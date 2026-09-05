//! CPU glue kernels for the layered Whisper pipeline.
//!
//! Everything the NPU cannot run executes here, on dequantized views of the
//! int8/int16 activations in the arena. The numerical definitions mirror
//! model/layered.py exactly; the Python int8 simulation is the reference for
//! every function in this file.
//!
//! Quantization conventions (from the layered pipeline):
//! - NPU-facing activations: int8, per-tensor scale/zero-point.
//! - Residual stream: int16 (int8 destroys the small per-block deltas).
//! - Attention q/k/v sites are symmetric (zp = 0), so QK^T needs no
//!   zero-point correction; softmax output is fixed at scale 1/256, zp -128.

use crate::dsp;

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Quant {
    pub scale: f32,
    pub zp: i32,
}

impl Quant {
    #[inline]
    pub fn q8(&self, x: f32) -> i8 {
        (dsp::round_i32(x / self.scale) + self.zp).clamp(-128, 127) as i8
    }

    #[inline]
    fn q16(&self, x: f32) -> i16 {
        (dsp::round_i32(x / self.scale) + self.zp).clamp(-32768, 32767) as i16
    }

    #[inline]
    fn dq8(&self, q: i8) -> f32 {
        (q as i32 - self.zp) as f32 * self.scale
    }

    #[inline]
    fn dq16(&self, q: i16) -> f32 {
        (q as i32 - self.zp) as f32 * self.scale
    }
}

/// Layernorm over channels for each frame, channel-planar [C][W] layout:
/// int16 residual in -> int8 out (an NPU submodel input). gamma/beta are
/// f32[C], streamed into the arena by the host.
pub fn ln_planar_i16_to_i8(
    src: &[i16],
    sq: Quant,
    gamma: &[f32],
    beta: &[f32],
    dst: &mut [i8],
    dq: Quant,
    ch: usize,
    w: usize,
) {
    for f in 0..w {
        let mut mean = 0.0f32;
        for c in 0..ch {
            mean += sq.dq16(src[c * w + f]);
        }
        mean /= ch as f32;
        let mut var = 0.0f32;
        for c in 0..ch {
            let d = sq.dq16(src[c * w + f]) - mean;
            var += d * d;
        }
        var /= ch as f32;
        let inv = 1.0 / libm::sqrtf(var + 1e-5);
        for c in 0..ch {
            let y = (sq.dq16(src[c * w + f]) - mean) * inv * gamma[c] + beta[c];
            dst[c * w + f] = dq.q8(y);
        }
    }
}

/// int8 -> int8 scalar map (GELU): exact via a host-computed 256-entry table.
pub fn lut_i8(lut: &[i8; 256], src: &[i8], dst: &mut [i8]) {
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = lut[(*s as i32 + 128) as usize];
    }
}

/// C[m,n] += A[m,k] * B[n,k]^T, int8 x int8 -> int32. `za` is A's zero-point
/// (B must be symmetric). Used for the logits tiles.
pub fn matmul_i8_bt(a: &[i8], za: i32, b: &[i8], acc: &mut [i32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let ar = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let br = &b[j * k..(j + 1) * k];
            let mut s = 0i32;
            for t in 0..k {
                s += (ar[t] as i32 - za) * br[t] as i32;
            }
            acc[i * n + j] = s;
        }
    }
}

pub const MAX_KEYS: usize = 640;

/// Working buffers for `attn_head`. 6.4 KB -- deliberately NOT stack
/// locals: at decode depth that frame reached below _stack_end and
/// overwrote the end of .bss (the Axon driver's state struct lives
/// there; hardware-observed as a wild register write mid-infer). The
/// caller parks this in memory that is idle during CPU attention.
pub struct AttnScratch {
    pub acc: [i32; MAX_KEYS],
    pub scores: [f32; MAX_KEYS],
    pub p16: [i16; MAX_KEYS],
}

/// One attention head, fused QK^T -> softmax -> probs x V, channel-planar
/// buffers. q/ctx are [hd, wq] (column stride qstride), k/v are [hd, tk]
/// (column stride kstride; tk <= kstride masks padding frames out of the
/// keys). zq/zk/zv are the q/k/v zero-points (the emitted submodels'
/// converter-chosen output quantization); probs are quantized to the fixed
/// 1/256 scale exactly as in the Python pipeline; ctx is written at
/// `ctx_q` (the out-projection submodel's input quantization).
#[allow(clippy::too_many_arguments)]
pub fn attn_head(
    q: &[i8],
    k: &[i8],
    v: &[i8],
    ctx: &mut [i8],
    hd: usize,
    wq: usize,
    qstride: usize,
    tk: usize,
    kstride: usize,
    zq: i32,
    zk: i32,
    zv: i32,
    score_mult: f32,
    v_scale: f32,
    ctx_q: Quant,
    s: &mut AttnScratch,
) {
    // Bit-exact restructure of the naive triple loop (measured ~21 cy/MAC:
    // stride-`qstride` column walks with a bounds check per access). Both
    // matmul phases run c-outer so the inner loop walks a CONTIGUOUS k/v
    // row through checked-once slices, and the zero-points come out of the
    // inner loops via the exact integer identity
    //   sum_c (q_c - zq)(k_c - zk) = sum_c (q_c - zq) k_c - zk sum_c (q_c - zq)
    // (and its probs x V analogue). All accumulation stays i32 in the same
    // algebraic terms, so scores, probs, and ctx match the previous
    // implementation (and the numpy mirror) bit for bit.
    let acc = &mut s.acc;
    let scores = &mut s.scores;
    let p16 = &mut s.p16;
    for qi in 0..wq {
        acc[..tk].fill(0);
        let mut qsum = 0i32;
        for c in 0..hd {
            let qc = q[c * qstride + qi] as i32 - zq;
            if qc == 0 {
                continue; // zero rank-1 update: skips a full key row
            }
            qsum += qc;
            let krow = &k[c * kstride..c * kstride + tk];
            for (a, &kv) in acc[..tk].iter_mut().zip(krow) {
                *a += qc * kv as i32;
            }
        }
        let corr = zk * qsum;
        let mut max = f32::MIN;
        for (s, &a) in scores[..tk].iter_mut().zip(&acc[..tk]) {
            *s = (a - corr) as f32 * score_mult;
            if *s > max {
                max = *s;
            }
        }
        let mut sum = 0.0f32;
        for s in scores[..tk].iter_mut() {
            *s = libm::expf(*s - max);
            sum += *s;
        }
        let inv = 1.0 / sum;
        let mut psum = 0i32;
        for (p, &s) in p16[..tk].iter_mut().zip(&scores[..tk]) {
            let pv = PROBS.q8(s * inv) as i32 + 128;
            *p = pv as i16;
            psum += pv;
        }
        let vcorr = zv * psum;
        for c in 0..hd {
            let vrow = &v[c * kstride..c * kstride + tk];
            let mut a = 0i32;
            for (&p, &vv) in p16[..tk].iter().zip(vrow) {
                a += p as i32 * vv as i32;
            }
            ctx[c * qstride + qi] = ctx_q.q8((a - vcorr) as f32 * (PROBS.scale * v_scale));
        }
    }
}

/// MLP fc2 recombination, elementwise: dst16 = qd(q_res(a16) + sum of the
/// four dequantized int8 partial outputs). Callers tile by element range to
/// bound the arena working set.
pub fn fc2_sum(
    parts: [&[i8]; 4],
    pq: &[Quant; 4],
    a: &[i16],
    qa: Quant,
    dst: &mut [i16],
    qd: Quant,
) {
    for i in 0..dst.len() {
        let mut s = qa.dq16(a[i]);
        for j in 0..4 {
            s += pq[j].dq8(parts[j][i]);
        }
        dst[i] = qd.q16(s);
    }
}

pub const PROBS: Quant = Quant {
    scale: 1.0 / 256.0,
    zp: -128,
};

/// Row-wise softmax over int32 scores (dequantized by `mult`), quantized to
/// the fixed probs scale. Scratch must hold one row of f32.
pub fn softmax_rows_quant(
    acc: &[i32],
    mult: f32,
    dst: &mut [i8],
    cols: usize,
    scratch: &mut [f32],
) {
    for (row, out) in acc.chunks_exact(cols).zip(dst.chunks_exact_mut(cols)) {
        let mut max = f32::MIN;
        for (i, &v) in row.iter().enumerate() {
            scratch[i] = v as f32 * mult;
            if scratch[i] > max {
                max = scratch[i];
            }
        }
        let mut sum = 0.0f32;
        for s in scratch[..cols].iter_mut() {
            *s = libm::expf(*s - max);
            sum += *s;
        }
        let inv = 1.0 / sum;
        for i in 0..cols {
            out[i] = PROBS.q8(scratch[i] * inv);
        }
    }
}

/// Requantize int32 accumulators (per-tensor `mult`) to an int8 site.
pub fn requant_i32_to_i8(acc: &[i32], mult: f32, dst: &mut [i8], dq: Quant) {
    for (a, d) in acc.iter().zip(dst.iter_mut()) {
        *d = dq.q8(*a as f32 * mult);
    }
}

/// Residual update: dst16 = a16 + b8 (dequantized add, requantized to dst).
pub fn add_i16_i8(a: &[i16], qa: Quant, b: &[i8], qb: Quant, dst: &mut [i16], qd: Quant) {
    for i in 0..dst.len() {
        dst[i] = qd.q16(qa.dq16(a[i]) + qb.dq8(b[i]));
    }
}

/// Entry into the residual stream: int8 activation + f32 vector (e.g. the
/// positional embedding or a host-supplied embedding row) -> int16 residual.
pub fn add_i8_f32_to_i16(a: &[i8], qa: Quant, b: &[f32], dst: &mut [i16], qd: Quant) {
    for i in 0..dst.len() {
        dst[i] = qd.q16(qa.dq8(a[i]) + b[i]);
    }
}

#[repr(C)]
pub struct ArgmaxState {
    pub best: f32,
    pub best_idx: u32,
}

/// Fold one logits tile into the running argmax. `mults[j]` dequantizes row j
/// (per-channel embedding weight scale x input scale); `idx[j]` is the row's
/// vocabulary id (the host streams only non-suppressed rows).
pub fn logits_max(acc: &[i32], mults: &[f32], idx: &[u32], state: &mut ArgmaxState) {
    for j in 0..acc.len() {
        let v = acc[j] as f32 * mults[j];
        if v > state.best {
            state.best = v;
            state.best_idx = idx[j];
        }
    }
}

// --- attention on the DSP extension ---------------------------------------------
//
// Same arithmetic as `attn_head` (i32 sums of the same products, the same
// f32 softmax and quantization expressions), restructured so the two
// matmul phases are dot products the SMLAD kernels in dsp.rs can run:
// keys are transposed to key-major int16 once per head (`AttnKv`), values
// are widened to int16, and queries are processed two at a time against
// two keys / two value rows per kernel call. Verified bit-identical to the
// golden by tools/attncheck.

pub const HD64: usize = 64;
const _: () = assert!(MAX_KEYS == dsp::ROW2);

/// Cycle split of `attn_head_kt` (QK dots, softmax rows, PV dots and
/// context quantization); the firmware prints it next to the attention
/// total and resets it. Host builds count nothing.
pub static mut ATTN_PROF: [u64; 3] = [0; 3];

#[inline(always)]
fn cyc() -> u32 {
    #[cfg(target_arch = "arm")]
    {
        cortex_m::peripheral::DWT::cycle_count()
    }
    #[cfg(not(target_arch = "arm"))]
    {
        0
    }
}

#[inline(always)]
fn prof(i: usize, t0: u32) {
    #[cfg(target_arch = "arm")]
    unsafe {
        ATTN_PROF[i] += cyc().wrapping_sub(t0) as u64;
    }
    #[cfg(not(target_arch = "arm"))]
    {
        let _ = (i, t0);
    }
}

/// Working buffers for `attn_head_kt`: two queries at a time (~13 KB).
/// The p16 rows sit exactly ROW2 elements apart, which the 2x2 kernel
/// bakes in as its second-row offset. Parked in the idle interlayer.
#[repr(C)]
pub struct AttnScratch2 {
    pub acc: [[i32; MAX_KEYS]; 2],
    pub scores: [[f32; MAX_KEYS]; 2],
    pub p16: [[i16; MAX_KEYS]; 2],
    pub q16: [[i16; HD64]; 2],
}

/// Keys and values in the layouts the kernels consume: `kt` key-major
/// int16 [tkp][64] (rows tk..tkp zero, tkp = tk rounded up to 8) and
/// `v16` int16 [64][MAX_KEYS] (columns tk..tkp zero).
pub struct AttnKv<'a> {
    pub kt: &'a [i16],
    pub v16: &'a [i16],
    pub tk: usize,
}

/// Elements of a full `kt` / `v16` buffer.
pub const KV16_LEN: usize = MAX_KEYS * HD64;

#[inline]
pub fn keys_padded(tk: usize) -> usize {
    tk.div_ceil(8) * 8
}

/// Fill `kt` from any int8 key layout through `get(channel, key)`.
#[allow(dead_code)] // reference form, exercised by tools/attncheck
pub fn attn_prepare_kt<F: Fn(usize, usize) -> i8>(get: F, tk: usize, kt: &mut [i16]) {
    let tkp = keys_padded(tk);
    assert!(tk >= 1 && tkp <= MAX_KEYS && kt.len() >= tkp * HD64);
    for j in 0..tk {
        let row = &mut kt[j * HD64..(j + 1) * HD64];
        for (c, r) in row.iter_mut().enumerate() {
            *r = get(c, j) as i16;
        }
    }
    kt[tk * HD64..tkp * HD64].fill(0);
}

/// Fill `v16` from any int8 value layout through `get(channel, key)`.
#[allow(dead_code)] // reference form, exercised by tools/attncheck
pub fn attn_prepare_v16<F: Fn(usize, usize) -> i8>(get: F, tk: usize, v16: &mut [i16]) {
    let tkp = keys_padded(tk);
    assert!(tk >= 1 && tkp <= MAX_KEYS && v16.len() >= KV16_LEN);
    for c in 0..HD64 {
        let row = &mut v16[c * MAX_KEYS..c * MAX_KEYS + tkp];
        for (j, r) in row[..tk].iter_mut().enumerate() {
            *r = get(c, j) as i16;
        }
        row[tk..].fill(0);
    }
}

/// One attention head over prepared keys/values; q/ctx are [64, wq] with
/// column stride `qstride` (channel-planar), exactly like `attn_head`.
/// `exp` is the softmax exponential (see `SoftmaxExp`): `ExpFn(libm::expf)`
/// reproduces the golden bit for bit, `ExpFn(dsp::exp_neg)` is the fast
/// polynomial, `ExpTable` the integer-indexed table (see the call sites).
#[allow(clippy::too_many_arguments)]
pub fn attn_head_kt<E: SoftmaxExp>(
    q: &[i8],
    kv: &AttnKv,
    ctx: &mut [i8],
    wq: usize,
    qstride: usize,
    zq: i32,
    zk: i32,
    zv: i32,
    score_mult: f32,
    v_scale: f32,
    ctx_q: Quant,
    s: &mut AttnScratch2,
    exp: &E,
) {
    let tk = kv.tk;
    let tkp = keys_padded(tk);
    assert!(tk >= 1 && tkp <= MAX_KEYS);
    assert!(kv.kt.len() >= tkp * HD64 && kv.v16.len() >= KV16_LEN);
    assert!(wq >= 1 && q.len() >= (HD64 - 1) * qstride + wq);
    assert!(ctx.len() >= (HD64 - 1) * qstride + wq);
    let ctx_mult = PROBS.scale * v_scale;
    let mut qi = 0;
    while qi < wq {
        // an odd trailing query is paired with itself; its twin's
        // results are simply not stored
        let two = qi + 1 < wq;
        let mut qsum = [0i32; 2];
        for c in 0..HD64 {
            let a = q[c * qstride + qi] as i32 - zq;
            let b = if two { q[c * qstride + qi + 1] as i32 - zq } else { a };
            s.q16[0][c] = a as i16;
            s.q16[1][c] = b as i16;
            qsum[0] += a;
            qsum[1] += b;
        }
        let t0 = cyc();
        let mut j = 0;
        while j < tkp {
            let r = unsafe {
                dsp::dot64_2x2(s.q16.as_ptr() as *const i16, kv.kt.as_ptr().add(j * HD64))
            };
            s.acc[0][j] = r[0];
            s.acc[0][j + 1] = r[1];
            s.acc[1][j] = r[2];
            s.acc[1][j + 1] = r[3];
            j += 2;
        }
        prof(0, t0);
        let t0 = cyc();
        // a lone query's twin row is the same query: its softmax is
        // skipped (the PV kernel still reads the row, whose contents are
        // then irrelevant because the twin's results are not stored)
        let mut psum = [0i32; 2];
        for row in 0..(if two { 2 } else { 1 }) {
            let (acc, scores, p16) = (&s.acc[row], &mut s.scores[row], &mut s.p16[row]);
            psum[row] = softmax_row(&acc[..tk], zk * qsum[row], score_mult,
                                    &mut scores[..tk], p16, tkp, exp);
        }
        prof(1, t0);
        let t0 = cyc();
        let vcorr = [zv * psum[0], zv * psum[1]];
        let mut c = 0;
        while c < HD64 {
            let r = unsafe {
                dsp::dot_pv_2x2(s.p16.as_ptr() as *const i16,
                                kv.v16.as_ptr().add(c * MAX_KEYS), tkp / 8)
            };
            ctx[c * qstride + qi] = ctx_q.q8((r[0] - vcorr[0]) as f32 * ctx_mult);
            ctx[(c + 1) * qstride + qi] = ctx_q.q8((r[1] - vcorr[0]) as f32 * ctx_mult);
            if two {
                ctx[c * qstride + qi + 1] = ctx_q.q8((r[2] - vcorr[1]) as f32 * ctx_mult);
                ctx[(c + 1) * qstride + qi + 1] =
                    ctx_q.q8((r[3] - vcorr[1]) as f32 * ctx_mult);
            }
            c += 2;
        }
        prof(2, t0);
        qi += 2;
    }
}

/// One query against int8 keys and values in their storage layouts, for
/// the decode cross-attention: `k` is key-major ([key][64] int8, at
/// least `keys_padded(tk)` rows), `v` tile-major ([tile][64 channels][64
/// keys] int8, `tk.div_ceil(64)` tiles), `q`/`ctx` are one column of a
/// [64, qstride] channel-planar tensor. Same sums and the same softmax as
/// `attn_head_kt`; the kernels widen the int8 operands in registers, so
/// nothing is transposed or widened per head.
#[allow(clippy::too_many_arguments)]
pub fn attn_head_x1_i8<E: SoftmaxExp>(
    q: &[i8],
    k: &[i8],
    v: &[i8],
    tk: usize,
    ctx: &mut [i8],
    qstride: usize,
    zq: i32,
    zk: i32,
    zv: i32,
    score_mult: f32,
    v_scale: f32,
    ctx_q: Quant,
    s: &mut AttnScratch2,
    exp: &E,
) {
    let tkp = keys_padded(tk);
    let tiles = tk.div_ceil(64);
    assert!(tk >= 1 && tkp <= MAX_KEYS);
    assert!(k.len() >= tkp * HD64 && v.len() >= tiles * HD64 * 64);
    assert!(q.len() >= (HD64 - 1) * qstride + 1 && ctx.len() >= (HD64 - 1) * qstride + 1);
    assert!(k.as_ptr() as usize % 4 == 0 && v.as_ptr() as usize % 4 == 0);
    let ctx_mult = PROBS.scale * v_scale;
    let mut qsum = 0i32;
    for c in 0..HD64 {
        let a = q[c * qstride] as i32 - zq;
        s.q16[0][dsp::perm4(c)] = a as i16;
        qsum += a;
    }
    let t0 = cyc();
    let mut j = 0;
    while j < tkp {
        let r = unsafe {
            dsp::dot64_q1k4_i8(s.q16[0].as_ptr(), k.as_ptr().add(j * HD64))
        };
        s.acc[0][j..j + 4].copy_from_slice(&r);
        j += 4;
    }
    prof(0, t0);
    let t0 = cyc();
    let psum = softmax_row(&s.acc[0][..tk], zk * qsum, score_mult,
                           &mut s.scores[0][..tk], &mut s.p16[0], tkp, exp);
    // the PV kernel reads the probabilities in perm4 order
    let p = &mut s.p16[0];
    let mut j = 0;
    while j < tkp {
        p.swap(j + 1, j + 2);
        j += 4;
    }
    prof(1, t0);
    let t0 = cyc();
    let vcorr = zv * psum;
    let mut c = 0;
    while c < HD64 {
        let mut acc = [0i32; 4];
        for i in 0..tiles {
            let n8 = (tkp - i * 64).min(64) / 8;
            acc = unsafe {
                dsp::dot_pv_q1v4_i8(s.p16[0].as_ptr().add(i * 64),
                                    v.as_ptr().add(i * HD64 * 64 + c * 64), n8, acc)
            };
        }
        for r in 0..4 {
            ctx[(c + r) * qstride] = ctx_q.q8((acc[r] - vcorr) as f32 * ctx_mult);
        }
        c += 4;
    }
    prof(2, t0);
}

/// The softmax exponential of `attn_head_kt`: fills `scores[j]` with
/// exp(x_j) for x_j = (acc_j - corr) * mult - max (<= 0, where max is
/// that expression at `amax`, the row's integer maximum) and returns the
/// sum of the row.
pub trait SoftmaxExp {
    fn row(&self, acc: &[i32], amax: i32, corr: i32, mult: f32, scores: &mut [f32]) -> f32;
}

/// The exponential as a function of the float argument, evaluated the
/// way the golden does it (two rounded products and a subtraction, then
/// the function).
pub struct ExpFn<F: Fn(f32) -> f32>(pub F);

impl<F: Fn(f32) -> f32> SoftmaxExp for ExpFn<F> {
    #[inline(always)]
    fn row(&self, acc: &[i32], amax: i32, corr: i32, mult: f32, scores: &mut [f32]) -> f32 {
        let max = (amax - corr) as f32 * mult;
        let mut sum = 0.0f32;
        for (s, &a) in scores.iter_mut().zip(acc) {
            *s = (self.0)((a - corr) as f32 * mult - max);
            sum += *s;
        }
        sum
    }
}

pub const EXP_LO_BITS: u32 = 10;
pub const EXP_LO: usize = 1 << EXP_LO_BITS;
pub const EXP_HI: usize = 256;

/// exp(-mult * d) for the integer score deficit d = amax - acc >= 0, as
/// the product of two table entries: `lo[d mod 1024]` and `hi[d / 1024]`.
///
/// The deficit is an integer and the tables are built from it in f64, so
/// each value is within ~1.5 ulp of the true exponential; the golden's
/// float path rounds two products and a difference per key first and
/// sits about as far from the true value in its own direction. Deficits
/// past the `hi` table (mult * 256 * 1024 and beyond, or wherever the
/// entry underflows) are 0, invisible next to a row sum >= 1. About 30
/// cycles per key against ~100 for the polynomial.
///
/// The f64 here runs in software: the M33 has no double-precision FPU,
/// and the build disables the `fp64` feature so this does not compile
/// to instructions the core would fault on.
#[repr(C)]
pub struct ExpTable {
    pub lo: [f32; EXP_LO],
    pub hi: [f32; EXP_HI],
    pub mult: f32,
}

pub const EXP_TABLE_BYTES: usize = core::mem::size_of::<ExpTable>();

impl ExpTable {
    /// (Re)build for a score multiplier. ~1300 f64 exponentials.
    pub fn build(&mut self, mult: f32) {
        let m = mult as f64;
        for (i, e) in self.lo.iter_mut().enumerate() {
            *e = libm::exp(-m * i as f64) as f32;
        }
        for (j, e) in self.hi.iter_mut().enumerate() {
            let x = -m * (EXP_LO * j) as f64;
            *e = if x < -87.0 { 0.0 } else { libm::exp(x) as f32 };
        }
        self.mult = mult;
    }

    #[inline(always)]
    pub fn eval(&self, d: u32) -> f32 {
        let hi = ((d >> EXP_LO_BITS) as usize).min(EXP_HI - 1);
        let lo = (d & (EXP_LO as u32 - 1)) as usize;
        // SAFETY: both indices are masked / clamped into their tables
        unsafe { *self.hi.get_unchecked(hi) * *self.lo.get_unchecked(lo) }
    }
}

impl SoftmaxExp for ExpTable {
    #[inline(always)]
    fn row(&self, acc: &[i32], amax: i32, _corr: i32, mult: f32, scores: &mut [f32]) -> f32 {
        assert!(self.mult.to_bits() == mult.to_bits());
        let mut sum = 0.0f32;
        for (s, &a) in scores.iter_mut().zip(acc) {
            let e = self.eval((amax - a) as u32);
            *s = e;
            sum += e;
        }
        sum
    }
}

/// One softmax row: dequantize the i32 scores, exponentiate, quantize
/// to the fixed 1/256 probability scale, stored +128-biased as int16
/// (zero beyond `tk` up to `tkp`). Returns the biased sum for the V
/// zero-point correction.
fn softmax_row<E: SoftmaxExp>(
    acc: &[i32],
    corr: i32,
    mult: f32,
    scores: &mut [f32],
    p16: &mut [i16; MAX_KEYS],
    tkp: usize,
    exp: &E,
) -> i32 {
    let tk = acc.len();
    // The row maximum is an integer maximum: the conversion to f32 and
    // the positive multiplier are both monotonic, so this is the value a
    // pass over the float scores would find, without storing them.
    let mut amax = i32::MIN;
    for &a in acc {
        if a > amax {
            amax = a;
        }
    }
    let sum = exp.row(acc, amax, corr, mult, scores);
    // PROBS.q8(s * inv) + 128, with the /(1/256) as an exact *256: the
    // power-of-two scaling commutes with the rounding of s * inv, so
    // folding it into the reciprocal quantizes identically. The biased
    // value clamp(r - 128, -128, 127) + 128 is clamp(r, 0, 255), one
    // saturating instruction.
    let inv256 = (1.0 / sum) * 256.0;
    let mut psum = 0i32;
    for (p, &s) in p16[..tk].iter_mut().zip(scores.iter()) {
        let pv = dsp::round_i32(s * inv256).clamp(0, 255);
        *p = pv as i16;
        psum += pv;
    }
    p16[tk..tkp].fill(0);
    psum
}
