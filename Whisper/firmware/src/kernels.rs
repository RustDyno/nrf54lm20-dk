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

#[repr(C)]
#[derive(Clone, Copy, Default)]
pub struct Quant {
    pub scale: f32,
    pub zp: i32,
}

impl Quant {
    #[inline]
    fn q8(&self, x: f32) -> i8 {
        (libm::roundf(x / self.scale) as i32 + self.zp).clamp(-128, 127) as i8
    }

    #[inline]
    fn q16(&self, x: f32) -> i16 {
        (libm::roundf(x / self.scale) as i32 + self.zp).clamp(-32768, 32767) as i16
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
