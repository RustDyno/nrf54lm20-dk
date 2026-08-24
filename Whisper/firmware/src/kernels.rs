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
#[derive(Clone, Copy)]
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

/// Layernorm over `cols`-wide rows: int16 residual in -> int8 out.
/// gamma/beta are f32, streamed into the arena by the host.
pub fn ln_i16_to_i8(
    src: &[i16],
    sq: Quant,
    gamma: &[f32],
    beta: &[f32],
    dst: &mut [i8],
    dq: Quant,
    cols: usize,
) {
    for (row, out) in src.chunks_exact(cols).zip(dst.chunks_exact_mut(cols)) {
        let mut mean = 0.0f32;
        for &v in row {
            mean += sq.dq16(v);
        }
        mean /= cols as f32;
        let mut var = 0.0f32;
        for &v in row {
            let d = sq.dq16(v) - mean;
            var += d * d;
        }
        var /= cols as f32;
        let inv = 1.0 / libm::sqrtf(var + 1e-5);
        for c in 0..cols {
            let y = (sq.dq16(row[c]) - mean) * inv * gamma[c] + beta[c];
            out[c] = dq.q8(y);
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
/// (B must be symmetric). Used for QK^T (per head) and the logits tiles.
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

/// C[m,n] = A[m,k] * B[k,n], int8 x int8 -> int32, B row-major [k,n].
/// Used for probs x V (V stays in its natural [frames, head_dim] layout).
pub fn matmul_i8_b(a: &[i8], za: i32, b: &[i8], acc: &mut [i32], m: usize, k: usize, n: usize) {
    for i in 0..m {
        let ar = &a[i * k..(i + 1) * k];
        for j in 0..n {
            let mut s = 0i32;
            for t in 0..k {
                s += (ar[t] as i32 - za) * b[t * n + j] as i32;
            }
            acc[i * n + j] = s;
        }
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
