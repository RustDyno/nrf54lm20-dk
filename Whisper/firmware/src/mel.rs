//! Whisper log-mel frontend on the M33.
//!
//! Mirrors whisper.audio.log_mel_spectrogram exactly: 400-sample Hann
//! window, hop 160, centered frames with reflect padding, |STFT|^2 (no
//! normalization), 80 mel filters, log10 with a 1e-10 floor. The global
//! max-minus-8 clamp and (x+4)/4 normalization are a second pass
//! (mel_normalize) once the whole chunk's max is known.
//!
//! The 400-point transform is a mixed-radix DIT FFT (400 = 5*5*4*4 with
//! real radix-4 leaves). Every twiddle W_400^m comes from the same
//! 400-entry cosine table the old direct DFT used (sin via the +300
//! index shift), so the tables and the SD image are unchanged; only the
//! summation order differs from the direct DFT (float rounding at the
//! ulp level, ~20x fewer operations). All tables (Hann, cos, mel
//! filters) are data supplied by the host/SD image, not baked in.

#[repr(C)]
pub struct MelParams {
    pub pcm: u32,      // i16 samples, scaled /32768 on the fly
    pub n_samples: u32,
    pub frame0: u32,   // first frame index to compute
    pub n_frames: u32,
    pub hann: u32,     // f32[400]
    pub cos_tab: u32,  // f32[400], cos(2*pi*i/400)
    pub filters: u32,  // f32[80*201], row-major [mel][bin]
    pub out: u32,      // f32[80 * n_frames], planar [mel][frame]
    pub max_acc: u32,  // f32 running max across calls
}

const N_FFT: usize = 400;
const HOP: usize = 160;
const N_BINS: usize = 201;
const N_MELS: usize = 80;

/// Compute log10-mel for frames [frame0, frame0+n_frames), updating the
/// running max. Frame t is centered at t*HOP with reflect padding.
pub unsafe fn mel_frames(p: &MelParams) {
    let pcm = core::slice::from_raw_parts(p.pcm as *const i16, p.n_samples as usize);
    let hann = core::slice::from_raw_parts(p.hann as *const f32, N_FFT);
    let cos_tab = core::slice::from_raw_parts(p.cos_tab as *const f32, N_FFT);
    let filters = core::slice::from_raw_parts(p.filters as *const f32, N_MELS * N_BINS);
    let out = core::slice::from_raw_parts_mut(
        p.out as *mut f32,
        N_MELS * p.n_frames as usize,
    );
    mel_frames_core(
        pcm,
        p.frame0 as usize,
        p.n_frames as usize,
        hann,
        cos_tab,
        filters,
        out,
        &mut *(p.max_acc as *mut f32),
    );
}

/// Safe-slice core of `mel_frames` (also compiled by the host-side
/// comparator harness, which cannot form the u32-address param block).
#[allow(clippy::too_many_arguments)]
pub fn mel_frames_core(
    pcm: &[i16],
    frame0: usize,
    n_frames: usize,
    hann: &[f32],
    cos_tab: &[f32],
    filters: &[f32],
    out: &mut [f32],
    max_acc: &mut f32,
) {
    // Nonzero span of each (triangular) filter row. Skipped entries are
    // exact 0.0 and every product is >= +0.0, so the trimmed sum is
    // bit-identical to the dense one.
    let mut lo = [0u16; N_MELS];
    let mut hi = [0u16; N_MELS];
    for m in 0..N_MELS {
        let fr = &filters[m * N_BINS..(m + 1) * N_BINS];
        let (mut a, mut b) = (N_BINS, 0);
        for (k, &v) in fr.iter().enumerate() {
            if v != 0.0 {
                if a == N_BINS {
                    a = k;
                }
                b = k + 1;
            }
        }
        lo[m] = if a == N_BINS { 0 } else { a as u16 };
        hi[m] = b as u16;
    }

    let n = pcm.len() as i32;
    let mut w = [0.0f32; N_FFT];
    let mut spec = [0.0f32; 2 * N_FFT];
    let mut power = [0.0f32; N_BINS];
    for f in 0..n_frames {
        let t = (frame0 + f) as i32;
        let start = t * HOP as i32 - (N_FFT / 2) as i32;
        for (i, wi) in w.iter_mut().enumerate() {
            let mut idx = start + i as i32;
            if idx < 0 {
                idx = -idx; // reflect
            }
            if idx >= n {
                idx = 2 * n - 2 - idx;
            }
            *wi = pcm[idx as usize] as f32 * (1.0 / 32768.0) * hann[i];
        }
        fft400_real(&w, cos_tab, &mut spec);
        for (k, pk) in power.iter_mut().enumerate() {
            let re = spec[2 * k];
            let im = spec[2 * k + 1];
            *pk = re * re + im * im;
        }
        for m in 0..N_MELS {
            let fr = &filters[m * N_BINS..(m + 1) * N_BINS];
            let mut s = 0.0f32;
            for k in lo[m] as usize..hi[m] as usize {
                s += fr[k] * power[k];
            }
            let v = libm::log10f(if s > 1e-10 { s } else { 1e-10 });
            out[m * n_frames + f] = v;
            if v > *max_acc {
                *max_acc = v;
            }
        }
    }
}

// --- 400-point mixed-radix FFT -----------------------------------------------
//
// Decimation in time over the fixed plan 400 -(r=5)-> 80 -(r=5)-> 16
// -(r=4)-> 4, with a hardcoded radix-4 leaf on the (real) input samples.
// A stage splits x into r residue sequences x_q[j] = x[j*r + q], sub-
// transforms each, then combines in place:
//   X[s + t*m] = sum_q (Y_q[s] * W_len^(q*s)) * W_r^(q*t)
// For every len dividing 400, W_len^e = W_400^(e * 400/len), so all
// twiddles are (cos_tab[i], -cos_tab[(i+300) % 400]) -- the identical
// table values the direct DFT consumed.

/// Full complex spectrum of 400 real samples: out[2k], out[2k+1] =
/// Re, Im of X[k] for k = 0..400 (the mel pass reads bins 0..=200).
fn fft400_real(x: &[f32; N_FFT], cos_tab: &[f32], out: &mut [f32; 2 * N_FFT]) {
    fft_real(x, 0, 1, out, N_FFT, cos_tab);
}

/// DIT FFT of the `len` real samples x[off], x[off+stride], ... into
/// interleaved complex out[0..2*len]. len is one of 400/80/16/4.
fn fft_real(x: &[f32], off: usize, stride: usize, out: &mut [f32], len: usize, cos_tab: &[f32]) {
    if len == 4 {
        // Radix-4 on real inputs (W_4 = -i): X1/X3 are conjugates.
        let a = x[off];
        let b = x[off + stride];
        let c = x[off + 2 * stride];
        let d = x[off + 3 * stride];
        out[0] = a + b + c + d;
        out[1] = 0.0;
        out[2] = a - c;
        out[3] = d - b;
        out[4] = a - b + c - d;
        out[5] = 0.0;
        out[6] = a - c;
        out[7] = b - d;
        return;
    }
    let r = if len == 16 { 4 } else { 5 };
    let m = len / r;
    for q in 0..r {
        fft_real(x, off + q * stride, stride * r, &mut out[2 * q * m..2 * (q + 1) * m], m, cos_tab);
    }
    if r == 4 {
        combine4(out, cos_tab);
    } else {
        combine5(out, len, cos_tab);
    }
}

#[inline]
fn twiddle(cos_tab: &[f32], i: usize) -> (f32, f32) {
    let j = i + 300;
    let j = if j >= N_FFT { j - N_FFT } else { j };
    (cos_tab[i], -cos_tab[j])
}

/// In-place radix-5 combine over out[0..2*len] (len = 400 or 80).
fn combine5(out: &mut [f32], len: usize, cos_tab: &[f32]) {
    let m = len / 5;
    let step = N_FFT / len; // W_len^1 = W_400^step
    // W_5^e = W_400^(80e)
    let mut w5r = [0.0f32; 5];
    let mut w5i = [0.0f32; 5];
    for e in 0..5 {
        let (re, im) = twiddle(cos_tab, (80 * e) % N_FFT);
        w5r[e] = re;
        w5i[e] = im;
    }
    let mut idx = [0usize; 5]; // idx[q] = (q*s*step) % 400, walked per s
    let mut zr = [0.0f32; 5];
    let mut zi = [0.0f32; 5];
    for s in 0..m {
        for q in 0..5 {
            let yr = out[2 * (q * m + s)];
            let yi = out[2 * (q * m + s) + 1];
            let (wr, wi) = twiddle(cos_tab, idx[q]);
            zr[q] = yr * wr - yi * wi;
            zi[q] = yr * wi + yi * wr;
            idx[q] += q * step;
            if idx[q] >= N_FFT {
                idx[q] -= N_FFT;
            }
        }
        for t in 0..5 {
            let mut ar = 0.0f32;
            let mut ai = 0.0f32;
            let mut e = 0usize; // (q*t) % 5
            for q in 0..5 {
                ar += zr[q] * w5r[e] - zi[q] * w5i[e];
                ai += zr[q] * w5i[e] + zi[q] * w5r[e];
                e += t;
                if e >= 5 {
                    e -= 5;
                }
            }
            out[2 * (t * m + s)] = ar;
            out[2 * (t * m + s) + 1] = ai;
        }
    }
}

/// In-place radix-4 combine over out[0..32] (len = 16, m = 4, step 25).
fn combine4(out: &mut [f32], cos_tab: &[f32]) {
    let mut idx = [0usize; 4];
    let mut zr = [0.0f32; 4];
    let mut zi = [0.0f32; 4];
    for s in 0..4 {
        for q in 0..4 {
            let yr = out[2 * (q * 4 + s)];
            let yi = out[2 * (q * 4 + s) + 1];
            let (wr, wi) = twiddle(cos_tab, idx[q]);
            zr[q] = yr * wr - yi * wi;
            zi[q] = yr * wi + yi * wr;
            idx[q] += q * 25;
            if idx[q] >= N_FFT {
                idx[q] -= N_FFT;
            }
        }
        // Radix-4 butterfly, W_4 = -i.
        let ar = zr[0] - zr[2];
        let ai = zi[0] - zi[2];
        let br = zr[1] - zr[3];
        let bi = zi[1] - zi[3];
        let sr = zr[0] + zr[2];
        let si = zi[0] + zi[2];
        let tr = zr[1] + zr[3];
        let ti = zi[1] + zi[3];
        out[2 * s] = sr + tr;
        out[2 * s + 1] = si + ti;
        out[2 * (4 + s)] = ar + bi;
        out[2 * (4 + s) + 1] = ai - br;
        out[2 * (8 + s)] = sr - tr;
        out[2 * (8 + s) + 1] = si - ti;
        out[2 * (12 + s)] = ar - bi;
        out[2 * (12 + s) + 1] = ai + br;
    }
}

#[repr(C)]
pub struct MelNormParams {
    pub mel: u32,     // f32[n], log10-mel
    pub n: u32,
    pub max_acc: u32, // f32, global max from the mel pass
    pub out: u32,     // i8[n]
    pub q: crate::kernels::Quant,
}

/// whisper's normalization: clamp to max-8, then (x+4)/4, then quantize.
pub unsafe fn mel_normalize(p: &MelNormParams) {
    let mel = core::slice::from_raw_parts(p.mel as *const f32, p.n as usize);
    let out = core::slice::from_raw_parts_mut(p.out as *mut i8, p.n as usize);
    let floor = *(p.max_acc as *const f32) - 8.0;
    for i in 0..mel.len() {
        let x = (if mel[i] < floor { floor } else { mel[i] } + 4.0) / 4.0;
        let q = (libm::roundf(x / p.q.scale) as i32 + p.q.zp).clamp(-128, 127);
        out[i] = q as i8;
    }
}
