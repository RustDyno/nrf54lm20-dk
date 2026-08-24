//! Whisper log-mel frontend on the M33.
//!
//! Mirrors whisper.audio.log_mel_spectrogram exactly: 400-sample Hann
//! window, hop 160, centered frames with reflect padding, |STFT|^2 (no
//! normalization), 80 mel filters, log10 with a 1e-10 floor. The global
//! max-minus-8 clamp and (x+4)/4 normalization are a second pass
//! (mel_normalize) once the whole chunk's max is known.
//!
//! The 400-point DFT is table-driven: one 400-entry cosine table covers
//! both cos and sin ((k*n) mod 400 walked incrementally), so there are no
//! trig calls and no power-of-two FFT constraint. All tables (Hann, cos,
//! mel filters) are data supplied by the host/SD image, not baked in.

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
    let max_acc = p.max_acc as *mut f32;

    let n = pcm.len() as i32;
    let mut w = [0.0f32; N_FFT];
    let mut power = [0.0f32; N_BINS];
    for f in 0..p.n_frames as usize {
        let t = (p.frame0 as usize + f) as i32;
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
        for (k, pk) in power.iter_mut().enumerate() {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            let mut idx = 0usize; // (k*n) mod 400, walked incrementally
            for &wn in w.iter() {
                re += wn * cos_tab[idx];
                im -= wn * cos_tab[(idx + 300) % N_FFT]; // sin via cos shift
                idx += k;
                if idx >= N_FFT {
                    idx -= N_FFT;
                }
            }
            *pk = re * re + im * im;
        }
        for m in 0..N_MELS {
            let fr = &filters[m * N_BINS..(m + 1) * N_BINS];
            let mut s = 0.0f32;
            for k in 0..N_BINS {
                s += fr[k] * power[k];
            }
            let v = libm::log10f(if s > 1e-10 { s } else { 1e-10 });
            out[m * p.n_frames as usize + f] = v;
            if v > *max_acc {
                *max_acc = v;
            }
        }
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
