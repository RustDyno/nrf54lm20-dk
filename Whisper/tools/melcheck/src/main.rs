//! Old-vs-new comparator for the firmware mel frontend.
//!
//! Compiles the real firmware mel.rs (the mixed-radix FFT) next to a
//! frozen copy of the old direct-DFT implementation, feeds both the
//! exact Hann/cos/filter tables from the built SD image, and compares:
//!   - log-mel f32 (max abs diff, old vs new, both vs an f64 DFT)
//!   - the int8 mel actually consumed by conv1 (enc.mel quant)
//!   - the three device drive patterns (sequential chunks, streaming
//!     chunks, one whole-run call) for bitwise self-consistency
//! plus a wall-clock ratio for the full 1200-frame pass.

#![allow(dead_code)]

use std::time::Instant;

#[path = "../../../firmware/src/kernels.rs"]
mod kernels;
#[path = "../../../firmware/src/mel.rs"]
mod mel;

const N_FFT: usize = 400;
const HOP: usize = 160;
const N_BINS: usize = 201;
const N_MELS: usize = 80;
const N_SAMPLES: usize = 192_000; // 12 s at 16 kHz (app.rs)
const N_FRAMES: usize = 1200;
const T: usize = 64; // frame tile (app.rs)
const CHUNK: usize = 10240; // streaming samples per PDM buffer (app.rs)
const N_CHUNKS: usize = 19;

// enc.mel quantization from model/out/scales.json.
const MEL_Q: kernels::Quant = kernels::Quant {
    scale: 0.007843137254901960,
    zp: -59,
};

/// Frozen copy of the pre-FFT mel_frames inner loops (direct table DFT,
/// dense filter rows), arithmetic order preserved exactly.
#[allow(clippy::too_many_arguments)]
fn mel_frames_ref(
    pcm: &[i16],
    frame0: usize,
    n_frames: usize,
    hann: &[f32],
    cos_tab: &[f32],
    filters: &[f32],
    out: &mut [f32],
    max_acc: &mut f32,
) {
    let n = pcm.len() as i32;
    let mut w = [0.0f32; N_FFT];
    let mut power = [0.0f32; N_BINS];
    for f in 0..n_frames {
        let t = (frame0 + f) as i32;
        let start = t * HOP as i32 - (N_FFT / 2) as i32;
        for (i, wi) in w.iter_mut().enumerate() {
            let mut idx = start + i as i32;
            if idx < 0 {
                idx = -idx;
            }
            if idx >= n {
                idx = 2 * n - 2 - idx;
            }
            *wi = pcm[idx as usize] as f32 * (1.0 / 32768.0) * hann[i];
        }
        for (k, pk) in power.iter_mut().enumerate() {
            let mut re = 0.0f32;
            let mut im = 0.0f32;
            let mut idx = 0usize;
            for &wn in w.iter() {
                re += wn * cos_tab[idx];
                im -= wn * cos_tab[(idx + 300) % N_FFT];
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
            out[m * n_frames + f] = v;
            if v > *max_acc {
                *max_acc = v;
            }
        }
    }
}

/// f64 reference: same windowed signal and tables (widened), exact-cos
/// twiddles, dense filters. Bounds how far each f32 path is from truth.
fn mel_frames_f64(pcm: &[i16], hann: &[f32], filters: &[f32]) -> Vec<f64> {
    let mut cos64 = [0.0f64; N_FFT];
    let mut sin64 = [0.0f64; N_FFT];
    for i in 0..N_FFT {
        let th = 2.0 * std::f64::consts::PI * i as f64 / N_FFT as f64;
        cos64[i] = th.cos();
        sin64[i] = th.sin();
    }
    let n = pcm.len() as i64;
    let mut out = vec![0.0f64; N_MELS * N_FRAMES];
    let mut w = [0.0f64; N_FFT];
    let mut power = [0.0f64; N_BINS];
    for f in 0..N_FRAMES {
        let start = f as i64 * HOP as i64 - (N_FFT / 2) as i64;
        for (i, wi) in w.iter_mut().enumerate() {
            let mut idx = start + i as i64;
            if idx < 0 {
                idx = -idx;
            }
            if idx >= n {
                idx = 2 * n - 2 - idx;
            }
            *wi = pcm[idx as usize] as f64 / 32768.0 * hann[i] as f64;
        }
        for (k, pk) in power.iter_mut().enumerate() {
            let mut re = 0.0f64;
            let mut im = 0.0f64;
            for (nn, &wn) in w.iter().enumerate() {
                let idx = nn * k % N_FFT;
                re += wn * cos64[idx];
                im -= wn * sin64[idx];
            }
            *pk = re * re + im * im;
        }
        for m in 0..N_MELS {
            let fr = &filters[m * N_BINS..(m + 1) * N_BINS];
            let mut s = 0.0f64;
            for k in 0..N_BINS {
                s += fr[k] as f64 * power[k];
            }
            out[m * N_FRAMES + f] = (if s > 1e-10 { s } else { 1e-10 }).log10();
        }
    }
    out
}

type MelFn = fn(&[i16], usize, usize, &[f32], &[f32], &[f32], &mut [f32], &mut f32);

/// Drive one impl the way mel_pass1 does (sequential chunks read back
/// from the card, block-rounded windows). Returns planar [80][1200].
fn drive_seq(f: MelFn, pcm: &[i16], hann: &[f32], cos: &[f32], filt: &[f32]) -> (Vec<f32>, f32) {
    let mut full = vec![0.0f32; N_MELS * N_FRAMES];
    let mut max = -1e30f32;
    for ci in 0..N_CHUNKS {
        let f0 = ci * T;
        let n_frames = if ci == 18 { 48 } else { T };
        let s0 = (f0 * HOP).saturating_sub(1280);
        let span = ((f0 + n_frames) * HOP + 256).min(N_SAMPLES) - s0;
        let bytes = (span * 2).div_ceil(512) * 512;
        let n_samples = (bytes / 2).min(N_SAMPLES - s0);
        let frame0 = (f0 * HOP - s0) / HOP;
        let mut out = vec![0.0f32; N_MELS * n_frames];
        f(&pcm[s0..s0 + n_samples], frame0, n_frames, hann, cos, filt, &mut out, &mut max);
        for m in 0..N_MELS {
            for fr in 0..n_frames {
                full[m * N_FRAMES + f0 + fr] = out[m * n_frames + fr];
            }
        }
    }
    (full, max)
}

/// Drive one impl the way record_mel does (streaming sliding window:
/// 1280-sample left halo + chunk + 256-sample right halo).
fn drive_stream(f: MelFn, pcm: &[i16], hann: &[f32], cos: &[f32], filt: &[f32]) -> (Vec<f32>, f32) {
    const WIN: usize = 12032;
    // The mic fills 19 whole chunks (194560 samples); everything past
    // N_SAMPLES is masked by the final call's n_samples and never read.
    let mut padded = pcm.to_vec();
    padded.resize(N_CHUNKS * CHUNK, 0);
    let pcm = &padded[..];
    let mut full = vec![0.0f32; N_MELS * N_FRAMES];
    let mut max = -1e30f32;
    let mut win = vec![0i16; WIN];
    let mut have = 0usize;
    let emit = |win: &[i16], ci: usize, n_samples: usize, n_frames: usize,
                    max: &mut f32, full: &mut [f32]| {
        let frame0 = if ci == 0 { 0 } else { 8 };
        let mut out = vec![0.0f32; N_MELS * n_frames];
        f(&win[..n_samples], frame0, n_frames, hann, cos, filt, &mut out, max);
        for m in 0..N_MELS {
            for fr in 0..n_frames {
                full[m * N_FRAMES + ci * T + fr] = out[m * n_frames + fr];
            }
        }
    };
    for k in 0..N_CHUNKS {
        let hop = &pcm[k * CHUNK..(k + 1) * CHUNK];
        if k == 0 {
            win[..CHUNK].copy_from_slice(hop);
            have = CHUNK;
            continue;
        }
        win[have..have + 256].copy_from_slice(&hop[..256]);
        emit(&win, k - 1, have + 256, T, &mut max, &mut full);
        win.copy_within(have - 1280..have, 0);
        win[1280..1280 + CHUNK].copy_from_slice(hop);
        have = 1280 + CHUNK;
    }
    emit(&win, N_CHUNKS - 1, N_SAMPLES - ((N_CHUNKS - 1) * CHUNK - 1280), 48, &mut max, &mut full);
    (full, max)
}

/// One whole-run call (frame0 = 0, all 1200 frames, full 12 s PCM).
fn drive_whole(f: MelFn, pcm: &[i16], hann: &[f32], cos: &[f32], filt: &[f32]) -> (Vec<f32>, f32) {
    let mut full = vec![0.0f32; N_MELS * N_FRAMES];
    let mut max = -1e30f32;
    f(pcm, 0, N_FRAMES, hann, cos, filt, &mut full, &mut max);
    (full, max)
}

/// whisper normalization + enc.mel int8 quant (mirrors mel_normalize).
fn normalize_i8(mel: &[f32], max: f32) -> Vec<i8> {
    let floor = max - 8.0;
    mel.iter()
        .map(|&v| {
            let x = (if v < floor { floor } else { v } + 4.0) / 4.0;
            (libm::roundf(x / MEL_Q.scale) as i32 + MEL_Q.zp).clamp(-128, 127) as i8
        })
        .collect()
}

fn max_diff_f32(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

fn max_diff_f64(a: &[f32], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(&x, &y)| (x as f64 - y).abs()).fold(0.0, f64::max)
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn i16_pm(&mut self, amp: i32) -> i16 {
        ((self.next() % (2 * amp as u32 + 1)) as i32 - amp) as i16
    }
}

fn signal(which: usize) -> (&'static str, Vec<i16>) {
    let mut r = Lcg(0x5eed_0000 + which as u64);
    let mut x = vec![0i16; N_SAMPLES];
    match which {
        0 => {
            for v in x.iter_mut() {
                *v = r.i16_pm(4); // near-silence with dither
            }
            ("silence+dither", x)
        }
        1 => {
            for v in x.iter_mut() {
                *v = r.i16_pm(20000);
            }
            ("white noise", x)
        }
        2 => {
            // AM harmonic stack, speech-ish envelope
            for (i, v) in x.iter_mut().enumerate() {
                let t = i as f64 / 16000.0;
                let env = (0.5 - 0.5 * (2.0 * std::f64::consts::PI * 2.3 * t).cos())
                    * (0.6 + 0.4 * (2.0 * std::f64::consts::PI * 0.35 * t).sin());
                let mut s = 0.0;
                for (h, a) in [(140.0, 0.5), (280.0, 0.3), (560.0, 0.15), (1120.0, 0.08)] {
                    s += a * (2.0 * std::f64::consts::PI * h * t).sin();
                }
                *v = ((s * env * 18000.0) as i32).clamp(-32768, 32767) as i16
                    + r.i16_pm(60);
            }
            ("harmonic stack", x)
        }
        3 => {
            // linear chirp 50 -> 7900 Hz
            for (i, v) in x.iter_mut().enumerate() {
                let t = i as f64 / 16000.0;
                let ph = 2.0 * std::f64::consts::PI * (50.0 * t + (7850.0 / 12.0) * t * t / 2.0);
                *v = (ph.sin() * 15000.0) as i16;
            }
            ("chirp", x)
        }
        4 => {
            for i in (0..N_SAMPLES).step_by(1600) {
                x[i] = 30000;
            }
            ("impulse train", x)
        }
        _ => {
            for v in x.iter_mut() {
                *v = r.i16_pm(300); // low-level noise (log floor regime)
            }
            ("low noise", x)
        }
    }
}

fn load_tables() -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let root = concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/out");
    let img = std::fs::read(format!("{root}/sd.img")).expect("model/out/sd.img");
    let idx: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(format!("{root}/sd-index.json")).expect("sd-index.json"),
    )
    .unwrap();
    let get = |name: &str, n: usize| -> Vec<f32> {
        let e = &idx[name];
        let off = e["lba"].as_u64().unwrap() as usize * 512;
        let bytes = e["bytes"].as_u64().unwrap() as usize;
        assert_eq!(bytes, n * 4, "{name} size");
        img[off..off + bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect()
    };
    (
        get("hann", N_FFT),
        get("melcos", N_FFT),
        get("melfilt", N_MELS * N_BINS),
    )
}

fn main() {
    let (hann, cos, filt) = load_tables();
    println!(
        "tables from sd.img: hann[0..3]={:?} cos[100]={} filt sum={:.4}",
        &hann[0..3],
        cos[100],
        filt.iter().sum::<f32>()
    );

    let mut worst_i8 = 0usize;
    for which in 0..6 {
        let (name, pcm) = signal(which);

        let (o_seq, o_max) = drive_seq(mel_frames_ref, &pcm, &hann, &cos, &filt);
        let (o_str, o_smax) = drive_stream(mel_frames_ref, &pcm, &hann, &cos, &filt);
        let (n_seq, n_max) = drive_seq(mel::mel_frames_core, &pcm, &hann, &cos, &filt);
        let (n_str, n_smax) = drive_stream(mel::mel_frames_core, &pcm, &hann, &cos, &filt);
        let (n_whole, _) = drive_whole(mel::mel_frames_core, &pcm, &hann, &cos, &filt);

        // Device drive patterns must agree with themselves bitwise.
        assert!(o_seq == o_str && o_max == o_smax, "{name}: old seq!=stream");
        assert!(n_seq == n_str && n_max == n_smax, "{name}: new seq!=stream");
        assert!(n_seq == n_whole, "{name}: new seq!=whole");

        let f64ref = mel_frames_f64(&pcm, &hann, &filt);
        let d_on = max_diff_f32(&o_seq, &n_seq);
        let d_of = max_diff_f64(&o_seq, &f64ref);
        let d_nf = max_diff_f64(&n_seq, &f64ref);

        let qi_o = normalize_i8(&o_seq, o_max);
        let qi_n = normalize_i8(&n_seq, n_max);
        let mism = qi_o
            .iter()
            .zip(&qi_n)
            .filter(|(a, b)| a != b)
            .count();
        let mism_max = qi_o
            .iter()
            .zip(&qi_n)
            .map(|(&a, &b)| (a as i32 - b as i32).abs())
            .max()
            .unwrap();
        worst_i8 = worst_i8.max(mism);
        println!(
            "{name:>15}: dlog(old,new) {d_on:.3e}  vs f64: old {d_of:.3e} new {d_nf:.3e}  \
             i8 mism {mism}/96000 (max {mism_max})  max {o_max:.4}/{n_max:.4}"
        );
    }

    // Wall-clock, full 1200-frame pass.
    let (_, pcm) = signal(2);
    let t0 = Instant::now();
    let _ = drive_seq(mel_frames_ref, &pcm, &hann, &cos, &filt);
    let t_old = t0.elapsed();
    let t0 = Instant::now();
    let _ = drive_seq(mel::mel_frames_core, &pcm, &hann, &cos, &filt);
    let t_new = t0.elapsed();
    println!(
        "host time, 1200 frames: old {:.1} ms, new {:.1} ms ({:.1}x)",
        t_old.as_secs_f64() * 1e3,
        t_new.as_secs_f64() * 1e3,
        t_old.as_secs_f64() / t_new.as_secs_f64()
    );
    println!("worst int8 mismatch count: {worst_i8}/96000");
}
