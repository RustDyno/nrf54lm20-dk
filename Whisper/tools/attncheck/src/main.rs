//! Bit-exactness checks of the firmware attention kernels and the DSP
//! primitives they use. Compiles the real kernels.rs / dsp.rs / q4.rs
//! (portable fallback bodies on the host) and compares:
//!
//! - attn_head (scalar) and attn_head_kt (the SMLAD-shaped kernel, with
//!   libm::expf) against a direct transliteration of the numpy golden
//!   (tape.py attn_golden) over decode/cross/encoder shapes: 0 bytes may
//!   differ.
//! - attn_head_kt with the fast exponential: reports how many context
//!   bytes move (expected: a handful of 1-LSB cases at most).
//! - dsp::exp_neg vs libm::expf over a dense sweep of the softmax domain.
//! - q4::unpack16 vs q4::unpack, dsp::byte_sum vs a byte loop.

#![allow(dead_code)]

#[path = "../../../firmware/src/dsp.rs"]
mod dsp;
#[path = "../../../firmware/src/kernels.rs"]
mod kernels;
#[path = "../../../firmware/src/q4.rs"]
mod q4;

use kernels::{AttnKv, AttnScratch, AttnScratch2, Quant, KV16_LEN, MAX_KEYS, PROBS};

/// tape.py attn_golden, one head, transliterated (i64 like numpy).
#[allow(clippy::too_many_arguments)]
fn golden(
    q: &[i8], k: &[i8], v: &[i8], ctx: &mut [i8],
    hd: usize, wq: usize, qstride: usize, tk: usize, kstride: usize,
    zq: i32, zk: i32, zv: i32, score_mult: f32, v_scale: f32, ctx_q: Quant,
) {
    for qi in 0..wq {
        let mut scores = vec![0f32; tk];
        for (j, s) in scores.iter_mut().enumerate() {
            let mut acc = 0i64;
            for c in 0..hd {
                acc += (q[c * qstride + qi] as i64 - zq as i64)
                    * (k[c * kstride + j] as i64 - zk as i64);
            }
            *s = acc as f32 * score_mult;
        }
        let m = scores.iter().fold(f32::MIN, |a, &b| a.max(b));
        let mut sum = 0f32;
        let e: Vec<f32> = scores.iter().map(|&s| {
            let x = libm::expf(s - m);
            sum += x;
            x
        }).collect();
        let inv = 1.0 / sum;
        let p8: Vec<i64> = e.iter().map(|&x| {
            (libm::roundf(x * inv * 256.0) as i64 - 128).clamp(-128, 127)
        }).collect();
        for c in 0..hd {
            let mut acc = 0i64;
            for (j, &p) in p8.iter().enumerate() {
                acc += (v[c * kstride + j] as i64 - zv as i64) * (p + 128);
            }
            let x = acc as f32 * (v_scale / 256.0);
            let qv = (libm::roundf(x / ctx_q.scale) as i32 + ctx_q.zp)
                .clamp(-128, 127);
            ctx[c * qstride + qi] = qv as i8;
        }
    }
}

struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    fn i8v(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| (self.next() % 256) as i32 as u8 as i8).collect()
    }
    fn u8v(&mut self, n: usize) -> Vec<u8> {
        (0..n).map(|_| (self.next() % 256) as u8).collect()
    }
}

fn ulp_dist(a: f32, b: f32) -> u32 {
    if a == b {
        return 0;
    }
    (a.to_bits() as i64 - b.to_bits() as i64).unsigned_abs() as u32
}

fn check_attention() -> bool {
    let hd = 64;
    // (wq, qstride, tk, kstride) for decode self, decode cross, encoder
    let shapes = [
        (1usize, 4usize, 1usize, 32usize),
        (1, 4, 5, 32),
        (1, 4, 32, 32),
        (1, 4, 600, 600),
        (1, 4, 192, 192),
        (64, 64, 192, 640),
        (64, 64, 600, 640),
        (64, 64, 640, 640),
        (3, 8, 7, 16),
    ];
    let mut scratch = Box::new(AttnScratch {
        acc: [0; MAX_KEYS],
        scores: [0.0; MAX_KEYS],
        p16: [0; MAX_KEYS],
    });
    let mut s2 = Box::new(AttnScratch2 {
        acc: [[0; MAX_KEYS]; 2],
        scores: [[0.0; MAX_KEYS]; 2],
        p16: [[0; MAX_KEYS]; 2],
        q16: [[0; 64]; 2],
    });
    let mut kt = vec![0i16; KV16_LEN];
    let mut v16 = vec![0i16; KV16_LEN];
    let mut worst_old = 0usize;
    let mut worst_kt = 0usize;
    let mut fast_total = 0usize;
    let mut fast_bytes = 0usize;
    let mut fast_maxdiff = 0i32;
    let mut r = Lcg(0xa77);
    for (case, &(wq, qs, tk, ks)) in shapes.iter().enumerate() {
        for rep in 0..3 {
            let q = r.i8v(hd * qs);
            let k = r.i8v(hd * ks);
            let v = r.i8v(hd * ks);
            let (zq, zk, zv) = ((r.next() % 11) as i32 - 5,
                                (r.next() % 11) as i32 - 5,
                                (r.next() % 11) as i32 - 5);
            let sm = 0.25f32 * 0.11 / 8.0;
            let vs = 0.17f32;
            let cq = Quant { scale: 0.06, zp: 3 };
            let mut a = vec![0i8; hd * qs];
            let mut b = vec![0i8; hd * qs];
            let mut c = vec![0i8; hd * qs];
            let mut d = vec![0i8; hd * qs];
            kernels::attn_head(&q, &k, &v, &mut a, hd, wq, qs, tk, ks,
                               zq, zk, zv, sm, vs, cq, &mut scratch);
            golden(&q, &k, &v, &mut b, hd, wq, qs, tk, ks,
                   zq, zk, zv, sm, vs, cq);
            kernels::attn_prepare_kt(|ch, j| k[ch * ks + j], tk, &mut kt);
            kernels::attn_prepare_v16(|ch, j| v[ch * ks + j], tk, &mut v16);
            let kv = AttnKv { kt: &kt, v16: &v16, tk };
            kernels::attn_head_kt(&q, &kv, &mut c, wq, qs, zq, zk, zv, sm, vs, cq,
                                  &mut s2, libm::expf);
            kernels::attn_head_kt(&q, &kv, &mut d, wq, qs, zq, zk, zv, sm, vs, cq,
                                  &mut s2, dsp::exp_neg);
            let diff_old = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            let diff_kt = c.iter().zip(&b).filter(|(x, y)| x != y).count();
            let diff_fast = d.iter().zip(&b).filter(|(x, y)| x != y).count();
            let maxd = d.iter().zip(&b).map(|(x, y)| (*x as i32 - *y as i32).abs())
                .max().unwrap_or(0);
            worst_old = worst_old.max(diff_old);
            worst_kt = worst_kt.max(diff_kt);
            fast_total += diff_fast;
            fast_bytes += hd * wq;
            fast_maxdiff = fast_maxdiff.max(maxd);
            if diff_old != 0 || diff_kt != 0 {
                println!("case {case} rep {rep} (wq={wq} tk={tk} ks={ks}): \
                          scalar {diff_old}, kt {diff_kt} bytes differ");
            }
        }
    }
    let _ = PROBS;
    println!("attention: scalar worst {worst_old}, SMLAD-shaped worst {worst_kt} \
              mismatching ctx bytes vs the golden (0 = bit-identical)");
    println!("attention with fast exp: {fast_total}/{fast_bytes} ctx bytes moved, \
              max |diff| {fast_maxdiff} LSB");
    worst_old == 0 && worst_kt == 0 && fast_maxdiff <= 1
}

fn check_exp() -> bool {
    let mut max_ulp = 0u32;
    let mut at = 0f32;
    let mut sum_ulp = 0u64;
    let mut n = 0u64;
    let mut x = -87.0f32;
    while x <= 0.0 {
        let a = dsp::exp_neg(x);
        let b = libm::expf(x);
        let u = ulp_dist(a, b);
        if u > max_ulp {
            max_ulp = u;
            at = x;
        }
        sum_ulp += u as u64;
        n += 1;
        x += 1.0e-4;
    }
    // exact zero at the flush point and beyond
    let below = dsp::exp_neg(-90.0) == 0.0 && dsp::exp_neg(-1000.0) == 0.0;
    println!("exp_neg vs libm::expf on [-87, 0] step 1e-4: max {max_ulp} ulp (at {at:.4}), \
              mean {:.3} ulp over {n} points; flush below -87: {below}",
             sum_ulp as f64 / n as f64);
    max_ulp <= 4 && below && dsp::exp_neg(0.0) == 1.0
}

fn check_q4_and_sum() -> bool {
    let mut r = Lcg(0x51);
    let mut ok = true;
    let mut tables = Box::new([0u32; q4::TABLES16]);
    q4::tables16(&mut tables);
    for _ in 0..40 {
        let groups = 2 * (1 + (r.next() % 40) as usize); // unpack needs 2G multiples
        let n = groups * q4::G;
        let amax = r.u8v(groups);
        let nibs: Vec<u8> = r.u8v(n / 2).iter().map(|&b| {
            // packer never emits nibble 0 (-8): keep to 1..15
            let lo = (b & 15).max(1);
            let hi = (b >> 4).max(1);
            lo | (hi << 4)
        }).collect();
        let mut d8 = vec![0i8; n];
        let mut d16 = vec![0i16; n];
        q4::unpack(&amax, &nibs, &mut d8);
        q4::unpack16(&tables, &amax, &nibs, &mut d16);
        if d8.iter().zip(&d16).any(|(a, b)| *a as i16 != *b) {
            ok = false;
            println!("unpack16 differs from unpack");
        }
    }
    for _ in 0..50 {
        let n = (r.next() % 3000) as usize;
        let b = r.u8v(n);
        let want = b.iter().fold(0u32, |s, &x| s.wrapping_add(x as u32));
        if dsp::byte_sum(&b) != want {
            ok = false;
            println!("byte_sum differs at n={n}");
        }
    }
    for _ in 0..1000 {
        let x = f32::from_bits(r.next());
        if x.is_nan() {
            continue;
        }
        let x = x.clamp(-3.0e9, 3.0e9);
        if dsp::round_i32(x) != libm::roundf(x) as i32 {
            ok = false;
            println!("round_i32 differs at {x}");
        }
    }
    println!("q4 unpack16 == unpack, byte_sum == byte loop, round_i32 == roundf: {}",
             if ok { "ok" } else { "FAILED" });
    ok
}

fn main() {
    let a = check_attention();
    let e = check_exp();
    let q = check_q4_and_sum();
    if a && e && q {
        println!("attncheck: all checks passed");
    } else {
        println!("attncheck: FAILED");
        std::process::exit(1);
    }
}
