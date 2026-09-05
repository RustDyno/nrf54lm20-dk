//! Bit-exactness checks of the firmware attention kernels and the DSP
//! primitives they use. Compiles the real kernels.rs / dsp.rs / q4.rs
//! (portable fallback bodies on the host) and compares:
//!
//! - attn_head (scalar) and attn_head_kt (the SMLAD-shaped kernel, with
//!   libm::expf) against a direct transliteration of the numpy golden
//!   (tape.py attn_golden) over decode/cross/encoder shapes: 0 bytes may
//!   differ.
//! - attn_head_kt with the fast exponential, and with the integer-indexed
//!   exponential table the encoder uses: reports how many context bytes
//!   move (expected: a handful of 1-LSB cases at most).
//! - dsp::exp_neg vs libm::expf over a dense sweep of the softmax domain.
//! - q4::unpack16 vs q4::unpack, dsp::byte_sum vs a byte loop.

#![allow(dead_code)]

#[path = "../../../firmware/src/dsp.rs"]
mod dsp;
#[path = "../../../firmware/src/kernels.rs"]
mod kernels;
#[path = "../../../firmware/src/q4.rs"]
mod q4;

use kernels::{AttnKv, AttnScratch, AttnScratch2, ExpFn, ExpTable, Quant, EXP_HI, EXP_LO,
              KV16_LEN, MAX_KEYS, PROBS};

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
    let mut tab = Box::new(ExpTable { lo: [0.0; EXP_LO], hi: [0.0; EXP_HI], mult: 0.0 });
    let mut worst_old = 0usize;
    let mut worst_kt = 0usize;
    let mut fast_total = 0usize;
    let mut fast_bytes = 0usize;
    let mut fast_maxdiff = 0i32;
    let mut tab_total = 0usize;
    let mut tab_maxdiff = 0i32;
    let mut r = Lcg(0xa77);
    // score multipliers spanning the image's attention sites (q k scales
    // 0.03..0.12, / 8) plus the harness's old fixed value
    let mults = [0.25f32 * 0.11 / 8.0, 4.26e-4, 7.9e-4, 9.5e-4, 1.4e-3];
    for (case, &(wq, qs, tk, ks)) in shapes.iter().enumerate() {
        for rep in 0..5 {
            let q = r.i8v(hd * qs);
            let k = r.i8v(hd * ks);
            let v = r.i8v(hd * ks);
            let (zq, zk, zv) = ((r.next() % 11) as i32 - 5,
                                (r.next() % 11) as i32 - 5,
                                (r.next() % 11) as i32 - 5);
            let sm = mults[rep];
            let vs = 0.17f32;
            let cq = Quant { scale: 0.06, zp: 3 };
            let mut a = vec![0i8; hd * qs];
            let mut b = vec![0i8; hd * qs];
            let mut c = vec![0i8; hd * qs];
            let mut d = vec![0i8; hd * qs];
            let mut e = vec![0i8; hd * qs];
            kernels::attn_head(&q, &k, &v, &mut a, hd, wq, qs, tk, ks,
                               zq, zk, zv, sm, vs, cq, &mut scratch);
            golden(&q, &k, &v, &mut b, hd, wq, qs, tk, ks,
                   zq, zk, zv, sm, vs, cq);
            kernels::attn_prepare_kt(|ch, j| k[ch * ks + j], tk, &mut kt);
            kernels::attn_prepare_v16(|ch, j| v[ch * ks + j], tk, &mut v16);
            let kv = AttnKv { kt: &kt, v16: &v16, tk };
            kernels::attn_head_kt(&q, &kv, &mut c, wq, qs, zq, zk, zv, sm, vs, cq,
                                  &mut s2, &ExpFn(libm::expf));
            kernels::attn_head_kt(&q, &kv, &mut d, wq, qs, zq, zk, zv, sm, vs, cq,
                                  &mut s2, &ExpFn(dsp::exp_neg));
            tab.build(sm);
            kernels::attn_head_kt(&q, &kv, &mut e, wq, qs, zq, zk, zv, sm, vs, cq,
                                  &mut s2, &*tab);
            let diff_old = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            let diff_kt = c.iter().zip(&b).filter(|(x, y)| x != y).count();
            let diff_fast = d.iter().zip(&b).filter(|(x, y)| x != y).count();
            let diff_tab = e.iter().zip(&b).filter(|(x, y)| x != y).count();
            let maxd = d.iter().zip(&b).map(|(x, y)| (*x as i32 - *y as i32).abs())
                .max().unwrap_or(0);
            let maxt = e.iter().zip(&b).map(|(x, y)| (*x as i32 - *y as i32).abs())
                .max().unwrap_or(0);
            worst_old = worst_old.max(diff_old);
            worst_kt = worst_kt.max(diff_kt);
            fast_total += diff_fast;
            fast_bytes += hd * wq;
            fast_maxdiff = fast_maxdiff.max(maxd);
            tab_total += diff_tab;
            tab_maxdiff = tab_maxdiff.max(maxt);
            if diff_old != 0 || diff_kt != 0 {
                println!("case {case} rep {rep} (wq={wq} tk={tk} ks={ks}): \
                          scalar {diff_old}, kt {diff_kt} bytes differ");
            }
        }
    }
    // the decode cross-attention form: one query, int8 keys key-major,
    // int8 values tile-major, must match the golden exactly with expf
    let mut worst_x1 = 0usize;
    for &tk in &[1usize, 5, 8, 63, 64, 65, 192, 582, 600, 640] {
        for rep in 0..3 {
            let tkp = kernels::keys_padded(tk);
            let tiles = tk.div_ceil(64);
            let ks = 640;
            let q = r.i8v(hd * 4);
            let k = r.i8v(hd * ks);
            let v = r.i8v(hd * ks);
            let (zq, zk, zv) = ((r.next() % 11) as i32 - 5,
                                (r.next() % 11) as i32 - 5,
                                (r.next() % 11) as i32 - 5);
            let sm = mults[rep];
            let vs = 0.17f32;
            let cq = Quant { scale: 0.06, zp: 3 };
            let mut kk = vec![0i8; tkp.max(tiles * 64) * hd];
            for j in 0..tk {
                for ch in 0..hd {
                    kk[j * hd + ch] = k[ch * ks + j];
                }
            }
            let mut vt = vec![0i8; tiles * hd * 64];
            for ch in 0..hd {
                for j in 0..tk {
                    vt[(j / 64) * hd * 64 + ch * 64 + j % 64] = v[ch * ks + j];
                }
            }
            let mut b = vec![0i8; hd * 4];
            let mut e = vec![0i8; hd * 4];
            golden(&q, &k, &v, &mut b, hd, 1, 4, tk, ks, zq, zk, zv, sm, vs, cq);
            kernels::attn_head_x1_i8(&q, &kk, &vt, tk, &mut e, 4, zq, zk, zv, sm, vs, cq,
                                     &mut s2, &ExpFn(libm::expf));
            let diff = e.iter().zip(&b).filter(|(x, y)| x != y).count();
            if diff != 0 {
                println!("x1_i8 tk={tk} rep {rep}: {diff} bytes differ");
            }
            worst_x1 = worst_x1.max(diff);
        }
    }
    println!("attention, one query on int8 K/V layouts: worst {worst_x1} \
              mismatching ctx bytes vs the golden (0 = bit-identical)");
    worst_kt = worst_kt.max(worst_x1);
    let _ = PROBS;
    println!("attention: scalar worst {worst_old}, SMLAD-shaped worst {worst_kt} \
              mismatching ctx bytes vs the golden (0 = bit-identical)");
    println!("attention with fast exp: {fast_total}/{fast_bytes} ctx bytes moved, \
              max |diff| {fast_maxdiff} LSB");
    println!("attention with the exp table: {tab_total}/{fast_bytes} ctx bytes moved, \
              max |diff| {tab_maxdiff} LSB");
    worst_old == 0 && worst_kt == 0 && fast_maxdiff <= 1 && tab_maxdiff <= 1
}

/// The table against the true exponential (f64) and against libm::expf
/// of the f32 argument over the deficits an attention row can produce.
fn check_exp_table() -> bool {
    let mut tab = Box::new(ExpTable { lo: [0.0; EXP_LO], hi: [0.0; EXP_HI], mult: 0.0 });
    let mut worst_true = 0u32;
    let mut worst_libm = 0u32;
    let mut ok = true;
    for &mult in &[4.26e-4f32, 7.9e-4, 1.4e-3, 3.0e-3] {
        tab.build(mult);
        if tab.eval(0) != 1.0 {
            ok = false;
        }
        for d in (0u32..300_000).step_by(7) {
            let x = -(mult as f64) * d as f64;
            let e = tab.eval(d);
            if x < -87.0 {
                // the flush point sits at a `hi` step (mult * 1024) below
                // -87; values in between are denormal-range and harmless
                if e != 0.0 && x < -87.0 - (mult as f64) * EXP_LO as f64 {
                    ok = false;
                }
                continue;
            }
            let t = libm::exp(x) as f32;
            worst_true = worst_true.max(ulp_dist(e, t));
            let xf = -(mult * d as f32);
            worst_libm = worst_libm.max(ulp_dist(e, libm::expf(xf)));
        }
    }
    println!("exp table vs f64 exp: max {worst_true} ulp; vs libm::expf of the f32 \
              argument: max {worst_libm} ulp; flush past -87: {ok}");
    ok && worst_true <= 3
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
    let t = check_exp_table();
    let q = check_q4_and_sum();
    if a && e && t && q {
        println!("attncheck: all checks passed");
    } else {
        println!("attncheck: FAILED");
        std::process::exit(1);
    }
}
