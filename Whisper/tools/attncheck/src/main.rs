//! Bit-exactness check of the firmware attention kernel after the
//! scratch-struct change: compiles the real kernels.rs and compares
//! attn_head against a direct transliteration of the numpy golden
//! (tape.py attn_golden) over decode/cross/encoder shapes.

#![allow(dead_code)]

#[path = "../../../firmware/src/kernels.rs"]
mod kernels;

use kernels::{AttnScratch, Quant, MAX_KEYS, PROBS};

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
}

fn main() {
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
    ];
    let mut scratch = Box::new(AttnScratch {
        acc: [0; MAX_KEYS],
        scores: [0.0; MAX_KEYS],
        p16: [0; MAX_KEYS],
    });
    let mut worst = 0usize;
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
            kernels::attn_head(&q, &k, &v, &mut a, hd, wq, qs, tk, ks,
                               zq, zk, zv, sm, vs, cq, &mut scratch);
            golden(&q, &k, &v, &mut b, hd, wq, qs, tk, ks,
                   zq, zk, zv, sm, vs, cq);
            let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
            worst = worst.max(diff);
            if diff != 0 {
                println!("case {case} rep {rep} (wq={wq} tk={tk} ks={ks}): {diff} bytes differ");
            }
        }
    }
    let _ = PROBS;
    if worst == 0 {
        println!("attncheck: all shapes bit-identical to the golden");
    } else {
        println!("attncheck: WORST {worst} mismatching ctx bytes");
    }
}
