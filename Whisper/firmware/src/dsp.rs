//! Cortex-M33 DSP-extension primitives with portable fallbacks.
//!
//! The host comparators (tools/attncheck, tools/q4check) compile this
//! file for x86, so every primitive has a plain-Rust body that is the
//! reference definition; the `target_arch = "arm"` bodies are inline
//! assembly that computes the same integer result, only faster:
//!
//! - SMLAD: two int16 x int16 products summed into an i32 accumulator
//!   per instruction (1 cycle), used for every dot product here.
//! - VCVTA: f32 -> i32 with round-half-away-from-zero and saturation,
//!   exactly `libm::roundf(x) as i32`, in one instruction.
//! - USADA8: sum of four byte lanes added to an accumulator, used for
//!   the blob integrity byte-sum (4 bytes per cycle instead of 3-4
//!   cycles per byte).
//!
//! The asm register classes for the FPU (`sreg`) need the `vfp2` target
//! feature, which the thumbv8m.main-none-eabihf target does not declare
//! at the Rust level even though the M33's FPv5 implements it: it is
//! enabled in .cargo/config.toml (rustc warns that the feature name is
//! unstable; the build is otherwise unaffected).

/// Row stride, in i16 elements, of the two-row operands of the 2x2 dot
/// kernels (probability rows and value rows). Baked into the asm as the
/// byte offset of the second row, so it is a fixed constant.
pub const ROW2: usize = 640;

/// f32 -> i32, round half away from zero, saturating (NaN -> 0). Equal
/// to `libm::roundf(x) as i32` for every input.
#[inline(always)]
pub fn round_i32(x: f32) -> i32 {
    #[cfg(target_arch = "arm")]
    {
        let r: i32;
        unsafe {
            core::arch::asm!(
                "vcvta.s32.f32 {o}, {i}",
                o = out(sreg) r,
                i = in(sreg) x,
                options(pure, nomem, nostack),
            );
        }
        r
    }
    #[cfg(not(target_arch = "arm"))]
    {
        libm::roundf(x) as i32
    }
}

/// Wrapping u32 sum of all bytes (the image's per-entry integrity sum).
///
/// 32 bytes per iteration: one LDM of eight words (the loads pipeline
/// one per cycle), eight USADA8 into two accumulators, so a 166 KB blob
/// sums in ~0.6 cycles per byte. Unaligned input takes the byte loop.
pub fn byte_sum(b: &[u8]) -> u32 {
    let mut s = 0u32;
    let n32 = if b.as_ptr() as usize % 4 == 0 { b.len() / 32 } else { 0 };
    #[cfg(target_arch = "arm")]
    {
        if n32 > 0 {
            let mut p = b.as_ptr();
            let mut n = n32;
            let mut s2 = 0u32;
            unsafe {
                core::arch::asm!(
                    "1:",
                    "ldmia {p}!, {{{w0}, {w1}, {w2}, {w3}, {w4}, {w5}, {w6}, {w7}}}",
                    "usada8 {s}, {w0}, {z}, {s}",
                    "usada8 {s2}, {w1}, {z}, {s2}",
                    "usada8 {s}, {w2}, {z}, {s}",
                    "usada8 {s2}, {w3}, {z}, {s2}",
                    "usada8 {s}, {w4}, {z}, {s}",
                    "usada8 {s2}, {w5}, {z}, {s2}",
                    "usada8 {s}, {w6}, {z}, {s}",
                    "usada8 {s2}, {w7}, {z}, {s2}",
                    "subs {n}, {n}, #1",
                    "bne 1b",
                    p = inout(reg) p,
                    n = inout(reg) n,
                    s = inout(reg) s,
                    s2 = inout(reg) s2,
                    z = in(reg) 0u32,
                    w0 = out(reg) _,
                    w1 = out(reg) _,
                    w2 = out(reg) _,
                    w3 = out(reg) _,
                    w4 = out(reg) _,
                    w5 = out(reg) _,
                    w6 = out(reg) _,
                    w7 = out(reg) _,
                    options(readonly, nostack),
                );
            }
            s = s.wrapping_add(s2);
            let _ = (p, n);
        }
    }
    #[cfg(not(target_arch = "arm"))]
    {
        for &x in &b[..n32 * 32] {
            s = s.wrapping_add(x as u32);
        }
    }
    for &x in &b[n32 * 32..] {
        s = s.wrapping_add(x as u32);
    }
    s
}

/// exp(x) for x <= 0, about 2 ulp from libm::expf, roughly 3x cheaper.
///
/// Range reduction as in musl (n = round(x*log2e), r = x - n*ln2 with a
/// two-part ln2), then a degree-7 Taylor polynomial for e^r on
/// |r| <= ln2/2 (truncation 6e-9 relative) scaled by 2^n through the
/// exponent field. Inputs below -87 return 0 where libm returns a
/// subnormal below 1.5e-38: invisible next to a softmax sum >= 1.
#[inline(always)]
pub fn exp_neg(x: f32) -> f32 {
    const LOG2E: f32 = 1.442_695_04;
    const LN2_HI: f32 = 6.931_457_52e-1; // top bits of ln2, n*LN2_HI exact
    const LN2_LO: f32 = 1.428_606_82e-6;
    if x < -87.0 {
        return 0.0;
    }
    let n = round_i32(x * LOG2E);
    let nf = n as f32;
    let r = (x - nf * LN2_HI) - nf * LN2_LO;
    // Horner, e^r = 1 + r(1 + r/2(1 + r/3(1 + r/4(1 + r/5(1 + r/6(1 + r/7))))))
    let mut p = 1.0 + r * (1.0 / 7.0);
    p = 1.0 + r * (1.0 / 6.0) * p;
    p = 1.0 + r * (1.0 / 5.0) * p;
    p = 1.0 + r * (1.0 / 4.0) * p;
    p = 1.0 + r * (1.0 / 3.0) * p;
    p = 1.0 + r * (1.0 / 2.0) * p;
    p = 1.0 + r * p;
    let scale = f32::from_bits(((n + 127) as u32) << 23);
    p * scale
}

/// Two queries x two keys over 64 int16 dims.
///
/// `q` points at query row 0 with row 1 following at +64 elements;
/// `k` likewise for the two keys. Returns
/// [q0.k0, q0.k1, q1.k0, q1.k1] as wrapping i32 sums (the callers'
/// operands are small enough that no sum can wrap).
///
/// # Safety
/// Both pointers must be readable for 128 i16 elements.
#[inline(always)]
pub unsafe fn dot64_2x2(q: *const i16, k: *const i16) -> [i32; 4] {
    #[cfg(target_arch = "arm")]
    {
        // One step = one int16 pair of every row: the two loads of a
        // row's second half use the post-incremented pointer, hence the
        // +124 (= 128 - 4) offsets. The four loads come first so that
        // they pipeline (consecutive loads issue one per cycle after the
        // first), then the four multiply-accumulates. Fully unrolled:
        // 32 steps.
        macro_rules! step {
            () => {
                concat!(
                    "ldr {t0}, [{qa}], #4\n",
                    "ldr {t1}, [{qa}, #124]\n",
                    "ldr {t2}, [{ka}], #4\n",
                    "ldr {t3}, [{ka}, #124]\n",
                    "smlad {a00}, {t0}, {t2}, {a00}\n",
                    "smlad {a01}, {t0}, {t3}, {a01}\n",
                    "smlad {a10}, {t1}, {t2}, {a10}\n",
                    "smlad {a11}, {t1}, {t3}, {a11}\n",
                )
            };
        }
        macro_rules! rep2 {
            ($s:expr) => {
                concat!($s, $s)
            };
        }
        macro_rules! rep8 {
            ($s:expr) => {
                rep2!(rep2!(rep2!($s)))
            };
        }
        let (mut a00, mut a01, mut a10, mut a11) = (0i32, 0i32, 0i32, 0i32);
        let mut qa = q;
        let mut ka = k;
        core::arch::asm!(
            rep8!(step!()),
            rep8!(step!()),
            rep8!(step!()),
            rep8!(step!()),
            qa = inout(reg) qa,
            ka = inout(reg) ka,
            a00 = inout(reg) a00,
            a01 = inout(reg) a01,
            a10 = inout(reg) a10,
            a11 = inout(reg) a11,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            options(pure, readonly, nostack),
        );
        let _ = (qa, ka);
        [a00, a01, a10, a11]
    }
    #[cfg(not(target_arch = "arm"))]
    {
        let q = core::slice::from_raw_parts(q, 128);
        let k = core::slice::from_raw_parts(k, 128);
        let mut a = [0i32; 4];
        for c in 0..64 {
            a[0] = a[0].wrapping_add(q[c] as i32 * k[c] as i32);
            a[1] = a[1].wrapping_add(q[c] as i32 * k[64 + c] as i32);
            a[2] = a[2].wrapping_add(q[64 + c] as i32 * k[c] as i32);
            a[3] = a[3].wrapping_add(q[64 + c] as i32 * k[64 + c] as i32);
        }
        a
    }
}

/// Two probability rows x two value rows over `n8 * 8` int16 keys.
///
/// `p` points at row 0 with row 1 at +ROW2 elements; `v` likewise.
/// Returns [p0.v0, p0.v1, p1.v0, p1.v1]. `n8` must be >= 1.
///
/// # Safety
/// Both row pairs must be readable for ROW2 + n8 * 8 elements.
#[inline(always)]
pub unsafe fn dot_pv_2x2(p: *const i16, v: *const i16, n8: usize) -> [i32; 4] {
    debug_assert!(n8 >= 1 && n8 * 8 <= ROW2);
    #[cfg(target_arch = "arm")]
    {
        // Second-row offset after the post-increment: ROW2 * 2 - 4. Loads
        // grouped ahead of the multiply-accumulates so they pipeline.
        macro_rules! step {
            () => {
                concat!(
                    "ldr {t0}, [{pa}], #4\n",
                    "ldr {t1}, [{pa}, #1276]\n",
                    "ldr {t2}, [{va}], #4\n",
                    "ldr {t3}, [{va}, #1276]\n",
                    "smlad {s00}, {t0}, {t2}, {s00}\n",
                    "smlad {s01}, {t0}, {t3}, {s01}\n",
                    "smlad {s10}, {t1}, {t2}, {s10}\n",
                    "smlad {s11}, {t1}, {t3}, {s11}\n",
                )
            };
        }
        const _: () = assert!(ROW2 * 2 - 4 == 1276);
        let (mut s00, mut s01, mut s10, mut s11) = (0i32, 0i32, 0i32, 0i32);
        let mut pa = p;
        let mut va = v;
        let mut n = n8;
        core::arch::asm!(
            "1:",
            step!(),
            step!(),
            step!(),
            step!(),
            "subs {n}, {n}, #1",
            "bne 1b",
            pa = inout(reg) pa,
            va = inout(reg) va,
            n = inout(reg) n,
            s00 = inout(reg) s00,
            s01 = inout(reg) s01,
            s10 = inout(reg) s10,
            s11 = inout(reg) s11,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            t3 = out(reg) _,
            options(pure, readonly, nostack),
        );
        let _ = (pa, va, n);
        [s00, s01, s10, s11]
    }
    #[cfg(not(target_arch = "arm"))]
    {
        let p = core::slice::from_raw_parts(p, ROW2 + n8 * 8);
        let v = core::slice::from_raw_parts(v, ROW2 + n8 * 8);
        let mut a = [0i32; 4];
        for j in 0..n8 * 8 {
            a[0] = a[0].wrapping_add(p[j] as i32 * v[j] as i32);
            a[1] = a[1].wrapping_add(p[j] as i32 * v[ROW2 + j] as i32);
            a[2] = a[2].wrapping_add(p[ROW2 + j] as i32 * v[j] as i32);
            a[3] = a[3].wrapping_add(p[ROW2 + j] as i32 * v[ROW2 + j] as i32);
        }
        a
    }
}

/// Two consecutive int16 rows (384 elements each, row 1 at +384) dotted
/// with one int16 vector of 384: [row0.h, row1.h].
///
/// # Safety
/// `rows` readable for 768 elements, `h` for 384.
#[inline(always)]
pub unsafe fn dot384_2rows(rows: *const i16, h: *const i16) -> [i32; 2] {
    #[cfg(target_arch = "arm")]
    {
        // Row 1 sits 768 bytes past row 0: 764 after the post-increment.
        macro_rules! step {
            () => {
                concat!(
                    "ldr {t0}, [{h}], #4\n",
                    "ldr {t1}, [{r}], #4\n",
                    "ldr {t2}, [{r}, #764]\n",
                    "smlad {s0}, {t0}, {t1}, {s0}\n",
                    "smlad {s1}, {t0}, {t2}, {s1}\n",
                )
            };
        }
        let (mut s0, mut s1) = (0i32, 0i32);
        let mut r = rows;
        let mut hp = h;
        let mut n = 48usize; // 192 pairs, 4 per iteration
        core::arch::asm!(
            "1:",
            step!(),
            step!(),
            step!(),
            step!(),
            "subs {n}, {n}, #1",
            "bne 1b",
            r = inout(reg) r,
            h = inout(reg) hp,
            n = inout(reg) n,
            s0 = inout(reg) s0,
            s1 = inout(reg) s1,
            t0 = out(reg) _,
            t1 = out(reg) _,
            t2 = out(reg) _,
            options(pure, readonly, nostack),
        );
        let _ = (r, hp, n);
        [s0, s1]
    }
    #[cfg(not(target_arch = "arm"))]
    {
        let rows = core::slice::from_raw_parts(rows, 768);
        let h = core::slice::from_raw_parts(h, 384);
        let mut a = [0i32; 2];
        for i in 0..384 {
            a[0] = a[0].wrapping_add(rows[i] as i32 * h[i] as i32);
            a[1] = a[1].wrapping_add(rows[384 + i] as i32 * h[i] as i32);
        }
        a
    }
}

/// int8 -> int16 widening, `dst[i] = src[i] as i16` for every element of
/// `src`.
///
/// Four elements per step on the DSP extension: SXTB16 sign-extends the
/// even bytes of a word, and the odd bytes after a rotate, into two
/// halfword pairs; PKHBT / PKHTB reorder them into consecutive pairs and
/// STRD stores both words. Both slices must be 4-byte aligned; a tail of
/// up to three elements is widened one at a time.
pub fn widen_i8_i16(src: &[i8], dst: &mut [i16]) {
    let n = src.len();
    assert!(dst.len() >= n);
    let n4 = n / 4;
    #[cfg(target_arch = "arm")]
    {
        if n4 > 0 {
            assert!(src.as_ptr() as usize % 4 == 0 && dst.as_ptr() as usize % 4 == 0);
            let mut s = src.as_ptr();
            let mut d = dst.as_mut_ptr();
            let mut k = n4;
            unsafe {
                core::arch::asm!(
                    "1:",
                    "ldr {w}, [{s}], #4",
                    "sxtb16 {e}, {w}",
                    "sxtb16 {o}, {w}, ror #8",
                    "pkhbt {p0}, {e}, {o}, lsl #16",
                    "pkhtb {p1}, {o}, {e}, asr #16",
                    "strd {p0}, {p1}, [{d}], #8",
                    "subs {k}, {k}, #1",
                    "bne 1b",
                    s = inout(reg) s,
                    d = inout(reg) d,
                    k = inout(reg) k,
                    w = out(reg) _,
                    e = out(reg) _,
                    o = out(reg) _,
                    p0 = out(reg) _,
                    p1 = out(reg) _,
                    options(nostack),
                );
            }
            let _ = (s, d, k);
        }
    }
    #[cfg(not(target_arch = "arm"))]
    {
        for i in 0..n4 * 4 {
            dst[i] = src[i] as i16;
        }
    }
    for i in n4 * 4..n {
        dst[i] = src[i] as i16;
    }
}

/// Two int8 rows (row 1 at +384 bytes) dotted with a permuted int16
/// vector. `h` holds, for every four columns 4i..4i+3, the pair
/// (h[4i], h[4i+2]) and then (h[4i+1], h[4i+3]): the order SXTB16
/// produces from a word of four int8 (the even bytes, then the odd bytes
/// after a rotate), so the rows are widened in registers and never
/// stored. Returns [row0 . h, row1 . h] as wrapping i32 sums.
///
/// # Safety
/// `rows` must be readable for 768 bytes and 4-byte aligned, `h` for
/// 384 i16 and 4-byte aligned (LDRD).
#[inline(always)]
pub unsafe fn dot384_2rows_i8(rows: *const i8, h: *const i16) -> [i32; 2] {
    #[cfg(target_arch = "arm")]
    {
        // One step = four columns of both rows. Row 1 sits 384 bytes past
        // row 0: 380 after the post-increment.
        macro_rules! step {
            () => {
                concat!(
                    "ldrd {ha}, {hb}, [{h}], #8\n",
                    "ldr {w0}, [{r}], #4\n",
                    "ldr {w1}, [{r}, #380]\n",
                    "sxtb16 {t}, {w0}\n",
                    "smlad {s0}, {t}, {ha}, {s0}\n",
                    "sxtb16 {t}, {w0}, ror #8\n",
                    "smlad {s0}, {t}, {hb}, {s0}\n",
                    "sxtb16 {t}, {w1}\n",
                    "smlad {s1}, {t}, {ha}, {s1}\n",
                    "sxtb16 {t}, {w1}, ror #8\n",
                    "smlad {s1}, {t}, {hb}, {s1}\n",
                )
            };
        }
        let (mut s0, mut s1) = (0i32, 0i32);
        let mut r = rows;
        let mut hp = h;
        let mut n = 24usize; // 96 steps of four columns, 4 per iteration
        core::arch::asm!(
            "1:",
            step!(),
            step!(),
            step!(),
            step!(),
            "subs {n}, {n}, #1",
            "bne 1b",
            r = inout(reg) r,
            h = inout(reg) hp,
            n = inout(reg) n,
            s0 = inout(reg) s0,
            s1 = inout(reg) s1,
            ha = out(reg) _,
            hb = out(reg) _,
            w0 = out(reg) _,
            w1 = out(reg) _,
            t = out(reg) _,
            options(pure, readonly, nostack),
        );
        let _ = (r, hp, n);
        [s0, s1]
    }
    #[cfg(not(target_arch = "arm"))]
    {
        let rows = core::slice::from_raw_parts(rows, 768);
        let h = core::slice::from_raw_parts(h, 384);
        let mut a = [0i32; 2];
        for i in 0..96 {
            let c = 4 * i;
            // h's pairs back to column order
            let hc = [h[c], h[c + 2], h[c + 1], h[c + 3]];
            for j in 0..4 {
                a[0] = a[0].wrapping_add(rows[c + j] as i32 * hc[j] as i32);
                a[1] = a[1].wrapping_add(rows[384 + c + j] as i32 * hc[j] as i32);
            }
        }
        a
    }
}

/// Index of column `c` of the hidden vector in the layout
/// `dot384_2rows_i8` expects (the middle two of every four swapped).
#[inline(always)]
pub fn perm4(c: usize) -> usize {
    (c & !3) | ((c & 1) << 1) | ((c >> 1) & 1)
}

/// One int16 query (64 elements in `perm4` order) against four
/// consecutive int8 key rows of 64: [q.k0, q.k1, q.k2, q.k3].
///
/// The key words are widened in registers with SXTB16 (even bytes, then
/// the odd bytes after a rotate), which is why the query is permuted:
/// each key word then meets the two query pairs it belongs with. 21
/// instructions per four columns of four keys.
///
/// # Safety
/// `q` readable for 64 i16 and 4-byte aligned; `k` readable for 256
/// bytes and 4-byte aligned.
#[inline(always)]
pub unsafe fn dot64_q1k4_i8(q: *const i16, k: *const i8) -> [i32; 4] {
    #[cfg(target_arch = "arm")]
    {
        // one step = four columns; rows 1..3 sit 64, 128, 192 bytes past
        // row 0 (60, 124, 188 after the post-increment)
        macro_rules! row {
            ($w:tt, $a:tt) => {
                concat!(
                    "sxtb16 {t}, {", $w, "}\n",
                    "smlad {", $a, "}, {t}, {qa}, {", $a, "}\n",
                    "sxtb16 {t}, {", $w, "}, ror #8\n",
                    "smlad {", $a, "}, {t}, {qb}, {", $a, "}\n",
                )
            };
        }
        macro_rules! step {
            () => {
                concat!(
                    "ldrd {qa}, {qb}, [{q}], #8\n",
                    "ldr {w0}, [{k}], #4\n",
                    "ldr {w1}, [{k}, #60]\n",
                    row!("w0", "a0"),
                    row!("w1", "a1"),
                    "ldr {w0}, [{k}, #124]\n",
                    "ldr {w1}, [{k}, #188]\n",
                    row!("w0", "a2"),
                    row!("w1", "a3"),
                )
            };
        }
        macro_rules! rep4 {
            ($s:expr) => {
                concat!($s, $s, $s, $s)
            };
        }
        let (mut a0, mut a1, mut a2, mut a3) = (0i32, 0i32, 0i32, 0i32);
        let mut qp = q;
        let mut kp = k;
        core::arch::asm!(
            rep4!(step!()),
            rep4!(step!()),
            rep4!(step!()),
            rep4!(step!()),
            q = inout(reg) qp,
            k = inout(reg) kp,
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            a2 = inout(reg) a2,
            a3 = inout(reg) a3,
            qa = out(reg) _,
            qb = out(reg) _,
            w0 = out(reg) _,
            w1 = out(reg) _,
            t = out(reg) _,
            options(pure, readonly, nostack),
        );
        let _ = (qp, kp);
        [a0, a1, a2, a3]
    }
    #[cfg(not(target_arch = "arm"))]
    {
        let q = core::slice::from_raw_parts(q, 64);
        let k = core::slice::from_raw_parts(k, 256);
        let mut a = [0i32; 4];
        for r in 0..4 {
            for c in 0..64 {
                a[r] = a[r].wrapping_add(k[r * 64 + c] as i32 * q[perm4(c)] as i32);
            }
        }
        a
    }
}

/// One int16 probability row (`n8 * 8` keys in `perm4` order) against
/// four int8 value rows 64 bytes apart: `acc[r] + p . v_r`.
///
/// The value rows are a tile's channel rows ([64 channels][64 keys]
/// int8), so a full row of keys is accumulated tile by tile through
/// `acc`. `n8` must be >= 1 and <= 8.
///
/// # Safety
/// `p` readable for n8 * 8 i16 and 4-byte aligned; `v` readable for
/// 192 + n8 * 8 bytes and 4-byte aligned.
#[inline(always)]
pub unsafe fn dot_pv_q1v4_i8(p: *const i16, v: *const i8, n8: usize, acc: [i32; 4]) -> [i32; 4] {
    debug_assert!((1..=8).contains(&n8));
    #[cfg(target_arch = "arm")]
    {
        macro_rules! row {
            ($w:tt, $a:tt) => {
                concat!(
                    "sxtb16 {t}, {", $w, "}\n",
                    "smlad {", $a, "}, {t}, {pa}, {", $a, "}\n",
                    "sxtb16 {t}, {", $w, "}, ror #8\n",
                    "smlad {", $a, "}, {t}, {pb}, {", $a, "}\n",
                )
            };
        }
        macro_rules! step {
            () => {
                concat!(
                    "ldrd {pa}, {pb}, [{p}], #8\n",
                    "ldr {w0}, [{v}], #4\n",
                    "ldr {w1}, [{v}, #60]\n",
                    row!("w0", "a0"),
                    row!("w1", "a1"),
                    "ldr {w0}, [{v}, #124]\n",
                    "ldr {w1}, [{v}, #188]\n",
                    row!("w0", "a2"),
                    row!("w1", "a3"),
                )
            };
        }
        let [mut a0, mut a1, mut a2, mut a3] = acc;
        let mut pp = p;
        let mut vp = v;
        let mut n = n8;
        core::arch::asm!(
            "1:",
            step!(),
            step!(),
            "subs {n}, {n}, #1",
            "bne 1b",
            p = inout(reg) pp,
            v = inout(reg) vp,
            n = inout(reg) n,
            a0 = inout(reg) a0,
            a1 = inout(reg) a1,
            a2 = inout(reg) a2,
            a3 = inout(reg) a3,
            pa = out(reg) _,
            pb = out(reg) _,
            w0 = out(reg) _,
            w1 = out(reg) _,
            t = out(reg) _,
            options(pure, readonly, nostack),
        );
        let _ = (pp, vp, n);
        [a0, a1, a2, a3]
    }
    #[cfg(not(target_arch = "arm"))]
    {
        let n = n8 * 8;
        let p = core::slice::from_raw_parts(p, n);
        let v = core::slice::from_raw_parts(v, 192 + n);
        let mut a = acc;
        for r in 0..4 {
            for j in 0..n {
                a[r] = a[r].wrapping_add(p[perm4(j)] as i32 * v[r * 64 + j] as i32);
            }
        }
        a
    }
}

/// On-target check of every asm body against a scalar evaluation of the
/// same definition, on pseudo-random data. The host comparators compile
/// only the portable bodies, so this is what actually exercises the
/// instructions. `scratch` needs 8 KB, 4-byte aligned. Returns a bit per
/// failing primitive (0 = all agree); on the host it is a no-op.
pub fn selftest(scratch: &mut [u8]) -> u32 {
    assert!(scratch.len() >= 8192 && scratch.as_ptr() as usize % 4 == 0);
    let mut x = 0x2545_f491u32;
    for b in scratch.iter_mut() {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *b = (x >> 24) as u8;
    }
    let mut bad = 0u32;
    let i8s = |off: usize, n: usize| -> &[i8] {
        // SAFETY: in-bounds view of the caller's scratch
        unsafe { core::slice::from_raw_parts(scratch.as_ptr().add(off) as *const i8, n) }
    };
    let i16s = |off: usize, n: usize| -> &[i16] {
        unsafe { core::slice::from_raw_parts(scratch.as_ptr().add(off) as *const i16, n) }
    };

    // round_i32: halves, negatives, large, tiny
    for &v in &[0.5f32, 1.5, 2.5, -0.5, -1.5, -2.5, 0.49999, -0.49999, 12345.5, -12345.5,
                3.0e9, -3.0e9, 1.0e-30, -0.0, 0.0] {
        if round_i32(v) != libm::roundf(v) as i32 {
            bad |= 1;
        }
    }
    // byte_sum with a tail that is not a multiple of 32, aligned and not
    {
        for &(off, n) in &[(0usize, 1003usize), (0, 32), (0, 31), (1, 1000), (4, 96)] {
            let b = &scratch[off..off + n];
            let want = b.iter().fold(0u32, |a, &v| a.wrapping_add(v as u32));
            if byte_sum(b) != want {
                bad |= 2;
            }
        }
    }
    // dot64_2x2: q rows at 0 and 64, k rows at 0 and 64 (i16 each)
    {
        let q = i16s(0, 128);
        let k = i16s(256, 128);
        let r = unsafe { dot64_2x2(q.as_ptr(), k.as_ptr()) };
        let mut w = [0i32; 4];
        for c in 0..64 {
            w[0] = w[0].wrapping_add(q[c] as i32 * k[c] as i32);
            w[1] = w[1].wrapping_add(q[c] as i32 * k[64 + c] as i32);
            w[2] = w[2].wrapping_add(q[64 + c] as i32 * k[c] as i32);
            w[3] = w[3].wrapping_add(q[64 + c] as i32 * k[64 + c] as i32);
        }
        if r != w {
            bad |= 4;
        }
    }
    // dot_pv_2x2 over 24 keys, rows ROW2 apart
    {
        let n8 = 3;
        let p = i16s(512, ROW2 + n8 * 8);
        let v = i16s(512 + 2 * (ROW2 + n8 * 8), ROW2 + n8 * 8);
        let r = unsafe { dot_pv_2x2(p.as_ptr(), v.as_ptr(), n8) };
        let mut w = [0i32; 4];
        for j in 0..n8 * 8 {
            w[0] = w[0].wrapping_add(p[j] as i32 * v[j] as i32);
            w[1] = w[1].wrapping_add(p[j] as i32 * v[ROW2 + j] as i32);
            w[2] = w[2].wrapping_add(p[ROW2 + j] as i32 * v[j] as i32);
            w[3] = w[3].wrapping_add(p[ROW2 + j] as i32 * v[ROW2 + j] as i32);
        }
        if r != w {
            bad |= 8;
        }
    }
    // dot384_2rows: int16 rows at 0 and 384, vector after them
    {
        let rows = i16s(4096, 768);
        let h = i16s(4096 + 1536, 384);
        let r = unsafe { dot384_2rows(rows.as_ptr(), h.as_ptr()) };
        let mut w = [0i32; 2];
        for i in 0..384 {
            w[0] = w[0].wrapping_add(rows[i] as i32 * h[i] as i32);
            w[1] = w[1].wrapping_add(rows[384 + i] as i32 * h[i] as i32);
        }
        if r != w {
            bad |= 16;
        }
    }
    // dot384_2rows_i8: int8 rows, the vector in perm4 order
    {
        let rows = i8s(6144, 768);
        let h = i16s(4096 + 1536, 384);
        let r = unsafe { dot384_2rows_i8(rows.as_ptr(), h.as_ptr()) };
        let mut w = [0i32; 2];
        for c in 0..384 {
            let hc = h[perm4(c)] as i32;
            w[0] = w[0].wrapping_add(rows[c] as i32 * hc);
            w[1] = w[1].wrapping_add(rows[384 + c] as i32 * hc);
        }
        if r != w {
            bad |= 32;
        }
    }
    // dot64_q1k4_i8: permuted int16 query, four int8 key rows
    {
        let q = i16s(0, 64);
        let k = i8s(6144, 256);
        let r = unsafe { dot64_q1k4_i8(q.as_ptr(), k.as_ptr()) };
        let mut w = [0i32; 4];
        for rr in 0..4 {
            for c in 0..64 {
                w[rr] = w[rr].wrapping_add(k[rr * 64 + c] as i32 * q[perm4(c)] as i32);
            }
        }
        if r != w {
            bad |= 128;
        }
    }
    // dot_pv_q1v4_i8 over 24 keys with a running accumulator
    {
        let n8 = 3;
        let p = i16s(512, 24);
        let v = i8s(6144, 192 + 24);
        let r = unsafe { dot_pv_q1v4_i8(p.as_ptr(), v.as_ptr(), n8, [1, 2, 3, 4]) };
        let mut w = [1i32, 2, 3, 4];
        for rr in 0..4 {
            for j in 0..24 {
                w[rr] = w[rr].wrapping_add(p[perm4(j)] as i32 * v[rr * 64 + j] as i32);
            }
        }
        if r != w {
            bad |= 256;
        }
    }
    // widen_i8_i16 with a three-element tail, into the scratch tail
    {
        let n = 259;
        let src: [i8; 259] = core::array::from_fn(|i| i8s(6144, 768)[i]);
        let mut dst = [0i16; 259];
        widen_i8_i16(&src, &mut dst);
        for i in 0..n {
            if dst[i] != src[i] as i16 {
                bad |= 64;
            }
        }
    }
    bad
}
