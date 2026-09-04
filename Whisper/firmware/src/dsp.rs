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
pub fn byte_sum(b: &[u8]) -> u32 {
    let mut s = 0u32;
    let n16 = b.len() / 16;
    #[cfg(target_arch = "arm")]
    {
        if n16 > 0 {
            let mut p = b.as_ptr();
            let mut n = n16;
            unsafe {
                core::arch::asm!(
                    "1:",
                    "ldr {w0}, [{p}], #4",
                    "ldr {w1}, [{p}], #4",
                    "ldr {w2}, [{p}], #4",
                    "ldr {w3}, [{p}], #4",
                    "usada8 {s}, {w0}, {z}, {s}",
                    "usada8 {s}, {w1}, {z}, {s}",
                    "usada8 {s}, {w2}, {z}, {s}",
                    "usada8 {s}, {w3}, {z}, {s}",
                    "subs {n}, {n}, #1",
                    "bne 1b",
                    p = inout(reg) p,
                    n = inout(reg) n,
                    s = inout(reg) s,
                    z = in(reg) 0u32,
                    w0 = out(reg) _,
                    w1 = out(reg) _,
                    w2 = out(reg) _,
                    w3 = out(reg) _,
                    options(readonly, nostack),
                );
            }
            let _ = (p, n);
        }
    }
    #[cfg(not(target_arch = "arm"))]
    {
        for &x in &b[..n16 * 16] {
            s = s.wrapping_add(x as u32);
        }
    }
    for &x in &b[n16 * 16..] {
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
        // +124 (= 128 - 4) offsets. Fully unrolled: 32 steps.
        macro_rules! step {
            () => {
                concat!(
                    "ldr {t0}, [{qa}], #4\n",
                    "ldr {t2}, [{ka}], #4\n",
                    "smlad {a00}, {t0}, {t2}, {a00}\n",
                    "ldr {t3}, [{ka}, #124]\n",
                    "smlad {a01}, {t0}, {t3}, {a01}\n",
                    "ldr {t0}, [{qa}, #124]\n",
                    "smlad {a10}, {t0}, {t2}, {a10}\n",
                    "smlad {a11}, {t0}, {t3}, {a11}\n",
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
        // Second-row offset after the post-increment: ROW2 * 2 - 4.
        macro_rules! step {
            () => {
                concat!(
                    "ldr {t0}, [{pa}], #4\n",
                    "ldr {t2}, [{va}], #4\n",
                    "smlad {s00}, {t0}, {t2}, {s00}\n",
                    "ldr {t3}, [{va}, #1276]\n",
                    "smlad {s01}, {t0}, {t3}, {s01}\n",
                    "ldr {t0}, [{pa}, #1276]\n",
                    "smlad {s10}, {t0}, {t2}, {s10}\n",
                    "smlad {s11}, {t0}, {t3}, {s11}\n",
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
                    "smlad {s0}, {t0}, {t1}, {s0}\n",
                    "ldr {t2}, [{r}, #764]\n",
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
