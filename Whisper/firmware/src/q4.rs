//! 4-bit weight unpacking, the integer-exact mirror of model/quant4.py
//! (keep the two in lockstep; tools/q4check cross-checks them against
//! shared test vectors).
//!
//! Groups of 64 consecutive int8 weights share one u8 amax; nibbles are
//! stored biased (+8). Reconstruction, round-half-away-from-zero in pure
//! integers:  w' = sign(nib) * min(127, (|nib| * amax * 2 + 7) / 14).
//!
//! Packed blob entry layout (little-endian u32 header words):
//!   "LAY4" | raw_len | w_off | n_weights | raw_sum
//!   raw[0..w_off] verbatim | amax[n/64] | nibbles[n/2]
//! raw_sum is the wrapping u32 byte-sum of the RAW blob the entry
//! expands to (head + reconstructed weights); the loader checks the
//! expansion against it once per boot. The embedding entry (embp4) is
//! chunks of 64 rows x 384 with the same amax|nibbles layout and no
//! header (block-padded per chunk).

pub const G: usize = 64;
pub const MAGIC: u32 = 0x3459_414C; // "LAY4"
pub const HDR: usize = 20; // five u32 header words

/// Reconstruction table for one group: lut[raw nibble 0..16].
/// Index 0 (nibble -8) is never emitted by the packer; it decodes to the
/// clamped -127 for defensiveness.
#[inline]
pub fn lut(amax: u8) -> [i8; 16] {
    let a = amax as i32;
    core::array::from_fn(|k| {
        let n = k as i32 - 8;
        let v = (n.abs() * a * 2 + 7) / 14;
        let v = if v > 127 { 127 } else { v };
        (if n < 0 { -v } else { v }) as i8
    })
}

/// Expand `n` weights (n % 128 == 0) from `nibs`/`amax` into `dst`.
///
/// Raw pointers because the slot loader expands in place: `dst` may
/// overlap the tail of the nibble stream provided the stream starts at
/// least n/2 bytes past `dst` (the writer advances 2 bytes per source
/// byte and never catches the reader; callers must check).
///
/// # Safety
/// `nibs` must be readable for n/2 bytes, `dst` writable for n bytes,
/// and any overlap must satisfy the margin above.
pub unsafe fn unpack_raw(amax: &[u8], nibs: *const u8, dst: *mut i8, n: usize) {
    debug_assert!(n % (2 * G) == 0 && amax.len() >= n / G);
    let mut src = nibs;
    let mut d = dst;
    for &a in &amax[..n / G] {
        let t = lut(a);
        for _ in 0..G / 2 {
            let b = *src as usize;
            *d = t[b & 15];
            *d.add(1) = t[b >> 4];
            src = src.add(1);
            d = d.add(2);
        }
    }
}

/// Safe wrapper for disjoint buffers (the embedding chunk path).
pub fn unpack(amax: &[u8], nibs: &[u8], dst: &mut [i8]) {
    let n = dst.len();
    assert!(nibs.len() >= n / 2);
    unsafe { unpack_raw(amax, nibs.as_ptr(), dst.as_mut_ptr(), n) }
}
