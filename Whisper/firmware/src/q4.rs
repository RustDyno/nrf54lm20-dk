//! 4-bit weight unpacking, the integer-exact mirror of model/quant4.py
//! (keep the two in lockstep; tools/q4check cross-checks them against
//! shared test vectors).
//!
//! Groups of consecutive int8 weights share one scale byte; nibbles are
//! stored biased (+8). Reconstruction, round-half-away-from-zero in pure
//! integers:  w' = sign(nib) * min(127, (|nib| * amax * 2 + 7) / 14).
//! Two codings: the LM-head chunks ("embc4") use G = 64 and the raw amax;
//! the decoder blobs ("LAY5") use GL = 16 and a 6-bit code for an odd
//! scale (chosen by the packer's error search), so a 64 x 256 pair table
//! expands two weights per lookup (expand_lvl).
//!
//! Packed blob entry layout (little-endian u32 header words):
//!   "LAY5" | raw_len | w_off | n_weights | raw_sum | base
//!   raw[0..w_off] verbatim | codes[n/GL] | nibbles[n/2]
//! raw_sum is the wrapping u32 byte-sum of the RAW blob the entry
//! expands to (head + reconstructed weights); the loader checks the
//! expansion against it once per boot; base is the RAM address the raw
//! blob was linked at. The embedding entry (embc4) is chunks of 64 rows
//! x 384 with the G = 64 amax|nibbles layout and no header (block-padded
//! per chunk).

pub const G: usize = 64;
/// Level-coded decoder blobs ("LAY5", see model/quant4.py): groups of GL
/// weights sharing an odd scale 1..127, stored as the 6-bit code
/// (scale >> 1). Header: magic, raw_len, w_off, n_weights, raw_sum,
/// base (RAM address the raw blob was linked at).
pub const GL: usize = 16;
pub const MAGIC5: u32 = 0x3559_414C; // "LAY5"
pub const HDR5: usize = 24;
/// Expansion table: u16[64 codes][256 nibble-byte values], low byte the
/// weight of the low nibble, high byte the weight of the high nibble.
pub const LVL_TABLE_LEN: usize = 64 * 256;
/// Blocks per LM-head embedding chunk ("embc4": 64 f32 scales, 64 u32
/// ids, 384 amax bytes, 12288 nibble bytes = 13184 B, block-padded).
pub const EMB_CHUNK_BLOCKS: usize = 26;

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

/// Safe wrapper for disjoint buffers (tools/q4check; the firmware's
/// embedding path now expands to int16 below).
#[allow(dead_code)]
pub fn unpack(amax: &[u8], nibs: &[u8], dst: &mut [i8]) {
    let n = dst.len();
    assert!(nibs.len() >= n / 2);
    unsafe { unpack_raw(amax, nibs.as_ptr(), dst.as_mut_ptr(), n) }
}

/// Build the level-coded expansion table (32 KB, once per decode).
pub fn table_lvl(out: &mut [u16; LVL_TABLE_LEN]) {
    for c in 0..64 {
        let a = (2 * c + 1) as i32;
        let lut: [u8; 16] = core::array::from_fn(|k| {
            let n = k as i32 - 8;
            let v = (n.abs() * a * 2 + 7) / 14;
            let v = if v > 127 { 127 } else { v };
            (if n < 0 { -v } else { v }) as i8 as u8
        });
        for b in 0..256 {
            out[c * 256 + b] = lut[b & 15] as u16 | ((lut[b >> 4] as u16) << 8);
        }
    }
}

/// Expand `n_groups` level-coded groups (GL weights each) from `codes`
/// and `nibs` into `dst` (n_groups * GL bytes, 4-byte aligned).
///
/// Hot path of the decode blob pipeline (147456 weights per blob, 56
/// blobs per token): each source word (8 nibbles) becomes two words of
/// four int8 weights through the group's 256-entry pair table.
///
/// # Safety
/// `codes` readable for n_groups bytes, `nibs` for n_groups * GL / 2
/// bytes (any alignment), `dst` writable for n_groups * GL bytes and
/// disjoint from `nibs`.
#[inline(never)]
pub unsafe fn expand_lvl(table: &[u16; LVL_TABLE_LEN], codes: *const u8, n_groups: usize,
                         nibs: *const u8, dst: *mut u8) {
    debug_assert!(dst as usize % 4 == 0);
    let mut sp = nibs as *const u32;
    let mut dp = dst as *mut u32;
    for g in 0..n_groups {
        let c = (*codes.add(g) & 63) as usize;
        let t = table.as_ptr().add(c * 256);
        for _ in 0..GL / 8 {
            let b = sp.read_unaligned();
            let w0 = *t.add((b & 255) as usize) as u32
                | ((*t.add(((b >> 8) & 255) as usize) as u32) << 16);
            let w1 = *t.add(((b >> 16) & 255) as usize) as u32
                | ((*t.add((b >> 24) as usize) as u32) << 16);
            *dp = w0;
            *dp.add(1) = w1;
            sp = sp.add(1);
            dp = dp.add(2);
        }
    }
}

/// Safe wrapper for disjoint buffers (tools/q4check).
#[allow(dead_code)]
pub fn expand_lvl_slice(table: &[u16; LVL_TABLE_LEN], codes: &[u8], nibs: &[u8],
                        dst: &mut [i8]) {
    let n = dst.len();
    assert!(n % GL == 0 && codes.len() >= n / GL && nibs.len() >= n / 2);
    assert!(dst.as_ptr() as usize % 4 == 0);
    unsafe { expand_lvl(table, codes.as_ptr(), n / GL, nibs.as_ptr(), dst.as_mut_ptr() as *mut u8) }
}

/// Reconstruction tables for every possible amax, as the u16 bit patterns
/// of the int16 weights (16 KB). Building one table costs ~300 cycles
/// (a division per entry); the LM head meets ~73 k groups per token, so
/// they are built once per decode and indexed by amax.
pub const TABLES16: usize = 256 * 16;

pub fn tables16(out: &mut [u32; TABLES16]) {
    for a in 0..256 {
        let t = lut(a as u8);
        for k in 0..16 {
            out[a * 16 + k] = t[k] as i16 as u16 as u32;
        }
    }
}

/// Expand `dst.len()` weights (a multiple of G) into int16.
///
/// Hot path of the LM head (4.7 M nibbles per token): each source word
/// (8 nibbles) becomes four u32 stores of two int16 weights each through
/// the group's 16-entry table.
pub fn unpack16(tables: &[u32; TABLES16], amax: &[u8], nibs: &[u8], dst: &mut [i16]) {
    let n = dst.len();
    assert!(n % G == 0 && nibs.len() >= n / 2 && amax.len() >= n / G);
    assert!(dst.as_ptr() as usize % 4 == 0);
    let mut sp = nibs.as_ptr();
    let mut dp = dst.as_mut_ptr() as *mut u32;
    for &a in &amax[..n / G] {
        let t = &tables[a as usize * 16..a as usize * 16 + 16];
        // SAFETY: bounds asserted above; G/2 source bytes and G/2 u32
        // (= G i16) destination words per group. Source words may be
        // unaligned (read_unaligned).
        unsafe {
            for _ in 0..G / 8 {
                let b = (sp as *const u32).read_unaligned();
                *dp = t[(b & 15) as usize] | (t[((b >> 4) & 15) as usize] << 16);
                *dp.add(1) = t[((b >> 8) & 15) as usize] | (t[((b >> 12) & 15) as usize] << 16);
                *dp.add(2) = t[((b >> 16) & 15) as usize] | (t[((b >> 20) & 15) as usize] << 16);
                *dp.add(3) = t[((b >> 24) & 15) as usize] | (t[(b >> 28) as usize] << 16);
                sp = sp.add(4);
                dp = dp.add(4);
            }
        }
    }
}
