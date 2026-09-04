//! Standalone Whisper: record from the PDM mic, transcribe, print over RTT.
//! No host in the data path -- weights and scratch live on the storage
//! device (USB stick via usb.rs host mode, or SD card via sd.rs; the
//! storage module picks at init and everything here is backend-agnostic).
//!
//! The pipeline mirrors the hardware-verified host drivers step for step
//! (tape-encoder and whisper-host decode); only the paging backend changed
//! from SWD to SD. Activations are stored TILE-MAJOR ([C, 64] chunks) so
//! every SD transfer is contiguous; attention K/V live as per-head 4 KB
//! blocks and are reassembled into planar buffers in RAM at use.
//!
//! The SD image (model/make_sd_image.py) provides all weights, tables and
//! the binary plan (quantization parameters + pruned vocabulary). The
//! image builder and `Plan::load` are ONE contract: same field order.

use crate::kernels::{self, Quant};
use crate::{display, mel, pdm, sd, slot, storage};
use rtt_target::rprintln;

use crate::dsp;

const C: usize = 384;
const HD: usize = 64;
const HEADS: usize = 6;
const T: usize = 64; // frame tile
const CTX: usize = 600;
const PAD_W: usize = 640;
const N_TILES: usize = PAD_W / T; // 10
const MEL_W: usize = 2 * PAD_W; // 1280 mel frames
const MEL_TILES: usize = MEL_W / T; // 20
const N_SAMPLES: usize = 192_000; // 12 s at 16 kHz
pub const MAX_TOKENS: usize = 32;
const BLOCKS: usize = 4;

const TILE8: usize = C * T; // 24576 B
const TILE16: usize = 2 * TILE8;
const HB: usize = HD * T; // head block, 4096 B
const HREG_BLOCKS: u32 = (HEADS * N_TILES * HB / storage::BLOCK) as u32; // 480

// --- SD scratch regions (block offsets from plan.scratch_lba) ----------------
const S_PCM: u32 = 0; // 750 blocks
// Marker block written by tools/mockusb when S_PCM already holds a fixed
// clip: the run then skips the microphone so it is reproducible and
// directly comparable to the host mirror. Sits in the gap between S_PCM's
// 750 blocks and S_MELF.
const S_INJECT: u32 = 750;
const S_MELF: u32 = 768; // pass-1 f32 mel chunks, 20 x 40 blocks
const S_MEL: u32 = 1600; // int8 mel tiles [80,64], 20 x 10 blocks
const S_A: u32 = 1856; // mel-rate int8 tiles, 20 x 48 blocks
const S_X: u32 = 2880; // int16 residual tiles, 10 x 96 blocks
const S_LN: u32 = 3904; // int8 tiles, 10 x 48
const S_O: u32 = 4416; // int8 tiles, 10 x 48
const S_QH: u32 = 4928; // head-block regions, 480 each
const S_KH: u32 = 5440;
const S_VH: u32 = 5952;
const S_CH: u32 = 6464;
const S_P: u32 = 6976; // fc2 partials, 4 x 480
const S_EO: u32 = 8960; // encoder output tiles, 480
const S_XKV: u32 = 9472; // cross K/V head blocks, 8 x 480 (l*2 + [k|v])

// --- arena layout -------------------------------------------------------------
// Encoder phase: A_IN holds the largest assembled input (conv2: 49920 B).
const A_IN: usize = 0; // to 50176
const A_OUT: usize = 50176; // 24576, to 74752
const A_AUX: usize = 74752; // 12288, to 87040
const A_GB: usize = 87040; // 3072, to 90112
const A_LUT: usize = 90112; // 512, to 90624

fn arena(off: usize, len: usize) -> &'static mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(crate::ARENA) as *mut u8).add(off),
            len,
        )
    }
}

fn arena_addr(off: usize) -> u32 {
    unsafe { (core::ptr::addr_of!(crate::ARENA) as *const u8).add(off) as u32 }
}

fn as_i8(off: usize, len: usize) -> &'static [i8] {
    unsafe { core::slice::from_raw_parts(arena_addr(off) as *const i8, len) }
}

fn as_i8_mut(off: usize, len: usize) -> &'static mut [i8] {
    unsafe { core::slice::from_raw_parts_mut(arena_addr(off) as *mut i8, len) }
}

fn as_i16(off: usize, len: usize) -> &'static [i16] {
    unsafe { core::slice::from_raw_parts(arena_addr(off) as *const i16, len) }
}

fn as_i16_mut(off: usize, len: usize) -> &'static mut [i16] {
    unsafe { core::slice::from_raw_parts_mut(arena_addr(off) as *mut i16, len) }
}

fn as_f32(off: usize, len: usize) -> &'static [f32] {
    unsafe { core::slice::from_raw_parts(arena_addr(off) as *const f32, len) }
}

/// The interlayer buffer is idle outside NPU runs; phases may borrow it
/// as scratch (mel filterbank) but nothing may park data there across an
/// NPU inference -- every blob DMAs activations through it.
fn interlayer(len: usize) -> &'static mut [u8] {
    interlayer_at(0, len)
}

fn interlayer_at(off: usize, len: usize) -> &'static mut [u8] {
    assert!(off + len <= crate::INTERLAYER_BUFFER_BYTES);
    unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(crate::nrf_axon_interlayer_buffer) as *mut u8).add(off),
            len,
        )
    }
}

// --- the weight slot as CPU scratch --------------------------------------------
//
// During CPU attention no blob is needed (the next NPU stage reloads its
// own in ~18 ms), so the 208 KB slot holds the key-major int16 keys, the
// int16 values and a staging area for one head's tile blocks. Anything
// that uses it sets Ctxt::loaded back to "nothing".
const SL_KT: usize = 0; // i16 [640][64], 81920 B
const SL_V16: usize = SL_KT + 2 * kernels::KV16_LEN; // i16 [64][640], 81920 B
const SL_STAGE: usize = SL_V16 + 2 * kernels::KV16_LEN; // one head's tile blocks
const _: () = assert!(SL_STAGE + N_TILES * HB <= slot::SLOT_BYTES);
// Interlayer buffer during CPU attention: the two-query scratch, then
// the head's context tiles staged for a single write.
const IL_SCRATCH2: usize = 0;
const IL_CTX: usize = 16384;
const _: () = assert!(core::mem::size_of::<kernels::AttnScratch2>() <= IL_CTX);
const _: () = assert!(IL_CTX + N_TILES * HB <= crate::INTERLAYER_BUFFER_BYTES);

fn slot_u8(off: usize, len: usize) -> &'static mut [u8] {
    assert!(off + len <= slot::SLOT_BYTES);
    unsafe { core::slice::from_raw_parts_mut((slot::SLOT_BASE + off) as *mut u8, len) }
}

fn slot_i8(off: usize, len: usize) -> &'static [i8] {
    assert!(off + len <= slot::SLOT_BYTES);
    unsafe { core::slice::from_raw_parts((slot::SLOT_BASE + off) as *const i8, len) }
}

fn slot_i16(off: usize, len: usize) -> &'static mut [i16] {
    assert!(off % 2 == 0 && off + 2 * len <= slot::SLOT_BYTES);
    unsafe { core::slice::from_raw_parts_mut((slot::SLOT_BASE + off) as *mut i16, len) }
}

fn attn_scratch2() -> &'static mut kernels::AttnScratch2 {
    let b = interlayer_at(IL_SCRATCH2, core::mem::size_of::<kernels::AttnScratch2>());
    unsafe { &mut *(b.as_mut_ptr() as *mut kernels::AttnScratch2) }
}

/// Softmax exponential for the DSP attention kernels: `dsp::exp_neg` is
/// within ~2 ulp of libm::expf at about a third of the cost; `false`
/// reproduces the numpy golden bit for bit (tools/attncheck reports what
/// the fast one moves).
const FAST_EXP: bool = true;

#[inline(always)]
fn softmax_exp(x: f32) -> f32 {
    if FAST_EXP {
        dsp::exp_neg(x)
    } else {
        libm::expf(x)
    }
}

/// Transposed keys for one head from its tile blocks staged in the slot
/// ([tile][64 channels][64 keys] int8): key-major int16 rows in SL_KT.
/// Tile-wise slice loops rather than the generic kernels::attn_prepare_kt
/// closure form: at decode this runs per token and per head, where the
/// index arithmetic was costing more than the attention itself.
fn prepare_kv_from_stage(tk: usize) {
    let tkp = kernels::keys_padded(tk);
    let stage = slot_i8(SL_STAGE, N_TILES * HB);
    let kt = slot_i16(SL_KT, kernels::KV16_LEN);
    let kp = kt.as_mut_ptr();
    for i in 0..tk.div_ceil(T) {
        let take = T.min(tk - i * T);
        for ch in 0..HD {
            let src = &stage[i * HB + ch * T..i * HB + ch * T + take];
            // SAFETY: destination index (i*T + col)*HD + ch < tk*HD <= KV16_LEN
            let mut d = unsafe { kp.add(i * T * HD + ch) };
            for &x in src {
                unsafe {
                    *d = x as i16;
                    d = d.add(HD);
                }
            }
        }
    }
    kt[tk * HD..tkp * HD].fill(0);
}

/// Widened values for one head, same staging: int16 [64][MAX_KEYS] in
/// SL_V16, columns tk..tkp zero.
fn prepare_v_from_stage(tk: usize) {
    let tkp = kernels::keys_padded(tk);
    let stage = slot_i8(SL_STAGE, N_TILES * HB);
    let v16 = slot_i16(SL_V16, kernels::KV16_LEN);
    for ch in 0..HD {
        let row = &mut v16[ch * kernels::MAX_KEYS..ch * kernels::MAX_KEYS + tkp];
        for i in 0..tk.div_ceil(T) {
            let take = T.min(tk - i * T);
            let src = &stage[i * HB + ch * T..i * HB + ch * T + take];
            let dst = &mut row[i * T..i * T + take];
            for (d, &x) in dst.iter_mut().zip(src) {
                *d = x as i16;
            }
        }
        row[tk..].fill(0);
    }
}

// --- CPU phase profile ---------------------------------------------------------
const P_ATTN: usize = 0;
const P_LM: usize = 1;
const P_UNPACK: usize = 2;
const P_SUM: usize = 3;
static mut PROF: [u64; 4] = [0; 4];

fn cycles() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

fn prof_add(i: usize, t0: u32) {
    unsafe {
        (*core::ptr::addr_of_mut!(PROF))[i] += cycles().wrapping_sub(t0) as u64;
    }
}

// --- SD image index + plan -----------------------------------------------------

pub const IMG_MAGIC: &[u8; 8] = b"WSPRIMG1";
const PLAN_MAGIC: u32 = 0x4E4C_5057; // "WPLN"

static mut INDEX: [u8; 8192] = [0; 8192];

#[derive(Clone, Copy, Default)]
pub struct Entry {
    pub lba: u32,
    pub len: u32,
}

fn lookup(name: &str) -> Option<Entry> {
    lookup_idx(name).map(|(e, _)| e)
}

/// Like lookup, but also returns the entry's index (key into SUMS).
fn lookup_idx(name: &str) -> Option<(Entry, usize)> {
    let idx = unsafe { &*core::ptr::addr_of!(INDEX) };
    let count = u32::from_le_bytes(idx[12..16].try_into().unwrap()) as usize;
    for i in 0..count.min(254) {
        let e = &idx[16 + 32 * i..16 + 32 * (i + 1)];
        let end = e[..24].iter().position(|&b| b == 0).unwrap_or(24);
        if &e[..end] == name.as_bytes() {
            return Some((
                Entry {
                    lba: u32::from_le_bytes(e[24..28].try_into().unwrap()),
                    len: u32::from_le_bytes(e[28..32].try_into().unwrap()),
                },
                i,
            ));
        }
    }
    None
}

/// Per-entry u32 byte-sums (image "sums" asset): SPI mode has CRC off,
/// so blob loads are verified against these and retried on mismatch.
static mut SUMS: [u32; 256] = [0; 256];
/// Packed entries whose expansion has been checked this boot (raw-sum
/// carried in the LAY4 header; drift is a hard error, checked once).
static mut Q4_VERIFIED: [u32; 8] = [0; 8];

fn load_sums() {
    if let Some(e) = lookup("sums") {
        let n = (e.len as usize / 4).min(256);
        let mut buf = [0u8; 1024];
        if e.len as usize <= buf.len()
            && storage::read_blocks(e.lba, buf.as_mut_ptr(),
                               e.len.div_ceil(storage::BLOCK as u32)) == 0
        {
            let sums = unsafe { &mut *core::ptr::addr_of_mut!(SUMS) };
            for i in 0..n {
                sums[i] = u32::from_le_bytes(
                    buf[i * 4..i * 4 + 4].try_into().unwrap());
            }
        }
    }
}

/// Fixed-capacity name builder for parameterized asset/blob names.
struct Name {
    buf: [u8; 23],
    len: usize,
}

impl Name {
    fn of(parts: &[&str]) -> Name {
        let mut n = Name { buf: [0; 23], len: 0 };
        for p in parts {
            let b = p.as_bytes();
            n.buf[n.len..n.len + b.len()].copy_from_slice(b);
            n.len += b.len();
        }
        n
    }

    fn s(&self) -> &str {
        core::str::from_utf8(&self.buf[..self.len]).unwrap()
    }
}

const DIGITS: [&str; 4] = ["0", "1", "2", "3"];
const PARTS: [&str; 4] = ["a", "b", "c", "d"];

/// Encoder submodel names follow common.submodel_names (block 0 is legacy).
fn enc_blob(l: usize, kind: &str, sub: usize) -> Name {
    match (l, kind) {
        (0, "q") => Name::of(&["wq0"]),
        (0, "k") => Name::of(&["wk0"]),
        (0, "v") => Name::of(&["wv0"]),
        (0, "out") => Name::of(&["wout0"]),
        (0, "fc1") => Name::of(&["wfc1", PARTS[sub]]),
        (0, "fc2p") => Name::of(&["wfc2p", DIGITS[sub]]),
        (_, "fc1") => Name::of(&["w", DIGITS[l], "fc1", PARTS[sub]]),
        (_, "fc2p") => Name::of(&["w", DIGITS[l], "fc2p", DIGITS[sub]]),
        (_, k) => Name::of(&["w", DIGITS[l], k]),
    }
}

fn dec_blob(l: usize, kind: &str, sub: usize) -> Name {
    match kind {
        "fc1" => Name::of(&["d", DIGITS[l], "fc1", PARTS[sub]]),
        "fc2p" => Name::of(&["d", DIGITS[l], "fc2p", DIGITS[sub]]),
        k => Name::of(&["d", DIGITS[l], k]),
    }
}

struct Rd<'a>(&'a [u8], usize);

impl<'a> Rd<'a> {
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.0[self.1..self.1 + 4].try_into().unwrap());
        self.1 += 4;
        v
    }

    fn q(&mut self) -> Quant {
        let scale = f32::from_le_bytes(self.0[self.1..self.1 + 4].try_into().unwrap());
        let zp = i32::from_le_bytes(self.0[self.1 + 4..self.1 + 8].try_into().unwrap());
        self.1 += 8;
        Quant { scale, zp }
    }

    fn q4(&mut self) -> [Quant; 4] {
        [self.q(), self.q(), self.q(), self.q()]
    }
}

#[derive(Clone, Copy, Default)]
struct EncBlockQ {
    ln1: Quant,
    q_out: Quant,
    k_out: Quant,
    v_out: Quant,
    ctx: Quant,
    out_out: Quant,
    res1: Quant,
    ln2: Quant,
    fc2p_out: [Quant; 4],
    res2: Quant,
}

#[derive(Clone, Copy, Default)]
struct DecBlockQ {
    ln1: Quant,
    q_out: Quant,
    k_out: Quant,
    v_out: Quant,
    ctx: Quant,
    out_out: Quant,
    res1: Quant,
    xln: Quant,
    xq_out: Quant,
    xk_out: Quant,
    xv_out: Quant,
    xctx: Quant,
    xout_out: Quant,
    res2: Quant,
    ln2: Quant,
    fc2p_out: [Quant; 4],
    res3: Quant,
}

struct Plan {
    scratch: u32,
    vocab_n: usize,
    n_sot: usize,
    sot: [u32; 4],
    eot: u32,
    n_blank: usize,
    blank: [u32; 4],
    conv1_in: Quant,
    conv2_in: Quant,
    gelu2: Quant,
    enc_x: Quant,
    enc_out: Quant,
    xk_in: Quant,
    dec_x: Quant,
    enc: [EncBlockQ; BLOCKS],
    dec: [DecBlockQ; BLOCKS],
}

static mut PLAN_BUF: [u8; 2048] = [0; 2048];

impl Plan {
    fn load() -> Result<Plan, i32> {
        let e = lookup("plan").ok_or(-901)?;
        let buf = unsafe { &mut *core::ptr::addr_of_mut!(PLAN_BUF) };
        if e.len as usize > buf.len() {
            return Err(-902);
        }
        let rc = storage::read_blocks(e.lba, buf.as_mut_ptr(),
                                 e.len.div_ceil(storage::BLOCK as u32));
        if rc != 0 {
            return Err(rc);
        }
        let mut r = Rd(buf, 0);
        if r.u32() != PLAN_MAGIC || r.u32() != 1 {
            return Err(-903);
        }
        let scratch = r.u32();
        let vocab_n = r.u32() as usize;
        let n_sot = r.u32() as usize;
        let sot = [r.u32(), r.u32(), r.u32(), r.u32()];
        let eot = r.u32();
        let n_blank = r.u32() as usize;
        let blank = [r.u32(), r.u32(), r.u32(), r.u32()];
        let conv1_in = r.q();
        let conv2_in = r.q();
        let gelu2 = r.q();
        let enc_x = r.q();
        let enc_out = r.q();
        let xk_in = r.q();
        let dec_x = r.q();
        let mut enc: [EncBlockQ; BLOCKS] = Default::default();
        for b in enc.iter_mut() {
            *b = EncBlockQ {
                ln1: r.q(), q_out: r.q(), k_out: r.q(), v_out: r.q(),
                ctx: r.q(), out_out: r.q(), res1: r.q(), ln2: r.q(),
                fc2p_out: r.q4(), res2: r.q(),
            };
        }
        let mut dec: [DecBlockQ; BLOCKS] = Default::default();
        for b in dec.iter_mut() {
            *b = DecBlockQ {
                ln1: r.q(), q_out: r.q(), k_out: r.q(), v_out: r.q(),
                ctx: r.q(), out_out: r.q(), res1: r.q(), xln: r.q(),
                xq_out: r.q(), xk_out: r.q(), xv_out: r.q(), xctx: r.q(),
                xout_out: r.q(), res2: r.q(), ln2: r.q(), fc2p_out: r.q4(),
                res3: r.q(),
            };
        }
        Ok(Plan {
            scratch, vocab_n, n_sot, sot, eot, n_blank, blank, conv1_in,
            conv2_in, gelu2, enc_x, enc_out, xk_in, dec_x, enc, dec,
        })
    }
}

// --- SD helpers ------------------------------------------------------------------

struct Ctxt {
    scratch: u32,
    loaded: Entry,
    /// VAD-chosen extent: active 64-frame tiles and attention context
    /// (keys). Storage layouts stay sized for N_TILES/CTX; these only
    /// bound the loops. Floor/cap enforced in mel_pass2.
    tiles: usize,
    ctx: usize,
}

macro_rules! try_rc {
    ($e:expr, $what:expr) => {{
        let rc = $e;
        if rc != 0 {
            rprintln!("FAIL {} rc={}", $what, rc);
            return Err(rc);
        }
    }};
}

impl Ctxt {
    fn read(&self, region: u32, byte_off: usize, off: usize, len: usize) -> i32 {
        storage::read_blocks(
            self.scratch + region + (byte_off / storage::BLOCK) as u32,
            arena(off, 0).as_mut_ptr(),
            (len / storage::BLOCK) as u32,
        )
    }

    fn write(&self, region: u32, byte_off: usize, off: usize, len: usize) -> i32 {
        storage::write_blocks(
            self.scratch + region + (byte_off / storage::BLOCK) as u32,
            arena(off, 0).as_ptr(),
            (len / storage::BLOCK) as u32,
        )
    }

    /// Whole blocks of a scratch region into any buffer.
    fn read_raw(&self, region: u32, byte_off: usize, dst: &mut [u8]) -> i32 {
        storage::read_blocks(
            self.scratch + region + (byte_off / storage::BLOCK) as u32,
            dst.as_mut_ptr(),
            (dst.len() / storage::BLOCK) as u32,
        )
    }

    fn write_raw(&self, region: u32, byte_off: usize, src: &[u8]) -> i32 {
        storage::write_blocks(
            self.scratch + region + (byte_off / storage::BLOCK) as u32,
            src.as_ptr(),
            (src.len() / storage::BLOCK) as u32,
        )
    }

    /// Load a whole named asset to an arena offset.
    fn asset(&self, name: &str, off: usize) -> Result<Entry, i32> {
        let e = lookup(name).ok_or(-901)?;
        let rc = storage::read_blocks(e.lba, arena(off, 0).as_mut_ptr(),
                                 e.len.div_ceil(storage::BLOCK as u32));
        if rc != 0 {
            return Err(rc);
        }
        Ok(e)
    }

    /// Blob into the slot (cached, sum-verified) + one NPU inference.
    fn npu(&mut self, blob: &str, input: usize, output: usize) -> i32 {
        let (e, idx) = match lookup_idx(blob) {
            Some(x) => x,
            None => {
                rprintln!("no blob {}", blob);
                return -904;
            }
        };
        if self.loaded.lba != e.lba {
            let expect = unsafe { (*core::ptr::addr_of!(SUMS))[idx] };
            let mut ok = false;
            for attempt in 0..3 {
                let rc = storage::read_blocks(e.lba, slot::SLOT_BASE as *mut u8,
                                         e.len.div_ceil(storage::BLOCK as u32));
                if rc != 0 {
                    return rc;
                }
                let bytes = unsafe {
                    core::slice::from_raw_parts(slot::SLOT_BASE as *const u8,
                                                e.len as usize)
                };
                let t0 = cycles();
                let sum = dsp::byte_sum(bytes);
                prof_add(P_SUM, t0);
                if expect == 0 || sum == expect {
                    ok = true;
                    break;
                }
                rprintln!("blob {} sum mismatch (got {:#x} want {:#x}, try {})",
                          blob, sum, expect, attempt + 1);
            }
            if !ok {
                return -905;
            }
            // "LAY4" packed entry: expand in place to the raw blob.
            let magic = unsafe { core::ptr::read(slot::SLOT_BASE as *const u32) };
            if magic == crate::q4::MAGIC {
                if let Err(rc) = unpack_slot(e, idx) {
                    return rc;
                }
            }
            self.loaded = e;
        }
        let _wd = crate::WdogGuard::arm();
        unsafe { slot::run(arena_addr(input), arena_addr(output), blob) }
    }
}

/// Largest weight region a packable blob may carry, in 64-weight groups
/// (bounds the amax staging copy; slot-sized blobs fit).
const AMAX_MAX: usize = 3328;

/// Expand a "LAY4" packed blob, just read (and sum-verified) at
/// SLOT_BASE, into the raw blob it encodes -- in place in the slot.
///
/// The packed bytes are first moved to the slot tail; the head is copied
/// back verbatim and the weights expand forward from w_off. The writer
/// advances 2 bytes per nibble byte consumed, so it never catches the
/// reader as long as the nibble stream starts >= n/2 bytes past the
/// weight region start -- checked below, guaranteed by the image builder
/// with ~56 KB of margin for the current blobs. The amax table is staged
/// out first (in the idle interlayer, NOT the stack -- see attn_scratch)
/// because the writer DOES cross it.
fn unpack_slot(e: Entry, idx: usize) -> Result<(), i32> {
    let blocks = e.len.div_ceil(storage::BLOCK as u32) as usize;
    let tail = slot::SLOT_BYTES - blocks * storage::BLOCK;
    unsafe {
        core::ptr::copy(slot::SLOT_BASE as *const u8,
                        (slot::SLOT_BASE + tail) as *mut u8, e.len as usize);
        let p = (slot::SLOT_BASE + tail) as *const u8;
        let word = |i: usize| -> usize {
            u32::from_le_bytes(core::slice::from_raw_parts(p.add(i * 4), 4)
                .try_into().unwrap()) as usize
        };
        let (raw_len, w_off, n) = (word(1), word(2), word(3));
        let raw_sum = word(4) as u32;
        let n_groups = n / crate::q4::G;
        let nib_off = tail + crate::q4::HDR + w_off + n_groups;
        if w_off + n != raw_len
            || raw_len > slot::SLOT_BYTES
            || n % (2 * crate::q4::G) != 0
            || n_groups > AMAX_MAX
            || crate::q4::HDR + w_off + n_groups + n / 2 != e.len as usize
            || raw_len > nib_off + n / 2
            || w_off > tail
        {
            rtt_target::rprintln!("blob unpack: bad LAY4 header");
            return Err(-906);
        }
        let amax = interlayer(n_groups);
        amax.copy_from_slice(
            core::slice::from_raw_parts(p.add(crate::q4::HDR + w_off), n_groups));
        core::ptr::copy_nonoverlapping(p.add(crate::q4::HDR),
                                       slot::SLOT_BASE as *mut u8, w_off);
        crate::q4::unpack_raw(amax,
                              (slot::SLOT_BASE + nib_off) as *const u8,
                              (slot::SLOT_BASE + w_off) as *mut i8, n);
        // once per boot per entry: catch packer/unpacker drift exactly
        let seen = (*core::ptr::addr_of!(Q4_VERIFIED))[idx >> 5]
            & (1 << (idx & 31)) != 0;
        if !seen {
            let raw = core::slice::from_raw_parts(slot::SLOT_BASE as *const u8,
                                                  raw_len);
            let sum = dsp::byte_sum(raw);
            if sum != raw_sum {
                rtt_target::rprintln!(
                    "blob unpack sum mismatch (got {:#x} want {:#x})", sum, raw_sum);
                return Err(-907);
            }
            (*core::ptr::addr_of_mut!(Q4_VERIFIED))[idx >> 5] |= 1 << (idx & 31);
        }
    }
    Ok(())
}

/// attn_head working buffers (6.4 KB), parked in the interlayer buffer:
/// CPU attention runs between NPU inferences, when the interlayer is
/// idle -- the same transient-use rule as the lm_head bounce. As stack
/// locals this frame reached below _stack_end at decode depth and
/// overwrote the driver state at the top of .bss (gl_axon_instances;
/// hardware-observed as a wild register write mid-infer, BFAR garbage).
fn attn_scratch() -> &'static mut kernels::AttnScratch {
    let b = interlayer(core::mem::size_of::<kernels::AttnScratch>());
    unsafe { &mut *(b.as_mut_ptr() as *mut kernels::AttnScratch) }
}

fn lut_apply(lut_off: usize, buf_off: usize, len: usize) {
    let lut: [i8; 256] = core::array::from_fn(|i| as_i8(lut_off, 256)[i]);
    for b in as_i8_mut(buf_off, len) {
        *b = lut[(*b as i32 + 128) as usize];
    }
}

// --- entry ------------------------------------------------------------------------

pub fn run() -> ! {
    // Optional transcript display; every display call no-ops when absent.
    if display::init() {
        rprintln!("standalone: OLED found");
        display::print("Whisper standalone\n");
    }
    rprintln!("standalone: storage init (USB stick, then SD)");
    let rc = storage::init();
    if rc != 0 {
        rprintln!("standalone: no storage ({}), staying in mailbox mode", rc);
        display::print("no storage\n");
        sd::diag(2);
        sd::release_pins();
        rprintln!("sd pins released (high-Z): external testers may drive the bus");
        crate::mailbox_loop();
    }
    rprintln!("standalone: model source: {}", storage::name());
    unsafe {
        let rc = storage::read_blocks(0, core::ptr::addr_of_mut!(INDEX) as *mut u8, 16);
        let idx = &*core::ptr::addr_of!(INDEX);
        if rc != 0 || &idx[..8] != IMG_MAGIC {
            rprintln!("standalone: no image (rc={}), staying in mailbox mode", rc);
            display::print("no card image\n");
            crate::mailbox_loop();
        }
    }
    let plan = match Plan::load() {
        Ok(p) => p,
        Err(rc) => {
            rprintln!("standalone: bad plan ({}), staying in mailbox mode", rc);
            display::print("bad card plan\n");
            crate::mailbox_loop();
        }
    };
    // Card/firmware match check: the image records the interlayer/psum
    // addresses of the ELF its blobs were linked against. A mismatch
    // means the card is STALE (blobs would DMA into a previous build's
    // buffer addresses -- garbage results or a wedged engine).
    if let Some(f) = lookup("fwid") {
        let mut b = [0u8; 8];
        if sd_read_bytes(f, 0, &mut b) == 0 {
            let il = u32::from_le_bytes(b[0..4].try_into().unwrap());
            let ps = u32::from_le_bytes(b[4..8].try_into().unwrap());
            let my_il =
                core::ptr::addr_of!(crate::nrf_axon_interlayer_buffer) as u32;
            let my_ps = core::ptr::addr_of!(crate::nrf_axon_psum_buffer) as u32;
            if il != my_il || ps != my_ps {
                rprintln!(
                    "standalone: CARD IMAGE IS STALE (card fwid {:#010x}/{:#010x},                      firmware {:#010x}/{:#010x}) -- re-dd model/out/sd.img",
                    il, ps, my_il, my_ps
                );
                display::print("STALE CARD: re-dd
");
                crate::mailbox_loop();
            }
        }
    } else {
        rprintln!("standalone: warning: image has no fwid (predates the check)");
    }
    load_sums();
    crate::platform::hold_axon();
    rprintln!("standalone: ready ({} kept vocabulary entries)", plan.vocab_n);
    loop {
        let mut ctx = Ctxt {
            scratch: plan.scratch,
            loaded: Entry::default(),
            tiles: N_TILES,
            ctx: CTX,
        };
        if utterance(&plan, &mut ctx).is_err() {
            rprintln!("(utterance aborted; retrying in a moment)");
            display::print("(retry)\n");
            cortex_m::asm::delay(128_000_000);
        }
    }
}

/// Streaming mode failed once (mel fell behind the mic): stay sequential.
static mut STREAM_MEL_OK: bool = true;

/// True when the storage backend is serving pre-loaded audio in S_PCM
/// (the mock-usb development rig). Checked every utterance so the same
/// firmware records live or replays a clip with no rebuild.
fn injected(c: &Ctxt) -> Option<u32> {
    if c.read(S_INJECT, 0, 0, storage::BLOCK) != 0 {
        return None;
    }
    let b = arena(0, 16);
    if &b[..8] != b"WMOCKAU1" {
        return None;
    }
    Some(u32::from_le_bytes([b[8], b[9], b[10], b[11]]))
}

/// Microphone bring-up aid: record windows and report the level of each,
/// stopping at the first one loud enough to be speech so that window's
/// audio is still sitting in S_PCM for the host to pull off the work
/// image. Skips the encoder and decoder entirely, so a window costs 12 s
/// rather than three minutes.
#[cfg(feature = "mic-check")]
#[inline(never)]
fn mic_check(c: &Ctxt) -> Result<(), i32> {
    const WINDOWS: u32 = 30;
    // Ambient noise measures a peak near 60, so this is comfortably above
    // the floor and still low enough to trip on quiet speech if the gain
    // really is as low as it looks.
    const TRIGGER: i32 = 200;
    rprintln!("=== mic check: {} windows of 12 s, speak into the mic ===", WINDOWS);
    for w in 0..WINDOWS {
        rprintln!("mic: window {}/{} recording...", w + 1, WINDOWS);
        display::clear();
        display::print("mic check\n");
        mic_reset();
        let ov = record_mel(c)?;
        let (peak, rms) = mic_level();
        rprintln!(
            "mic: window {} peak {} ({:.1} dBFS) rms {:.1}, {} overruns",
            w + 1,
            peak,
            20.0 * libm::log10f(if peak > 0 { peak as f32 } else { 1.0 } / 32768.0),
            rms,
            ov
        );
        if peak >= TRIGGER {
            rprintln!("mic: window {} kept in S_PCM (peak >= {})", w + 1, TRIGGER);
            return Ok(());
        }
    }
    rprintln!("mic: nothing reached peak {}; last window kept in S_PCM", TRIGGER);
    Ok(())
}

fn utterance(plan: &Plan, c: &mut Ctxt) -> Result<(), i32> {
    #[cfg(feature = "mic-check")]
    {
        let _ = plan;
        let r = mic_check(c);
        rprintln!("mic check done ({:?}); idling so the audio survives", r);
        crate::mailbox_loop();
    }

    // Decoder iteration without paying for a full encoder pass: the cross
    // K/V scratch persists on the card (dd stops before the scratch
    // blocks), so a build with this feature jumps straight to decode
    // against the previous complete run's encoder output.
    if cfg!(feature = "decode-only") {
        rprintln!("");
        rprintln!("=== decode-only (cross K/V scratch of the last full run) ===");
        let r = decode(plan, c);
        rprintln!("");
        rprintln!("decode-only pass finished ({:?}); mailbox mode", r);
        crate::mailbox_loop();
    }
    rprintln!("");
    if let Some(n) = injected(c) {
        rprintln!("=== injected audio ({} samples, no mic) ===", n);
        display::clear();
        display::print("== injected clip\n");
        mel_tables(c)?;
        mel_pass1(c)?;
    } else {
    rprintln!("=== speak now (up to 12 s) ===");
    display::clear();
    display::print("== speak now (12 s max)\n");
    // Mel pass 1 overlaps the recording (chunk-sized PDM buffers). The
    // old direct DFT overran the mic deterministically (17/boot, each
    // 64-frame chunk cost more than its 640 ms period); the mixed-radix
    // FFT computes a chunk in a small fraction of that, leaving the
    // 20 KB SD spill as the only variable per-chunk cost.
    const TRY_STREAM_MEL: bool = true;
    let stream_ok =
        unsafe { core::ptr::read_volatile(core::ptr::addr_of!(STREAM_MEL_OK)) };
    if TRY_STREAM_MEL && stream_ok {
        // mel pass 1 overlaps the recording (chunk-sized PDM buffers); an
        // overrun means the M33 could not keep up -- lost audio, so abort
        // this utterance and fall back to the sequential path for good.
        let ov = record_mel(c)?;
        if ov > 0 {
            unsafe { STREAM_MEL_OK = false };
            rprintln!(
                "warning: {} overruns streaming mel; falling back to \
                 sequential mel from now on",
                ov
            );
            return Err(-908);
        }
    } else {
        record(c)?;
        rprintln!("mel...");
        mel_tables(c)?;
        mel_pass1(c)?;
    }
    }
    let (tiles, actx) = mel_pass2(plan, c)?;
    sd_stats("mel");
    fp(c, S_MEL, "mel8");
    c.tiles = tiles;
    c.ctx = actx;
    display::print("encoding...\n");
    rprintln!("encoder...");
    encoder(plan, c)?;
    sd_stats("encoder");
    fp(c, S_EO, "enc");
    rprintln!("cross K/V...");
    cross_kv(plan, c)?;
    sd_stats("cross");
    fp(c, S_XKV, "xkv");
    rprintln!("decoding...");
    display::print("decoding:\n");
    let r = decode(plan, c);
    sd_stats("decode");
    r
}

/// Debug fingerprint: byte-sum of a scratch region's first block, to
/// localize where the pipeline stops responding to the audio (two runs
/// with different audio must differ at every live stage).
fn fp(c: &Ctxt, region: u32, label: &str) {
    let rc = c.read(region, 0, 0, 512);
    if rc == 0 {
        rprintln!("fp[{}]: {:#07x}", label, dsp::byte_sum(arena(0, 512)));
    }
}

/// Per-phase SD throughput line (drains the counters). Rates well below
/// the session's first-utterance numbers implicate the card, not code.
fn sd_stats(phase: &str) {
    let (rb, rc, wb, wc) = storage::stats_take();
    rprintln!(
        "sd[{}]: rd {} KB / {} ms, wr {} KB / {} ms",
        phase, rb / 1024, rc / 128_000, wb / 1024, wc / 128_000
    );
    let p = unsafe { core::mem::replace(&mut *core::ptr::addr_of_mut!(PROF), [0; 4]) };
    rprintln!(
        "cpu[{}]: attn {} ms, lm {} ms, unpack {} ms, sums {} ms",
        phase, p[P_ATTN] / 128_000, p[P_LM] / 128_000, p[P_UNPACK] / 128_000,
        p[P_SUM] / 128_000
    );
}

// --- record + mel pass 1 --------------------------------------------------------
//
// Streaming layout: the PDM ping-pong buffers are one mel chunk each
// (10240 samples = 0.64 s), so while the DMA fills one buffer the CPU has a
// whole chunk period to window the previous one, run mel_frames, and spill
// the f32 chunk to S_MELF. Chunk c's STFT window needs samples
// [c*10240 - 1280, c*10240 + 10496) (left reflect/alignment halo + right
// STFT halo), i.e. the tail of buffer c-1, all of buffer c, and the first
// 256 samples of buffer c+1 -- assembled in a 12032-sample sliding window.
// The windows and sample counts are byte-identical to the sequential pass 1
// (mel_pass1), so the mel output is bit-exact either way.

const CHUNK: usize = 10240; // samples per PDM buffer / mel chunk
const N_CHUNKS: usize = 19; // 18 x 64 frames + 1 x 48 (1200 real frames)
const WIN: usize = 12032; // 1280 halo + CHUNK + 256 halo

// Arena offsets shared by the record/mel phases. Asset loads round up to
// whole SD blocks, so the table slots are block-sized (1600 B tables in
// 2048 B slots) and nothing overlaps a neighbor's rounding tail.
const R_HANN: usize = 0; // 1600 in 2048
const R_COS: usize = 2048; // 1600 in 2048
const R_MAX: usize = 4096; // 4
const R_WIN: usize = 4104; // 24064 (streaming window / fallback PCM staging)
const R_MELF: usize = 28168; // 20480 (f32 mel chunk)
const R_PDM0: usize = 48648; // 20480
const R_PDM1: usize = 69128; // 20480 (ends 89608 < arena top)
const R_TILE: usize = 48648; // pass 2 int8 tile staging (PDM idle by then)
const R_MEANS: usize = 89608; // 4800: per-frame mean log-mel for the online VAD

/// Recording ends early once speech has been heard and SIL_CHUNKS whole
/// chunks (0.64 s each) after it stayed silent, never before MIN_CHUNKS.
/// The two bounds keep every frame the encoder can touch inside the
/// recording -- the VAD context floor (192 frames = 6 chunks), the 0.64 s
/// margin, the round-up to a 128-frame tile and conv1's one-frame halo --
/// so the encoder input is identical to the full 12 s capture's whenever
/// the VAD picks the same endpoint. A pause longer than SIL_CHUNKS ends
/// the utterance: that is the trade.
const SIL_CHUNKS: usize = 3;
const MIN_CHUNKS: usize = 7;
/// Chunks the last capture produced (N_CHUNKS = the full 12 s).
static mut REC_CHUNKS: usize = N_CHUNKS;

// Each pipeline phase keeps its scratch on the stack, and the stack has
// ~16.5 KB between the top of RAM and .bss. Inlined into one another the
// phases' frames SUM -- decode's ~8 KB of buffers then stay live through
// mel, whose mel_frames needs 6 KB of its own, and the run dies in an
// MSPLIM stack-overflow fault whose backtrace names mel, not decode.
// Keeping the phases out of line makes each frame live only while that
// phase runs. (Found the hard way: the sequential mel path faulted here
// the first time anything exercised it.)
#[inline(never)]
fn mel_tables(c: &Ctxt) -> Result<(), i32> {
    // filterbank borrows the interlayer buffer; small tables in the arena
    let filt = lookup("melfilt").ok_or(-901)?;
    try_rc!(storage::read_blocks(filt.lba, interlayer(0).as_mut_ptr(), 126), "filt");
    c.asset("hann", R_HANN)?;
    c.asset("melcos", R_COS)?;
    arena(R_MAX, 4).copy_from_slice(&(-1e30f32).to_le_bytes());
    Ok(())
}

/// Per-frame mean log-mel of the chunk just computed (R_MELF, planar
/// [80][n_frames]) into R_MEANS -- the same sums mel_pass2 forms from the
/// spilled data, so the online and offline VAD see identical numbers.
fn chunk_means(ci: usize, n_frames: usize) {
    let m = as_f32(R_MELF, 80 * n_frames);
    let means = unsafe {
        core::slice::from_raw_parts_mut(arena_addr(R_MEANS) as *mut f32, 1200)
    };
    for f in 0..n_frames {
        let mut sum = 0f32;
        for r in 0..80 {
            sum += m[r * n_frames + f];
        }
        means[ci * T + f] = sum / 80.0;
    }
}

/// Per-frame speech test over mean log-mel energies: louder than the
/// quietest frame + 10 dB and within 20 dB of the loudest. Returns
/// (floor, peak, thresh, last speech frame if any).
fn vad_scan(means: &[f32]) -> (f32, f32, f32, Option<usize>) {
    let mut floor = f32::MAX;
    let mut peak = f32::MIN;
    for &v in means {
        if v < floor {
            floor = v;
        }
        if v > peak {
            peak = v;
        }
    }
    let thresh = (floor + VAD_THRESH_LOG10).max(peak - VAD_PEAK_DROP_LOG10);
    let mut last = None;
    for (f, &v) in means.iter().enumerate() {
        if v > thresh {
            last = Some(f);
        }
    }
    (floor, peak, thresh, last)
}

/// Whole silent chunks after the last speech chunk, over `k` chunks.
fn silent_tail(k: usize) -> usize {
    let means = as_f32(R_MEANS, k * T);
    match vad_scan(means).3 {
        Some(last) => (k - 1) - last / T,
        None => 0,
    }
}

fn mel_chunk(ci: usize, n_samples: usize, n_frames: usize) {
    let p = mel::MelParams {
        pcm: arena_addr(R_WIN),
        n_samples: n_samples as u32,
        frame0: if ci == 0 { 0 } else { 8 },
        n_frames: n_frames as u32,
        hann: arena_addr(R_HANN),
        cos_tab: arena_addr(R_COS),
        filters: interlayer(0).as_ptr() as u32,
        out: arena_addr(R_MELF),
        max_acc: arena_addr(R_MAX),
    };
    unsafe { mel::mel_frames(&p) };
}

/// Record 12 s while computing mel pass 1 in the buffer gaps. Returns the
/// PDM overrun count (any overrun lost audio: caller must discard).
#[inline(never)]
fn record_mel(c: &Ctxt) -> Result<u32, i32> {
    mel_tables(c)?;

    let b0 = as_i16_mut(R_PDM0, CHUNK);
    let b1 = as_i16_mut(R_PDM1, CHUNK);
    let mut stream =
        unsafe { pdm::Pdm::init(crate::MIC_CLK, crate::MIC_DIN).start(b0, b1) };
    stream.next_buffer(); // warmup chunk (startup overrun + mic DC settle)
    stream.overruns = 0;

    let mut have = 0usize; // valid samples in the sliding window
    let mut n_chunks = N_CHUNKS;
    for k in 0..N_CHUNKS {
        let hop = stream.next_buffer(); // buffer k; DMA now fills the other
        // Development rig only: keep a copy of what the microphone
        // actually heard. The streaming path never otherwise persists its
        // PCM (only the sequential fallback writes S_PCM), which leaves a
        // live-mic run with nothing to inspect when its mel looks wrong.
        #[cfg(feature = "mock-usb")]
        spill_pcm(c, k, hop);
        let win = as_i16_mut(R_WIN, WIN);
        if k == 0 {
            win[..CHUNK].copy_from_slice(hop);
            have = CHUNK;
            continue;
        }
        // chunk k-1: append buffer k's first 256 samples (right halo)
        win[have..have + 256].copy_from_slice(&hop[..256]);
        mel_chunk(k - 1, have + 256, T);
        chunk_means(k - 1, T);
        try_rc!(c.write(S_MELF, (k - 1) * 20480, R_MELF, 20480), "mel spill");
        if k >= MIN_CHUNKS && silent_tail(k) >= SIL_CHUNKS {
            n_chunks = k; // chunks 0..k-1 are complete and spilled
            break;
        }
        // slide: 1280-sample left halo, then all of buffer k
        win.copy_within(have - 1280..have, 0);
        win[1280..1280 + CHUNK].copy_from_slice(hop);
        have = 1280 + CHUNK;
    }
    let ov = stream.overruns;
    stream.stop();
    if n_chunks == N_CHUNKS {
        // final chunk (48 frames) needs no further input from the mic: the
        // window already covers [18*CHUNK - 1280, N_SAMPLES)
        mel_chunk(N_CHUNKS - 1, N_SAMPLES - ((N_CHUNKS - 1) * CHUNK - 1280), 48);
        chunk_means(N_CHUNKS - 1, 48);
        try_rc!(c.write(S_MELF, (N_CHUNKS - 1) * 20480, R_MELF,
                        (80 * 48 * 4usize).div_ceil(storage::BLOCK) * storage::BLOCK),
                "mel spill");
    } else {
        rprintln!("rec: silence, stopped after {} chunks ({} ms)",
                  n_chunks, n_chunks * CHUNK / 16);
    }
    unsafe { REC_CHUNKS = n_chunks };
    Ok(ov)
}

/// Level of the window being recorded, accumulated as its chunks arrive.
#[cfg(feature = "mock-usb")]
static mut MIC_PEAK: i32 = 0;
#[cfg(feature = "mock-usb")]
static mut MIC_SUMSQ: u64 = 0;
#[cfg(feature = "mock-usb")]
static mut MIC_N: u32 = 0;

#[cfg(feature = "mock-usb")]
fn mic_reset() {
    unsafe {
        MIC_PEAK = 0;
        MIC_SUMSQ = 0;
        MIC_N = 0;
    }
}

/// (peak, rms) of the window just recorded.
#[cfg(feature = "mock-usb")]
fn mic_level() -> (i32, f32) {
    unsafe {
        let n = MIC_N.max(1) as f32;
        (MIC_PEAK, libm::sqrtf(MIC_SUMSQ as f32 / n))
    }
}

/// Copy one recorded chunk into S_PCM, clipped to the region's 750 blocks
/// (19 chunks of 10240 samples overrun 192000 by 2560, and S_INJECT sits
/// immediately after).
#[cfg(feature = "mock-usb")]
fn spill_pcm(c: &Ctxt, k: usize, hop: &[i16]) {
    unsafe {
        for &x in hop {
            let a = (x as i32).abs();
            if a > MIC_PEAK {
                MIC_PEAK = a;
            }
            MIC_SUMSQ += (x as i32 * x as i32) as u64;
            MIC_N += 1;
        }
    }
    let off = k * CHUNK * 2;
    if off >= N_SAMPLES * 2 {
        return;
    }
    let bytes = (N_SAMPLES * 2 - off).min(CHUNK * 2);
    let src = hop.as_ptr() as u32;
    let arena_off = (src - arena_addr(0)) as usize;
    let rc = c.write(S_PCM, off, arena_off, bytes);
    if rc != 0 {
        rprintln!("pcm spill rc={}", rc);
    }
}

/// Sequential fallback: plain recording to SD (used when streaming mel
/// once fell behind; pass 1 then reads the PCM back from the card).
#[inline(never)]
fn record(c: &Ctxt) -> Result<(), i32> {
    const HOP: usize = 320;
    let ring = arena(0, 16 * HOP * 2);
    let b0 = unsafe { &mut (*core::ptr::addr_of_mut!(crate::PDM_BUF0)).0 };
    let b1 = unsafe { &mut (*core::ptr::addr_of_mut!(crate::PDM_BUF1)).0 };
    let mut stream =
        unsafe { pdm::Pdm::init(crate::MIC_CLK, crate::MIC_DIN).start(b0, b1) };
    stream.next_buffer(); // warmup hop
    stream.overruns = 0;
    let total = N_SAMPLES * 2;
    let mut filled = 0usize;
    let mut written = 0usize;
    while written < total {
        let hop = stream.next_buffer();
        ring[filled * HOP * 2..(filled + 1) * HOP * 2].copy_from_slice(unsafe {
            core::slice::from_raw_parts(hop.as_ptr() as *const u8, HOP * 2)
        });
        filled += 1;
        if filled == 8 {
            try_rc!(c.write(S_PCM, written, 0, 8 * HOP * 2), "pcm");
            written += 8 * HOP * 2;
            filled = 0;
        }
    }
    let ov = stream.overruns;
    stream.stop();
    if ov > 0 {
        rprintln!("warning: {} recording overruns", ov);
    }
    Ok(())
}

// --- mel (sequential pass 1 + shared pass 2) ---------------------------------------

/// Fallback pass 1: 18 chunks of 64 frames + 1 of 48 (1200 real frames),
/// PCM read back from the card. The chunk's PCM window starts 8 frames
/// (1280 samples) early so that (a) the reflect halo has real samples and
/// (b) the SD byte offset stays block-aligned (1280 samples = 2560 B, lcm
/// of 160 and 256). Windows are identical to record_mel's streaming ones.
#[inline(never)]
fn mel_pass1(c: &Ctxt) -> Result<(), i32> {
    unsafe { REC_CHUNKS = N_CHUNKS };
    for ci in 0..N_CHUNKS {
        let f0 = ci * T;
        let n_frames = if ci == 18 { 48 } else { T };
        let s0 = (f0 * 160).saturating_sub(1280);
        let span = ((f0 + n_frames) * 160 + 256).min(N_SAMPLES) - s0;
        let bytes = (span * 2).div_ceil(storage::BLOCK) * storage::BLOCK;
        try_rc!(c.read(S_PCM, s0 * 2, R_WIN, bytes), "pcm rd");
        let p = mel::MelParams {
            pcm: arena_addr(R_WIN),
            n_samples: (bytes / 2).min(N_SAMPLES - s0) as u32,
            frame0: ((f0 * 160 - s0) / 160) as u32,
            n_frames: n_frames as u32,
            hann: arena_addr(R_HANN),
            cos_tab: arena_addr(R_COS),
            filters: interlayer(0).as_ptr() as u32,
            out: arena_addr(R_MELF),
            max_acc: arena_addr(R_MAX),
        };
        unsafe { mel::mel_frames(&p) };
        try_rc!(c.write(S_MELF, ci * 20480, R_MELF,
                        (80 * n_frames * 4).div_ceil(512) * 512),
                "mel spill");
    }
    Ok(())
}

/// VAD floor/cap in encoder frames (mel rate is 2x): never below 192
/// frames (3.84 s) of context, never above the full 600.
const VAD_FLOOR_CTX: usize = 192;
const VAD_MARGIN_CTX: usize = 32; // 0.64 s past the last speech frame
const VAD_THRESH_LOG10: f32 = 1.0; // 10 dB over the noise floor
// A frame must also be within 20 dB of the loudest frame. The floor-only
// criterion never trimmed on hardware: trailing room noise sat more than
// 10 dB above the quietest frame, pinning the endpoint at frame 1199.
const VAD_PEAK_DROP_LOG10: f32 = 2.0;

/// Peak log10-mel of the calibration clip (out/ref.npz mel_chunk max
/// 1.46126 un-normalized: 1.46126 * 4 - 4). Recorded audio is levelled to
/// this so the encoder sees the distribution it was calibrated on.
const MEL_TARGET_MAX: f32 = 1.845;
/// Ceiling on the correction, in log10 power units (4.0 = 40 dB of audio
/// gain). Past this an utterance is silence, and lifting it just
/// amplifies the noise floor to speech level.
const MEL_MAX_LIFT: f32 = 4.0;
/// Floor, so a shouted utterance is brought down as well as up.
const MEL_MIN_LIFT: f32 = -2.0;
/// Set false to get whisper's stock absolute normalization back.
const MEL_AUTOLEVEL: bool = true;

/// How far this utterance's log-mel has to move to sit where the
/// calibration clip did.
///
/// whisper's normalization clamps relative to the utterance peak but then
/// applies an absolute +4.0 offset, so it does NOT normalize level: audio
/// 31 dB below the reference (which is what the PDM mic delivers at
/// ordinary speaking volume) lands pinned against the int8 floor and
/// transcribes as nonsense. Correcting in the log-mel domain is exactly
/// equivalent to having applied the matching gain to the samples, but it
/// costs nothing -- pass 1 already accumulated the peak -- and it cannot
/// clip. Proven on hardware: the same mic recording scaled by 36x on the
/// host transcribed correctly where the original did not.
fn mel_lift() -> f32 {
    if !MEL_AUTOLEVEL {
        return 0.0;
    }
    let observed = f32::from_le_bytes(arena(R_MAX, 4).try_into().unwrap());
    let lift = MEL_TARGET_MAX - observed;
    if lift > MEL_MAX_LIFT {
        MEL_MAX_LIFT
    } else if lift < MEL_MIN_LIFT {
        MEL_MIN_LIFT
    } else {
        lift
    }
}

/// Pass 2: normalize into int8 mel tiles; pad frames = quantized 0.0.
/// Needs the global max in R_MAX from either pass 1. Also scans the f32
/// mel energies for speech (mean log-mel per frame vs the quietest
/// frame) and returns the VAD extent (tiles, ctx) for the encoder.
#[inline(never)]
fn mel_pass2(plan: &Plan, c: &Ctxt) -> Result<(usize, usize), i32> {
    let pad = quant8(0.0, plan.conv1_in);
    let lift = mel_lift();
    rprintln!(
        "mel: peak {:.3}, level correction {:+.3} log10 ({:+.1} dB of audio gain)",
        f32::from_le_bytes(arena(R_MAX, 4).try_into().unwrap()),
        lift,
        10.0 * lift
    );
    // per-mel-frame mean energies staged in the (idle) R_WIN region
    let means = unsafe {
        core::slice::from_raw_parts_mut(arena_addr(R_WIN) as *mut f32, 1200)
    };
    let rec = unsafe { REC_CHUNKS };
    let n_frames = if rec >= N_CHUNKS { 1200 } else { rec * T };
    for mt in 0..MEL_TILES {
        let dst = as_i8_mut(R_TILE, 80 * T);
        if mt < 18 && mt < rec {
            try_rc!(c.read(S_MELF, mt * 20480, R_MELF, 20480), "melf rd");
            let m = as_f32(R_MELF, 80 * T);
            for f in 0..T {
                let mut sum = 0f32;
                for r in 0..80 {
                    sum += m[r * T + f];
                }
                means[mt * T + f] = sum / 80.0;
            }
            let p = mel::MelNormParams {
                mel: arena_addr(R_MELF),
                n: (80 * T) as u32,
                max_acc: arena_addr(R_MAX),
                out: arena_addr(R_TILE),
                q: plan.conv1_in,
                lift,
            };
            unsafe { mel::mel_normalize(&p) };
        } else if mt == 18 && rec >= N_CHUNKS {
            // 48 real frames stored planar [80,48]; expand to [80,64]
            try_rc!(c.read(S_MELF, mt * 20480, R_MELF, 15360), "melf rd");
            let m = as_f32(R_MELF, 80 * 48);
            for f in 0..48 {
                let mut sum = 0f32;
                for r in 0..80 {
                    sum += m[r * 48 + f];
                }
                means[mt * T + f] = sum / 80.0;
            }
            let p = mel::MelNormParams {
                mel: arena_addr(R_MELF),
                n: (80 * 48) as u32,
                max_acc: arena_addr(R_MAX),
                out: arena_addr(R_TILE + 80 * T), // staging past the tile
                q: plan.conv1_in,
                lift,
            };
            unsafe { mel::mel_normalize(&p) };
            let st = as_i8(R_TILE + 80 * T, 80 * 48);
            dst.fill(pad);
            for r in 0..80 {
                dst[r * T..r * T + 48].copy_from_slice(&st[r * 48..(r + 1) * 48]);
            }
        } else {
            dst.fill(pad);
        }
        try_rc!(c.write(S_MEL, mt * 80 * T, R_TILE, 80 * T), "mel8 wr");
    }
    // Endpoint: last mel frame that is both louder than (quietest frame
    // + 10 dB) and within 20 dB of the loudest frame, plus margin,
    // floored and capped, rounded up to whole tiles.
    let (floor, peak, thresh, last) = vad_scan(&means[..n_frames]);
    let last = last.unwrap_or(0);
    let ctx = (last / 2 + VAD_MARGIN_CTX).clamp(VAD_FLOOR_CTX, CTX);
    let tiles = ctx.div_ceil(T);
    rprintln!(
        "vad: floor {:.2} peak {:.2} thresh {:.2}, speech to mel frame {} of {} -> ctx {} ({} tiles of {})",
        floor, peak, thresh, last, n_frames, ctx, tiles, N_TILES
    );
    Ok((tiles, ctx))
}

fn quant8(x: f32, q: Quant) -> i8 {
    ((libm::roundf(x / q.scale) as i32) + q.zp).clamp(-128, 127) as i8
}

// --- encoder ------------------------------------------------------------------------

/// Assemble a halo-padded planar [rows, t_in] input at A_IN from a
/// tile-major region; out-of-range columns get `pad`. Tiles stage through
/// A_OUT (free before an NPU run, and large enough for [384, 64]).
fn assemble_halo(c: &Ctxt, region: u32, rows: usize, n_tiles: usize,
                 col0: i32, t_in: usize, pad: i8) -> Result<(), i32> {
    as_i8_mut(A_IN, rows * t_in).fill(pad);
    let t0 = col0.div_euclid(T as i32);
    let t1 = (col0 + t_in as i32 - 1).div_euclid(T as i32);
    for ti in t0..=t1 {
        if ti < 0 || ti >= n_tiles as i32 {
            continue;
        }
        try_rc!(c.read(region, ti as usize * rows * T, A_OUT, rows * T),
                "halo tile");
        let src = as_i8(A_OUT, rows * T);
        let dst = as_i8_mut(A_IN, rows * t_in);
        let tile_c0 = ti * T as i32;
        let lo = col0.max(tile_c0);
        let hi = (col0 + t_in as i32).min(tile_c0 + T as i32);
        for r in 0..rows {
            for col in lo..hi {
                dst[r * t_in + (col - col0) as usize] =
                    src[r * T + (col - tile_c0) as usize];
            }
        }
    }
    Ok(())
}

fn ln_region(c: &mut Ctxt, gb: &str, sq: Quant, dq: Quant) -> Result<(), i32> {
    c.asset(gb, A_GB)?;
    for i in 0..c.tiles {
        try_rc!(c.read(S_X, i * TILE16, A_IN, TILE16), "ln in");
        kernels::ln_planar_i16_to_i8(
            as_i16(A_IN, C * T), sq,
            as_f32(A_GB, C), as_f32(A_GB + 4 * C, C),
            as_i8_mut(A_OUT, C * T), dq, C, T,
        );
        try_rc!(c.write(S_LN, i * TILE8, A_OUT, TILE8), "ln out");
    }
    Ok(())
}

fn res_add(c: &Ctxt, region8: u32, qa: Quant, qb: Quant, qd: Quant) -> Result<(), i32> {
    const CH: usize = 12288;
    for ci in 0..(C * c.tiles * T) / CH {
        try_rc!(c.read(S_X, ci * CH * 2, A_IN, CH * 2), "res a");
        try_rc!(c.read(region8, ci * CH, A_AUX, CH), "res b");
        // add in place: kernel reads index-aligned, safe to alias
        let a = as_i16(A_IN, CH);
        let dst = unsafe {
            core::slice::from_raw_parts_mut(arena_addr(A_IN) as *mut i16, CH)
        };
        kernels::add_i16_i8(a, qa, as_i8(A_AUX, CH), qb, dst, qd);
        try_rc!(c.write(S_X, ci * CH * 2, A_IN, CH * 2), "res w");
    }
    Ok(())
}

#[inline(never)]
fn encoder(plan: &Plan, c: &mut Ctxt) -> Result<(), i32> {
    let zin = quant8(0.0, plan.conv1_in);

    crate::crumb(0x511);
    // conv1 + gelu1 (mel tiles [80,64] -> mel-rate tiles [384,64] in S_A)
    c.asset("g1lut", A_LUT)?;
    let mel_tiles = 2 * c.tiles;
    for i in 0..mel_tiles {
        assemble_halo(c, S_MEL, 80, mel_tiles, i as i32 * T as i32 - 1, T + 2, zin)?;
        try_rc!(c.npu("wconv1", A_IN, A_OUT), "wconv1");
        lut_apply(A_LUT, A_OUT, TILE8);
        try_rc!(c.write(S_A, i * TILE8, A_OUT, TILE8), "c1 wr");
    }

    crate::crumb(0x512);
    // conv2 parts + gelu2 -> int8 tiles in S_LN ([384,64], parts at row offsets)
    let z2 = quant8(0.0, plan.conv2_in);
    for pi in 0..3usize {
        let nm = Name::of(&["wconv2", PARTS[pi]]);
        c.asset(Name::of(&["g2lut", DIGITS[pi]]).s(), A_LUT)?;
        for i in 0..c.tiles {
            assemble_halo(c, S_A, C, 2 * c.tiles, i as i32 * 128 - 1, 130, z2)?;
            try_rc!(c.npu(nm.s(), A_IN, A_OUT), "wconv2");
            lut_apply(A_LUT, A_OUT, 128 * T);
            try_rc!(c.write(S_LN, i * TILE8 + pi * 128 * T, A_OUT, 128 * T),
                    "c2 wr");
        }
    }

    crate::crumb(0x513);
    // + positional embedding (tile-major f32 on the card) -> int16 S_X
    let pos = lookup("posenc").ok_or(-901)?;
    const CH: usize = 8192;
    for ci in 0..(C * c.tiles * T) / CH {
        try_rc!(c.read(S_LN, ci * CH, A_AUX, CH), "pos a");
        try_rc!(storage::read_blocks(pos.lba + (ci * CH * 4 / storage::BLOCK) as u32,
                                arena(A_IN, 0).as_mut_ptr(),
                                (CH * 4 / storage::BLOCK) as u32), "pos b");
        kernels::add_i8_f32_to_i16(
            as_i8(A_AUX, CH), plan.gelu2, as_f32(A_IN, CH),
            as_i16_mut(A_IN + CH * 4, CH), plan.enc_x,
        );
        try_rc!(c.write(S_X, ci * CH * 2, A_IN + CH * 4, CH * 2), "pos w");
    }

    let mut sq = plan.enc_x;
    for l in 0..BLOCKS {
        let bq = plan.enc[l];
        crate::crumb(0x520 + (l as u32) * 0x10);
        ln_region(c, Name::of(&["e", DIGITS[l], "ln1_gb"]).s(), sq, bq.ln1)?;
        crate::crumb(0x521 + (l as u32) * 0x10);
        for (kind, reg) in [("q", S_QH), ("k", S_KH), ("v", S_VH)] {
            let nm = enc_blob(l, kind, 0);
            for i in 0..c.tiles {
                try_rc!(c.read(S_LN, i * TILE8, A_IN, TILE8), "proj in");
                try_rc!(c.npu(nm.s(), A_IN, A_OUT), "proj");
                for h in 0..HEADS {
                    try_rc!(c.write(reg, (h * N_TILES + i) * HB,
                                    A_OUT + h * HB, HB), "hb wr");
                }
            }
        }
        // attention: a head's K, V and Q tile blocks are contiguous in
        // their regions, so each is one read; keys/values go transposed
        // and widened into the idle slot, every tile's context is staged
        // in the interlayer and written with one command per head
        crate::crumb(0x524 + (l as u32) * 0x10);
        let sm = bq.q_out.scale * bq.k_out.scale / 8.0;
        let stage = c.tiles * HB;
        c.loaded = Entry::default();
        for h in 0..HEADS {
            try_rc!(c.read_raw(S_KH, h * N_TILES * HB, slot_u8(SL_STAGE, stage)), "k rd");
            let t0 = cycles();
            prepare_kv_from_stage(c.ctx);
            prof_add(P_ATTN, t0);
            try_rc!(c.read_raw(S_VH, h * N_TILES * HB, slot_u8(SL_STAGE, stage)), "v rd");
            let t0 = cycles();
            prepare_v_from_stage(c.ctx);
            prof_add(P_ATTN, t0);
            try_rc!(c.read_raw(S_QH, h * N_TILES * HB, slot_u8(SL_STAGE, stage)), "q rd");
            let t0 = cycles();
            let kv = kernels::AttnKv {
                kt: slot_i16(SL_KT, kernels::KV16_LEN),
                v16: slot_i16(SL_V16, kernels::KV16_LEN),
                tk: c.ctx,
            };
            for i in 0..c.tiles {
                let ctx = interlayer_at(IL_CTX + i * HB, HB);
                let ctx = unsafe {
                    core::slice::from_raw_parts_mut(ctx.as_mut_ptr() as *mut i8, HB)
                };
                kernels::attn_head_kt(
                    &slot_i8(SL_STAGE, stage)[i * HB..(i + 1) * HB], &kv, ctx, T, T,
                    bq.q_out.zp, bq.k_out.zp, bq.v_out.zp,
                    sm, bq.v_out.scale, bq.ctx, attn_scratch2(), softmax_exp,
                );
            }
            prof_add(P_ATTN, t0);
            try_rc!(c.write_raw(S_CH, h * N_TILES * HB, interlayer_at(IL_CTX, stage)), "ctx wr");
        }
        // out-projection
        crate::crumb(0x525 + (l as u32) * 0x10);
        let nm = enc_blob(l, "out", 0);
        for i in 0..c.tiles {
            for h in 0..HEADS {
                try_rc!(c.read(S_CH, (h * N_TILES + i) * HB, A_IN + h * HB, HB),
                        "ctx rd");
            }
            try_rc!(c.npu(nm.s(), A_IN, A_OUT), "out");
            try_rc!(c.write(S_O, i * TILE8, A_OUT, TILE8), "o wr");
        }
        crate::crumb(0x526 + (l as u32) * 0x10);
        res_add(c, S_O, sq, bq.out_out, bq.res1)?;
        crate::crumb(0x527 + (l as u32) * 0x10);

        // mlp
        ln_region(c, Name::of(&["e", DIGITS[l], "ln2_gb"]).s(), bq.res1, bq.ln2)?;
        for j in 0..4usize {
            let f1 = enc_blob(l, "fc1", j);
            let p2 = enc_blob(l, "fc2p", j);
            c.asset(Name::of(&["e", DIGITS[l], "lut", DIGITS[j]]).s(), A_LUT)?;
            for i in 0..c.tiles {
                try_rc!(c.read(S_LN, i * TILE8, A_IN, TILE8), "fc in");
                try_rc!(c.npu(f1.s(), A_IN, A_OUT), "fc1");
                lut_apply(A_LUT, A_OUT, TILE8);
                try_rc!(c.npu(p2.s(), A_OUT, A_IN), "fc2p");
                try_rc!(c.write(S_P + j as u32 * HREG_BLOCKS, i * TILE8,
                                A_IN, TILE8), "p wr");
            }
        }
        // recombination: x16 += sum of dequantized partials
        crate::crumb(0x528 + (l as u32) * 0x10);
        const CH2: usize = 8192;
        for ci in 0..(C * c.tiles * T) / CH2 {
            try_rc!(c.read(S_X, ci * CH2 * 2, A_IN, CH2 * 2), "s x");
            for j in 0..4usize {
                try_rc!(c.read(S_P + j as u32 * HREG_BLOCKS, ci * CH2,
                               A_IN + CH2 * 2 + j * CH2, CH2), "s p");
            }
            let parts = [
                as_i8(A_IN + CH2 * 2, CH2),
                as_i8(A_IN + CH2 * 3, CH2),
                as_i8(A_IN + CH2 * 4, CH2),
                as_i8(A_IN + CH2 * 5, CH2),
            ];
            let a = as_i16(A_IN, CH2);
            let dst = unsafe {
                core::slice::from_raw_parts_mut(arena_addr(A_IN) as *mut i16, CH2)
            };
            kernels::fc2_sum(parts, &bq.fc2p_out, a, bq.res1, dst, bq.res2);
            try_rc!(c.write(S_X, ci * CH2 * 2, A_IN, CH2 * 2), "s w");
        }
        sq = bq.res2;
    }

    // final layernorm -> encoder output tiles (int8, enc_out quant)
    crate::crumb(0x570);
    c.asset("lnpost_gb", A_GB)?;
    for i in 0..c.tiles {
        try_rc!(c.read(S_X, i * TILE16, A_IN, TILE16), "lp in");
        kernels::ln_planar_i16_to_i8(
            as_i16(A_IN, C * T), sq,
            as_f32(A_GB, C), as_f32(A_GB + 4 * C, C),
            as_i8_mut(A_OUT, C * T), plan.enc_out, C, T,
        );
        try_rc!(c.write(S_EO, i * TILE8, A_OUT, TILE8), "lp wr");
    }
    Ok(())
}

// --- cross K/V ---------------------------------------------------------------------

#[inline(never)]
fn cross_kv(plan: &Plan, c: &mut Ctxt) -> Result<(), i32> {
    crate::crumb(0x571);
    for l in 0..BLOCKS {
        for (which, kind) in [(0u32, "xk"), (1u32, "xv")] {
            let nm = dec_blob(l, kind, 0);
            for i in 0..c.tiles {
                try_rc!(c.read(S_EO, i * TILE8, A_IN, TILE8), "eo rd");
                // enc.out quant != the submodels' input quant: requantize
                // (the lesson that once garbled the transcript)
                for b in as_i8_mut(A_IN, TILE8) {
                    let f = (*b as i32 - plan.enc_out.zp) as f32 * plan.enc_out.scale;
                    *b = quant8(f, plan.xk_in);
                }
                try_rc!(c.npu(nm.s(), A_IN, A_OUT), "xkv");
                for h in 0..HEADS {
                    try_rc!(c.write(S_XKV + (l as u32 * 2 + which) * HREG_BLOCKS,
                                    (h * N_TILES + i) * HB, A_OUT + h * HB, HB),
                            "xkv wr");
                }
            }
        }
    }
    Ok(())
}

// --- decode ------------------------------------------------------------------------

// Self-attention KV cache: [layer][k|v] planar [384, MAX_TOKENS] int8.
pub static mut SELF_KV: [[i8; C * MAX_TOKENS]; 2 * BLOCKS] =
    [[0; C * MAX_TOKENS]; 2 * BLOCKS];

// Token-rate arena (decode phase): [384,4] tensors.
const W4: usize = 4;
const D_X16: usize = 0; // 3072 int16
const D_LN: usize = 3072; // 1536
const D_Q: usize = 4608;
const D_K: usize = 6144;
const D_V: usize = 7680;
const D_CTX: usize = 9216;
const D_O: usize = 10752;
const D_P: usize = 12288; // 4 x 1536
const D_GB: usize = 18432; // 3072
const D_LUT: usize = 21504; // 512
// LM head: one 64-row chunk of the embedding widened to int16 (49152 B,
// to 74240; the arena's top 8 bytes are the crash breadcrumb). Cross K/V
// no longer stage here: they go through the idle weight slot.
const D_ROWS: usize = 25088;
// 4-bit reconstruction tables for every amax (16 KB, built once per
// decode), to 90624.
const D_LUT16: usize = 74240;
const _: () = assert!(D_LUT16 + 4 * crate::q4::TABLES16 <= crate::ARENA_BYTES - 8);

fn lut16_tables() -> &'static mut [u32; crate::q4::TABLES16] {
    unsafe { &mut *(arena_addr(D_LUT16) as *mut [u32; crate::q4::TABLES16]) }
}

fn sd_read_bytes(e: Entry, byte_off: usize, dst: &mut [u8]) -> i32 {
    // unaligned helper via a bounce block (small reads only)
    let mut bounce = [0u8; 1024];
    let lba = e.lba + (byte_off / storage::BLOCK) as u32;
    let skew = byte_off % storage::BLOCK;
    let blocks = (skew + dst.len()).div_ceil(storage::BLOCK);
    if blocks > 2 {
        return -930; // larger reads go through storage::read_blocks directly
    }
    let rc = storage::read_blocks(lba, bounce.as_mut_ptr(), blocks as u32);
    if rc != 0 {
        return rc;
    }
    dst.copy_from_slice(&bounce[skew..skew + dst.len()]);
    0
}

#[inline(never)]
fn decode(plan: &Plan, c: &mut Ctxt) -> Result<(), i32> {
    let embf = lookup("embf").ok_or(-901)?;
    let embc = lookup("embc4").ok_or(-901)?; // LM-head chunks (2026-09 image)
    let ids = lookup("embpids").ok_or(-901)?;
    let posd = lookup("posdec").ok_or(-901)?;
    let vtb = lookup("vocabtb").ok_or(-901)?;
    let fin = lookup("final_gb.bin").ok_or(-901)?;

    unsafe {
        for m in (*core::ptr::addr_of_mut!(SELF_KV)).iter_mut() {
            m.fill(0);
        }
    }
    crate::q4::tables16(lut16_tables());
    let mut n_tok = 0usize; // cache length
    let mut token = plan.sot[0];
    let mut next_sot = 1usize;
    let mut printed = 0usize;
    // Accumulate the decoded token pieces; emitted on one "Detected:" line.
    let mut transcript = [0u8; 256];
    let mut tlen = 0usize;

    for step in 0..(plan.n_sot - 1 + MAX_TOKENS) {
        // x16 = quantize(embf[kept_pos(token)] + posdec[step])
        let pos_kept = kept_position(ids, token)?;
        let mut row = [0f32; C];
        let mut buf = [0u8; C * 4];
        // embf rows are 1536 B = 3 blocks, block-aligned by construction
        try_rc!(storage::read_blocks(embf.lba + (pos_kept * 3) as u32,
                                buf.as_mut_ptr(), 3), "embf");
        for (i, r) in row.iter_mut().enumerate() {
            *r = f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
        }
        try_rc!(storage::read_blocks(posd.lba + (step * 3) as u32,
                                buf.as_mut_ptr(), 3), "posd");
        let x16 = as_i16_mut(D_X16, C * W4);
        x16.fill(0);
        for i in 0..C {
            let p = f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
            let v = ((libm::roundf((row[i] + p) / plan.dec_x.scale) as i32)
                + plan.dec_x.zp)
                .clamp(-32768, 32767) as i16;
            x16[i * W4] = v;
        }

        let mut sq = plan.dec_x;
        for l in 0..BLOCKS {
            let bq = plan.dec[l];
            dec_ln(c, Name::of(&["b", DIGITS[l], "_ln1_gb.bin"]).s(),
                   D_X16, sq, D_LN, bq.ln1)?;
            try_rc!(c.npu(dec_blob(l, "q", 0).s(), D_LN, D_Q), "dq");
            try_rc!(c.npu(dec_blob(l, "k", 0).s(), D_LN, D_K), "dk");
            try_rc!(c.npu(dec_blob(l, "v", 0).s(), D_LN, D_V), "dv");
            // append column n_tok to the cache (planar stride MAX_TOKENS)
            unsafe {
                let kv = &mut *core::ptr::addr_of_mut!(SELF_KV);
                for ci in 0..C {
                    kv[l * 2][ci * MAX_TOKENS + n_tok] = as_i8(D_K, C * W4)[ci * W4];
                    kv[l * 2 + 1][ci * MAX_TOKENS + n_tok] =
                        as_i8(D_V, C * W4)[ci * W4];
                }
            }
            let t = n_tok + 1;
            let smq = bq.q_out.scale * bq.k_out.scale / 8.0;
            unsafe {
                let kv = &*core::ptr::addr_of!(SELF_KV);
                for h in 0..HEADS {
                    kernels::attn_head(
                        &as_i8(D_Q, C * W4)[h * HD * W4..(h + 1) * HD * W4],
                        &kv[l * 2][h * HD * MAX_TOKENS..(h + 1) * HD * MAX_TOKENS],
                        &kv[l * 2 + 1][h * HD * MAX_TOKENS..(h + 1) * HD * MAX_TOKENS],
                        &mut as_i8_mut(D_CTX, C * W4)[h * HD * W4..(h + 1) * HD * W4],
                        HD, 1, W4, t, MAX_TOKENS,
                        bq.q_out.zp, bq.k_out.zp, bq.v_out.zp,
                        smq, bq.v_out.scale, bq.ctx, attn_scratch(),
                    );
                }
            }
            try_rc!(c.npu(dec_blob(l, "out", 0).s(), D_CTX, D_O), "dout");
            dec_add(D_X16, sq, D_O, bq.out_out, bq.res1);
            sq = bq.res1;

            // cross-attention
            dec_ln(c, Name::of(&["b", DIGITS[l], "_xln_gb.bin"]).s(),
                   D_X16, sq, D_LN, bq.xln)?;
            try_rc!(c.npu(dec_blob(l, "xq", 0).s(), D_LN, D_Q), "dxq");
            let smx = bq.xq_out.scale * bq.xk_out.scale / 8.0;
            // each head's K and V tile blocks: one read each into the
            // slot (idle between dxq and dxout), transposed/widened there
            let stage = c.tiles * HB;
            c.loaded = Entry::default();
            for h in 0..HEADS {
                let kreg = S_XKV + (l as u32 * 2) * HREG_BLOCKS;
                let vreg = S_XKV + (l as u32 * 2 + 1) * HREG_BLOCKS;
                try_rc!(c.read_raw(kreg, h * N_TILES * HB, slot_u8(SL_STAGE, stage)), "xk rd");
                let t0 = cycles();
                prepare_kv_from_stage(c.ctx);
                prof_add(P_ATTN, t0);
                try_rc!(c.read_raw(vreg, h * N_TILES * HB, slot_u8(SL_STAGE, stage)), "xv rd");
                let t0 = cycles();
                prepare_v_from_stage(c.ctx);
                let kv = kernels::AttnKv {
                    kt: slot_i16(SL_KT, kernels::KV16_LEN),
                    v16: slot_i16(SL_V16, kernels::KV16_LEN),
                    tk: c.ctx,
                };
                kernels::attn_head_kt(
                    &as_i8(D_Q, C * W4)[h * HD * W4..(h + 1) * HD * W4], &kv,
                    &mut as_i8_mut(D_CTX, C * W4)[h * HD * W4..(h + 1) * HD * W4],
                    1, W4, bq.xq_out.zp, bq.xk_out.zp, bq.xv_out.zp,
                    smx, bq.xv_out.scale, bq.xctx, attn_scratch2(), softmax_exp,
                );
                prof_add(P_ATTN, t0);
            }
            try_rc!(c.npu(dec_blob(l, "xout", 0).s(), D_CTX, D_O), "dxout");
            dec_add(D_X16, sq, D_O, bq.xout_out, bq.res2);
            sq = bq.res2;

            // mlp
            dec_ln(c, Name::of(&["b", DIGITS[l], "_ln2_gb.bin"]).s(),
                   D_X16, sq, D_LN, bq.ln2)?;
            for j in 0..4usize {
                try_rc!(c.npu(dec_blob(l, "fc1", j).s(), D_LN, D_Q), "dfc1");
                c.asset(Name::of(&["b", DIGITS[l], "_lut", DIGITS[j], ".bin"]).s(),
                        D_LUT)?;
                lut_apply(D_LUT, D_Q, C * W4);
                try_rc!(c.npu(dec_blob(l, "fc2p", j).s(), D_Q, D_P + j * C * W4),
                        "dfc2p");
            }
            {
                let parts = [
                    as_i8(D_P, C * W4),
                    as_i8(D_P + C * W4, C * W4),
                    as_i8(D_P + 2 * C * W4, C * W4),
                    as_i8(D_P + 3 * C * W4, C * W4),
                ];
                let a = as_i16(D_X16, C * W4);
                let dst = unsafe {
                    core::slice::from_raw_parts_mut(arena_addr(D_X16) as *mut i16,
                                                    C * W4)
                };
                kernels::fc2_sum(parts, &bq.fc2p_out, a, sq, dst, bq.res3);
            }
            sq = bq.res3;
        }
        n_tok += 1;

        if step < plan.n_sot - 1 {
            token = plan.sot[next_sot];
            next_sot += 1;
            continue;
        }

        // LM head on the CPU: f32 layernorm + pruned-vocab argmax
        let mut gb = [0u8; 3072];
        try_rc!(storage::read_blocks(fin.lba, gb.as_mut_ptr(), 6), "fin gb");
        let mut hid = [0f32; C];
        let x16 = as_i16(D_X16, C * W4);
        let mut mean = 0f32;
        for i in 0..C {
            hid[i] = (x16[i * W4] as i32 - sq.zp) as f32 * sq.scale;
            mean += hid[i];
        }
        mean /= C as f32;
        let mut var = 0f32;
        for h in hid.iter() {
            var += (h - mean) * (h - mean);
        }
        var /= C as f32;
        let inv = 1.0 / libm::sqrtf(var + 1e-5);
        for (i, h) in hid.iter_mut().enumerate() {
            let g = f32::from_le_bytes(gb[i * 4..i * 4 + 4].try_into().unwrap());
            let b = f32::from_le_bytes(gb[(C + i) * 4..(C + i) * 4 + 4].try_into().unwrap());
            *h = (*h - mean) * inv * g + b;
        }

        let out_idx = step - (plan.n_sot - 1);
        rprintln!("hid[0,1,2,383] {:.4} {:.4} {:.4} {:.4}",
                  hid[0], hid[1], hid[2], hid[C - 1]);
        let best = lm_head(plan, embc, &hid, out_idx == 0)?;
        rprintln!("tok id {}", best);
        if best == plan.eot {
            print_detected(&transcript, tlen);
            rprintln!("=== done ({} tokens) ===", printed);
            return Ok(());
        }
        append_token(vtb, kept_position(ids, best)?, &mut transcript, &mut tlen)?;
        printed += 1;
        token = best;
        if n_tok >= MAX_TOKENS {
            print_detected(&transcript, tlen);
            rprintln!("=== token budget reached ===");
            return Ok(());
        }
    }
    print_detected(&transcript, tlen);
    Ok(())
}

fn dec_ln(c: &Ctxt, gb: &str, src: usize, sq: Quant, dst: usize,
          dq: Quant) -> Result<(), i32> {
    c.asset(gb, D_GB)?;
    kernels::ln_planar_i16_to_i8(
        as_i16(src, C * W4), sq, as_f32(D_GB, C), as_f32(D_GB + 4 * C, C),
        as_i8_mut(dst, C * W4), dq, C, W4,
    );
    Ok(())
}

fn dec_add(x16: usize, qa: Quant, b8: usize, qb: Quant, qd: Quant) {
    let a = as_i16(x16, C * W4);
    let dst = unsafe {
        core::slice::from_raw_parts_mut(arena_addr(x16) as *mut i16, C * W4)
    };
    kernels::add_i16_i8(a, qa, as_i8(b8, C * W4), qb, dst, qd);
}

/// Argmax over the pruned embedding, streamed in 64-row chunks.
///
/// Each "embc4" chunk is one read: the rows' f32 scales and token ids,
/// then the 4-bit weights (group amax + nibbles), which are expanded to
/// int16 and dotted on SMLAD with the hidden vector quantized to a
/// per-utterance int16 grid (gate: model/lm16_check.py, zero argmax
/// flips on the golden decode). Pad rows carry scale 0 and input-only
/// rows (SOT etc) scale -1: both skipped.
///
/// (An exact bound-sorted early exit was measured and rejected: Whisper
/// LM-head cosines are so small that even the loosest row's Cauchy-Schwarz
/// bound sits ~3x above the best logit -- 0 of 12228 rows prunable.)
fn lm_head(plan: &Plan, embc: Entry, hid: &[f32; C], first: bool) -> Result<u32, i32> {
    const ROWS: usize = 64;
    const CH_SCL: usize = 0; // f32[64]
    const CH_IDS: usize = ROWS * 4; // u32[64]
    const CH_AMAX: usize = 2 * ROWS * 4; // u8[64 * 6]
    const CH_NIBS: usize = CH_AMAX + ROWS * C / crate::q4::G;
    const CH_USED: usize = CH_NIBS + ROWS * C / 2;
    const CH_BLOCKS: usize = CH_USED.div_ceil(storage::BLOCK);
    const _: () = assert!(CH_BLOCKS == crate::q4::EMB_CHUNK_BLOCKS);

    let mut hmax = 0f32;
    for &h in hid.iter() {
        hmax = hmax.max(libm::fabsf(h));
    }
    let hs = if hmax > 0.0 { hmax / 32767.0 } else { 1.0 };
    let mut hq = [0i16; C];
    for (q, &h) in hq.iter_mut().zip(hid.iter()) {
        *q = dsp::round_i32(h / hs).clamp(-32767, 32767) as i16;
    }

    let mut best = f32::MIN;
    let mut best2 = f32::MIN;
    let mut best_id = plan.eot;
    let rows16 = as_i16_mut(D_ROWS, ROWS * C);
    for chunk in 0..plan.vocab_n.div_ceil(ROWS) {
        // the packed chunk bounces through the interlayer buffer:
        // transient use between NPU runs is fine (nothing persists)
        let il = interlayer(CH_BLOCKS * storage::BLOCK);
        try_rc!(storage::read_blocks(embc.lba + (chunk * CH_BLOCKS) as u32,
                                     il.as_mut_ptr(), CH_BLOCKS as u32),
                "embc rd");
        let t0 = cycles();
        crate::q4::unpack16(lut16_tables(), &il[CH_AMAX..CH_NIBS], &il[CH_NIBS..CH_USED],
                            rows16);
        prof_add(P_UNPACK, t0);
        let t0 = cycles();
        let n = ROWS.min(plan.vocab_n - chunk * ROWS);
        let mut r = 0;
        while r < ROWS {
            let d = unsafe { dsp::dot384_2rows(rows16.as_ptr().add(r * C), hq.as_ptr()) };
            for k in 0..2 {
                let row = r + k;
                let s = f32::from_le_bytes(
                    il[CH_SCL + row * 4..CH_SCL + row * 4 + 4].try_into().unwrap());
                if row >= n || s <= 0.0 {
                    continue;
                }
                let logit = d[k] as f32 * (s * hs);
                if logit > best {
                    let id = u32::from_le_bytes(
                        il[CH_IDS + row * 4..CH_IDS + row * 4 + 4].try_into().unwrap());
                    if first
                        && (id == plan.eot || plan.blank[..plan.n_blank].contains(&id))
                    {
                        continue; // blank suppression on the first sampled token
                    }
                    best2 = best;
                    best = logit;
                    best_id = id;
                } else if logit > best2 {
                    best2 = logit;
                }
            }
            r += 2;
        }
        prof_add(P_LM, t0);
    }
    rprintln!("lm: id {} logit {:.3} (2nd {:.3})", best_id, best, best2);
    Ok(best_id)
}

/// Binary search the kept-id table for a token id -> kept position.
fn kept_position(ids: Entry, token: u32) -> Result<usize, i32> {
    let mut lo = 0usize;
    let mut hi = {
        // count = entry length / 4
        (ids.len / 4) as usize
    };
    let mut b = [0u8; 4];
    while lo < hi {
        let mid = (lo + hi) / 2;
        try_rc!(sd_read_bytes(ids, mid * 4, &mut b), "ids bs");
        let v = u32::from_le_bytes(b);
        if v == token {
            return Ok(mid);
        }
        if v < token {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    rprintln!("token {} not in kept vocabulary", token);
    Err(-906)
}

/// Append a kept token's text piece from the vocabulary table to the
/// running transcript buffer (`buf[..len]`), truncating if it fills.
fn append_token(vtb: Entry, kept_pos: usize, buf: &mut [u8; 256],
                len: &mut usize) -> Result<(), i32> {
    let mut offs = [0u8; 8];
    try_rc!(sd_read_bytes(vtb, 4 + kept_pos * 4, &mut offs), "vtb off");
    let o0 = u32::from_le_bytes(offs[0..4].try_into().unwrap()) as usize;
    let o1 = u32::from_le_bytes(offs[4..8].try_into().unwrap()) as usize;
    let n = (o1 - o0).min(48);
    let mut sbuf = [0u8; 48];
    // strings start after the offset table: 4 + (n+1)*4 bytes in
    let n_off = {
        let mut nb = [0u8; 4];
        try_rc!(sd_read_bytes(vtb, 0, &mut nb), "vtb n");
        u32::from_le_bytes(nb) as usize
    };
    let base = 4 + (n_off + 1) * 4;
    try_rc!(sd_read_bytes(vtb, base + o0, &mut sbuf[..n]), "vtb s");
    let take = n.min(buf.len() - *len);
    buf[*len..*len + take].copy_from_slice(&sbuf[..take]);
    *len += take;
    Ok(())
}

/// Emit the decoded transcript on its own line, prefixed "Detected: ",
/// to both the RTT log and the OLED (leading BPE space trimmed).
fn print_detected(buf: &[u8; 256], len: usize) {
    if let Ok(s) = core::str::from_utf8(&buf[..len]) {
        let s = s.trim_start();
        rprintln!("");
        rprintln!("Detected: {}", s);
        display::print("\nDetected: ");
        display::print(s);
        display::print("\n");
    }
}
