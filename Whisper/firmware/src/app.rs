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
#[cfg(feature = "sd-card")]
use crate::sd;
use crate::{display, mel, pdm, slot, storage};
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
const S_XKV: u32 = 9472; // cross K/V head blocks, 8 x 480 (l*2 + [k|v]); K key-major

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
// Softmax exponential table for the encoder's attention, above the
// staging area: nothing else reaches this high in the slot (the largest
// blob is 166 KB), so it survives the block's blob loads and is rebuilt
// per block for that block's score multiplier.
const SL_EXPTAB: usize = SL_STAGE + N_TILES * HB;
const _: () = assert!(SL_EXPTAB % 4 == 0);
const _: () = assert!(SL_EXPTAB + kernels::EXP_TABLE_BYTES <= slot::SLOT_BYTES);
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

fn exp_table() -> &'static mut kernels::ExpTable {
    let b = slot_u8(SL_EXPTAB, kernels::EXP_TABLE_BYTES);
    unsafe { &mut *(b.as_mut_ptr() as *mut kernels::ExpTable) }
}

/// Softmax exponential for the decode cross-attention: `dsp::exp_neg` is
/// within ~2 ulp of libm::expf at about a third of the cost; `false`
/// reproduces the numpy golden bit for bit (tools/attncheck reports what
/// the fast one moves). The encoder uses the integer-indexed table
/// (kernels::ExpTable) instead: its ~1300-entry rebuild per score
/// multiplier pays off over a block's 2 M keys, not over a decode
/// layer's 3.5 K.
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
fn prepare_kv_from(stage: &[i8], tk: usize) {
    let tkp = kernels::keys_padded(tk);
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
fn prepare_v_from(stage: &[i8], tk: usize) {
    let tkp = kernels::keys_padded(tk);
    let v16 = slot_i16(SL_V16, kernels::KV16_LEN);
    for ch in 0..HD {
        let row = &mut v16[ch * kernels::MAX_KEYS..ch * kernels::MAX_KEYS + tkp];
        for i in 0..tk.div_ceil(T) {
            let take = T.min(tk - i * T);
            dsp::widen_i8_i16(&stage[i * HB + ch * T..i * HB + ch * T + take],
                              &mut row[i * T..i * T + take]);
        }
        row[tk..].fill(0);
    }
}

/// One 64x64 int8 head block transposed: [row][col] -> [col][row].
fn transpose64(src: &[i8], dst: &mut [i8]) {
    for r in 0..T {
        for (c, &x) in src[r * T..(r + 1) * T].iter().enumerate() {
            dst[c * T + r] = x;
        }
    }
}

// --- CPU phase profile ---------------------------------------------------------
const P_ATTN: usize = 0;
const P_LM: usize = 1;
const P_UNPACK: usize = 2;
const P_SUM: usize = 3;
const P_NPU: usize = 4;
static mut PROF: [u64; 5] = [0; 5];

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
/// carried in the LAY5 header; drift is a hard error, checked once).
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
    /// Block address of a byte offset inside a scratch region.
    fn lba(&self, region: u32, byte_off: usize) -> u32 {
        self.scratch + region + (byte_off / storage::BLOCK) as u32
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
        let e = match lookup(name) {
            Some(e) => e,
            None => {
                rprintln!("FAIL no asset {}", name);
                return Err(-901);
            }
        };
        let rc = storage::read_blocks(e.lba, arena(off, 0).as_mut_ptr(),
                                 e.len.div_ceil(storage::BLOCK as u32));
        if rc != 0 {
            rprintln!("FAIL asset {} rc={}", name, rc);
            return Err(rc);
        }
        Ok(e)
    }

    /// Blob into the slot (cached, sum-verified) + one NPU inference on
    /// arena offsets.
    fn npu(&mut self, blob: &str, input: usize, output: usize) -> i32 {
        self.npu_addr(blob, arena_addr(input), arena_addr(output))
    }

    /// Same on absolute activation addresses (any RAM the tensor fits).
    fn npu_addr(&mut self, blob: &str, input: u32, output: u32) -> i32 {
        let rc = self.load(blob);
        if rc != 0 {
            return rc;
        }
        self.run_loaded(blob, input, output)
    }

    /// Run the blob already in the slot (after `load`); split from it so
    /// a pipeline can start a DMA transfer between the blob read and the
    /// NPU run.
    fn run_loaded(&mut self, blob: &str, input: u32, output: u32) -> i32 {
        let _wd = crate::WdogGuard::arm();
        let t0 = cycles();
        let rc = unsafe { slot::run(input, output, blob) };
        prof_add(P_NPU, t0);
        rc
    }

    /// `run_loaded` with a transfer queue pumped from the inference wait
    /// loop, so several small transfers (a tile's six head-block writes)
    /// complete behind one NPU run. The queue's error, if any, surfaces
    /// at its next pump or drain.
    fn run_pumped(&mut self, blob: &str, input: u32, output: u32, q: &mut IoQueue) -> i32 {
        unsafe { PUMP_Q = q as *mut IoQueue };
        crate::platform::set_wait_hook(Some(pump_hook));
        let rc = self.run_loaded(blob, input, output);
        crate::platform::set_wait_hook(None);
        unsafe { PUMP_Q = core::ptr::null_mut() };
        rc
    }

    /// Blob into the slot (cached by address, sum-verified).
    fn load(&mut self, blob: &str) -> i32 {
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
            self.loaded = e;
        }
        0
    }
}

// --- storage/compute overlap ----------------------------------------------------
//
// A short queue of block transfers issued one after another through the
// split-phase storage interface. `pump` is called between units of CPU
// work: it retires a finished transfer and starts the next, so the DMA
// runs while the CPU computes. `drain` blocks for whatever is left. The
// first error stops the queue and is reported by whichever call sees it.
struct IoOp {
    read: bool,
    lba: u32,
    buf: u32,
    count: u32,
    what: &'static str,
}

struct IoQueue {
    ops: [IoOp; IOQ_MAX],
    n: usize,
    next: usize,
    rc: i32,
}

const IOQ_MAX: usize = 8;

/// The queue `run_pumped` is advancing, for the wait-loop hook.
static mut PUMP_Q: *mut IoQueue = core::ptr::null_mut();

fn pump_hook() {
    // SAFETY: set for the duration of one run_pumped call, whose queue
    // outlives it; the hook runs in thread mode inside that call.
    unsafe {
        let q = PUMP_Q;
        if !q.is_null() {
            let _ = (*q).pump();
        }
    }
}

impl IoQueue {
    const fn new() -> IoQueue {
        const E: IoOp = IoOp { read: true, lba: 0, buf: 0, count: 0, what: "" };
        IoQueue { ops: [E; IOQ_MAX], n: 0, next: 0, rc: 0 }
    }

    fn push(&mut self, read: bool, lba: u32, buf: &[u8], what: &'static str) {
        assert!(self.n < IOQ_MAX && buf.len() % storage::BLOCK == 0);
        self.ops[self.n] = IoOp {
            read,
            lba,
            buf: buf.as_ptr() as u32,
            count: (buf.len() / storage::BLOCK) as u32,
            what,
        };
        self.n += 1;
    }

    /// Retire a completed transfer and start the next one; never blocks.
    fn pump(&mut self) -> Result<(), i32> {
        if self.rc != 0 {
            return Err(self.rc);
        }
        if storage::busy() {
            if !storage::poll() {
                return Ok(());
            }
            let rc = storage::finish();
            if rc != 0 {
                return self.fail(self.next - 1, rc);
            }
        }
        if self.next < self.n {
            let op = &self.ops[self.next];
            let rc = if op.read {
                storage::read_start(op.lba, op.buf as *mut u8, op.count)
            } else {
                storage::write_start(op.lba, op.buf as *const u8, op.count)
            };
            self.next += 1;
            if rc != 0 {
                return self.fail(self.next - 1, rc);
            }
        }
        Ok(())
    }

    fn fail(&mut self, i: usize, rc: i32) -> Result<(), i32> {
        rprintln!("FAIL {} rc={}", self.ops[i].what, rc);
        self.rc = rc;
        Err(rc)
    }

    /// Block until every queued transfer is through, then reset.
    fn drain(&mut self) -> Result<(), i32> {
        loop {
            self.pump()?;
            if self.next == self.n && !storage::busy() {
                break;
            }
        }
        self.n = 0;
        self.next = 0;
        Ok(())
    }
}

/// Wait for one split-phase transfer started outside a queue.
fn io_wait(what: &str) -> Result<(), i32> {
    let rc = storage::finish();
    if rc != 0 {
        rprintln!("FAIL {} rc={}", what, rc);
        return Err(rc);
    }
    Ok(())
}

fn io_start(read: bool, lba: u32, buf: &[u8], what: &str) -> Result<(), i32> {
    let rc = if read {
        storage::read_start(lba, buf.as_ptr() as *mut u8, (buf.len() / storage::BLOCK) as u32)
    } else {
        storage::write_start(lba, buf.as_ptr(), (buf.len() / storage::BLOCK) as u32)
    };
    if rc != 0 {
        rprintln!("FAIL {} rc={}", what, rc);
        return Err(rc);
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
    lut_apply_to(lut_off, as_i8_mut(buf_off, len));
}

fn lut_apply_to(lut_off: usize, buf: &mut [i8]) {
    let lut: [i8; 256] = core::array::from_fn(|i| as_i8(lut_off, 256)[i]);
    for b in buf {
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
    if cfg!(feature = "mock-usb") {
        rprintln!("standalone: storage init (host image over USB device mode)");
    } else {
        rprintln!("standalone: storage init (USB stick, {} tries)", storage::USB_TRIES);
    }
    // the DSP-extension kernels against their scalar definitions (the
    // arena is free this early)
    let bad = dsp::selftest(arena(0, 8192));
    if bad == 0 {
        rprintln!("standalone: dsp kernels self-test ok");
    } else {
        rprintln!("standalone: DSP KERNEL SELF-TEST FAILED (mask {:#x}); results will be wrong",
                  bad);
        display::print("DSP SELFTEST FAIL\n");
    }
    let rc = storage::init();
    if rc != 0 {
        rprintln!("standalone: no storage ({}), staying in mailbox mode", rc);
        display::print("no storage\n");
        #[cfg(feature = "sd-card")]
        {
            sd::diag(2);
            sd::release_pins();
            rprintln!("sd pins released (high-Z): external testers may drive the bus");
        }
        crate::mailbox_loop();
    }
    rprintln!("standalone: model source: {}", storage::name());
    unsafe {
        // The first read after enumeration is the one a stick may still
        // be waking up for; give it a few tries before giving up.
        let mut rc = -650;
        for attempt in 0..3 {
            rc = storage::read_blocks(0, core::ptr::addr_of_mut!(INDEX) as *mut u8, 16);
            if rc == 0 {
                break;
            }
            rprintln!("standalone: index read {}/3 failed rc={}", attempt + 1, rc);
            cortex_m::asm::delay(64_000_000); // 0.5 s
        }
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
    // A pipeline that failed mid-flight leaves its transfer pending; every
    // blocking storage call would then refuse with -495.
    if storage::busy() {
        let _ = storage::finish();
    }
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
        capture_done(false);
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
/// The ms figures are CPU time spent in storage calls: split-phase
/// transfers only count the part that was not hidden behind compute.
fn chase_stats() {
    let st = unsafe { &mut *core::ptr::addr_of_mut!(CHASE_STATS) };
    if st[0] > 0 {
        rprintln!("chase: {} blobs, first poll {} KB landed, {} KB of {} KB nibbles expanded before completion, {}.{} polls per blob",
                  st[0], st[1] / st[0] / 1024, st[2] / st[0] / 1024,
                  C * C / 2 / 1024, st[3] / st[0], st[3] * 10 / st[0] % 10);
    }
    *st = [0; 4];
}

fn sd_stats(phase: &str) {
    chase_stats();
    let (rb, rc, wb, wc) = storage::stats_take();
    rprintln!(
        "sd[{}]: rd {} KB / {} ms, wr {} KB / {} ms",
        phase, rb / 1024, rc / 128_000, wb / 1024, wc / 128_000
    );
    let p = unsafe { core::mem::replace(&mut *core::ptr::addr_of_mut!(PROF), [0; 5]) };
    let a = unsafe {
        core::mem::replace(&mut *core::ptr::addr_of_mut!(kernels::ATTN_PROF), [0; 3])
    };
    let v = unsafe { core::mem::replace(&mut *core::ptr::addr_of_mut!(slot::VALIDATE_CYCLES), 0) };
    rprintln!(
        "cpu[{}]: attn {} ms (qk {}, softmax {}, pv {}), lm {} ms, unpack {} ms, \
         sums {} ms, npu {} ms (validate {})",
        phase, p[P_ATTN] / 128_000, a[0] / 128_000, a[1] / 128_000, a[2] / 128_000,
        p[P_LM] / 128_000, p[P_UNPACK] / 128_000, p[P_SUM] / 128_000, p[P_NPU] / 128_000,
        v / 128_000
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

/// Latency bookkeeping for the line printed after the transcript: when
/// the audio was complete (uptime ms), whether it came from the mic, and
/// how much of the recording followed the last speech frame.
static mut CAPTURE_END_MS: u32 = 0;
static mut CAPTURE_LIVE: bool = false;
static mut SPEECH_TAIL_MS: u32 = 0;

fn capture_done(live: bool) {
    unsafe {
        CAPTURE_END_MS = crate::uptime_ms();
        CAPTURE_LIVE = live;
    }
}

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
    capture_done(true);
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
    capture_done(true);
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
    // mel frames are 10 ms apart: recording that followed the speech
    unsafe { SPEECH_TAIL_MS = ((n_frames - 1 - last) * 10) as u32 };
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
            // Tile pipeline: tile i-1's six head-block writes drain behind
            // tile i's NPU run (queue pumped from the inference wait), so
            // the input and output buffers alternate: inputs in the two
            // halves of A_IN, outputs in A_OUT and the idle KV cache.
            let nm = enc_blob(l, kind, 0);
            let mut ioq = IoQueue::new();
            try_rc!(c.load(nm.s()), "proj load");
            try_rc!(c.read(S_LN, 0, A_IN, TILE8), "proj in");
            for i in 0..c.tiles {
                let x = if i % 2 == 0 { A_IN } else { A_IN + TILE8 };
                let (oaddr, out): (u32, &[u8]) = if i % 2 == 0 {
                    (arena_addr(A_OUT), arena(A_OUT, TILE8))
                } else {
                    (kvscratch_addr(KV_PO), kvscratch(KV_PO, TILE8))
                };
                try_rc!(c.run_pumped(nm.s(), arena_addr(x), oaddr, &mut ioq), "proj");
                ioq.drain()?;
                if i + 1 < c.tiles {
                    let xn = if i % 2 == 0 { A_IN + TILE8 } else { A_IN };
                    try_rc!(c.read(S_LN, (i + 1) * TILE8, xn, TILE8), "proj in");
                }
                for h in 0..HEADS {
                    ioq.push(false, c.lba(reg, (h * N_TILES + i) * HB),
                             &out[h * HB..(h + 1) * HB], "hb wr");
                }
                ioq.pump()?;
            }
            ioq.drain()?;
        }
        // attention: a head's K, V and Q tile blocks are contiguous in
        // their regions, so each is one read; keys/values go transposed
        // and widened into the idle slot. While head h computes, the DMA
        // writes head h-1's context tiles and fetches head h+1's K, V
        // and Q into the arena / KV scratch (pumped every 16 queries).
        crate::crumb(0x524 + (l as u32) * 0x10);
        let sm = bq.q_out.scale * bq.k_out.scale / 8.0;
        let stage = c.tiles * HB;
        c.loaded = Entry::default();
        let t0 = cycles();
        exp_table().build(sm);
        prof_add(P_ATTN, t0);
        let mut ioq = IoQueue::new();
        ioq.push(true, c.lba(S_KH, 0), arena(A_PK, stage), "k rd");
        ioq.push(true, c.lba(S_VH, 0), arena(A_PV, stage), "v rd");
        ioq.push(true, c.lba(S_QH, 0), kvscratch(KV_PQ, stage), "q rd");
        ioq.drain()?;
        for h in 0..HEADS {
            let t0 = cycles();
            prepare_kv_from(as_i8(A_PK, stage), c.ctx);
            prepare_v_from(as_i8(A_PV, stage), c.ctx);
            slot_u8(SL_STAGE, stage).copy_from_slice(kvscratch(KV_PQ, stage));
            prof_add(P_ATTN, t0);
            if h > 0 {
                let prev = if (h - 1) % 2 == 0 {
                    interlayer_at(IL_CTX, stage)
                } else {
                    kvscratch(KV_CTX1, stage)
                };
                ioq.push(false, c.lba(S_CH, (h - 1) * N_TILES * HB), prev, "ctx wr");
            }
            if h + 1 < HEADS {
                let nb = (h + 1) * N_TILES * HB;
                ioq.push(true, c.lba(S_KH, nb), arena(A_PK, stage), "k rd");
                ioq.push(true, c.lba(S_VH, nb), arena(A_PV, stage), "v rd");
                ioq.push(true, c.lba(S_QH, nb), kvscratch(KV_PQ, stage), "q rd");
            }
            ioq.pump()?;
            let t0 = cycles();
            let kv = kernels::AttnKv {
                kt: slot_i16(SL_KT, kernels::KV16_LEN),
                v16: slot_i16(SL_V16, kernels::KV16_LEN),
                tk: c.ctx,
            };
            const QSTEP: usize = 16;
            for i in 0..c.tiles {
                let ctx = if h % 2 == 0 {
                    interlayer_at(IL_CTX + i * HB, HB)
                } else {
                    kvscratch(KV_CTX1 + i * HB, HB)
                };
                let ctx = unsafe {
                    core::slice::from_raw_parts_mut(ctx.as_mut_ptr() as *mut i8, HB)
                };
                let q = &slot_i8(SL_STAGE, stage)[i * HB..(i + 1) * HB];
                for q0 in (0..T).step_by(QSTEP) {
                    kernels::attn_head_kt(
                        &q[q0..], &kv, &mut ctx[q0..], QSTEP, T,
                        bq.q_out.zp, bq.k_out.zp, bq.v_out.zp,
                        sm, bq.v_out.scale, bq.ctx, attn_scratch2(), &*exp_table(),
                    );
                    ioq.pump()?;
                }
            }
            prof_add(P_ATTN, t0);
            ioq.drain()?;
        }
        {
            let last = if (HEADS - 1) % 2 == 0 {
                interlayer_at(IL_CTX, stage)
            } else {
                kvscratch(KV_CTX1, stage)
            };
            try_rc!(c.write_raw(S_CH, (HEADS - 1) * N_TILES * HB, last), "ctx wr");
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
        // Tiles in pairs per blob load: fc1 runs on both tiles of a pair
        // (outputs in A_OUT and the KV scratch), then fc2p on both, so
        // the two blobs alternate every other tile and their reloads,
        // most of the encoder's storage reads, halve. The previous
        // pair's partial writes drain behind the fc1 runs, the next
        // pair's input reads behind the fc2p runs (queue pumped from the
        // inference wait). The blobs load synchronously between, when
        // nothing is in flight.
        for j in 0..4usize {
            let f1 = enc_blob(l, "fc1", j);
            let p2 = enc_blob(l, "fc2p", j);
            c.asset(Name::of(&["e", DIGITS[l], "lut", DIGITS[j]]).s(), A_LUT)?;
            let preg = S_P + j as u32 * HREG_BLOCKS;
            let mut ioq = IoQueue::new();
            try_rc!(c.read(S_LN, 0, A_X0, TILE8), "fc in");
            if c.tiles > 1 {
                try_rc!(c.read(S_LN, TILE8, A_X1, TILE8), "fc in");
            }
            let mut i = 0;
            while i < c.tiles {
                let two = i + 1 < c.tiles;
                try_rc!(c.load(f1.s()), "fc1 load");
                if i > 0 {
                    ioq.push(false, c.lba(preg, (i - 2) * TILE8), kvscratch(KV_P0, TILE8), "p wr");
                    ioq.push(false, c.lba(preg, (i - 1) * TILE8), kvscratch(KV_P1, TILE8), "p wr");
                    ioq.pump()?;
                }
                try_rc!(c.run_pumped(f1.s(), arena_addr(A_X0), arena_addr(A_OUT), &mut ioq), "fc1");
                if two {
                    try_rc!(c.run_pumped(f1.s(), arena_addr(A_X1), kvscratch_addr(KV_F1), &mut ioq),
                            "fc1");
                }
                ioq.drain()?;
                try_rc!(c.load(p2.s()), "fc2p load");
                if i + 2 < c.tiles {
                    ioq.push(true, c.lba(S_LN, (i + 2) * TILE8), arena(A_X0, TILE8), "fc in");
                    if i + 3 < c.tiles {
                        ioq.push(true, c.lba(S_LN, (i + 3) * TILE8), arena(A_X1, TILE8), "fc in");
                    }
                    ioq.pump()?;
                }
                lut_apply(A_LUT, A_OUT, TILE8);
                try_rc!(c.run_pumped(p2.s(), arena_addr(A_OUT), kvscratch_addr(KV_P0), &mut ioq),
                        "fc2p");
                if two {
                    let f1b = kvscratch(KV_F1, TILE8);
                    let f1b = unsafe {
                        core::slice::from_raw_parts_mut(f1b.as_mut_ptr() as *mut i8, TILE8)
                    };
                    lut_apply_to(A_LUT, f1b);
                    try_rc!(c.run_pumped(p2.s(), kvscratch_addr(KV_F1), kvscratch_addr(KV_P1),
                                         &mut ioq), "fc2p");
                }
                ioq.drain()?;
                i += 2;
            }
            // the last pair's partials
            let first = if c.tiles % 2 == 0 { c.tiles - 2 } else { c.tiles - 1 };
            try_rc!(c.write_raw(preg, first * TILE8, kvscratch(KV_P0, TILE8)), "p wr");
            if c.tiles % 2 == 0 {
                try_rc!(c.write_raw(preg, (first + 1) * TILE8, kvscratch(KV_P1, TILE8)), "p wr");
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
    // enc.out quant != the submodels' input quant: requantize (the lesson
    // that once garbled the transcript)
    let requant = |c: &Ctxt, i: usize, x: usize| -> Result<(), i32> {
        try_rc!(c.read(S_EO, i * TILE8, x, TILE8), "eo rd");
        for b in as_i8_mut(x, TILE8) {
            let f = (*b as i32 - plan.enc_out.zp) as f32 * plan.enc_out.scale;
            *b = quant8(f, plan.xk_in);
        }
        Ok(())
    };
    for l in 0..BLOCKS {
        for (which, kind) in [(0u32, "xk"), (1u32, "xv")] {
            // Same tile pipeline as the encoder projections: tile i-1's
            // head-block writes drain behind tile i's NPU run. Inputs
            // alternate in the halves of A_IN, outputs and the transposed
            // key blocks in the idle KV cache.
            let nm = dec_blob(l, kind, 0);
            let reg = S_XKV + (l as u32 * 2 + which) * HREG_BLOCKS;
            let mut ioq = IoQueue::new();
            try_rc!(c.load(nm.s()), "xkv load");
            requant(c, 0, A_IN)?;
            for i in 0..c.tiles {
                let (x, o, t) = if i % 2 == 0 {
                    (A_IN, KV_XO0, KV_XT0)
                } else {
                    (A_IN + TILE8, KV_XO1, KV_XT1)
                };
                try_rc!(c.run_pumped(nm.s(), arena_addr(x), kvscratch_addr(o), &mut ioq),
                        "xkv");
                ioq.drain()?;
                if i + 1 < c.tiles {
                    requant(c, i + 1, if i % 2 == 0 { A_IN + TILE8 } else { A_IN })?;
                }
                // Keys go out key-major ([64 keys][64 channels] per head
                // block) so decode widens them straight into the attention
                // kernel's layout instead of transposing every head on
                // every token. Values keep the NPU's channel-major layout.
                let src = if which == 0 {
                    for h in 0..HEADS {
                        let blk = kvscratch(o + h * HB, HB);
                        let blk = unsafe {
                            core::slice::from_raw_parts(blk.as_ptr() as *const i8, HB)
                        };
                        let dst = kvscratch(t + h * HB, HB);
                        let dst = unsafe {
                            core::slice::from_raw_parts_mut(dst.as_mut_ptr() as *mut i8, HB)
                        };
                        transpose64(blk, dst);
                    }
                    t
                } else {
                    o
                };
                for h in 0..HEADS {
                    ioq.push(false, c.lba(reg, (h * N_TILES + i) * HB),
                             kvscratch(src + h * HB, HB), "xkv wr");
                }
                ioq.pump()?;
            }
            ioq.drain()?;
        }
    }
    Ok(())
}

// --- decode ------------------------------------------------------------------------

// Self-attention KV cache: [layer][k|v] planar [384, MAX_TOKENS] int8.
// Word-aligned because the encoder borrows it as DMA scratch (KV_*).
#[repr(C, align(4))]
pub struct KvCache(pub [[i8; C * MAX_TOKENS]; 2 * BLOCKS]);
pub static mut SELF_KV: KvCache = KvCache([[0; C * MAX_TOKENS]; 2 * BLOCKS]);
pub const KV_BYTES: usize = 2 * BLOCKS * C * MAX_TOKENS; // 98304

/// The KV cache as encoder-phase scratch (it is zeroed when decode starts).
fn kvscratch(off: usize, len: usize) -> &'static mut [u8] {
    assert!(off + len <= KV_BYTES);
    unsafe {
        core::slice::from_raw_parts_mut(
            (core::ptr::addr_of_mut!(SELF_KV) as *mut u8).add(off), len)
    }
}

fn kvscratch_addr(off: usize) -> u32 {
    unsafe { (core::ptr::addr_of!(SELF_KV) as *const u8).add(off) as u32 }
}

// Encoder attention pipeline buffers: the next head's K/V/Q tile blocks
// land here by DMA while the current head computes (STAGE_BYTES each),
// and the context tiles alternate between two buffers so one head's write
// overlaps the next head's attention. The MLP tile pipeline reuses the
// same memory for its double-buffered input and fc2 partial.
const STAGE_BYTES: usize = N_TILES * HB; // 40960
const A_PK: usize = 0; // arena: next head's K blocks
const A_PV: usize = STAGE_BYTES; // arena: next head's V blocks
const _: () = assert!(A_PV + STAGE_BYTES <= crate::ARENA_BYTES);
const KV_PQ: usize = 0; // kv scratch: next head's Q blocks
const KV_CTX1: usize = STAGE_BYTES; // kv scratch: second context buffer
const _: () = assert!(KV_CTX1 + STAGE_BYTES <= KV_BYTES);
// MLP tile pipeline: input tiles alternate A_X0/A_X1, fc2 partials KV_P0/KV_P1.
const A_X0: usize = 0;
const A_X1: usize = TILE8;
const _: () = assert!(A_X1 + TILE8 <= A_OUT);
const KV_P0: usize = 0;
const KV_P1: usize = TILE8;
// second tile's fc1 output (the first's is A_OUT)
const KV_F1: usize = 2 * TILE8;
const _: () = assert!(KV_F1 + TILE8 <= KV_BYTES);
// projection pipeline: odd tiles' NPU output
const KV_PO: usize = 0;
// cross K/V pipeline: NPU output per parity, then the transposed K blocks
const KV_XO0: usize = 0;
const KV_XO1: usize = TILE8;
const KV_XT0: usize = 2 * TILE8;
const KV_XT1: usize = 3 * TILE8;
const _: () = assert!(KV_XT1 + TILE8 <= KV_BYTES);
const _: () = assert!(2 * TILE8 <= A_OUT); // two input tiles in A_IN

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
// One decoder layer's constants ("dc<l>" image entry: ln1, xln and ln2
// gamma/beta, then the four GELU tables), read once per layer per step
// instead of seven small reads.
const D_DC: usize = 18432;
const DC_LN1: usize = 0;
const DC_XLN: usize = 4 * 2 * C;
const DC_LN2: usize = 2 * 4 * 2 * C;
const DC_LUT: usize = 3 * 4 * 2 * C; // 4 x 256
const DC_BYTES: usize = DC_LUT + 4 * 256; // 10240
// The rest of the arena (from 28672) is the landing zone below.

// --- decode blob pipeline -----------------------------------------------------
//
// The per-token decoder blobs are stored 4-bit ("LAY5", model/quant4.py)
// and linked at DEC_BASE, the top DEC_BYTES of the slot. A packed entry
// is read by DMA into the landing zone LAND (the free top of the arena
// and the bottom of the slot, which are contiguous) while the NPU runs
// the previous blob at DEC_BASE; the CPU then expands it into DEC_BASE
// through the level table. The cross-attention K/V stages and the
// LM head's chunk buffers borrow the DEC region between blob runs. A raw
// int8 decoder blob (an older image, linked at the slot base) does not
// fit the landing zone and runs through the unpipelined Ctxt::npu.
const ARENA_BASE: usize = 0x2003_2000; // memory.x ARENA (checked at decode start)
const DEC_BYTES: usize = 152 * 1024; // largest per-token blob: 152356 B
const DEC_OFF: usize = slot::SLOT_BYTES - DEC_BYTES; // 57344
const DEC_BASE: usize = slot::SLOT_BASE + DEC_OFF; // 0x20059000
const LAND_ARENA: usize = D_DC + DC_BYTES; // 28672
const LAND_BASE: usize = ARENA_BASE + LAND_ARENA;
const LAND_BYTES: usize = (crate::ARENA_BYTES - LAND_ARENA) + DEC_OFF; // 131072
const _: () = assert!(ARENA_BASE + crate::ARENA_BYTES == slot::SLOT_BASE);
const _: () = assert!(LAND_BASE % 4 == 0 && DEC_BASE % 4 == 0);
// the crash breadcrumbs at the top of the slot are written during every
// NPU run: nothing the decode lands, expands or stages may reach them
const CRUMB_OFF: usize = crate::CRUMB_BASE - slot::SLOT_BASE;
const _: () = assert!(LAND_BASE + LAND_BYTES <= crate::CRUMB_BASE);
const _: () = assert!(SL_EXPTAB + kernels::EXP_TABLE_BYTES <= CRUMB_OFF);
// DEC region between blob runs: the cross-attention key stages (two
// heads alternate) and value stage; the LM head's chunk double buffer
// (up to 49 blocks each) and its int16 rows for the 4-bit chunks.
const DE_XK0: usize = DEC_OFF;
const DE_XV: usize = DEC_OFF + 2 * STAGE_BYTES;
const _: () = assert!(DE_XV + STAGE_BYTES <= CRUMB_OFF);
const DE_CHUNK: usize = DEC_OFF;
const DE_ROWS: usize = DEC_OFF + 2 * 49 * storage::BLOCK;
const _: () = assert!(DE_ROWS + 2 * 64 * C <= CRUMB_OFF);
// Interlayer buffer during decode: attention scratch below IL_KEEP (what
// a blob may touch: checked against each blob's declared use), then the
// level expansion table and the 4-bit LM-head tables, persistent for the
// whole decode (nothing else touches the interlayer between the runs).
const IL_KEEP: usize = 16384;
const IL_LVL: usize = IL_KEEP;
const IL_LUT16: usize = IL_LVL + 2 * crate::q4::LVL_TABLE_LEN; // 49152
const _: () = assert!(IL_LUT16 + 4 * crate::q4::TABLES16 <= crate::INTERLAYER_BUFFER_BYTES);
const _: () = assert!(core::mem::size_of::<kernels::AttnScratch>() <= IL_KEEP);
const _: () = assert!(core::mem::size_of::<kernels::AttnScratch2>() <= IL_KEEP);

fn lut16_tables() -> &'static mut [u32; crate::q4::TABLES16] {
    let b = interlayer_at(IL_LUT16, 4 * crate::q4::TABLES16);
    unsafe { &mut *(b.as_mut_ptr() as *mut [u32; crate::q4::TABLES16]) }
}

fn lvl_table() -> &'static mut [u16; crate::q4::LVL_TABLE_LEN] {
    let b = interlayer_at(IL_LVL, 2 * crate::q4::LVL_TABLE_LEN);
    unsafe { &mut *(b.as_mut_ptr() as *mut [u16; crate::q4::LVL_TABLE_LEN]) }
}

fn land_u8(len: usize) -> &'static mut [u8] {
    assert!(len <= LAND_BYTES);
    unsafe { core::slice::from_raw_parts_mut(LAND_BASE as *mut u8, len) }
}

/// Packed decoder blobs flow through here (see the constants above).
struct Pipe {
    /// Entry whose packed bytes are complete in LAND.
    landed: Option<(Entry, usize)>,
    /// Entry whose read into LAND is in flight.
    pending: Option<(Entry, usize)>,
}

impl Pipe {
    const fn new() -> Pipe {
        Pipe { landed: None, pending: None }
    }

    /// Whether an entry fits the landing zone (packed); raw blobs do not.
    fn packed(e: Entry) -> bool {
        (e.len as usize) <= LAND_BYTES
    }

    /// Start reading `name` into LAND (nothing else may be pending).
    fn prefetch(&mut self, name: &str) -> Result<(), i32> {
        let (e, idx) = match lookup_idx(name) {
            Some(x) => x,
            None => {
                rprintln!("no blob {}", name);
                return Err(-904);
            }
        };
        if !Self::packed(e) || self.pending.is_some() {
            return Ok(());
        }
        let bytes = e.len.div_ceil(storage::BLOCK as u32) as usize * storage::BLOCK;
        io_start(true, e.lba, land_u8(bytes), "blob prefetch")?;
        self.pending = Some((e, idx));
        Ok(())
    }

    /// Wait for a pending prefetch, so blocking transfers may follow.
    fn settle(&mut self) -> Result<(), i32> {
        if let Some(p) = self.pending.take() {
            io_wait("blob prefetch")?;
            self.landed = Some(p);
        }
        Ok(())
    }

    /// Run blob `name` on arena offsets: land it if it is not yet (read,
    /// sum-verified with retries), expand it into DEC_BASE, start the
    /// read of `next`, then infer.
    fn run(&mut self, c: &mut Ctxt, name: &str, input: usize, output: usize,
           next: Option<&str>) -> i32 {
        let (e, idx) = match lookup_idx(name) {
            Some(x) => x,
            None => {
                rprintln!("no blob {}", name);
                return -904;
            }
        };
        // this entry's read in flight: expand it behind the DMA (any
        // other pending read is waited for first)
        let chase = self.pending.map(|(p, _)| p.lba) == Some(e.lba);
        if !chase {
            if let Err(rc) = self.settle() {
                return rc;
            }
        }
        if !Self::packed(e) {
            self.landed = None;
            return c.npu(name, input, output);
        }
        let blocks = e.len.div_ceil(storage::BLOCK as u32);
        let expect = unsafe { (*core::ptr::addr_of!(SUMS))[idx] };
        let mut ok = false;
        if chase {
            self.pending = None;
            self.landed = None;
            match chase_landing(e) {
                Ok(sum) if expect == 0 || sum == expect => ok = true,
                Ok(sum) => rprintln!("blob {} sum mismatch (got {:#x} want {:#x}, chased)",
                                     name, sum, expect),
                Err(rc) => return rc,
            }
            if ok {
                if let Err(rc) = verify_expanded(idx) {
                    return rc;
                }
            }
        }
        for attempt in 0..3 {
            if ok {
                break;
            }
            if self.landed.map(|(l, _)| l.lba) != Some(e.lba) {
                let rc = storage::read_blocks(e.lba, LAND_BASE as *mut u8, blocks);
                if rc != 0 {
                    return rc;
                }
                self.landed = Some((e, idx));
            }
            let t0 = cycles();
            let sum = dsp::byte_sum(land_u8(e.len as usize));
            prof_add(P_SUM, t0);
            if expect == 0 || sum == expect {
                let t0 = cycles();
                let rc = expand_landed(e);
                prof_add(P_UNPACK, t0);
                if let Err(rc) = rc {
                    return rc;
                }
                if let Err(rc) = verify_expanded(idx) {
                    return rc;
                }
                ok = true;
                break;
            }
            rprintln!("blob {} sum mismatch (got {:#x} want {:#x}, try {})",
                      name, sum, expect, attempt + 1);
            self.landed = None;
        }
        if !ok {
            return -905;
        }
        self.landed = None;
        if let Some(n) = next {
            if let Err(rc) = self.prefetch(n) {
                return rc;
            }
        }
        let _wd = crate::WdogGuard::arm();
        let t0 = cycles();
        let rc = unsafe {
            slot::run_at(DEC_BASE, arena_addr(input), arena_addr(output), name)
        };
        prof_add(P_NPU, t0);
        rc
    }
}

/// Check every expansion against the header's raw sum instead of once
/// per boot per entry (validation builds: proves the DMA chase never
/// reads a byte before it has landed; ~0.9 ms per blob).
const VERIFY_EVERY_EXPANSION: bool = false;

/// The validated "LAY5" header of the entry in LAND:
/// (raw_len, w_off, n_weights, raw_sum).
fn lay5_header(e: Entry) -> Result<(usize, usize, usize, u32), i32> {
    let p = LAND_BASE as *const u8;
    let word = |i: usize| -> usize { unsafe { *(p.add(i * 4) as *const u32) as usize } };
    if word(0) != crate::q4::MAGIC5 as usize {
        rprintln!("blob: not a LAY5 entry ({:#x})", word(0));
        return Err(-906);
    }
    let (raw_len, w_off, n) = (word(1), word(2), word(3));
    let raw_sum = word(4) as u32;
    let base = word(5);
    let n_groups = n / crate::q4::GL;
    if base != DEC_BASE
        || w_off + n != raw_len
        || DEC_OFF + raw_len > CRUMB_OFF
        || w_off % 4 != 0
        || n % (2 * crate::q4::GL) != 0
        || crate::q4::HDR5 + w_off + n_groups + n / 2 != e.len as usize
    {
        rprintln!("blob unpack: bad LAY5 header");
        return Err(-906);
    }
    Ok((raw_len, w_off, n, raw_sum))
}

/// Expand groups [from, to) of the entry in LAND into the raw blob at
/// DEC_BASE (the head has been copied).
fn expand_groups(w_off: usize, n_groups: usize, from: usize, to: usize) {
    let p = LAND_BASE as *const u8;
    let codes = crate::q4::HDR5 + w_off;
    let nibs = codes + n_groups;
    unsafe {
        crate::q4::expand_lvl(
            lvl_table(),
            p.add(codes + from),
            to - from,
            p.add(nibs + from * crate::q4::GL / 2),
            (DEC_BASE + w_off + from * crate::q4::GL) as *mut u8,
        );
    }
}

/// Expand the "LAY5" entry that is complete in LAND into the raw blob at
/// DEC_BASE: head verbatim, then the level-coded weights through the
/// table.
fn expand_landed(e: Entry) -> Result<(), i32> {
    let (_, w_off, n, _) = lay5_header(e)?;
    let n_groups = n / crate::q4::GL;
    unsafe {
        core::ptr::copy_nonoverlapping((LAND_BASE + crate::q4::HDR5) as *const u8,
                                       DEC_BASE as *mut u8, w_off);
    }
    expand_groups(w_off, n_groups, 0, n_groups);
    Ok(())
}

/// Expand the "LAY5" entry whose read into LAND is in flight, behind the
/// DMA: header, head and each group as soon as storage::landed() covers
/// it, and the packed byte sum the same way. Waits for the transfer at
/// the end and returns the sum (the caller compares it with the image's).
/// On the rig this hides the expansion (2.6 ms) and the sum (0.4 ms) of
/// an 88 KB blob under its own 4 ms read.
/// Chase statistics for the phase line: blobs chased, bytes landed at
/// the first poll, nibble bytes expanded before the transfer completed,
/// loop iterations. Tells how much of a blob's read the NPU run hid and
/// how much of the expansion the read hid.
static mut CHASE_STATS: [u64; 4] = [0; 4];

fn chase_landing(e: Entry) -> Result<u32, i32> {
    let len = e.len as usize;
    let mut first_got: Option<usize> = None;
    let mut pre_done_groups = 0usize;
    let mut iters = 0u64;
    let mut hdr: Option<(usize, usize)> = None; // (w_off, n_groups)
    let mut head_done = false;
    let mut groups = 0usize;
    let mut summed = 0usize;
    let mut sum = 0u32;
    loop {
        let done = storage::poll();
        // whole 32-byte units, for the word-wise byte sum
        let got = if done { len } else { storage::landed().min(len) & !31 };
        iters += 1;
        if first_got.is_none() {
            first_got = Some(got);
        }
        if hdr.is_none() && got >= crate::q4::HDR5 {
            match lay5_header(e) {
                Ok((_, w_off, n, _)) => hdr = Some((w_off, n / crate::q4::GL)),
                Err(rc) => {
                    let _ = storage::finish();
                    return Err(rc);
                }
            }
        }
        if let Some((w_off, n_groups)) = hdr {
            let nib_off = crate::q4::HDR5 + w_off + n_groups;
            if !head_done && got >= nib_off {
                unsafe {
                    core::ptr::copy_nonoverlapping((LAND_BASE + crate::q4::HDR5) as *const u8,
                                                   DEC_BASE as *mut u8, w_off);
                }
                head_done = true;
            }
            if head_done {
                let avail = (got.saturating_sub(nib_off) / (crate::q4::GL / 2)).min(n_groups);
                if avail > groups {
                    let t0 = cycles();
                    expand_groups(w_off, n_groups, groups, avail);
                    prof_add(P_UNPACK, t0);
                    groups = avail;
                }
            }
        }
        if got > summed {
            let t0 = cycles();
            sum = sum.wrapping_add(dsp::byte_sum(&land_u8(got)[summed..got]));
            prof_add(P_SUM, t0);
            summed = got;
        }
        if done {
            break;
        }
        pre_done_groups = groups;
    }
    let rc = storage::finish();
    if rc != 0 {
        rprintln!("FAIL blob prefetch rc={}", rc);
        return Err(rc);
    }
    unsafe {
        let st = &mut *core::ptr::addr_of_mut!(CHASE_STATS);
        st[0] += 1;
        st[1] += first_got.unwrap_or(0) as u64;
        st[2] += (pre_done_groups * crate::q4::GL / 2) as u64;
        st[3] += iters;
    }
    match hdr {
        Some((_, n_groups)) if head_done && groups == n_groups && summed == len => Ok(sum),
        _ => {
            rprintln!("blob chase: incomplete ({} of {} B)", summed, len);
            Err(-906)
        }
    }
}

/// Once per boot per entry (or always, VERIFY_EVERY_EXPANSION): the
/// expansion at DEC_BASE against the header's raw sum (packer/expander
/// drift, a byte read before it landed) and the blob's declared
/// interlayer use against IL_KEEP (the tables parked above it must
/// survive its run).
fn verify_expanded(idx: usize) -> Result<(), i32> {
    let seen = unsafe { (*core::ptr::addr_of!(Q4_VERIFIED))[idx >> 5] & (1 << (idx & 31)) != 0 };
    if seen && !VERIFY_EVERY_EXPANSION {
        return Ok(());
    }
    let p = LAND_BASE as *const u8;
    let word = |i: usize| -> usize { unsafe { *(p.add(i * 4) as *const u32) as usize } };
    let (raw_len, raw_sum) = (word(1), word(4) as u32);
    let raw = unsafe { core::slice::from_raw_parts(DEC_BASE as *const u8, raw_len) };
    let sum = dsp::byte_sum(raw);
    if sum != raw_sum {
        rprintln!("blob unpack sum mismatch (got {:#x} want {:#x})", sum, raw_sum);
        return Err(-907);
    }
    match unsafe { slot::interlayer_needed(DEC_BASE) } {
        Some(need) if need as usize <= IL_KEEP => {}
        other => {
            rprintln!("blob: interlayer use {:?} exceeds the decode limit {}", other, IL_KEEP);
            return Err(-909);
        }
    }
    unsafe { (*core::ptr::addr_of_mut!(Q4_VERIFIED))[idx >> 5] |= 1 << (idx & 31) };
    Ok(())
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
    let embc4 = lookup("embc4").ok_or(-901)?; // LM-head chunks (2026-09 image)
    let embc8 = if LM_ROWS_INT8 { lookup("embc8") } else { None };
    let ids = lookup("embpids").ok_or(-901)?;
    let posd = lookup("posdec").ok_or(-901)?;
    let vtb = lookup("vocabtb").ok_or(-901)?;
    let fin = lookup("final_gb.bin").ok_or(-901)?;
    // per-layer constant bundles (images from 2026-09-05 on); an older
    // image falls back to the individual entries
    let bundled = lookup("dc0").is_some();
    // the vocabulary table's string count, read once instead of per token
    let vtb_n = {
        let mut nb = [0u8; 4];
        try_rc!(sd_read_bytes(vtb, 0, &mut nb), "vtb n");
        u32::from_le_bytes(nb) as usize
    };

    unsafe {
        for m in (*core::ptr::addr_of_mut!(SELF_KV)).0.iter_mut() {
            m.fill(0);
        }
    }
    if arena_addr(0) as usize != ARENA_BASE {
        rprintln!("FAIL arena not at {:#x}", ARENA_BASE);
        return Err(-910);
    }
    // the slot is scratch and pipeline space from here on
    c.loaded = Entry::default();
    crate::q4::table_lvl(lvl_table());
    if embc8.is_none() {
        crate::q4::tables16(lut16_tables());
    }
    let mut pipe = Pipe::new();
    let mut n_tok = 0usize; // cache length
    let mut token = plan.sot[0];
    let mut next_sot = 1usize;
    let mut printed = 0usize;
    // Accumulate the decoded token pieces; emitted on one "Detected:" line.
    let mut transcript = [0u8; 256];
    let mut tlen = 0usize;

    for step in 0..(plan.n_sot - 1 + MAX_TOKENS) {
        // the first blob's prefetch (from the previous step) must land
        // before these blocking reads
        pipe.settle()?;
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
            pipe.settle()?;
            if bundled {
                c.asset(Name::of(&["dc", DIGITS[l]]).s(), D_DC)?;
            } else {
                c.asset(Name::of(&["b", DIGITS[l], "_ln1_gb.bin"]).s(), D_DC + DC_LN1)?;
                c.asset(Name::of(&["b", DIGITS[l], "_xln_gb.bin"]).s(), D_DC + DC_XLN)?;
                c.asset(Name::of(&["b", DIGITS[l], "_ln2_gb.bin"]).s(), D_DC + DC_LN2)?;
                for j in 0..4usize {
                    c.asset(Name::of(&["b", DIGITS[l], "_lut", DIGITS[j], ".bin"]).s(),
                            D_DC + DC_LUT + j * 256)?;
                }
            }
            dec_ln(D_X16, sq, D_DC + DC_LN1, D_LN, bq.ln1);
            let (nk, nv, nout) = (dec_blob(l, "k", 0), dec_blob(l, "v", 0), dec_blob(l, "out", 0));
            let (nxq, nxout) = (dec_blob(l, "xq", 0), dec_blob(l, "xout", 0));
            try_rc!(pipe.run(c, dec_blob(l, "q", 0).s(), D_LN, D_Q, Some(nk.s())), "dq");
            try_rc!(pipe.run(c, nk.s(), D_LN, D_K, Some(nv.s())), "dk");
            try_rc!(pipe.run(c, nv.s(), D_LN, D_V, Some(nout.s())), "dv");
            // append column n_tok to the cache (planar stride MAX_TOKENS)
            unsafe {
                let kv = &mut (*core::ptr::addr_of_mut!(SELF_KV)).0;
                for ci in 0..C {
                    kv[l * 2][ci * MAX_TOKENS + n_tok] = as_i8(D_K, C * W4)[ci * W4];
                    kv[l * 2 + 1][ci * MAX_TOKENS + n_tok] =
                        as_i8(D_V, C * W4)[ci * W4];
                }
            }
            let t = n_tok + 1;
            let smq = bq.q_out.scale * bq.k_out.scale / 8.0;
            unsafe {
                let kv = &(*core::ptr::addr_of!(SELF_KV)).0;
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
            try_rc!(pipe.run(c, nout.s(), D_CTX, D_O, Some(nxq.s())), "dout");
            dec_add(D_X16, sq, D_O, bq.out_out, bq.res1);
            sq = bq.res1;

            // cross-attention
            dec_ln(D_X16, sq, D_DC + DC_XLN, D_LN, bq.xln);
            try_rc!(pipe.run(c, nxq.s(), D_LN, D_Q, Some(nxout.s())), "dxq");
            // the stage reads below need the channel: let dxout land first
            pipe.settle()?;
            let smx = bq.xq_out.scale * bq.xk_out.scale / 8.0;
            // each head's K and V tile blocks: one read each, K (key-major
            // as cross_kv wrote it) and V staged in the DEC region, which
            // is idle between dxq and dxout; the kernels consume both int8
            // layouts directly. Head h's V read runs under head h-1's
            // compute, head h+1's K read under head h's.
            let stage = c.tiles * HB;
            c.loaded = Entry::default();
            let kreg = S_XKV + (l as u32 * 2) * HREG_BLOCKS;
            let vreg = S_XKV + (l as u32 * 2 + 1) * HREG_BLOCKS;
            let kbuf = |h: usize| slot_u8(DE_XK0 + (h % 2) * STAGE_BYTES, stage);
            io_start(true, c.lba(kreg, 0), kbuf(0), "xk rd")?;
            io_wait("xk rd")?;
            io_start(true, c.lba(vreg, 0), slot_u8(DE_XV, stage), "xv rd")?;
            for h in 0..HEADS {
                io_wait("xv rd")?;
                if h + 1 < HEADS {
                    io_start(true, c.lba(kreg, (h + 1) * N_TILES * HB), kbuf(h + 1), "xk rd")?;
                }
                let t0 = cycles();
                kernels::attn_head_x1_i8(
                    &as_i8(D_Q, C * W4)[h * HD * W4..(h + 1) * HD * W4],
                    slot_i8(DE_XK0 + (h % 2) * STAGE_BYTES, stage),
                    slot_i8(DE_XV, stage), c.ctx,
                    &mut as_i8_mut(D_CTX, C * W4)[h * HD * W4..(h + 1) * HD * W4],
                    W4, bq.xq_out.zp, bq.xk_out.zp, bq.xv_out.zp,
                    smx, bq.xv_out.scale, bq.xctx, attn_scratch2(),
                    &kernels::ExpFn(softmax_exp),
                );
                prof_add(P_ATTN, t0);
                if h + 1 < HEADS {
                    io_wait("xk rd")?;
                    io_start(true, c.lba(vreg, (h + 1) * N_TILES * HB),
                             slot_u8(DE_XV, stage), "xv rd")?;
                }
            }
            let nf = dec_blob(l, "fc1", 0);
            try_rc!(pipe.run(c, nxout.s(), D_CTX, D_O, Some(nf.s())), "dxout");
            dec_add(D_X16, sq, D_O, bq.xout_out, bq.res2);
            sq = bq.res2;

            // mlp
            dec_ln(D_X16, sq, D_DC + DC_LN2, D_LN, bq.ln2);
            for j in 0..4usize {
                let n2 = dec_blob(l, "fc2p", j);
                try_rc!(pipe.run(c, dec_blob(l, "fc1", j).s(), D_LN, D_Q, Some(n2.s())),
                        "dfc1");
                lut_apply(D_DC + DC_LUT + j * 256, D_Q, C * W4);
                // after the last partial: the next layer's q, or the next
                // step's first blob
                let after = if j < 3 {
                    dec_blob(l, "fc1", j + 1)
                } else if l + 1 < BLOCKS {
                    dec_blob(l + 1, "q", 0)
                } else {
                    dec_blob(0, "q", 0)
                };
                try_rc!(pipe.run(c, n2.s(), D_Q, D_P + j * C * W4, Some(after.s())),
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

        // LM head on the CPU: f32 layernorm + pruned-vocab argmax (its
        // reads need the channel; the chunk buffers sit in DEC)
        pipe.settle()?;
        c.loaded = Entry::default();
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
        let (best, best_pos) = lm_head(plan, embc4, embc8, &hid, out_idx == 0)?;
        rprintln!("tok id {}", best);
        if best == plan.eot {
            print_detected(&transcript, tlen);
            rprintln!("=== done ({} tokens) ===", printed);
            return Ok(());
        }
        append_token(vtb, vtb_n, best_pos, &mut transcript, &mut tlen)?;
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

/// Layernorm of the token-rate residual with gamma/beta at arena offset
/// `gb` (f32[C] each).
fn dec_ln(src: usize, sq: Quant, gb: usize, dst: usize, dq: Quant) {
    kernels::ln_planar_i16_to_i8(
        as_i16(src, C * W4), sq, as_f32(gb, C), as_f32(gb + 4 * C, C),
        as_i8_mut(dst, C * W4), dq, C, W4,
    );
}

fn dec_add(x16: usize, qa: Quant, b8: usize, qb: Quant, qd: Quant) {
    let a = as_i16(x16, C * W4);
    let dst = unsafe {
        core::slice::from_raw_parts_mut(arena_addr(x16) as *mut i16, C * W4)
    };
    kernels::add_i16_i8(a, qa, as_i8(b8, C * W4), qb, dst, qd);
}

/// Prefer the int8 LM-head rows ("embc8") when the image carries them:
/// no nibble unpack, at twice the read (4.7 MB per token). Pays on a
/// stick reading 20 MB/s or more; below that the 4-bit chunks win.
const LM_ROWS_INT8: bool = true;

/// Argmax over the pruned embedding, streamed in 64-row chunks.
///
/// Each chunk is one read: the rows' f32 scales and token ids, then the
/// weights. "embc4" carries them 4-bit (group amax + nibbles), expanded
/// to int16 and dotted on SMLAD; "embc8" carries the int8 rows, widened
/// in registers by the SXTB16 kernel (dsp::dot384_2rows_i8, which takes
/// the hidden vector with the middle two of every four columns swapped).
/// The hidden vector is quantized to a per-utterance int16 grid (gate:
/// model/lm16_check.py, zero argmax flips on the golden decode for both
/// row formats). Pad rows carry scale 0 and input-only rows (SOT etc)
/// scale -1: both skipped.
///
/// (An exact bound-sorted early exit was measured and rejected: Whisper
/// LM-head cosines are so small that even the loosest row's Cauchy-Schwarz
/// bound sits ~3x above the best logit -- 0 of 12228 rows prunable.)
fn lm_head(plan: &Plan, embc4: Entry, embc8: Option<Entry>, hid: &[f32; C],
           first: bool) -> Result<(u32, usize), i32> {
    const ROWS: usize = 64;
    const CH_SCL: usize = 0; // f32[64]
    const CH_IDS: usize = ROWS * 4; // u32[64]
    // 4-bit chunk: group amax bytes, then the nibbles
    const CH_AMAX: usize = 2 * ROWS * 4; // u8[64 * 6]
    const CH_NIBS: usize = CH_AMAX + ROWS * C / crate::q4::G;
    const CH4_USED: usize = CH_NIBS + ROWS * C / 2;
    const CH4_BLOCKS: usize = CH4_USED.div_ceil(storage::BLOCK);
    const _: () = assert!(CH4_BLOCKS == crate::q4::EMB_CHUNK_BLOCKS);
    // int8 chunk: the rows themselves
    const CH_ROWS: usize = 2 * ROWS * 4; // i8[64][384]
    const CH8_BLOCKS: usize = (CH_ROWS + ROWS * C) / storage::BLOCK;
    const _: () = assert!((CH_ROWS + ROWS * C) % storage::BLOCK == 0 && CH8_BLOCKS == 49);
    // chunks alternate between two buffers in the idle DEC region: chunk
    // i+1 streams in while chunk i is unpacked and dotted
    const _: () = assert!(DE_CHUNK + 2 * CH8_BLOCKS * storage::BLOCK <= DE_ROWS);

    let (embc, ch_blocks) = match embc8 {
        Some(e) => (e, CH8_BLOCKS),
        None => (embc4, CH4_BLOCKS),
    };
    let int8 = embc8.is_some();
    let ch_bytes = ch_blocks * storage::BLOCK;

    let mut hmax = 0f32;
    for &h in hid.iter() {
        hmax = hmax.max(libm::fabsf(h));
    }
    let hs = if hmax > 0.0 { hmax / 32767.0 } else { 1.0 };
    // word-aligned for the kernels' paired loads
    #[repr(C, align(4))]
    struct Hq([i16; C]);
    let mut hq = Hq([0i16; C]);
    for (i, &h) in hid.iter().enumerate() {
        let at = if int8 { dsp::perm4(i) } else { i };
        hq.0[at] = dsp::round_i32(h / hs).clamp(-32767, 32767) as i16;
    }

    let mut best = f32::MIN;
    let mut best2 = f32::MIN;
    let mut best_id = plan.eot;
    let mut best_pos = 0usize; // its kept position (= row index)
    let rows16 = slot_i16(DE_ROWS, ROWS * C);
    let n_chunks = plan.vocab_n.div_ceil(ROWS);
    io_start(true, embc.lba, slot_u8(DE_CHUNK, ch_bytes), "embc rd")?;
    for chunk in 0..n_chunks {
        io_wait("embc rd")?;
        if chunk + 1 < n_chunks {
            io_start(true, embc.lba + ((chunk + 1) * ch_blocks) as u32,
                     slot_u8(DE_CHUNK + ((chunk + 1) % 2) * ch_bytes, ch_bytes), "embc rd")?;
        }
        let il = slot_u8(DE_CHUNK + (chunk % 2) * ch_bytes, ch_bytes);
        if !int8 {
            let t0 = cycles();
            crate::q4::unpack16(lut16_tables(), &il[CH_AMAX..CH_NIBS], &il[CH_NIBS..CH4_USED],
                                rows16);
            prof_add(P_UNPACK, t0);
        }
        let t0 = cycles();
        let n = ROWS.min(plan.vocab_n - chunk * ROWS);
        let mut r = 0;
        while r < ROWS {
            let d = unsafe {
                if int8 {
                    dsp::dot384_2rows_i8(il.as_ptr().add(CH_ROWS + r * C) as *const i8,
                                         hq.0.as_ptr())
                } else {
                    dsp::dot384_2rows(rows16.as_ptr().add(r * C), hq.0.as_ptr())
                }
            };
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
                    best_pos = chunk * ROWS + row;
                } else if logit > best2 {
                    best2 = logit;
                }
            }
            r += 2;
        }
        prof_add(P_LM, t0);
    }
    rprintln!("lm: id {} logit {:.3} (2nd {:.3})", best_id, best, best2);
    Ok((best_id, best_pos))
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

/// Append a kept token's text piece from the vocabulary table (`n_off`
/// strings) to the running transcript buffer (`buf[..len]`), truncating
/// if it fills.
fn append_token(vtb: Entry, n_off: usize, kept_pos: usize, buf: &mut [u8; 256],
                len: &mut usize) -> Result<(), i32> {
    let mut offs = [0u8; 8];
    try_rc!(sd_read_bytes(vtb, 4 + kept_pos * 4, &mut offs), "vtb off");
    let o0 = u32::from_le_bytes(offs[0..4].try_into().unwrap()) as usize;
    let o1 = u32::from_le_bytes(offs[4..8].try_into().unwrap()) as usize;
    let n = (o1 - o0).min(48);
    let mut sbuf = [0u8; 48];
    // strings start after the offset table: 4 + (n+1)*4 bytes in
    let base = 4 + (n_off + 1) * 4;
    try_rc!(sd_read_bytes(vtb, base + o0, &mut sbuf[..n]), "vtb s");
    let take = n.min(buf.len() - *len);
    buf[*len..*len + take].copy_from_slice(&sbuf[..take]);
    *len += take;
    Ok(())
}

/// Emit the decoded transcript on its own line, prefixed "Detected: ",
/// to both the RTT log and the OLED (leading BPE space trimmed), then the
/// time from the end of speech to this line.
fn print_detected(buf: &[u8; 256], len: usize) {
    if let Ok(s) = core::str::from_utf8(&buf[..len]) {
        let s = s.trim_start();
        rprintln!("");
        rprintln!("Detected: {}", s);
        display::print("\nDetected: ");
        display::print(s);
        display::print("\n");
    }
    let (end, live, tail) = unsafe { (CAPTURE_END_MS, CAPTURE_LIVE, SPEECH_TAIL_MS) };
    let processing = crate::uptime_ms().wrapping_sub(end);
    if live {
        // the mic kept recording for `tail` after the last speech frame
        rprintln!(
            "latency: {} ms from end of speech (recording tail {} ms + processing {} ms)",
            tail + processing, tail, processing
        );
    } else {
        rprintln!("latency: {} ms of processing from end of clip (injected audio)",
                  processing);
    }
    oled_line(format_args!("({} ms)\n", if live { tail + processing } else { processing }));
}

/// Formatted text to the OLED without an allocator (48-byte line).
fn oled_line(args: core::fmt::Arguments) {
    struct Buf {
        b: [u8; 48],
        n: usize,
    }
    impl core::fmt::Write for Buf {
        fn write_str(&mut self, s: &str) -> core::fmt::Result {
            let take = s.len().min(self.b.len() - self.n);
            self.b[self.n..self.n + take].copy_from_slice(&s.as_bytes()[..take]);
            self.n += take;
            Ok(())
        }
    }
    let mut buf = Buf { b: [0; 48], n: 0 };
    let _ = core::fmt::write(&mut buf, args);
    if let Ok(s) = core::str::from_utf8(&buf.b[..buf.n]) {
        display::print(s);
    }
}
