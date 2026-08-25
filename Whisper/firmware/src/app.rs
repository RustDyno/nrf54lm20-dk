//! Standalone Whisper: record from the PDM mic, transcribe, print over RTT.
//! No host in the data path -- weights and scratch live on the SD card.
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
use crate::{display, mel, pdm, sd, slot};
use rtt_target::{rprint, rprintln};

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
const HREG_BLOCKS: u32 = (HEADS * N_TILES * HB / sd::BLOCK) as u32; // 480

// --- SD scratch regions (block offsets from plan.scratch_lba) ----------------
const S_PCM: u32 = 0; // 750 blocks
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
// Attention phase overlay (no A_IN/A_OUT/A_AUX use):
const A_K: usize = 0; // 40960 planar [64, 640]
const A_V: usize = 40960; // 40960
const A_Q: usize = 81920; // 4096
const A_TMP: usize = 86016; // 4096 block scratch / ctx out

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

/// The interlayer buffer is idle outside NPU runs; phases borrow it as
/// scratch (mel filterbank, embedding scale table).
fn interlayer(len: usize) -> &'static mut [u8] {
    unsafe {
        core::slice::from_raw_parts_mut(
            core::ptr::addr_of_mut!(crate::nrf_axon_interlayer_buffer) as *mut u8,
            len,
        )
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
    let idx = unsafe { &*core::ptr::addr_of!(INDEX) };
    let count = u32::from_le_bytes(idx[12..16].try_into().unwrap()) as usize;
    for i in 0..count.min(254) {
        let e = &idx[16 + 32 * i..16 + 32 * (i + 1)];
        let end = e[..24].iter().position(|&b| b == 0).unwrap_or(24);
        if &e[..end] == name.as_bytes() {
            return Some(Entry {
                lba: u32::from_le_bytes(e[24..28].try_into().unwrap()),
                len: u32::from_le_bytes(e[28..32].try_into().unwrap()),
            });
        }
    }
    None
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
        let rc = sd::read_blocks(e.lba, buf.as_mut_ptr(),
                                 e.len.div_ceil(sd::BLOCK as u32));
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
        sd::read_blocks(
            self.scratch + region + (byte_off / sd::BLOCK) as u32,
            arena(off, 0).as_mut_ptr(),
            (len / sd::BLOCK) as u32,
        )
    }

    fn write(&self, region: u32, byte_off: usize, off: usize, len: usize) -> i32 {
        sd::write_blocks(
            self.scratch + region + (byte_off / sd::BLOCK) as u32,
            arena(off, 0).as_ptr(),
            (len / sd::BLOCK) as u32,
        )
    }

    /// Load a whole named asset to an arena offset.
    fn asset(&self, name: &str, off: usize) -> Result<Entry, i32> {
        let e = lookup(name).ok_or(-901)?;
        let rc = sd::read_blocks(e.lba, arena(off, 0).as_mut_ptr(),
                                 e.len.div_ceil(sd::BLOCK as u32));
        if rc != 0 {
            return Err(rc);
        }
        Ok(e)
    }

    /// Blob into the slot (cached) + one NPU inference.
    fn npu(&mut self, blob: &str, input: usize, output: usize) -> i32 {
        let e = match lookup(blob) {
            Some(e) => e,
            None => {
                rprintln!("no blob {}", blob);
                return -904;
            }
        };
        if self.loaded.lba != e.lba {
            let rc = sd::read_blocks(e.lba, slot::SLOT_BASE as *mut u8,
                                     e.len.div_ceil(sd::BLOCK as u32));
            if rc != 0 {
                return rc;
            }
            self.loaded = e;
        }
        let _wd = crate::WdogGuard::arm();
        unsafe { slot::run(arena_addr(input), arena_addr(output)) }
    }
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
    rprintln!("standalone: SD init");
    let rc = sd::init();
    if rc != 0 {
        rprintln!("standalone: no SD ({}), staying in mailbox mode", rc);
        display::print("no SD card\n");
        sd::diag(2);
        sd::release_pins();
        rprintln!("sd pins released (high-Z): external testers may drive the bus");
        crate::mailbox_loop();
    }
    unsafe {
        let rc = sd::read_blocks(0, core::ptr::addr_of_mut!(INDEX) as *mut u8, 16);
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

fn utterance(plan: &Plan, c: &mut Ctxt) -> Result<(), i32> {
    rprintln!("");
    rprintln!("=== speak now (12 s) ===");
    display::clear();
    display::print("== speak now (12 s)\n");
    if unsafe { STREAM_MEL_OK } {
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
    let (tiles, actx) = mel_pass2(plan, c)?;
    c.tiles = tiles;
    c.ctx = actx;
    display::print("encoding...\n");
    rprintln!("encoder...");
    encoder(plan, c)?;
    rprintln!("cross K/V...");
    cross_kv(plan, c)?;
    rprintln!("decoding...");
    display::print("decoding:\n");
    decode(plan, c)
}

// --- record + mel pass 1 --------------------------------------------------------
//
// Streaming layout: the PDM ping-pong buffers are one mel chunk each
// (10240 samples = 0.64 s), so while the DMA fills one buffer the CPU has a
// whole chunk period to window the previous one, run mel_frames, and spill
// the f32 chunk to S_MELF. Chunk c's DFT window needs samples
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

fn mel_tables(c: &Ctxt) -> Result<(), i32> {
    // filterbank borrows the interlayer buffer; small tables in the arena
    let filt = lookup("melfilt").ok_or(-901)?;
    try_rc!(sd::read_blocks(filt.lba, interlayer(0).as_mut_ptr(), 126), "filt");
    c.asset("hann", R_HANN)?;
    c.asset("melcos", R_COS)?;
    arena(R_MAX, 4).copy_from_slice(&(-1e30f32).to_le_bytes());
    Ok(())
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
fn record_mel(c: &Ctxt) -> Result<u32, i32> {
    mel_tables(c)?;

    let b0 = as_i16_mut(R_PDM0, CHUNK);
    let b1 = as_i16_mut(R_PDM1, CHUNK);
    let mut stream =
        unsafe { pdm::Pdm::init(crate::MIC_CLK, crate::MIC_DIN).start(b0, b1) };
    stream.next_buffer(); // warmup chunk (startup overrun + mic DC settle)
    stream.overruns = 0;

    let mut have = 0usize; // valid samples in the sliding window
    for k in 0..N_CHUNKS {
        let hop = stream.next_buffer(); // buffer k; DMA now fills the other
        let win = as_i16_mut(R_WIN, WIN);
        if k == 0 {
            win[..CHUNK].copy_from_slice(hop);
            have = CHUNK;
            continue;
        }
        // chunk k-1: append buffer k's first 256 samples (right halo)
        win[have..have + 256].copy_from_slice(&hop[..256]);
        mel_chunk(k - 1, have + 256, T);
        try_rc!(c.write(S_MELF, (k - 1) * 20480, R_MELF, 20480), "mel spill");
        // slide: 1280-sample left halo, then all of buffer k
        win.copy_within(have - 1280..have, 0);
        win[1280..1280 + CHUNK].copy_from_slice(hop);
        have = 1280 + CHUNK;
    }
    let ov = stream.overruns;
    stream.stop();
    // final chunk (48 frames) needs no further input from the mic: the
    // window already covers [18*CHUNK - 1280, N_SAMPLES)
    mel_chunk(N_CHUNKS - 1, N_SAMPLES - ((N_CHUNKS - 1) * CHUNK - 1280), 48);
    try_rc!(c.write(S_MELF, (N_CHUNKS - 1) * 20480, R_MELF,
                    (80 * 48 * 4usize).div_ceil(sd::BLOCK) * sd::BLOCK),
            "mel spill");
    Ok(ov)
}

/// Sequential fallback: plain recording to SD (used when streaming mel
/// once fell behind; pass 1 then reads the PCM back from the card).
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
fn mel_pass1(c: &Ctxt) -> Result<(), i32> {
    for ci in 0..N_CHUNKS {
        let f0 = ci * T;
        let n_frames = if ci == 18 { 48 } else { T };
        let s0 = (f0 * 160).saturating_sub(1280);
        let span = ((f0 + n_frames) * 160 + 256).min(N_SAMPLES) - s0;
        let bytes = (span * 2).div_ceil(sd::BLOCK) * sd::BLOCK;
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

/// Pass 2: normalize into int8 mel tiles; pad frames = quantized 0.0.
/// Needs the global max in R_MAX from either pass 1. Also scans the f32
/// mel energies for speech (mean log-mel per frame vs the quietest
/// frame) and returns the VAD extent (tiles, ctx) for the encoder.
fn mel_pass2(plan: &Plan, c: &Ctxt) -> Result<(usize, usize), i32> {
    let pad = quant8(0.0, plan.conv1_in);
    // per-mel-frame mean energies staged in the (idle) R_WIN region
    let means = unsafe {
        core::slice::from_raw_parts_mut(arena_addr(R_WIN) as *mut f32, 1200)
    };
    for mt in 0..MEL_TILES {
        let dst = as_i8_mut(R_TILE, 80 * T);
        if mt < 18 {
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
            };
            unsafe { mel::mel_normalize(&p) };
        } else if mt == 18 {
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
    // Endpoint: last mel frame louder than (quietest frame + 10 dB),
    // plus margin, floored and capped, rounded up to whole tiles.
    let mut floor = f32::MAX;
    for &v in means[..1200].iter() {
        if v < floor {
            floor = v;
        }
    }
    let mut last = 0usize;
    for (f, &v) in means[..1200].iter().enumerate() {
        if v > floor + VAD_THRESH_LOG10 {
            last = f;
        }
    }
    let ctx = (last / 2 + VAD_MARGIN_CTX).clamp(VAD_FLOOR_CTX, CTX);
    let tiles = ctx.div_ceil(T);
    rprintln!(
        "vad: speech to mel frame {} -> ctx {} ({} tiles of {})",
        last, ctx, tiles, N_TILES
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

/// Reassemble per-head [64,64] blocks from `region` into a planar
/// [64, out_w] buffer at `dst_off`, dropping columns >= out_w.
fn assemble_head(c: &Ctxt, region: u32, head: usize, dst_off: usize,
                 out_w: usize) -> Result<(), i32> {
    for i in 0..c.tiles {
        try_rc!(c.read(region, (head * N_TILES + i) * HB, A_TMP, HB), "hb rd");
        let tmp = as_i8(A_TMP, HB);
        let dst = as_i8_mut(dst_off, HD * out_w);
        let c0 = i * T;
        let take = T.min(out_w.saturating_sub(c0));
        for r in 0..HD {
            for col in 0..take {
                dst[r * out_w + c0 + col] = tmp[r * T + col];
            }
        }
    }
    Ok(())
}

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
        try_rc!(sd::read_blocks(pos.lba + (ci * CH * 4 / sd::BLOCK) as u32,
                                arena(A_IN, 0).as_mut_ptr(),
                                (CH * 4 / sd::BLOCK) as u32), "pos b");
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
        // attention
        crate::crumb(0x524 + (l as u32) * 0x10);
        let sm = bq.q_out.scale * bq.k_out.scale / 8.0;
        let aw = c.tiles * T;
        for h in 0..HEADS {
            assemble_head(c, S_KH, h, A_K, aw)?;
            assemble_head(c, S_VH, h, A_V, aw)?;
            for i in 0..c.tiles {
                try_rc!(c.read(S_QH, (h * N_TILES + i) * HB, A_Q, HB), "q rd");
                kernels::attn_head(
                    as_i8(A_Q, HB), as_i8(A_K, HD * aw), as_i8(A_V, HD * aw),
                    as_i8_mut(A_TMP, HB), HD, T, T, c.ctx, aw,
                    bq.q_out.zp, bq.k_out.zp, bq.v_out.zp,
                    sm, bq.v_out.scale, bq.ctx,
                );
                try_rc!(c.write(S_CH, (h * N_TILES + i) * HB, A_TMP, HB), "ctx wr");
            }
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
// Cross K/V planar [64, 600] pair: 25088 + 2 x 38400 = 101888, inside the
// arena's usable span (the top 8 bytes are the crash breadcrumb). Head
// blocks stage through D_P, which is idle during cross-attention.
const D_BIGK: usize = 25088;
const D_BIGV: usize = 25088 + HD * CTX;

fn sd_read_bytes(e: Entry, byte_off: usize, dst: &mut [u8]) -> i32 {
    // unaligned helper via a bounce block (small reads only)
    let mut bounce = [0u8; 1024];
    let lba = e.lba + (byte_off / sd::BLOCK) as u32;
    let skew = byte_off % sd::BLOCK;
    let blocks = (skew + dst.len()).div_ceil(sd::BLOCK);
    debug_assert!(blocks <= 2);
    let rc = sd::read_blocks(lba, bounce.as_mut_ptr(), blocks as u32);
    if rc != 0 {
        return rc;
    }
    dst.copy_from_slice(&bounce[skew..skew + dst.len()]);
    0
}

fn decode(plan: &Plan, c: &mut Ctxt) -> Result<(), i32> {
    let embf = lookup("embf").ok_or(-901)?;
    let embp = lookup("embp").ok_or(-901)?;
    let scl = lookup("embpscl").ok_or(-901)?;
    let ids = lookup("embpids").ok_or(-901)?;
    let posd = lookup("posdec").ok_or(-901)?;
    let vtb = lookup("vocabtb").ok_or(-901)?;
    let fin = lookup("final_gb.bin").ok_or(-901)?;
    // row scales live in the interlayer buffer for the whole decode
    try_rc!(sd::read_blocks(scl.lba, interlayer(0).as_mut_ptr(),
                            scl.len.div_ceil(sd::BLOCK as u32)), "scl");

    unsafe {
        for m in (*core::ptr::addr_of_mut!(SELF_KV)).iter_mut() {
            m.fill(0);
        }
    }
    let mut n_tok = 0usize; // cache length
    let mut token = plan.sot[0];
    let mut next_sot = 1usize;
    let mut printed = 0usize;

    for step in 0..(plan.n_sot - 1 + MAX_TOKENS) {
        // x16 = quantize(embf[kept_pos(token)] + posdec[step])
        let pos_kept = kept_position(ids, token)?;
        let mut row = [0f32; C];
        let mut buf = [0u8; C * 4];
        // embf rows are 1536 B = 3 blocks, block-aligned by construction
        try_rc!(sd::read_blocks(embf.lba + (pos_kept * 3) as u32,
                                buf.as_mut_ptr(), 3), "embf");
        for (i, r) in row.iter_mut().enumerate() {
            *r = f32::from_le_bytes(buf[i * 4..i * 4 + 4].try_into().unwrap());
        }
        try_rc!(sd::read_blocks(posd.lba + (step * 3) as u32,
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
                        smq, bq.v_out.scale, bq.ctx,
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
            for h in 0..HEADS {
                assemble_head_ctx(c, S_XKV, l * 2, h, D_BIGK)?;
                assemble_head_ctx(c, S_XKV, l * 2 + 1, h, D_BIGV)?;
                kernels::attn_head(
                    &as_i8(D_Q, C * W4)[h * HD * W4..(h + 1) * HD * W4],
                    as_i8(D_BIGK, HD * c.ctx), as_i8(D_BIGV, HD * c.ctx),
                    &mut as_i8_mut(D_CTX, C * W4)[h * HD * W4..(h + 1) * HD * W4],
                    HD, 1, W4, c.ctx, c.ctx,
                    bq.xq_out.zp, bq.xk_out.zp, bq.xv_out.zp,
                    smx, bq.xv_out.scale, bq.xctx,
                );
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
        try_rc!(sd_read_bytes(fin, 0, &mut gb), "fin gb");
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
        let best = lm_head(plan, embp, ids, &hid, out_idx == 0)?;
        if best == plan.eot {
            rprintln!("");
            rprintln!("=== done ({} tokens) ===", printed);
            display::print("\n== done\n");
            return Ok(());
        }
        print_token(vtb, kept_position(ids, best)?)?;
        printed += 1;
        token = best;
        if n_tok >= MAX_TOKENS {
            rprintln!("");
            rprintln!("=== token budget reached ===");
            display::print("\n== token budget\n");
            return Ok(());
        }
    }
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

/// Cross K/V head reassembly into planar [64, CTX] (pad columns dropped).
fn assemble_head_ctx(c: &Ctxt, base: u32, matrix: usize, head: usize,
                     dst_off: usize) -> Result<(), i32> {
    let w = c.ctx;
    for i in 0..c.tiles {
        try_rc!(c.read(base + matrix as u32 * HREG_BLOCKS,
                       (head * N_TILES + i) * HB, D_P, HB),
                "xhb");
        let tmp = as_i8(D_P, HB);
        let dst = as_i8_mut(dst_off, HD * w);
        let c0 = i * T;
        let take = T.min(w.saturating_sub(c0));
        for r in 0..HD {
            for col in 0..take {
                dst[r * w + c0 + col] = tmp[r * T + col];
            }
        }
    }
    Ok(())
}

/// Argmax over the pruned int8 embedding, streamed in 64-row chunks.
///
/// (An exact bound-sorted early exit was measured and rejected: Whisper
/// LM-head cosines are so small that even the loosest row's Cauchy-Schwarz
/// bound sits ~3x above the best logit -- 0 of 12228 rows prunable.)
fn lm_head(plan: &Plan, embp: Entry, ids: Entry, hid: &[f32; C],
           first: bool) -> Result<u32, i32> {
    let mut best = f32::MIN;
    let mut best_row = 0usize;
    const ROWS: usize = 64; // 64 x 384 = 24576 B per chunk
    let scl_all = interlayer(0); // f32 row scales, loaded at decode start
    let mut row_id = [0u8; 4];
    for chunk in 0..plan.vocab_n.div_ceil(ROWS) {
        let r0 = chunk * ROWS;
        let n = ROWS.min(plan.vocab_n - r0);
        let rc = sd::read_blocks(embp.lba + (r0 * C / sd::BLOCK) as u32,
                                 arena(D_BIGK, 0).as_mut_ptr(),
                                 (n * C).div_ceil(sd::BLOCK) as u32);
        if rc != 0 {
            return Err(rc);
        }
        let rows = as_i8(D_BIGK, n * C);
        for r in 0..n {
            let s = f32::from_le_bytes(
                scl_all[(r0 + r) * 4..(r0 + r) * 4 + 4].try_into().unwrap());
            if s <= 0.0 {
                continue; // input-only row (SOT etc), marked by the image
            }
            let mut acc = 0f32;
            for i in 0..C {
                acc += rows[r * C + i] as f32 * hid[i];
            }
            let logit = acc * s;
            if logit > best {
                if first {
                    // blank suppression on the first sampled token
                    sd_read_bytes(ids, (r0 + r) * 4, &mut row_id);
                    let id = u32::from_le_bytes(row_id);
                    if id == plan.eot
                        || plan.blank[..plan.n_blank].contains(&id)
                    {
                        continue;
                    }
                }
                best = logit;
                best_row = r0 + r;
            }
        }
    }
    let mut b = [0u8; 4];
    try_rc!(sd_read_bytes(ids, best_row * 4, &mut b), "ids");
    Ok(u32::from_le_bytes(b))
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

/// Print a kept token's text piece from the vocabulary table.
fn print_token(vtb: Entry, kept_pos: usize) -> Result<(), i32> {
    let mut offs = [0u8; 8];
    try_rc!(sd_read_bytes(vtb, 4 + kept_pos * 4, &mut offs), "vtb off");
    let o0 = u32::from_le_bytes(offs[0..4].try_into().unwrap()) as usize;
    let o1 = u32::from_le_bytes(offs[4..8].try_into().unwrap()) as usize;
    let len = (o1 - o0).min(48);
    let mut sbuf = [0u8; 48];
    // strings start after the offset table: 4 + (n+1)*4 bytes in
    let n_off = {
        let mut nb = [0u8; 4];
        try_rc!(sd_read_bytes(vtb, 0, &mut nb), "vtb n");
        u32::from_le_bytes(nb) as usize
    };
    let base = 4 + (n_off + 1) * 4;
    try_rc!(sd_read_bytes(vtb, base + o0, &mut sbuf[..len]), "vtb s");
    if let Ok(s) = core::str::from_utf8(&sbuf[..len]) {
        rprint!("{}", s);
        display::print(s);
    }
    Ok(())
}
