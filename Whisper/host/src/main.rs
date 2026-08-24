//! Host driver for the layer-streaming Whisper firmware.
//!
//!   whisper-host selftest <firmware.elf> <blob.bin> <input.bin> <expect.bin>
//!   whisper-host tape <firmware.elf> <tape.json>
//!
//! selftest: stream one blob + golden input, run it on the NPU, compare.
//! tape: play a schedule emitted by model/tape.py. The tape is five generic
//! ops (load blob, write file to address, mailbox cmd, check memory against
//! a file, read memory to a file), so new device kernels need no host
//! changes; all structure lives in the generator. File paths are relative
//! to the tape's directory. Addresses are absolute (ARENA and SLOT are at
//! fixed addresses in the firmware).

use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use object::{Object, ObjectSymbol};
use probe_rs::config::Registry;
use probe_rs::probe::list::Lister;
use probe_rs::rtt::{Rtt, ScanRegion};
use probe_rs::{flashing, Core, MemoryInterface, Permissions, Session};
use serde::Deserialize;

mod decode;

const CHIP: &str = "nRF54LM20B";
const CHIP_DESCRIPTION: &str = include_str!("../../firmware/targets/nRF54LM20B.yaml");

/// Must match firmware/src/slot.rs and firmware/memory.x.
pub(crate) const SLOT_BASE: u64 = 0x2004_B000;
const MAILBOX_MAGIC: u32 = 0x4C41_5952;

// Mailbox field offsets (repr(C) in firmware/src/main.rs).
const MB_MAGIC: u64 = 0;
const MB_CMD_SEQ: u64 = 4;
const MB_CMD: u64 = 8;
const MB_ARGS: u64 = 12;
const MB_ACK_SEQ: u64 = 44;
const MB_STATUS: u64 = 48;

pub(crate) const CMD_PING: u32 = 1;
pub(crate) const CMD_RUN_NPU: u32 = 2;

pub(crate) struct Mailbox {
    pub base: u64,
    pub seq: u32,
}

impl Mailbox {
    pub(crate) fn call(&mut self, core: &mut Core, cmd: u32, args: &[u32]) -> Result<i32> {
        let mut a = [0u32; 8];
        a[..args.len()].copy_from_slice(args);
        core.write_32(self.base + MB_ARGS, &a)?;
        core.write_word_32(self.base + MB_CMD, cmd)?;
        self.seq += 1;
        core.write_word_32(self.base + MB_CMD_SEQ, self.seq)?;
        let deadline = Instant::now() + Duration::from_secs(20);
        loop {
            let ack = core.read_word_32(self.base + MB_ACK_SEQ)?;
            if ack == self.seq {
                return Ok(core.read_word_32(self.base + MB_STATUS)? as i32);
            }
            if Instant::now() > deadline {
                bail!("mailbox timeout on cmd {cmd} (ack {ack}, want {})", self.seq);
            }
            std::thread::sleep(Duration::from_millis(1));
        }
    }
}

/// Flash the firmware, reset, wait for the mailbox to come up.
pub(crate) fn setup(elf: &str) -> Result<(Session, u64, Option<u64>)> {
    let elf_data = std::fs::read(elf).context("reading firmware ELF")?;
    let elf_obj = object::File::parse(&*elf_data).context("parsing ELF")?;
    let sym = |name: &str| -> Option<u64> {
        elf_obj
            .symbols()
            .find(|s| s.name() == Ok(name))
            .map(|s| s.address())
    };
    let mailbox_addr = sym("MAILBOX").ok_or_else(|| anyhow!("MAILBOX not in ELF"))?;
    let rtt_addr = sym("_SEGGER_RTT");

    let lister = Lister::new();
    let probes = lister.list_all();
    let info = probes
        .first()
        .ok_or_else(|| anyhow!("no debug probe found (is the DK plugged in?)"))?;
    let mut probe = lister.open(info).context("opening probe")?;
    // Raise the SWD clock if the probe allows it (this J-Link OB caps at
    // 2000 kHz; throughput is ~74 KB/s -- see NOTES on the bottleneck).
    for khz in [4000, 2000, 1000] {
        if let Ok(actual) = probe.set_speed(khz) {
            eprintln!("probe speed: {actual} kHz");
            break;
        }
    }
    let mut registry = Registry::from_builtin_families();
    registry
        .add_target_family_from_yaml(CHIP_DESCRIPTION)
        .context("registering nRF54LM20B target")?;
    let mut session = probe
        .attach_with_registry(CHIP, Permissions::default(), &registry)
        .context("attaching to target")?;

    eprintln!("flashing {elf} ...");
    // Double-buffered RRAM programming corrupts one word per 4 KB page on
    // this target (probe-rs + cloned nRF54LM20B yaml); verify to make any
    // recurrence loud instead of a heisen-crash.
    let mut opts = flashing::DownloadOptions::default();
    opts.disable_double_buffering = true;
    opts.verify = true;
    let loader = flashing::build_loader(&mut session, elf, flashing::Format::Elf(Default::default()), None)
        .context("building flash loader")?;
    loader
        .commit(&mut session, opts)
        .context("flashing (verified)")?;
    {
        let mut core = session.core(0)?;
        core.reset_and_halt(Duration::from_millis(500))?;
        // RAM survives reflash: clear any stale magic from a previous run so
        // the wait below only passes once THIS boot republished it.
        core.write_word_32(mailbox_addr + MB_MAGIC, 0)?;
        core.run()?;
    }
    {
        let mut core = session.core(0)?;
        // The watchdog may reset once on a wedged engine; allow time for it.
        let deadline = Instant::now() + Duration::from_secs(6);
        loop {
            match core.read_word_32(mailbox_addr + MB_MAGIC)? {
                m if m == MAILBOX_MAGIC => break,
                0x4641_494C => {
                    let rc = core.read_word_32(mailbox_addr + MB_STATUS)? as i32;
                    bail!("Axon driver init failed with rc {rc}");
                }
                m if Instant::now() > deadline => bail!(
                    "firmware not ready (magic {m:#010x}: {})",
                    match m {
                        0x424F_4F54 => "hung in Axon init -- power-cycle the board",
                        0 => "never reached main",
                        _ => "unknown state",
                    }
                ),
                _ => std::thread::sleep(Duration::from_millis(20)),
            }
        }
    }
    Ok((session, mailbox_addr, rtt_addr))
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("selftest") if args.len() == 5 => {
            selftest(&args[1], &args[2], &args[3], &args[4])
        }
        Some("tape") if args.len() == 3 => tape(&args[1], &args[2]),
        Some("decode") if args.len() == 4 => decode::decode(&args[1], &args[2], &args[3]),
        Some("halt") => halt_info(),
        _ => bail!(
            "usage: whisper-host selftest <firmware.elf> <blob.bin> <input.bin> <expect.bin>\n\
             \x20      whisper-host tape <firmware.elf> <tape.json>\n\
             \x20      whisper-host decode <firmware.elf> <plan_dir> <blobs_dir>"
        ),
    }
}

// --- tape player ------------------------------------------------------------

#[derive(Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
enum Step {
    /// Load a compiled layer blob into the slot (skipped if already loaded).
    Blob { file: String },
    /// Write a data file to an absolute device address.
    Write { file: String, addr: u64 },
    /// Issue a mailbox command; nonzero status aborts the tape.
    Cmd { code: u32, #[serde(default)] args: Vec<u32> },
    /// Compare device memory against a file (byte length = file length).
    /// `width` is the element size (1 = int8, 2 = int16 little-endian);
    /// tolerances apply per element, not per byte.
    Check {
        file: String,
        addr: u64,
        label: String,
        #[serde(default)] tol: i32,
        #[serde(default = "default_width")] width: u32,
    },
    /// Dump device memory to a file.
    Read { file: String, addr: u64, len: u32 },
}

fn default_width() -> u32 {
    1
}

#[derive(Deserialize)]
struct Tape {
    steps: Vec<Step>,
}

fn tape(elf: &str, tape_path: &str) -> Result<()> {
    let tape: Tape = serde_json::from_str(
        &std::fs::read_to_string(tape_path).context("reading tape")?,
    )
    .context("parsing tape")?;
    let dir = Path::new(tape_path).parent().unwrap_or(Path::new("."));

    let (mut session, mailbox_addr, rtt_addr) = setup(elf)?;
    let mut core = session.core(0)?;
    let mut mb = Mailbox {
        base: mailbox_addr,
        seq: core.read_word_32(mailbox_addr + MB_CMD_SEQ)?,
    };
    mb.call(&mut core, CMD_PING, &[])?;

    let t0 = Instant::now();
    let mut loaded_blob = String::new();
    let mut streamed = 0usize;
    let (mut checks, mut failed) = (0u32, 0u32);
    for (i, step) in tape.steps.iter().enumerate() {
        match step {
            Step::Blob { file } => {
                if *file != loaded_blob {
                    let data = std::fs::read(dir.join(file))
                        .with_context(|| format!("blob {file}"))?;
                    core.write(SLOT_BASE, &data)?;
                    streamed += data.len();
                    loaded_blob = file.clone();
                }
            }
            Step::Write { file, addr } => {
                let data = std::fs::read(dir.join(file))
                    .with_context(|| format!("data {file}"))?;
                core.write(*addr, &data)?;
                streamed += data.len();
            }
            Step::Cmd { code, args } => {
                let rc = match mb.call(&mut core, *code, args) {
                    Ok(rc) => rc,
                    Err(e) => {
                        drain_rtt(&mut core, rtt_addr);
                        return Err(e);
                    }
                };
                if rc != 0 {
                    drain_rtt(&mut core, rtt_addr);
                    bail!("step {i}: cmd {code} failed with status {rc}");
                }
            }
            Step::Check { file, addr, label, tol, width } => {
                let expect = std::fs::read(dir.join(file))
                    .with_context(|| format!("expect {file}"))?;
                let mut got = vec![0u8; expect.len()];
                core.read_8(*addr, &mut got)?;
                let (mut diffs, mut maxerr) = (0usize, 0i32);
                let elem = |b: &[u8], i: usize| -> i32 {
                    match width {
                        2 => i16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as i32,
                        _ => b[i] as i8 as i32,
                    }
                };
                for i in 0..expect.len() / *width as usize {
                    let d = (elem(&got, i) - elem(&expect, i)).abs();
                    if d > *tol {
                        diffs += 1;
                    }
                    maxerr = maxerr.max(d);
                }
                checks += 1;
                if diffs == 0 {
                    eprintln!("  check {label}: PASS ({} B, max |err| {maxerr})", got.len());
                } else {
                    failed += 1;
                    eprintln!(
                        "  check {label}: FAIL {diffs}/{} bytes over tol {tol} (max |err| {maxerr})",
                        got.len()
                    );
                    std::fs::write(dir.join(format!("{label}.got.bin")), &got).ok();
                }
            }
            Step::Read { file, addr, len } => {
                let mut data = vec![0u8; *len as usize];
                core.read_8(*addr, &mut data)?;
                std::fs::write(dir.join(file), &data)?;
            }
        }
    }
    drain_rtt(&mut core, rtt_addr);
    eprintln!(
        "tape done: {} steps, {} KB streamed, {:.2} s, checks {}/{} passed",
        tape.steps.len(),
        streamed / 1024,
        t0.elapsed().as_secs_f64(),
        checks - failed,
        checks
    );
    if failed > 0 {
        bail!("{failed} checks failed");
    }
    Ok(())
}

// --- single-blob selftest ---------------------------------------------------

fn selftest(elf: &str, blob: &str, input: &str, expect: &str) -> Result<()> {
    let blob_data = std::fs::read(blob).context("reading blob")?;
    let input_data = std::fs::read(input).context("reading input")?;
    let expect_data = std::fs::read(expect).context("reading expected output")?;

    let (mut session, mailbox_addr, rtt_addr) = setup(elf)?;
    let mut core = session.core(0)?;
    let mut mb = Mailbox {
        base: mailbox_addr,
        seq: core.read_word_32(mailbox_addr + MB_CMD_SEQ)?,
    };
    let rc = mb.call(&mut core, CMD_PING, &[])?;
    eprintln!("ping -> {rc:#x}");

    // Arena base mirrors firmware memory.x (fixed region).
    let arena_addr: u64 = 0x2003_2000;
    eprintln!("loading blob ({} B) + input ({} B)", blob_data.len(), input_data.len());
    let t0 = Instant::now();
    core.write(SLOT_BASE, &blob_data)?;
    core.write(arena_addr, &input_data)?;
    eprintln!(
        "  streamed in {:.2} s ({:.0} KB/s)",
        t0.elapsed().as_secs_f64(),
        (blob_data.len() + input_data.len()) as f64 / 1024.0 / t0.elapsed().as_secs_f64()
    );

    let out_addr = arena_addr + input_data.len() as u64;
    let t0 = Instant::now();
    let rc = mb.call(&mut core, CMD_RUN_NPU, &[arena_addr as u32, out_addr as u32])?;
    eprintln!("run_npu -> {rc} in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    if rc != 0 {
        drain_rtt(&mut core, rtt_addr);
        bail!("NPU run failed with {rc}");
    }

    let mut out = vec![0u8; expect_data.len()];
    core.read_8(out_addr, &mut out)?;
    let diffs = out
        .iter()
        .zip(expect_data.iter())
        .filter(|(a, b)| a != b)
        .count();
    drain_rtt(&mut core, rtt_addr);
    if diffs == 0 {
        eprintln!("PASS: output bit-exact vs the TFLite interpreter ({} B)", out.len());
        Ok(())
    } else {
        let maxerr = out
            .iter()
            .zip(expect_data.iter())
            .map(|(a, b)| (*a as i8 as i32 - *b as i8 as i32).abs())
            .max()
            .unwrap_or(0);
        bail!("FAIL: {diffs}/{} bytes differ (max |err| {maxerr})", out.len());
    }
}

/// Debug aid: attach without flashing, halt, dump PC/LR/SP + PRIMASK.
fn halt_info() -> Result<()> {
    let lister = Lister::new();
    let probes = lister.list_all();
    let info = probes.first().ok_or_else(|| anyhow!("no probe"))?;
    let probe = lister.open(info)?;
    let mut registry = Registry::from_builtin_families();
    registry.add_target_family_from_yaml(CHIP_DESCRIPTION)?;
    let mut session = probe.attach_with_registry(CHIP, Permissions::default(), &registry)?;
    let mut core = session.core(0)?;
    core.halt(Duration::from_millis(500))?;
    let pc: u64 = core.read_core_reg(core.program_counter())?;
    let lr: u64 = core.read_core_reg(core.return_address())?;
    let sp: u64 = core.read_core_reg(core.stack_pointer())?;
    eprintln!("pc {pc:#010x}  lr {lr:#010x}  sp {sp:#010x}");
    let mut stack = [0u32; 16];
    core.read_32(sp, &mut stack)?;
    eprintln!("stack: {:08x?}", stack);
    core.run()?;
    Ok(())
}

/// Print whatever the firmware logged so failures come with context.
fn drain_rtt(core: &mut Core, rtt_addr: Option<u64>) {
    let region = match rtt_addr {
        Some(a) => ScanRegion::Exact(a),
        None => ScanRegion::Ram,
    };
    if let Ok(mut rtt) = Rtt::attach_region(core, &region) {
        if let Some(ch) = rtt.up_channel(0) {
            let mut buf = [0u8; 2048];
            if let Ok(n) = ch.read(core, &mut buf) {
                if n > 0 {
                    eprint!("[fw] {}", String::from_utf8_lossy(&buf[..n]));
                }
            }
        }
    }
}
