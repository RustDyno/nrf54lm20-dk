//! Blob selftest driver: flash the executor firmware, stream one compiled
//! layer blob into the slot, run it on the NPU against a golden input, and
//! compare the output with the TFLite interpreter's (bit-exactness is the
//! pass bar, as established by the KWS project).
//!
//!   cargo run --release -- <firmware.elf> <blob.bin> <input.bin> <expect.bin>
//!
//! This is the seed of the full tape orchestrator (M2): the mailbox client
//! below is the complete device protocol; the tape player will drive the
//! same calls from a schedule emitted by model/.

use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use object::{Object, ObjectSymbol};
use probe_rs::config::Registry;
use probe_rs::probe::list::Lister;
use probe_rs::rtt::{Rtt, ScanRegion};
use probe_rs::{flashing, Core, MemoryInterface, Permissions};

const CHIP: &str = "nRF54LM20B";
const CHIP_DESCRIPTION: &str = include_str!("../../firmware/targets/nRF54LM20B.yaml");

/// Must match firmware/src/slot.rs and firmware/memory.x.
const SLOT_BASE: u64 = 0x2004_B000;
const MAILBOX_MAGIC: u32 = 0x4C41_5952;

// Mailbox field offsets (repr(C) in firmware/src/main.rs).
const MB_MAGIC: u64 = 0;
const MB_CMD_SEQ: u64 = 4;
const MB_CMD: u64 = 8;
const MB_ARGS: u64 = 12;
const MB_ACK_SEQ: u64 = 44;
const MB_STATUS: u64 = 48;

const CMD_PING: u32 = 1;
const CMD_RUN_NPU: u32 = 2;

struct Mailbox {
    base: u64,
    seq: u32,
}

impl Mailbox {
    fn call(&mut self, core: &mut Core, cmd: u32, args: &[u32]) -> Result<i32> {
        let mut a = [0u32; 8];
        a[..args.len()].copy_from_slice(args);
        core.write_32(self.base + MB_ARGS, &a)?;
        core.write_word_32(self.base + MB_CMD, cmd)?;
        self.seq += 1;
        core.write_word_32(self.base + MB_CMD_SEQ, self.seq)?;
        let deadline = Instant::now() + Duration::from_secs(10);
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

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [elf, blob, input, expect] = args.as_slice() else {
        bail!("usage: whisper-host <firmware.elf> <blob.bin> <input.bin> <expect.bin>");
    };
    let blob_data = std::fs::read(blob).context("reading blob")?;
    let input_data = std::fs::read(input).context("reading input")?;
    let expect_data = std::fs::read(expect).context("reading expected output")?;

    let elf_data = std::fs::read(elf).context("reading firmware ELF")?;
    let elf_obj = object::File::parse(&*elf_data).context("parsing ELF")?;
    let sym = |name: &str| -> Result<u64> {
        elf_obj
            .symbols()
            .find(|s| s.name() == Ok(name))
            .map(|s| s.address())
            .ok_or_else(|| anyhow!("symbol {name} not found in firmware ELF"))
    };
    let mailbox_addr = sym("MAILBOX")?;
    let arena_addr = sym("ARENA")?;
    let interlayer_addr = sym("nrf_axon_interlayer_buffer")?;
    let rtt_addr = sym("_SEGGER_RTT").ok();

    // Open the probe, flash, reset into the image (same flow as PDM-MIC).
    let lister = Lister::new();
    let probes = lister.list_all();
    let info = probes
        .first()
        .ok_or_else(|| anyhow!("no debug probe found (is the DK plugged in?)"))?;
    let mut probe = lister.open(info).context("opening probe")?;
    // Raise the SWD clock if the probe allows it (the J-Link OB rejects
    // values outside its table; throughput is link-command-bound anyway,
    // ~26 KB/s -- see NOTES on the streaming bottleneck).
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
    flashing::download_file(&mut session, elf, flashing::FormatKind::Elf).context("flashing")?;
    {
        let mut core = session.core(0)?;
        core.reset_and_halt(Duration::from_millis(500))?;
        // RAM survives reflash: clear any stale magic from a previous run so
        // the wait below only passes once THIS boot republished it.
        core.write_word_32(mailbox_addr + MB_MAGIC, 0)?;
        core.run()?;
    }

    let mut core = session.core(0)?;

    // Wait for the firmware to finish init (mailbox magic appears).
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if core.read_word_32(mailbox_addr + MB_MAGIC)? == MAILBOX_MAGIC {
            break;
        }
        if Instant::now() > deadline {
            bail!("firmware did not publish the mailbox magic (Axon init failed?)");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let mut mb = Mailbox {
        base: mailbox_addr,
        seq: core.read_word_32(mailbox_addr + MB_CMD_SEQ)?,
    };

    let rc = mb.call(&mut core, CMD_PING, &[])?;
    eprintln!("ping -> {rc:#x}");

    // Stream the blob into the slot and the golden input into the arena.
    eprintln!("loading blob ({} B) + input ({} B)", blob_data.len(), input_data.len());
    let t0 = Instant::now();
    // `write` picks word-sized accesses (write_8 byte access crawls at
    // ~7 KB/s over the J-Link OB); slot and arena are word-aligned.
    core.write(SLOT_BASE, &blob_data)?;
    core.write(arena_addr, &input_data)?;
    eprintln!("  streamed in {:.2} s ({:.0} KB/s)",
        t0.elapsed().as_secs_f64(),
        (blob_data.len() + input_data.len()) as f64 / 1024.0 / t0.elapsed().as_secs_f64());

    let out_addr = arena_addr + input_data.len() as u64;
    let t0 = Instant::now();
    let rc = mb.call(&mut core, CMD_RUN_NPU, &[arena_addr as u32, out_addr as u32])?;
    eprintln!("run_npu -> {rc} in {:.1} ms", t0.elapsed().as_secs_f64() * 1e3);
    if rc != 0 {
        drain_rtt(&mut core, &elf_data, rtt_addr);
        bail!("NPU run failed with {rc}");
    }

    let mut out = vec![0u8; expect_data.len()];
    core.read_8(out_addr, &mut out)?;
    // Keep both the packed output and the raw interlayer image around for
    // layout analysis when the comparison fails.
    std::fs::write("out-packed.bin", &out).ok();
    let mut il = vec![0u8; expect_data.len()];
    core.read_8(interlayer_addr, &mut il)?;
    std::fs::write("out-interlayer.bin", &il).ok();
    let diffs = out
        .iter()
        .zip(expect_data.iter())
        .filter(|(a, b)| a != b)
        .count();
    drain_rtt(&mut core, &elf_data, rtt_addr);
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

/// Print whatever the firmware logged so failures come with context.
fn drain_rtt(core: &mut Core, elf_data: &[u8], rtt_addr: Option<u64>) {
    let region = match rtt_addr {
        Some(a) => ScanRegion::Exact(a),
        None => ScanRegion::Ram,
    };
    let _ = elf_data;
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
