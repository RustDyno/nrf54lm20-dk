//! Serves the Whisper model image to the DK over the USB device-mode link.
//!
//! The firmware's "mock-usb" build asks this daemon for 512-byte blocks
//! instead of reading a USB stick (firmware/src/mockblk.rs). That buys
//! three things the stick cannot: the image can be rebuilt and re-served
//! without unplugging anything, the audio the device "records" can be a
//! fixed clip so runs are comparable, and every scratch block the device
//! writes lands back in the work file where the analysis tooling can read
//! it -- mel, encoder output and cross K/V come off a run for free.
//!
//! Usage:
//!     mockusb serve [--img model/out/sd.img] [--work out/mock-work.img]
//!                   [--audio clip.wav] [--port /dev/ttyACMn] [--fresh]

mod tty;

use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{bail, Context, Result};

const BLOCK: u64 = 512;

const REQ_MAGIC: u32 = 0x514B_4C42; // "BLKQ"
const RSP_MAGIC: u32 = 0x524B_4C42; // "BLKR"
const OP_READ: u8 = 1;
const OP_WRITE: u8 = 2;
const OP_INFO: u8 = 3;

// Scratch layout, mirrored from firmware/src/app.rs. Offsets are blocks
// past the image's scratch_lba.
const S_PCM: u64 = 0; // 750 blocks, 12 s of int16 at 16 kHz
const S_INJECT: u64 = 750; // marker block: PCM above is pre-loaded
/// Room past every scratch region the firmware touches (the highest is
/// S_XKV + 8*480 = 13312, and the sdtest tape pokes scratch + 20000).
const SCRATCH_BLOCKS: u64 = 20608;

const INJECT_MAGIC: &[u8; 8] = b"WMOCKAU1";
const N_SAMPLES: usize = 192_000; // 12 s at 16 kHz, as the firmware expects

fn sum32(buf: &[u8]) -> u32 {
    let mut s = 0u32;
    for c in buf.chunks_exact(4) {
        s = s
            .rotate_left(1)
            .wrapping_add(u32::from_le_bytes([c[0], c[1], c[2], c[3]]));
    }
    s
}

fn get32(b: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([b[o], b[o + 1], b[o + 2], b[o + 3]])
}

fn put32(b: &mut [u8], o: usize, v: u32) {
    b[o..o + 4].copy_from_slice(&v.to_le_bytes());
}

// --- image -----------------------------------------------------------------

/// Read scratch_lba out of the image's own plan blob, so the daemon never
/// has to be told a layout the image already records. The header is
/// "WSPRIMG1", u32 version, u32 entries, then 32-byte {name[24], u32
/// lba, u32 bytes} entries; the plan starts with magic "WPLN", u32
/// version, u32 scratch_lba.
fn scratch_lba(img: &Path) -> Result<u64> {
    let mut f = File::open(img).with_context(|| format!("open {}", img.display()))?;
    let mut hdr = vec![0u8; 16 * BLOCK as usize];
    f.read_exact(&mut hdr)?;
    if &hdr[..8] != b"WSPRIMG1" {
        bail!("{} is not a Whisper model image", img.display());
    }
    let n = get32(&hdr, 12) as usize;
    for i in 0..n {
        let e = 16 + i * 32;
        let name = String::from_utf8_lossy(&hdr[e..e + 24])
            .trim_end_matches('\0')
            .to_string();
        if name == "plan" {
            let lba = get32(&hdr, e + 24) as u64;
            f.seek(SeekFrom::Start(lba * BLOCK))?;
            let mut p = [0u8; 16];
            f.read_exact(&mut p)?;
            if get32(&p, 0) != 0x4E4C_5057 {
                bail!("plan blob has the wrong magic");
            }
            return Ok(get32(&p, 8) as u64);
        }
    }
    bail!("no plan entry in {}", img.display())
}

/// Copy the pristine image into a writable work file sized to cover the
/// scratch regions, so the device's spills have somewhere to land and stay
/// readable afterwards.
fn prepare_work(img: &Path, work: &Path, fresh: bool) -> Result<u64> {
    let scratch = scratch_lba(img)?;
    let total = scratch + SCRATCH_BLOCKS;
    if fresh || !work.exists() {
        if let Some(d) = work.parent() {
            std::fs::create_dir_all(d).ok();
        }
        std::fs::copy(img, work)
            .with_context(|| format!("copy {} -> {}", img.display(), work.display()))?;
        println!("mockusb: work image {} (fresh copy)", work.display());
    } else {
        println!("mockusb: work image {} (reused)", work.display());
    }
    let f = OpenOptions::new().write(true).open(work)?;
    f.set_len(total * BLOCK)?;
    println!(
        "mockusb: {} blocks, scratch at {} ({} MB total)",
        total,
        scratch,
        total * BLOCK / (1 << 20)
    );
    Ok(scratch)
}

/// 16-bit mono 16 kHz samples from a .wav or headerless .raw/.pcm file.
fn load_pcm(path: &Path) -> Result<Vec<i16>> {
    let mut b = Vec::new();
    File::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .read_to_end(&mut b)?;
    if b.len() >= 12 && &b[..4] == b"RIFF" && &b[8..12] == b"WAVE" {
        let mut pos = 12;
        let mut fmt = None;
        while pos + 8 <= b.len() {
            let id = &b[pos..pos + 4];
            let sz = get32(&b, pos + 4) as usize;
            let body = pos + 8;
            if id == b"fmt " && body + 16 <= b.len() {
                let channels = u16::from_le_bytes([b[body + 2], b[body + 3]]);
                let rate = get32(&b, body + 4);
                let bits = u16::from_le_bytes([b[body + 14], b[body + 15]]);
                if channels != 1 || rate != 16000 || bits != 16 {
                    bail!(
                        "{}: need mono 16 kHz 16-bit, got {} ch / {} Hz / {} bit \
                         (convert with: ffmpeg -i in -ac 1 -ar 16000 -c:a pcm_s16le out.wav)",
                        path.display(),
                        channels,
                        rate,
                        bits
                    );
                }
                fmt = Some(());
            } else if id == b"data" {
                if fmt.is_none() {
                    bail!("{}: data chunk before fmt", path.display());
                }
                let end = (body + sz).min(b.len());
                return Ok(b[body..end]
                    .chunks_exact(2)
                    .map(|c| i16::from_le_bytes([c[0], c[1]]))
                    .collect());
            }
            pos = body + sz + (sz & 1);
        }
        bail!("{}: no data chunk", path.display());
    }
    Ok(b.chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Write the clip into the PCM scratch region and drop the marker block
/// that tells the firmware to skip the microphone.
fn inject_audio(work: &Path, scratch: u64, audio: &Path) -> Result<()> {
    let mut pcm = load_pcm(audio)?;
    let given = pcm.len();
    pcm.resize(N_SAMPLES, 0); // pad or truncate to the fixed 12 s window
    let mut f = OpenOptions::new().write(true).open(work)?;
    f.seek(SeekFrom::Start((scratch + S_PCM) * BLOCK))?;
    let mut bytes = Vec::with_capacity(N_SAMPLES * 2);
    for s in &pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    f.write_all(&bytes)?;

    let mut marker = vec![0u8; BLOCK as usize];
    marker[..8].copy_from_slice(INJECT_MAGIC);
    put32(&mut marker, 8, N_SAMPLES as u32);
    put32(&mut marker, 12, 16000);
    f.seek(SeekFrom::Start((scratch + S_INJECT) * BLOCK))?;
    f.write_all(&marker)?;
    f.sync_all()?;
    println!(
        "mockusb: injected {} ({} samples given, {} served); the device will skip the mic",
        audio.display(),
        given,
        N_SAMPLES
    );
    Ok(())
}

/// Remove the marker so the next run records from the microphone again.
fn clear_inject(work: &Path, scratch: u64) -> Result<()> {
    let mut f = OpenOptions::new().write(true).open(work)?;
    f.seek(SeekFrom::Start((scratch + S_INJECT) * BLOCK))?;
    f.write_all(&vec![0u8; BLOCK as usize])?;
    f.sync_all()?;
    println!("mockusb: injection marker cleared; the device will use the mic");
    Ok(())
}

// --- port ---------------------------------------------------------------------

/// Find the CDC-ACM port the firmware presents (pid.codes prototyping ids,
/// matching usbdev.rs), together with the kernel's device number for it.
///
/// The device number matters on reconnect: after a reflash the old tty
/// node lingers for a moment, and opening it yields a session that dies
/// immediately -- but not before the host ACKs and then discards the one
/// request the device patiently sent. Waiting for a *different* devnum is
/// what distinguishes the re-enumerated board from its own corpse.
fn find_port() -> Result<(PathBuf, u32)> {
    let mut found = Vec::new();
    for e in std::fs::read_dir("/sys/class/tty")? {
        let e = e?;
        let name = e.file_name().to_string_lossy().to_string();
        if !name.starts_with("ttyACM") {
            continue;
        }
        // .../ttyACMn/device is the interface; its parent is the device.
        let dev = e.path().join("device");
        let Ok(iface) = std::fs::canonicalize(&dev) else {
            continue;
        };
        let Some(usbdev) = iface.parent() else { continue };
        let vid = std::fs::read_to_string(usbdev.join("idVendor")).unwrap_or_default();
        let pid = std::fs::read_to_string(usbdev.join("idProduct")).unwrap_or_default();
        if vid.trim() == "1209" && pid.trim() == "0001" {
            let devnum = std::fs::read_to_string(usbdev.join("devnum"))
                .ok()
                .and_then(|s| s.trim().parse::<u32>().ok())
                .unwrap_or(0);
            found.push((PathBuf::from("/dev").join(&name), devnum));
        }
    }
    found.sort();
    match found.len() {
        0 => bail!("no 1209:0001 port"),
        _ => Ok(found.remove(0)),
    }
}

/// The device only enumerates once its firmware reaches storage init, so
/// the daemon is routinely started first; wait for the port rather than
/// making the operator race it.
fn wait_for_port(secs: u64, skip_devnum: u32) -> Result<(PathBuf, u32)> {
    let t0 = Instant::now();
    let mut said = false;
    loop {
        if let Ok((p, devnum)) = find_port() {
            if devnum != skip_devnum {
                // Let udev finish creating the node and applying its
                // permissions before the first open.
                std::thread::sleep(std::time::Duration::from_millis(300));
                return Ok((p, devnum));
            }
        }
        if t0.elapsed().as_secs() >= secs {
            bail!(
                "no 1209:0001 CDC-ACM port appeared in {secs}s. Is the mock-usb firmware \
                 flashed and J3 cabled to this PC? (check: lsusb -d 1209:0001)"
            );
        }
        if !said {
            println!("mockusb: waiting for the device to enumerate...");
            said = true;
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

// --- serve --------------------------------------------------------------------

struct Stats {
    rd: u64,
    wr: u64,
    reqs: u64,
    t0: Instant,
    last: Instant,
}

/// One connected session. Returns when the link drops (the usual cause is
/// a reflash or a board reset), so the caller can wait for the device to
/// come back instead of making the operator restart the daemon on every
/// firmware iteration.
fn serve(port: &Path, work: &Path, total_blocks: u64) -> Result<()> {
    let mut link = tty::Tty::open(port)?;
    let mut img = OpenOptions::new().read(true).write(true).open(work)?;
    println!("mockusb: serving {} on {}", work.display(), port.display());

    let mut req = [0u8; BLOCK as usize];
    let mut rsp = [0u8; BLOCK as usize];
    let mut payload = Vec::new();
    let mut st = Stats {
        rd: 0,
        wr: 0,
        reqs: 0,
        t0: Instant::now(),
        last: Instant::now(),
    };

    loop {
        link.read_exact(&mut req)?;
        let magic = get32(&req, 0);
        if magic != REQ_MAGIC {
            // Framing is fixed-size in both directions, so this should be
            // unreachable; say so loudly rather than serve garbage.
            let head: Vec<String> = req[..32].iter().map(|b| format!("{b:02x}")).collect();
            let nonzero = req.iter().filter(|&&b| b != 0).count();
            bail!(
                "framing lost: request magic {magic:#010x}, {nonzero}/512 nonzero bytes, \
                 head {}",
                head.join(" ")
            );
        }
        let op = req[4];
        let lba = get32(&req, 8) as u64;
        let count = get32(&req, 12) as u64;
        let claim = get32(&req, 16);
        let tag = get32(&req, 20);

        rsp.fill(0);
        put32(&mut rsp, 0, RSP_MAGIC);
        // Echoed so the device can prove the reply is the one it asked for.
        put32(&mut rsp, 16, tag);
        st.reqs += 1;

        match op {
            OP_INFO => {
                put32(&mut rsp, 8, total_blocks as u32);
                link.write_all(&rsp)?;
                println!("mockusb: device attached, serving {total_blocks} blocks");
            }
            OP_READ => {
                let bytes = (count * BLOCK) as usize;
                if lba + count > total_blocks {
                    put32(&mut rsp, 4, (-1i32) as u32);
                    link.write_all(&rsp)?;
                    eprintln!("mockusb: read past end (lba {lba} x{count})");
                    continue;
                }
                payload.resize(bytes, 0);
                img.seek(SeekFrom::Start(lba * BLOCK))?;
                img.read_exact(&mut payload)?;
                put32(&mut rsp, 8, count as u32);
                put32(&mut rsp, 12, sum32(&payload));
                link.write_all(&rsp)?;
                link.write_all(&payload)?;
                st.rd += bytes as u64;
            }
            OP_WRITE => {
                let bytes = (count * BLOCK) as usize;
                payload.resize(bytes, 0);
                link.read_exact(&mut payload)?;
                let got = sum32(&payload);
                if got != claim {
                    put32(&mut rsp, 4, (-2i32) as u32);
                    link.write_all(&rsp)?;
                    eprintln!("mockusb: write checksum {got:#010x} != {claim:#010x}");
                    continue;
                }
                if lba + count > total_blocks {
                    put32(&mut rsp, 4, (-1i32) as u32);
                    link.write_all(&rsp)?;
                    eprintln!("mockusb: write past end (lba {lba} x{count})");
                    continue;
                }
                img.seek(SeekFrom::Start(lba * BLOCK))?;
                img.write_all(&payload)?;
                link.write_all(&rsp)?;
                st.wr += bytes as u64;
            }
            _ => {
                put32(&mut rsp, 4, (-3i32) as u32);
                link.write_all(&rsp)?;
            }
        }

        if st.last.elapsed().as_secs() >= 5 {
            let el = st.t0.elapsed().as_secs_f64();
            println!(
                "mockusb: {:.1} MB read / {:.1} MB written, {} requests, {:.2} MB/s",
                st.rd as f64 / 1e6,
                st.wr as f64 / 1e6,
                st.reqs,
                (st.rd + st.wr) as f64 / 1e6 / el.max(1e-9)
            );
            st.last = Instant::now();
        }
    }
}

// --- cli ------------------------------------------------------------------------

fn usage() -> ! {
    eprintln!(
        "usage: mockusb serve [options]\n\
         \n\
           --img <path>     pristine model image   (default ../../model/out/sd.img)\n\
           --work <path>    writable copy served   (default ../../model/out/mock-work.img)\n\
           --audio <path>   inject a wav/raw clip as the recording, skipping the mic\n\
           --no-audio       clear a previous injection (device records from the mic)\n\
           --port <dev>     CDC-ACM port           (default: autodetect 1209:0001)\n\
           --fresh          re-copy the work image, discarding the last run's scratch\n\
           --prep-only      prepare the work image and exit without serving\n"
    );
    std::process::exit(2)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        usage();
    }
    if args[0] != "serve" {
        usage();
    }

    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()?;
    let mut img = root.join("model/out/sd.img");
    let mut work = root.join("model/out/mock-work.img");
    let mut audio: Option<PathBuf> = None;
    let mut no_audio = false;
    let mut port: Option<PathBuf> = None;
    let mut fresh = false;
    let mut prep_only = false;

    let mut i = 1;
    while i < args.len() {
        let need = |i: usize| -> String {
            if i + 1 >= args.len() {
                usage();
            }
            args[i + 1].clone()
        };
        match args[i].as_str() {
            "--img" => {
                img = PathBuf::from(need(i));
                i += 2;
            }
            "--work" => {
                work = PathBuf::from(need(i));
                i += 2;
            }
            "--audio" => {
                audio = Some(PathBuf::from(need(i)));
                i += 2;
            }
            "--port" => {
                port = Some(PathBuf::from(need(i)));
                i += 2;
            }
            "--no-audio" => {
                no_audio = true;
                i += 1;
            }
            "--fresh" => {
                fresh = true;
                i += 1;
            }
            "--prep-only" => {
                prep_only = true;
                i += 1;
            }
            _ => usage(),
        }
    }

    let scratch = prepare_work(&img, &work, fresh)?;
    if let Some(a) = &audio {
        inject_audio(&work, scratch, a)?;
    } else if no_audio {
        clear_inject(&work, scratch)?;
    }
    if prep_only {
        return Ok(());
    }

    let total = scratch + SCRATCH_BLOCKS;
    // Devnum of the session that just ended, so the next connection waits
    // for a genuinely re-enumerated device rather than the stale node.
    let mut stale = 0u32;
    loop {
        let (p, devnum) = match &port {
            Some(p) => (p.clone(), 0),
            None => wait_for_port(600, stale)?,
        };
        match serve(&p, &work, total) {
            Ok(()) => {}
            Err(e) => println!("mockusb: link down ({e}); waiting for the device again"),
        }
        stale = devnum;
    }
}
