//! Mock block device: the model image lives in a file on a PC, which
//! serves 512-byte blocks over the CDC-ACM link in usbdev.rs.
//!
//! This exists so the whole standalone pipeline can be exercised on real
//! silicon without a USB stick in the loop: the image can be rebuilt and
//! re-served in a second (no dd, no unplugging), the audio the device
//! "records" can be a fixed clip written into the image, and every scratch
//! block the device writes lands in the host's file where the comparison
//! tooling can read it. Same 512-byte block contract as sd.rs and usb.rs,
//! so nothing above storage.rs knows the difference.
//!
//! Framing: every exchange is a whole number of 512-byte units, so a short
//! packet never appears mid-stream and the two sides cannot slip out of
//! step. The device is always the initiator -- a USB device cannot start a
//! transfer, so the host daemon sits in a read loop waiting for requests.
//!
//!     device -> host   request frame (512 B)   [+ payload for WRITE]
//!     host   -> device response frame (512 B)  [+ payload for READ]

use core::ptr::{addr_of, addr_of_mut};

use rtt_target::rprintln;

use crate::usbdev;

pub const BLOCK: usize = 512;

const REQ_MAGIC: u32 = 0x514B_4C42; // "BLKQ"
const RSP_MAGIC: u32 = 0x524B_4C42; // "BLKR"

const OP_INFO: u8 = 3;
const OP_READ: u8 = 1;
const OP_WRITE: u8 = 2;

/// Enumeration budget. The daemon is normally already waiting, but a cold
/// PC-side start (or a `cargo run` of the daemon) needs a moment.
const ENUM_MS: u32 = 8000;
/// How long the device holds a handshake request open waiting for the
/// daemon to open the port. The send blocks until the host starts polling,
/// so this is really "how long to wait for the daemon to start".
const HANDSHAKE_SEND_MS: u32 = 20_000;
/// Once the request is away the daemon answers immediately, so a short
/// wait here just means a mangled first attempt (see tools/mockusb
/// tty.rs: the daemon flushes whatever arrived before it could put the
/// port in raw mode) is retried in seconds rather than tens of them.
const HANDSHAKE_REPLY_MS: u32 = 2500;

#[repr(C, align(4))]
struct Frame([u8; BLOCK]);

static mut REQ: Frame = Frame([0; BLOCK]);
static mut RSP: Frame = Frame([0; BLOCK]);
/// Unaligned-caller staging, same role as usb.rs's BOUNCE.
static mut BOUNCE: Frame = Frame([0; BLOCK]);

/// Monotonic request tag, echoed by the daemon. A response carrying the
/// wrong tag means the two sides have slipped a frame apart, which is
/// worth saying out loud rather than quietly acting on the wrong answer.
static mut TAG: u32 = 0;
static mut READY: bool = false;
static mut BLOCKS: u32 = 0;

static mut RD_BYTES: u64 = 0;
static mut RD_CYC: u64 = 0;
static mut WR_BYTES: u64 = 0;
static mut WR_CYC: u64 = 0;

pub fn stats_take() -> (u64, u64, u64, u64) {
    unsafe {
        let s = (RD_BYTES, RD_CYC, WR_BYTES, WR_CYC);
        RD_BYTES = 0;
        RD_CYC = 0;
        WR_BYTES = 0;
        WR_CYC = 0;
        s
    }
}

/// Additive word checksum over a block run. Cheap enough to run on every
/// transfer (a whole utterance moves a few hundred MB, costing ~1 s total)
/// and it turns a framing slip into a loud error instead of silently wrong
/// weights -- which is the failure this rig exists to hunt.
fn sum32(addr: u32, bytes: usize) -> u32 {
    let mut s = 0u32;
    let p = addr as *const u32;
    for i in 0..bytes / 4 {
        s = s
            .rotate_left(1)
            .wrapping_add(unsafe { core::ptr::read_volatile(p.add(i)) });
    }
    s
}

fn put32(buf: &mut [u8], off: usize, v: u32) {
    buf[off..off + 4].copy_from_slice(&v.to_le_bytes());
}

fn get32(buf: &[u8], off: usize) -> u32 {
    u32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]])
}

/// Build and send a request frame.
fn request(op: u8, lba: u32, count: u32, sum: u32, to_ms: u32) -> i32 {
    unsafe {
        let f = &mut (*addr_of_mut!(REQ)).0;
        f.fill(0);
        put32(f, 0, REQ_MAGIC);
        f[4] = op;
        put32(f, 8, lba);
        put32(f, 12, count);
        put32(f, 16, sum);
        TAG = TAG.wrapping_add(1);
        put32(f, 20, TAG);
    }
    usbdev::send(addr_of!(REQ) as u32, BLOCK, to_ms)
}

/// Read the response frame; returns (status, count, sum).
fn response(to_ms: u32) -> Result<(i32, u32, u32), i32> {
    let rc = usbdev::recv(addr_of_mut!(RSP) as u32, BLOCK, to_ms);
    if rc != 0 {
        return Err(rc);
    }
    let f = unsafe { &(*addr_of!(RSP)).0 };
    if get32(f, 0) != RSP_MAGIC {
        rprintln!("mock: bad response magic {:#010x}", get32(f, 0));
        return Err(-640);
    }
    let tag = get32(f, 16);
    if tag != unsafe { TAG } {
        rprintln!("mock: response tag {} != {} (stream desync)", tag, unsafe { TAG });
        return Err(-648);
    }
    Ok((get32(f, 4) as i32, get32(f, 8), get32(f, 12)))
}

/// Bring up the link and ask the daemon what it is serving.
pub fn init() -> i32 {
    unsafe { READY = false };
    let rc = usbdev::init(ENUM_MS);
    if rc != 0 {
        return rc;
    }
    // One patient exchange, never abandoned. The daemon can only find the
    // port after enumeration, so it routinely starts second -- but the
    // request simply sits in the endpoint until the host opens the port
    // and begins polling. Retrying instead would leave the daemon's reply
    // to a gave-up-on request in the pipe, where the next request reads it
    // as its own answer.
    rprintln!("mock: waiting for the host daemon (tools/mockusb serve)...");
    let mut last = -647;
    for attempt in 0..5 {
        if attempt > 0 {
            // Anything the daemon sent in answer to the attempt we just
            // gave up on has to go before the next one, or it would be
            // read as the new reply.
            usbdev::drain();
        }
        let rc = request(OP_INFO, 0, 0, 0, HANDSHAKE_SEND_MS);
        if rc != 0 {
            last = rc;
            continue;
        }
        match response(HANDSHAKE_REPLY_MS) {
            Ok((0, blocks, _)) => {
                unsafe {
                    BLOCKS = blocks;
                    READY = true;
                }
                rprintln!(
                    "mock: host image ready, {} blocks ({} MB)",
                    blocks,
                    (blocks as u64 * BLOCK as u64) >> 20
                );
                return 0;
            }
            Ok((st, _, _)) => {
                rprintln!("mock: daemon refused INFO ({})", st);
                return -641;
            }
            Err(e) => last = e,
        }
    }
    rprintln!("mock: no daemon answered ({})", last);
    usbdev::diag("handshake-failed");
    last
}

pub fn read_blocks(lba: u32, dst: *mut u8, count: u32) -> i32 {
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let rc = rw(lba, dst as u32, count, true);
    unsafe {
        RD_CYC += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
        RD_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

pub fn write_blocks(lba: u32, src: *const u8, count: u32) -> i32 {
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let rc = rw(lba, src as u32, count, false);
    unsafe {
        WR_CYC += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
        WR_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

fn rw(lba: u32, buf: u32, count: u32, read: bool) -> i32 {
    if !unsafe { core::ptr::read_volatile(addr_of!(READY)) } {
        return -642;
    }
    if count == 0 {
        return 0;
    }
    if buf & 3 != 0 {
        // Endpoint DMA wants word alignment; stage block by block. No
        // caller does this on a hot path.
        let bo = addr_of_mut!(BOUNCE) as u32;
        for i in 0..count {
            let dst = buf + i * BLOCK as u32;
            let rc = if read {
                let rc = rw_aligned(lba + i, bo, 1, true);
                unsafe {
                    core::ptr::copy_nonoverlapping(bo as *const u8, dst as *mut u8, BLOCK)
                };
                rc
            } else {
                unsafe {
                    core::ptr::copy_nonoverlapping(dst as *const u8, bo as *mut u8, BLOCK)
                };
                rw_aligned(lba + i, bo, 1, false)
            };
            if rc != 0 {
                return rc;
            }
        }
        return 0;
    }
    rw_aligned(lba, buf, count, read)
}

fn rw_aligned(lba: u32, buf: u32, count: u32, read: bool) -> i32 {
    let bytes = count as usize * BLOCK;
    // A PC serving a page-cached file answers in microseconds; the budget
    // only has to cover a cold file and the tty round trip.
    let to_ms = 3000 + (bytes >> 10) as u32 * 2;

    if read {
        let rc = request(OP_READ, lba, count, 0, to_ms);
        if rc != 0 {
            return rc;
        }
        let (st, n, sum) = match response(to_ms) {
            Ok(v) => v,
            Err(e) => return e,
        };
        if st != 0 {
            rprintln!("mock: read lba {} x{} refused ({})", lba, count, st);
            return -643;
        }
        if n != count {
            return -644;
        }
        let rc = usbdev::recv(buf, bytes, to_ms);
        if rc != 0 {
            return rc;
        }
        let got = sum32(buf, bytes);
        if got != sum {
            rprintln!(
                "mock: read lba {} x{} checksum {:#010x} != {:#010x}",
                lba,
                count,
                got,
                sum
            );
            return -645;
        }
        0
    } else {
        let rc = request(OP_WRITE, lba, count, sum32(buf, bytes), to_ms);
        if rc != 0 {
            return rc;
        }
        let rc = usbdev::send(buf, bytes, to_ms);
        if rc != 0 {
            return rc;
        }
        match response(to_ms) {
            Ok((0, _, _)) => 0,
            Ok((st, _, _)) => {
                rprintln!("mock: write lba {} x{} refused ({})", lba, count, st);
                -646
            }
            Err(e) => e,
        }
    }
}
