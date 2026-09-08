//! USB host: model image on a USB stick, bulk-only mass storage over the
//! USBHS host driver (`hal::usbhs::host`).
//!
//! The stick is powered by the 5 V the DK feeds into the connector's VBUS
//! (see README wiring). If VBUS is absent, init() fails in ~100 ms with
//! -600 and the storage layer falls back to the SD card.
//!
//! This module is the transport above the host driver: enumeration at
//! address 0, the mass-storage interface, the bulk-only (BOT) command
//! phases with their recovery ceremonies, the SCSI commands the model
//! image needs, and the split-phase block reads that let the CPU work
//! while a payload lands. Polled throughout, DWT-timed deadlines,
//! cumulative transfer stats.

use core::ptr::{addr_of, addr_of_mut};

use rtt_target::rprintln;

use crate::hal::usbhs::host::{Direction, Endpoint, EndpointType, Host, Pid, Transfer, TransferError};
use crate::hal::usbhs::{delay_ms, elapsed_ms, now, Error};
use crate::usbproto as proto;

pub const BLOCK: usize = 512;

/// The host driver, built on the first probe from the board's USBHS
/// singleton and kept for the rest of the run.
static mut HOST: Option<Host<'static>> = None;

fn host() -> Option<&'static mut Host<'static>> {
    let slot = unsafe { &mut *addr_of_mut!(HOST) };
    if slot.is_none() {
        let usb = crate::board::get().usbhs.take()?;
        *slot = Some(Host::new_blocking(usb));
    }
    slot.as_mut()
}

/// The driver once init() has built it.
fn h() -> &'static mut Host<'static> {
    host().expect("usb host driver")
}

fn code(e: Error) -> i32 {
    match e {
        Error::NoVbus => -600,
        Error::Xo24mTimeout => -601,
        Error::CoreNotResponding => -602,
        Error::AhbNotIdle => -603,
        Error::ResetTimeout => -604,
        Error::ModeRefused => -605,
    }
}

fn xfer_code(e: TransferError) -> i32 {
    match e {
        TransferError::Timeout => -650,
        TransferError::Stall => -651,
        TransferError::Transaction => -652,
        TransferError::Babble => -653,
        TransferError::DataToggle => -654,
        TransferError::Ahb => -655,
        TransferError::FrameOverrun => -656,
        TransferError::Unknown => -657,
    }
}

// --- state -------------------------------------------------------------------

struct Dev {
    addr: u8,
    mps0: u16,
    msc: proto::MscIface,
    /// Next PID per bulk direction (the core reports the follow-on PID
    /// after each transfer; carrying it is the data toggle).
    pid_in: Pid,
    pid_out: Pid,
    tag: u32,
    ready: bool,
}

const DEV_INIT: Dev = Dev {
    addr: 0,
    mps0: 64,
    msc: proto::MscIface {
        cfg_value: 0,
        ifnum: 0,
        ep_in: 0,
        ep_out: 0,
        mps_in: 0,
        mps_out: 0,
    },
    pid_in: Pid::Data0,
    pid_out: Pid::Data0,
    tag: 0,
    ready: false,
};

static mut DEV: Dev = DEV_INIT;

/// DMA staging: control-IN data, CSW, and the unaligned-destination
/// fallback all land here first. Word alignment satisfies HCDMA.
#[repr(C, align(4))]
struct Bounce([u8; BLOCK]);
static mut BOUNCE: Bounce = Bounce([0; BLOCK]);

#[repr(C, align(4))]
struct Cbw([u8; 32]);
static mut CBW: Cbw = Cbw([0; 32]);

/// CSW staging. MUST be distinct from BOUNCE: small SCSI responses
/// (capacity, sense, inquiry) are DMA'd into BOUNCE and the CSW read
/// follows in the same command -- sharing the buffer clobbers the first
/// 13 bytes of the response (hardware-observed: READ CAPACITY parsed the
/// CSW tag as a 64 MB block size). Full MPS of room because IN transfers
/// are programmed in whole-packet multiples.
#[repr(C, align(4))]
struct Csw([u8; BLOCK]);
static mut CSWBUF: Csw = Csw([0; BLOCK]);

#[repr(C, align(4))]
struct SetupBuf([u8; 8]);
static mut SETUP: SetupBuf = SetupBuf([0; 8]);

// Cumulative transfer accounting, same contract as sd::stats_take.
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

fn bounce_addr() -> u32 {
    addr_of_mut!(BOUNCE) as u32
}

fn dev() -> &'static mut Dev {
    unsafe { &mut *addr_of_mut!(DEV) }
}

// --- endpoints ---------------------------------------------------------------

fn ep0(direction: Direction) -> Endpoint {
    let d = dev();
    Endpoint {
        device_address: d.addr,
        number: 0,
        direction,
        ep_type: EndpointType::Control,
        max_packet_size: d.mps0,
    }
}

fn bulk_ep(dir_in: bool) -> Endpoint {
    let d = dev();
    let (number, max_packet_size) = if dir_in {
        (d.msc.ep_in, d.msc.mps_in)
    } else {
        (d.msc.ep_out, d.msc.mps_out)
    };
    Endpoint {
        device_address: d.addr,
        number,
        direction: if dir_in { Direction::In } else { Direction::Out },
        ep_type: EndpointType::Bulk,
        max_packet_size,
    }
}

/// Control transfer on endpoint 0. `dlen` bytes of IN data land in BOUNCE
/// (callers copy out); OUT data stages are not needed by this driver.
fn control_in(sp: [u8; 8], dlen: usize) -> i32 {
    let mps = dev().mps0 as usize;
    unsafe {
        (*addr_of_mut!(SETUP)).0 = sp;
    }
    let sp_addr = addr_of!(SETUP) as u32;
    if let Err(e) = h().transfer(&ep0(Direction::Out), Pid::Setup, sp_addr, 8, 200) {
        return xfer_code(e);
    }
    if dlen > 0 {
        debug_assert!(dlen.div_ceil(mps) * mps <= BLOCK);
        let rounded = dlen.div_ceil(mps) * mps;
        if let Err(e) = h().transfer(&ep0(Direction::In), Pid::Data1, bounce_addr(), rounded, 500) {
            return xfer_code(e);
        }
    }
    // Status stage: opposite direction of the data stage (OUT here), or IN
    // for a no-data request. Zero length, always DATA1.
    let status = if dlen == 0 { Direction::In } else { Direction::Out };
    match h().transfer(&ep0(status), Pid::Data1, bounce_addr(), 0, 200) {
        Ok(_) => 0,
        Err(e) => xfer_code(e),
    }
}

fn bulk(dir_in: bool, dma: u32, len: usize, to_ms: u32) -> i32 {
    let t = bulk_arm(dir_in, dma, len);
    loop {
        if let Some(rc) = bulk_check(dir_in, &t, to_ms) {
            return rc;
        }
    }
}

/// Arm a bulk transfer on the MSC endpoint of `dir_in`.
fn bulk_arm(dir_in: bool, dma: u32, len: usize) -> Transfer {
    let d = dev();
    let pid = if dir_in { d.pid_in } else { d.pid_out };
    h().start_transfer(&bulk_ep(dir_in), pid, dma, len)
}

/// Completion check for `bulk_arm`; carries the data toggle forward on
/// success. None while the transfer is still running.
fn bulk_check(dir_in: bool, t: &Transfer, to_ms: u32) -> Option<i32> {
    match h().poll_transfer(t, to_ms)? {
        Ok(next) => {
            let d = dev();
            if dir_in {
                d.pid_in = next;
            } else {
                d.pid_out = next;
            }
            Some(0)
        }
        Err(e) => Some(xfer_code(e)),
    }
}

/// CLEAR_FEATURE(ENDPOINT_HALT) after a bulk STALL; the endpoint toggle
/// resets to DATA0 on both sides.
fn clear_halt(dir_in: bool) {
    let d = dev();
    let ep = if dir_in { d.msc.ep_in as u16 | 0x80 } else { d.msc.ep_out as u16 };
    let rc = control_in(
        proto::setup(0x02, proto::REQ_CLEAR_FEATURE, proto::FEATURE_ENDPOINT_HALT, ep, 0),
        0,
    );
    if rc == 0 {
        let d = dev();
        if dir_in {
            d.pid_in = Pid::Data0;
        } else {
            d.pid_out = Pid::Data0;
        }
    }
}

// --- bulk-only transport -----------------------------------------------------

/// Largest data-phase chunk per channel run: PktCnt is 10 bits, so at most
/// 1023 max-size packets. Full blocks only, so both MPS cases divide evenly.
fn chunk_bytes(mps: usize) -> usize {
    (1023 * mps / BLOCK) * BLOCK
}

/// One BOT command: CBW, data phase (dlen bytes at dma, direction dir_in),
/// CSW. Returns 0, the SCSI-failed marker -670 (sense data pending), or a
/// transport error -- after which the device's BOT state machine has been
/// realigned so the NEXT command starts clean.
fn bot(cb: &[u8], dir_in: bool, dma: u32, dlen: usize, data_to_ms: u32) -> i32 {
    let rc = bot_inner(cb, dir_in, dma, dlen, data_to_ms);
    if rc != 0 && rc != -670 {
        bot_recover();
    }
    rc
}

fn bot_inner(cb: &[u8], dir_in: bool, dma: u32, dlen: usize, data_to_ms: u32) -> i32 {
    let d = dev();
    d.tag = d.tag.wrapping_add(1);
    let tag = d.tag;
    unsafe {
        proto::build_cbw(&mut (*addr_of_mut!(CBW)).0, tag, dlen as u32, dir_in, cb);
    }
    let rc = bulk(false, addr_of!(CBW) as u32, proto::CBW_LEN, 500);
    if rc != 0 {
        return rc;
    }
    let mps = if dir_in { d.msc.mps_in } else { d.msc.mps_out } as usize;
    let mut off = 0usize;
    while off < dlen {
        let n = (dlen - off).min(chunk_bytes(mps));
        // IN transfers must be programmed in whole packets; the buffer has
        // the rounded-up room (BOUNCE for short SCSI responses, whole
        // blocks for data). A short packet ends the transfer early.
        let prog = if dir_in { n.div_ceil(mps) * mps } else { n };
        let rc = bulk(dir_in, dma + off as u32, prog, data_to_ms);
        if rc == -651 {
            // Data-phase STALL: the device truncated the transfer (normal
            // for short SCSI responses). Recover the endpoint and read the
            // CSW, which still carries the command status.
            clear_halt(dir_in);
            break;
        }
        if rc != 0 {
            return rc;
        }
        off += n;
    }
    csw_phase(tag)
}

/// CSW on bulk IN, one retry after a STALL (BOT 1.0, 6.7.2), checked
/// against the command's tag.
fn csw_phase(tag: u32) -> i32 {
    let mps_in = dev().msc.mps_in as usize;
    let csw_addr = addr_of_mut!(CSWBUF) as u32;
    let mut rc = bulk(true, csw_addr, mps_in, 500);
    if rc == -651 {
        clear_halt(true);
        rc = bulk(true, csw_addr, mps_in, 500);
    }
    if rc != 0 {
        return rc;
    }
    let csw = unsafe { &(&(*addr_of!(CSWBUF)).0)[..proto::CSW_LEN] };
    match proto::check_csw(csw, tag) {
        Ok(0) => 0,
        Ok(1) => -670,
        Ok(_) => -671, // phase error: device wants a reset
        Err(e) => e,
    }
}

/// Bulk-only mass storage reset + endpoint recovery (BOT 1.0, 5.3.4).
/// A timed-out data phase kills our channel but leaves the DEVICE's BOT
/// state machine mid-command; without this, every later transfer fails
/// against a desynced stick (hardware-observed after the first slow
/// write burst).
fn bot_recover() {
    let ifnum = dev().msc.ifnum as u16;
    let _ = control_in(proto::setup(0x21, proto::REQ_MSC_RESET, 0, ifnum, 0), 0);
    clear_halt(true);
    clear_halt(false);
}

/// Read the sense data after a -670 so the stick can clear its UNIT
/// ATTENTION state; returns the sense key (or an error).
fn request_sense() -> i32 {
    let rc = bot(&proto::cdb_request_sense(18), true, bounce_addr(), 18, 500);
    if rc != 0 && rc != -670 {
        return rc;
    }
    (unsafe { (*addr_of!(BOUNCE)).0[2] } & 0x0F) as i32
}

// --- bring-up ----------------------------------------------------------------

/// Bring up the port, enumerate the stick, and get its SCSI unit ready.
/// Returns 0 or a negative stage-tagged error (-600 = no VBUS: nothing is
/// wired, the caller should fall back to SD quietly).
pub fn init() -> i32 {
    *dev() = DEV_INIT;

    let Some(host) = host() else {
        // The peripheral is owned by the device-mode driver (mock rig).
        return -606;
    };
    if let Err(e) = host.power_up() {
        if e == Error::ModeRefused {
            let (mode, hw) = host.otg_mode();
            rprintln!("usb: host mode refused (GHWCFG2={:#010x} OTGMODE={})", hw, mode);
            host.power_down();
        }
        return code(e);
    }

    host.port_power(true);
    // Connect detection: the stick is expected to be attached already, so
    // a short window is enough (and keeps a wedge-free mailbox budget).
    if !host.wait_connect(400) {
        rprintln!("usb: VBUS present but no device on the port");
        host.power_down();
        return -610;
    }
    delay_ms(100); // attach debounce (USB 2.0, 7.1.7.3)

    let speed = match host.port_reset() {
        Ok(s) => s,
        Err(_) => {
            host.power_down();
            return -611;
        }
    };
    use crate::hal::usbhs::host::Speed;
    let speed_code = match speed {
        Speed::High => 0,
        Speed::Full => 1,
        Speed::Low => 2,
    };
    rprintln!("usb: port enabled, speed {} (0=HS 1=FS)", speed_code);
    if speed == Speed::Low {
        rprintln!("usb: low-speed device is not a stick");
        host.power_down();
        return -612;
    }
    delay_ms(20); // reset recovery

    // Enumeration at address 0. High speed fixes MPS0 at 64; full speed
    // reports it in byte 7 of the first descriptor read.
    let mut rc = control_in(
        proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_DEVICE, 0, 8),
        8,
    );
    if rc != 0 {
        // One retry: some sticks reject the very first transaction after
        // reset while their firmware is still settling.
        delay_ms(20);
        rc = control_in(
            proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_DEVICE, 0, 8),
            8,
        );
    }
    if rc != 0 {
        rprintln!("usb: first descriptor read failed rc={}", rc);
        h().power_down();
        return -620;
    }
    {
        let d = dev();
        d.mps0 = unsafe { (*addr_of!(BOUNCE)).0[7] } as u16;
        if d.mps0 < 8 {
            h().power_down();
            return -621;
        }
    }

    if control_in(proto::setup(0x00, proto::REQ_SET_ADDRESS, 1, 0, 0), 0) != 0 {
        h().power_down();
        return -622;
    }
    dev().addr = 1;
    delay_ms(5); // SET_ADDRESS recovery (USB 2.0, 9.2.6.3)

    if control_in(
        proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_DEVICE, 0, 18),
        18,
    ) != 0
    {
        h().power_down();
        return -623;
    }
    let (vid, pid) = unsafe {
        let b = &(*addr_of!(BOUNCE)).0;
        (
            u16::from_le_bytes([b[8], b[9]]),
            u16::from_le_bytes([b[10], b[11]]),
        )
    };

    if control_in(
        proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_CONFIG, 0, 9),
        9,
    ) != 0
    {
        h().power_down();
        return -624;
    }
    let total = unsafe {
        let b = &(*addr_of!(BOUNCE)).0;
        (u16::from_le_bytes([b[2], b[3]]) as usize).min(BLOCK - 64)
    };
    if total < 9
        || control_in(
            proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_CONFIG, 0, total as u16),
            total,
        ) != 0
    {
        h().power_down();
        return -625;
    }
    let msc = {
        let cfg = unsafe { &(&(*addr_of!(BOUNCE)).0)[..total] };
        match proto::parse_config(cfg) {
            Ok(m) => m,
            Err(e) => {
                rprintln!("usb: no mass-storage interface ({})", e);
                h().power_down();
                return -626;
            }
        }
    };
    dev().msc = msc;

    if control_in(
        proto::setup(0x00, proto::REQ_SET_CONFIGURATION, msc.cfg_value as u16, 0, 0),
        0,
    ) != 0
    {
        h().power_down();
        return -627;
    }
    delay_ms(5);

    // SCSI bring-up: sticks report a power-on UNIT ATTENTION until a
    // REQUEST SENSE collects it, and a stick that was reset mid-command
    // (previous session interrupted) can take seconds of internal
    // recovery before the LUN is ready -- hardware-observed after a
    // reflash landed mid-utterance. Budget accordingly; every wait in
    // here is DWT-bounded, so no watchdog rides along.
    let start = now();
    let mut last_sense = -1;
    loop {
        let rc = bot(&proto::cdb_test_unit_ready(), false, bounce_addr(), 0, 500);
        if rc == 0 {
            break;
        }
        if rc == -670 || rc == -671 {
            let key = request_sense();
            if key != last_sense {
                rprintln!("usb: unit not ready (sense key {})", key);
                last_sense = key;
            }
        }
        if elapsed_ms(start, 5000) {
            rprintln!("usb: unit never became ready (last sense key {})", last_sense);
            h().power_down();
            return -640;
        }
        delay_ms(20);
    }

    if bot(&proto::cdb_read_capacity10(), true, bounce_addr(), 8, 500) != 0 {
        h().power_down();
        return -641;
    }
    let (last_lba, blklen) = unsafe {
        let b = &(*addr_of!(BOUNCE)).0;
        (
            u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
            u32::from_be_bytes([b[4], b[5], b[6], b[7]]),
        )
    };
    if blklen != BLOCK as u32 {
        rprintln!("usb: stick block size {} unsupported", blklen);
        h().power_down();
        return -642;
    }

    // INQUIRY is cosmetic; failure does not gate readiness.
    let name_ok = bot(&proto::cdb_inquiry(36), true, bounce_addr(), 36, 500) == 0;
    dev().ready = true;
    rprintln!(
        "usb: {} stick {:04x}:{:04x}, {} MB",
        if speed == Speed::High { "high-speed" } else { "full-speed" },
        vid,
        pid,
        ((last_lba as u64 + 1) * BLOCK as u64) >> 20
    );
    if name_ok {
        let b = unsafe { &(*addr_of!(BOUNCE)).0 };
        let mut name = [b' '; 24];
        for (i, n) in name.iter_mut().enumerate() {
            let c = b[8 + i];
            *n = if (0x20..0x7F).contains(&c) { c } else { b'?' };
        }
        rprintln!("usb: \"{}\"", core::str::from_utf8(&name).unwrap_or("?"));
    }
    0
}

// --- block interface (same contract as sd.rs) --------------------------------

pub fn read_blocks(lba: u32, dst: *mut u8, count: u32) -> i32 {
    let t0 = now();
    let rc = rw_blocks(lba, dst as u32, count, true);
    unsafe {
        RD_CYC += now().wrapping_sub(t0) as u64;
        RD_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

pub fn write_blocks(lba: u32, src: *const u8, count: u32) -> i32 {
    let t0 = now();
    let rc = rw_blocks(lba, src as u32, count, false);
    unsafe {
        WR_CYC += now().wrapping_sub(t0) as u64;
        WR_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

fn rw_blocks(lba: u32, buf: u32, count: u32, read: bool) -> i32 {
    if !dev().ready {
        return -660;
    }
    if count == 0 {
        return 0;
    }
    if buf & 3 != 0 {
        // HCDMA wants word alignment; stage block-by-block. No caller
        // does this on a hot path.
        for i in 0..count {
            let rc = if read {
                let rc = rw_aligned(lba + i, bounce_addr(), 1, true);
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        bounce_addr() as *const u8,
                        (buf + i * BLOCK as u32) as *mut u8,
                        BLOCK,
                    );
                }
                rc
            } else {
                unsafe {
                    core::ptr::copy_nonoverlapping(
                        (buf + i * BLOCK as u32) as *const u8,
                        bounce_addr() as *mut u8,
                        BLOCK,
                    );
                }
                rw_aligned(lba + i, bounce_addr(), 1, false)
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
    // READ(10)/WRITE(10) carry a 16-bit block count; every caller is far
    // below that, but split anyway rather than trust it.
    let mut done = 0u32;
    while done < count {
        let n = (count - done).min(65_535);
        let bytes = n as usize * BLOCK;
        let cdb: [u8; 10] = if read {
            proto::cdb_read10(lba + done, n as u16)
        } else {
            proto::cdb_write10(lba + done, n as u16)
        };
        // Budgets differ by direction: reads stream at bus speed, but
        // cheap flash controllers stall for SECONDS on a write burst
        // (mapping-table rebuilds, GC; hardware-observed >340 ms on the
        // very first write), and a large stick can take over half a
        // second to serve its very first read after enumeration
        // (hardware-observed: a 128 GB stick timed out the 8 KB index
        // read at 516 ms). All waits stay DWT-bounded.
        let to_ms = if read {
            2000 + (bytes >> 10) as u32 * 2
        } else {
            3000 + (bytes >> 10) as u32 * 4
        };
        let rc = bot(&cdb, read, buf + done * BLOCK as u32, bytes, to_ms);
        if rc == -670 {
            request_sense();
            return if read { -661 } else { -662 };
        }
        if rc != 0 {
            return rc;
        }
        done += n;
    }
    0
}

// --- split-phase block interface -----------------------------------------------
//
// One READ(10)/WRITE(10) whose data phase runs on the channel's DMA while
// the CPU does something else: `start` sends the CBW and arms the first
// data chunk, `poll` advances the command (next chunk, CSW) whenever the
// caller checks in, `finish` blocks for the rest. The BOT transport is
// strictly one command at a time, so there is at most one in flight;
// storage.rs enforces that above this layer. Cycle accounting counts only
// the time the CPU spent inside these calls, i.e. the storage stall that
// the overlap failed to hide.

struct AsyncOp {
    active: bool,
    read: bool,
    buf: u32,
    bytes: usize,
    off: usize,
    /// Bytes programmed for the running data chunk.
    prog: usize,
    xfer: Transfer,
    to_ms: u32,
    tag: u32,
    rc: i32,
    stage: Stage,
    /// One CSW re-read after a STALL (BOT 1.0, 6.7.2) has been used.
    csw_retried: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Data,
    Csw,
    Done,
}

static mut ASYNC: AsyncOp = AsyncOp {
    active: false,
    read: false,
    buf: 0,
    bytes: 0,
    off: 0,
    prog: 0,
    xfer: Transfer::NONE,
    to_ms: 0,
    tag: 0,
    rc: 0,
    stage: Stage::Done,
    csw_retried: false,
};

fn async_op() -> &'static mut AsyncOp {
    unsafe { &mut *addr_of_mut!(ASYNC) }
}

pub fn read_start(lba: u32, dst: *mut u8, count: u32) -> i32 {
    async_start(lba, dst as u32, count, true)
}

pub fn write_start(lba: u32, src: *const u8, count: u32) -> i32 {
    async_start(lba, src as u32, count, false)
}

fn async_start(lba: u32, buf: u32, count: u32, read: bool) -> i32 {
    let a = async_op();
    if a.active {
        return -495;
    }
    let t0 = now();
    // Cases the split path does not cover run synchronously and report
    // through the same finish(): unaligned buffers, empty and oversized
    // requests, and a stick that is not there.
    if buf & 3 != 0 || count == 0 || count > 65_535 || !dev().ready {
        a.rc = rw_blocks(lba, buf, count, read);
        a.active = true;
        a.stage = Stage::Done;
        a.read = read;
        async_account(read, count, t0);
        return 0;
    }
    let bytes = count as usize * BLOCK;
    let cdb: [u8; 10] = if read {
        proto::cdb_read10(lba, count as u16)
    } else {
        proto::cdb_write10(lba, count as u16)
    };
    let d = dev();
    d.tag = d.tag.wrapping_add(1);
    let tag = d.tag;
    unsafe {
        proto::build_cbw(&mut (*addr_of_mut!(CBW)).0, tag, bytes as u32, read, &cdb);
    }
    let rc = bulk(false, addr_of!(CBW) as u32, proto::CBW_LEN, 500);
    *a = AsyncOp {
        active: true,
        read,
        buf,
        bytes,
        off: 0,
        prog: 0,
        xfer: Transfer::NONE,
        to_ms: if read {
            2000 + (bytes >> 10) as u32 * 2
        } else {
            3000 + (bytes >> 10) as u32 * 4
        },
        tag,
        rc,
        stage: Stage::Data,
        csw_retried: false,
    };
    if rc != 0 {
        bot_recover();
        a.stage = Stage::Done; // xfer_finish() reports rc
    } else {
        async_arm_chunk();
    }
    async_account(read, count, t0);
    0
}

fn async_account(read: bool, count: u32, t0: u32) {
    let dt = now().wrapping_sub(t0) as u64;
    unsafe {
        if read {
            RD_CYC += dt;
            RD_BYTES += count as u64 * BLOCK as u64;
        } else {
            WR_CYC += dt;
            WR_BYTES += count as u64 * BLOCK as u64;
        }
    }
}

fn async_arm_chunk() {
    let a = async_op();
    let d = dev();
    let mps = if a.read { d.msc.mps_in } else { d.msc.mps_out } as usize;
    let n = (a.bytes - a.off).min(chunk_bytes(mps));
    a.prog = n;
    a.xfer = bulk_arm(a.read, a.buf + a.off as u32, n);
}

/// Bytes of the pending read known to be in memory: the finished chunks
/// plus what the channel has written of the current one, behind a
/// two-packet margin for a write still on its way through the bus.
/// Meaningful while xfer_poll() is false.
pub fn xfer_landed() -> usize {
    let a = async_op();
    if !a.active {
        return 0;
    }
    match a.stage {
        Stage::Data if a.read => {
            let left = h().bytes_remaining();
            let mps = dev().msc.mps_in as usize;
            (a.off + a.prog.saturating_sub(left)).saturating_sub(2 * mps).min(a.bytes)
        }
        _ => a.bytes,
    }
}

/// True when the command has run to completion (rc ready for xfer_finish()).
pub fn xfer_poll() -> bool {
    let read = async_op().read;
    let t0 = now();
    let done = poll_inner();
    async_account(read, 0, t0);
    done
}

fn poll_inner() -> bool {
    let a = async_op();
    if !a.active || a.stage == Stage::Done {
        return true;
    }
    let dir_in = a.stage == Stage::Csw || a.read;
    let xfer = a.xfer;
    let rc = match bulk_check(dir_in, &xfer, a.to_ms) {
        None => return false,
        Some(rc) => rc,
    };
    match a.stage {
        Stage::Data => {
            if rc == 0 {
                a.off += a.prog;
                if a.off < a.bytes {
                    async_arm_chunk();
                    return false;
                }
            } else if rc == -651 {
                // Data-phase STALL: the device truncated the transfer; the
                // CSW still carries the status.
                clear_halt(a.read);
            } else {
                return async_done(rc);
            }
            async_arm_csw();
            false
        }
        Stage::Csw => {
            if rc == -651 && !a.csw_retried {
                a.csw_retried = true;
                clear_halt(true);
                async_arm_csw();
                return false;
            }
            if rc != 0 {
                return async_done(rc);
            }
            let csw = unsafe { &(&(*addr_of!(CSWBUF)).0)[..proto::CSW_LEN] };
            let rc = match proto::check_csw(csw, a.tag) {
                Ok(0) => 0,
                Ok(1) => {
                    request_sense();
                    if a.read { -661 } else { -662 }
                }
                Ok(_) => -671, // phase error: device wants a reset
                Err(e) => e,
            };
            if rc != 0 && rc != -661 && rc != -662 {
                bot_recover();
            }
            let a = async_op();
            a.rc = rc;
            a.stage = Stage::Done;
            true
        }
        Stage::Done => true,
    }
}

/// CSW on bulk IN: the command's status, read on the channel's DMA like
/// the data so a slow device's completion wait is not a CPU stall.
fn async_arm_csw() {
    let a = async_op();
    let mps_in = dev().msc.mps_in as usize;
    a.xfer = bulk_arm(true, addr_of_mut!(CSWBUF) as u32, mps_in);
    a.to_ms = 500;
    a.stage = Stage::Csw;
}

/// Transport error: realign the device's BOT state machine and report.
fn async_done(rc: i32) -> bool {
    let a = async_op();
    a.rc = rc;
    a.stage = Stage::Done;
    bot_recover();
    true
}

/// Block until the in-flight command is done; returns its result. A call
/// with nothing in flight returns 0.
pub fn xfer_finish() -> i32 {
    let a = async_op();
    if !a.active {
        return 0;
    }
    let read = a.read;
    let t0 = now();
    while !poll_inner() {}
    async_account(read, 0, t0);
    let a = async_op();
    a.active = false;
    a.rc
}
