//! USB host: model image on a USB stick behind the USBHS peripheral.
//!
//! The nRF54LM20's USBHS wraps a Synopsys DWC2 dual-role core. The
//! datasheet's feature list says "USB 2.0 device" and "no OTG", but the
//! hardwired configuration register disagrees on the part that matters:
//! GHWCFG2.OTGMODE = 2 (non-HNP/non-SRP OTG = HOST and device), 16 host
//! channels, internal DMA, and GUSBCFG.FORCEHSTMODE is documented as
//! valid exactly for that OTG mode. The wrapper's SOF publish register
//! even documents its host-mode behavior. So: force host mode, drive the
//! root port directly (no hub support), and speak bulk-only mass storage.
//!
//! Board reality on the DK: the chip has no VBUS sourcing -- the VBUS pin
//! is an INPUT that powers the PHY's signaling rail (VREGUSB). Host mode
//! therefore needs 5 V fed into the nRF USB connector's VBUS from the
//! board's own 5 V rail (see README wiring). The stick is powered by that
//! same 5 V. If VBUS is absent, init() fails in ~100 ms with -600 and the
//! storage layer falls back to the SD card.
//!
//! Driver shape mirrors sd.rs: polled, no interrupts (NVIC line stays
//! disabled; we read HCINT/HPRT directly), DWT-timed deadlines, cumulative
//! transfer stats. Every transfer runs on HOST CHANNEL 0 ONLY: bulk-only
//! transport is strictly sequential, one channel reprogrammed per transfer
//! is enough, and it sidesteps the HC[n] address-stride ambiguity in the
//! datasheet (0x18 per entry where Synopsys hardware canonically decodes
//! 0x20). The core retries NAKs in hardware (buffer DMA mode), so a
//! transfer either completes, errors, or hits our deadline.

use core::ptr::{read_volatile, write_volatile};

use crate::usbproto as proto;
use rtt_target::rprintln;

// --- register bases (secure aliases, datasheet v1.0-3) -----------------------

const CLOCK_BASE: usize = 0x5010_E000;
const CLK_TASKS_XO24MSTART: usize = 0x024;
const CLK_EVENTS_XO24MSTARTED: usize = 0x11C;

const VREG_BASE: usize = 0x5012_1000; // VREGUSB
const VREG_TASKS_START: usize = 0x000;
const VREG_TASKS_STOP: usize = 0x004;
const VREG_EVENTS_VBUSDETECTED: usize = 0x104;

const WRAP_BASE: usize = 0x5005_A000; // USBHS wrapper
const WRAP_TASKS_START: usize = 0x000;
const WRAP_TASKS_STOP: usize = 0x004;
const WRAP_ENABLE: usize = 0x400; // bit0 CORE, bit1 PHY
// The wrapper's STATUS register (0x408) is deliberately unused: its CORE
// ready bit never asserts on this part (hardware-observed) even with the
// core alive; GSNPSID readback is the working readiness check.

const CORE_BASE: usize = 0x5002_0000; // USBHSCORE (DWC2)
const GOTGCTL: usize = 0x000;
pub(crate) const GAHBCFG: usize = 0x008;
pub(crate) const GUSBCFG: usize = 0x00C;
pub(crate) const GRSTCTL: usize = 0x010;
pub(crate) const GINTSTS: usize = 0x014;
pub(crate) const GRXFSIZ: usize = 0x024;
pub(crate) const GNPTXFSIZ: usize = 0x028;
pub(crate) const GSNPSID: usize = 0x040;
pub(crate) const GHWCFG2: usize = 0x048;
const HPTXFSIZ: usize = 0x100;
const HCFG: usize = 0x400;
const HPRT: usize = 0x440;
const HCCHAR0: usize = 0x500;
const HCINT0: usize = 0x508;
const HCINTMSK0: usize = 0x50C;
const HCTSIZ0: usize = 0x510;
const HCDMA0: usize = 0x514;

// GOTGCTL session overrides: the host state machine wants A-session/VBUS
// valid from the PHY; force both so port power does not depend on how the
// wrapper routes the VBUS comparator in host mode.
const OTG_VBVALID_OV: u32 = (1 << 2) | (1 << 3);
const OTG_AVALID_OV: u32 = (1 << 4) | (1 << 5);

pub(crate) const GRSTCTL_CSFTRST: u32 = 1 << 0;
pub(crate) const GRSTCTL_RXFFLSH: u32 = 1 << 4;
pub(crate) const GRSTCTL_TXFFLSH: u32 = 1 << 5;
// HARDWARE-CONFIRMED: this core is DWC2 v5.00b (GSNPSID 0x4F54500B), and
// since v4.20a soft reset is a handshake -- CSftRst does NOT self-clear.
// The core sets CSftRstDone (bit 29, absent from the SVD/datasheet) and
// software must then write both bits back to 0. Polling for self-clear
// hangs forever with the reset long since finished.
pub(crate) const GRSTCTL_CSFTRSTDONE: u32 = 1 << 29;
pub(crate) const GRSTCTL_AHBIDLE: u32 = 1 << 31;
const GUSBCFG_FRCHSTMODE: u32 = 1 << 29;
pub(crate) const GAHBCFG_DMAEN: u32 = 1 << 5;
pub(crate) const GAHBCFG_BURST_INCR4: u32 = 3 << 1;
const GINTSTS_CURMOD_HOST: u32 = 1 << 0;

const HPRT_CONNSTS: u32 = 1 << 0;
const HPRT_CONNDET: u32 = 1 << 1;
const HPRT_ENA: u32 = 1 << 2;
const HPRT_ENCHNG: u32 = 1 << 3;
const HPRT_OCCHNG: u32 = 1 << 5;
const HPRT_RST: u32 = 1 << 8;
const HPRT_PWR: u32 = 1 << 12;
// Write-1-to-clear bits (PRTENA included: writing 1 DISABLES the port);
// every read-modify-write must mask these out.
const HPRT_W1C: u32 = HPRT_CONNDET | HPRT_ENA | HPRT_ENCHNG | HPRT_OCCHNG;

const CH_EPTYPE_CTRL: u32 = 0 << 18;
const CH_EPTYPE_BULK: u32 = 2 << 18;
const CH_CHDIS: u32 = 1 << 30;
const CH_CHENA: u32 = 1 << 31;

const HCI_XFERCOMPL: u32 = 1 << 0;
const HCI_CHHLTD: u32 = 1 << 1;
const HCI_AHBERR: u32 = 1 << 2;
const HCI_STALL: u32 = 1 << 3;
const HCI_XACTERR: u32 = 1 << 7;
const HCI_BBLERR: u32 = 1 << 8;
const HCI_FRMOVRUN: u32 = 1 << 9;
const HCI_DTGLERR: u32 = 1 << 10;

// HCTSIZ.Pid encoding.
const PID_DATA0: u32 = 0;
const PID_DATA1: u32 = 2;
const PID_SETUP: u32 = 3;

pub const BLOCK: usize = 512;

#[inline]
fn clk(off: usize) -> *mut u32 {
    (CLOCK_BASE + off) as *mut u32
}

#[inline]
fn vreg(off: usize) -> *mut u32 {
    (VREG_BASE + off) as *mut u32
}

#[inline]
fn wrap(off: usize) -> *mut u32 {
    (WRAP_BASE + off) as *mut u32
}

#[inline]
pub(crate) fn core_reg(off: usize) -> *mut u32 {
    (CORE_BASE + off) as *mut u32
}

const CYC_PER_MS: u32 = 128_000; // DWT at the 128 MHz core clock

pub(crate) fn ms_wait(ms: u32) {
    let start = cortex_m::peripheral::DWT::cycle_count();
    while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < ms * CYC_PER_MS {}
}

/// Poll `reg` until `(read & mask) == want` or `ms` elapses.
pub(crate) fn poll(reg: *mut u32, mask: u32, want: u32, ms: u32) -> bool {
    let start = cortex_m::peripheral::DWT::cycle_count();
    loop {
        if unsafe { read_volatile(reg) } & mask == want {
            return true;
        }
        if cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) >= ms * CYC_PER_MS {
            return false;
        }
    }
}

// --- state -------------------------------------------------------------------

#[derive(Default)]
struct Dev {
    addr: u32,
    mps0: u32,
    msc: proto::MscIface,
    /// Next HCTSIZ.Pid per bulk direction (the core reports the follow-on
    /// PID in HCTSIZ after each transfer; carrying it is the data toggle).
    pid_in: u32,
    pid_out: u32,
    tag: u32,
    ready: bool,
}

static mut DEV: Dev = Dev {
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
    pid_in: PID_DATA0,
    pid_out: PID_DATA0,
    tag: 0,
    ready: false,
};

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
    core::ptr::addr_of_mut!(BOUNCE) as u32
}

// --- channel 0 transfer engine -----------------------------------------------

fn chr(dev_addr: u32, ep: u32, dir_in: bool, eptype: u32, mps: u32) -> u32 {
    mps | ep << 11 | (dir_in as u32) << 15 | eptype | 1 << 20 | dev_addr << 22
}

/// One transfer on host channel 0 (buffer DMA). For IN, `len` must be a
/// multiple of `mps` and the buffer must have room for all of it; a short
/// packet from the device completes the transfer early. Returns the next
/// PID the endpoint expects.
fn ch0(chr_v: u32, pid: u32, dma: u32, len: usize, mps: usize, to_ms: u32) -> Result<u32, i32> {
    let pkts = if len == 0 { 1 } else { len.div_ceil(mps) as u32 };
    unsafe {
        write_volatile(core_reg(HCINT0), 0xFFFF_FFFF);
        write_volatile(core_reg(HCINTMSK0), 0); // polled: no propagation
        write_volatile(core_reg(HCTSIZ0), len as u32 | pkts << 19 | pid << 29);
        write_volatile(core_reg(HCDMA0), dma);
        write_volatile(core_reg(HCCHAR0), chr_v | CH_CHENA);
    }
    if !poll(core_reg(HCINT0), HCI_CHHLTD, HCI_CHHLTD, to_ms) {
        // Deadline (device NAKing forever, or gone): request a halt and
        // give the core a moment to return the channel.
        unsafe {
            write_volatile(core_reg(HCCHAR0), chr_v | CH_CHENA | CH_CHDIS);
        }
        poll(core_reg(HCINT0), HCI_CHHLTD, HCI_CHHLTD, 5);
        return Err(-650);
    }
    let ints = unsafe { read_volatile(core_reg(HCINT0)) };
    if ints & HCI_XFERCOMPL != 0 {
        return Ok(unsafe { read_volatile(core_reg(HCTSIZ0)) } >> 29 & 3);
    }
    Err(if ints & HCI_STALL != 0 {
        -651
    } else if ints & HCI_XACTERR != 0 {
        -652
    } else if ints & HCI_BBLERR != 0 {
        -653
    } else if ints & HCI_DTGLERR != 0 {
        -654
    } else if ints & HCI_AHBERR != 0 {
        -655
    } else if ints & HCI_FRMOVRUN != 0 {
        -656
    } else {
        -657
    })
}

/// Control transfer on endpoint 0. `dlen` bytes of IN data land in BOUNCE
/// (callers copy out); OUT data stages are not needed by this driver.
fn control_in(sp: [u8; 8], dlen: usize) -> i32 {
    let (addr, mps) = unsafe { ((*core::ptr::addr_of!(DEV)).addr, (*core::ptr::addr_of!(DEV)).mps0) };
    let mpsu = mps as usize;
    unsafe {
        (*core::ptr::addr_of_mut!(SETUP)).0 = sp;
    }
    let sp_addr = core::ptr::addr_of!(SETUP) as u32;
    if let Err(e) = ch0(chr(addr, 0, false, CH_EPTYPE_CTRL, mps), PID_SETUP, sp_addr, 8, mpsu, 200) {
        return e;
    }
    if dlen > 0 {
        debug_assert!(dlen.div_ceil(mpsu) * mpsu <= BLOCK);
        let rounded = dlen.div_ceil(mpsu) * mpsu;
        if let Err(e) = ch0(
            chr(addr, 0, true, CH_EPTYPE_CTRL, mps),
            PID_DATA1,
            bounce_addr(),
            rounded,
            mpsu,
            500,
        ) {
            return e;
        }
    }
    // Status stage: opposite direction of the data stage (OUT here), or IN
    // for a no-data request. Zero length, always DATA1.
    let status_in = dlen == 0;
    match ch0(
        chr(addr, 0, status_in, CH_EPTYPE_CTRL, mps),
        PID_DATA1,
        bounce_addr(),
        0,
        mpsu,
        200,
    ) {
        Ok(_) => 0,
        Err(e) => e,
    }
}

fn bulk(dir_in: bool, dma: u32, len: usize, to_ms: u32) -> i32 {
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DEV) };
    let (ep, mps, pid) = if dir_in {
        (d.msc.ep_in, d.msc.mps_in, d.pid_in)
    } else {
        (d.msc.ep_out, d.msc.mps_out, d.pid_out)
    };
    match ch0(
        chr(d.addr, ep as u32, dir_in, CH_EPTYPE_BULK, mps as u32),
        pid,
        dma,
        len,
        mps as usize,
        to_ms,
    ) {
        Ok(next) => {
            if dir_in {
                d.pid_in = next;
            } else {
                d.pid_out = next;
            }
            0
        }
        Err(e) => e,
    }
}

/// CLEAR_FEATURE(ENDPOINT_HALT) after a bulk STALL; the endpoint toggle
/// resets to DATA0 on both sides.
fn clear_halt(dir_in: bool) {
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DEV) };
    let ep = if dir_in { d.msc.ep_in as u16 | 0x80 } else { d.msc.ep_out as u16 };
    let rc = control_in(
        proto::setup(0x02, proto::REQ_CLEAR_FEATURE, proto::FEATURE_ENDPOINT_HALT, ep, 0),
        0,
    );
    if rc == 0 {
        if dir_in {
            d.pid_in = PID_DATA0;
        } else {
            d.pid_out = PID_DATA0;
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
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DEV) };
    d.tag = d.tag.wrapping_add(1);
    let tag = d.tag;
    unsafe {
        proto::build_cbw(&mut (*core::ptr::addr_of_mut!(CBW)).0, tag, dlen as u32, dir_in, cb);
    }
    let rc = bulk(false, core::ptr::addr_of!(CBW) as u32, proto::CBW_LEN, 500);
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
    // CSW on bulk IN. One retry after a STALL (BOT 1.0, 6.7.2).
    let mps_in = d.msc.mps_in as usize;
    let csw_addr = core::ptr::addr_of_mut!(CSWBUF) as u32;
    let mut rc = bulk(true, csw_addr, mps_in, 500);
    if rc == -651 {
        clear_halt(true);
        rc = bulk(true, csw_addr, mps_in, 500);
    }
    if rc != 0 {
        return rc;
    }
    let csw = unsafe { &(&(*core::ptr::addr_of!(CSWBUF)).0)[..proto::CSW_LEN] };
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
    let ifnum = unsafe { (*core::ptr::addr_of!(DEV)).msc.ifnum } as u16;
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
    (unsafe { (*core::ptr::addr_of!(BOUNCE)).0[2] } & 0x0F) as i32
}

// --- bring-up ----------------------------------------------------------------

fn hprt_set(bits: u32) {
    unsafe {
        let v = read_volatile(core_reg(HPRT));
        write_volatile(core_reg(HPRT), (v & !HPRT_W1C) | bits);
    }
}

pub(crate) fn power_down() {
    unsafe {
        write_volatile(wrap(WRAP_TASKS_STOP), 1);
        write_volatile(wrap(WRAP_ENABLE), 0);
        write_volatile(vreg(VREG_TASKS_STOP), 1);
    }
}

/// Power and clock bring-up shared by both roles: 24 MHz PHY reference,
/// VBUS detection, wrapper enable, and the v4.20a+ core soft-reset
/// handshake. Leaves the core reset and idle, in whatever mode the
/// hardware defaults to -- the caller then forces host (init, below) or
/// device (usbdev) mode. Returns 0, or -600 for "no VBUS wired" and
/// -601..-604 for the bring-up stage that timed out.
pub(crate) fn platform_up() -> i32 {
    // The USB PHY reference is the 24 MHz PLL off HFXO (PHY.CLOCK reset
    // FSEL already selects 24 MHz); nothing else in this firmware starts
    // the crystal.
    unsafe {
        write_volatile(clk(CLK_TASKS_XO24MSTART), 1);
    }
    if !poll(clk(CLK_EVENTS_XO24MSTARTED), 1, 1, 100) {
        return -601;
    }

    // VBUS check first: without 5 V on the VBUS pin the PHY has no
    // signaling rail and there is nothing to talk to. VBUSDETECTED is an
    // EDGE event: if VREGUSB is already running (a previous init this
    // power cycle; soft reset does not fully reset peripherals, erratum
    // [63]) a bare re-START never re-fires it -- hardware-observed as a
    // silent -600 with VBUS present. Stop first to force a fresh
    // detection cycle.
    unsafe {
        write_volatile(vreg(VREG_TASKS_STOP), 1);
    }
    ms_wait(2);
    unsafe {
        write_volatile(vreg(VREG_EVENTS_VBUSDETECTED), 0);
        write_volatile(vreg(VREG_TASKS_START), 1);
    }
    if !poll(vreg(VREG_EVENTS_VBUSDETECTED), 1, 1, 100) {
        power_down();
        return -600;
    }

    unsafe {
        write_volatile(wrap(WRAP_ENABLE), 3); // CORE + PHY
    }
    ms_wait(1); // PHY clock start (the H20 quirk waits 45 us here)
    unsafe {
        write_volatile(wrap(WRAP_TASKS_START), 1);
    }
    // HARDWARE-CONFIRMED: the wrapper's STATUS.CORE bit never asserts on
    // this part even with the core fully alive and register access
    // working, so it cannot be the readiness gate. GSNPSID answering
    // with the Synopsys signature ("OT" in the top bytes) is.
    if !poll(core_reg(GSNPSID), 0xFFFF_0000, 0x4F54_0000, 50) {
        power_down();
        return -602;
    }

    // Core soft reset, then force host mode (the force bit survives the
    // reset per the datasheet, but ordering this way needs no such trust).
    if !poll(core_reg(GRSTCTL), GRSTCTL_AHBIDLE, GRSTCTL_AHBIDLE, 50) {
        power_down();
        return -603;
    }
    unsafe {
        write_volatile(core_reg(GRSTCTL), GRSTCTL_CSFTRST);
    }
    // v4.20a+ handshake: wait for CSftRstDone, then clear both bits.
    if !poll(core_reg(GRSTCTL), GRSTCTL_CSFTRSTDONE, GRSTCTL_CSFTRSTDONE, 50) {
        power_down();
        return -604;
    }
    unsafe {
        write_volatile(core_reg(GRSTCTL), 0);
    }
    if !poll(core_reg(GRSTCTL), GRSTCTL_AHBIDLE, GRSTCTL_AHBIDLE, 50) {
        power_down();
        return -604;
    }
    0
}

/// Bring up the port, enumerate the stick, and get its SCSI unit ready.
/// Returns 0 or a negative stage-tagged error (-600 = no VBUS: nothing is
/// wired, the caller should fall back to SD quietly).
pub fn init() -> i32 {
    unsafe {
        *core::ptr::addr_of_mut!(DEV) = Dev {
            mps0: 64,
            pid_in: PID_DATA0,
            pid_out: PID_DATA0,
            ..Default::default()
        };
    }

    let rc = platform_up();
    if rc != 0 {
        return rc;
    }

    unsafe {
        let cfg = read_volatile(core_reg(GUSBCFG));
        write_volatile(core_reg(GUSBCFG), cfg | GUSBCFG_FRCHSTMODE);
        let otg = read_volatile(core_reg(GOTGCTL));
        write_volatile(core_reg(GOTGCTL), otg | OTG_VBVALID_OV | OTG_AVALID_OV);
    }
    // The mode change is specified to take up to 25 ms.
    if !poll(core_reg(GINTSTS), GINTSTS_CURMOD_HOST, GINTSTS_CURMOD_HOST, 60) {
        let hw = unsafe { read_volatile(core_reg(GHWCFG2)) };
        rprintln!("usb: host mode refused (GHWCFG2={:#010x} OTGMODE={})", hw, hw & 7);
        power_down();
        return -605;
    }

    unsafe {
        // Buffer DMA, INCR4 bursts; global interrupt output stays masked.
        write_volatile(core_reg(GAHBCFG), GAHBCFG_DMAEN | GAHBCFG_BURST_INCR4);
        // FIFO carve-up in 32-bit words (12160 available): RX 1024,
        // non-periodic TX 512, periodic TX 256 (unused, must still fit).
        write_volatile(core_reg(GRXFSIZ), 0x400);
        write_volatile(core_reg(GNPTXFSIZ), 0x200 << 16 | 0x400);
        write_volatile(core_reg(HPTXFSIZ), 0x100 << 16 | 0x600);
        write_volatile(core_reg(GRSTCTL), GRSTCTL_TXFFLSH | 0x10 << 6);
    }
    poll(core_reg(GRSTCTL), GRSTCTL_TXFFLSH, 0, 10);
    unsafe {
        write_volatile(core_reg(GRSTCTL), GRSTCTL_RXFFLSH);
    }
    poll(core_reg(GRSTCTL), GRSTCTL_RXFFLSH, 0, 10);
    unsafe {
        // HCFG reset state is right for the internal UTMI HS PHY
        // (FSLSPclkSel = 30/60 MHz, FS/LS-only support off, buffer DMA).
        write_volatile(core_reg(HCFG), read_volatile(core_reg(HCFG)) & !0x7);
    }

    hprt_set(HPRT_PWR);
    // Connect detection: the stick is expected to be attached already, so
    // a short window is enough (and keeps a wedge-free mailbox budget).
    if !poll(core_reg(HPRT), HPRT_CONNSTS, HPRT_CONNSTS, 400) {
        rprintln!("usb: VBUS present but no device on the port");
        power_down();
        return -610;
    }
    unsafe {
        write_volatile(core_reg(HPRT), (read_volatile(core_reg(HPRT)) & !HPRT_W1C) | HPRT_CONNDET);
    }
    ms_wait(100); // attach debounce (USB 2.0, 7.1.7.3)

    hprt_set(HPRT_RST);
    ms_wait(60);
    unsafe {
        let v = read_volatile(core_reg(HPRT));
        write_volatile(core_reg(HPRT), v & !(HPRT_W1C | HPRT_RST));
    }
    if !poll(core_reg(HPRT), HPRT_ENA, HPRT_ENA, 100) {
        power_down();
        return -611;
    }
    let hprt = unsafe { read_volatile(core_reg(HPRT)) };
    unsafe {
        write_volatile(core_reg(HPRT), (hprt & !HPRT_W1C) | HPRT_ENCHNG | HPRT_CONNDET);
    }
    let speed = hprt >> 17 & 3;
    rprintln!("usb: port enabled, speed {} (0=HS 1=FS)", speed);
    if speed == 2 {
        rprintln!("usb: low-speed device is not a stick");
        power_down();
        return -612;
    }
    ms_wait(20); // reset recovery

    // Enumeration at address 0. High speed fixes MPS0 at 64; full speed
    // reports it in byte 7 of the first descriptor read.
    let mut rc = control_in(
        proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_DEVICE, 0, 8),
        8,
    );
    if rc != 0 {
        // One retry: some sticks reject the very first transaction after
        // reset while their firmware is still settling.
        ms_wait(20);
        rc = control_in(
            proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_DEVICE, 0, 8),
            8,
        );
    }
    if rc != 0 {
        rprintln!("usb: first descriptor read failed rc={}", rc);
        power_down();
        return -620;
    }
    unsafe {
        let d = &mut *core::ptr::addr_of_mut!(DEV);
        d.mps0 = (*core::ptr::addr_of!(BOUNCE)).0[7] as u32;
        if d.mps0 < 8 {
            power_down();
            return -621;
        }
    }

    if control_in(proto::setup(0x00, proto::REQ_SET_ADDRESS, 1, 0, 0), 0) != 0 {
        power_down();
        return -622;
    }
    unsafe {
        (*core::ptr::addr_of_mut!(DEV)).addr = 1;
    }
    ms_wait(5); // SET_ADDRESS recovery (USB 2.0, 9.2.6.3)

    if control_in(
        proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_DEVICE, 0, 18),
        18,
    ) != 0
    {
        power_down();
        return -623;
    }
    let (vid, pid) = unsafe {
        let b = &(*core::ptr::addr_of!(BOUNCE)).0;
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
        power_down();
        return -624;
    }
    let total = unsafe {
        let b = &(*core::ptr::addr_of!(BOUNCE)).0;
        (u16::from_le_bytes([b[2], b[3]]) as usize).min(BLOCK - 64)
    };
    if total < 9
        || control_in(
            proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_CONFIG, 0, total as u16),
            total,
        ) != 0
    {
        power_down();
        return -625;
    }
    let msc = {
        let cfg = unsafe { &(&(*core::ptr::addr_of!(BOUNCE)).0)[..total] };
        match proto::parse_config(cfg) {
            Ok(m) => m,
            Err(e) => {
                rprintln!("usb: no mass-storage interface ({})", e);
                power_down();
                return -626;
            }
        }
    };
    unsafe {
        (*core::ptr::addr_of_mut!(DEV)).msc = msc;
    }

    if control_in(
        proto::setup(0x00, proto::REQ_SET_CONFIGURATION, msc.cfg_value as u16, 0, 0),
        0,
    ) != 0
    {
        power_down();
        return -627;
    }
    ms_wait(5);

    // SCSI bring-up: sticks report a power-on UNIT ATTENTION until a
    // REQUEST SENSE collects it, and a stick that was reset mid-command
    // (previous session interrupted) can take seconds of internal
    // recovery before the LUN is ready -- hardware-observed after a
    // reflash landed mid-utterance. Budget accordingly; every wait in
    // here is DWT-bounded, so no watchdog rides along.
    let start = cortex_m::peripheral::DWT::cycle_count();
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
        if cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) > 5000 * CYC_PER_MS {
            rprintln!("usb: unit never became ready (last sense key {})", last_sense);
            power_down();
            return -640;
        }
        ms_wait(20);
    }

    if bot(&proto::cdb_read_capacity10(), true, bounce_addr(), 8, 500) != 0 {
        power_down();
        return -641;
    }
    let (last_lba, blklen) = unsafe {
        let b = &(*core::ptr::addr_of!(BOUNCE)).0;
        (
            u32::from_be_bytes([b[0], b[1], b[2], b[3]]),
            u32::from_be_bytes([b[4], b[5], b[6], b[7]]),
        )
    };
    if blklen != BLOCK as u32 {
        rprintln!("usb: stick block size {} unsupported", blklen);
        power_down();
        return -642;
    }

    // INQUIRY is cosmetic; failure does not gate readiness.
    let name_ok = bot(&proto::cdb_inquiry(36), true, bounce_addr(), 36, 500) == 0;
    unsafe {
        (*core::ptr::addr_of_mut!(DEV)).ready = true;
    }
    rprintln!(
        "usb: {} stick {:04x}:{:04x}, {} MB",
        if speed == 0 { "high-speed" } else { "full-speed" },
        vid,
        pid,
        ((last_lba as u64 + 1) * BLOCK as u64) >> 20
    );
    if name_ok {
        let b = unsafe { &(*core::ptr::addr_of!(BOUNCE)).0 };
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
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let rc = rw_blocks(lba, dst as u32, count, true);
    unsafe {
        RD_CYC += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
        RD_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

pub fn write_blocks(lba: u32, src: *const u8, count: u32) -> i32 {
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let rc = rw_blocks(lba, src as u32, count, false);
    unsafe {
        WR_CYC += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
        WR_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

fn rw_blocks(lba: u32, buf: u32, count: u32, read: bool) -> i32 {
    if !unsafe { (*core::ptr::addr_of!(DEV)).ready } {
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
        // very first write). All waits stay DWT-bounded.
        let to_ms = if read {
            500 + (bytes >> 10) as u32 * 2
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
