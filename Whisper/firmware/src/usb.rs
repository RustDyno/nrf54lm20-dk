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
//! Registers go through the PAC (`embassy_nrf::pac`): the wrapper
//! (USBHS), the PHY rail (VREGUSB), the 24 MHz reference (CLOCK) and the
//! DWC2 core (USBHSCORE, whose SVD carries the host channel block as
//! `hc(n)`). The HAL has a device-mode driver for this core but no host
//! mode, and this driver's shape mirrors sd.rs anyway: polled, no
//! interrupts (NVIC line stays disabled; we read HCINT/HPRT directly),
//! DWT-timed deadlines, cumulative transfer stats. Every transfer runs on
//! HOST CHANNEL 0 ONLY: bulk-only transport is strictly sequential, one
//! channel reprogrammed per transfer is enough, and it sidesteps the HC[n]
//! address-stride ambiguity in the datasheet (0x18 per entry where
//! Synopsys hardware canonically decodes 0x20; the SVD says 0x1C -- only
//! channel 0's base of 0x500 is beyond doubt). The core retries NAKs in
//! hardware (buffer DMA mode), so a transfer either completes, errors, or
//! hits our deadline.

use embassy_nrf::pac;
use pac::usbhscore::regs;
use pac::usbhscore::vals::{
    Avalidovval, CharEptype, Dmaen, Ec, Epdir, Epnum, Fslspclksel, Fslssupp, GintstsCurmod,
    GrstctlTxfnum, Hbstlen, Pid, Vbvalidovval,
};

use crate::usbproto as proto;
use rtt_target::rprintln;

// --- peripherals (secure aliases) --------------------------------------------

const CLOCK: pac::clock::Clock = pac::CLOCK;
const VREG: pac::vregusb::Vregusb = pac::VREGUSB;
const WRAP: pac::usbhs::Usbhs = pac::USBHS;
/// The DWC2 core. Shared with usbdev.rs, which drives the same core in the
/// opposite role.
pub(crate) const CORE: pac::usbhscore::Usbhscore = pac::USBHSCORE;
// The wrapper's STATUS register is deliberately unused: its CORE ready
// bit never asserts on this part (hardware-observed) even with the core
// alive; GSNPSID readback is the working readiness check.

// HARDWARE-CONFIRMED: this core is DWC2 v5.00b (GSNPSID 0x4F54500B), and
// since v4.20a soft reset is a handshake -- CSftRst does NOT self-clear.
// The core sets CSftRstDone (bit 29, absent from the datasheet but present
// in the SVD) and software must then write both bits back to 0. Polling
// for self-clear hangs forever with the reset long since finished.

// HCTSIZ.Pid encoding (host side: 3 is SETUP, which the SVD's device-flavored
// enum names MDATA).
const PID_DATA0: u32 = 0;
const PID_DATA1: u32 = 2;
const PID_SETUP: u32 = 3;

pub const BLOCK: usize = 512;

const CYC_PER_MS: u32 = 128_000; // DWT at the 128 MHz core clock

pub(crate) fn ms_wait(ms: u32) {
    let start = cortex_m::peripheral::DWT::cycle_count();
    while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < ms * CYC_PER_MS {}
}

/// Poll `cond` until it holds or `ms` elapses.
pub(crate) fn poll(cond: impl Fn() -> bool, ms: u32) -> bool {
    let start = cortex_m::peripheral::DWT::cycle_count();
    loop {
        if cond() {
            return true;
        }
        if cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) >= ms * CYC_PER_MS {
            return false;
        }
    }
}

#[inline]
fn hc0() -> pac::usbhscore::Hc {
    CORE.hc(0)
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

/// HCCHAR value for one endpoint: multi-count 1, channel not yet enabled.
fn chr(dev_addr: u32, ep: u32, dir_in: bool, eptype: CharEptype, mps: u32) -> regs::Char {
    let mut c = regs::Char(0);
    c.set_mps(mps as u16);
    c.set_epnum(Epnum::from_bits(ep as u8));
    c.set_epdir(if dir_in { Epdir::In } else { Epdir::Out });
    c.set_eptype(eptype);
    c.set_ec(Ec::Transone);
    c.set_devaddr(dev_addr as u8);
    c
}

/// One transfer on host channel 0 (buffer DMA). For IN, `len` must be a
/// multiple of `mps` and the buffer must have room for all of it; a short
/// packet from the device completes the transfer early. Returns the next
/// PID the endpoint expects.
fn ch0(chr_v: regs::Char, pid: u32, dma: u32, len: usize, mps: usize, to_ms: u32) -> Result<u32, i32> {
    ch0_arm(chr_v, pid, dma, len, mps);
    let start = cortex_m::peripheral::DWT::cycle_count();
    loop {
        if let Some(r) = ch0_check(chr_v, start, to_ms) {
            return r;
        }
    }
}

/// Program and enable channel 0; the core runs the transfer on its own
/// from here (buffer DMA), so the CPU is free until `ch0_check` says the
/// channel halted.
fn ch0_arm(chr_v: regs::Char, pid: u32, dma: u32, len: usize, mps: usize) {
    let pkts = if len == 0 { 1 } else { len.div_ceil(mps) as u32 };
    let hc = hc0();
    hc.int().write(|w| w.0 = 0xFFFF_FFFF);
    hc.intmsk().write_value(regs::Intmsk(0)); // polled: no propagation
    hc.tsiz().write(|w| {
        w.set_xfersize(len as u32);
        w.set_pktcnt(pkts as u16);
        w.set_pid(Pid::from_bits(pid as u8));
    });
    hc.dma().write_value(dma);
    let mut c = chr_v;
    c.set_chena(true);
    hc.char().write_value(c);
}

/// Non-blocking completion check for an armed channel: None while it is
/// still running and within `to_ms` of `start` (a DWT cycle stamp).
fn ch0_check(chr_v: regs::Char, start: u32, to_ms: u32) -> Option<Result<u32, i32>> {
    let hc = hc0();
    let ints = hc.int().read();
    if !ints.chhltd() {
        if cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < to_ms * CYC_PER_MS {
            return None;
        }
        // Deadline (device NAKing forever, or gone): request a halt and
        // give the core a moment to return the channel.
        let mut c = chr_v;
        c.set_chena(true);
        c.set_chdis(true);
        hc.char().write_value(c);
        poll(|| hc.int().read().chhltd(), 5);
        return Some(Err(-650));
    }
    if ints.xfercompl() {
        return Some(Ok(hc.tsiz().read().pid().to_bits() as u32));
    }
    Some(Err(if ints.stall() {
        -651
    } else if ints.xacterr() {
        -652
    } else if ints.bblerr() {
        -653
    } else if ints.datatglerr() {
        -654
    } else if ints.ahberr() {
        -655
    } else if ints.frmovrun() {
        -656
    } else {
        -657
    }))
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
    if let Err(e) = ch0(chr(addr, 0, false, CharEptype::Ctrl, mps), PID_SETUP, sp_addr, 8, mpsu, 200) {
        return e;
    }
    if dlen > 0 {
        debug_assert!(dlen.div_ceil(mpsu) * mpsu <= BLOCK);
        let rounded = dlen.div_ceil(mpsu) * mpsu;
        if let Err(e) = ch0(
            chr(addr, 0, true, CharEptype::Ctrl, mps),
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
        chr(addr, 0, status_in, CharEptype::Ctrl, mps),
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
    let chr_v = bulk_arm(dir_in, dma, len);
    let start = cortex_m::peripheral::DWT::cycle_count();
    loop {
        if let Some(rc) = bulk_check(dir_in, chr_v, start, to_ms) {
            return rc;
        }
    }
}

/// Arm a bulk transfer on the MSC endpoint of `dir_in`; returns the
/// HCCHAR value `bulk_check` needs.
fn bulk_arm(dir_in: bool, dma: u32, len: usize) -> regs::Char {
    let d = unsafe { &*core::ptr::addr_of!(DEV) };
    let (ep, mps, pid) = if dir_in {
        (d.msc.ep_in, d.msc.mps_in, d.pid_in)
    } else {
        (d.msc.ep_out, d.msc.mps_out, d.pid_out)
    };
    let chr_v = chr(d.addr, ep as u32, dir_in, CharEptype::Bulk, mps as u32);
    ch0_arm(chr_v, pid, dma, len, mps as usize);
    chr_v
}

/// Completion check for `bulk_arm`; carries the data toggle forward on
/// success. None while the transfer is still running.
fn bulk_check(dir_in: bool, chr_v: regs::Char, start: u32, to_ms: u32) -> Option<i32> {
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DEV) };
    match ch0_check(chr_v, start, to_ms)? {
        Ok(next) => {
            if dir_in {
                d.pid_in = next;
            } else {
                d.pid_out = next;
            }
            Some(0)
        }
        Err(e) => Some(e),
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
    csw_phase(tag)
}

/// CSW on bulk IN, one retry after a STALL (BOT 1.0, 6.7.2), checked
/// against the command's tag.
fn csw_phase(tag: u32) -> i32 {
    let mps_in = unsafe { (*core::ptr::addr_of!(DEV)).msc.mps_in } as usize;
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

/// Read-modify-write HPRT with its write-1-to-clear bits masked out
/// (PRTENA included: writing 1 DISABLES the port), then apply `f`.
fn hprt_rmw(f: impl FnOnce(&mut regs::Hprt)) {
    let mut v = CORE.hprt().read();
    v.set_prtconndet(false);
    v.set_prtena(false);
    v.set_prtenchng(false);
    v.set_prtovrcurrchng(false);
    f(&mut v);
    CORE.hprt().write_value(v);
}

pub(crate) fn power_down() {
    WRAP.tasks_stop().write_value(1);
    WRAP.enable().write(|w| {
        w.set_core(false);
        w.set_phy(false);
    });
    VREG.tasks_stop().write_value(1);
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
    CLOCK.tasks_xo24mstart().write_value(1);
    if !poll(|| CLOCK.events_xo24mstarted().read() != 0, 100) {
        return -601;
    }

    // VBUS check first: without 5 V on the VBUS pin the PHY has no
    // signaling rail and there is nothing to talk to. VBUSDETECTED is an
    // EDGE event: if VREGUSB is already running (a previous init this
    // power cycle; soft reset does not fully reset peripherals, erratum
    // [63]) a bare re-START never re-fires it -- hardware-observed as a
    // silent -600 with VBUS present. Stop first to force a fresh
    // detection cycle.
    VREG.tasks_stop().write_value(1);
    ms_wait(2);
    VREG.events_vbusdetected().write_value(0);
    VREG.tasks_start().write_value(1);
    if !poll(|| VREG.events_vbusdetected().read() != 0, 100) {
        power_down();
        return -600;
    }

    WRAP.enable().write(|w| {
        w.set_core(true);
        w.set_phy(true);
    });
    ms_wait(1); // PHY clock start (the H20 quirk waits 45 us here)
    WRAP.tasks_start().write_value(1);
    // HARDWARE-CONFIRMED: the wrapper's STATUS.CORE bit never asserts on
    // this part even with the core fully alive and register access
    // working, so it cannot be the readiness gate. GSNPSID answering
    // with the Synopsys signature ("OT" in the top bytes) is.
    if !poll(|| CORE.gsnpsid().read() & 0xFFFF_0000 == 0x4F54_0000, 50) {
        power_down();
        return -602;
    }

    // Core soft reset, then force host mode (the force bit survives the
    // reset per the datasheet, but ordering this way needs no such trust).
    if !poll(|| CORE.grstctl().read().ahbidle(), 50) {
        power_down();
        return -603;
    }
    CORE.grstctl().write(|w| w.set_csftrst(true));
    // v4.20a+ handshake: wait for CSftRstDone, then clear both bits.
    if !poll(|| CORE.grstctl().read().csftrstdone(), 50) {
        power_down();
        return -604;
    }
    CORE.grstctl().write_value(regs::Grstctl(0));
    if !poll(|| CORE.grstctl().read().ahbidle(), 50) {
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

    CORE.gusbcfg().modify(|w| w.set_forcehstmode(true));
    // GOTGCTL session overrides: the host state machine wants A-session
    // and VBUS valid from the PHY; force both so port power does not
    // depend on how the wrapper routes the VBUS comparator in host mode.
    CORE.gotgctl().modify(|w| {
        w.set_vbvalidoven(true);
        w.set_vbvalidovval(Vbvalidovval::Set1);
        w.set_avalidoven(true);
        w.set_avalidovval(Avalidovval::Value1);
    });
    // The mode change is specified to take up to 25 ms.
    if !poll(|| CORE.gintsts().read().curmod() == GintstsCurmod::Host, 60) {
        let hw = CORE.ghwcfg2().read();
        rprintln!(
            "usb: host mode refused (GHWCFG2={:#010x} OTGMODE={})",
            hw.0,
            hw.otgmode().to_bits()
        );
        power_down();
        return -605;
    }

    // Buffer DMA, INCR4 bursts; global interrupt output stays masked.
    CORE.gahbcfg().write(|w| {
        w.set_dmaen(Dmaen::Dmamode);
        w.set_hbstlen(Hbstlen::Word16orincr4);
    });
    // FIFO carve-up in 32-bit words (12160 available): RX 1024,
    // non-periodic TX 512, periodic TX 256 (unused, must still fit).
    // Written as whole words (depth in the high half, start address in
    // the low): the SVD declares these size and start fields as 10 bits,
    // so the typed setters silently turn 1024 into 0, while the core
    // takes 1024-word values (hardware-observed: this carve-up runs).
    CORE.grxfsiz().write_value(regs::Grxfsiz(0x400));
    CORE.gnptxfsiz().write_value(regs::Gnptxfsiz(0x200 << 16 | 0x400));
    CORE.hptxfsiz().write_value(regs::Hptxfsiz(0x100 << 16 | 0x600));
    CORE.grstctl().write(|w| {
        w.set_txfflsh(true);
        w.set_txfnum(GrstctlTxfnum::Txf16); // all TX FIFOs
    });
    poll(|| !CORE.grstctl().read().txfflsh(), 10);
    CORE.grstctl().write(|w| w.set_rxfflsh(true));
    poll(|| !CORE.grstctl().read().rxfflsh(), 10);
    // HCFG reset state is right for the internal UTMI HS PHY
    // (FSLSPclkSel = 30/60 MHz, FS/LS-only support off, buffer DMA).
    CORE.hcfg().modify(|w| {
        w.set_fslspclksel(Fslspclksel::Clk3060);
        w.set_fslssupp(Fslssupp::Hsfsls);
    });

    hprt_rmw(|w| w.set_prtpwr(true));
    // Connect detection: the stick is expected to be attached already, so
    // a short window is enough (and keeps a wedge-free mailbox budget).
    if !poll(|| CORE.hprt().read().prtconnsts(), 400) {
        rprintln!("usb: VBUS present but no device on the port");
        power_down();
        return -610;
    }
    hprt_rmw(|w| w.set_prtconndet(true)); // clear the connect event
    ms_wait(100); // attach debounce (USB 2.0, 7.1.7.3)

    hprt_rmw(|w| w.set_prtrst(true));
    ms_wait(60);
    hprt_rmw(|w| w.set_prtrst(false));
    if !poll(|| CORE.hprt().read().prtena(), 100) {
        power_down();
        return -611;
    }
    let hprt = CORE.hprt().read();
    hprt_rmw(|w| {
        w.set_prtenchng(true);
        w.set_prtconndet(true);
    });
    let speed = hprt.prtspd().to_bits() as u32;
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
    chr_v: regs::Char,
    start: u32,
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
    chr_v: regs::Char(0),
    start: 0,
    to_ms: 0,
    tag: 0,
    rc: 0,
    stage: Stage::Done,
    csw_retried: false,
};

pub fn read_start(lba: u32, dst: *mut u8, count: u32) -> i32 {
    async_start(lba, dst as u32, count, true)
}

pub fn write_start(lba: u32, src: *const u8, count: u32) -> i32 {
    async_start(lba, src as u32, count, false)
}

fn async_start(lba: u32, buf: u32, count: u32, read: bool) -> i32 {
    let a = unsafe { &mut *core::ptr::addr_of_mut!(ASYNC) };
    if a.active {
        return -495;
    }
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    // Cases the split path does not cover run synchronously and report
    // through the same finish(): unaligned buffers, empty and oversized
    // requests, and a stick that is not there.
    if buf & 3 != 0 || count == 0 || count > 65_535 || !unsafe { (*core::ptr::addr_of!(DEV)).ready } {
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
    let d = unsafe { &mut *core::ptr::addr_of_mut!(DEV) };
    d.tag = d.tag.wrapping_add(1);
    let tag = d.tag;
    unsafe {
        proto::build_cbw(&mut (*core::ptr::addr_of_mut!(CBW)).0, tag, bytes as u32, read, &cdb);
    }
    let rc = bulk(false, core::ptr::addr_of!(CBW) as u32, proto::CBW_LEN, 500);
    *a = AsyncOp {
        active: true,
        read,
        buf,
        bytes,
        off: 0,
        prog: 0,
        chr_v: regs::Char(0),
        start: 0,
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
    let dt = cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
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
    let a = unsafe { &mut *core::ptr::addr_of_mut!(ASYNC) };
    let d = unsafe { &*core::ptr::addr_of!(DEV) };
    let mps = if a.read { d.msc.mps_in } else { d.msc.mps_out } as usize;
    let n = (a.bytes - a.off).min(chunk_bytes(mps));
    a.prog = n;
    a.chr_v = bulk_arm(a.read, a.buf + a.off as u32, n);
    a.start = cortex_m::peripheral::DWT::cycle_count();
}

/// Bytes of the pending read known to be in memory: the finished chunks
/// plus what the channel has written of the current one (HCTSIZ.XferSize
/// counts down per packet written out of the RxFIFO), behind a two-packet
/// margin for a write still on its way through the bus. Meaningful while
/// xfer_poll() is false.
pub fn xfer_landed() -> usize {
    let a = unsafe { &*core::ptr::addr_of!(ASYNC) };
    if !a.active {
        return 0;
    }
    match a.stage {
        Stage::Data if a.read => {
            let left = hc0().tsiz().read().xfersize() as usize;
            let mps = unsafe { (*core::ptr::addr_of!(DEV)).msc.mps_in } as usize;
            (a.off + a.prog.saturating_sub(left)).saturating_sub(2 * mps).min(a.bytes)
        }
        _ => a.bytes,
    }
}

/// True when the command has run to completion (rc ready for xfer_finish()).
pub fn xfer_poll() -> bool {
    let a = unsafe { &*core::ptr::addr_of!(ASYNC) };
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let done = poll_inner();
    async_account(a.read, 0, t0);
    done
}

fn poll_inner() -> bool {
    let a = unsafe { &mut *core::ptr::addr_of_mut!(ASYNC) };
    if !a.active || a.stage == Stage::Done {
        return true;
    }
    let rc = match bulk_check(a.stage == Stage::Csw || a.read, a.chr_v, a.start, a.to_ms) {
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
            let csw = unsafe { &(&(*core::ptr::addr_of!(CSWBUF)).0)[..proto::CSW_LEN] };
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
    let a = unsafe { &mut *core::ptr::addr_of_mut!(ASYNC) };
    let mps_in = unsafe { (*core::ptr::addr_of!(DEV)).msc.mps_in } as usize;
    a.chr_v = bulk_arm(true, core::ptr::addr_of_mut!(CSWBUF) as u32, mps_in);
    a.start = cortex_m::peripheral::DWT::cycle_count();
    a.to_ms = 500;
    a.stage = Stage::Csw;
}

/// Transport error: realign the device's BOT state machine and report.
fn async_done(rc: i32) -> bool {
    let a = unsafe { &mut *core::ptr::addr_of_mut!(ASYNC) };
    a.rc = rc;
    a.stage = Stage::Done;
    bot_recover();
    true
}

/// Block until the in-flight command is done; returns its result. A call
/// with nothing in flight returns 0.
pub fn xfer_finish() -> i32 {
    let a = unsafe { &mut *core::ptr::addr_of_mut!(ASYNC) };
    if !a.active {
        return 0;
    }
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    while !poll_inner() {}
    async_account(a.read, 0, t0);
    a.active = false;
    a.rc
}
