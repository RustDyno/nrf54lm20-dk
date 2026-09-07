//! USB DEVICE mode: a PC stands in for the model-image storage.
//!
//! Same DWC2 core as usb.rs, opposite role. usb.rs forces HOST so the chip
//! can read a USB stick directly; this module forces DEVICE (the role the
//! datasheet documents) so a PC can serve the image instead. Only one role
//! can be live at a time -- they are the same peripheral -- so the build
//! picks one (feature "mock-usb"). mockblk.rs speaks the block protocol on
//! top of the bulk pipes here.
//!
//! The interface is CDC-ACM rather than vendor-specific so that Linux binds
//! its in-tree cdc-acm driver: the port appears as /dev/ttyACMn, readable
//! and writable by the `dialout` group, with no udev rule and no root. A
//! vendor interface would need both. The class costs one notification
//! endpoint this code never sends on, and three class requests.
//!
//! Polled, buffer DMA, no interrupts -- the shape of usb.rs and sd.rs, on
//! the PAC's USBHSCORE register block. The HAL's device driver for this
//! core is not used: it is interrupt driven and async, and mockblk.rs
//! chases landing transfers by reading DOEPTSIZ directly. The NVIC line
//! stays disabled; GINTSTS and the per-endpoint DOEPINT/DIEPINT registers
//! are read directly.

use core::ptr::{addr_of, addr_of_mut, read_volatile, write_volatile};

use embassy_nrf::pac::usbhscore::regs;
use embassy_nrf::pac::usbhscore::vals::{
    Devspd, Diepctl0Mps, Diepctl1Eptype, Diepctl1Txfnum, Diepctl2Eptype, Diepctl2Txfnum,
    Dmaen, Doepctl1Eptype, GintstsCurmod, GrstctlTxfnum, Hbstlen, Supcnt,
};
use rtt_target::rprintln;

use crate::usb::{ms_wait, platform_up, poll, power_down, CORE};

pub const MPS0: usize = 64;
pub const MPS_BULK: usize = 512;
// Endpoint 1 IN and 1 OUT carry the block protocol; endpoint 2 IN is the
// CDC notification pipe (never written to). The register accessors below
// are per endpoint (diepctl1, doepctl1, diepctl2...), so the numbers are
// fixed here rather than indexed.
const EP_BULK: usize = 1;
const EP_NOTIFY: usize = 2;

// --- descriptors --------------------------------------------------------------
// VID 0x1209 / PID 0x0001 is pid.codes' explicitly allocated prototyping
// pair -- not a real vendor's id.

const VID: u16 = 0x1209;
const PID: u16 = 0x0001;

#[rustfmt::skip]
const DESC_DEVICE: [u8; 18] = [
    18, 0x01,
    0x00, 0x02,             // USB 2.00
    0x02, 0x00, 0x00,       // class CDC (the union is described per-interface)
    MPS0 as u8,
    VID as u8, (VID >> 8) as u8,
    PID as u8, (PID >> 8) as u8,
    0x01, 0x00,             // bcdDevice 0.01
    1, 2, 3,                // iManufacturer, iProduct, iSerial
    1,                      // one configuration
];

// Answered so a high-speed host knows what the other speed would look
// like; the core enumerates at high speed and stays there.
#[rustfmt::skip]
const DESC_QUALIFIER: [u8; 10] = [
    10, 0x06, 0x00, 0x02, 0x02, 0x00, 0x00, MPS0 as u8, 1, 0,
];

const CONFIG_LEN: usize = 67;

#[rustfmt::skip]
const DESC_CONFIG: [u8; CONFIG_LEN] = [
    // configuration: two interfaces, bus powered, 100 mA
    9, 0x02, CONFIG_LEN as u8, 0x00, 2, 1, 0, 0x80, 50,
    // interface 0: CDC communications / abstract control model
    9, 0x04, 0, 0, 1, 0x02, 0x02, 0x00, 0,
    5, 0x24, 0x00, 0x10, 0x01,          // header, CDC 1.10
    5, 0x24, 0x01, 0x00, 1,             // call management, data on interface 1
    4, 0x24, 0x02, 0x02,                // ACM: supports line coding / state
    5, 0x24, 0x06, 0, 1,                // union: control 0, subordinate 1
    // notification endpoint (required by the class, never written to)
    7, 0x05, 0x80 | EP_NOTIFY as u8, 0x03, 16, 0, 16,
    // interface 1: CDC data, the two bulk pipes the block protocol uses
    9, 0x04, 1, 0, 2, 0x0A, 0x00, 0x00, 0,
    7, 0x05, EP_BULK as u8, 0x02, (MPS_BULK & 0xFF) as u8, (MPS_BULK >> 8) as u8, 0,
    7, 0x05, 0x80 | EP_BULK as u8, 0x02, (MPS_BULK & 0xFF) as u8, (MPS_BULK >> 8) as u8, 0,
];

const DESC_LANG: [u8; 4] = [4, 0x03, 0x09, 0x04]; // en-US

/// UTF-16LE string descriptor built into `dst`; returns its length.
fn string_desc(s: &str, dst: &mut [u8]) -> usize {
    let n = 2 + s.len() * 2;
    dst[0] = n as u8;
    dst[1] = 0x03;
    for (i, ch) in s.bytes().enumerate() {
        dst[2 + i * 2] = ch;
        dst[3 + i * 2] = 0;
    }
    n
}

// --- state ---------------------------------------------------------------------

#[repr(C, align(4))]
struct Buf64([u8; 64]);
/// EP0 IN staging. The configuration descriptor is the longest control
/// response at 67 bytes, and EP0's PktCnt is 2 bits, so three packets of
/// MPS0 is both the ceiling the core can send and ample room.
#[repr(C, align(4))]
struct Buf192([u8; 3 * MPS0]);

/// SETUP landing zone. SUPCnt = 3 lets the core stack up to three
/// back-to-back SETUP packets without an intervening re-arm.
static mut SETUP: Buf64 = Buf64([0; 64]);
/// Control IN staging (descriptors are copied here: DMA reads RAM).
static mut EP0IN: Buf192 = Buf192([0; 3 * MPS0]);
/// Staging for drain(): somewhere to dump bytes we are discarding.
#[repr(C, align(4))]
struct Buf512([u8; MPS_BULK]);
static mut DRAINBUF: Buf512 = Buf512([0; MPS_BULK]);

/// Control OUT data stage. XferSize must be programmed in whole packets,
/// so even a 7-byte SET_LINE_CODING needs MPS0 of room.
static mut EP0OUT: Buf64 = Buf64([0; 64]);
static mut CONFIGURED: bool = false;
static mut ADDRESS: u32 = 0;
/// Set when the host resets the port after we were configured: the link is
/// gone and every transfer should fail rather than hang.
static mut RESET_AFTER_CONFIG: bool = false;

pub fn configured() -> bool {
    unsafe { read_volatile(addr_of!(CONFIGURED)) }
}

// --- endpoint plumbing ---------------------------------------------------------

fn ep0_arm_setup() {
    CORE.doepdma0().write_value(addr_of_mut!(SETUP) as u32);
    CORE.doeptsiz0().write(|w| {
        w.set_supcnt(Supcnt::Threepacket);
        w.set_pktcnt(true);
        w.set_xfersize(24);
    });
    CORE.doepctl0().modify(|w| {
        w.set_epena(true);
        w.set_cnak(true);
    });
}

/// Arm EP0 OUT for a data or status stage of at most one packet.
fn ep0_arm_out(len: u32) {
    CORE.doepdma0().write_value(addr_of_mut!(EP0OUT) as u32);
    CORE.doeptsiz0().write(|w| {
        w.set_pktcnt(true);
        w.set_xfersize(len as u8);
    });
    CORE.doepctl0().modify(|w| {
        w.set_epena(true);
        w.set_cnak(true);
    });
}

/// Send up to 3 packets on EP0 IN (PktCnt is 2 bits there) and wait for the
/// core to hand them over, then arm the status OUT.
fn ep0_in(data: &[u8], req_len: usize) {
    let n = data.len().min(req_len).min(3 * MPS0);
    unsafe {
        let buf = &mut (*addr_of_mut!(EP0IN)).0;
        buf[..n].copy_from_slice(&data[..n]);
    }
    let pkts = if n == 0 { 1 } else { n.div_ceil(MPS0) } as u8;
    CORE.diepdma0().write_value(addr_of_mut!(EP0IN) as u32);
    CORE.dieptsiz0().write(|w| {
        w.set_pktcnt(pkts);
        w.set_xfersize(n as u8);
    });
    CORE.diepctl0().modify(|w| {
        w.set_epena(true);
        w.set_cnak(true);
    });
    // The status stage is a zero-length OUT; arm it now so the host never
    // sees a NAK storm after a short descriptor.
    poll(|| CORE.diepint0().read().xfercompl(), 50);
    CORE.diepint0().write(|w| w.set_xfercompl(true));
    ep0_arm_out(0);
}

/// Zero-length IN: the status stage of a control transfer with no data.
fn ep0_status_in() {
    CORE.diepdma0().write_value(addr_of_mut!(EP0IN) as u32);
    CORE.dieptsiz0().write(|w| w.set_pktcnt(1));
    CORE.diepctl0().modify(|w| {
        w.set_epena(true);
        w.set_cnak(true);
    });
    poll(|| CORE.diepint0().read().xfercompl(), 50);
    CORE.diepint0().write(|w| w.set_xfercompl(true));
}

fn ep0_stall() {
    CORE.diepctl0().modify(|w| w.set_stall(true));
    CORE.doepctl0().modify(|w| w.set_stall(true));
    ep0_arm_setup();
}

fn activate_data_endpoints() {
    CORE.diepctl1().write(|w| {
        w.set_mps(MPS_BULK as u16);
        w.set_usbactep(true);
        w.set_eptype(Diepctl1Eptype::Bulk);
        w.set_txfnum(Diepctl1Txfnum::Txfifo1);
        w.set_setd0pid(true);
        w.set_snak(true);
    });
    CORE.doepctl1().write(|w| {
        w.set_mps(MPS_BULK as u16);
        w.set_usbactep(true);
        w.set_eptype(Doepctl1Eptype::Bulk);
        w.set_setd0pid(true);
        w.set_snak(true);
    });
    CORE.diepctl2().write(|w| {
        w.set_mps(16);
        w.set_usbactep(true);
        w.set_eptype(Diepctl2Eptype::Interrup);
        w.set_txfnum(Diepctl2Txfnum::Txfifo2);
        w.set_setd0pid(true);
        w.set_snak(true);
    });
}

// --- control transfers ----------------------------------------------------------

fn handle_setup() {
    let sp = unsafe { (*addr_of!(SETUP)).0 };
    let bm = sp[0];
    let req = sp[1];
    let val = u16::from_le_bytes([sp[2], sp[3]]);
    let idx = u16::from_le_bytes([sp[4], sp[5]]);
    let len = u16::from_le_bytes([sp[6], sp[7]]) as usize;

    let recip_std = bm & 0x60 == 0x00;
    let class = bm & 0x60 == 0x20;
    let dev_to_host = bm & 0x80 != 0;

    if class {
        // CDC: the kernel driver issues these before it will open the port.
        // SET_LINE_CODING carries 7 bytes we have no use for (there is no
        // UART behind this); accepting them is the whole requirement.
        match req {
            0x20 => {
                // SET_LINE_CODING: data stage, then status IN
                ep0_arm_out(MPS0 as u32);
                if poll(|| CORE.doepint0().read().xfercompl(), 50) {
                    CORE.doepint0().write(|w| w.set_xfercompl(true));
                }
                ep0_status_in();
            }
            0x21 => {
                // GET_LINE_CODING: 115200 8N1, entirely nominal
                let lc = [0x00u8, 0xC2, 0x01, 0x00, 0x00, 0x00, 0x08];
                ep0_in(&lc, len);
            }
            0x22 => ep0_status_in(), // SET_CONTROL_LINE_STATE (DTR/RTS)
            0x23 => ep0_status_in(), // SEND_BREAK
            _ => ep0_stall(),
        }
        ep0_arm_setup();
        return;
    }

    if !recip_std {
        ep0_stall();
        return;
    }

    match (req, dev_to_host) {
        // GET_DESCRIPTOR
        (0x06, true) => {
            let dtype = (val >> 8) as u8;
            let dindex = (val & 0xFF) as u8;
            match (dtype, dindex) {
                (1, _) => ep0_in(&DESC_DEVICE, len),
                (2, _) => ep0_in(&DESC_CONFIG, len),
                (6, _) => ep0_in(&DESC_QUALIFIER, len),
                (3, 0) => ep0_in(&DESC_LANG, len),
                (3, n) => {
                    let mut tmp = [0u8; 64];
                    let s = match n {
                        1 => "nRF54LM20-DK",
                        2 => "Whisper mock storage",
                        _ => "WHISPER1",
                    };
                    let l = string_desc(s, &mut tmp);
                    ep0_in(&tmp[..l], len);
                }
                _ => ep0_stall(),
            }
        }
        // SET_ADDRESS: DWC2 wants the address programmed BEFORE the status
        // stage goes out, not after it completes.
        (0x05, false) => {
            let addr = (val & 0x7F) as u32;
            unsafe { ADDRESS = addr };
            CORE.dcfg().modify(|w| w.set_devaddr(addr as u8));
            ep0_status_in();
        }
        (0x09, false) => {
            // SET_CONFIGURATION
            if val == 0 {
                unsafe { CONFIGURED = false };
            } else {
                activate_data_endpoints();
                unsafe { CONFIGURED = true };
            }
            ep0_status_in();
        }
        (0x08, true) => ep0_in(&[unsafe { CONFIGURED } as u8], len), // GET_CONFIGURATION
        (0x00, true) => ep0_in(&[0, 0], len),                        // GET_STATUS
        (0x0A, true) => ep0_in(&[0], len),                           // GET_INTERFACE
        (0x0B, false) => ep0_status_in(),                            // SET_INTERFACE
        (0x01, false) | (0x03, false) => ep0_status_in(),            // CLEAR/SET_FEATURE
        _ => {
            let _ = idx;
            ep0_stall();
        }
    }
    ep0_arm_setup();
}

/// Service global events and endpoint 0. Deliberately touches NO endpoint
/// but 0: the bulk transfer routines own DIEPINT/DOEPINT for EP1 and would
/// lose completions to a pump() call that cleared them.
pub fn pump() {
    let g = CORE.gintsts().read();

    if g.usbrst() {
        CORE.gintsts().write(|w| w.set_usbrst(true));
        // A reset after we were configured means the host tore the link
        // down; the caller needs to see that rather than block forever.
        unsafe {
            if CONFIGURED {
                RESET_AFTER_CONFIG = true;
            }
            CONFIGURED = false;
        }
        CORE.dctl().modify(|w| w.set_cgoutnak(true));
        CORE.dcfg().modify(|w| w.set_devaddr(0));
        ep0_arm_setup();
    }

    if g.enumdone() {
        CORE.gintsts().write(|w| w.set_enumdone(true));
        let spd = CORE.dsts().read().enumspd().to_bits();
        // EP0 MPS is an enum there (0 = 64 B), which is what we advertise
        // at either speed, so the reset value already suits.
        CORE.diepctl0().modify(|w| w.set_mps(Diepctl0Mps::Bytes64));
        CORE.dctl().modify(|w| w.set_cgnpinnak(true));
        rprintln!("usbdev: enumerated, speed {} (0=HS 1=FS)", spd);
        ep0_arm_setup();
    }

    // SETUP arrival and control data completions both land on EP0 OUT.
    let o = CORE.doepint0().read();
    if o.setup() {
        CORE.doepint0().write(|w| {
            w.set_setup(true);
            w.set_xfercompl(true);
        });
        handle_setup();
    } else if o.xfercompl() {
        CORE.doepint0().write(|w| w.set_xfercompl(true));
    }
}

// --- bulk transfers ---------------------------------------------------------------

const CYC_PER_MS: u32 = 128_000;

fn now() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

/// Wrap-safe transfer deadline.
///
/// DWT is a 32-bit cycle counter: at 128 MHz it wraps every ~33 s, so a
/// single `now - start` cannot express a longer wait, and the obvious
/// `to_ms * CYC_PER_MS` overflows u32 at the same point -- a 60 s budget
/// silently became 26.5 s and reported a spurious timeout. Accumulating
/// the deltas between polls is exact for any budget, because the poll
/// loops iterate far faster than the counter wraps.
struct Budget {
    last: u32,
    acc: u64,
    limit: u64,
}

impl Budget {
    fn new(ms: u32) -> Budget {
        Budget {
            last: now(),
            acc: 0,
            limit: ms as u64 * CYC_PER_MS as u64,
        }
    }

    fn expired(&mut self) -> bool {
        let n = now();
        self.acc += n.wrapping_sub(self.last) as u64;
        self.last = n;
        self.acc >= self.limit
    }
}

/// One bulk IN transfer of `len` bytes from a word-aligned buffer. A length
/// that is not a whole number of packets ends with a short packet, which is
/// exactly how the host learns the transfer is over.
fn ep_in_xfer(dma: u32, len: usize, to_ms: u32) -> i32 {
    ep_in_arm(dma, len);
    let mut budget = Budget::new(to_ms);
    loop {
        if let Some(rc) = ep_in_check(&mut budget) {
            return rc;
        }
    }
}

fn ep_in_arm(dma: u32, len: usize) {
    let pkts = if len == 0 { 1 } else { len.div_ceil(MPS_BULK) } as u16;
    // The endpoint DMA reads a buffer the CPU has just written; make those
    // stores visible before the transfer is armed.
    cortex_m::asm::dmb();
    CORE.diepint1().write(|w| w.0 = !0);
    CORE.diepdma1().write_value(dma);
    CORE.dieptsiz1().write(|w| {
        w.set_pktcnt(pkts);
        w.set_xfersize(len as u32);
    });
    CORE.diepctl1().modify(|w| {
        w.set_epena(true);
        w.set_cnak(true);
    });
}

/// Non-blocking completion check for `ep_in_arm`; services the control
/// endpoint on the way. None while the transfer is still running.
fn ep_in_check(budget: &mut Budget) -> Option<i32> {
    let i = CORE.diepint1().read();
    if i.xfercompl() {
        CORE.diepint1().write(|w| w.set_xfercompl(true));
        return Some(0);
    }
    if i.ahberr() {
        ep_abort(true);
        return Some(-621);
    }
    pump();
    if unsafe { RESET_AFTER_CONFIG } {
        ep_abort(true);
        return Some(-623);
    }
    if budget.expired() {
        ep_abort(true);
        return Some(-620);
    }
    None
}

/// One bulk OUT transfer into a word-aligned buffer with `cap` bytes of
/// room (rounded down to whole packets). Returns bytes received: the core
/// reports the shortfall as the residual XferSize, and a short packet ends
/// the transfer early, so callers must loop until they have what they need.
fn ep_out_xfer(dma: u32, cap: usize, to_ms: u32) -> Result<usize, i32> {
    let want = ep_out_arm(dma, cap);
    let mut budget = Budget::new(to_ms);
    loop {
        if let Some(r) = ep_out_check(want, &mut budget) {
            return r;
        }
    }
}

/// Returns the number of bytes the transfer was programmed for.
fn ep_out_arm(dma: u32, cap: usize) -> usize {
    let pkts = (cap / MPS_BULK).max(1) as u16;
    let want = pkts as usize * MPS_BULK;
    CORE.doepint1().write(|w| w.0 = !0);
    CORE.doepdma1().write_value(dma);
    CORE.doeptsiz1().write(|w| {
        w.set_pktcnt(pkts);
        w.set_xfersize(want as u32);
    });
    CORE.doepctl1().modify(|w| {
        w.set_epena(true);
        w.set_cnak(true);
    });
    want
}

fn ep_out_check(want: usize, budget: &mut Budget) -> Option<Result<usize, i32>> {
    let i = CORE.doepint1().read();
    if i.xfercompl() {
        CORE.doepint1().write(|w| w.set_xfercompl(true));
        let left = CORE.doeptsiz1().read().xfersize();
        return Some(Ok(want - left as usize));
    }
    if i.ahberr() {
        ep_abort(false);
        return Some(Err(-621));
    }
    pump();
    if unsafe { RESET_AFTER_CONFIG } {
        ep_abort(false);
        return Some(Err(-623));
    }
    if budget.expired() {
        ep_abort(false);
        return Some(Err(-622));
    }
    None
}

/// Disable a bulk endpoint that is still armed, so the next transfer's
/// programming is not ignored. A transfer that times out (daemon not
/// running, host gone) leaves EPENA set, and the databook's disable
/// ceremony -- global NAK, then EPDis, then a TX FIFO flush for IN -- is
/// the only way back. Without this a single timeout wedges the link until
/// the board is reset.
fn ep_abort(is_in: bool) {
    if is_in {
        if !CORE.diepctl1().read().epena() {
            return;
        }
        CORE.dctl().modify(|w| w.set_sgnpinnak(true));
        poll(|| CORE.gintsts().read().ginnakeff(), 10);
        CORE.diepctl1().modify(|w| {
            w.set_epdis(true);
            w.set_snak(true);
        });
        poll(|| CORE.diepint1().read().epdisbld(), 10);
        CORE.diepint1().write(|w| w.0 = !0);
        // Stale packets in the endpoint's TxFIFO would go out ahead of
        // the next transfer's first packet.
        CORE.grstctl().write(|w| {
            w.set_txfflsh(true);
            w.set_txfnum(GrstctlTxfnum::Txf1);
        });
        poll(|| !CORE.grstctl().read().txfflsh(), 10);
        CORE.dctl().modify(|w| w.set_cgnpinnak(true));
    } else {
        if !CORE.doepctl1().read().epena() {
            return;
        }
        CORE.dctl().modify(|w| w.set_sgoutnak(true));
        poll(|| CORE.gintsts().read().goutnakeff(), 10);
        CORE.doepctl1().modify(|w| {
            w.set_epdis(true);
            w.set_snak(true);
        });
        poll(|| CORE.doepint1().read().epdisbld(), 10);
        CORE.doepint1().write(|w| w.0 = !0);
        CORE.dctl().modify(|w| w.set_cgoutnak(true));
    }
}

/// Send exactly `len` bytes. The buffer must be word aligned (DMA reads it).
///
/// A transfer whose length is a whole number of max-size packets ends with
/// no short packet, and the host has no way to know it is over: the
/// kernel's bulk IN URB is larger than one packet (cdc-acm asks for
/// maxp*2), so it keeps polling for the rest and the bytes we sent sit in
/// the host controller, never reaching userspace. Every frame here is
/// exactly 512 bytes, which is precisely that case -- so terminate with a
/// zero-length packet. Cost is one extra packet per transfer; without it
/// the link looks like it works from the device side (the packet is ACKed
/// and the data toggle advances) while the daemon blocks forever.
pub fn send(dma: u32, len: usize, to_ms: u32) -> i32 {
    let mut off = 0usize;
    while off < len {
        // PktCnt is 10 bits: 1023 packets is the most one transfer can carry.
        let n = (len - off).min(1023 * MPS_BULK);
        let rc = ep_in_xfer(dma + off as u32, n, to_ms);
        if rc != 0 {
            return rc;
        }
        off += n;
    }
    if len % MPS_BULK == 0 {
        return ep_in_xfer(dma, 0, to_ms);
    }
    0
}

/// Receive exactly `len` bytes into a word-aligned buffer. `len` must be a
/// whole number of packets: an OUT transfer can always deliver up to MPS,
/// so a caller wanting less must stage through recv_small.
pub fn recv(dma: u32, len: usize, to_ms: u32) -> i32 {
    let mut off = 0usize;
    while off < len {
        let n = (len - off).min(1023 * MPS_BULK);
        match ep_out_xfer(dma + off as u32, n, to_ms) {
            Ok(got) => {
                if got == 0 {
                    return -624; // zero-length packet: host lost framing
                }
                off += got;
            }
            Err(e) => return e,
        }
    }
    0
}

/// One-line dump of everything that decides whether a bulk IN packet can
/// actually leave: FIFO carve-up, endpoint state, and the core's view of
/// the bus.
pub fn diag(tag: &str) {
    rprintln!(
        "usbdev[{}]: GINTSTS={:#010x} DSTS={:#010x} DCTL={:#010x} GHWCFG3={:#010x} GDFIFOCFG={:#010x}",
        tag,
        CORE.gintsts().read().0,
        CORE.dsts().read().0,
        CORE.dctl().read().0,
        CORE.ghwcfg3().read().0,
        CORE.gdfifocfg().read().0
    );
    rprintln!(
        "usbdev[{}]: GRXFSIZ={:#010x} GNPTXFSIZ={:#010x} TXF1={:#010x} TXF2={:#010x}",
        tag,
        CORE.grxfsiz().read().0,
        CORE.gnptxfsiz().read().0,
        CORE.dieptxf(0).read().0,
        CORE.dieptxf(1).read().0
    );
    rprintln!(
        "usbdev[{}]: DIEPCTL1={:#010x} DIEPTSIZ1={:#010x} DIEPINT1={:#010x} DTXFSTS1={:#010x}",
        tag,
        CORE.diepctl1().read().0,
        CORE.dieptsiz1().read().0,
        CORE.diepint1().read().0,
        CORE.dtxfsts1().read().0
    );
    rprintln!(
        "usbdev[{}]: DOEPCTL1={:#010x} DOEPTSIZ1={:#010x} DOEPINT1={:#010x}",
        tag,
        CORE.doepctl1().read().0,
        CORE.doeptsiz1().read().0,
        CORE.doepint1().read().0
    );
}

/// Swallow anything the host has already queued for us.
///
/// If the device ever abandons a request, the daemon still answers it, and
/// that answer would be read as the reply to the NEXT request -- one frame
/// out of step for the rest of the run. Draining before a retry is what
/// makes retrying safe; it ends when a read finds nothing left.
pub fn drain() {
    for _ in 0..64 {
        if ep_out_xfer(addr_of_mut!(DRAINBUF) as u32, MPS_BULK, 40).is_err() {
            return;
        }
    }
}

// --- bring-up ---------------------------------------------------------------------

/// Force device mode, enumerate against the PC, and wait to be configured.
/// Returns 0, or a negative stage-tagged code (-600 = no VBUS: nothing is
/// plugged in, so the caller can fall back quietly).
pub fn init(wait_ms: u32) -> i32 {
    unsafe {
        write_volatile(addr_of_mut!(CONFIGURED), false);
        ADDRESS = 0;
        RESET_AFTER_CONFIG = false;
    }

    let rc = platform_up();
    if rc != 0 {
        return rc;
    }

    CORE.gusbcfg().modify(|w| {
        w.set_forcehstmode(false);
        w.set_forcedevmode(true);
    });
    // Mode changes are specified to take up to 25 ms; CURMOD reads 0 in
    // device mode.
    if !poll(|| CORE.gintsts().read().curmod() == GintstsCurmod::Device, 60) {
        rprintln!("usbdev: device mode refused");
        power_down();
        return -630;
    }

    // Hold the bus off until the endpoints and FIFOs are programmed, so
    // the PC's first reset finds a device that can answer.
    CORE.dctl().modify(|w| w.set_sftdiscon(true));
    CORE.gahbcfg().write(|w| {
        w.set_dmaen(Dmaen::Dmamode);
        w.set_hbstlen(Hbstlen::Word16orincr4);
    });
    CORE.dcfg().write(|w| w.set_devspd(Devspd::Usbhs20));

    // FIFO carve-up in 32-bit words. The shared RX FIFO needs
    // (4*ctrl_eps + 6) + (MPS/4 + 1)*packets + 2*out_eps + 1; 640 words
    // covers four 512-byte packets in flight with room to spare.
    // Written as whole words (depth in the high half, start address in
    // the low): the SVD declares these size and start fields as 10 bits,
    // so the typed setters silently turn the 1024-word EP1 FIFO into a
    // zero-word one (hardware-observed: DTXFSTS1 = 0, every IN transfer
    // times out), while the core takes the value.
    let rx = 640u32;
    let np = 256u32; // EP0 IN
    let tx1 = 1024u32; // EP1 IN bulk
    let tx2 = 64u32; // EP2 IN notifications
    CORE.grxfsiz().write_value(regs::Grxfsiz(rx));
    CORE.gnptxfsiz().write_value(regs::Gnptxfsiz((np << 16) | rx));
    // dieptxf(0) is DIEPTXF1 (EP1 IN), dieptxf(1) is DIEPTXF2.
    CORE.dieptxf(0).write_value(regs::Dieptxf((tx1 << 16) | (rx + np)));
    CORE.dieptxf(1).write_value(regs::Dieptxf((tx2 << 16) | (rx + np + tx1)));
    // The databook requires the endpoint-info block to sit above every
    // FIFO. Only that base address is ours to set -- the low half is
    // the core's own total-size value, so read-modify-write it.
    let epinfo = rx + np + tx1 + tx2;
    CORE.gdfifocfg().modify(|w| w.set_epinfobaseaddr(epinfo as u16));
    // Clock gating off: a gated PHY/core clock silently swallows
    // transfers.
    CORE.pcgcctl().write_value(regs::Pcgcctl(0));

    CORE.grstctl().write(|w| {
        w.set_txfflsh(true);
        w.set_txfnum(GrstctlTxfnum::Txf16); // all TX FIFOs
    });
    poll(|| !CORE.grstctl().read().txfflsh(), 10);
    CORE.grstctl().write(|w| w.set_rxfflsh(true));
    poll(|| !CORE.grstctl().read().rxfflsh(), 10);

    // Interrupt lines stay masked at the NVIC; these masks only gate
    // the status bits this code polls.
    CORE.diepmsk().write_value(regs::Diepmsk(0));
    CORE.doepmsk().write_value(regs::Doepmsk(0));
    CORE.daintmsk().write_value(regs::Daintmsk(0));
    CORE.gintsts().write(|w| w.0 = !0);
    CORE.dctl().modify(|w| w.set_pwronprgdone(true));
    // Hold the disconnect long enough for the host to actually register a
    // detach before we re-attach. A reset that leaves VBUS up (a reflash,
    // say) can otherwise present so brief a gap that the host never
    // notices, keeps its old view of the device, never issues a bus reset,
    // and enumeration simply never happens -- the device then sits in
    // DSTS.SuspSts with no USBRst forever. USB 2.0 debounce is 100 ms.
    ms_wait(150);
    // Attach: the PC now sees a device and starts enumeration.
    CORE.dctl().modify(|w| w.set_sftdiscon(false));

    let mut budget = Budget::new(wait_ms);
    while !configured() {
        pump();
        if budget.expired() {
            rprintln!(
                "usbdev: not configured after {} ms (DSTS={:#010x} GINTSTS={:#010x})",
                wait_ms,
                CORE.dsts().read().0,
                CORE.gintsts().read().0
            );
            power_down();
            return -631;
        }
    }
    // A reset seen during enumeration is normal; only one after we are
    // configured means the link died.
    unsafe { RESET_AFTER_CONFIG = false };
    rprintln!("usbdev: configured (address {})", unsafe { ADDRESS });
    0
}

// --- split-phase bulk transfers -------------------------------------------------
//
// mockblk.rs runs a block payload on the endpoint DMA while the CPU works:
// `*_start` arms the first packet run, `*_check` re-arms the remainder
// (and the terminating ZLP for a send) each time it is called, and only
// reports once everything is through. One transfer per direction at a
// time, which is all the block protocol ever has.

pub struct Xfer {
    dma: u32,
    len: usize,
    off: usize,
    /// Bytes the running packet run was programmed for.
    want: usize,
    /// A send that is a whole number of packets ends with a ZLP: true
    /// once that packet has been armed.
    zlp_armed: bool,
    budget: Budget,
}

pub fn send_start(dma: u32, len: usize, to_ms: u32) -> Xfer {
    let mut x = Xfer { dma, len, off: 0, want: 0, zlp_armed: false, budget: Budget::new(to_ms) };
    send_arm_next(&mut x);
    x
}

fn send_arm_next(x: &mut Xfer) {
    if x.off < x.len {
        let n = (x.len - x.off).min(1023 * MPS_BULK);
        ep_in_arm(x.dma + x.off as u32, n);
        x.want = n;
    } else {
        ep_in_arm(x.dma, 0);
        x.want = 0;
        x.zlp_armed = true;
    }
}

/// None while still sending; Some(rc) once the whole payload (and its
/// terminating ZLP, when one is due) has gone out.
pub fn send_check(x: &mut Xfer) -> Option<i32> {
    let rc = ep_in_check(&mut x.budget)?;
    if rc != 0 {
        return Some(rc);
    }
    if x.zlp_armed {
        return Some(0);
    }
    x.off += x.want;
    if x.off < x.len || x.len % MPS_BULK == 0 {
        send_arm_next(x);
        return None;
    }
    Some(0)
}

pub fn recv_start(dma: u32, len: usize, to_ms: u32) -> Xfer {
    let mut x = Xfer { dma, len, off: 0, want: 0, zlp_armed: false, budget: Budget::new(to_ms) };
    recv_arm_next(&mut x);
    x
}

fn recv_arm_next(x: &mut Xfer) {
    let n = (x.len - x.off).min(1023 * MPS_BULK);
    x.want = ep_out_arm(x.dma + x.off as u32, n);
}

/// Bytes of a running receive known to be in memory: the packets the core
/// has written out of its FIFO (DOEPTSIZ.XferSize counts the programmed
/// run down per packet) behind a two-packet margin for a write still on
/// its way through the bus.
pub fn recv_landed(x: &Xfer) -> usize {
    let left = CORE.doeptsiz1().read().xfersize() as usize;
    (x.off + x.want.saturating_sub(left)).saturating_sub(2 * MPS_BULK).min(x.len)
}

/// None while still receiving; Some(rc) once `len` bytes have landed.
pub fn recv_check(x: &mut Xfer) -> Option<i32> {
    match ep_out_check(x.want, &mut x.budget)? {
        Ok(got) => {
            if got == 0 {
                return Some(-624); // zero-length packet: host lost framing
            }
            x.off += got;
            if x.off < x.len {
                recv_arm_next(x);
                return None;
            }
            Some(0)
        }
        Err(e) => Some(e),
    }
}
