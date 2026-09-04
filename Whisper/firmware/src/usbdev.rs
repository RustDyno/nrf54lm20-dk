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
//! Polled, buffer DMA, no interrupts -- the shape of usb.rs and sd.rs. The
//! NVIC line stays disabled; GINTSTS and the per-endpoint DOEPINT/DIEPINT
//! registers are read directly.

use core::ptr::{addr_of, addr_of_mut, read_volatile, write_volatile};

use rtt_target::rprintln;

use crate::usb::{
    core_reg, ms_wait, platform_up, poll, power_down, GAHBCFG, GAHBCFG_BURST_INCR4,
    GAHBCFG_DMAEN, GINTSTS, GNPTXFSIZ, GRSTCTL, GRSTCTL_RXFFLSH, GRSTCTL_TXFFLSH, GRXFSIZ,
    GUSBCFG,
};

// --- device-mode registers (datasheet v1.0-3 USBHSCORE map) ------------------

const GHWCFG3: usize = 0x04C;
const GDFIFOCFG: usize = 0x05C;
const DIEPTXF1: usize = 0x104; // DIEPTXF[n] = 0x104 + (n-1)*4
const PCGCCTL: usize = 0xE00;
const DTXFSTS: usize = 0x18;
const DCFG: usize = 0x800;
const DCTL: usize = 0x804;
const DSTS: usize = 0x808;
const DIEPMSK: usize = 0x810;
const DOEPMSK: usize = 0x814;
const DAINTMSK: usize = 0x81C;

// Endpoint register files: IN at 0x900, OUT at 0xB00, stride 0x20 per
// endpoint (the datasheet map lists DIEPCTL1 = 0x920, DOEPCTL1 = 0xB20 --
// unlike the host channel block, whose stride the datasheet gives as 0x18).
const EP_CTL: usize = 0x00;
const EP_INT: usize = 0x08;
const EP_TSIZ: usize = 0x10;
const EP_DMA: usize = 0x14;

#[inline]
fn diep(n: usize, off: usize) -> *mut u32 {
    core_reg(0x900 + n * 0x20 + off)
}

#[inline]
fn doep(n: usize, off: usize) -> *mut u32 {
    core_reg(0xB00 + n * 0x20 + off)
}

const GUSBCFG_FRCDEVMODE: u32 = 1 << 30;
const GUSBCFG_FRCHSTMODE: u32 = 1 << 29;
const GINTSTS_CURMOD_HOST: u32 = 1 << 0;
const GINTSTS_GINNAKEFF: u32 = 1 << 6;
const GINTSTS_GOUTNAKEFF: u32 = 1 << 7;
const GINTSTS_USBRST: u32 = 1 << 12;
const GINTSTS_ENUMDONE: u32 = 1 << 13;

const DCFG_DEVSPD_HS: u32 = 0;
const DCFG_DEVADDR_SHIFT: u32 = 4;
const DCFG_DEVADDR_MASK: u32 = 0x7F << DCFG_DEVADDR_SHIFT;

const DCTL_SFTDISCON: u32 = 1 << 1;
const DCTL_SGNPINNAK: u32 = 1 << 7;
const DCTL_CGNPINNAK: u32 = 1 << 8;
const DCTL_SGOUTNAK: u32 = 1 << 9;
const DCTL_CGOUTNAK: u32 = 1 << 10;
const DCTL_PWRONPRGDONE: u32 = 1 << 11;

const DSTS_ENUMSPD_SHIFT: u32 = 1;
const DSTS_ENUMSPD_MASK: u32 = 3 << DSTS_ENUMSPD_SHIFT;

const EPCTL_USBACTEP: u32 = 1 << 15;
const EPCTL_EPTYPE_BULK: u32 = 2 << 18;
const EPCTL_EPTYPE_INTR: u32 = 3 << 18;
const EPCTL_STALL: u32 = 1 << 21;
const EPCTL_TXFNUM_SHIFT: u32 = 22;
const EPCTL_CNAK: u32 = 1 << 26;
const EPCTL_SNAK: u32 = 1 << 27;
const EPCTL_SETD0PID: u32 = 1 << 28;
const EPCTL_EPDIS: u32 = 1 << 30;
const EPCTL_EPENA: u32 = 1 << 31;

const EPINT_XFERCOMPL: u32 = 1 << 0;
const EPINT_EPDISBLD: u32 = 1 << 1;
const EPINT_AHBERR: u32 = 1 << 2;
const EPINT_SETUP: u32 = 1 << 3; // OUT endpoints only

// Transfer-size encodings differ between endpoint 0 and the rest:
// EP0 has XferSize[6:0] and a 2-bit (IN) / 1-bit (OUT) PktCnt, EPn has
// XferSize[18:0] and PktCnt[28:19].
const TSIZ0_PKTCNT_SHIFT: u32 = 19;
const TSIZ0_SUPCNT_SHIFT: u32 = 29;
const TSIZN_PKTCNT_SHIFT: u32 = 19;

pub const MPS0: usize = 64;
pub const MPS_BULK: usize = 512;
const EP_BULK: usize = 1; // EP1 IN and EP1 OUT
const EP_NOTIFY: usize = 2; // EP2 IN, CDC notifications (never sent)

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
    unsafe {
        write_volatile(doep(0, EP_DMA), addr_of_mut!(SETUP) as u32);
        write_volatile(
            doep(0, EP_TSIZ),
            (3 << TSIZ0_SUPCNT_SHIFT) | (1 << TSIZ0_PKTCNT_SHIFT) | 24,
        );
        let c = read_volatile(doep(0, EP_CTL));
        write_volatile(doep(0, EP_CTL), c | EPCTL_EPENA | EPCTL_CNAK);
    }
}

/// Arm EP0 OUT for a data or status stage of at most one packet.
fn ep0_arm_out(len: u32) {
    unsafe {
        write_volatile(doep(0, EP_DMA), addr_of_mut!(EP0OUT) as u32);
        write_volatile(doep(0, EP_TSIZ), (1 << TSIZ0_PKTCNT_SHIFT) | len);
        let c = read_volatile(doep(0, EP_CTL));
        write_volatile(doep(0, EP_CTL), c | EPCTL_EPENA | EPCTL_CNAK);
    }
}

/// Send up to 3 packets on EP0 IN (PktCnt is 2 bits there) and wait for the
/// core to hand them over, then arm the status OUT.
fn ep0_in(data: &[u8], req_len: usize) {
    let n = data.len().min(req_len).min(3 * MPS0);
    unsafe {
        let buf = &mut (*addr_of_mut!(EP0IN)).0;
        buf[..n].copy_from_slice(&data[..n]);
        let pkts = if n == 0 { 1 } else { n.div_ceil(MPS0) } as u32;
        write_volatile(diep(0, EP_DMA), addr_of_mut!(EP0IN) as u32);
        write_volatile(diep(0, EP_TSIZ), (pkts << TSIZ0_PKTCNT_SHIFT) | n as u32);
        let c = read_volatile(diep(0, EP_CTL));
        write_volatile(diep(0, EP_CTL), c | EPCTL_EPENA | EPCTL_CNAK);
    }
    // The status stage is a zero-length OUT; arm it now so the host never
    // sees a NAK storm after a short descriptor.
    poll(diep(0, EP_INT), EPINT_XFERCOMPL, EPINT_XFERCOMPL, 50);
    unsafe { write_volatile(diep(0, EP_INT), EPINT_XFERCOMPL) };
    ep0_arm_out(0);
}

/// Zero-length IN: the status stage of a control transfer with no data.
fn ep0_status_in() {
    unsafe {
        write_volatile(diep(0, EP_DMA), addr_of_mut!(EP0IN) as u32);
        write_volatile(diep(0, EP_TSIZ), 1 << TSIZ0_PKTCNT_SHIFT);
        let c = read_volatile(diep(0, EP_CTL));
        write_volatile(diep(0, EP_CTL), c | EPCTL_EPENA | EPCTL_CNAK);
    }
    poll(diep(0, EP_INT), EPINT_XFERCOMPL, EPINT_XFERCOMPL, 50);
    unsafe { write_volatile(diep(0, EP_INT), EPINT_XFERCOMPL) };
}

fn ep0_stall() {
    unsafe {
        let c = read_volatile(diep(0, EP_CTL));
        write_volatile(diep(0, EP_CTL), c | EPCTL_STALL);
        let c = read_volatile(doep(0, EP_CTL));
        write_volatile(doep(0, EP_CTL), c | EPCTL_STALL);
    }
    ep0_arm_setup();
}

fn activate_data_endpoints() {
    unsafe {
        write_volatile(
            diep(EP_BULK, EP_CTL),
            MPS_BULK as u32
                | EPCTL_USBACTEP
                | EPCTL_EPTYPE_BULK
                | ((EP_BULK as u32) << EPCTL_TXFNUM_SHIFT)
                | EPCTL_SETD0PID
                | EPCTL_SNAK,
        );
        write_volatile(
            doep(EP_BULK, EP_CTL),
            MPS_BULK as u32 | EPCTL_USBACTEP | EPCTL_EPTYPE_BULK | EPCTL_SETD0PID | EPCTL_SNAK,
        );
        write_volatile(
            diep(EP_NOTIFY, EP_CTL),
            16 | EPCTL_USBACTEP
                | EPCTL_EPTYPE_INTR
                | ((EP_NOTIFY as u32) << EPCTL_TXFNUM_SHIFT)
                | EPCTL_SETD0PID
                | EPCTL_SNAK,
        );
    }
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
                if poll(doep(0, EP_INT), EPINT_XFERCOMPL, EPINT_XFERCOMPL, 50) {
                    unsafe { write_volatile(doep(0, EP_INT), EPINT_XFERCOMPL) };
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
            unsafe {
                ADDRESS = addr;
                let c = read_volatile(core_reg(DCFG)) & !DCFG_DEVADDR_MASK;
                write_volatile(core_reg(DCFG), c | (addr << DCFG_DEVADDR_SHIFT));
            }
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
    let g = unsafe { read_volatile(core_reg(GINTSTS)) };

    if g & GINTSTS_USBRST != 0 {
        unsafe {
            write_volatile(core_reg(GINTSTS), GINTSTS_USBRST);
            // A reset after we were configured means the host tore the link
            // down; the caller needs to see that rather than block forever.
            if CONFIGURED {
                RESET_AFTER_CONFIG = true;
            }
            CONFIGURED = false;
            write_volatile(core_reg(DCTL), read_volatile(core_reg(DCTL)) | DCTL_CGOUTNAK);
            write_volatile(core_reg(DCFG), read_volatile(core_reg(DCFG)) & !DCFG_DEVADDR_MASK);
        }
        ep0_arm_setup();
    }

    if g & GINTSTS_ENUMDONE != 0 {
        unsafe { write_volatile(core_reg(GINTSTS), GINTSTS_ENUMDONE) };
        let spd = unsafe { (read_volatile(core_reg(DSTS)) & DSTS_ENUMSPD_MASK) >> DSTS_ENUMSPD_SHIFT };
        // EP0 MPS is an enum there (0 = 64 B), which is what we advertise
        // at either speed, so the reset value already suits.
        unsafe {
            let c = read_volatile(diep(0, EP_CTL));
            write_volatile(diep(0, EP_CTL), c & !3);
            write_volatile(core_reg(DCTL), read_volatile(core_reg(DCTL)) | DCTL_CGNPINNAK);
        }
        rprintln!("usbdev: enumerated, speed {} (0=HS 1=FS)", spd);
        ep0_arm_setup();
    }

    // SETUP arrival and control data completions both land on EP0 OUT.
    let o = unsafe { read_volatile(doep(0, EP_INT)) };
    if o & EPINT_SETUP != 0 {
        unsafe { write_volatile(doep(0, EP_INT), EPINT_SETUP | EPINT_XFERCOMPL) };
        handle_setup();
    } else if o & EPINT_XFERCOMPL != 0 {
        unsafe { write_volatile(doep(0, EP_INT), EPINT_XFERCOMPL) };
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
    let pkts = if len == 0 { 1 } else { len.div_ceil(MPS_BULK) } as u32;
    // The endpoint DMA reads a buffer the CPU has just written; make those
    // stores visible before the transfer is armed.
    cortex_m::asm::dmb();
    unsafe {
        write_volatile(diep(EP_BULK, EP_INT), !0);
        write_volatile(diep(EP_BULK, EP_DMA), dma);
        write_volatile(
            diep(EP_BULK, EP_TSIZ),
            (pkts << TSIZN_PKTCNT_SHIFT) | len as u32,
        );
        let c = read_volatile(diep(EP_BULK, EP_CTL));
        write_volatile(diep(EP_BULK, EP_CTL), c | EPCTL_EPENA | EPCTL_CNAK);
    }
    let mut budget = Budget::new(to_ms);
    loop {
        let i = unsafe { read_volatile(diep(EP_BULK, EP_INT)) };
        if i & EPINT_XFERCOMPL != 0 {
            unsafe { write_volatile(diep(EP_BULK, EP_INT), EPINT_XFERCOMPL) };
            return 0;
        }
        if i & EPINT_AHBERR != 0 {
            ep_abort(true);
            return -621;
        }
        pump();
        if unsafe { RESET_AFTER_CONFIG } {
            ep_abort(true);
            return -623;
        }
        if budget.expired() {
            ep_abort(true);
            return -620;
        }
    }
}

/// One bulk OUT transfer into a word-aligned buffer with `cap` bytes of
/// room (rounded down to whole packets). Returns bytes received: the core
/// reports the shortfall as the residual XferSize, and a short packet ends
/// the transfer early, so callers must loop until they have what they need.
fn ep_out_xfer(dma: u32, cap: usize, to_ms: u32) -> Result<usize, i32> {
    let pkts = (cap / MPS_BULK).max(1) as u32;
    let want = pkts as usize * MPS_BULK;
    unsafe {
        write_volatile(doep(EP_BULK, EP_INT), !0);
        write_volatile(doep(EP_BULK, EP_DMA), dma);
        write_volatile(
            doep(EP_BULK, EP_TSIZ),
            (pkts << TSIZN_PKTCNT_SHIFT) | want as u32,
        );
        let c = read_volatile(doep(EP_BULK, EP_CTL));
        write_volatile(doep(EP_BULK, EP_CTL), c | EPCTL_EPENA | EPCTL_CNAK);
    }
    let mut budget = Budget::new(to_ms);
    loop {
        let i = unsafe { read_volatile(doep(EP_BULK, EP_INT)) };
        if i & EPINT_XFERCOMPL != 0 {
            unsafe { write_volatile(doep(EP_BULK, EP_INT), EPINT_XFERCOMPL) };
            let left = unsafe { read_volatile(doep(EP_BULK, EP_TSIZ)) } & 0x7FFFF;
            return Ok(want - left as usize);
        }
        if i & EPINT_AHBERR != 0 {
            ep_abort(false);
            return Err(-621);
        }
        pump();
        if unsafe { RESET_AFTER_CONFIG } {
            ep_abort(false);
            return Err(-623);
        }
        if budget.expired() {
            ep_abort(false);
            return Err(-622);
        }
    }
}

/// Disable a bulk endpoint that is still armed, so the next transfer's
/// programming is not ignored. A transfer that times out (daemon not
/// running, host gone) leaves EPENA set, and the databook's disable
/// ceremony -- global NAK, then EPDis, then a TX FIFO flush for IN -- is
/// the only way back. Without this a single timeout wedges the link until
/// the board is reset.
fn ep_abort(is_in: bool) {
    unsafe {
        let ep = if is_in { diep(EP_BULK, EP_CTL) } else { doep(EP_BULK, EP_CTL) };
        let int = if is_in { diep(EP_BULK, EP_INT) } else { doep(EP_BULK, EP_INT) };
        if read_volatile(ep) & EPCTL_EPENA == 0 {
            return;
        }
        let dctl = core_reg(DCTL);
        if is_in {
            write_volatile(dctl, read_volatile(dctl) | DCTL_SGNPINNAK);
            poll(core_reg(GINTSTS), GINTSTS_GINNAKEFF, GINTSTS_GINNAKEFF, 10);
        } else {
            write_volatile(dctl, read_volatile(dctl) | DCTL_SGOUTNAK);
            poll(core_reg(GINTSTS), GINTSTS_GOUTNAKEFF, GINTSTS_GOUTNAKEFF, 10);
        }
        write_volatile(ep, read_volatile(ep) | EPCTL_EPDIS | EPCTL_SNAK);
        poll(int, EPINT_EPDISBLD, EPINT_EPDISBLD, 10);
        write_volatile(int, !0);
        if is_in {
            // Stale packets in the endpoint's TxFIFO would go out ahead of
            // the next transfer's first packet.
            write_volatile(
                core_reg(GRSTCTL),
                GRSTCTL_TXFFLSH | ((EP_BULK as u32) << 6),
            );
            poll(core_reg(GRSTCTL), GRSTCTL_TXFFLSH, 0, 10);
            write_volatile(dctl, read_volatile(dctl) | DCTL_CGNPINNAK);
        } else {
            write_volatile(dctl, read_volatile(dctl) | DCTL_CGOUTNAK);
        }
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
    unsafe {
        rprintln!(
            "usbdev[{}]: GINTSTS={:#010x} DSTS={:#010x} DCTL={:#010x} GHWCFG3={:#010x} GDFIFOCFG={:#010x}",
            tag,
            read_volatile(core_reg(GINTSTS)),
            read_volatile(core_reg(DSTS)),
            read_volatile(core_reg(DCTL)),
            read_volatile(core_reg(GHWCFG3)),
            read_volatile(core_reg(GDFIFOCFG))
        );
        rprintln!(
            "usbdev[{}]: GRXFSIZ={:#010x} GNPTXFSIZ={:#010x} TXF1={:#010x} TXF2={:#010x}",
            tag,
            read_volatile(core_reg(GRXFSIZ)),
            read_volatile(core_reg(GNPTXFSIZ)),
            read_volatile(core_reg(DIEPTXF1)),
            read_volatile(core_reg(DIEPTXF1 + 4))
        );
        rprintln!(
            "usbdev[{}]: DIEPCTL1={:#010x} DIEPTSIZ1={:#010x} DIEPINT1={:#010x} DTXFSTS1={:#010x}",
            tag,
            read_volatile(diep(EP_BULK, EP_CTL)),
            read_volatile(diep(EP_BULK, EP_TSIZ)),
            read_volatile(diep(EP_BULK, EP_INT)),
            read_volatile(diep(EP_BULK, DTXFSTS))
        );
        rprintln!(
            "usbdev[{}]: DOEPCTL1={:#010x} DOEPTSIZ1={:#010x} DOEPINT1={:#010x}",
            tag,
            read_volatile(doep(EP_BULK, EP_CTL)),
            read_volatile(doep(EP_BULK, EP_TSIZ)),
            read_volatile(doep(EP_BULK, EP_INT))
        );
    }
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
        CONFIGURED = false;
        ADDRESS = 0;
        RESET_AFTER_CONFIG = false;
    }

    let rc = platform_up();
    if rc != 0 {
        return rc;
    }

    unsafe {
        let cfg = read_volatile(core_reg(GUSBCFG));
        write_volatile(
            core_reg(GUSBCFG),
            (cfg & !GUSBCFG_FRCHSTMODE) | GUSBCFG_FRCDEVMODE,
        );
    }
    // Mode changes are specified to take up to 25 ms; CURMOD reads 0 in
    // device mode.
    if !poll(core_reg(GINTSTS), GINTSTS_CURMOD_HOST, 0, 60) {
        rprintln!("usbdev: device mode refused");
        power_down();
        return -630;
    }

    unsafe {
        // Hold the bus off until the endpoints and FIFOs are programmed, so
        // the PC's first reset finds a device that can answer.
        write_volatile(core_reg(DCTL), read_volatile(core_reg(DCTL)) | DCTL_SFTDISCON);
        write_volatile(core_reg(GAHBCFG), GAHBCFG_DMAEN | GAHBCFG_BURST_INCR4);
        write_volatile(core_reg(DCFG), DCFG_DEVSPD_HS);

        // FIFO carve-up in 32-bit words. The shared RX FIFO needs
        // (4*ctrl_eps + 6) + (MPS/4 + 1)*packets + 2*out_eps + 1; 640 words
        // covers four 512-byte packets in flight with room to spare.
        let rx = 640u32;
        let np = 256u32; // EP0 IN
        let tx1 = 1024u32; // EP1 IN bulk
        let tx2 = 64u32; // EP2 IN notifications
        write_volatile(core_reg(GRXFSIZ), rx);
        write_volatile(core_reg(GNPTXFSIZ), (np << 16) | rx);
        write_volatile(core_reg(DIEPTXF1), (tx1 << 16) | (rx + np));
        write_volatile(core_reg(DIEPTXF1 + 4), (tx2 << 16) | (rx + np + tx1));
        // The databook requires the endpoint-info block to sit above every
        // FIFO. Only that base address is ours to set -- the low half is
        // the core's own total-size value, so read-modify-write it.
        let epinfo = rx + np + tx1 + tx2;
        let g = read_volatile(core_reg(GDFIFOCFG)) & 0xFFFF;
        write_volatile(core_reg(GDFIFOCFG), (epinfo << 16) | g);
        // Clock gating off: a gated PHY/core clock silently swallows
        // transfers.
        write_volatile(core_reg(PCGCCTL), 0);

        write_volatile(core_reg(GRSTCTL), GRSTCTL_TXFFLSH | (0x10 << 6));
    }
    poll(core_reg(GRSTCTL), GRSTCTL_TXFFLSH, 0, 10);
    unsafe { write_volatile(core_reg(GRSTCTL), GRSTCTL_RXFFLSH) };
    poll(core_reg(GRSTCTL), GRSTCTL_RXFFLSH, 0, 10);

    unsafe {
        // Interrupt lines stay masked at the NVIC; these masks only gate
        // the status bits this code polls.
        write_volatile(core_reg(DIEPMSK), 0);
        write_volatile(core_reg(DOEPMSK), 0);
        write_volatile(core_reg(DAINTMSK), 0);
        write_volatile(core_reg(GINTSTS), !0);
        write_volatile(core_reg(DCTL), read_volatile(core_reg(DCTL)) | DCTL_PWRONPRGDONE);
    }
    ms_wait(2);
    unsafe {
        // Attach: the PC now sees a device and starts enumeration.
        write_volatile(core_reg(DCTL), read_volatile(core_reg(DCTL)) & !DCTL_SFTDISCON);
    }

    let mut budget = Budget::new(wait_ms);
    while !configured() {
        pump();
        if budget.expired() {
            rprintln!(
                "usbdev: not configured after {} ms (DSTS={:#010x} GINTSTS={:#010x})",
                wait_ms,
                unsafe { read_volatile(core_reg(DSTS)) },
                unsafe { read_volatile(core_reg(GINTSTS)) }
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
