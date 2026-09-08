//! USB DEVICE mode: a PC stands in for the model-image storage.
//!
//! Same core as usb.rs, opposite role: the USBHS device driver
//! (`hal::usbhs::device`) with this module's CDC-ACM identity on top.
//! Only one role can be live at a time -- they are the same peripheral --
//! so the build picks one (feature "mock-usb"). mockblk.rs speaks the
//! block protocol on top of the bulk pipes here.
//!
//! The interface is CDC-ACM rather than vendor-specific so that Linux binds
//! its in-tree cdc-acm driver: the port appears as /dev/ttyACMn, readable
//! and writable by the `dialout` group, with no udev rule and no root. A
//! vendor interface would need both. The class costs one notification
//! endpoint this code never sends on, and three class requests.
//!
//! Polled: the control requests, the bus-event pump and the framing above
//! the bulk pipes live here; the endpoint and FIFO plumbing is the driver's.

use core::ptr::{addr_of, addr_of_mut, read_volatile, write_volatile};

use rtt_target::rprintln;

use crate::hal::usbhs::device::{BusEvent, Config, Device, Direction, EndpointStatus, Ep0Out, Speed};
use crate::hal::usbhs::{elapsed_ms, now, Error};

pub const MPS0: usize = 64;
pub const MPS_BULK: usize = 512;
const EP_BULK: u8 = 1; // EP1 IN and EP1 OUT
const EP_NOTIFY: u8 = 2; // EP2 IN, CDC notifications (never sent)

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
    7, 0x05, 0x80 | EP_NOTIFY, 0x03, 16, 0, 16,
    // interface 1: CDC data, the two bulk pipes the block protocol uses
    9, 0x04, 1, 0, 2, 0x0A, 0x00, 0x00, 0,
    7, 0x05, EP_BULK, 0x02, (MPS_BULK & 0xFF) as u8, (MPS_BULK >> 8) as u8, 0,
    7, 0x05, 0x80 | EP_BULK, 0x02, (MPS_BULK & 0xFF) as u8, (MPS_BULK >> 8) as u8, 0,
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

/// The device driver, built on the first init() from the board's USBHS
/// singleton and kept for the rest of the run.
static mut DRIVER: Option<Device<'static>> = None;

fn dev() -> &'static mut Device<'static> {
    let slot = unsafe { &mut *addr_of_mut!(DRIVER) };
    if slot.is_none() {
        let usb = crate::board::get().usbhs.take().expect("USBHS is owned by the host driver");
        *slot = Some(Device::new_blocking(usb));
    }
    slot.as_mut().unwrap()
}

#[repr(C, align(4))]
struct Buf64([u8; 64]);
/// EP0 IN staging. The configuration descriptor is the longest control
/// response at 67 bytes, and EP0's PktCnt is 2 bits, so three packets of
/// MPS0 is both the ceiling the core can send and ample room.
#[repr(C, align(4))]
struct Buf192([u8; 3 * MPS0]);

/// SETUP landing zone.
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
static mut ADDRESS: u8 = 0;
/// Set when the host resets the port after we were configured: the link is
/// gone and every transfer should fail rather than hang.
static mut RESET_AFTER_CONFIG: bool = false;

pub fn configured() -> bool {
    unsafe { read_volatile(addr_of!(CONFIGURED)) }
}

// --- endpoint 0 ----------------------------------------------------------------

fn ep0_arm_setup() {
    dev().ep0_arm_setup(addr_of_mut!(SETUP) as *mut u8);
}

fn ep0_arm_out(len: usize) {
    dev().ep0_arm_out(addr_of_mut!(EP0OUT) as *mut u8, len);
}

/// Send up to 3 packets on EP0 IN and wait for the core to hand them over,
/// then arm the status OUT.
fn ep0_in(data: &[u8], req_len: usize) {
    let n = data.len().min(req_len).min(3 * MPS0);
    unsafe {
        let buf = &mut (*addr_of_mut!(EP0IN)).0;
        buf[..n].copy_from_slice(&data[..n]);
    }
    dev().ep0_in(addr_of!(EP0IN) as *const u8, n, 50);
    // The status stage is a zero-length OUT; arm it now so the host never
    // sees a NAK storm after a short descriptor.
    ep0_arm_out(0);
}

/// Zero-length IN: the status stage of a control transfer with no data.
fn ep0_status_in() {
    dev().ep0_in(addr_of!(EP0IN) as *const u8, 0, 50);
}

fn ep0_stall() {
    dev().ep0_stall();
    ep0_arm_setup();
}

fn activate_data_endpoints() {
    let d = dev();
    d.configure_bulk(EP_BULK, Direction::In, MPS_BULK as u16);
    d.configure_bulk(EP_BULK, Direction::Out, MPS_BULK as u16);
    d.configure_interrupt_in(EP_NOTIFY, 16);
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
                ep0_arm_out(MPS0);
                dev().ep0_out_wait(50);
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
            let addr = (val & 0x7F) as u8;
            unsafe { ADDRESS = addr };
            dev().set_address(addr);
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

/// Service bus events and endpoint 0. Touches no other endpoint: the bulk
/// transfer routines own their completions.
pub fn pump() {
    match dev().poll_bus() {
        BusEvent::Reset => {
            // A reset after we were configured means the host tore the
            // link down; the caller needs to see that rather than block
            // forever.
            unsafe {
                if CONFIGURED {
                    RESET_AFTER_CONFIG = true;
                }
                CONFIGURED = false;
            }
            ep0_arm_setup();
        }
        BusEvent::Enumerated(speed) => {
            rprintln!(
                "usbdev: enumerated, speed {} (0=HS 1=FS)",
                if speed == Speed::High { 0 } else { 1 }
            );
            ep0_arm_setup();
        }
        BusEvent::None => {}
    }

    // SETUP arrival and control data completions both land on EP0 OUT.
    if dev().ep0_out_event() == Ep0Out::Setup {
        handle_setup();
    }
}

// --- bulk transfers ---------------------------------------------------------------

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

const CYC_PER_MS: u64 = 128_000;

impl Budget {
    fn new(ms: u32) -> Budget {
        Budget {
            last: now(),
            acc: 0,
            limit: ms as u64 * CYC_PER_MS,
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
    dev().in_arm(EP_BULK, MPS_BULK, dma, len);
}

/// Non-blocking completion check for `ep_in_arm`; services the control
/// endpoint on the way. None while the transfer is still running.
fn ep_in_check(budget: &mut Budget) -> Option<i32> {
    match dev().in_status(EP_BULK) {
        EndpointStatus::Complete(_) => return Some(0),
        EndpointStatus::AhbError => {
            dev().abort(EP_BULK, Direction::In);
            return Some(-621);
        }
        EndpointStatus::Busy => {}
    }
    pump();
    if unsafe { RESET_AFTER_CONFIG } {
        dev().abort(EP_BULK, Direction::In);
        return Some(-623);
    }
    if budget.expired() {
        dev().abort(EP_BULK, Direction::In);
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
    dev().out_arm(EP_BULK, MPS_BULK, dma, cap)
}

fn ep_out_check(want: usize, budget: &mut Budget) -> Option<Result<usize, i32>> {
    match dev().out_status(EP_BULK, want) {
        EndpointStatus::Complete(got) => return Some(Ok(got)),
        EndpointStatus::AhbError => {
            dev().abort(EP_BULK, Direction::Out);
            return Some(Err(-621));
        }
        EndpointStatus::Busy => {}
    }
    pump();
    if unsafe { RESET_AFTER_CONFIG } {
        dev().abort(EP_BULK, Direction::Out);
        return Some(Err(-623));
    }
    if budget.expired() {
        dev().abort(EP_BULK, Direction::Out);
        return Some(Err(-622));
    }
    None
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
    let s = dev().snapshot(EP_BULK);
    rprintln!(
        "usbdev[{}]: GINTSTS={:#010x} DSTS={:#010x} DCTL={:#010x} GHWCFG3={:#010x} GDFIFOCFG={:#010x}",
        tag, s.gintsts, s.dsts, s.dctl, s.ghwcfg3, s.gdfifocfg
    );
    rprintln!(
        "usbdev[{}]: GRXFSIZ={:#010x} GNPTXFSIZ={:#010x} TXF1={:#010x} TXF2={:#010x}",
        tag, s.grxfsiz, s.gnptxfsiz, s.dieptxf1, s.dieptxf2
    );
    rprintln!(
        "usbdev[{}]: DIEPCTL1={:#010x} DIEPTSIZ1={:#010x} DIEPINT1={:#010x} DTXFSTS1={:#010x}",
        tag, s.diepctl, s.dieptsiz, s.diepint, s.dtxfsts
    );
    rprintln!(
        "usbdev[{}]: DOEPCTL1={:#010x} DOEPTSIZ1={:#010x} DOEPINT1={:#010x}",
        tag, s.doepctl, s.doeptsiz, s.doepint
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

    // FIFO carve-up: EP0 256 words, EP1 IN bulk 1024, EP2 IN notifications 64.
    let config = Config {
        rx_fifo_words: 640,
        tx_fifo_words: [256, 1024, 64],
    };
    if let Err(e) = dev().power_up(config) {
        return match e {
            Error::NoVbus => -600,
            Error::Xo24mTimeout => -601,
            Error::CoreNotResponding => -602,
            Error::AhbNotIdle => -603,
            Error::ResetTimeout => -604,
            Error::ModeRefused => {
                rprintln!("usbdev: device mode refused");
                dev().power_down();
                -630
            }
        };
    }
    // Attach: the PC now sees a device and starts enumeration.
    dev().attach();

    let mut budget = Budget::new(wait_ms);
    while !configured() {
        pump();
        if budget.expired() {
            let s = dev().snapshot(EP_BULK);
            rprintln!(
                "usbdev: not configured after {} ms (DSTS={:#010x} GINTSTS={:#010x})",
                wait_ms, s.dsts, s.gintsts
            );
            dev().power_down();
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
/// has written out of its FIFO behind a two-packet margin for a write
/// still on its way through the bus.
pub fn recv_landed(x: &Xfer) -> usize {
    let left = dev().out_remaining(EP_BULK);
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

/// Unused elapsed-time helper kept for callers with a single deadline.
#[allow(dead_code)]
fn deadline_passed(start: u32, ms: u32) -> bool {
    elapsed_ms(start, ms)
}
