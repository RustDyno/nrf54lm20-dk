//! Block-device dispatch: the model image is the same raw 512-byte-block
//! image whether it sits on a USB stick (usb.rs, USB host mode) or on the
//! SD card (sd.rs). The stick is probed up to three times (with no 5 V
//! wired to VBUS a probe fails in ~100 ms with -600); the SD card is only
//! tried after that with the `sd-card` feature. Everything above this
//! layer (app.rs, the mailbox SD commands, the host tape tools) is
//! backend-agnostic.

#[cfg(feature = "mock-usb")]
use crate::mockblk;
use crate::{sd, usb};
use rtt_target::rprintln;

pub const BLOCK: usize = 512;

#[derive(Clone, Copy, PartialEq)]
pub enum Backend {
    None,
    Sd,
    Usb,
    /// Blocks served by a PC over the USB device-mode link (feature
    /// "mock-usb"). Development rig only -- see mockblk.rs.
    Mock,
}

static mut BACKEND: Backend = Backend::None;

pub fn backend() -> Backend {
    unsafe { BACKEND }
}

pub fn name() -> &'static str {
    match backend() {
        Backend::Usb => "USB stick",
        Backend::Sd => "SD card",
        Backend::Mock => "host image (mock USB)",
        Backend::None => "none",
    }
}

/// Stick probes before giving up: one that is still enumerating after
/// power-up answers on a later try.
pub const USB_TRIES: u32 = 3;

/// Probe the USB stick USB_TRIES times, then (with `sd-card`) the SD
/// card. Returns 0 with a backend selected, else the last probe's error.
/// Pins/power of a failed probe are left clean.
pub fn init() -> i32 {
    unsafe { BACKEND = Backend::None };

    // The mock and the USB stick are the same peripheral in opposite
    // roles, so a mock build never probes host mode: forcing host after
    // device mode would tear down a working link.
    #[cfg(feature = "mock-usb")]
    {
        let mrc = mockblk::init();
        if mrc == 0 {
            unsafe { BACKEND = Backend::Mock };
            return 0;
        }
        // No SD fallback in a mock build: the card is not wired on this
        // rig, and its probe spends minutes in pin diagnostics that only
        // delay the real error.
        rprintln!("storage: mock usb failed rc={} (no SD fallback in mock builds)", mrc);
        return mrc;
    }

    #[cfg(not(feature = "mock-usb"))]
    {
    let mut urc = -600;
    for attempt in 1..=USB_TRIES {
        urc = usb::init();
        if urc == 0 {
            unsafe { BACKEND = Backend::Usb };
            return 0;
        }
        // -600 is plain "nothing wired"; anything else means a stick was
        // in reach and failed partway.
        rprintln!("storage: usb stick probe {}/{} failed rc={}", attempt, USB_TRIES, urc);
        if attempt < USB_TRIES {
            cortex_m::asm::delay(64_000_000); // 0.5 s at 128 MHz
        }
    }
    #[cfg(feature = "sd-card")]
    {
        rprintln!("storage: trying the SD card");
        let src = sd::init();
        if src == 0 {
            unsafe { BACKEND = Backend::Sd };
            return 0;
        }
        return src;
    }
    #[cfg(not(feature = "sd-card"))]
    urc
    }
}

pub fn read_blocks(lba: u32, dst: *mut u8, count: u32) -> i32 {
    if busy() {
        rprintln!("storage: blocking read with a transfer pending");
        return -495;
    }
    match backend() {
        Backend::Usb => usb::read_blocks(lba, dst, count),
        Backend::Sd => sd::read_blocks(lba, dst, count),
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::read_blocks(lba, dst, count),
        _ => -490,
    }
}

pub fn write_blocks(lba: u32, src: *const u8, count: u32) -> i32 {
    if busy() {
        rprintln!("storage: blocking write with a transfer pending");
        return -495;
    }
    match backend() {
        Backend::Usb => usb::write_blocks(lba, src, count),
        Backend::Sd => sd::write_blocks(lba, src, count),
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::write_blocks(lba, src, count),
        _ => -490,
    }
}

// --- split-phase transfers --------------------------------------------------
//
// `read_start`/`write_start` issue a transfer whose data phase runs on the
// backend's DMA; `poll` advances it without blocking (call it between
// units of CPU work); `finish` blocks for the result. One transfer in
// flight at a time: a second start, or a blocking read/write while one
// is pending, returns -495 rather than quietly serializing, so a missing
// finish() shows up as an error at the call site. The SD backend has no
// split path and completes inside start().

static mut PENDING: bool = false;
#[cfg(feature = "sd-card")]
static mut SD_RC: i32 = 0;

pub fn busy() -> bool {
    unsafe { PENDING }
}

pub fn read_start(lba: u32, dst: *mut u8, count: u32) -> i32 {
    if busy() {
        return -495;
    }
    let rc = match backend() {
        Backend::Usb => usb::read_start(lba, dst, count),
        #[cfg(feature = "sd-card")]
        Backend::Sd => {
            unsafe { SD_RC = sd::read_blocks(lba, dst, count) };
            0
        }
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::read_start(lba, dst, count),
        _ => -490,
    };
    if rc == 0 {
        unsafe { PENDING = true };
    }
    rc
}

pub fn write_start(lba: u32, src: *const u8, count: u32) -> i32 {
    if busy() {
        return -495;
    }
    let rc = match backend() {
        Backend::Usb => usb::write_start(lba, src, count),
        #[cfg(feature = "sd-card")]
        Backend::Sd => {
            unsafe { SD_RC = sd::write_blocks(lba, src, count) };
            0
        }
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::write_start(lba, src, count),
        _ => -490,
    };
    if rc == 0 {
        unsafe { PENDING = true };
    }
    rc
}

/// True once the pending transfer is complete (or none is pending).
pub fn poll() -> bool {
    if !busy() {
        return true;
    }
    match backend() {
        Backend::Usb => usb::xfer_poll(),
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::xfer_poll(),
        _ => true,
    }
}

/// Bytes of the pending read that are already in memory (a conservative
/// count read from the DMA engine), so a consumer can start on the head
/// of a transfer while its tail is still landing. Meaningful while
/// poll() is false; backends without a live count report everything.
pub fn landed() -> usize {
    if !busy() {
        return usize::MAX;
    }
    match backend() {
        Backend::Usb => usb::xfer_landed(),
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::xfer_landed(),
        _ => usize::MAX,
    }
}

/// Wait for the pending transfer and return its result (0 if none).
pub fn finish() -> i32 {
    if !busy() {
        return 0;
    }
    unsafe { PENDING = false };
    match backend() {
        Backend::Usb => usb::xfer_finish(),
        #[cfg(feature = "sd-card")]
        Backend::Sd => unsafe { SD_RC },
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::xfer_finish(),
        _ => -490,
    }
}

/// (read bytes, read cycles, written bytes, write cycles) since last call.
/// Cycles are CPU time spent inside storage calls: for split-phase
/// transfers that is only the part the overlap did not hide.
pub fn stats_take() -> (u64, u64, u64, u64) {
    match backend() {
        Backend::Usb => usb::stats_take(),
        Backend::Sd => sd::stats_take(),
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::stats_take(),
        _ => (0, 0, 0, 0),
    }
}
