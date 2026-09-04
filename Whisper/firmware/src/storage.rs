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
    match backend() {
        Backend::Usb => usb::read_blocks(lba, dst, count),
        Backend::Sd => sd::read_blocks(lba, dst, count),
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::read_blocks(lba, dst, count),
        _ => -490,
    }
}

pub fn write_blocks(lba: u32, src: *const u8, count: u32) -> i32 {
    match backend() {
        Backend::Usb => usb::write_blocks(lba, src, count),
        Backend::Sd => sd::write_blocks(lba, src, count),
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::write_blocks(lba, src, count),
        _ => -490,
    }
}

/// (read bytes, read cycles, written bytes, write cycles) since last call.
pub fn stats_take() -> (u64, u64, u64, u64) {
    match backend() {
        Backend::Usb => usb::stats_take(),
        Backend::Sd => sd::stats_take(),
        #[cfg(feature = "mock-usb")]
        Backend::Mock => mockblk::stats_take(),
        _ => (0, 0, 0, 0),
    }
}
