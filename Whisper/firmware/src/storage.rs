//! Block-device dispatch: the model image is the same raw 512-byte-block
//! image whether it sits on a USB stick (usb.rs, USB host mode) or on the
//! SD card (sd.rs). Probe order is USB first -- with no 5 V wired to VBUS
//! it fails in ~100 ms (-600) -- then SD. Everything above this layer
//! (app.rs, the mailbox SD commands, the host tape tools) is
//! backend-agnostic.

use crate::{sd, usb};
use rtt_target::rprintln;

pub const BLOCK: usize = 512;

#[derive(Clone, Copy, PartialEq)]
pub enum Backend {
    None,
    Sd,
    Usb,
}

static mut BACKEND: Backend = Backend::None;

pub fn backend() -> Backend {
    unsafe { BACKEND }
}

pub fn name() -> &'static str {
    match backend() {
        Backend::Usb => "USB stick",
        Backend::Sd => "SD card",
        Backend::None => "none",
    }
}

/// Probe USB then SD. Returns 0 with a backend selected, or the SD error
/// (the USB code is logged; SD is the last resort and its code is the one
/// existing tooling knows). Pins/power of a failed probe are left clean.
pub fn init() -> i32 {
    unsafe { BACKEND = Backend::None };
    let urc = usb::init();
    if urc == 0 {
        unsafe { BACKEND = Backend::Usb };
        return 0;
    }
    if urc != -600 {
        // -600 is plain "nothing wired"; anything else means a stick was
        // in reach and failed partway -- worth a loud line.
        rprintln!("storage: usb host failed rc={}, trying SD", urc);
    }
    let src = sd::init();
    if src == 0 {
        unsafe { BACKEND = Backend::Sd };
        return 0;
    }
    src
}

pub fn read_blocks(lba: u32, dst: *mut u8, count: u32) -> i32 {
    match backend() {
        Backend::Usb => usb::read_blocks(lba, dst, count),
        Backend::Sd => sd::read_blocks(lba, dst, count),
        Backend::None => -490,
    }
}

pub fn write_blocks(lba: u32, src: *const u8, count: u32) -> i32 {
    match backend() {
        Backend::Usb => usb::write_blocks(lba, src, count),
        Backend::Sd => sd::write_blocks(lba, src, count),
        Backend::None => -490,
    }
}

/// (read bytes, read cycles, written bytes, write cycles) since last call.
pub fn stats_take() -> (u64, u64, u64, u64) {
    match backend() {
        Backend::Usb => usb::stats_take(),
        Backend::Sd => sd::stats_take(),
        Backend::None => (0, 0, 0, 0),
    }
}
