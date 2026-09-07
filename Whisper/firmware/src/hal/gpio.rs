//! Pin helpers for the drivers in this tree.
//!
//! Inside embassy-nrf a driver configures its pins through the crate-private
//! `SealedPin` methods (`conf()`, `set_high()`, `block()`); from outside the
//! crate only the pin's PSEL bits are reachable, which carry the same
//! information. These helpers rebuild the port lookup from them. Upstream,
//! every use maps one-to-one onto the `SealedPin` method of the same name.

use embassy_nrf::gpio::{AnyPin, Pin};
use embassy_nrf::pac;
use embassy_nrf::pac::common::{Reg, RW};
use embassy_nrf::pac::gpio::regs::PinCnf;
use embassy_nrf::pac::shared::regs::Psel;
use embassy_nrf::pac::shared::vals::Connect;

/// The PSEL value that disconnects a peripheral signal from every pin.
pub const DISCONNECTED: Psel = {
    let mut v = Psel(0);
    v.set_connect(Connect::Disconnected);
    v
};

/// The GPIO register block of port `port`.
pub fn block(port: u8) -> pac::gpio::Gpio {
    match port {
        0 => pac::P0,
        1 => pac::P1,
        2 => pac::P2,
        3 => pac::P3,
        _ => unreachable!(),
    }
}

/// The pin's PIN_CNF register.
pub fn conf(pin: &AnyPin) -> Reg<PinCnf, RW> {
    let psel = pin.psel_bits();
    block(psel.port()).pin_cnf(psel.pin() as usize)
}

pub fn set_high(pin: &AnyPin) {
    let psel = pin.psel_bits();
    block(psel.port()).outset().write(|w| w.set_pin(psel.pin() as usize, true));
}

pub fn set_low(pin: &AnyPin) {
    let psel = pin.psel_bits();
    block(psel.port()).outclr().write(|w| w.set_pin(psel.pin() as usize, true));
}

pub fn is_high(pin: &AnyPin) -> bool {
    let psel = pin.psel_bits();
    block(psel.port()).in_().read().pin(psel.pin() as usize)
}

/// Return a pin to its reset state (input buffer disconnected, no pull).
pub fn deconfigure(pin: &AnyPin) {
    conf(pin).write(|w| w.set_input(pac::gpio::vals::Input::Disconnect));
}
