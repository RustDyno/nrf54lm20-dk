//! Drivers written in embassy-nrf's shape, for peripherals (or modes) the
//! HAL does not cover on the nRF54LM20, so that each can be offered
//! upstream as it stands:
//!
//! - `pdm`: the nRF54L PDM (PDM20/PDM21) with a blocking double-buffer
//!   sampler. embassy-nrf's `pdm` module is not built for `_nrf54l`; the
//!   nRF54L block has PRESCALER/CLKSELECT/RATIO instead of PDMCLKCTRL and
//!   a byte-counted MAXCNT.
//! - `usbhs`: the USBHS block driven polled, in both roles. embassy-nrf has
//!   an interrupt-driven device driver for it (via embassy-usb-synopsys-otg)
//!   and no host mode at all.
//! - `spim`: SPIM with a software chip select held across transfers, a
//!   bit-banged low-speed phase on the same pins, and the erratum [8]
//!   workaround for the 128 MHz instance.
//! - `axons`: the register block of the Axon NPU, absent from the SVD and
//!   so from nrf-pac; written the way chiptool would generate it.
//!
//! Conventions followed: a `Config` with `Default`, an `Error` enum, a
//! driver struct owning `Peri<'d, _>` singletons, `new_blocking`
//! constructors for interrupt-free drivers, sealed `Instance` traits with
//! `regs()`, `Drop` returning pins and PSELs to reset state, and the
//! `blocking_` prefix on APIs that spin. `gpio` carries the few pin
//! helpers that are `pub(crate)` inside embassy-nrf (`SealedPin::conf`
//! and friends); upstream, the drivers would call those instead.

pub mod axons;
pub mod gpio;
pub mod pdm;
