//! USBHS: the Synopsys DWC2 dual-role core behind the nRF54LM20's USB
//! port, driven polled in either role.
//!
//! The datasheet's feature list says "USB 2.0 device" and "no OTG", but
//! the hardwired configuration register disagrees on the part that
//! matters: GHWCFG2.OTGMODE = 2 (non-HNP/non-SRP OTG = HOST and device),
//! 16 host channels, internal DMA, and GUSBCFG.FORCEHSTMODE is documented
//! as valid exactly for that OTG mode. [`host::Host`] forces host mode and
//! drives the root port directly (no hub support); [`device::Device`]
//! forces device mode. Only one can exist: both take the USBHS singleton.
//!
//! Board reality on the DK: the chip has no VBUS sourcing -- the VBUS pin
//! is an INPUT that powers the PHY's signaling rail (VREGUSB). Host mode
//! therefore needs 5 V fed into the connector's VBUS from the board's own
//! 5 V rail; the device role gets it from the PC. Either way, no VBUS is
//! reported as [`Error::NoVbus`] within ~100 ms of power-up.
//!
//! Registers are the PAC's USBHS wrapper, VREGUSB, CLOCK and USBHSCORE
//! blocks. No interrupts: the NVIC line stays masked and every wait is
//! bounded on the DWT cycle counter (an upstream version would take its
//! deadlines from embassy-time).

// A build uses one role; the other's API is still part of the driver.
#[cfg_attr(not(feature = "mock-usb"), allow(dead_code))]
pub mod device;
#[cfg_attr(feature = "mock-usb", allow(dead_code))]
pub mod host;

use core::marker::PhantomData;

use embassy_nrf::pac;
use embassy_nrf::pac::usbhscore::regs;
use embassy_nrf::pac::usbhscore::vals::GintstsCurmod;
use embassy_nrf::{Peri, PeripheralType};

/// Bring-up error: the stage that timed out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// The 24 MHz PHY reference never started.
    Xo24mTimeout,
    /// VREGUSB reported no VBUS: nothing is wired to the port.
    NoVbus,
    /// The core did not answer with its Synopsys signature.
    CoreNotResponding,
    /// The AHB never went idle for the soft reset.
    AhbNotIdle,
    /// The soft-reset handshake never completed.
    ResetTimeout,
    /// The core refused the requested role.
    ModeRefused,
}

const CYC_PER_MS: u32 = 128_000; // DWT at the 128 MHz core clock

/// Spin for `ms` milliseconds.
pub(crate) fn delay_ms(ms: u32) {
    let start = cortex_m::peripheral::DWT::cycle_count();
    while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < ms * CYC_PER_MS {}
}

/// Poll `cond` until it holds or `ms` milliseconds elapse.
pub(crate) fn wait_for(cond: impl Fn() -> bool, ms: u32) -> bool {
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

/// A DWT cycle stamp, for deadlines spanning several polls.
pub(crate) fn now() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}

pub(crate) fn elapsed_ms(start: u32, ms: u32) -> bool {
    now().wrapping_sub(start) >= ms * CYC_PER_MS
}

/// Power, clock and core state shared by both roles.
pub(crate) struct Platform<'d> {
    pub(crate) core: pac::usbhscore::Usbhscore,
    pub(crate) wrap: pac::usbhs::Usbhs,
    _phantom: PhantomData<&'d ()>,
}

impl<'d> Platform<'d> {
    pub(crate) fn new<T: Instance>(_usb: Peri<'d, T>) -> Self {
        Self {
            core: T::core_regs(),
            wrap: T::wrap_regs(),
            _phantom: PhantomData,
        }
    }

    /// 24 MHz PHY reference, VBUS detection, wrapper enable, and the
    /// v4.20a+ core soft-reset handshake. Leaves the core reset and idle,
    /// in whatever mode the hardware defaults to.
    pub(crate) fn power_up(&mut self) -> Result<(), Error> {
        // The USB PHY reference is the 24 MHz PLL off HFXO (PHY.CLOCK reset
        // FSEL already selects 24 MHz); nothing else in this firmware
        // starts the crystal.
        pac::CLOCK.tasks_xo24mstart().write_value(1);
        if !wait_for(|| pac::CLOCK.events_xo24mstarted().read() != 0, 100) {
            return Err(Error::Xo24mTimeout);
        }

        // VBUS check first: without 5 V on the VBUS pin the PHY has no
        // signaling rail and there is nothing to talk to. VBUSDETECTED is
        // an EDGE event: if VREGUSB is already running (a previous
        // power-up this power cycle; soft reset does not fully reset
        // peripherals, erratum [63]) a bare re-START never re-fires it --
        // hardware-observed as a silent no-VBUS with VBUS present. Stop
        // first to force a fresh detection cycle.
        let vreg = pac::VREGUSB;
        vreg.tasks_stop().write_value(1);
        delay_ms(2);
        vreg.events_vbusdetected().write_value(0);
        vreg.tasks_start().write_value(1);
        if !wait_for(|| vreg.events_vbusdetected().read() != 0, 100) {
            self.power_down();
            return Err(Error::NoVbus);
        }

        self.wrap.enable().write(|w| {
            w.set_core(true);
            w.set_phy(true);
        });
        delay_ms(1); // PHY clock start (the H20 quirk waits 45 us here)
        self.wrap.tasks_start().write_value(1);
        // HARDWARE-CONFIRMED: the wrapper's STATUS.CORE bit never asserts
        // on this part even with the core fully alive and register access
        // working, so it cannot be the readiness gate. GSNPSID answering
        // with the Synopsys signature ("OT" in the top bytes) is.
        let core = self.core;
        if !wait_for(|| core.gsnpsid().read() & 0xFFFF_0000 == 0x4F54_0000, 50) {
            self.power_down();
            return Err(Error::CoreNotResponding);
        }

        // HARDWARE-CONFIRMED: this core is DWC2 v5.00b (GSNPSID
        // 0x4F54500B), and since v4.20a soft reset is a handshake --
        // CSftRst does NOT self-clear. The core sets CSftRstDone (absent
        // from the datasheet, present in the SVD) and software must then
        // write both bits back to 0. Polling for self-clear hangs forever
        // with the reset long since finished.
        if !wait_for(|| core.grstctl().read().ahbidle(), 50) {
            self.power_down();
            return Err(Error::AhbNotIdle);
        }
        core.grstctl().write(|w| w.set_csftrst(true));
        if !wait_for(|| core.grstctl().read().csftrstdone(), 50) {
            self.power_down();
            return Err(Error::ResetTimeout);
        }
        core.grstctl().write_value(regs::Grstctl(0));
        if !wait_for(|| core.grstctl().read().ahbidle(), 50) {
            self.power_down();
            return Err(Error::ResetTimeout);
        }
        Ok(())
    }

    pub(crate) fn power_down(&mut self) {
        self.wrap.tasks_stop().write_value(1);
        self.wrap.enable().write(|w| {
            w.set_core(false);
            w.set_phy(false);
        });
        pac::VREGUSB.tasks_stop().write_value(1);
    }

    /// Force the role. The mode change is specified to take up to 25 ms.
    pub(crate) fn force_mode(&mut self, host: bool) -> Result<(), Error> {
        let core = self.core;
        core.gusbcfg().modify(|w| {
            w.set_forcehstmode(host);
            w.set_forcedevmode(!host);
        });
        let want = if host { GintstsCurmod::Host } else { GintstsCurmod::Device };
        if !wait_for(|| core.gintsts().read().curmod() == want, 60) {
            return Err(Error::ModeRefused);
        }
        Ok(())
    }

    /// Flush all TX FIFOs and the RX FIFO.
    pub(crate) fn flush_fifos(&mut self) {
        let core = self.core;
        core.grstctl().write(|w| {
            w.set_txfflsh(true);
            w.set_txfnum(pac::usbhscore::vals::GrstctlTxfnum::Txf16); // all TX FIFOs
        });
        wait_for(|| !core.grstctl().read().txfflsh(), 10);
        core.grstctl().write(|w| w.set_rxfflsh(true));
        wait_for(|| !core.grstctl().read().rxfflsh(), 10);
    }
}

pub(crate) trait SealedInstance {
    fn wrap_regs() -> pac::usbhs::Usbhs;
    fn core_regs() -> pac::usbhscore::Usbhscore;
}

/// USBHS peripheral instance.
#[allow(private_bounds)]
pub trait Instance: SealedInstance + PeripheralType + 'static + Send {}

impl SealedInstance for embassy_nrf::peripherals::USBHS {
    fn wrap_regs() -> pac::usbhs::Usbhs {
        pac::USBHS
    }
    fn core_regs() -> pac::usbhscore::Usbhscore {
        pac::USBHSCORE
    }
}

impl Instance for embassy_nrf::peripherals::USBHS {}
