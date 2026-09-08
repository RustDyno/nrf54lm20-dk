//! SPIM with a software chip select, a bit-banged low-speed phase on the
//! same pins, and the erratum [8] workaround: the transport an SD card in
//! SPI mode needs.
//!
//! Three things keep this off the HAL's `Spim`:
//!
//! - The SD protocol holds CS asserted across many transfers, so CSN is a
//!   GPIO here and the SPIM's CSN is disconnected. On this part the SPIM's
//!   EVENTS_END is tied to the hardware-CSN transaction framing and never
//!   fires with CSN disconnected (hardware-observed: both DMA END events
//!   set, EVENTS_END stuck 0). Completion is therefore both DMA
//!   directions done, followed by STOP to close the engine's transaction
//!   state.
//! - The 128 MHz instance cannot divide below ~1 MHz, above the 400 kHz
//!   SD initialization cap, so the card's init phase is bit-banged on the
//!   same pins at ~250 kHz before the peripheral takes over.
//! - Erratum [8] "SPIM: Wrong data is transmitted on MOSI" (Engineering
//!   B): with CPHA=0 and PRESCALER > 2, a first transmitted bit of 1
//!   corrupts the data. Workaround per the errata doc: CSNDUR >=
//!   PRESCALER/2 + 1, write 0x82 to offset 0xC84 before each START, and
//!   0x00 back once STARTED has fired. That register is not in the SVD.
//!
//! Mode 0, MSB first, over-read character 0xFF. DMA waits are bounded on
//! the DWT cycle counter; a transfer that times out marks the driver
//! faulted and every later transfer fails fast (reads 0xFF) rather than
//! piling timeouts on every byte.

#![allow(dead_code)]

use core::marker::PhantomData;

use embassy_nrf::gpio::{AnyPin, Pin as GpioPin};
use embassy_nrf::pac;
use embassy_nrf::pac::common::{Reg, RW};
use embassy_nrf::pac::gpio::vals as gpiovals;
use embassy_nrf::pac::shared::vals::Connect;
use embassy_nrf::pac::spim::vals;
use embassy_nrf::{Peri, PeripheralType};

use super::gpio::{self, DISCONNECTED};

/// Erratum [8] workaround register, absent from the SVD.
const ERRATA8_OFFSET: usize = 0xC84;

/// Pad drive for SCK and MOSI once the peripheral takes over.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Drive {
    /// Standard drive on both halves.
    Standard,
    /// Extra-high drive on both halves with the fast-pad slew at its
    /// highest (GPIOHSPADCTRL.BIAS): what 32 MHz on the P2 pads needs.
    /// Only the P2 pads have it.
    ExtraHigh,
}

/// SPIM driver Config.
#[derive(Clone)]
pub struct Config {
    /// SCK = instance clock / divisor (4..126 on the 128 MHz instance,
    /// 2..126 on the 16 MHz ones).
    pub divisor: u8,
    /// Over-read character: what MOSI carries past the end of `tx`.
    pub orc: u8,
    /// Drive for SCK and MOSI in the peripheral phase. The bit-banged
    /// phase always uses standard drive: extra-high edges ring hard on
    /// jumper wiring, and a ring on SCK re-crossing the card's threshold
    /// is a phantom clock.
    pub drive: Drive,
    /// Half period of the bit-banged clock in DWT cycles (256 at 128 MHz
    /// is 2 us, i.e. 250 kHz).
    pub bitbang_half_cycles: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            divisor: 4,
            orc: 0xFF,
            drive: Drive::Standard,
            bitbang_half_cycles: 256,
        }
    }
}

/// SPIM error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A DMA transfer did not complete within 20 ms; the driver is
    /// faulted from here on.
    Timeout,
    /// A transfer was refused because an earlier one timed out.
    Faulted,
}

/// Which stage of a DMA transfer timed out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Started,
    DmaEnd,
}

/// The register state behind a fault, for a diagnostic dump.
#[derive(Debug, Clone, Copy)]
pub struct FaultSnapshot {
    pub stage: Stage,
    pub events_started: u32,
    pub events_end: u32,
    pub enable: u32,
    pub prescaler: u32,
    pub config: u32,
    pub dma_rx_end: u32,
    pub dma_rx_ready: u32,
    pub dma_rx_buserror: u32,
    pub dma_rx_match: [u32; 4],
    pub dma_tx_end: u32,
    pub dma_tx_ready: u32,
    pub dma_tx_buserror: u32,
    pub rx_buserror_address: u32,
    pub tx_buserror_address: u32,
}

/// One of the driver's lines, for the wiring diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Line {
    Sck,
    Mosi,
    Miso,
    Cs,
}

/// SPIM with software chip select.
pub struct SpimSoftCs<'d> {
    r: pac::spim::Spim,
    sck: Peri<'d, AnyPin>,
    mosi: Peri<'d, AnyPin>,
    miso: Peri<'d, AnyPin>,
    cs: Peri<'d, AnyPin>,
    config: Config,
    /// True from construction until [`SpimSoftCs::engage`] hands the pins
    /// to the peripheral. All traffic funnels through
    /// [`SpimSoftCs::blocking_transfer`], so both phases share every code
    /// path.
    bitbang: bool,
    /// Bit-bang with the two data pins' roles exchanged (diagnostic).
    swapped: bool,
    fault: Option<FaultSnapshot>,
    _phantom: PhantomData<&'d ()>,
}

/// Real pointers even for zero-length directions: empty-slice pointers
/// are dangling, and the nRF54 EasyDMA validates bus addresses
/// (TERMINATEONBUSERROR machinery) where nRF52 did not.
static mut DMA_DUMMY: [u8; 4] = [0; 4];

#[inline]
fn dwt_delay(cycles: u32) {
    let start = cortex_m::peripheral::DWT::cycle_count();
    while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < cycles {}
}

impl<'d> SpimSoftCs<'d> {
    /// Create the driver in its bit-banged phase: SCK/MOSI/CS outputs
    /// (SCK idle low, MOSI/CS idle high), MISO input with pull-up,
    /// standard drive, the peripheral disabled.
    pub fn new_blocking<T: Instance>(
        _spim: Peri<'d, T>,
        sck: Peri<'d, impl GpioPin>,
        mosi: Peri<'d, impl GpioPin>,
        miso: Peri<'d, impl GpioPin>,
        cs: Peri<'d, impl GpioPin>,
        config: Config,
    ) -> Self {
        let mut this = Self {
            r: T::regs(),
            sck: sck.into(),
            mosi: mosi.into(),
            miso: miso.into(),
            cs: cs.into(),
            config,
            bitbang: true,
            swapped: false,
            fault: None,
            _phantom: PhantomData,
        };
        this.enter_bitbang();
        this
    }

    /// (Re)configure the pins for the bit-banged phase and disable the
    /// peripheral. The current data-pin role assignment is kept.
    pub fn enter_bitbang(&mut self) {
        self.r.enable().write(|w| w.set_enable(vals::Enable::Disabled));
        self.bitbang = true;
        gpio::set_low(&self.sck);
        gpio::set_high(&self.cs);
        Self::cnf_output(&self.sck, Drive::Standard);
        Self::cnf_output(&self.cs, Drive::Standard);
        self.config_data_pins();
    }

    fn cnf_output(pin: &AnyPin, drive: Drive) {
        gpio::conf(pin).write(|w| {
            w.set_dir(gpiovals::Dir::Output);
            w.set_input(gpiovals::Input::Disconnect);
            if drive == Drive::ExtraHigh {
                w.set_drive0(gpiovals::Drive::E);
                w.set_drive1(gpiovals::Drive::E);
            }
        });
    }

    fn cnf_input_pullup(pin: &AnyPin) {
        gpio::conf(pin).write(|w| {
            w.set_dir(gpiovals::Dir::Input);
            w.set_input(gpiovals::Input::Connect);
            w.set_pull(gpiovals::Pull::Pullup);
        });
    }

    fn data_out(&self) -> &AnyPin {
        if self.swapped { &self.miso } else { &self.mosi }
    }

    fn data_in(&self) -> &AnyPin {
        if self.swapped { &self.mosi } else { &self.miso }
    }

    /// Configure the data pins for the current role assignment.
    fn config_data_pins(&mut self) {
        gpio::set_high(self.data_out());
        Self::cnf_output(self.data_out(), Drive::Standard);
        // PIN_CNF.DIR is the same physical register as DIR: this also
        // turns the former output back into an input.
        Self::cnf_input_pullup(self.data_in());
    }

    /// Diagnostic: bit-bang with the two data pins' roles exchanged. If
    /// the far side answers only like this, its data wires are crossed.
    pub fn swap_data_pins(&mut self, swapped: bool) {
        self.swapped = swapped;
        self.config_data_pins();
    }

    /// Assert chip select (drive CS low).
    pub fn cs_assert(&mut self) {
        gpio::set_low(&self.cs);
    }

    /// Release chip select (drive CS high).
    pub fn cs_release(&mut self) {
        gpio::set_high(&self.cs);
    }

    /// Hand SCK/MOSI/MISO to the peripheral (CS stays a GPIO) and enable
    /// it with the configured divisor and drive. Clears a fault.
    pub fn engage(&mut self) {
        let r = self.r;
        let cfg = self.config.clone();
        if cfg.drive == Drive::ExtraHigh {
            pac::GPIOHSPADCTRL_S.bias().write(|w| w.set_hsbias(0x3));
        }
        Self::cnf_output(&self.sck, cfg.drive);
        Self::cnf_output(&self.mosi, cfg.drive);
        r.psel().sck().write_value(self.sck.psel_bits());
        r.psel().mosi().write_value(self.mosi.psel_bits());
        r.psel().miso().write_value(self.miso.psel_bits());
        // CS is ours: leave CSN disconnected.
        r.psel().csn().write(|w| w.set_connect(Connect::Disconnected));
        r.config().write(|w| {
            // mode 0, MSB first
            w.set_order(vals::Order::MsbFirst);
            w.set_cpha(vals::Cpha::Leading);
            w.set_cpol(vals::Cpol::ActiveHigh);
        });
        r.orc().write(|w| w.set_orc(cfg.orc));
        r.prescaler().write(|w| w.set_divisor(cfg.divisor));
        r.iftiming().csndur().write(|w| w.set_csndur(cfg.divisor / 2 + 1)); // erratum [8]
        r.enable().write(|w| w.set_enable(vals::Enable::Enabled));
        self.bitbang = false;
        self.fault = None;
    }

    /// True while the pins are bit-banged.
    pub fn is_bitbang(&self) -> bool {
        self.bitbang
    }

    /// The fault that put the driver in fail-fast mode, if any.
    pub fn fault(&self) -> Option<FaultSnapshot> {
        self.fault
    }

    /// One byte over the bit-banged pins, mode 0: MOSI changes on the
    /// falling edge, both sides sample on the rising edge.
    pub fn bitbang_byte(&mut self, tx: u8) -> u8 {
        let half = self.config.bitbang_half_cycles;
        let mut rx = 0u8;
        for bit in (0..8).rev() {
            if tx & (1 << bit) != 0 {
                gpio::set_high(self.data_out());
            } else {
                gpio::set_low(self.data_out());
            }
            dwt_delay(half);
            gpio::set_high(&self.sck);
            if gpio::is_high(self.data_in()) {
                rx |= 1 << bit;
            }
            dwt_delay(half);
            gpio::set_low(&self.sck);
        }
        rx
    }

    /// One full-duplex transaction: send `tx` (the over-read character
    /// past its end), receive `rx.len()` bytes into `rx`. Bit-banged or
    /// DMA-driven, per phase.
    pub fn blocking_transfer(&mut self, tx: &[u8], rx: &mut [u8]) -> Result<(), Error> {
        if self.bitbang {
            for &b in tx {
                self.bitbang_byte(b);
            }
            for r in rx.iter_mut() {
                *r = self.bitbang_byte(0xFF);
            }
            return Ok(());
        }
        if self.fault.is_some() {
            for r in rx.iter_mut() {
                *r = 0xFF;
            }
            return Err(Error::Faulted);
        }
        let r = self.r;
        let dummy = core::ptr::addr_of_mut!(DMA_DUMMY) as u32;
        let txp = if tx.is_empty() { dummy } else { tx.as_ptr() as u32 };
        let rxp = if rx.is_empty() { dummy } else { rx.as_mut_ptr() as u32 };
        r.dma().tx().ptr().write_value(txp);
        r.dma().tx().maxcnt().write(|w| w.set_maxcnt(tx.len() as u16));
        r.dma().rx().ptr().write_value(rxp);
        r.dma().rx().maxcnt().write(|w| w.set_maxcnt(rx.len() as u16));
        r.events_started().write_value(0);
        r.events_end().write_value(0);
        let errata8 = self.config.divisor > 2;
        if errata8 {
            self.errata8_reg().write_value(0x82);
        }
        r.events_dma().rx().end().write_value(0);
        r.events_dma().tx().end().write_value(0);
        r.events_stopped().write_value(0);
        r.tasks_start().write_value(1);
        let ok_started = Self::wait_event(r.events_started());
        if errata8 {
            self.errata8_reg().write_value(0x00);
        }
        // Completion = both DMA directions done (EVENTS_END never fires
        // with CSN disconnected); then STOP closes the engine's
        // transaction state.
        if !ok_started {
            self.fault = Some(self.snapshot(Stage::Started));
            return Err(Error::Timeout);
        }
        if !(Self::wait_event(r.events_dma().rx().end()) && Self::wait_event(r.events_dma().tx().end())) {
            self.fault = Some(self.snapshot(Stage::DmaEnd));
            return Err(Error::Timeout);
        }
        r.tasks_stop().write_value(1);
        // Best-effort: erratum [69] says STOPPED can fail to assert in
        // corner cases; a bounded wait keeps that from wedging us.
        let stop_start = cortex_m::peripheral::DWT::cycle_count();
        while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(stop_start) < 128_000 {
            if r.events_stopped().read() != 0 {
                break;
            }
        }
        r.events_stopped().write_value(0);
        Ok(())
    }

    pub fn blocking_write(&mut self, tx: &[u8]) -> Result<(), Error> {
        self.blocking_transfer(tx, &mut [])
    }

    pub fn blocking_read(&mut self, rx: &mut [u8]) -> Result<(), Error> {
        self.blocking_transfer(&[], rx)
    }

    fn errata8_reg(&self) -> Reg<u32, RW> {
        unsafe { Reg::from_ptr((self.r.as_ptr() as *mut u8).add(ERRATA8_OFFSET) as *mut u32) }
    }

    /// Wait up to 20 ms (DWT-timed) for an event register.
    fn wait_event(event: Reg<u32, RW>) -> bool {
        let start = cortex_m::peripheral::DWT::cycle_count();
        while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < 2_560_000 {
            if event.read() != 0 {
                return true;
            }
        }
        false
    }

    fn snapshot(&self, stage: Stage) -> FaultSnapshot {
        let r = self.r;
        let rx = r.events_dma().rx();
        let tx = r.events_dma().tx();
        FaultSnapshot {
            stage,
            events_started: r.events_started().read(),
            events_end: r.events_end().read(),
            enable: r.enable().read().0,
            prescaler: r.prescaler().read().0,
            config: r.config().read().0,
            dma_rx_end: rx.end().read(),
            dma_rx_ready: rx.ready().read(),
            dma_rx_buserror: rx.buserror().read(),
            dma_rx_match: [
                rx.match_(0).read(),
                rx.match_(1).read(),
                rx.match_(2).read(),
                rx.match_(3).read(),
            ],
            dma_tx_end: tx.end().read(),
            dma_tx_ready: tx.ready().read(),
            dma_tx_buserror: tx.buserror().read(),
            rx_buserror_address: r.dma().rx().buserroraddress().read(),
            tx_buserror_address: r.dma().tx().buserroraddress().read(),
        }
    }

    // --- wiring diagnostics -------------------------------------------

    /// Drive one of the output lines to a static level (bit-banged phase).
    pub fn drive_line(&mut self, line: Line, high: bool) {
        let pin = match line {
            Line::Sck => &self.sck,
            Line::Mosi => &self.mosi,
            Line::Cs => &self.cs,
            Line::Miso => return,
        };
        if high {
            gpio::set_high(pin);
        } else {
            gpio::set_low(pin);
        }
    }

    /// Pull MISO down (or back up) and report the level the SoC reads.
    pub fn miso_pull(&mut self, pull_up: bool) {
        gpio::conf(&self.miso).write(|w| {
            w.set_dir(gpiovals::Dir::Input);
            w.set_input(gpiovals::Input::Connect);
            w.set_pull(if pull_up { gpiovals::Pull::Pullup } else { gpiovals::Pull::Pulldown });
        });
    }

    pub fn miso_level(&self) -> bool {
        gpio::is_high(&self.miso)
    }

    /// Release all four pins to high-impedance inputs (no pulls), so an
    /// external master can drive the shared wires while this side stays
    /// powered and attached.
    pub fn release_pins(&mut self) {
        self.r.enable().write(|w| w.set_enable(vals::Enable::Disabled));
        for pin in [&self.sck, &self.mosi, &self.miso, &self.cs] {
            gpio::conf(pin).write(|w| {
                w.set_dir(gpiovals::Dir::Input);
                w.set_input(gpiovals::Input::Disconnect);
                w.set_pull(gpiovals::Pull::Disabled);
            });
        }
        self.bitbang = true;
    }
}

impl<'d> Drop for SpimSoftCs<'d> {
    fn drop(&mut self) {
        let r = self.r;
        r.enable().write(|w| w.set_enable(vals::Enable::Disabled));
        r.psel().sck().write_value(DISCONNECTED);
        r.psel().mosi().write_value(DISCONNECTED);
        r.psel().miso().write_value(DISCONNECTED);
        for pin in [&self.sck, &self.mosi, &self.miso, &self.cs] {
            gpio::deconfigure(pin);
        }
    }
}

pub(crate) trait SealedInstance {
    fn regs() -> pac::spim::Spim;
}

/// SPIM peripheral instance.
#[allow(private_bounds)]
pub trait Instance: SealedInstance + PeripheralType + 'static + Send {}

macro_rules! impl_spim {
    ($type:ident, $pac_type:ident) => {
        impl SealedInstance for embassy_nrf::peripherals::$type {
            fn regs() -> pac::spim::Spim {
                pac::$pac_type
            }
        }
        impl Instance for embassy_nrf::peripherals::$type {}
    };
}

impl_spim!(SERIAL00, SPIM00);
impl_spim!(SERIAL22, SPIM22);
