//! Pulse Density Modulation (PDM) microphone driver for the nRF54L series.
//!
//! The nRF54L PDM block differs from the nRF52's: the clock is
//! CLKSELECT (32 MHz peripheral clock or the audio clock) divided by
//! PRESCALER, the decimation RATIO is a register of its own, and
//! SAMPLE.MAXCNT counts bytes rather than samples. This driver is
//! interrupt free: [`Pdm::blocking_stream`] runs the EasyDMA double buffer
//! the way a polled capture loop needs it, with each finished buffer handed
//! back as samples and late buffer swaps counted as overruns.
//!
//! Register usage, offsets and the PSEL encoding come from the PAC.

#![allow(dead_code)]

use core::marker::PhantomData;

use embassy_nrf::gpio::{AnyPin, Pin as GpioPin};
use embassy_nrf::pac;
use embassy_nrf::pac::gpio::vals as gpiovals;
use embassy_nrf::pac::pdm::vals;
pub use embassy_nrf::pac::pdm::vals::Ratio;
use embassy_nrf::{Peri, PeripheralType};
use fixed::types::I7F1;

use super::gpio::{self, DISCONNECTED};

/// The largest buffer one DMA run can fill: MAXCNT is a 15-bit byte count.
pub const MAX_SAMPLES: usize = 0x7FFF / 2;

/// PDM microphone interface.
pub struct Pdm<'d> {
    r: pac::pdm::Pdm,
    clk: Peri<'d, AnyPin>,
    din: Peri<'d, AnyPin>,
    _phantom: PhantomData<&'d ()>,
}

/// PDM error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// A buffer is longer than one DMA run can fill.
    BufferTooLong,
    /// A buffer is empty.
    BufferZeroLength,
    /// The two buffers of a stream have different lengths.
    BufferLengthMismatch,
}

impl<'d> Pdm<'d> {
    /// Create a PDM driver without an interrupt binding: the only API is
    /// the polled [`Pdm::blocking_stream`].
    pub fn new_blocking<T: Instance>(
        pdm: Peri<'d, T>,
        clk: Peri<'d, impl GpioPin>,
        din: Peri<'d, impl GpioPin>,
        config: Config,
    ) -> Self {
        Self::new_inner(pdm, clk.into(), din.into(), config)
    }

    fn new_inner<T: Instance>(_pdm: Peri<'d, T>, clk: Peri<'d, AnyPin>, din: Peri<'d, AnyPin>, config: Config) -> Self {
        let r = T::regs();

        // setup gpio pins: DIN input, CLK output starting low
        gpio::conf(&din).write(|w| {
            w.set_dir(gpiovals::Dir::Input);
            w.set_input(gpiovals::Input::Connect);
        });
        r.psel().din().write_value(din.psel_bits());
        gpio::set_low(&clk);
        gpio::conf(&clk).write(|w| {
            w.set_dir(gpiovals::Dir::Output);
            w.set_input(gpiovals::Input::Disconnect);
        });
        r.psel().clk().write_value(clk.psel_bits());

        // configure
        r.clkselect().write(|w| w.set_src(config.clock_source.into()));
        r.prescaler().write(|w| w.set_divisor(config.prescaler));
        r.ratio().write(|w| w.set_ratio(config.ratio));
        r.mode().write(|w| {
            w.set_operation(config.operation_mode.into());
            w.set_edge(config.edge.into());
        });

        Self::_set_gain(r, config.gain_left, config.gain_right);

        // Disable all events interrupts
        r.intenclr().write(|w| w.0 = 0xFFFF_FFFF);

        r.enable().write(|w| w.set_enable(true));

        Self {
            r,
            clk,
            din,
            _phantom: PhantomData,
        }
    }

    fn _set_gain(r: pac::pdm::Pdm, gain_left: I7F1, gain_right: I7F1) {
        // 0.5 dB per step around 0x28 = 0 dB; the hardware range is
        // -20 dB (0x00) to +20 dB (0x50).
        let left = gain_left.saturating_add(I7F1::from_bits(0x28)).to_bits().clamp(0, 0x50) as u8;
        let right = gain_right.saturating_add(I7F1::from_bits(0x28)).to_bits().clamp(0, 0x50) as u8;
        r.gainl().write(|w| w.set_gainl(vals::Gain::from_bits(left)));
        r.gainr().write(|w| w.set_gainr(vals::Gain::from_bits(right)));
    }

    /// Adjust the gain.
    pub fn set_gain(&mut self, gain_left: I7F1, gain_right: I7F1) {
        Self::_set_gain(self.r, gain_left, gain_right);
    }

    /// Start sampling into `buf0`, ping-ponging with `buf1`. Each
    /// [`Stream::next_buffer`] call blocks until the buffer in flight is
    /// full and returns it; dropping the stream stops sampling.
    pub fn blocking_stream<'b>(
        &'b mut self,
        buf0: &'b mut [i16],
        buf1: &'b mut [i16],
    ) -> Result<Stream<'b>, Error> {
        let len = buf0.len();
        if len == 0 {
            return Err(Error::BufferZeroLength);
        }
        if len > MAX_SAMPLES {
            return Err(Error::BufferTooLong);
        }
        if buf1.len() != len {
            return Err(Error::BufferLengthMismatch);
        }
        let r = self.r;
        r.events_started().write_value(0);
        r.events_end().write_value(0);
        r.events_stopped().write_value(0);

        Self::set_buffer(r, buf0.as_ptr(), len);
        r.tasks_start().write_value(1);
        // First STARTED: buf0 is now being filled.
        while r.events_started().read() == 0 {}
        r.events_started().write_value(0);

        Ok(Stream {
            r,
            bufs: [buf0, buf1],
            filling: 0,
            len,
            overruns: 0,
            _phantom: PhantomData,
        })
    }

    /// MAXCNT counts bytes on this block.
    fn set_buffer(r: pac::pdm::Pdm, buf: *const i16, len: usize) {
        r.sample().ptr().write_value(buf as u32);
        r.sample().maxcnt().write(|w| w.set_buffsize((len * 2) as u16));
    }
}

/// A running double-buffered capture.
pub struct Stream<'b> {
    r: pac::pdm::Pdm,
    bufs: [&'b mut [i16]; 2],
    filling: usize,
    len: usize,
    overruns: u32,
    _phantom: PhantomData<&'b mut ()>,
}

impl<'b> Stream<'b> {
    /// Queue the idle buffer, wait for the one in flight to fill, and
    /// return it. If the caller arrives after the in-flight buffer already
    /// completed, the hardware reused the stale pointer and samples were
    /// lost: that is counted in [`Stream::overruns`].
    pub fn next_buffer(&mut self) -> &[i16] {
        let next = self.filling ^ 1;
        if self.r.events_started().read() != 0 {
            self.overruns += 1;
        }
        Pdm::set_buffer(self.r, self.bufs[next].as_ptr(), self.len);
        while self.r.events_started().read() == 0 {}
        self.r.events_started().write_value(0);
        let done = self.filling;
        self.filling = next;
        &self.bufs[done][..]
    }

    /// Buffers lost to a late [`Stream::next_buffer`] since the last clear.
    pub fn overruns(&self) -> u32 {
        self.overruns
    }

    pub fn clear_overruns(&mut self) {
        self.overruns = 0;
    }
}

impl<'b> Drop for Stream<'b> {
    fn drop(&mut self) {
        self.r.tasks_stop().write_value(1);
        while self.r.events_stopped().read() == 0 {}
        self.r.events_stopped().write_value(0);
    }
}

/// PDM microphone driver Config.
pub struct Config {
    /// Use stereo or mono operation.
    pub operation_mode: OperationMode,
    /// On which edge the left channel should be sampled.
    pub edge: Edge,
    /// Source of the PDM clock.
    pub clock_source: ClockSource,
    /// PDM clock = source / prescaler (divisor of the 32 MHz source; 25
    /// gives 1.28 MHz).
    pub prescaler: u8,
    /// Decimation ratio (1.28 MHz / 80 = 16 kHz).
    pub ratio: Ratio,
    /// Gain left in dB.
    pub gain_left: I7F1,
    /// Gain right in dB.
    pub gain_right: I7F1,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            operation_mode: OperationMode::Mono,
            edge: Edge::LeftFalling,
            clock_source: ClockSource::Pclk32M,
            prescaler: 25,
            ratio: Ratio::Ratio80,
            gain_left: I7F1::ZERO,
            gain_right: I7F1::ZERO,
        }
    }
}

/// PDM operation mode.
#[derive(PartialEq)]
pub enum OperationMode {
    /// Mono (1 channel).
    Mono,
    /// Stereo (2 channels).
    Stereo,
}

impl From<OperationMode> for vals::Operation {
    fn from(mode: OperationMode) -> Self {
        match mode {
            OperationMode::Mono => vals::Operation::Mono,
            OperationMode::Stereo => vals::Operation::Stereo,
        }
    }
}

/// PDM edge polarity.
#[derive(PartialEq)]
pub enum Edge {
    /// Left edge is rising.
    LeftRising,
    /// Left edge is falling.
    LeftFalling,
}

impl From<Edge> for vals::Edge {
    fn from(edge: Edge) -> Self {
        match edge {
            Edge::LeftRising => vals::Edge::LeftRising,
            Edge::LeftFalling => vals::Edge::LeftFalling,
        }
    }
}

/// PDM clock source.
#[derive(PartialEq)]
pub enum ClockSource {
    /// The 32 MHz peripheral clock.
    Pclk32M,
    /// The audio clock.
    Aclk,
}

impl From<ClockSource> for vals::Src {
    fn from(src: ClockSource) -> Self {
        match src {
            ClockSource::Pclk32M => vals::Src::Pclk32m,
            ClockSource::Aclk => vals::Src::Aclk,
        }
    }
}

impl<'d> Drop for Pdm<'d> {
    fn drop(&mut self) {
        self.r.tasks_stop().write_value(1);
        self.r.enable().write(|w| w.set_enable(false));
        self.r.psel().din().write_value(DISCONNECTED);
        self.r.psel().clk().write_value(DISCONNECTED);
        gpio::deconfigure(&self.clk);
        gpio::deconfigure(&self.din);
    }
}

pub(crate) trait SealedInstance {
    fn regs() -> pac::pdm::Pdm;
}

/// PDM peripheral instance.
#[allow(private_bounds)]
pub trait Instance: SealedInstance + PeripheralType + 'static + Send {}

macro_rules! impl_pdm {
    ($type:ident, $pac_type:ident) => {
        impl SealedInstance for embassy_nrf::peripherals::$type {
            fn regs() -> pac::pdm::Pdm {
                pac::$pac_type
            }
        }
        impl Instance for embassy_nrf::peripherals::$type {}
    };
}

impl_pdm!(PDM20, PDM20);
impl_pdm!(PDM21, PDM21);
