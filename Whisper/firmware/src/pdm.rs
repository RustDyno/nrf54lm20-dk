//! Minimal bare-metal driver for the nRF54LM20 PDM peripheral (PDM v2).
//!
//! Ported from the PDM-MIC capture project: the PDM block clocks the MEMS mic,
//! decimates the 1-bit stream to 16-bit signed PCM at 16 kHz and writes it to
//! RAM via EasyDMA. Polled ping-pong double buffering; here each buffer is one
//! MFCC hop (320 samples = 20 ms) and is returned as samples, not bytes.
//!
//! Registers are reached through the PAC (`embassy_nrf::pac`), which carries
//! the byte-counted MAXCNT, the PRESCALER/RATIO clock model and the PSEL
//! encoding. The HAL's own PDM driver is async only, and this capture loop
//! has to stay polled: the hop-level overrun accounting below relies on
//! seeing EVENTS_STARTED itself.

use embassy_nrf::pac;
use pac::gpio::vals::{Dir, Input, Pull};
use pac::pdm::vals::{Edge, Gain, Operation, Ratio, Src};
use pac::shared::vals::Connect;

// PDM20 (secure alias, the core boots secure). PDM21 is the other instance.
const PDM: pac::pdm::Pdm = pac::PDM20;

// The mic pins live on port 1.
const PORT1: pac::gpio::Gpio = pac::P1;

// Digital gain, 0.5 dB per step around 0x28 = 0 dB (0x00 = -20 dB,
// 0x50 = +20 dB). Applied inside the peripheral ahead of the 16-bit
// output, so unlike a later software scale it keeps detail that would
// otherwise be truncated away.
//
// +12 dB (24 steps). Measured at 0 dB the mic delivered ordinary speech
// at an active rms of ~156 of 32768 -- about 7 bits of the 16 -- with a
// noise floor near 3. Four times that still leaves speech peaks (~10x
// the rms) an order of magnitude below full scale, so it buys headroom
// back without risking clipped speech; a desk knock may clip, which is
// harmless. The rest of the shortfall against the calibration clip is
// taken out per-utterance in the log-mel domain, where it cannot clip at
// all (app.rs mel_lift).
const GAIN: Gain = Gain::from_bits(0x28 + 24);

// Clocking: PDM_CLK = 32 MHz / 25 = 1.28 MHz, / RATIO 80 = 16000 Hz exactly.
const PRESCALER_DIV: u8 = 25;
#[allow(dead_code)]
pub const SAMPLE_RATE_HZ: u32 = 16_000;

/// Poll the live level of a port-1 pin `samples` times. With PDM running, a
/// working mic makes DIN toggle (both counts > 0); a dead line stays stuck.
pub fn probe_pin_activity(pin: Pin, samples: u32) -> (u32, u32) {
    let mut high = 0;
    let mut low = 0;
    for _ in 0..samples {
        if PORT1.in_().read().pin(pin.pin as usize) {
            high += 1;
        } else {
            low += 1;
        }
    }
    (high, low)
}

#[derive(Clone, Copy)]
pub struct Pin {
    pub port: u8,
    pub pin: u8,
}

impl Pin {
    /// PSEL value: PIN in bits [4:0], PORT in [6:5], CONNECT (bit 31) clear.
    fn psel(self) -> pac::shared::regs::Psel {
        let mut v = pac::shared::regs::Psel(0);
        v.set_pin(self.pin);
        v.set_port(self.port);
        v.set_connect(Connect::Connected);
        v
    }
}

fn configure_gpio(clk: Pin, din: Pin) {
    // CLK: output, input buffer disconnected, start low.
    PORT1.outclr().write(|w| w.set_pin(clk.pin as usize, true));
    PORT1.pin_cnf(clk.pin as usize).write(|w| {
        w.set_dir(Dir::Output);
        w.set_input(Input::Disconnect);
    });
    // DIN: input, input buffer connected, no pull.
    PORT1.pin_cnf(din.pin as usize).write(|w| {
        w.set_dir(Dir::Input);
        w.set_input(Input::Connect);
        w.set_pull(Pull::Disabled);
    });
}

pub struct Pdm;

impl Pdm {
    /// Configure the peripheral and its pins. Unsafe because the caller
    /// hands the EasyDMA engine RAM through [`Pdm::start`]: nothing else may
    /// touch those buffers while a capture runs.
    pub unsafe fn init(clk: Pin, din: Pin) -> Self {
        configure_gpio(clk, din);

        PDM.psel().clk().write_value(clk.psel());
        PDM.psel().din().write_value(din.psel());

        PDM.clkselect().write(|w| w.set_src(Src::Pclk32m));
        PDM.prescaler().write(|w| w.set_divisor(PRESCALER_DIV));
        PDM.ratio().write(|w| w.set_ratio(Ratio::Ratio80));
        PDM.mode().write(|w| {
            w.set_operation(Operation::Mono);
            w.set_edge(Edge::LeftRising); // board-validated for this mic
        });
        PDM.gainl().write(|w| w.set_gainl(GAIN));
        PDM.gainr().write(|w| w.set_gainr(GAIN));

        PDM.enable().write(|w| w.set_enable(true));
        Pdm
    }

    /// MAXCNT on PDM v2 is a *byte* count, so it is `len * 2`.
    #[inline(always)]
    fn set_buffer(&self, buf: *const i16, len: usize) {
        PDM.sample().ptr().write_value(buf as u32);
        PDM.sample().maxcnt().write(|w| w.set_buffsize((len * 2) as u16));
    }

    #[inline(always)]
    fn clear_started(&self) {
        PDM.events_started().write_value(0);
    }

    #[inline(always)]
    fn started_pending(&self) -> bool {
        PDM.events_started().read() != 0
    }

    #[inline(always)]
    fn wait_started(&self) {
        while PDM.events_started().read() == 0 {
            cortex_m::asm::nop();
        }
    }

    /// Start sampling into `buf0`, returning a [`Stream`] that ping-pongs
    /// between `buf0` and `buf1`. Equal length, both in RAM.
    pub unsafe fn start<'b>(self, buf0: &'b mut [i16], buf1: &'b mut [i16]) -> Stream<'b> {
        let len = buf0.len();
        PDM.events_started().write_value(0);
        PDM.events_end().write_value(0);
        PDM.events_stopped().write_value(0);

        self.set_buffer(buf0.as_ptr(), len);
        PDM.tasks_start().write_value(1);
        // First STARTED: buf0 is now being filled.
        self.wait_started();
        self.clear_started();

        Stream {
            pdm: self,
            bufs: [buf0, buf1],
            filling: 0,
            len,
            overruns: 0,
        }
    }
}

/// A running double-buffered capture. Each [`Stream::next_buffer`] call blocks
/// until a 20 ms hop is full and returns its samples.
pub struct Stream<'b> {
    pdm: Pdm,
    bufs: [&'b mut [i16]; 2],
    filling: usize,
    len: usize,
    /// Hops where the caller was too slow to queue the next buffer before the
    /// in-flight one filled: the hardware reused the old pointer and samples
    /// were lost. Monitored by the main loop.
    pub overruns: u32,
}

impl<'b> Stream<'b> {
    pub fn next_buffer(&mut self) -> &[i16] {
        let next = self.filling ^ 1;
        // If STARTED is already pending we arrived after the in-flight
        // buffer completed: its successor pointer was stale -> overrun.
        if self.pdm.started_pending() {
            self.overruns += 1;
        }
        // Queue the other buffer before the current one finishes.
        self.pdm.set_buffer(self.bufs[next].as_ptr(), self.len);
        // Wait for the hardware to latch it and swap; `filling` is now full.
        self.pdm.wait_started();
        self.pdm.clear_started();
        let done = self.filling;
        self.filling = next;
        &self.bufs[done][..]
    }

    #[allow(dead_code)]
    pub fn stop(self) {
        PDM.tasks_stop().write_value(1);
        while PDM.events_stopped().read() == 0 {
            cortex_m::asm::nop();
        }
        PDM.enable().write(|w| w.set_enable(false));
    }
}
