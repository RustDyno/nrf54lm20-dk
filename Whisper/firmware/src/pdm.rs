//! Minimal bare-metal driver for the nRF54LM20 PDM peripheral (PDM v2).
//!
//! Ported from the PDM-MIC capture project: the PDM block clocks the MEMS mic,
//! decimates the 1-bit stream to 16-bit signed PCM at 16 kHz and writes it to
//! RAM via EasyDMA. Polled ping-pong double buffering; here each buffer is one
//! MFCC hop (320 samples = 20 ms) and is returned as samples, not bytes.
//!
//! Register offsets, the byte-counted MAXCNT, the PRESCALER/RATIO clock model
//! and the PSEL encoding all come from the nRF54LM20 SVD / MDK.

use core::ptr::{read_volatile, write_volatile};

// PDM20, secure alias (the core boots secure). PDM21 lives at 0x500D1000.
const PDM_BASE: usize = 0x500D_0000;

const TASKS_START: usize = 0x000;
const TASKS_STOP: usize = 0x004;
const EVENTS_STARTED: usize = 0x100;
const EVENTS_STOPPED: usize = 0x104;
const EVENTS_END: usize = 0x108;
const ENABLE: usize = 0x500;
const MODE: usize = 0x508;
const GAINL: usize = 0x518;
const GAINR: usize = 0x51C;
const RATIO: usize = 0x520;
const PSEL_CLK: usize = 0x540;
const PSEL_DIN: usize = 0x544;
const CLKSELECT: usize = 0x54C;
const SAMPLE_PTR: usize = 0x560;
const SAMPLE_MAXCNT: usize = 0x564;
const PRESCALER: usize = 0x580;

const ENABLE_ENABLED: u32 = 1;
const MODE_OPERATION_MONO: u32 = 1 << 0;
const MODE_EDGE_LEFTRISING: u32 = 0 << 1; // board-validated for this mic

// Digital gain, 0.5 dB per step around 0x28 = 0 dB (0x00 = -20 dB, 0x50 = +20 dB).
const GAIN: u32 = 0x28;

// Clocking: PDM_CLK = 32 MHz / 25 = 1.28 MHz, / RATIO 80 = 16000 Hz exactly.
const RATIO_80: u32 = 0x4;
const CLKSELECT_PCLK32M: u32 = 0x0;
const PRESCALER_DIV: u32 = 25;
#[allow(dead_code)]
pub const SAMPLE_RATE_HZ: u32 = 16_000;

// --- GPIO (port 1, secure alias). Used to pre-configure the CLK/DIN pins. ---
const P1_BASE: usize = 0x500D_8200;
const GPIO_IN: usize = 0x00C;
const GPIO_OUTCLR: usize = 0x008;
const GPIO_PIN_CNF: usize = 0x080; // PIN_CNF[n] = 0x080 + 4*n
const PIN_CNF_DIR_OUTPUT: u32 = 1 << 0;
const PIN_CNF_INPUT_DISCONNECT: u32 = 1 << 1;

/// Poll the live level of a port-1 pin `samples` times. With PDM running, a
/// working mic makes DIN toggle (both counts > 0); a dead line stays stuck.
pub fn probe_pin_activity(pin: Pin, samples: u32) -> (u32, u32) {
    let mut high = 0;
    let mut low = 0;
    for _ in 0..samples {
        let level = unsafe { read_volatile((P1_BASE + GPIO_IN) as *const u32) };
        if (level >> pin.pin) & 1 != 0 {
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
    /// Nordic PSEL encoding: PIN in bits [4:0], PORT in [6:5], CONNECT=0 in [31].
    const fn psel(self) -> u32 {
        ((self.port as u32) << 5) | (self.pin as u32)
    }
}

#[inline(always)]
unsafe fn wr(base: usize, off: usize, val: u32) {
    write_volatile((base + off) as *mut u32, val);
}

#[inline(always)]
unsafe fn rd(base: usize, off: usize) -> u32 {
    read_volatile((base + off) as *const u32)
}

unsafe fn configure_gpio(clk: Pin, din: Pin) {
    // CLK: output, input buffer disconnected, start low.
    wr(P1_BASE, GPIO_OUTCLR, 1 << clk.pin);
    wr(
        P1_BASE,
        GPIO_PIN_CNF + 4 * clk.pin as usize,
        PIN_CNF_DIR_OUTPUT | PIN_CNF_INPUT_DISCONNECT,
    );
    // DIN: input, input buffer connected, no pull (PIN_CNF = 0).
    wr(P1_BASE, GPIO_PIN_CNF + 4 * din.pin as usize, 0);
}

pub struct Pdm;

impl Pdm {
    pub unsafe fn init(clk: Pin, din: Pin) -> Self {
        configure_gpio(clk, din);

        wr(PDM_BASE, PSEL_CLK, clk.psel());
        wr(PDM_BASE, PSEL_DIN, din.psel());

        wr(PDM_BASE, CLKSELECT, CLKSELECT_PCLK32M);
        wr(PDM_BASE, PRESCALER, PRESCALER_DIV);
        wr(PDM_BASE, RATIO, RATIO_80);
        wr(PDM_BASE, MODE, MODE_OPERATION_MONO | MODE_EDGE_LEFTRISING);
        wr(PDM_BASE, GAINL, GAIN);
        wr(PDM_BASE, GAINR, GAIN);

        wr(PDM_BASE, ENABLE, ENABLE_ENABLED);
        Pdm
    }

    /// MAXCNT on PDM v2 is a *byte* count, so it is `len * 2`.
    #[inline(always)]
    unsafe fn set_buffer(&self, buf: *const i16, len: usize) {
        wr(PDM_BASE, SAMPLE_PTR, buf as u32);
        wr(PDM_BASE, SAMPLE_MAXCNT, (len * 2) as u32);
    }

    #[inline(always)]
    unsafe fn clear_started(&self) {
        wr(PDM_BASE, EVENTS_STARTED, 0);
    }

    #[inline(always)]
    unsafe fn started_pending(&self) -> bool {
        rd(PDM_BASE, EVENTS_STARTED) != 0
    }

    #[inline(always)]
    unsafe fn wait_started(&self) {
        while rd(PDM_BASE, EVENTS_STARTED) == 0 {
            cortex_m::asm::nop();
        }
    }

    /// Start sampling into `buf0`, returning a [`Stream`] that ping-pongs
    /// between `buf0` and `buf1`. Equal length, both in RAM.
    pub unsafe fn start<'b>(self, buf0: &'b mut [i16], buf1: &'b mut [i16]) -> Stream<'b> {
        let len = buf0.len();
        wr(PDM_BASE, EVENTS_STARTED, 0);
        wr(PDM_BASE, EVENTS_END, 0);
        wr(PDM_BASE, EVENTS_STOPPED, 0);

        self.set_buffer(buf0.as_ptr(), len);
        wr(PDM_BASE, TASKS_START, 1);
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
        unsafe {
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
        }
        let done = self.filling;
        self.filling = next;
        &self.bufs[done][..]
    }

    #[allow(dead_code)]
    pub fn stop(self) {
        unsafe {
            wr(PDM_BASE, TASKS_STOP, 1);
            while rd(PDM_BASE, EVENTS_STOPPED) == 0 {
                cortex_m::asm::nop();
            }
            wr(PDM_BASE, ENABLE, 0);
        }
    }
}
