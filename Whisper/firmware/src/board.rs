//! The peripheral singletons this firmware uses, taken once from
//! `embassy_nrf::init()` and parked here for the drivers.
//!
//! Drivers are built where they are needed, the embassy way (a `Peri`
//! per peripheral and pin): the microphone per recording (reborrowed, so
//! the singletons stay here), the display when the standalone app probes
//! the panel, the storage backends when the storage layer picks one. The
//! one-shot users take their singletons out of the `Option`s, so a second
//! attempt to build the same driver is a visible `None`, not a silent
//! second owner.
//!
//! Wiring, all on the DK expansion header P17 (see the module docs of
//! each driver for the pin table):
//!   PDM mic     CLK P1.23, DIN P1.24        (PDM20)
//!   OLED        SDA P3.2,  SCL P3.3         (TWIM22)
//!   SD card     SCK P2.01, MOSI P2.02, MISO P2.04, CS P2.05 (SPIM00),
//!               or SCK P3.3, MOSI P3.0, MISO P3.1, CS P3.2 (SPIM22,
//!               `sd-spim22`: the OLED's serial box and pins)
//!   USB         J3 (USBHS)

#[cfg(not(feature = "sd-spim22"))]
use embassy_nrf::peripherals::{P2_01, P2_02, P2_04, P2_05, SERIAL00};
#[cfg(feature = "sd-spim22")]
use embassy_nrf::peripherals::{P3_00, P3_01};
use embassy_nrf::peripherals::{P1_23, P1_24, P3_02, P3_03, PDM20, SERIAL22, USBHS};
use embassy_nrf::{Peri, Peripherals};
use fixed::types::I7F1;

use crate::hal::pdm;

/// The microphone configuration this board was calibrated with: 32 MHz /
/// 25 = 1.28 MHz PDM clock, decimated by 80 to 16 kHz, mono, left channel
/// on the rising edge (board-validated for this mic).
///
/// +12 dB of digital gain, applied inside the peripheral ahead of the
/// 16-bit output, so unlike a later software scale it keeps detail that
/// would otherwise be truncated away. Measured at 0 dB the mic delivered
/// ordinary speech at an active rms of ~156 of 32768 -- about 7 bits of
/// the 16 -- with a noise floor near 3. Four times that still leaves
/// speech peaks (~10x the rms) an order of magnitude below full scale, so
/// it buys headroom back without risking clipped speech; a desk knock may
/// clip, which is harmless. The rest of the shortfall against the
/// calibration clip is taken out per-utterance in the log-mel domain,
/// where it cannot clip at all (app.rs mel_lift).
pub fn mic_config() -> pdm::Config {
    let mut config = pdm::Config::default();
    config.edge = pdm::Edge::LeftRising;
    config.gain_left = I7F1::from_num(12);
    config.gain_right = I7F1::from_num(12);
    config
}

/// The microphone: reborrowed per recording.
pub struct Mic {
    pub pdm: Peri<'static, PDM20>,
    pub clk: Peri<'static, P1_23>,
    pub din: Peri<'static, P1_24>,
}

/// The OLED bus: taken once by the display.
#[cfg(not(feature = "sd-spim22"))]
pub struct Oled {
    pub twim: Peri<'static, SERIAL22>,
    pub sda: Peri<'static, P3_02>,
    pub scl: Peri<'static, P3_03>,
}

/// The SD card bus on the high-speed SPIM and its dedicated P2 pins.
#[cfg(not(feature = "sd-spim22"))]
pub struct SdBus {
    pub spim: Peri<'static, SERIAL00>,
    pub sck: Peri<'static, P2_01>,
    pub mosi: Peri<'static, P2_02>,
    pub miso: Peri<'static, P2_04>,
    pub cs: Peri<'static, P2_05>,
}

/// The SD card bus of the diagnostic wiring: SPIM22 on plain P3 pins.
#[cfg(feature = "sd-spim22")]
pub struct SdBus {
    pub spim: Peri<'static, SERIAL22>,
    pub sck: Peri<'static, P3_03>,
    pub mosi: Peri<'static, P3_00>,
    pub miso: Peri<'static, P3_01>,
    pub cs: Peri<'static, P3_02>,
}

pub struct Board {
    pub mic: Mic,
    #[cfg(not(feature = "sd-spim22"))]
    pub oled: Option<Oled>,
    pub sd: Option<SdBus>,
    pub usbhs: Option<Peri<'static, USBHS>>,
}

static mut BOARD: Option<Board> = None;

/// Park the singletons. Called once from main().
pub fn init(p: Peripherals) {
    let board = Board {
        mic: Mic {
            pdm: p.PDM20,
            clk: p.P1_23,
            din: p.P1_24,
        },
        #[cfg(not(feature = "sd-spim22"))]
        oled: Some(Oled {
            twim: p.SERIAL22,
            sda: p.P3_02,
            scl: p.P3_03,
        }),
        #[cfg(not(feature = "sd-spim22"))]
        sd: Some(SdBus {
            spim: p.SERIAL00,
            sck: p.P2_01,
            mosi: p.P2_02,
            miso: p.P2_04,
            cs: p.P2_05,
        }),
        #[cfg(feature = "sd-spim22")]
        sd: Some(SdBus {
            spim: p.SERIAL22,
            sck: p.P3_03,
            mosi: p.P3_00,
            miso: p.P3_01,
            cs: p.P3_02,
        }),
        usbhs: Some(p.USBHS),
    };
    unsafe { BOARD = Some(board) };
}

/// The board. The firmware is single threaded and every driver is built
/// and used from thread mode, so one caller at a time holds this.
pub fn get() -> &'static mut Board {
    unsafe { (*core::ptr::addr_of_mut!(BOARD)).as_mut().expect("board::init first") }
}
