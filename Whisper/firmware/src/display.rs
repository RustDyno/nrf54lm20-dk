//! SSD1306 128x64 OLED over TWIM22: live transcript output for the
//! standalone transcriber. The display is OPTIONAL -- init() probes the
//! bus and every call becomes a no-op when nothing answers, so the build
//! runs unchanged without the module wired.
//!
//! Wiring (expansion board header P17, 3.3 V; the module's own pull-ups
//! plus the internal ones):
//!   SCL -> P3.3 (P17 pin 14)    SDA -> P3.2 (P17 pin 13)
//! These are the pins the SD card vacated when it moved to SPIM00, and
//! TWIM22 is the serial instance the SD vacated (the 16 MHz serial boxes
//! reach ports P1/P3 only; same register map as SPIM22 at the same base).
//!
//! The bus is the HAL's TWIM driver (`embassy_nrf::twim`) in blocking mode:
//! one write transaction per call, STOP shortcut on the LASTTX event, an
//! address or data NACK reported as an error (the HAL needs the port-3
//! GPIO fix carried in vendor/embassy-nrf for these pins). The driver is
//! built when the standalone app probes the panel, so the mailbox executor
//! never touches the bus, and it stays alive for the rest of the run even
//! when nothing answers: erratum [105] wedges the peripheral if it is
//! disabled while a target stretches the clock. Its blocking wait has no
//! timeout: a target that holds SCL forever holds the transcriber.
//!
//! Console model: 21 columns x 8 rows of 5x7 glyphs in 6x8 cells,
//! append-with-wrap, scroll-up when full, full redraw per print call
//! (~1 KB over the bus, ~27 ms at 400 kHz -- nothing at token cadence).

#[cfg(not(feature = "sd-spim22"))]
use embassy_nrf::bind_interrupts;
#[cfg(not(feature = "sd-spim22"))]
use embassy_nrf::peripherals::SERIAL22;
#[cfg(not(feature = "sd-spim22"))]
use embassy_nrf::twim::{self, Twim};

// The handler never runs in blocking use (no INTEN bit is ever set), but
// `Twim::new` unmasks the NVIC line, so main.rs routes SERIAL22 here.
#[cfg(not(feature = "sd-spim22"))]
bind_interrupts!(struct Irqs {
    SERIAL22 => twim::InterruptHandler<SERIAL22>;
});

pub const COLS: usize = 21;
pub const ROWS: usize = 8;

#[cfg(not(feature = "sd-spim22"))]
static mut TWIM: Option<Twim<'static>> = None;
/// Every write here comes from a RAM buffer, so the driver's flash-copy
/// staging buffer can be empty.
#[cfg(not(feature = "sd-spim22"))]
static mut NO_RAM_STAGING: [u8; 0] = [];

static mut PRESENT: bool = false;
static mut ADDR7: u8 = 0x3C;
static mut GRID: [u8; COLS * ROWS] = [b' '; COLS * ROWS];
static mut CUR_ROW: usize = 0;
static mut CUR_COL: usize = 0;

/// One I2C write transaction. Returns false when the target NACKs the
/// address or a byte (or when no driver was ever built).
fn twi_write(addr: u8, buf: &[u8]) -> bool {
    #[cfg(not(feature = "sd-spim22"))]
    {
        match unsafe { &mut *core::ptr::addr_of_mut!(TWIM) } {
            Some(t) => match t.blocking_write(addr, buf) {
                Ok(()) => true,
                Err(twim::Error::AddressNack) => false,
                Err(e) => {
                    // Anything but the "no panel" answer is worth a line.
                    rtt_target::rprintln!("display: twim write to {:#04x} failed: {:?}", addr, e);
                    false
                }
            },
            None => false,
        }
    }
    #[cfg(feature = "sd-spim22")]
    {
        let _ = (addr, buf);
        false
    }
}

fn cmd(bytes: &[u8]) -> bool {
    let mut buf = [0u8; 8];
    buf[1..1 + bytes.len()].copy_from_slice(bytes); // buf[0]=0x00 control
    twi_write(unsafe { ADDR7 }, &buf[..1 + bytes.len()])
}

/// Build the TWIM driver on the board's OLED singletons: 400 kHz, internal
/// pull-ups on both lines, standard drive with open-drain '1'.
#[cfg(not(feature = "sd-spim22"))]
fn open_bus() -> bool {
    unsafe {
        if (*core::ptr::addr_of!(TWIM)).is_some() {
            return true;
        }
    }
    let Some(oled) = crate::board::get().oled.take() else {
        return false;
    };
    let mut config = twim::Config::default();
    config.frequency = twim::Frequency::K400;
    config.sda_pullup = true;
    config.scl_pullup = true;
    let staging = unsafe { &mut *core::ptr::addr_of_mut!(NO_RAM_STAGING) };
    let twim = Twim::new(oled.twim, Irqs, oled.sda, oled.scl, config, staging);
    unsafe { TWIM = Some(twim) };
    true
}

/// Probe for the display and bring it up. Safe to call when absent.
pub fn init() -> bool {
    // The sd-spim22 diagnostic build owns this serial box and these pins.
    if cfg!(feature = "sd-spim22") {
        return false;
    }
    #[cfg(not(feature = "sd-spim22"))]
    if !open_bus() {
        return false;
    }
    unsafe {
        PRESENT = true; // provisionally, for the probe writes

        // 0x3C is the common SSD1306 address, 0x3D the alternate strap.
        ADDR7 = 0x3C;
        if !cmd(&[0xAE]) {
            ADDR7 = 0x3D;
            if !cmd(&[0xAE]) {
                // The driver stays built and enabled (erratum [105]).
                PRESENT = false;
                return false;
            }
        }
    }
    // Standard SSD1306 128x64 bring-up, horizontal addressing.
    let ok = cmd(&[0xD5, 0x80]) // clock divide
        && cmd(&[0xA8, 0x3F]) // multiplex 64
        && cmd(&[0xD3, 0x00]) // display offset
        && cmd(&[0x40]) // start line 0
        && cmd(&[0x8D, 0x14]) // charge pump on
        && cmd(&[0x20, 0x00]) // horizontal addressing
        && cmd(&[0xA1]) // segment remap
        && cmd(&[0xC8]) // COM scan descending
        && cmd(&[0xDA, 0x12]) // COM pins
        && cmd(&[0x81, 0x7F]) // contrast
        && cmd(&[0xD9, 0xF1]) // precharge
        && cmd(&[0xDB, 0x40]) // VCOM detect
        && cmd(&[0xA4]) // resume from RAM
        && cmd(&[0xA6]) // normal (not inverted)
        && cmd(&[0xAF]); // display on
    unsafe { PRESENT = ok };
    if ok {
        clear();
    }
    ok
}

pub fn clear() {
    unsafe {
        if !PRESENT {
            return;
        }
        (*core::ptr::addr_of_mut!(GRID)).fill(b' ');
        CUR_ROW = 0;
        CUR_COL = 0;
    }
    render();
}

/// Append text: wraps at the right edge, '\n' breaks, scrolls when full.
/// Non-ASCII input renders as '?' (UTF-8 continuation bytes are skipped).
pub fn print(s: &str) {
    unsafe {
        if !PRESENT {
            return;
        }
        let grid = &mut *core::ptr::addr_of_mut!(GRID);
        for b in s.bytes() {
            if b & 0xC0 == 0x80 {
                continue; // UTF-8 continuation
            }
            if b == b'\n' {
                CUR_COL = 0;
                CUR_ROW += 1;
            } else {
                if CUR_COL == COLS {
                    CUR_COL = 0;
                    CUR_ROW += 1;
                }
                if CUR_ROW == ROWS {
                    grid.copy_within(COLS.., 0);
                    grid[COLS * (ROWS - 1)..].fill(b' ');
                    CUR_ROW = ROWS - 1;
                }
                let ch = if (0x20..=0x7E).contains(&b) { b } else { b'?' };
                grid[CUR_ROW * COLS + CUR_COL] = ch;
                CUR_COL += 1;
            }
            if CUR_ROW == ROWS && CUR_COL == 0 {
                // newline past the last row: scroll now so the next
                // character lands on a fresh bottom line
                grid.copy_within(COLS.., 0);
                grid[COLS * (ROWS - 1)..].fill(b' ');
                CUR_ROW = ROWS - 1;
            }
        }
    }
    render();
}

/// Redraw the whole panel from the character grid, one page per burst.
fn render() {
    unsafe {
        if !PRESENT {
            return;
        }
        let grid = &*core::ptr::addr_of!(GRID);
        for row in 0..ROWS {
            if !(cmd(&[0x21, 0, 127]) && cmd(&[0x22, row as u8, row as u8])) {
                return;
            }
            let mut buf = [0u8; 1 + 128];
            buf[0] = 0x40; // data control byte
            for col in 0..COLS {
                let ch = grid[row * COLS + col] as usize;
                let glyph = &FONT[(ch - 0x20) * 5..(ch - 0x20) * 5 + 5];
                buf[1 + col * 6..1 + col * 6 + 5].copy_from_slice(glyph);
            }
            if !twi_write(ADDR7, &buf) {
                return;
            }
        }
    }
}

// Classic 5x7 ASCII font, one column per byte, LSB at the top (0x20-0x7E).
#[rustfmt::skip]
static FONT: [u8; 95 * 5] = [
    0x00,0x00,0x00,0x00,0x00, // space
    0x00,0x00,0x5F,0x00,0x00, // !
    0x00,0x07,0x00,0x07,0x00, // "
    0x14,0x7F,0x14,0x7F,0x14, // #
    0x24,0x2A,0x7F,0x2A,0x12, // $
    0x23,0x13,0x08,0x64,0x62, // %
    0x36,0x49,0x55,0x22,0x50, // &
    0x00,0x05,0x03,0x00,0x00, // '
    0x00,0x1C,0x22,0x41,0x00, // (
    0x00,0x41,0x22,0x1C,0x00, // )
    0x14,0x08,0x3E,0x08,0x14, // *
    0x08,0x08,0x3E,0x08,0x08, // +
    0x00,0x50,0x30,0x00,0x00, // ,
    0x08,0x08,0x08,0x08,0x08, // -
    0x00,0x60,0x60,0x00,0x00, // .
    0x20,0x10,0x08,0x04,0x02, // /
    0x3E,0x51,0x49,0x45,0x3E, // 0
    0x00,0x42,0x7F,0x40,0x00, // 1
    0x42,0x61,0x51,0x49,0x46, // 2
    0x21,0x41,0x45,0x4B,0x31, // 3
    0x18,0x14,0x12,0x7F,0x10, // 4
    0x27,0x45,0x45,0x45,0x39, // 5
    0x3C,0x4A,0x49,0x49,0x30, // 6
    0x01,0x71,0x09,0x05,0x03, // 7
    0x36,0x49,0x49,0x49,0x36, // 8
    0x06,0x49,0x49,0x29,0x1E, // 9
    0x00,0x36,0x36,0x00,0x00, // :
    0x00,0x56,0x36,0x00,0x00, // ;
    0x08,0x14,0x22,0x41,0x00, // <
    0x14,0x14,0x14,0x14,0x14, // =
    0x00,0x41,0x22,0x14,0x08, // >
    0x02,0x01,0x51,0x09,0x06, // ?
    0x32,0x49,0x79,0x41,0x3E, // @
    0x7E,0x11,0x11,0x11,0x7E, // A
    0x7F,0x49,0x49,0x49,0x36, // B
    0x3E,0x41,0x41,0x41,0x22, // C
    0x7F,0x41,0x41,0x22,0x1C, // D
    0x7F,0x49,0x49,0x49,0x41, // E
    0x7F,0x09,0x09,0x09,0x01, // F
    0x3E,0x41,0x49,0x49,0x7A, // G
    0x7F,0x08,0x08,0x08,0x7F, // H
    0x00,0x41,0x7F,0x41,0x00, // I
    0x20,0x40,0x41,0x3F,0x01, // J
    0x7F,0x08,0x14,0x22,0x41, // K
    0x7F,0x40,0x40,0x40,0x40, // L
    0x7F,0x02,0x0C,0x02,0x7F, // M
    0x7F,0x04,0x08,0x10,0x7F, // N
    0x3E,0x41,0x41,0x41,0x3E, // O
    0x7F,0x09,0x09,0x09,0x06, // P
    0x3E,0x41,0x51,0x21,0x5E, // Q
    0x7F,0x09,0x19,0x29,0x46, // R
    0x46,0x49,0x49,0x49,0x31, // S
    0x01,0x01,0x7F,0x01,0x01, // T
    0x3F,0x40,0x40,0x40,0x3F, // U
    0x1F,0x20,0x40,0x20,0x1F, // V
    0x3F,0x40,0x38,0x40,0x3F, // W
    0x63,0x14,0x08,0x14,0x63, // X
    0x07,0x08,0x70,0x08,0x07, // Y
    0x61,0x51,0x49,0x45,0x43, // Z
    0x00,0x7F,0x41,0x41,0x00, // [
    0x02,0x04,0x08,0x10,0x20, // backslash
    0x00,0x41,0x41,0x7F,0x00, // ]
    0x04,0x02,0x01,0x02,0x04, // ^
    0x40,0x40,0x40,0x40,0x40, // _
    0x00,0x01,0x02,0x04,0x00, // `
    0x20,0x54,0x54,0x54,0x78, // a
    0x7F,0x48,0x44,0x44,0x38, // b
    0x38,0x44,0x44,0x44,0x20, // c
    0x38,0x44,0x44,0x48,0x7F, // d
    0x38,0x54,0x54,0x54,0x18, // e
    0x08,0x7E,0x09,0x01,0x02, // f
    0x0C,0x52,0x52,0x52,0x3E, // g
    0x7F,0x08,0x04,0x04,0x78, // h
    0x00,0x44,0x7D,0x40,0x00, // i
    0x20,0x40,0x44,0x3D,0x00, // j
    0x7F,0x10,0x28,0x44,0x00, // k
    0x00,0x41,0x7F,0x40,0x00, // l
    0x7C,0x04,0x18,0x04,0x78, // m
    0x7C,0x08,0x04,0x04,0x78, // n
    0x38,0x44,0x44,0x44,0x38, // o
    0x7C,0x14,0x14,0x14,0x08, // p
    0x08,0x14,0x14,0x18,0x7C, // q
    0x7C,0x08,0x04,0x04,0x08, // r
    0x48,0x54,0x54,0x54,0x20, // s
    0x04,0x3F,0x44,0x40,0x20, // t
    0x3C,0x40,0x40,0x20,0x7C, // u
    0x1C,0x20,0x40,0x20,0x1C, // v
    0x3C,0x40,0x30,0x40,0x3C, // w
    0x44,0x28,0x10,0x28,0x44, // x
    0x0C,0x50,0x50,0x50,0x3C, // y
    0x44,0x64,0x54,0x4C,0x44, // z
    0x00,0x08,0x36,0x41,0x00, // {
    0x00,0x00,0x7F,0x00,0x00, // |
    0x00,0x41,0x36,0x08,0x00, // }
    0x08,0x04,0x08,0x10,0x08, // ~
];
