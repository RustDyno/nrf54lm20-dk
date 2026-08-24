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
//! Register map from the nRF54LM20A SVD: the nRF54 TWIM has no STARTTX
//! task -- a write is TASKS_DMA.TX.START, then software stops it on the
//! LASTTX event (the datasheet's recommended non-shortcut pattern; the
//! event fires when the last byte STARTS, and a poll loop reacts well
//! within one 22.5 us byte time at 400 kHz).
//!
//! Console model: 21 columns x 8 rows of 5x7 glyphs in 6x8 cells,
//! append-with-wrap, scroll-up when full, full redraw per print call
//! (~1 KB over the bus, ~27 ms at 400 kHz -- nothing at token cadence).

use core::ptr::{read_volatile, write_volatile};

const TWIM_BASE: usize = 0x500C_8000; // TWIM22, secure alias
const P3_BASE: usize = 0x500D_8600; // GPIO port 3, secure alias

const TASKS_STOP: usize = 0x004;
const TASKS_TX_START: usize = 0x050; // TASKS_DMA.TX.START
const EVENTS_STOPPED: usize = 0x104;
const EVENTS_ERROR: usize = 0x114;
const EVENTS_LASTTX: usize = 0x138;
const ERRORSRC: usize = 0x4C4;
const ENABLE: usize = 0x500;
const FREQUENCY: usize = 0x524;
const ADDRESS: usize = 0x588;
const PSEL_SCL: usize = 0x600;
const PSEL_SDA: usize = 0x604;
const TX_PTR: usize = 0x73C; // DMA.TX.PTR
const TX_MAXCNT: usize = 0x740; // DMA.TX.MAXCNT

const ENABLE_TWIM: u32 = 6;
const FREQ_K400: u32 = 0x0640_0000;

const GPIO_PIN_CNF: usize = 0x080;
// input buffer connected, pull-up, DRIVE0=S0 DRIVE1=D1 (open drain '1')
const CNF_TWI: u32 = (3 << 2) | (2 << 10);

const PIN_SCL: u32 = 3;
const PIN_SDA: u32 = 2;
const PORT: u32 = 3;

pub const COLS: usize = 21;
pub const ROWS: usize = 8;

static mut PRESENT: bool = false;
static mut ADDR7: u32 = 0x3C;
static mut GRID: [u8; COLS * ROWS] = [b' '; COLS * ROWS];
static mut CUR_ROW: usize = 0;
static mut CUR_COL: usize = 0;

#[inline]
fn twim(off: usize) -> *mut u32 {
    (TWIM_BASE + off) as *mut u32
}

/// One I2C write transaction. Returns false on NACK/timeout (and disarms
/// the display on timeout so a flaky wire cannot wedge the transcriber).
fn twi_write(addr: u32, buf: &[u8]) -> bool {
    unsafe {
        write_volatile(twim(ADDRESS), addr);
        write_volatile(twim(EVENTS_STOPPED), 0);
        write_volatile(twim(EVENTS_ERROR), 0);
        write_volatile(twim(EVENTS_LASTTX), 0);
        write_volatile(twim(TX_PTR), buf.as_ptr() as u32);
        write_volatile(twim(TX_MAXCNT), buf.len() as u32);
        write_volatile(twim(TASKS_TX_START), 1);
        // Generous vs the longest frame (129 B at 400 kHz = 3.3 ms), tiny
        // vs the boot budget when the bus is stuck.
        let mut ok = false;
        for _ in 0..1_000_000u32 {
            if read_volatile(twim(EVENTS_ERROR)) != 0 {
                break;
            }
            if read_volatile(twim(EVENTS_LASTTX)) != 0 {
                ok = true;
                break;
            }
        }
        write_volatile(twim(TASKS_STOP), 1);
        let mut stopped = false;
        for _ in 0..1_000_000u32 {
            if read_volatile(twim(EVENTS_STOPPED)) != 0 {
                stopped = true;
                break;
            }
        }
        if read_volatile(twim(EVENTS_ERROR)) != 0 {
            ok = false;
        }
        let src = read_volatile(twim(ERRORSRC));
        if src != 0 {
            write_volatile(twim(ERRORSRC), src); // write-1-to-clear
        }
        if !stopped {
            PRESENT = false; // bus wedged: give up on the display
        }
        ok && stopped
    }
}

fn cmd(bytes: &[u8]) -> bool {
    let mut buf = [0u8; 8];
    buf[1..1 + bytes.len()].copy_from_slice(bytes); // buf[0]=0x00 control
    twi_write(unsafe { ADDR7 }, &buf[..1 + bytes.len()])
}

/// Probe for the display and bring it up. Safe to call when absent.
pub fn init() -> bool {
    unsafe {
        for pin in [PIN_SCL, PIN_SDA] {
            write_volatile(
                (P3_BASE + GPIO_PIN_CNF + 4 * pin as usize) as *mut u32,
                CNF_TWI,
            );
        }
        write_volatile(twim(ENABLE), 0);
        write_volatile(twim(PSEL_SCL), (PORT << 5) | PIN_SCL);
        write_volatile(twim(PSEL_SDA), (PORT << 5) | PIN_SDA);
        write_volatile(twim(FREQUENCY), FREQ_K400);
        write_volatile(twim(ENABLE), ENABLE_TWIM);
        PRESENT = true; // provisionally, for the probe writes

        // 0x3C is the common SSD1306 address, 0x3D the alternate strap.
        ADDR7 = 0x3C;
        if !cmd(&[0xAE]) {
            ADDR7 = 0x3D;
            if !cmd(&[0xAE]) {
                // Leave TWIM enabled: erratum [105] wedges the peripheral
                // if it is disabled while a target stretches the clock.
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
