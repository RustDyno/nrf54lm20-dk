//! SD card in SPI mode on SPIM22 (the DK's "nordic expansion" SPI).
//!
//! Wiring (microSD breakout to the DK expansion header, 3.3 V):
//!   SCK  -> P3.3    MOSI -> P3.0    MISO -> P3.1    CS -> P3.2
//! These match the board devicetree's `nordic_expansion_spi` pinout, so
//! standard expansion shields agree with us.
//!
//! The card holds the model image (blobs, LUTs, params, embeddings) written
//! raw by model/make_sd_image.py -- no filesystem, just 512-byte blocks
//! addressed by the image's index. Reads are the hot path (weights); writes
//! back activation spill during standalone encoding.
//!
//! Register map from the nRF54LM20A SVD (nrf-pac 0.4.0); the B variant on
//! this DK shares it. SPIM22 is a 16 MHz-domain instance: SCK = 16 MHz /
//! PRESCALER.DIVISOR, so 8 MHz data clock, 250 kHz during card init.

use core::ptr::{read_volatile, write_volatile};

const SPIM_BASE: usize = 0x500C_8000; // SPIM22, secure alias
const P3_BASE: usize = 0x500D_8600; // GPIO port 3, secure alias

// SPIM register offsets (SVD: GLOBAL_SPIM00, all instances share the map).
const TASKS_START: usize = 0x000;
const EVENTS_END: usize = 0x108;
const ENABLE: usize = 0x500;
const PRESCALER: usize = 0x52C;
const CONFIG: usize = 0x554;
const ORC: usize = 0x5C0;
const PSEL_SCK: usize = 0x600;
const PSEL_MOSI: usize = 0x604;
const PSEL_MISO: usize = 0x608;
const PSEL_CSN: usize = 0x610;
const RX_PTR: usize = 0x704;
const RX_MAXCNT: usize = 0x708;
const TX_PTR: usize = 0x73C;
const TX_MAXCNT: usize = 0x740;

// GPIO port offsets.
const OUTSET: usize = 0x004;
const OUTCLR: usize = 0x008;
const DIRSET: usize = 0x014;
const PIN_CNF: usize = 0x080;

const PIN_SCK: u32 = 3;
const PIN_MOSI: u32 = 0;
const PIN_MISO: u32 = 1;
const PIN_CS: u32 = 2;
const PORT: u32 = 3;

const DIV_INIT: u32 = 64; // 16 MHz / 64 = 250 kHz (SD init needs <= 400 kHz)
const DIV_FAST: u32 = 2; // 16 MHz / 2 = 8 MHz

pub const BLOCK: usize = 512;

#[inline]
fn spim(off: usize) -> *mut u32 {
    (SPIM_BASE + off) as *mut u32
}

#[inline]
fn gpio(off: usize) -> *mut u32 {
    (P3_BASE + off) as *mut u32
}

fn cs(low: bool) {
    unsafe {
        write_volatile(gpio(if low { OUTCLR } else { OUTSET }), 1 << PIN_CS);
    }
}

/// One full-duplex SPI transaction: send `tx` (0xFF-filled past its end),
/// receive `rx_len` bytes into `rx`. Polled on EVENTS_END.
fn xfer(tx: &[u8], rx: &mut [u8]) {
    unsafe {
        write_volatile(spim(TX_PTR), tx.as_ptr() as u32);
        write_volatile(spim(TX_MAXCNT), tx.len() as u32);
        write_volatile(spim(RX_PTR), rx.as_mut_ptr() as u32);
        write_volatile(spim(RX_MAXCNT), rx.len() as u32);
        write_volatile(spim(EVENTS_END), 0);
        write_volatile(spim(TASKS_START), 1);
        while read_volatile(spim(EVENTS_END)) == 0 {}
    }
}

fn send(tx: &[u8]) {
    let mut sink = [0u8; 0];
    xfer(tx, &mut sink);
}

fn recv(rx: &mut [u8]) {
    xfer(&[], rx); // ORC=0xFF keeps MOSI high
}

fn recv1() -> u8 {
    let mut b = [0u8; 1];
    recv(&mut b);
    b[0]
}

/// Send a command frame, return the R1 response (poll up to 16 bytes).
fn command(cmd: u8, arg: u32, crc: u8) -> u8 {
    let frame = [
        0x40 | cmd,
        (arg >> 24) as u8,
        (arg >> 16) as u8,
        (arg >> 8) as u8,
        arg as u8,
        crc,
    ];
    send(&frame);
    for _ in 0..16 {
        let r = recv1();
        if r & 0x80 == 0 {
            return r;
        }
    }
    0xFF
}

static mut HIGH_CAPACITY: bool = false;

/// Byte-addressed cards multiply the LBA by the block size.
fn card_addr(lba: u32) -> u32 {
    if unsafe { HIGH_CAPACITY } {
        lba
    } else {
        lba * BLOCK as u32
    }
}

/// Bring up SPIM22 + the card: SPI-mode entry, v2 negotiation, fast clock.
/// Returns 0, or a negative stage-tagged error (-2xx = stage xx).
pub fn init() -> i32 {
    unsafe {
        // CS as a plain GPIO output, idle high; MISO gets a pull-up.
        write_volatile(gpio(PIN_CNF + 4 * PIN_CS as usize), 0x3);
        write_volatile(gpio(OUTSET), 1 << PIN_CS);
        write_volatile(gpio(DIRSET), 1 << PIN_CS);
        write_volatile(gpio(PIN_CNF + 4 * PIN_MISO as usize), 0xC); // input, pull-up

        write_volatile(spim(ENABLE), 0);
        write_volatile(spim(PSEL_SCK), (PORT << 5) | PIN_SCK);
        write_volatile(spim(PSEL_MOSI), (PORT << 5) | PIN_MOSI);
        write_volatile(spim(PSEL_MISO), (PORT << 5) | PIN_MISO);
        write_volatile(spim(PSEL_CSN), 1 << 31); // CS is ours, disconnect
        write_volatile(spim(CONFIG), 0); // mode 0, MSB first
        write_volatile(spim(ORC), 0xFF);
        write_volatile(spim(PRESCALER), DIV_INIT);
        write_volatile(spim(ENABLE), 7);
    }

    // >= 74 clocks with CS high puts the card in SPI-command mode.
    cs(false);
    cs(true);
    let mut warmup = [0u8; 10];
    recv(&mut warmup);

    // CMD0: software reset -> idle state.
    cs(false);
    let r = command(0, 0, 0x95);
    if r != 0x01 {
        cs(true);
        return -200 - r as i32;
    }

    // CMD8: v2 check pattern.
    let r = command(8, 0x1AA, 0x87);
    let v2 = r == 0x01;
    if v2 {
        let mut r7 = [0u8; 4];
        recv(&mut r7);
        if r7[2] & 0x0F != 0x01 || r7[3] != 0xAA {
            cs(true);
            return -210;
        }
    }

    // ACMD41 until the card leaves idle (HCS set for v2).
    let mut ok = false;
    for _ in 0..2500 {
        command(55, 0, 0xFF);
        let r = command(41, if v2 { 1 << 30 } else { 0 }, 0xFF);
        if r == 0 {
            ok = true;
            break;
        }
        if r != 0x01 {
            cs(true);
            return -220 - r as i32;
        }
    }
    if !ok {
        cs(true);
        return -230;
    }

    // CMD58: OCR -> block vs byte addressing.
    let r = command(58, 0, 0xFF);
    if r != 0 {
        cs(true);
        return -240 - r as i32;
    }
    let mut ocr = [0u8; 4];
    recv(&mut ocr);
    unsafe { HIGH_CAPACITY = ocr[0] & 0x40 != 0 };
    cs(true);
    recv1(); // 8 clocks after CS release

    unsafe {
        write_volatile(spim(ENABLE), 0);
        write_volatile(spim(PRESCALER), DIV_FAST);
        write_volatile(spim(ENABLE), 7);
    }
    0
}

/// Read `count` 512-byte blocks starting at `lba` into `dst` (CMD18).
pub fn read_blocks(lba: u32, dst: *mut u8, count: u32) -> i32 {
    cs(false);
    let r = command(18, card_addr(lba), 0xFF);
    if r != 0 {
        cs(true);
        return -300 - r as i32;
    }
    for i in 0..count {
        // wait for the data token
        let mut token = 0xFFu8;
        for _ in 0..200_000 {
            token = recv1();
            if token != 0xFF {
                break;
            }
        }
        if token != 0xFE {
            cs(true);
            return -310;
        }
        let blk = unsafe {
            core::slice::from_raw_parts_mut(dst.add((i as usize) * BLOCK), BLOCK)
        };
        recv(blk);
        let mut crc = [0u8; 2];
        recv(&mut crc); // CRC not checked (off in SPI mode)
    }
    command(12, 0, 0xFF); // stop transmission
    // the card holds the line busy (0x00) while finishing
    for _ in 0..200_000 {
        if recv1() == 0xFF {
            cs(true);
            return 0;
        }
    }
    cs(true);
    -320
}

/// Write `count` 512-byte blocks starting at `lba` from `src` (CMD25).
pub fn write_blocks(lba: u32, src: *const u8, count: u32) -> i32 {
    cs(false);
    let r = command(25, card_addr(lba), 0xFF);
    if r != 0 {
        cs(true);
        return -400 - r as i32;
    }
    for i in 0..count {
        send(&[0xFF, 0xFC]); // gap + multi-block data token
        let blk = unsafe {
            core::slice::from_raw_parts(src.add((i as usize) * BLOCK), BLOCK)
        };
        send(blk);
        send(&[0xFF, 0xFF]); // dummy CRC
        let resp = recv1();
        if resp & 0x1F != 0x05 {
            cs(true);
            return -410 - (resp & 0x1F) as i32;
        }
        let mut busy = false;
        for _ in 0..500_000 {
            if recv1() == 0xFF {
                busy = true;
                break;
            }
        }
        if !busy {
            cs(true);
            return -420;
        }
    }
    send(&[0xFD]); // stop tran token
    recv1();
    for _ in 0..500_000 {
        if recv1() == 0xFF {
            cs(true);
            return 0;
        }
    }
    cs(true);
    -430
}
