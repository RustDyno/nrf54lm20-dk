//! SD card in SPI mode on SPIM00, the high-speed SPI instance (32 MHz).
//!
//! Wiring (microSD breakout to the DK expansion board header P17, 3.3 V --
//! set VDD:nRF to 3.3 V and route P2.00-P2.05 to the headers with the Board
//! Configurator app; by default the analog switches connect them to the
//! on-board NOR flash instead):
//!   SCK  -> P2.01 (P17 pin 22)   MOSI -> P2.02 (P17 pin 23)
//!   MISO -> P2.04 (P17 pin 25)   CS   -> P2.05 (P17 pin 26)
//! These are SPIM00's dedicated pins (HSSPI.SCK/MOSI/MISO/CSN in the pin
//! assignment tables); the 16 MHz-domain SPIM instances cannot exceed 8 MHz
//! and cannot reach port P2 at all.
//!
//! The card holds the model image (blobs, LUTs, params, embeddings) written
//! raw by model/make_sd_image.py -- no filesystem, just 512-byte blocks
//! addressed by the image's index. Reads are the hot path (weights); writes
//! back activation spill during standalone encoding.
//!
//! Register map from the nRF54LM20A SVD (nrf-pac 0.4.0); the B variant on
//! this DK shares it. SPIM00 is a 128 MHz-domain instance: SCK = 128 MHz /
//! PRESCALER.DIVISOR with DIVISOR in 4..126, so 32 MHz at DIVISOR=4. The
//! minimum divided clock (~1.02 MHz) is above the 400 kHz SD initialization
//! cap, so card init is bit-banged on the same pins at ~250 kHz and the
//! peripheral takes over for the data phase.

use core::ptr::{read_volatile, write_volatile};

// Default bus: SPIM00 at 32 MHz on the dedicated P2 pins (through the
// DK's analog switches). The `sd-spim22` feature instead uses SPIM22 at
// 8 MHz on plain P3 pins (the original wiring: SCK P3.3, MOSI P3.0,
// MISO P3.1, CS P3.2 = P17 pins 14/9/10/13) -- no analog switches in
// the path, standard pads. Diagnostic fallback; the OLED (same serial
// box and pins) is disabled under it.
#[cfg(not(feature = "sd-spim22"))]
const SPIM_BASE: usize = 0x5004_D000; // SPIM00, secure alias
#[cfg(not(feature = "sd-spim22"))]
const GPIO_BASE: usize = 0x5005_0400; // GPIO port 2 (fast pads), secure alias
#[cfg(not(feature = "sd-spim22"))]
const HSPAD_BASE: usize = 0x5005_0400; // GPIOHSPADCTRL overlays the P2 block

#[cfg(feature = "sd-spim22")]
const SPIM_BASE: usize = 0x500C_8000; // SPIM22, secure alias
#[cfg(feature = "sd-spim22")]
const GPIO_BASE: usize = 0x500D_8600; // GPIO port 3, secure alias

// SPIM register offsets (SVD: GLOBAL_SPIM00, all instances share the map).
const TASKS_START: usize = 0x000;
const EVENTS_STARTED: usize = 0x100;
const EVENTS_END: usize = 0x108;
const ENABLE: usize = 0x500;
const PRESCALER: usize = 0x52C;
const CONFIG: usize = 0x554;
const IFTIMING_CSNDUR: usize = 0x5B0;
const ORC: usize = 0x5C0;
// Erratum [8] "SPIM: Wrong data is transmitted on MOSI" (Engineering B):
// with CPHA=0 and PRESCALER > 2 (always true on SPIM00, minimum 4), a
// first transmitted bit of 1 corrupts the data. Workaround per the errata
// doc: CSNDUR >= PRESCALER/2 + 1, write 0x82 to offset 0xC84 before each
// START, and 0x00 back once STARTED has fired.
const ERRATA8_REG: usize = 0xC84;
const PSEL_SCK: usize = 0x600;
const PSEL_MOSI: usize = 0x604;
const PSEL_MISO: usize = 0x608;
const PSEL_CSN: usize = 0x610;
const RX_PTR: usize = 0x704;
const RX_MAXCNT: usize = 0x708;
const TX_PTR: usize = 0x73C;
const TX_MAXCNT: usize = 0x740;

// GPIO port offsets.
const IN: usize = 0x00C;
const OUTSET: usize = 0x004;
const OUTCLR: usize = 0x008;
const DIRSET: usize = 0x014;
const PIN_CNF: usize = 0x080;

// GPIOHSPADCTRL.BIAS: slew control for P2 pads in E0E1 drive. HSBIAS is the
// two low bits; the datasheet says to always use the highest slew (3).
#[cfg(not(feature = "sd-spim22"))]
const HSPAD_BIAS: usize = 0x030;
#[cfg(not(feature = "sd-spim22"))]
const HSBIAS_MAX: u32 = 0x3;

// PIN_CNF: DIR[0], INPUT[1], PULL[3:2], DRIVE0[9:8], DRIVE1[11:10].
// Fast switching on P2 requires extra-high drive on both halves (E0=E1=3).
#[cfg(not(feature = "sd-spim22"))]
const CNF_E0E1: u32 = (3 << 8) | (3 << 10);
const CNF_OUT: u32 = 0x3; // output, input buffer disconnected
const CNF_IN_PULLUP: u32 = 0xC; // input buffer connected, pull-up

#[cfg(not(feature = "sd-spim22"))]
const PIN_SCK: u32 = 1;
#[cfg(not(feature = "sd-spim22"))]
const PIN_MOSI: u32 = 2;
#[cfg(not(feature = "sd-spim22"))]
const PIN_MISO: u32 = 4;
#[cfg(not(feature = "sd-spim22"))]
const PIN_CS: u32 = 5;
#[cfg(not(feature = "sd-spim22"))]
const PORT: u32 = 2;
#[cfg(not(feature = "sd-spim22"))]
const DIV_FAST: u32 = 4; // 128 MHz / 4 = 32 MHz

#[cfg(feature = "sd-spim22")]
const PIN_SCK: u32 = 3;
#[cfg(feature = "sd-spim22")]
const PIN_MOSI: u32 = 0;
#[cfg(feature = "sd-spim22")]
const PIN_MISO: u32 = 1;
#[cfg(feature = "sd-spim22")]
const PIN_CS: u32 = 2;
#[cfg(feature = "sd-spim22")]
const PORT: u32 = 3;
#[cfg(feature = "sd-spim22")]
const DIV_FAST: u32 = 2; // 16 MHz / 2 = 8 MHz

// Bit-bang half period for card init: 256 cycles at 128 MHz = 2 us ->
// 250 kHz, timed with the DWT cycle counter (asm::delay pacing varies
// with instruction fetch behavior; DWT is exact).
const BB_HALF_CYCLES: u32 = 256;

#[inline]
fn dwt_delay(cycles: u32) {
    let start = cortex_m::peripheral::DWT::cycle_count();
    while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < cycles {}
}

pub const BLOCK: usize = 512;

/// True from reset until init() hands the pins to the SPIM peripheral. All
/// traffic funnels through xfer(), so the two phases share every code path.
static mut BITBANG: bool = true;

/// Diagnostic: bit-bang with the two data pins' ROLES exchanged. If the
/// card answers only like this, the physical MOSI/MISO wires are crossed.
static mut SWAP_DATA: bool = false;

#[inline]
fn data_out_pin() -> u32 {
    if unsafe { SWAP_DATA } { PIN_MISO } else { PIN_MOSI }
}

#[inline]
fn data_in_pin() -> u32 {
    if unsafe { SWAP_DATA } { PIN_MOSI } else { PIN_MISO }
}

/// (Re)configure the data pins for the current role assignment.
unsafe fn config_data_pins() {
    let o = data_out_pin();
    let i = data_in_pin();
    write_volatile(gpio(OUTSET), 1 << o);
    write_volatile(gpio(PIN_CNF + 4 * o as usize), CNF_OUT);
    write_volatile(gpio(DIRSET), 1 << o);
    // PIN_CNF.DIR is the same physical register as DIR: this also turns
    // the former output back into an input.
    write_volatile(gpio(PIN_CNF + 4 * i as usize), CNF_IN_PULLUP);
}

#[inline]
fn spim(off: usize) -> *mut u32 {
    (SPIM_BASE + off) as *mut u32
}

#[inline]
fn gpio(off: usize) -> *mut u32 {
    (GPIO_BASE + off) as *mut u32
}

fn cs(low: bool) {
    unsafe {
        write_volatile(gpio(if low { OUTCLR } else { OUTSET }), 1 << PIN_CS);
    }
}

fn bb_byte(tx: u8) -> u8 {
    let mosi = data_out_pin();
    let miso = data_in_pin();
    let mut rx = 0u8;
    for bit in (0..8).rev() {
        unsafe {
            // Mode 0: MOSI changes on the falling edge, both sides sample on
            // the rising edge.
            write_volatile(
                gpio(if tx & (1 << bit) != 0 { OUTSET } else { OUTCLR }),
                1 << mosi,
            );
            dwt_delay(BB_HALF_CYCLES);
            write_volatile(gpio(OUTSET), 1 << PIN_SCK);
            if read_volatile(gpio(IN)) & (1 << miso) != 0 {
                rx |= 1 << bit;
            }
            dwt_delay(BB_HALF_CYCLES);
            write_volatile(gpio(OUTCLR), 1 << PIN_SCK);
        }
    }
    rx
}

/// One full-duplex SPI transaction: send `tx` (0xFF-filled past its end),
/// receive `rx_len` bytes into `rx`. DMA-driven on the peripheral, or
/// bit-banged during card init.
fn xfer(tx: &[u8], rx: &mut [u8]) {
    if unsafe { BITBANG } {
        for &b in tx {
            bb_byte(b);
        }
        for r in rx.iter_mut() {
            *r = bb_byte(0xFF);
        }
        return;
    }
    unsafe {
        write_volatile(spim(TX_PTR), tx.as_ptr() as u32);
        write_volatile(spim(TX_MAXCNT), tx.len() as u32);
        write_volatile(spim(RX_PTR), rx.as_mut_ptr() as u32);
        write_volatile(spim(RX_MAXCNT), rx.len() as u32);
        write_volatile(spim(EVENTS_STARTED), 0);
        write_volatile(spim(EVENTS_END), 0);
        if DIV_FAST > 2 {
            // erratum [8] applies only above PRESCALER 2
            write_volatile(spim(ERRATA8_REG), 0x82);
        }
        write_volatile(spim(TASKS_START), 1);
        while read_volatile(spim(EVENTS_STARTED)) == 0 {}
        if DIV_FAST > 2 {
            write_volatile(spim(ERRATA8_REG), 0x00);
        }
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

/// CMD0 with retries; returns the last R1 (0xFF = total silence). Leaves
/// CS low on success, high on failure.
fn cmd0_probe() -> u8 {
    let mut r = 0xFF;
    for _ in 0..8 {
        cs(false);
        recv1(); // 8 clocks with CS low before the frame
        r = command(0, 0, 0x95);
        if r == 0x01 {
            return r;
        }
        cs(true);
        recv1(); // 8 deselected clocks between attempts
    }
    r
}

/// Bring up the card (bit-banged SPI-mode entry + v2 negotiation), then hand
/// the pins to SPIM00 at 32 MHz. Returns 0, or a negative stage-tagged error
/// (-2xx = stage xx, -460 = data wires crossed).
pub fn init() -> i32 {
    unsafe {
        BITBANG = true;
        SWAP_DATA = false;
        write_volatile(spim(ENABLE), 0);

        // SCK/MOSI/CS as outputs (SCK idle low, MOSI/CS idle high), MISO
        // input with pull-up. STANDARD drive for the init phase: E0E1's
        // nanosecond edges ring hard on jumper wiring, and a ring on SCK
        // re-crossing the card's threshold is a phantom clock -- the C3
        // line monitor showed the card receiving a bit-perfect CMD0
        // (3342/3392 expected SCK edges, 81/80 MOSI, 16/16 CS) and
        // staying mute; a softer driver (ESP32-C3) talked to the same
        // card at the same speed without issue.
        write_volatile(gpio(OUTCLR), 1 << PIN_SCK);
        write_volatile(gpio(OUTSET), (1 << PIN_MOSI) | (1 << PIN_CS));
        for pin in [PIN_SCK, PIN_MOSI, PIN_CS] {
            write_volatile(gpio(PIN_CNF + 4 * pin as usize), CNF_OUT);
        }
        write_volatile(gpio(DIRSET), (1 << PIN_SCK) | (1 << PIN_MOSI) | (1 << PIN_CS));
        write_volatile(gpio(PIN_CNF + 4 * PIN_MISO as usize), CNF_IN_PULLUP);
    }

    // >= 74 clocks with CS high puts the card in SPI-command mode; send
    // 160 (some cards want extra right after power-up).
    cs(true);
    let mut warmup = [0u8; 20];
    recv(&mut warmup);

    // CMD0: software reset -> idle state. Retried: real cards commonly
    // ignore the first attempt(s) after power-up.
    let r = cmd0_probe();
    if r == 0x00 {
        // 0x00 in the response slot is either a live card answering out
        // of alignment / out of idle, or MISO stuck low. A real card
        // returns to 0xFF idle after its response; a stuck line reads
        // 0x00 forever.
        let mut post = [0u8; 4];
        recv(&mut post);
        cs(true);
        use rtt_target::rprintln;
        if post == [0u8; 4] {
            rprintln!("sd: MISO reads permanently LOW (stuck line/short)");
            return -461;
        }
        rprintln!("sd: card RESPONDED but R1=00 (bit slip or already");
        rprintln!("sd: initialized): contact is marginal -- reseat/rewire");
        return -200;
    }
    if r == 0xFF {
        // Total silence: probe with the data-pin roles exchanged. The
        // card itself is the one witness that cannot be mis-tapped -- if
        // it answers like this, the two data wires are crossed.
        unsafe {
            SWAP_DATA = true;
            config_data_pins();
        }
        let r_swapped = cmd0_probe();
        unsafe {
            SWAP_DATA = false;
            config_data_pins();
        }
        cs(true);
        if r_swapped != 0xFF {
            use rtt_target::rprintln;
            rprintln!("sd: the card answers ONLY with the data pins swapped:");
            rprintln!("sd: MOSI/MISO wires are CROSSED at the breakout.");
            rprintln!("sd: exchange the two data wires (SPIM needs them straight).");
            return -460;
        }
        return -200 - r as i32;
    }
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

    // Data phase: hand SCK/MOSI/MISO to the SPIM (CS stays a GPIO). Only
    // now raise SCK/MOSI to extra-high drive with the fast pad slew --
    // 32 MHz needs it; CS switches once per transaction and stays soft.
    // (P2 fast pads only; the SPIM22 fallback runs standard pads at 8 MHz.)
    unsafe {
        #[cfg(not(feature = "sd-spim22"))]
        {
            write_volatile((HSPAD_BASE + HSPAD_BIAS) as *mut u32, HSBIAS_MAX);
            for pin in [PIN_SCK, PIN_MOSI] {
                write_volatile(gpio(PIN_CNF + 4 * pin as usize), CNF_OUT | CNF_E0E1);
            }
        }
        write_volatile(spim(PSEL_SCK), (PORT << 5) | PIN_SCK);
        write_volatile(spim(PSEL_MOSI), (PORT << 5) | PIN_MOSI);
        write_volatile(spim(PSEL_MISO), (PORT << 5) | PIN_MISO);
        write_volatile(spim(PSEL_CSN), 1 << 31); // CS is ours, disconnect
        write_volatile(spim(CONFIG), 0); // mode 0, MSB first
        write_volatile(spim(ORC), 0xFF);
        write_volatile(spim(PRESCALER), DIV_FAST);
        write_volatile(spim(IFTIMING_CSNDUR), DIV_FAST / 2 + 1); // erratum [8]
        write_volatile(spim(ENABLE), 7);
        BITBANG = false;
    }
    0
}

/// Wiring diagnostic for a failed init: holds each driven line at
/// DMM-visible static levels (measure at the CARD SOCKET pads, not the
/// header, to test the whole path), exercises MISO's pulls, and runs a
/// MOSI->MISO loopback probe (jumper the two at the breakout, card out,
/// to prove the full digital path both ways). Assumes init() already
/// configured the pins; leaves them in the idle state.
pub fn diag(cycles: u32) {
    use rtt_target::rprintln;
    const SEC: u32 = 128_000_000; // 1 s of DWT cycles at 128 MHz
    unsafe { BITBANG = true };
    for c in 0..cycles {
        rprintln!("sd diag {}/{} (measure at the card socket pads):", c + 1, cycles);
        for (name, pin, idle_high) in [
            ("SCK  P2.01", PIN_SCK, false),
            ("MOSI P2.02", PIN_MOSI, true),
            ("CS   P2.05", PIN_CS, true),
        ] {
            rprintln!("  {} LOW for 3 s...", name);
            unsafe { write_volatile(gpio(OUTCLR), 1 << pin) };
            dwt_delay(3 * SEC);
            rprintln!("  {} HIGH for 3 s...", name);
            unsafe { write_volatile(gpio(OUTSET), 1 << pin) };
            dwt_delay(3 * SEC);
            if !idle_high {
                unsafe { write_volatile(gpio(OUTCLR), 1 << pin) };
            }
        }
        rprintln!("  MISO P2.04 pull-DOWN for 3 s (a breakout pull-up may hold");
        rprintln!("  the node mid-rail; the read below shows the SoC's view)...");
        unsafe {
            write_volatile(gpio(PIN_CNF + 4 * PIN_MISO as usize), 0x4);
        }
        dwt_delay(3 * SEC);
        let down = unsafe { read_volatile(gpio(IN)) >> PIN_MISO } & 1;
        unsafe {
            write_volatile(gpio(PIN_CNF + 4 * PIN_MISO as usize), CNF_IN_PULLUP);
        }
        dwt_delay(SEC / 100);
        let up = unsafe { read_volatile(gpio(IN)) >> PIN_MISO } & 1;
        rprintln!("  MISO input reads: pulled-down={} pulled-up={}", down, up);
        let mut ok = 0;
        for &b in &[0xA5u8, 0x3C, 0x0F, 0x81] {
            let got = bb_byte(b);
            rprintln!("  loopback sent {:02X} got {:02X}", b, got);
            if got == b {
                ok += 1;
            }
        }
        rprintln!("  MOSI->MISO loopback (needs jumper, card out): {}/4", ok);
    }
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
