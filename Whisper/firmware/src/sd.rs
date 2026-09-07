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
//! Registers go through the PAC (`embassy_nrf::pac`). The HAL's SPIM driver
//! is not used: its transfers wait on EVENTS_END, which never fires on this
//! part while CSN is disconnected (hardware-observed, see xfer), it cannot
//! keep CS asserted across a multi-transfer SD command, and it has no
//! erratum [8] handling. SPIM00 is a 128 MHz-domain instance: SCK = 128 MHz
//! / PRESCALER.DIVISOR with DIVISOR in 4..126, so 32 MHz at DIVISOR=4. The
//! minimum divided clock (~1.02 MHz) is above the 400 kHz SD initialization
//! cap, so card init is bit-banged on the same pins at ~250 kHz and the
//! peripheral takes over for the data phase.

use embassy_nrf::pac;
use pac::common::{Reg, RW};
#[cfg(not(feature = "sd-spim22"))]
use pac::gpio::vals::Drive;
use pac::gpio::vals::{Dir, Input, Pull};
use pac::shared::vals::Connect;
use pac::spim::vals::{Cpha, Cpol, Enable, Order};

// Default bus: SPIM00 at 32 MHz on the dedicated P2 pins (through the
// DK's analog switches). The `sd-spim22` feature instead uses SPIM22 at
// 8 MHz on plain P3 pins (the original wiring: SCK P3.3, MOSI P3.0,
// MISO P3.1, CS P3.2 = P17 pins 14/9/10/13) -- no analog switches in
// the path, standard pads. Diagnostic fallback; the OLED (same serial
// box and pins) is disabled under it.
#[cfg(not(feature = "sd-spim22"))]
const SPIM: pac::spim::Spim = pac::SPIM00;
#[cfg(not(feature = "sd-spim22"))]
const GPIO: pac::gpio::Gpio = pac::P2; // fast pads

#[cfg(feature = "sd-spim22")]
const SPIM: pac::spim::Spim = pac::SPIM22;
#[cfg(feature = "sd-spim22")]
const GPIO: pac::gpio::Gpio = pac::P3_S; // port 3 has no unsuffixed alias

// Erratum [8] "SPIM: Wrong data is transmitted on MOSI" (Engineering B):
// with CPHA=0 and PRESCALER > 2 (always true on SPIM00, minimum 4), a
// first transmitted bit of 1 corrupts the data. Workaround per the errata
// doc: CSNDUR >= PRESCALER/2 + 1, write 0x82 to offset 0xC84 before each
// START, and 0x00 back once STARTED has fired. The register is not in the
// SVD, so it is the one raw offset in this driver.
const ERRATA8_OFFSET: usize = 0xC84;

fn errata8_reg() -> Reg<u32, RW> {
    unsafe { Reg::from_ptr((SPIM.as_ptr() as *mut u8).add(ERRATA8_OFFSET) as *mut u32) }
}

// GPIOHSPADCTRL.BIAS: slew control for P2 pads in E0E1 drive. HSBIAS is the
// two low bits; the datasheet says to always use the highest slew (3).
#[cfg(not(feature = "sd-spim22"))]
const HSBIAS_MAX: u8 = 0x3;

#[cfg(not(feature = "sd-spim22"))]
const PIN_SCK: usize = 1;
#[cfg(not(feature = "sd-spim22"))]
const PIN_MOSI: usize = 2;
#[cfg(not(feature = "sd-spim22"))]
const PIN_MISO: usize = 4;
#[cfg(not(feature = "sd-spim22"))]
const PIN_CS: usize = 5;
#[cfg(not(feature = "sd-spim22"))]
const PORT: u8 = 2;
#[cfg(not(feature = "sd-spim22"))]
const DIV_FAST: u8 = 4; // 128 MHz / 4 = 32 MHz

#[cfg(feature = "sd-spim22")]
const PIN_SCK: usize = 3;
#[cfg(feature = "sd-spim22")]
const PIN_MOSI: usize = 0;
#[cfg(feature = "sd-spim22")]
const PIN_MISO: usize = 1;
#[cfg(feature = "sd-spim22")]
const PIN_CS: usize = 2;
#[cfg(feature = "sd-spim22")]
const PORT: u8 = 3;
#[cfg(feature = "sd-spim22")]
const DIV_FAST: u8 = 2; // 16 MHz / 2 = 8 MHz

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
fn data_out_pin() -> usize {
    if unsafe { SWAP_DATA } { PIN_MISO } else { PIN_MOSI }
}

#[inline]
fn data_in_pin() -> usize {
    if unsafe { SWAP_DATA } { PIN_MOSI } else { PIN_MISO }
}

// --- GPIO helpers (port register level: the pins change role between the
// bit-banged init phase and the SPIM data phase, and the diagnostics
// drive them by hand).

#[inline]
fn pin_high(pin: usize) {
    GPIO.outset().write(|w| w.set_pin(pin, true));
}

#[inline]
fn pin_low(pin: usize) {
    GPIO.outclr().write(|w| w.set_pin(pin, true));
}

#[inline]
fn pin_read(pin: usize) -> bool {
    GPIO.in_().read().pin(pin)
}

/// Output, input buffer disconnected, standard drive.
fn cnf_out(pin: usize) {
    GPIO.pin_cnf(pin).write(|w| {
        w.set_dir(Dir::Output);
        w.set_input(Input::Disconnect);
    });
}

/// Output with extra-high drive on both halves (E0E1): fast switching on
/// the P2 pads needs it.
#[cfg(not(feature = "sd-spim22"))]
fn cnf_out_e0e1(pin: usize) {
    GPIO.pin_cnf(pin).write(|w| {
        w.set_dir(Dir::Output);
        w.set_input(Input::Disconnect);
        w.set_drive0(Drive::E);
        w.set_drive1(Drive::E);
    });
}

/// Input buffer connected, pull-up.
fn cnf_in_pullup(pin: usize) {
    GPIO.pin_cnf(pin).write(|w| {
        w.set_dir(Dir::Input);
        w.set_input(Input::Connect);
        w.set_pull(Pull::Pullup);
    });
}

/// (Re)configure the data pins for the current role assignment.
fn config_data_pins() {
    let o = data_out_pin();
    let i = data_in_pin();
    pin_high(o);
    cnf_out(o);
    GPIO.dirset().write(|w| w.set_pin(o, true));
    // PIN_CNF.DIR is the same physical register as DIR: this also turns
    // the former output back into an input.
    cnf_in_pullup(i);
}

// ROOT CAUSE of the long -455 hunt lived here: the old helper was
// `cs(low: bool)` but every call site passed `cs(false)` meaning
// "assert" -- so the warmup clocks ran with the card SELECTED and every
// command frame went out DESELECTED, which a card ignores by design.
// The C3 sniffer showed the inverted CS phase from its first capture.
// Explicit names so the polarity is visible at every call site:

/// Assert chip select (drive CS LOW: card listens).
fn cs_assert() {
    pin_low(PIN_CS);
}

/// Release chip select (drive CS HIGH: card deselected).
fn cs_release() {
    pin_high(PIN_CS);
}

fn bb_byte(tx: u8) -> u8 {
    let mosi = data_out_pin();
    let miso = data_in_pin();
    let mut rx = 0u8;
    for bit in (0..8).rev() {
        // Mode 0: MOSI changes on the falling edge, both sides sample on
        // the rising edge.
        if tx & (1 << bit) != 0 {
            pin_high(mosi);
        } else {
            pin_low(mosi);
        }
        dwt_delay(BB_HALF_CYCLES);
        pin_high(PIN_SCK);
        if pin_read(miso) {
            rx |= 1 << bit;
        }
        dwt_delay(BB_HALF_CYCLES);
        pin_low(PIN_SCK);
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
        if SPIM_FAULT {
            // a previous transfer faulted: fail fast instead of piling
            // 20 ms timeouts on every subsequent byte
            for r in rx.iter_mut() {
                *r = 0xFF;
            }
            return;
        }
        // Real pointers even for zero-length directions: empty-slice
        // pointers are dangling, and the nRF54 EasyDMA validates bus
        // addresses (TERMINATEONBUSERROR machinery) where nRF52 did not.
        static mut DMA_DUMMY: [u8; 4] = [0; 4];
        let dummy = core::ptr::addr_of_mut!(DMA_DUMMY) as u32;
        let txp = if tx.is_empty() { dummy } else { tx.as_ptr() as u32 };
        let rxp = if rx.is_empty() { dummy } else { rx.as_mut_ptr() as u32 };
        SPIM.dma().tx().ptr().write_value(txp);
        SPIM.dma().tx().maxcnt().write(|w| w.set_maxcnt(tx.len() as u16));
        SPIM.dma().rx().ptr().write_value(rxp);
        SPIM.dma().rx().maxcnt().write(|w| w.set_maxcnt(rx.len() as u16));
        SPIM.events_started().write_value(0);
        SPIM.events_end().write_value(0);
        if DIV_FAST > 2 {
            // erratum [8] applies only above PRESCALER 2
            errata8_reg().write_value(0x82);
        }
        SPIM.events_dma().rx().end().write_value(0);
        SPIM.events_dma().tx().end().write_value(0);
        SPIM.events_stopped().write_value(0);
        SPIM.tasks_start().write_value(1);
        let ok_started = spim_wait(SPIM.events_started());
        if DIV_FAST > 2 {
            errata8_reg().write_value(0x00);
        }
        // The nRF54 SPIM's EVENTS_END is tied to the hardware-CSN
        // transaction framing, and our CSN is disconnected (the SD
        // protocol holds CS across many transfers): with software CS the
        // END event never fires (hardware-observed: both DMA END events
        // set, EVENTS_END stuck 0). Completion = both DMA directions
        // done; then STOP closes the engine's transaction state.
        let ok_end = ok_started
            && spim_wait(SPIM.events_dma().rx().end())
            && spim_wait(SPIM.events_dma().tx().end());
        if !ok_end {
            spim_fault_dump(if ok_started { "DMA END" } else { "STARTED" });
            return;
        }
        SPIM.tasks_stop().write_value(1);
        // Best-effort: erratum [69] says STOPPED can fail to assert in
        // corner cases; a bounded wait keeps that from wedging us.
        let stop_start = cortex_m::peripheral::DWT::cycle_count();
        while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(stop_start)
            < 128_000
        {
            if SPIM.events_stopped().read() != 0 {
                break;
            }
        }
        SPIM.events_stopped().write_value(0);
    }
}

/// Sticky fault flag: set on the first SPIM event timeout; read_blocks /
/// write_blocks turn it into a distinct error code.
static mut SPIM_FAULT: bool = false;

/// Wait up to 20 ms (DWT-timed) for an event register.
fn spim_wait(event: Reg<u32, RW>) -> bool {
    let start = cortex_m::peripheral::DWT::cycle_count();
    while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < 2_560_000 {
        if event.read() != 0 {
            return true;
        }
    }
    false
}

/// One-shot diagnostic dump when a SPIM transfer times out.
unsafe fn spim_fault_dump(which: &str) {
    use rtt_target::{rprint, rprintln};
    SPIM_FAULT = true;
    rprintln!("sd: SPIM transfer timed out waiting for {}", which);
    rprintln!(
        "sd: STARTED={} END={} ENABLE={:#x} PRESC={} CONFIG={:#x}",
        SPIM.events_started().read(),
        SPIM.events_end().read(),
        SPIM.enable().read().0,
        SPIM.prescaler().read().0,
        SPIM.config().read().0,
    );
    let rx = SPIM.events_dma().rx();
    let tx = SPIM.events_dma().tx();
    rprint!("sd: EVENTS_DMA rx end/ready/buserr/match0-3:");
    rprint!(" {:x} {:x} {:x}", rx.end().read(), rx.ready().read(), rx.buserror().read());
    for i in 0..4 {
        rprint!(" {:x}", rx.match_(i).read());
    }
    rprintln!(
        " tx end/ready/buserr: {:x} {:x} {:x}",
        tx.end().read(),
        tx.ready().read(),
        tx.buserror().read()
    );
    rprintln!(
        "sd: RX buserr @{:#010x} TX buserr @{:#010x}",
        SPIM.dma().rx().buserroraddress().read(),
        SPIM.dma().tx().buserroraddress().read(),
    );
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
/// CS low on success, high on failure. The first attempt's 16 poll bytes
/// are printed afterwards -- the SoC's actual received data, the one
/// quantity no external instrument has to be trusted for.
fn cmd0_probe() -> u8 {
    use rtt_target::{rprint, rprintln};
    let mut trace = [0u8; 16];
    let mut r = 0xFF;
    for attempt in 0..8 {
        cs_assert();
        recv1(); // 8 clocks with CS low before the frame
        if attempt == 0 {
            send(&[0x40, 0, 0, 0, 0, 0x95]);
            r = 0xFF;
            for t in trace.iter_mut() {
                *t = recv1();
                if r == 0xFF && *t & 0x80 == 0 {
                    r = *t;
                }
            }
        } else {
            r = command(0, 0, 0x95);
        }
        if r == 0x01 {
            break;
        }
        cs_release();
        recv1(); // 8 deselected clocks between attempts
    }
    rprint!("sd: CMD0 poll bytes:");
    for t in trace {
        rprint!(" {:02X}", t);
    }
    rprintln!(" (r={:02X})", r);
    r
}

fn psel(pin: usize) -> pac::shared::regs::Psel {
    let mut v = pac::shared::regs::Psel(0);
    v.set_pin(pin as u8);
    v.set_port(PORT);
    v.set_connect(Connect::Connected);
    v
}

/// Bring up the card (bit-banged SPI-mode entry + v2 negotiation), then hand
/// the pins to SPIM00 at 32 MHz. Returns 0, or a negative stage-tagged error
/// (-2xx = stage xx, -460 = data wires crossed).
pub fn init() -> i32 {
    unsafe {
        BITBANG = true;
        SWAP_DATA = false;
        SPIM_FAULT = false;
    }
    SPIM.enable().write(|w| w.set_enable(Enable::Disabled));

    // SCK/MOSI/CS as outputs (SCK idle low, MOSI/CS idle high), MISO
    // input with pull-up. STANDARD drive for the init phase: E0E1's
    // nanosecond edges ring hard on jumper wiring, and a ring on SCK
    // re-crossing the card's threshold is a phantom clock -- the C3
    // line monitor showed the card receiving a bit-perfect CMD0
    // (3342/3392 expected SCK edges, 81/80 MOSI, 16/16 CS) and
    // staying mute; a softer driver (ESP32-C3) talked to the same
    // card at the same speed without issue.
    pin_low(PIN_SCK);
    GPIO.outset().write(|w| {
        w.set_pin(PIN_MOSI, true);
        w.set_pin(PIN_CS, true);
    });
    for pin in [PIN_SCK, PIN_MOSI, PIN_CS] {
        cnf_out(pin);
    }
    GPIO.dirset().write(|w| {
        w.set_pin(PIN_SCK, true);
        w.set_pin(PIN_MOSI, true);
        w.set_pin(PIN_CS, true);
    });
    cnf_in_pullup(PIN_MISO);

    // >= 74 clocks with CS high puts the card in SPI-command mode; send
    // 160 (some cards want extra right after power-up).
    cs_release();
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
        cs_release();
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
        unsafe { SWAP_DATA = true };
        config_data_pins();
        let r_swapped = cmd0_probe();
        unsafe { SWAP_DATA = false };
        config_data_pins();
        cs_release();
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
        cs_release();
        return -200 - r as i32;
    }

    // CMD8: v2 check pattern.
    let r = command(8, 0x1AA, 0x87);
    let v2 = r == 0x01;
    if v2 {
        let mut r7 = [0u8; 4];
        recv(&mut r7);
        if r7[2] & 0x0F != 0x01 || r7[3] != 0xAA {
            cs_release();
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
            cs_release();
            return -220 - r as i32;
        }
    }
    if !ok {
        cs_release();
        return -230;
    }

    // CMD58: OCR -> block vs byte addressing.
    let r = command(58, 0, 0xFF);
    if r != 0 {
        cs_release();
        return -240 - r as i32;
    }
    let mut ocr = [0u8; 4];
    recv(&mut ocr);
    unsafe { HIGH_CAPACITY = ocr[0] & 0x40 != 0 };
    cs_release();
    recv1(); // 8 clocks after CS release

    // Data phase: hand SCK/MOSI/MISO to the SPIM (CS stays a GPIO). Only
    // now raise SCK/MOSI to extra-high drive with the fast pad slew --
    // 32 MHz needs it; CS switches once per transaction and stays soft.
    // (P2 fast pads only; the SPIM22 fallback runs standard pads at 8 MHz.)
    #[cfg(not(feature = "sd-spim22"))]
    {
        pac::GPIOHSPADCTRL_S.bias().write(|w| w.set_hsbias(HSBIAS_MAX));
        for pin in [PIN_SCK, PIN_MOSI] {
            cnf_out_e0e1(pin);
        }
    }
    SPIM.psel().sck().write_value(psel(PIN_SCK));
    SPIM.psel().mosi().write_value(psel(PIN_MOSI));
    SPIM.psel().miso().write_value(psel(PIN_MISO));
    // CS is ours: leave CSN disconnected.
    SPIM.psel().csn().write(|w| w.set_connect(Connect::Disconnected));
    SPIM.config().write(|w| {
        // mode 0, MSB first
        w.set_order(Order::MsbFirst);
        w.set_cpha(Cpha::Leading);
        w.set_cpol(Cpol::ActiveHigh);
    });
    SPIM.orc().write(|w| w.set_orc(0xFF));
    SPIM.prescaler().write(|w| w.set_divisor(DIV_FAST));
    SPIM.iftiming().csndur().write(|w| w.set_csndur(DIV_FAST / 2 + 1)); // erratum [8]
    SPIM.enable().write(|w| w.set_enable(Enable::Enabled));
    unsafe { BITBANG = false };
    0
}

/// Release all four SD pins to high-impedance inputs (no pulls). Called
/// after a failed init so an external master (the C3 tester) can drive
/// the shared wires while the DK stays powered and attached.
pub fn release_pins() {
    for pin in [PIN_SCK, PIN_MOSI, PIN_MISO, PIN_CS] {
        GPIO.pin_cnf(pin).write(|w| {
            w.set_dir(Dir::Input);
            w.set_input(Input::Disconnect);
            w.set_pull(Pull::Disabled);
        });
    }
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
            pin_low(pin);
            dwt_delay(3 * SEC);
            rprintln!("  {} HIGH for 3 s...", name);
            pin_high(pin);
            dwt_delay(3 * SEC);
            if !idle_high {
                pin_low(pin);
            }
        }
        rprintln!("  MISO P2.04 pull-DOWN for 3 s (a breakout pull-up may hold");
        rprintln!("  the node mid-rail; the read below shows the SoC's view)...");
        GPIO.pin_cnf(PIN_MISO).write(|w| {
            w.set_dir(Dir::Input);
            w.set_input(Input::Connect);
            w.set_pull(Pull::Pulldown);
        });
        dwt_delay(3 * SEC);
        let down = pin_read(PIN_MISO) as u32;
        cnf_in_pullup(PIN_MISO);
        dwt_delay(SEC / 100);
        let up = pin_read(PIN_MISO) as u32;
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

// Cumulative transfer accounting (bytes and DWT cycles), drained by
// stats_take. Motivation: the second utterance of a session ran its
// encoder ~2x slower than the first with identical work; per-phase
// throughput numbers are the only way to tell a degrading card
// (internal garbage collection after heavy scratch writes) from a
// firmware regression.
static mut RD_BYTES: u64 = 0;
static mut RD_CYC: u64 = 0;
static mut WR_BYTES: u64 = 0;
static mut WR_CYC: u64 = 0;

/// Read and reset the cumulative transfer counters:
/// (read bytes, read cycles, written bytes, write cycles).
pub fn stats_take() -> (u64, u64, u64, u64) {
    unsafe {
        let s = (RD_BYTES, RD_CYC, WR_BYTES, WR_CYC);
        RD_BYTES = 0;
        RD_CYC = 0;
        WR_BYTES = 0;
        WR_CYC = 0;
        s
    }
}

/// Read `count` 512-byte blocks starting at `lba` into `dst` (CMD18).
pub fn read_blocks(lba: u32, dst: *mut u8, count: u32) -> i32 {
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let rc = read_blocks_inner(lba, dst, count);
    unsafe {
        RD_CYC += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
        RD_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

fn read_blocks_inner(lba: u32, dst: *mut u8, count: u32) -> i32 {
    if unsafe { SPIM_FAULT } {
        return -470;
    }
    cs_assert();
    let r = command(18, card_addr(lba), 0xFF);
    if r != 0 {
        cs_release();
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
            cs_release();
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
            cs_release();
            return if unsafe { SPIM_FAULT } { -470 } else { 0 };
        }
    }
    cs_release();
    -320
}

/// Write `count` 512-byte blocks starting at `lba` from `src` (CMD25).
pub fn write_blocks(lba: u32, src: *const u8, count: u32) -> i32 {
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let rc = write_blocks_inner(lba, src, count);
    unsafe {
        WR_CYC += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
        WR_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

fn write_blocks_inner(lba: u32, src: *const u8, count: u32) -> i32 {
    if unsafe { SPIM_FAULT } {
        return -471;
    }
    cs_assert();
    let r = command(25, card_addr(lba), 0xFF);
    if r != 0 {
        cs_release();
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
            cs_release();
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
            cs_release();
            return -420;
        }
    }
    send(&[0xFD]); // stop tran token
    recv1();
    for _ in 0..500_000 {
        if recv1() == 0xFF {
            cs_release();
            return if unsafe { SPIM_FAULT } { -471 } else { 0 };
        }
    }
    cs_release();
    -430
}
