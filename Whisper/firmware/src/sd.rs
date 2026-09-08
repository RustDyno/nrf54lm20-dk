//! SD card in SPI mode on SPIM00, the high-speed SPI instance (32 MHz),
//! over the software-chip-select SPIM driver (`hal::spim`).
//!
//! Wiring (microSD breakout to the DK expansion board header P17, 3.3 V --
//! set VDD:nRF to 3.3 V and route P2.00-P2.05 to the headers with the Board
//! Configurator app; by default the analog switches connect them to the
//! on-board NOR flash instead):
//!   SCK  -> P2.01 (P17 pin 22)   MOSI -> P2.02 (P17 pin 23)
//!   MISO -> P2.04 (P17 pin 25)   CS   -> P2.05 (P17 pin 26)
//! These are SPIM00's dedicated pins (HSSPI.SCK/MOSI/MISO/CSN in the pin
//! assignment tables); the 16 MHz-domain SPIM instances cannot exceed 8 MHz
//! and cannot reach port P2 at all. The `sd-spim22` feature instead uses
//! SPIM22 at 8 MHz on plain P3 pins (the original wiring: SCK P3.3, MOSI
//! P3.0, MISO P3.1, CS P3.2 = P17 pins 14/9/10/13) -- no analog switches
//! in the path, standard pads. Diagnostic fallback; the OLED (same serial
//! box and pins) is disabled under it.
//!
//! The card holds the model image (blobs, LUTs, params, embeddings) written
//! raw by model/make_sd_image.py -- no filesystem, just 512-byte blocks
//! addressed by the image's index. Reads are the hot path (weights); writes
//! back activation spill during standalone encoding.
//!
//! Card init (SPI-mode entry, v2 negotiation) runs on the driver's
//! bit-banged phase at ~250 kHz -- the 128 MHz instance cannot divide down
//! to the 400 kHz initialization cap -- and the peripheral takes over for
//! the data phase at 32 MHz.

use rtt_target::rprintln;

use crate::hal::spim::{self, Drive, Line, SpimSoftCs};

pub const BLOCK: usize = 512;

struct SdCard {
    bus: SpimSoftCs<'static>,
    high_capacity: bool,
}

/// The card, built on the first init() from the board's SD bus
/// singletons and kept for the rest of the run.
static mut CARD: Option<SdCard> = None;

fn bus_config() -> spim::Config {
    let mut config = spim::Config::default();
    if cfg!(feature = "sd-spim22") {
        config.divisor = 2; // 16 MHz / 2 = 8 MHz, standard pads
        config.drive = Drive::Standard;
    } else {
        config.divisor = 4; // 128 MHz / 4 = 32 MHz
        // Fast switching on the P2 pads needs extra-high drive on both
        // halves; CS switches once per transaction and stays soft.
        config.drive = Drive::ExtraHigh;
    }
    config
}

fn card() -> Option<&'static mut SdCard> {
    let slot = unsafe { &mut *core::ptr::addr_of_mut!(CARD) };
    if slot.is_none() {
        let b = crate::board::get().sd.take()?;
        *slot = Some(SdCard {
            bus: SpimSoftCs::new_blocking(b.spim, b.sck, b.mosi, b.miso, b.cs, bus_config()),
            high_capacity: false,
        });
    }
    slot.as_mut()
}

/// One full-duplex transaction: send `tx` (0xFF-filled past its end),
/// receive `rx.len()` bytes into `rx`. The first DMA timeout is dumped;
/// later transfers fail fast (0xFF) until the next init().
fn xfer(bus: &mut SpimSoftCs<'static>, tx: &[u8], rx: &mut [u8]) {
    if let Err(spim::Error::Timeout) = bus.blocking_transfer(tx, rx) {
        if let Some(f) = bus.fault() {
            fault_dump(&f);
        }
    }
}

/// One-shot diagnostic dump when a SPIM transfer times out.
fn fault_dump(f: &spim::FaultSnapshot) {
    rprintln!(
        "sd: SPIM transfer timed out waiting for {}",
        match f.stage {
            spim::Stage::Started => "STARTED",
            spim::Stage::DmaEnd => "DMA END",
        }
    );
    rprintln!(
        "sd: STARTED={} END={} ENABLE={:#x} PRESC={} CONFIG={:#x}",
        f.events_started,
        f.events_end,
        f.enable,
        f.prescaler,
        f.config,
    );
    rprintln!(
        "sd: EVENTS_DMA rx end/ready/buserr/match0-3: {:x} {:x} {:x} {:x} {:x} {:x} {:x} tx end/ready/buserr: {:x} {:x} {:x}",
        f.dma_rx_end,
        f.dma_rx_ready,
        f.dma_rx_buserror,
        f.dma_rx_match[0],
        f.dma_rx_match[1],
        f.dma_rx_match[2],
        f.dma_rx_match[3],
        f.dma_tx_end,
        f.dma_tx_ready,
        f.dma_tx_buserror,
    );
    rprintln!(
        "sd: RX buserr @{:#010x} TX buserr @{:#010x}",
        f.rx_buserror_address,
        f.tx_buserror_address,
    );
}

fn send(bus: &mut SpimSoftCs<'static>, tx: &[u8]) {
    xfer(bus, tx, &mut []);
}

fn recv(bus: &mut SpimSoftCs<'static>, rx: &mut [u8]) {
    xfer(bus, &[], rx); // ORC=0xFF keeps MOSI high
}

fn recv1(bus: &mut SpimSoftCs<'static>) -> u8 {
    let mut b = [0u8; 1];
    recv(bus, &mut b);
    b[0]
}

/// Send a command frame, return the R1 response (poll up to 16 bytes).
fn command(bus: &mut SpimSoftCs<'static>, cmd: u8, arg: u32, crc: u8) -> u8 {
    let frame = [
        0x40 | cmd,
        (arg >> 24) as u8,
        (arg >> 16) as u8,
        (arg >> 8) as u8,
        arg as u8,
        crc,
    ];
    send(bus, &frame);
    for _ in 0..16 {
        let r = recv1(bus);
        if r & 0x80 == 0 {
            return r;
        }
    }
    0xFF
}

/// Byte-addressed cards multiply the LBA by the block size.
fn card_addr(c: &SdCard, lba: u32) -> u32 {
    if c.high_capacity {
        lba
    } else {
        lba * BLOCK as u32
    }
}

/// CMD0 with retries; returns the last R1 (0xFF = total silence). Leaves
/// CS low on success, high on failure. The first attempt's 16 poll bytes
/// are printed afterwards -- the SoC's actual received data, the one
/// quantity no external instrument has to be trusted for.
fn cmd0_probe(bus: &mut SpimSoftCs<'static>) -> u8 {
    use rtt_target::rprint;
    let mut trace = [0u8; 16];
    let mut r = 0xFF;
    for attempt in 0..8 {
        bus.cs_assert();
        recv1(bus); // 8 clocks with CS low before the frame
        if attempt == 0 {
            send(bus, &[0x40, 0, 0, 0, 0, 0x95]);
            r = 0xFF;
            for t in trace.iter_mut() {
                *t = recv1(bus);
                if r == 0xFF && *t & 0x80 == 0 {
                    r = *t;
                }
            }
        } else {
            r = command(bus, 0, 0, 0x95);
        }
        if r == 0x01 {
            break;
        }
        bus.cs_release();
        recv1(bus); // 8 deselected clocks between attempts
    }
    rprint!("sd: CMD0 poll bytes:");
    for t in trace {
        rprint!(" {:02X}", t);
    }
    rprintln!(" (r={:02X})", r);
    r
}

/// Bring up the card (bit-banged SPI-mode entry + v2 negotiation), then hand
/// the pins to SPIM00 at 32 MHz. Returns 0, or a negative stage-tagged error
/// (-2xx = stage xx, -460 = data wires crossed).
pub fn init() -> i32 {
    let Some(c) = card() else {
        return -480; // the bus singletons are gone
    };
    let bus = &mut c.bus;
    // Standard drive for the init phase: extra-high edges ring hard on
    // jumper wiring, and a ring on SCK re-crossing the card's threshold
    // is a phantom clock -- the C3 line monitor showed the card receiving
    // a bit-perfect CMD0 (3342/3392 expected SCK edges, 81/80 MOSI, 16/16
    // CS) and staying mute; a softer driver (ESP32-C3) talked to the same
    // card at the same speed without issue.
    bus.swap_data_pins(false);
    bus.enter_bitbang();

    // >= 74 clocks with CS high puts the card in SPI-command mode; send
    // 160 (some cards want extra right after power-up).
    bus.cs_release();
    let mut warmup = [0u8; 20];
    recv(bus, &mut warmup);

    // CMD0: software reset -> idle state. Retried: real cards commonly
    // ignore the first attempt(s) after power-up.
    let r = cmd0_probe(bus);
    if r == 0x00 {
        // 0x00 in the response slot is either a live card answering out
        // of alignment / out of idle, or MISO stuck low. A real card
        // returns to 0xFF idle after its response; a stuck line reads
        // 0x00 forever.
        let mut post = [0u8; 4];
        recv(bus, &mut post);
        bus.cs_release();
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
        bus.swap_data_pins(true);
        let r_swapped = cmd0_probe(bus);
        bus.swap_data_pins(false);
        bus.cs_release();
        if r_swapped != 0xFF {
            rprintln!("sd: the card answers ONLY with the data pins swapped:");
            rprintln!("sd: MOSI/MISO wires are CROSSED at the breakout.");
            rprintln!("sd: exchange the two data wires (SPIM needs them straight).");
            return -460;
        }
        return -200 - r as i32;
    }
    if r != 0x01 {
        bus.cs_release();
        return -200 - r as i32;
    }

    // CMD8: v2 check pattern.
    let r = command(bus, 8, 0x1AA, 0x87);
    let v2 = r == 0x01;
    if v2 {
        let mut r7 = [0u8; 4];
        recv(bus, &mut r7);
        if r7[2] & 0x0F != 0x01 || r7[3] != 0xAA {
            bus.cs_release();
            return -210;
        }
    }

    // ACMD41 until the card leaves idle (HCS set for v2).
    let mut ok = false;
    for _ in 0..2500 {
        command(bus, 55, 0, 0xFF);
        let r = command(bus, 41, if v2 { 1 << 30 } else { 0 }, 0xFF);
        if r == 0 {
            ok = true;
            break;
        }
        if r != 0x01 {
            bus.cs_release();
            return -220 - r as i32;
        }
    }
    if !ok {
        bus.cs_release();
        return -230;
    }

    // CMD58: OCR -> block vs byte addressing.
    let r = command(bus, 58, 0, 0xFF);
    if r != 0 {
        bus.cs_release();
        return -240 - r as i32;
    }
    let mut ocr = [0u8; 4];
    recv(bus, &mut ocr);
    c.high_capacity = ocr[0] & 0x40 != 0;
    c.bus.cs_release();
    recv1(&mut c.bus); // 8 clocks after CS release

    // Data phase: hand SCK/MOSI/MISO to the SPIM (CS stays a GPIO).
    c.bus.engage();
    0
}

/// Release all four SD pins to high-impedance inputs (no pulls). Called
/// after a failed init so an external master (the C3 tester) can drive
/// the shared wires while the DK stays powered and attached.
pub fn release_pins() {
    if let Some(c) = card() {
        c.bus.release_pins();
    }
}

/// Wiring diagnostic for a failed init: holds each driven line at
/// DMM-visible static levels (measure at the CARD SOCKET pads, not the
/// header, to test the whole path), exercises MISO's pulls, and runs a
/// MOSI->MISO loopback probe (jumper the two at the breakout, card out,
/// to prove the full digital path both ways). Leaves the pins in the
/// bit-banged idle state.
pub fn diag(cycles: u32) {
    const SEC: u32 = 128_000_000; // 1 s of DWT cycles at 128 MHz
    let Some(c) = card() else {
        return;
    };
    let bus = &mut c.bus;
    bus.enter_bitbang();
    for cyc in 0..cycles {
        rprintln!("sd diag {}/{} (measure at the card socket pads):", cyc + 1, cycles);
        for (name, line, idle_high) in [
            ("SCK  P2.01", Line::Sck, false),
            ("MOSI P2.02", Line::Mosi, true),
            ("CS   P2.05", Line::Cs, true),
        ] {
            rprintln!("  {} LOW for 3 s...", name);
            bus.drive_line(line, false);
            dwt_delay(3 * SEC);
            rprintln!("  {} HIGH for 3 s...", name);
            bus.drive_line(line, true);
            dwt_delay(3 * SEC);
            if !idle_high {
                bus.drive_line(line, false);
            }
        }
        rprintln!("  MISO P2.04 pull-DOWN for 3 s (a breakout pull-up may hold");
        rprintln!("  the node mid-rail; the read below shows the SoC's view)...");
        bus.miso_pull(false);
        dwt_delay(3 * SEC);
        let down = bus.miso_level() as u32;
        bus.miso_pull(true);
        dwt_delay(SEC / 100);
        let up = bus.miso_level() as u32;
        rprintln!("  MISO input reads: pulled-down={} pulled-up={}", down, up);
        let mut ok = 0;
        for &b in &[0xA5u8, 0x3C, 0x0F, 0x81] {
            let got = bus.bitbang_byte(b);
            rprintln!("  loopback sent {:02X} got {:02X}", b, got);
            if got == b {
                ok += 1;
            }
        }
        rprintln!("  MOSI->MISO loopback (needs jumper, card out): {}/4", ok);
    }
}

#[inline]
fn dwt_delay(cycles: u32) {
    let start = cortex_m::peripheral::DWT::cycle_count();
    while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(start) < cycles {}
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
    let rc = match card() {
        Some(c) => read_blocks_inner(c, lba, dst, count),
        None => -470,
    };
    unsafe {
        RD_CYC += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
        RD_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

fn read_blocks_inner(c: &mut SdCard, lba: u32, dst: *mut u8, count: u32) -> i32 {
    if c.bus.fault().is_some() {
        return -470;
    }
    let addr = card_addr(c, lba);
    let bus = &mut c.bus;
    bus.cs_assert();
    let r = command(bus, 18, addr, 0xFF);
    if r != 0 {
        bus.cs_release();
        return -300 - r as i32;
    }
    for i in 0..count {
        // wait for the data token
        let mut token = 0xFFu8;
        for _ in 0..200_000 {
            token = recv1(bus);
            if token != 0xFF {
                break;
            }
        }
        if token != 0xFE {
            bus.cs_release();
            return -310;
        }
        let blk = unsafe {
            core::slice::from_raw_parts_mut(dst.add((i as usize) * BLOCK), BLOCK)
        };
        recv(bus, blk);
        let mut crc = [0u8; 2];
        recv(bus, &mut crc); // CRC not checked (off in SPI mode)
    }
    command(bus, 12, 0, 0xFF); // stop transmission
    // the card holds the line busy (0x00) while finishing
    for _ in 0..200_000 {
        if recv1(bus) == 0xFF {
            bus.cs_release();
            return if bus.fault().is_some() { -470 } else { 0 };
        }
    }
    bus.cs_release();
    -320
}

/// Write `count` 512-byte blocks starting at `lba` from `src` (CMD25).
pub fn write_blocks(lba: u32, src: *const u8, count: u32) -> i32 {
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let rc = match card() {
        Some(c) => write_blocks_inner(c, lba, src, count),
        None => -471,
    };
    unsafe {
        WR_CYC += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
        WR_BYTES += count as u64 * BLOCK as u64;
    }
    rc
}

fn write_blocks_inner(c: &mut SdCard, lba: u32, src: *const u8, count: u32) -> i32 {
    if c.bus.fault().is_some() {
        return -471;
    }
    let addr = card_addr(c, lba);
    let bus = &mut c.bus;
    bus.cs_assert();
    let r = command(bus, 25, addr, 0xFF);
    if r != 0 {
        bus.cs_release();
        return -400 - r as i32;
    }
    for i in 0..count {
        send(bus, &[0xFF, 0xFC]); // gap + multi-block data token
        let blk = unsafe {
            core::slice::from_raw_parts(src.add((i as usize) * BLOCK), BLOCK)
        };
        send(bus, blk);
        send(bus, &[0xFF, 0xFF]); // dummy CRC
        let resp = recv1(bus);
        if resp & 0x1F != 0x05 {
            bus.cs_release();
            return -410 - (resp & 0x1F) as i32;
        }
        let mut busy = false;
        for _ in 0..500_000 {
            if recv1(bus) == 0xFF {
                busy = true;
                break;
            }
        }
        if !busy {
            bus.cs_release();
            return -420;
        }
    }
    send(bus, &[0xFD]); // stop tran token
    recv1(bus);
    for _ in 0..500_000 {
        if recv1(bus) == 0xFF {
            bus.cs_release();
            return if bus.fault().is_some() { -471 } else { 0 };
        }
    }
    bus.cs_release();
    -430
}
