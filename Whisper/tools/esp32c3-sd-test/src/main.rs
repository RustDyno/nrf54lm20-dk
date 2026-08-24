//! Standalone SD breakout tester for an ESP32-C3 board.
//!
//! Purpose: prove (or disprove) that the microSD breakout + card work at
//! all, independent of the nRF54 DK. Bit-bangs the same SPI-mode init
//! sequence as the Whisper firmware (CMD0 retries, CMD8, ACMD41, CMD58)
//! at 250 kHz, then reads block 0 and checks the Whisper image magic.
//! Verbose: every stage prints its raw response bytes.
//!
//! Wiring (3.3 V breakout, no level shifter):
//!   SCK  -> GPIO4    MISO -> GPIO5    MOSI -> GPIO6    CS -> GPIO7
//!   VCC  -> 3V3      GND  -> GND
//!
//! Build and run (board on USB):
//!   cargo run --release
//! Output goes over the native USB (USB-Serial-JTAG). On boards that
//! only expose a USB-UART bridge, switch esp-println's feature to
//! "uart" in Cargo.toml.
//!
//! The test loops forever, one full attempt every 3 s, so it can be
//! probed at leisure.

#![no_std]
#![no_main]

use esp_hal::delay::Delay;
use esp_hal::gpio::{Input, InputConfig, Level, Output, OutputConfig, Pull};
use esp_hal::main;
use esp_println::{print, println};

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic: {}", info);
    loop {}
}

struct Bus<'d> {
    sck: Output<'d>,
    mosi: Output<'d>,
    cs: Output<'d>,
    miso: Input<'d>,
    delay: Delay,
}

impl Bus<'_> {
    /// SPI mode 0 at ~250 kHz: MOSI changes on the falling edge, both
    /// sides sample on the rising edge.
    fn xfer(&mut self, tx: u8) -> u8 {
        let mut rx = 0u8;
        for bit in (0..8).rev() {
            if tx & (1 << bit) != 0 {
                self.mosi.set_high();
            } else {
                self.mosi.set_low();
            }
            self.delay.delay_micros(2);
            self.sck.set_high();
            if self.miso.is_high() {
                rx |= 1 << bit;
            }
            self.delay.delay_micros(2);
            self.sck.set_low();
        }
        rx
    }

    /// Command frame, then poll up to 16 bytes for R1 (bit 7 clear).
    /// Prints the poll trail when `verbose`.
    fn cmd(&mut self, cmd: u8, arg: u32, crc: u8, verbose: bool) -> u8 {
        for b in [
            0x40 | cmd,
            (arg >> 24) as u8,
            (arg >> 16) as u8,
            (arg >> 8) as u8,
            arg as u8,
            crc,
        ] {
            self.xfer(b);
        }
        let mut r1 = 0xFF;
        if verbose {
            print!("    CMD{} poll:", cmd);
        }
        for _ in 0..16 {
            let b = self.xfer(0xFF);
            if verbose {
                print!(" {:02X}", b);
            }
            if b & 0x80 == 0 {
                r1 = b;
                break;
            }
        }
        if verbose {
            println!(" -> R1={:02X}", r1);
        }
        r1
    }
}

fn attempt(bus: &mut Bus<'_>, verbose: bool) -> bool {
    // 160 warmup clocks, CS high.
    bus.cs.set_high();
    for _ in 0..20 {
        bus.xfer(0xFF);
    }

    // CMD0 with retries, CS low.
    let mut r = 0xFF;
    for i in 0..8 {
        bus.cs.set_low();
        bus.xfer(0xFF);
        r = bus.cmd(0, 0, 0x95, verbose && i == 0);
        if r == 0x01 {
            println!("  CMD0 ok on attempt {} (R1=01, idle)", i + 1);
            break;
        }
        bus.cs.set_high();
        bus.xfer(0xFF);
    }
    if r != 0x01 {
        println!("  CMD0 FAILED after 8 attempts (last R1={:02X})", r);
        println!("  -> card never responded: breakout/socket/card problem");
        bus.cs.set_high();
        return false;
    }

    // CMD8: v2 check pattern.
    let r = bus.cmd(8, 0x1AA, 0x87, verbose);
    let v2 = r == 0x01;
    if v2 {
        let echo: [u8; 4] = core::array::from_fn(|_| bus.xfer(0xFF));
        println!(
            "  CMD8 R1=01, R7={:02X} {:02X} {:02X} {:02X} (expect .. .. 01 AA)",
            echo[0], echo[1], echo[2], echo[3]
        );
    } else {
        println!("  CMD8 R1={:02X} (v1 card)", r);
    }

    // ACMD41 until out of idle.
    let mut tries = 0u32;
    loop {
        bus.cmd(55, 0, 0xFF, false);
        let r = bus.cmd(41, if v2 { 1 << 30 } else { 0 }, 0xFF, false);
        tries += 1;
        if r == 0x00 {
            println!("  ACMD41 ready after {} tries", tries);
            break;
        }
        if r != 0x01 || tries > 2000 {
            println!("  ACMD41 FAILED (R1={:02X} after {} tries)", r, tries);
            bus.cs.set_high();
            return false;
        }
    }

    // CMD58: OCR.
    let r = bus.cmd(58, 0, 0xFF, false);
    let ocr: [u8; 4] = core::array::from_fn(|_| bus.xfer(0xFF));
    let hc = ocr[0] & 0x40 != 0;
    println!(
        "  CMD58 R1={:02X}, OCR={:02X} {:02X} {:02X} {:02X} ({})",
        r,
        ocr[0],
        ocr[1],
        ocr[2],
        ocr[3],
        if hc { "SDHC/SDXC, block addressed" } else { "SDSC, byte addressed" }
    );

    // CMD17: read block 0, look for the Whisper image magic.
    let addr = if hc { 0 } else { 0 };
    let r = bus.cmd(17, addr, 0xFF, false);
    if r != 0 {
        println!("  CMD17 FAILED (R1={:02X})", r);
        bus.cs.set_high();
        return false;
    }
    let mut token = 0xFF;
    for _ in 0..100_000u32 {
        token = bus.xfer(0xFF);
        if token != 0xFF {
            break;
        }
    }
    if token != 0xFE {
        println!("  CMD17: no data token (got {:02X})", token);
        bus.cs.set_high();
        return false;
    }
    let mut block = [0u8; 512];
    for b in block.iter_mut() {
        *b = bus.xfer(0xFF);
    }
    bus.xfer(0xFF); // CRC
    bus.xfer(0xFF);
    bus.cs.set_high();
    bus.xfer(0xFF);

    print!("  block 0:");
    for b in &block[..16] {
        print!(" {:02X}", b);
    }
    println!();
    if &block[..8] == b"WSPRIMG1" {
        println!("  Whisper image magic FOUND");
    } else {
        println!("  (no Whisper image magic; card readable but not the image)");
    }
    true
}

#[main]
fn main() -> ! {
    let p = esp_hal::init(esp_hal::Config::default());
    let mut bus = Bus {
        sck: Output::new(p.GPIO4, Level::Low, OutputConfig::default()),
        mosi: Output::new(p.GPIO6, Level::High, OutputConfig::default()),
        cs: Output::new(p.GPIO7, Level::High, OutputConfig::default()),
        miso: Input::new(p.GPIO5, InputConfig::default().with_pull(Pull::Up)),
        delay: Delay::new(),
    };
    println!();
    println!("esp32c3 SD breakout tester");
    println!("wiring: SCK=GPIO4 MISO=GPIO5 MOSI=GPIO6 CS=GPIO7, 3V3+GND");
    let mut n = 0u32;
    loop {
        n += 1;
        println!("--- attempt {} ---", n);
        let ok = attempt(&mut bus, n == 1);
        println!("--- attempt {}: {} ---", n, if ok { "PASS" } else { "FAIL" });
        bus.delay.delay_millis(3000);
    }
}
