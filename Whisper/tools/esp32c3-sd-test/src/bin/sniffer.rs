//! SPI sniffer: timestamped edge capture + protocol decode at ~100 ns
//! resolution, using raw GPIO_IN register reads and the RISC-V cycle
//! counter. Wired like the monitor (parallel taps, C3 drives nothing):
//!   GPIO4 = SCK    GPIO5 = MISO/DO    GPIO6 = MOSI/DI    GPIO7 = CS
//!
//! Arms on CS falling edge, records every line transition with a cycle
//! timestamp until CS stays high for ~1 s or the buffer fills, then:
//!   - per-line edge counts
//!   - SCK half-period min/avg/max (catches runts and stretched pulses)
//!   - the decoded DI byte stream (sampled on SCK rising while CS low),
//!     which is exactly what the card's command input sees
//!   - the decoded DO stream (any card response)
//! Re-arms afterwards; loops forever.
//!
//!   cargo run --release --bin sniffer

#![no_std]
#![no_main]

use esp_hal::gpio::{Input, InputConfig, Pull};
use esp_hal::main;
use esp_println::{print, println};

esp_bootloader_esp_idf::esp_app_desc!();

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic: {}", info);
    loop {}
}

// ESP32-C3 GPIO input register: all pins in one read.
const GPIO_IN_REG: *const u32 = (0x6000_4000 + 0x003C) as *const u32;
const B_SCK: u32 = 1 << 4;
const B_MISO: u32 = 1 << 5;
const B_MOSI: u32 = 1 << 6;
const B_CS: u32 = 1 << 7;
const MASK: u32 = B_SCK | B_MISO | B_MOSI | B_CS;

const CAP: usize = 16384;
static mut TS: [u32; CAP] = [0; CAP];
static mut VAL: [u32; CAP] = [0; CAP];

/// ESP32-C3 has no standard mcycle; use Espressif's machine performance
/// counter CSRs: PCER (0x7E0) event=cycles, PCMR (0x7E1) enable, PCCR
/// (0x7E2) the running count.
fn cycles_init() {
    unsafe {
        core::arch::asm!("csrwi 0x7E0, 1", "csrwi 0x7E1, 1");
    }
}

#[inline(always)]
fn cycles() -> u32 {
    let c: u32;
    unsafe { core::arch::asm!("csrr {}, 0x7E2", out(reg) c) };
    c
}

#[inline(always)]
fn sample() -> u32 {
    unsafe { core::ptr::read_volatile(GPIO_IN_REG) & MASK }
}

#[main]
fn main() -> ! {
    let p = esp_hal::init(esp_hal::Config::default());
    // Keep the pins alive as inputs; sampling bypasses the HAL for speed.
    let cfg = InputConfig::default().with_pull(Pull::None);
    let _keep = (
        Input::new(p.GPIO4, cfg),
        Input::new(p.GPIO5, cfg),
        Input::new(p.GPIO6, cfg),
        Input::new(p.GPIO7, cfg),
    );
    let delay = esp_hal::delay::Delay::new();

    // Calibrate the cycle counter against the HAL delay once.
    cycles_init();
    let c0 = cycles();
    delay.delay_millis(100);
    let cyc_per_us = cycles().wrapping_sub(c0) / 100_000;
    println!();
    println!("SPI sniffer armed ({} cycles/us). Waiting for CS low...", cyc_per_us);

    loop {
        // Arm: wait for CS falling edge.
        while sample() & B_CS == 0 {}
        while sample() & B_CS != 0 {}
        let t0 = cycles();
        let mut n = 0usize;
        let mut prev = sample();
        unsafe {
            TS[0] = 0;
            VAL[0] = prev;
        }
        n += 1;
        // ~1 s of idle-high CS ends the capture.
        let idle_limit = cyc_per_us * 1_000_000 / 1;
        let mut cs_high_since: Option<u32> = None;
        loop {
            let v = sample();
            let t = cycles().wrapping_sub(t0);
            if v != prev {
                if n < CAP {
                    unsafe {
                        TS[n] = t;
                        VAL[n] = v;
                    }
                    n += 1;
                } else {
                    break;
                }
                prev = v;
            }
            if v & B_CS != 0 {
                match cs_high_since {
                    None => cs_high_since = Some(t),
                    Some(s) => {
                        if t.wrapping_sub(s) > idle_limit {
                            break;
                        }
                    }
                }
            } else {
                cs_high_since = None;
            }
        }

        // ---- report ----------------------------------------------------
        let ts = unsafe { &TS[..n] };
        let val = unsafe { &VAL[..n] };
        let mut edge_count = [0u32; 4];
        for w in 1..n {
            let ch = val[w] ^ val[w - 1];
            for (i, b) in [B_SCK, B_MISO, B_MOSI, B_CS].iter().enumerate() {
                if ch & b != 0 {
                    edge_count[i] += 1;
                }
            }
        }
        println!();
        println!("capture: {} transitions", n);
        for (i, name) in ["SCK ", "MISO", "MOSI", "CS  "].iter().enumerate() {
            println!("  {}: {} edges", name, edge_count[i]);
        }

        // SCK half-period stats.
        let (mut last, mut min, mut max) = (0u32, u32::MAX, 0u32);
        let mut sum = 0u64;
        let mut cnt = 0u32;
        let mut have_last = false;
        for w in 1..n {
            if (val[w] ^ val[w - 1]) & B_SCK != 0 {
                if have_last {
                    let d = ts[w].wrapping_sub(last);
                    min = min.min(d);
                    max = max.max(d);
                    sum += d as u64;
                    cnt += 1;
                }
                last = ts[w];
                have_last = true;
            }
        }
        if cnt > 0 {
            let c = cyc_per_us.max(1);
            println!(
                "  SCK half-period: min {}.{:02} us, avg {}.{:02} us, max {}.{:02} us ({} halves)",
                min / c, (min % c) * 100 / c,
                (sum as u32 / cnt) / c, ((sum as u32 / cnt) % c) * 100 / c,
                max / c, (max % c) * 100 / c,
                cnt
            );
        }

        // Byte decode on SCK rising edges while CS low.
        for (line, bit) in [("DI (card command in)", B_MOSI), ("DO (card response)", B_MISO)] {
            println!("  {} bytes:", line);
            print!("   ");
            let mut sh = 0u32;
            let mut nb = 0u32;
            let mut printed = 0u32;
            let mut cs_was_high = true;
            for w in 1..n {
                if val[w] & B_CS != 0 {
                    if !cs_was_high {
                        // CS release: byte boundary marker
                        if nb != 0 {
                            print!(" +{}bits", nb);
                            nb = 0;
                            sh = 0;
                        }
                        print!(" |");
                        printed += 1;
                    }
                    cs_was_high = true;
                    continue;
                }
                cs_was_high = false;
                if (val[w] ^ val[w - 1]) & B_SCK != 0 && val[w] & B_SCK != 0 {
                    sh = (sh << 1) | u32::from(val[w] & bit != 0);
                    nb += 1;
                    if nb == 8 {
                        print!(" {:02X}", sh);
                        printed += 1;
                        sh = 0;
                        nb = 0;
                    }
                }
                if printed > 0 && printed % 24 == 0 {
                    println!();
                    print!("   ");
                    printed += 1; // avoid re-triggering on the same count
                }
                if printed > 400 {
                    print!(" ...");
                    break;
                }
            }
            println!();
        }
        println!("re-armed. Waiting for CS low...");
    }
}
