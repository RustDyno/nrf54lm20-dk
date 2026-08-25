//! Passive line monitor: the ESP32-C3 as a poor man's logic analyzer.
//!
//! Wire GPIO4/5/6/7 in PARALLEL with the SD breakout's SCK/MISO/MOSI/CS
//! (same assignment as the tester) while the nRF54 DK drives the card,
//! with ALL grounds common (DK, C3, breakout). The C3 only listens
//! (inputs, no pulls) and prints per-second edge counts and levels, so a
//! DK init burst shows up as thousands of SCK edges, MOSI activity, and
//! CS low time. A line that stays flat at the card while the DK claims
//! to drive it is the broken path.
//!
//!   cargo run --release --bin monitor
//!
//! Expected during one DK init attempt (~130 ms, 250 kHz): SCK ~65000
//! edges, MOSI hundreds, CS a couple of low/high transitions with real
//! low dwell, MISO flat high if the card stays silent.

#![no_std]
#![no_main]

use esp_hal::gpio::{Input, InputConfig, Pull};
use esp_hal::main;
use esp_hal::time::{Duration, Instant};
use esp_println::println;

esp_bootloader_esp_idf::esp_app_desc!();

#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    println!("panic: {}", info);
    loop {}
}

#[main]
fn main() -> ! {
    let p = esp_hal::init(esp_hal::Config::default());
    let cfg = InputConfig::default().with_pull(Pull::None);
    let pins = [
        ("SCK ", Input::new(p.GPIO4, cfg)),
        ("MISO", Input::new(p.GPIO5, cfg)),
        ("MOSI", Input::new(p.GPIO6, cfg)),
        ("CS  ", Input::new(p.GPIO7, cfg)),
    ];
    println!();
    println!("passive SD line monitor: GPIO4=SCK 5=MISO 6=MOSI 7=CS");
    println!("(all grounds common; C3 drives nothing)");
    loop {
        let mut edges = [0u32; 4];
        let mut lows = [0u32; 4];
        let mut samples = 0u32;
        let mut prev = [false; 4];
        for (i, (_, pin)) in pins.iter().enumerate() {
            prev[i] = pin.is_high();
        }
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(1) {
            samples += 1;
            for (i, (_, pin)) in pins.iter().enumerate() {
                let level = pin.is_high();
                if level != prev[i] {
                    edges[i] += 1;
                    prev[i] = level;
                }
                if !level {
                    lows[i] += 1;
                }
            }
        }
        let mut any = false;
        for e in edges {
            if e > 0 {
                any = true;
            }
        }
        if any {
            println!("activity ({} samples/s):", samples);
            for (i, (name, _)) in pins.iter().enumerate() {
                println!(
                    "  {}: {:6} edges, low {:3}% of samples",
                    name,
                    edges[i],
                    lows[i] / (samples / 100).max(1)
                );
            }
        } else {
            println!(
                "quiet ({} samples/s; levels SCK={} MISO={} MOSI={} CS={})",
                samples,
                prev[0] as u8,
                prev[1] as u8,
                prev[2] as u8,
                prev[3] as u8
            );
        }
    }
}
