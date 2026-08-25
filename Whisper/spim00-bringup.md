# SPIM00 SD bring-up: every step, sourced

Zero-assumption audit of everything the firmware does (or must do) to run
an SD card on the nRF54LM20B, bit-banged init on the P2 pins plus the
SPIM00 32 MHz data phase. Every step lists its source and its verification
status. Assumptions with no primary source are flagged.

Sources:

- DS    nRF54LM20A/B Datasheet v1.0 (datasheets/), section given
- PAC   nrf-pac 0.4.0, src/chips/nrf54lm20a-app/pac.rs (SVD-generated,
        the authoritative register map)
- ERR   nRF54LM20B Engineering B Errata v1.1 (datasheets/)
- SCH   PCA10184 DK schematic 0.7.0 (hardware files), sheet given
- UG    nRF54LM20 DK HW User Guide v0.7.0
- SD    SD Physical Layer Simplified Specification (SPI mode chapters)
- BENCH measured on this hardware during bring-up (instrument given)

Status legend: [OK] source + hardware-verified, [SRC] sourced but not yet
exercised on hardware, [ASSUME] no primary source found -- flagged.

## 1. System bring-up (main.rs)

| # | Step | Register/value | Source | Status |
|---|---|---|---|---|
| 1.1 | CPU boots at 64 MHz; switch to 128 MHz at start of main, before any HF peripheral | OSCILLATORS.PLL.FREQ (0x5012_0800) = 1 (CK128M); poll CURRENTFREQ (0x5012_0804) | DS 5.5.3 ("The device starts at 64 MHz"); PAC oscillators::Pll, vals::Freq (Ck128m=0x01, Ck64m=0x03) | [OK] grace window halved 48 s -> 24 s on hardware |
| 1.2 | Instruction cache is off at reset; enable it | ICACHE (0xE008_2000): TASKS_INVALIDATECACHE (+0x08)=1, ENABLE (+0x404)=1, then ISB | PAC cache::Cache (register offsets); no DS statement of the reset state found -- measured | [OK] loop pacing improved 24 s -> 8.9 s; ~16 -> ~6 cycles per 2-instruction iteration |
| 1.3 | DWT cycle counter for exact bit-bang timing | DCB trace enable + DWT.CYCCNT via cortex-m crate | ARMv8-M standard; cortex-m crate | [OK] SCK measured 222 kHz on scope, matching 2 us halves + loop overhead |

## 2. GPIO pad configuration, bit-bang init phase (sd.rs init)

| # | Step | Register/value | Source | Status |
|---|---|---|---|---|
| 2.1 | P2 port base and register map | P2_S = 0x5005_0400; OUT 0x00, OUTSET 0x04, OUTCLR 0x08, IN 0x0C, DIR 0x10, DIRSET 0x14, PIN_CNF[n] 0x80+4n | PAC (P2_S, gpio::Gpio impl) | [OK] pins demonstrably drive and read (scope, DMM, loopback) |
| 2.2 | PIN_CNF field layout | DIR[0], INPUT[1], PULL[3:2], DRIVE0[9:8], DRIVE1[11:10], SENSE[17:16], CTRLSEL[30:28] | PAC regs::PinCnf bit ops; DS 8.9 PIN_CNF description | [OK] |
| 2.3 | CTRLSEL must be 0 (GPIO/PSEL control) | Full-register PIN_CNF writes set CTRLSEL=0 implicitly | DS 8.9 ("GPIO or peripherals with PSEL registers" = 0x0); PAC | [OK] pins respond to GPIO writes |
| 2.4 | Outputs SCK/MOSI/CS: standard drive for init | PIN_CNF = 0x3 (DIR=out, INPUT=disconnect, S0S1 drive) | DS 8.9 drive table (S0=0/S1=0 default) | [OK] levels 0.00-0.03 V low / 3.27 V high at breakout (DMM) |
| 2.5 | MISO input with pull-up | PIN_CNF = 0xC (input connected, PULL=pull-up) | DS 8.9 (Pullup=3 at [3:2]) | [OK] jumpered loopback echoes 4/4 byte-exact |
| 2.6 | E0E1 extra-high drive NOT used during init | -- | DS 8.9.6 requires E0E1+HSBIAS only for fast switching; standard drive suffices at 250 kHz | [OK] scope shows clean square at the card |
| 2.7 | Pin idle states before card init | SCK low (OUTCLR), MOSI+CS high (OUTSET), then DIRSET | SD 6.4 (card expects CS and DI high during power-on clocks) | [OK] scope/DMM |
| 2.8 | Reset state of P2.00-05 pads before our writes | -- | ASSUME: no DS statement found for QSPI-capable pins' PIN_CNF/CTRLSEL reset values. Mitigated: we overwrite the whole PIN_CNF, forcing every field | [ASSUME, mitigated] |

## 3. SD SPI-mode initialization (bit-banged, sd.rs)

| # | Step | Value | Source | Status |
|---|---|---|---|---|
| 3.1 | Init clock 100-400 kHz | 250 kHz (DWT-timed 2 us half-periods) | SD 6.4.1 (identification frequency range) | [OK] 222 kHz measured incl. overhead |
| 3.2 | >= 74 clocks, CS and DI high, after power-up | 160 clocks (20 x 0xFF) | SD 6.4.1.1 | [OK] on C3 (identical code passes); scope-visible on DK |
| 3.3 | CMD0 with CS low + valid CRC enters SPI mode | frame 40 00 00 00 00 95, retried 8x with deselected gaps | SD 7.2.1 (SPI mode entry), CRC7 for CMD0 = 0x95 | [OK] on C3: R1=01 first try. DK: card never answers (poll bytes all FF) -- see section 7 |
| 3.4 | R1 within NCR <= 8 bytes | poll 16 bytes | SD 7.5.1 (NCR) | [OK] on C3 (R1 in byte 2) |
| 3.5 | CMD8 check pattern (v2), ACMD41 w/ HCS, CMD58 OCR, byte-vs-block addressing | standard sequence | SD 7.2.1 flowchart | [OK] on C3 end-to-end incl. CMD17 block-0 read |
| 3.6 | Data-pin swap probe (diagnostic) | retry CMD0 with MOSI/MISO roles exchanged; -460 if only swapped answers | own diagnostic; SD unaffected | [OK] both mappings delivered (monitor: 80+80 edges), both silent |

## 4. SPIM00 data phase (sd.rs, after successful init -- NOT YET REACHED)

| # | Step | Register/value | Source | Status |
|---|---|---|---|---|
| 4.1 | SPIM00 base + shared register map | 0x5004_D000; EVENTS_STARTED 0x100, EVENTS_END 0x108, ENABLE 0x500, PRESCALER 0x52C, CONFIG 0x554, IFTIMING.CSNDUR 0x5B0, ORC 0x5C0, PSEL.SCK/MOSI/MISO/CSN 0x600/604/608/610, DMA RX 0x704/708, TX 0x73C/740 | PAC (SPIM00_S, spim impl) | [SRC] |
| 4.2 | 128 MHz core clock, prescaler 4..126 -> 32 MHz | PRESCALER = 4 | DS 8.19 instances table ("Peripheral core frequency is 128 MHz. Prescaler divisor range is 4..126") | [SRC] |
| 4.3 | Init below 400 kHz impossible on SPIM00 (min ~1.02 MHz) -> bit-bang the init phase | -- | DS 8.19 instances table (divisor cap 126) | [OK] design consequence |
| 4.4 | ENABLE value for SPIM function | ENABLE = 7 | PAC spim vals (Enabled = 0x07) | [SRC] |
| 4.5 | Dedicated pins: SPIM00 cannot use other ports | SCK=P2.01, MOSI=P2.02, MISO=P2.04, CSN=P2.05 (CS kept as GPIO) | DS 8.9.5 (clock pins rule), DS ch 10 pin assignments (HSSPI.x functions); DS 8.19 instances ("Use GPIO port P2" for 00-series vs P1/P3 for 2x-series) | [SRC] |
| 4.6 | E0E1 drive + max pad slew for 32 MHz | PIN_CNF DRIVE0=E0, DRIVE1=E1 on SCK/MOSI; GPIOHSPADCTRL.BIAS.HSBIAS=3 (0x5005_0430) | DS 8.9.6 (E0E1 + "recommended ... highest slew"); PAC (GPIOHSPADCTRL overlays P2 base, BIAS +0x30) | [SRC] |
| 4.7 | Erratum [8]: MOSI corruption at CPHA=0, PRESCALER>2 | CSNDUR = PRESCALER/2+1; write 0x82 to +0xC84 before START, 0x00 after STARTED | ERR 3.2 [8] (applies to Eng B and Rev 1) | [SRC] |
| 4.8 | Erratum [69]: STOPPED may not fire with RXDELAY>0 | not in scope: driver never uses STOP/STOPPED | ERR 3.15 | [OK] by design |
| 4.9 | PSEL encoding | PIN[4:0], PORT[6:5], CONNECT[31] | PAC; hardware-validated by the PDM driver on this chip family | [SRC] |

## 5. Board-level path (DK PCA10184)

| # | Step | Detail | Source | Status |
|---|---|---|---|---|
| 5.1 | P2.00-05 route SoC -> TMUX1574 pair (U15/U16, D=SoC, S_A=headers, S_B=NOR QSPI) | SEL shared, 100k pull-up via SB5, driven by board controller; EN active | SCH sheet 7 | [OK] loopback passes through both data channels |
| 5.2 | Mux supply VDD:SW | dedicated rail; default assembly powers it normally (R42/R44) | SCH sheet 4 note | [OK] 3.3 V measured at C61/C62 |
| 5.3 | Board Configurator "Enable external flash" OFF routes P2.00-05 to headers (P4, mirrored on P17 21-26) | board controller drives SEL | UG 2.4 + Board Configurator tooltip; SCH sheet 7 | [OK] all four lines conduct to headers |
| 5.4 | Mux bypass: SoC-side test points | TP13=P2.00, TP29=P2.01, TP30=P2.02, TP31=P2.03, TP32=P2.04, TP33=P2.05 | SCH sheet 7 (GPIO_P2 harness) | [OK] wired; behavior unchanged (still no card response) |
| 5.5 | VDD:IO is a voltage-follower buffer; do not power the card from it | card powered from external 3.3 V (ESP32-C3 3V3) | UG 2.2 ("voltage follower ... leakage currents") | [OK] 3.28 V at breakout under load |

## 6. Verification matrix (what has been PROVEN, and by what)

| Property | Instrument | Result |
|---|---|---|
| CPU 128 MHz + icache | RTT timestamps | OK |
| SCK at card: 3.3 V square, 222 kHz | Tek MSO2024 | OK |
| MOSI/CS driven lows at breakout | DMM (0.03 / 0.00 V) | OK |
| All highs | DMM (3.27 V) | OK |
| Card VCC / grounds | DMM (3.28 V, 0 V offset) | OK |
| SoC transmits AND reads through full path | jumpered loopback, byte-exact 4/4 | OK |
| Frames delivered on both data-pin mappings | C3 passive monitor (edge counts match theory exactly) | OK |
| SoC's received poll bytes | firmware CMD0 trace | all FF -- card silent |
| Same card + breakout + protocol from ESP32-C3 | C3 tester | PASS 3/3 incl. block-0 read |
| Mux bypass (SoC pads direct via TPs) | tape test | still no response |

## 7. RESOLVED: the root cause (2026-08-25, commit 046de3c)

The firmware's CS polarity was inverted at every call site: the helper
was `cs(low: bool)` (true = drive low) but the blind-built M5 call sites
passed `cs(false)` meaning "assert". Warmup clocks ran with the card
SELECTED; every command frame went out DESELECTED -- ignored by every
card, by design. The C3 sniffer's decode showed the inverted CS phase
(warmup inside CS-low, frames inside CS-high) from its first capture;
it was misattributed to wiring three times because the code was assumed
to match its comments. Every other instrument (DMM via diag, loopback,
scope single-line shots) exercised paths that bypass the cs() helper.
Fixed by replacing the helper with cs_assert()/cs_release().

Post-mortem lessons, for the record:
- A boolean parameter whose sense can be misread at call sites
  (`cs(true)` = "low"?) is a defect class of its own; use named
  functions for polarity.
- "Verified" claims age: every DC/loopback verification predated some
  later rewiring and was silently trusted past its expiry.
- When one instrument (the protocol-context sniffer) repeatedly
  contradicts the mental model while spot-check instruments agree with
  it, believe the protocol-context instrument: the spot checks were
  measuring different code paths.

## 8. Superseded: the earlier open contradiction

A card that answers a byte-identical, speed-identical CMD0 sequence from
an ESP32-C3 ignores the same sequence delivered from the nRF54LM20B's own
pads, with every measurable signal property verified equal. No documented
software step is missing per sections 1-4. Remaining actions, in order:

1. **Second card on the clean bench.** Both cards failed in eras with
   real, since-fixed faults (unpowered card, 3.8 V rail, taps, muxes
   unbypassed). Only the SDSC card has been tried on the fully verified
   setup. (BENCH)
2. **Same-hour C3 control.** Re-run the C3 tester on the current physical
   harness (DK wire ends lifted) to re-prove the shared segment TODAY,
   then re-attach and re-run the DK within minutes. (BENCH)
3. **Sniffer byte-decode at the test points** during a DK attempt
   (tools/esp32c3-sd-test, `--bin sniffer`): decodes the exact DI
   bitstream + SCK half-period min/avg/max at 100 ns resolution. If the
   decoded frame is 40 00 00 00 00 95 with clean 2 us cells and the card
   still ignores it, capture the identical decode during a C3 PASS and
   diff the two -- any difference (inter-byte gaps, CS timing, half-period
   jitter) is then the answer by construction. (BENCH)
4. **Nordic DevZone case.** If 1-3 leave the contradiction standing, this
   document plus the two decode captures is a complete, reproducible case
   for Nordic support: nRF54LM20B (Engineering B / PAAA-Bxx) GPIO
   bit-banged SPI, SD cards do not respond, identical sequence from
   another MCU works, all listed errata applied. An unlisted silicon
   erratum on the P2 QSPI-capable pads is at that point a live
   possibility that only Nordic can confirm.

## 9. Explicitly retired theories (for the record)

Wiring swaps (disproven: card-witness swap probe), nRF cannot read MISO
(disproven: 4/4 loopback), analog switch misrouting or damage (disproven:
DC + loopback through both channels; then bypassed entirely), VDD:SW
starvation (3.3 V measured), 3.8 V card overvoltage (fixed; persisted
after), unpowered card during ESP-off runs (fixed; persisted after),
drive strength / ringing (soft drive + series R + short wires; no
change), erratum [8] (applies to the SPIM phase, not bit-bang; workaround
in place regardless), ground loop (strap + 0 V offset verified), CPU
clock / icache timing distortions (fixed; bit-bang is DWT-timed and
scope-verified).
