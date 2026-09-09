# embassy-nrf patches

`embassy-nrf-0.11.0-nrf54lm20.patch` is the diff between crates.io
`embassy-nrf 0.11.0` and `../vendor/embassy-nrf`, which the firmware uses
through `[patch.crates-io]`. It applies with `patch -p1` from the root of
an embassy checkout (paths are `a/embassy-nrf/...`).

Each hunk is meant to go upstream as it stands. Regenerate after editing
the vendored copy:

    R=$(ls -d ~/.cargo/registry/src/*/embassy-nrf-0.11.0)
    (cd "$R" && diff -ruN --exclude=Cargo.toml.orig --exclude=.cargo_vcs_info.json \
        --exclude=Cargo.lock . ../../vendor/embassy-nrf) > embassy-nrf-0.11.0-nrf54lm20.patch
    # then rewrite the ---/+++ paths to a/embassy-nrf and b/embassy-nrf

## What it changes, and why

**GPIO port 3 (`gpio.rs`, `Cargo.toml`, chip file).** `SealedPin::block()`
and `port()` matched ports 0-2 and hit `unreachable_unchecked()` for
anything else, while the nRF54LM20 chip table declares `P3_00..P3_12` as
real pins. Handing any P3 pin to a driver was therefore undefined
behavior, not an error: no panic, no fault, the optimizer deleting the
constructor and the code after it (hardware-observed with `Twim` on
P3.02/P3.03). Adds a `_gpio-p3` feature, enables it for `_nrf54lm20`, and
gives `Port` and both lookups their port-3 arm. The `P3` alias joins the
secure and non-secure re-export lists.

**PDM20/PDM21 singletons (chip file).** The LM20 has two PDM instances
and no `peripherals!` entries for them, so no driver could take one. The
`pdm` module itself is not built for `_nrf54l` (the nRF54L block has
PRESCALER/CLKSELECT/RATIO instead of PDMCLKCTRL and a byte-counted
MAXCNT), so this only adds the singletons; the driver lives in
`../src/hal/pdm.rs` and is the natural follow-up PR.

**Wait for the clock speed (`lib.rs`).** `init()` wrote `PLL.FREQ` and
returned. The switch is not instant, so a peripheral set up immediately
afterwards, or any cycle-counted delay, ran against the old frequency:
a 3 s window measured 48 s on hardware when the firmware did this itself
before waiting. Now `init()` polls `PLL.CURRENTFREQ` until it matches.

**Instruction cache (`lib.rs`).** ICACHE is disabled at reset, and with
it off every taken branch refetches from RRAM through fixed wait states
(measured here: about 5x slower than the core clock suggests on
branch-heavy code). Adds `Config::cache: CacheConfig { icache }`,
alongside the existing `dcdc` and `clock_speed` boot configuration.
Default is `false`: enabling it is a behavior change for existing users,
and an application that writes RRAM and then executes it has to
invalidate the cache itself.
