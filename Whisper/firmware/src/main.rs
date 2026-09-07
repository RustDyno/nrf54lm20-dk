//! Whisper layer-streaming executor.
//!
//! The firmware is a dumb step machine: the host (host/ tool) resolves symbol
//! addresses from this ELF, streams per-layer Axon blobs into the SLOT region
//! and activations/parameters into ARENA over SWD, and drives execution
//! through MAILBOX commands. All Whisper structure (the "tape") lives on the
//! host; the device only knows how to run the current slot blob on the NPU
//! and how to execute the CPU glue kernels.
//!
//! Hardware bring-up findings (vector table, power cycling, RAM ceiling) are
//! inherited from the KWS project; see ../../KWS/NOTES.md.

#![no_std]
#![no_main]

use core::sync::atomic::{AtomicU32, Ordering};

use cortex_m_rt::{entry, exception, ExceptionFrame};
use embassy_nrf::pac;
use rtt_target::{rprintln, rtt_init, ChannelMode};

// A silent panic loop cost a debugging session (an out-of-bounds index in
// the LM head presented as "decode stuck"): print the panic over RTT and
// leave a crumb before spinning.
#[panic_handler]
fn panic(info: &core::panic::PanicInfo) -> ! {
    crumb2(0xDEAD);
    rprintln!("PANIC: {}", info);
    loop {}
}

mod app;
mod bindings;
mod display;
mod dsp;
mod kernels;
mod libm_shims;
mod mel;
#[cfg(feature = "mock-usb")]
mod mockblk;
#[allow(dead_code)]
mod pdm;
mod platform;
mod q4;
#[cfg_attr(not(feature = "sd-card"), allow(dead_code))]
mod sd;
mod slot;
mod storage;
mod usb;
#[cfg(feature = "mock-usb")]
mod usbdev;
mod usbproto;

use kernels::Quant;

// Storage backing the C `extern uint32_t nrf_axon_interlayer_buffer[]` and
// `nrf_axon_psum_buffer[]`. Sizes must match the -D defines in build.rs.
pub const INTERLAYER_BUFFER_BYTES: usize = 65536;
const PSUM_BUFFER_BYTES: usize = 4096;

// Pinned in the fixed AXONBUF region (memory.x) so blob binaries -- which
// embed these two addresses -- survive firmware relinks. NOLOAD: zeroed
// explicitly at boot, not by cortex-m-rt.
#[no_mangle]
#[link_section = ".axonbuf.interlayer"]
pub static mut nrf_axon_interlayer_buffer: [u32; INTERLAYER_BUFFER_BYTES / 4] =
    [0; INTERLAYER_BUFFER_BYTES / 4];

#[no_mangle]
#[link_section = ".axonbuf.psum"]
pub static mut nrf_axon_psum_buffer: [u32; PSUM_BUFFER_BYTES / 4] = [0; PSUM_BUFFER_BYTES / 4];

/// Activation / parameter arena at a FIXED address (memory.x ARENA region,
/// 0x20032000) so the tape generator can emit absolute addresses. The tape
/// owns the layout: every mailbox command carries absolute addresses from
/// its allocation plan. Activations are channel-planar [C][W]
/// (hardware-verified Axon layout).
pub const ARENA_BYTES: usize = 100 * 1024;

#[no_mangle]
#[link_section = ".arena"]
pub static mut ARENA: [u8; ARENA_BYTES] = [0; ARENA_BYTES];

// --- Device interrupt vector table (AXONS IRQ 86; see ../../npu/src/main.rs).
//
// Hand-built rather than the PAC's: the AXONS block is absent from the
// public SVD, so the PAC's table (its `rt` feature) has no AXONS handler
// and vectors IRQ 86 to a null slot. The PAC's `rt` feature therefore
// stays off, and this table is the only one in the image.

const AXONS_IRQN: usize = 86;
// Cover every IRQ slot the LM20 has (VREGUSB, the highest, is 289): a
// stray unmasked interrupt then lands in default_irq_handler's bkpt loop
// instead of executing whatever .text follows the table.
const VECTOR_SLOTS: usize = 290;

unsafe extern "C" fn default_irq_handler() {
    loop {
        cortex_m::asm::bkpt();
    }
}

unsafe extern "C" fn axons_irq_handler() {
    bindings::nrf_axon_handle_interrupt();
}

// HAL drivers bind their interrupt handlers as exported symbols named
// after the IRQ (display.rs binds TWIM22's SERIAL22). The handlers never
// fire in blocking use, but the constructors unmask the NVIC lines, so
// the table routes them properly instead of into the bkpt loop.
#[cfg(not(feature = "sd-spim22"))]
extern "C" {
    fn SERIAL22();
}

const fn vector_table() -> [unsafe extern "C" fn(); VECTOR_SLOTS] {
    let mut t = [default_irq_handler as unsafe extern "C" fn(); VECTOR_SLOTS];
    t[AXONS_IRQN] = axons_irq_handler;
    #[cfg(not(feature = "sd-spim22"))]
    {
        t[pac::Interrupt::SERIAL22 as usize] = SERIAL22;
    }
    t
}

#[no_mangle]
#[link_section = ".vector_table.interrupts"]
pub static __INTERRUPTS: [unsafe extern "C" fn(); VECTOR_SLOTS] = vector_table();

// --- Crash breadcrumb ---------------------------------------------------------
// The last two words of the SLOT region (no section is placed there, so
// they survive a reset; no blob or slot-borrowing phase reaches the top
// 8 bytes -- app.rs asserts it). Written at each execution milestone and
// reported at the next boot, so a watchdog reset names its victim. They
// used to sit at the top of the arena, until the decode blob pipeline
// made that part of a DMA landing zone: an NPU-phase crumb written while
// a blob was landing corrupted the transfer.

pub const CRUMB_BASE: usize = slot::SLOT_BASE + slot::SLOT_BYTES - 8;
const BREADCRUMB: *mut u32 = CRUMB_BASE as *mut u32;
/// Second noinit word: the NPU slot phase (0x201 entering infer,
/// 0x202 returned), so it no longer overwrites the pipeline stage.
const BREADCRUMB2: *mut u32 = (CRUMB_BASE + 4) as *mut u32;

pub fn crumb(v: u32) {
    unsafe { core::ptr::write_volatile(BREADCRUMB, v) };
}

pub fn crumb2(v: u32) {
    unsafe { core::ptr::write_volatile(BREADCRUMB2, v) };
}

// --- Hang watchdog (ported from the KWS firmware) ----------------------------
// A debug session killed mid-inference can wedge the Axon engine across soft
// resets; the next driver call then blocks forever. SysTick counts while a
// driver call is in flight; over budget -> chip reset, and the boot-time
// power cycle in platform::init clears the engine. (A wedge inside a
// PRIMASK critical section still needs a board power cycle.)

const WDOG_DISARMED: u32 = u32::MAX;
const WDOG_LIMIT_TICKS: u32 = 200; // 200 x 10 ms = 2 s per driver call
static WDOG_TICKS: AtomicU32 = AtomicU32::new(WDOG_DISARMED);

/// SysTick periods since boot (10 ms each): a wall clock for spans the
/// 33 s DWT cycle counter cannot cover.
static UPTIME_TICKS: AtomicU32 = AtomicU32::new(0);

pub fn uptime_ms() -> u32 {
    UPTIME_TICKS.load(Ordering::Relaxed).wrapping_mul(10)
}

#[exception]
unsafe fn HardFault(ef: &ExceptionFrame) -> ! {
    // Print the exception frame and fault status over RTT, then spin so
    // the attached host can drain the message (no bkpt: keep RTT alive).
    let crumb_val = unsafe { core::ptr::read_volatile(BREADCRUMB) };
    let crumb2_val = unsafe { core::ptr::read_volatile(BREADCRUMB2) };
    rprintln!(
        "HARDFAULT pc={:#010x} lr={:#010x} xpsr={:#010x} crumbs={:#x}/{:#x} frame@{:#010x}",
        ef.pc(),
        ef.lr(),
        ef.xpsr(),
        crumb_val,
        crumb2_val,
        ef as *const ExceptionFrame as usize
    );
    rprintln!(
        "  r0={:#010x} r1={:#010x} r2={:#010x} r3={:#010x} r12={:#010x}",
        ef.r0(),
        ef.r1(),
        ef.r2(),
        ef.r3(),
        ef.r12()
    );
    unsafe {
        rprintln!(
            "  CFSR={:#010x} HFSR={:#010x} MMFAR={:#010x} BFAR={:#010x}",
            core::ptr::read_volatile(0xE000_ED28 as *const u32),
            core::ptr::read_volatile(0xE000_ED2C as *const u32),
            core::ptr::read_volatile(0xE000_ED34 as *const u32),
            core::ptr::read_volatile(0xE000_ED38 as *const u32)
        );
    }
    loop {}
}

#[exception]
fn SysTick() {
    UPTIME_TICKS.fetch_add(1, Ordering::Relaxed);
    let t = WDOG_TICKS.load(Ordering::Relaxed);
    if t != WDOG_DISARMED {
        if t >= WDOG_LIMIT_TICKS {
            // Name the reset in the log: a silent sys_reset here looked like
            // a memory fault for days (the debugger's reset catch reported
            // it as "Exception" at the reset handler).
            let c = unsafe { core::ptr::read_volatile(BREADCRUMB) };
            let c2 = unsafe { core::ptr::read_volatile(BREADCRUMB2) };
            rprintln!("watchdog: driver call stuck >2 s (crumbs {:#x}/{:#x}), resetting", c, c2);
            cortex_m::asm::delay(12_800_000); // ~100 ms for the host to drain RTT
            cortex_m::peripheral::SCB::sys_reset();
        }
        WDOG_TICKS.store(t + 1, Ordering::Relaxed);
    }
}

pub(crate) struct WdogGuard;

impl WdogGuard {
    pub(crate) fn arm() -> Self {
        WDOG_TICKS.store(0, Ordering::Relaxed);
        WdogGuard
    }
}

impl Drop for WdogGuard {
    fn drop(&mut self) {
        WDOG_TICKS.store(WDOG_DISARMED, Ordering::Relaxed);
    }
}

// --- Mailbox protocol ---------------------------------------------------------
// Host: write `args` + `cmd`, then increment `cmd_seq`. Firmware: on
// cmd_seq != ack_seq, execute, write `status`, then set ack_seq = cmd_seq.
// `magic` stages the boot so the host can tell a hang from a failure:
// BOOT (entered main) -> LAYR (Axon init ok, executor running), or FAIL
// with the init rc in `status`.

pub const MAILBOX_MAGIC: u32 = 0x4C41_5952; // "LAYR"
pub const MAILBOX_BOOT: u32 = 0x424F_4F54; // "BOOT"
pub const MAILBOX_FAIL: u32 = 0x4641_494C; // "FAIL"

#[repr(C)]
pub struct Mailbox {
    pub magic: u32,
    pub cmd_seq: u32,
    pub cmd: u32,
    pub args: [u32; 8],
    pub ack_seq: u32,
    pub status: i32,
}

#[no_mangle]
pub static mut MAILBOX: Mailbox = Mailbox {
    magic: 0, // set to MAILBOX_MAGIC once init succeeded
    cmd_seq: 0,
    cmd: 0,
    args: [0; 8],
    ack_seq: 0,
    status: 0,
};

// Command codes (host mirrors these).
const CMD_PING: u32 = 1;
const CMD_RUN_NPU: u32 = 2; // args: input addr|0, output addr|0
const CMD_LUT8: u32 = 3; // args: lut, src, dst, len
const CMD_MATMUL_BT: u32 = 4; // args: a, b, acc, m, k, n, za
const CMD_SOFTMAX: u32 = 6; // args: acc, dst, rows, cols, mult(f32 bits)
const CMD_REQUANT: u32 = 7; // args: acc, dst, len, mult(f32), scale(f32), zp
const CMD_LN: u32 = 8; // args: param block addr (LnParams)
const CMD_ADD16: u32 = 9; // args: param block addr (AddParams)
const CMD_ADDPOS: u32 = 10; // args: param block addr (AddPosParams)
const CMD_LOGITS_MAX: u32 = 11; // args: acc, mults, idx, len, state
const CMD_ATTN_HEAD: u32 = 12; // args: param block addr (AttnParams)
const CMD_FC2SUM: u32 = 13; // args: param block addr (Fc2SumParams)
const CMD_SD_INIT: u32 = 20;
const CMD_SD_READ: u32 = 21; // args: lba, dst addr, block count
const CMD_SD_WRITE: u32 = 22; // args: lba, src addr, block count
const CMD_MEL: u32 = 23; // args: param block addr (mel::MelParams)
const CMD_MELNORM: u32 = 24; // args: param block addr (mel::MelNormParams)
const CMD_RECORD: u32 = 25; // args: dst addr, sample count (16 kHz i16 mono)

/// Host-written parameter block for CMD_LN (all addresses absolute).
/// Layout is channel-planar: src is i16[ch][w], dst i8[ch][w].
#[repr(C)]
struct LnParams {
    src: u32,
    dst: u32,
    ch: u32,
    w: u32,
    gamma: u32,
    beta: u32,
    sq: Quant,
    dq: Quant,
}

/// CMD_ADD16: dst16 = a16 + b8.
#[repr(C)]
struct AddParams {
    a: u32,
    b: u32,
    dst: u32,
    len: u32,
    qa: Quant,
    qb: Quant,
    qd: Quant,
}

/// CMD_ADDPOS: dst16 = a8 + f32 vector.
#[repr(C)]
struct AddPosParams {
    a: u32,
    b: u32,
    dst: u32,
    len: u32,
    qa: Quant,
    qd: Quant,
}

/// CMD_ATTN_HEAD: one fused attention head over channel-planar buffers.
#[repr(C)]
struct AttnParams {
    q: u32,
    k: u32,
    v: u32,
    ctx: u32,
    hd: u32,
    wq: u32,
    qstride: u32,
    tk: u32,
    kstride: u32,
    zq: i32,
    zk: i32,
    zv: i32,
    score_mult: f32,
    v_scale: f32,
    ctx_q: Quant,
}

/// CMD_FC2SUM: residual + four dequantized fc2 partials -> int16.
#[repr(C)]
struct Fc2SumParams {
    p: [u32; 4],
    a: u32,
    dst: u32,
    len: u32,
    pq: [Quant; 4],
    qa: Quant,
    qd: Quant,
}

// Softmax scratch: one dequantized score row. Bounds CMD_SOFTMAX cols.
const MAX_SOFTMAX_COLS: usize = 640;
static mut SOFTMAX_SCRATCH: [f32; MAX_SOFTMAX_COLS] = [0.0; MAX_SOFTMAX_COLS];

#[inline]
unsafe fn sl<T>(addr: u32, len: u32) -> &'static [T] {
    core::slice::from_raw_parts(addr as *const T, len as usize)
}

#[inline]
unsafe fn sl_mut<T>(addr: u32, len: u32) -> &'static mut [T] {
    core::slice::from_raw_parts_mut(addr as *mut T, len as usize)
}

unsafe fn dispatch(cmd: u32, a: &[u32; 8]) -> i32 {
    match cmd {
        CMD_PING => 0x50494E47, // "PING"
        CMD_RUN_NPU => {
            let _wd = WdogGuard::arm();
            slot::run(a[0], a[1], "host")
        }
        CMD_LUT8 => {
            let lut: &[i8] = sl(a[0], 256);
            kernels::lut_i8(
                lut.try_into().unwrap(),
                sl(a[1], a[3]),
                sl_mut(a[2], a[3]),
            );
            0
        }
        CMD_MATMUL_BT => {
            kernels::matmul_i8_bt(
                sl(a[0], a[3] * a[4]),
                a[6] as i32,
                sl(a[1], a[5] * a[4]),
                sl_mut(a[2], a[3] * a[5]),
                a[3] as usize,
                a[4] as usize,
                a[5] as usize,
            );
            0
        }
        CMD_ATTN_HEAD => {
            let p = &*(a[0] as *const AttnParams);
            if p.tk as usize > kernels::MAX_KEYS {
                return -1;
            }
            kernels::attn_head(
                sl(p.q, p.hd * p.qstride),
                sl(p.k, p.hd * p.kstride),
                sl(p.v, p.hd * p.kstride),
                sl_mut(p.ctx, p.hd * p.qstride),
                p.hd as usize,
                p.wq as usize,
                p.qstride as usize,
                p.tk as usize,
                p.kstride as usize,
                p.zq,
                p.zk,
                p.zv,
                p.score_mult,
                p.v_scale,
                p.ctx_q,
                // interlayer is idle between host-driven NPU runs, same
                // as the standalone attention phases
                &mut *(core::ptr::addr_of_mut!(nrf_axon_interlayer_buffer)
                    as *mut kernels::AttnScratch),
            );
            0
        }
        CMD_FC2SUM => {
            let p = &*(a[0] as *const Fc2SumParams);
            kernels::fc2_sum(
                [
                    sl(p.p[0], p.len),
                    sl(p.p[1], p.len),
                    sl(p.p[2], p.len),
                    sl(p.p[3], p.len),
                ],
                &p.pq,
                sl(p.a, p.len),
                p.qa,
                sl_mut(p.dst, p.len),
                p.qd,
            );
            0
        }
        CMD_SOFTMAX => {
            if a[3] as usize > MAX_SOFTMAX_COLS {
                return -1;
            }
            kernels::softmax_rows_quant(
                sl(a[0], a[2] * a[3]),
                f32::from_bits(a[4]),
                sl_mut(a[1], a[2] * a[3]),
                a[3] as usize,
                &mut *core::ptr::addr_of_mut!(SOFTMAX_SCRATCH),
            );
            0
        }
        CMD_REQUANT => {
            kernels::requant_i32_to_i8(
                sl(a[0], a[2]),
                f32::from_bits(a[3]),
                sl_mut(a[1], a[2]),
                Quant {
                    scale: f32::from_bits(a[4]),
                    zp: a[5] as i32,
                },
            );
            0
        }
        CMD_LN => {
            let p = &*(a[0] as *const LnParams);
            kernels::ln_planar_i16_to_i8(
                sl(p.src, p.ch * p.w),
                p.sq,
                sl(p.gamma, p.ch),
                sl(p.beta, p.ch),
                sl_mut(p.dst, p.ch * p.w),
                p.dq,
                p.ch as usize,
                p.w as usize,
            );
            0
        }
        CMD_ADD16 => {
            let p = &*(a[0] as *const AddParams);
            kernels::add_i16_i8(
                sl(p.a, p.len),
                p.qa,
                sl(p.b, p.len),
                p.qb,
                sl_mut(p.dst, p.len),
                p.qd,
            );
            0
        }
        CMD_ADDPOS => {
            let p = &*(a[0] as *const AddPosParams);
            kernels::add_i8_f32_to_i16(
                sl(p.a, p.len),
                p.qa,
                sl(p.b, p.len),
                sl_mut(p.dst, p.len),
                p.qd,
            );
            0
        }
        CMD_SD_INIT => {
            // Backend-agnostic: probes the USB stick (then the card with
            // the sd-card feature). No watchdog here: every wait in both
            // drivers is DWT-bounded, and a stick recovering from an
            // interrupted session needs a multi-second ready budget that
            // would trip the 2 s limit.
            let rc = storage::init();
            #[cfg(feature = "sd-card")]
            if rc != 0 {
                // leave the SD bus high-Z so external testers can drive it
                sd::release_pins();
            }
            rc
        }
        // No watchdog on storage transfers either: USB write budgets run
        // past the 2 s limit by design (slow-stick stalls), and every
        // wait in both drivers is DWT-bounded.
        CMD_SD_READ => storage::read_blocks(a[0], a[1] as *mut u8, a[2]),
        CMD_SD_WRITE => storage::write_blocks(a[0], a[1] as *const u8, a[2]),
        CMD_MEL => {
            let _wd = WdogGuard::arm();
            mel::mel_frames(&*(a[0] as *const mel::MelParams));
            0
        }
        CMD_MELNORM => {
            mel::mel_normalize(&*(a[0] as *const mel::MelNormParams));
            0
        }
        // Recording runs longer than the watchdog budget; the PDM stream is
        // hardware-validated (KWS) and self-limiting, so it is not armed.
        CMD_RECORD => record(a[0] as *mut i16, a[1] as usize),
        CMD_LOGITS_MAX => {
            kernels::logits_max(
                sl(a[0], a[3]),
                sl(a[1], a[3]),
                sl(a[2], a[3]),
                &mut *(a[4] as *mut kernels::ArgmaxState),
            );
            0
        }
        _ => -100,
    }
}

// PDM mic (same pins the KWS project validated on this DK).
pub(crate) const MIC_CLK: pdm::Pin = pdm::Pin { port: 1, pin: 23 };
pub(crate) const MIC_DIN: pdm::Pin = pdm::Pin { port: 1, pin: 24 };

// One PDM hop = 20 ms; ping-pong pair for the record command.
#[repr(C, align(4))]
pub(crate) struct PdmBuf(pub [i16; 320]);
pub(crate) static mut PDM_BUF0: PdmBuf = PdmBuf([0; 320]);
pub(crate) static mut PDM_BUF1: PdmBuf = PdmBuf([0; 320]);

/// Record `n` 16 kHz samples into `dst` (blocking). Returns overrun count.
unsafe fn record(dst: *mut i16, n: usize) -> i32 {
    let b0 = &mut (*core::ptr::addr_of_mut!(PDM_BUF0)).0;
    let b1 = &mut (*core::ptr::addr_of_mut!(PDM_BUF1)).0;
    let mut stream = pdm::Pdm::init(MIC_CLK, MIC_DIN).start(b0, b1);
    // Drop the first hop: the very first session after flashing counts one
    // startup overrun (KWS finding) and the mic's DC settle lands there too.
    stream.next_buffer();
    stream.overruns = 0;
    let mut written = 0usize;
    while written < n {
        let hop = stream.next_buffer();
        let take = hop.len().min(n - written);
        core::ptr::copy_nonoverlapping(hop.as_ptr(), dst.add(written), take);
        written += take;
    }
    let overruns = stream.overruns;
    stream.stop();
    overruns as i32
}

#[entry]
fn main() -> ! {
    // HAL bring-up. OSCILLATORS.PLL.FREQ selects the MCU-domain (CPU)
    // clock: the device BOOTS AT 64 MHz (datasheet 5.5.3) and must be
    // switched to 128 MHz when the CPU starts, before any high-frequency
    // peripheral is enabled. Found the hard way: the 3 s host-grace window
    // took 48 s on hardware. init() requests it (CK128) and also applies
    // the FICR trim values, unlocks the debug port, disables the glitch
    // detectors, enables the DC/DC regulator and starts the RC LFCLK --
    // the system bring-up Zephyr does on this part. The FLPR coprocessor
    // is left alone: nothing here loads it.
    let mut config = embassy_nrf::config::Config::default();
    config.clock_speed = embassy_nrf::config::ClockSpeed::CK128;
    config.flpr_reset = embassy_nrf::config::FlprReset::Leave;
    let p = embassy_nrf::init(config);
    // The switch takes a moment: wait for it before enabling anything
    // clocked from it.
    for _ in 0..1_000_000 {
        if pac::OSCILLATORS.pll().currentfreq().read().currentfreq()
            == pac::oscillators::vals::Currentfreq::Ck128m
        {
            break;
        }
    }
    // The instruction cache (ICACHE, PPB region) is DISABLED at reset;
    // without it every taken branch refetches from RRAM through fixed
    // wait states. Measured on this loop-heavy firmware: ~16 CPU cycles
    // per 2-instruction delay iteration, i.e. code ran ~5x slower than
    // the core clock suggests.
    pac::ICACHE.tasks_invalidatecache().write_value(1);
    cortex_m::asm::delay(64);
    pac::ICACHE.enable().write(|w| w.set_enable(true));
    cortex_m::asm::isb();

    // Park the display's serial instance and pins; the standalone app
    // builds the driver when it probes the panel.
    display::attach(p.SERIAL22, p.P3_02, p.P3_03);

    let mut cp = cortex_m::Peripherals::take();
    unsafe {
        // An attached debugger arms DEMCR vector catch (VC_HARDERR etc.),
        // which halts the core at exception ENTRY -- our HardFault handler
        // never runs and no fault dump is printed. Clear the catch bits
        // (0: VC_CORERESET, 4..10: VC_MMERR..VC_HARDERR) so faults vector
        // into the handler; TRCENA and the rest are preserved.
        if let Some(cp) = cp.as_mut() {
            cp.DCB.demcr.modify(|v| v & !0x0000_07F1);
        }
        // Hardware stack-limit guard (ARMv8-M MSPLIM): the stack shares
        // its 132 K with .bss and the margin is tight -- a deep frame
        // once dipped past _stack_end and silently corrupted the Axon
        // driver's state at the top of .bss (wild register write, wild
        // BFAR). With the limit armed the same bug is an immediate
        // STKOF fault at the exact instruction instead.
        extern "C" {
            static _stack_end: u32;
        }
        cortex_m::register::msplim::write(
            core::ptr::addr_of!(_stack_end) as u32);
        // The AXONBUF region is NOLOAD (fixed-address interlayer/psum
        // buffers): zero it here since cortex-m-rt only zeroes .bss.
        core::ptr::write_bytes(
            core::ptr::addr_of_mut!(nrf_axon_interlayer_buffer) as *mut u8,
            0,
            INTERLAYER_BUFFER_BYTES,
        );
        core::ptr::write_bytes(
            core::ptr::addr_of_mut!(nrf_axon_psum_buffer) as *mut u8,
            0,
            PSUM_BUFFER_BYTES,
        );
    }
    let channels = rtt_init! {
        up: {
            0: {
                size: 2048,
                // NoBlockSkip: never freeze the firmware when no host reads.
                mode: ChannelMode::NoBlockSkip,
                name: "log",
            }
        }
    };
    rtt_target::set_print_channel(channels.up.0);

    let died_at = unsafe { core::ptr::read_volatile(BREADCRUMB) };
    let died_at2 = unsafe { core::ptr::read_volatile(BREADCRUMB2) };
    rprintln!(
        "boot: whisper fw build {} (previous life died at {:#x}/{:#x})",
        env!("BUILD_ID"),
        died_at,
        died_at2
    );
    crumb(0x100);

    let mb = core::ptr::addr_of_mut!(MAILBOX);
    unsafe {
        core::ptr::write_volatile(core::ptr::addr_of_mut!((*mb).magic), MAILBOX_BOOT);
    }

    // Cycle counter for timing; SysTick for the hang watchdog.
    if let Some(cp) = cp.as_mut() {
        cp.DCB.enable_trace();
        cp.DWT.enable_cycle_counter();
        cp.SYST
            .set_clock_source(cortex_m::peripheral::syst::SystClkSource::Core);
        cp.SYST.set_reload(1_280_000 - 1); // 10 ms at 128 MHz
        cp.SYST.clear_current();
        cp.SYST.enable_interrupt();
        cp.SYST.enable_counter();
    }

    let rc = {
        let _wd = WdogGuard::arm();
        platform::init()
    };
    rprintln!("whisper executor: axon init rc={}", rc);
    rprintln!(
        "slot @ {:#010x} ({}K)  arena @ {:#010x} ({}K)",
        slot::SLOT_BASE,
        slot::SLOT_BYTES / 1024,
        core::ptr::addr_of!(ARENA) as usize,
        ARENA_BYTES / 1024
    );
    // Logging the buffer addresses also RETAINS them: nothing else references
    // these statics (runtime-loaded blobs reach them via embedded absolute
    // addresses), and --gc-sections would otherwise drop the storage the NPU
    // DMAs into.
    rprintln!(
        "interlayer @ {:#010x} ({}K)  psum @ {:#010x} ({}K)",
        core::ptr::addr_of!(nrf_axon_interlayer_buffer) as usize,
        INTERLAYER_BUFFER_BYTES / 1024,
        core::ptr::addr_of!(nrf_axon_psum_buffer) as usize,
        PSUM_BUFFER_BYTES / 1024
    );

    unsafe {
        if rc == 0 {
            core::ptr::write_volatile(core::ptr::addr_of_mut!((*mb).magic), MAILBOX_MAGIC);
        } else {
            core::ptr::write_volatile(core::ptr::addr_of_mut!((*mb).status), rc);
            core::ptr::write_volatile(core::ptr::addr_of_mut!((*mb).magic), MAILBOX_FAIL);
        }
    }

    // Grace window: a connected host (tape player / decode driver) issues
    // its PING right after the magic appears. If one does, stay a mailbox
    // executor; otherwise go standalone (which itself falls back here when
    // no SD image is present). DWT-timed to exactly 3 s wall time. (The earlier 5 s
    // "misses" were actually watchdog deaths mid-SD_WRITE re-zeroing
    // ack_seq; the host in fact pings within ~1 s of the magic.) --
    // asm::delay pacing shrank with the icache fix and the host once lost
    // the race.
    let grace_start = cortex_m::peripheral::DWT::cycle_count();
    while cortex_m::peripheral::DWT::cycle_count().wrapping_sub(grace_start)
        < 3 * 128_000_000
    {
        unsafe {
            let seq = core::ptr::read_volatile(core::ptr::addr_of!((*mb).cmd_seq));
            if seq != core::ptr::read_volatile(core::ptr::addr_of!((*mb).ack_seq)) {
                rprintln!("host detected: mailbox mode");
                mailbox_loop();
            }
        }
    }
    app::run();
}

/// The host-driven executor (also the fallback when standalone cannot start).
pub fn mailbox_loop() -> ! {
    let mb = core::ptr::addr_of_mut!(MAILBOX);
    unsafe {
        loop {
            let seq = core::ptr::read_volatile(core::ptr::addr_of!((*mb).cmd_seq));
            if seq == core::ptr::read_volatile(core::ptr::addr_of!((*mb).ack_seq)) {
                continue;
            }
            let cmd = core::ptr::read_volatile(core::ptr::addr_of!((*mb).cmd));
            let args = core::ptr::read_volatile(core::ptr::addr_of!((*mb).args));
            crumb(0x300 + cmd);
            let status = dispatch(cmd, &args);
            crumb(0x400 + cmd);
            core::ptr::write_volatile(core::ptr::addr_of_mut!((*mb).status), status);
            core::ptr::write_volatile(core::ptr::addr_of_mut!((*mb).ack_seq), seq);
        }
    }
}
