//! Bare-metal implementation of the Axon platform interface
//! (`nrf_axon_platform.h`). The pre-compiled driver calls these; we provide the
//! RTOS-equivalent primitives without Zephyr.
//!
//! Configured for a single-threaded application using synchronous inference.
//! Reservation is trivial (one owner), the user-event is a simple flag, and a
//! driver-event is processed inline (the header explicitly permits calling
//! `nrf_axon_process_driver_event()` directly on bare metal).

use core::ffi::c_void;
use core::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use crate::bindings;

/// Base address of the AXONS peripheral (secure alias).
///
/// From the nRF54LM20B MDK: `NRF_AXONS_S_BASE = 0x50056000`
/// (non-secure alias `NRF_AXONS_NS_BASE = 0x40056000`). The CPU boots secure
/// without TrustZone/SPU setup, so use the secure alias.
pub const AXON_BASE_ADDR: usize = 0x5005_6000;

/// AXONS interrupt line (`AXONS_IRQn`). Model inference blocks in EVENT mode
/// (the driver hardcodes it), so the completion IRQ must be unmasked in the
/// NVIC -- that is the platform's job (Zephyr does IRQ_CONNECT + irq_enable);
/// the driver blob only enables the peripheral-side interrupt.
pub const AXONS_IRQN: u16 = 86;

#[derive(Clone, Copy)]
struct AxonsIrq;

unsafe impl cortex_m::interrupt::InterruptNumber for AxonsIrq {
    fn number(self) -> u16 {
        AXONS_IRQN
    }
}

/// ENABLE register offset within the AXONS block (MDK: ENABLE @ 0x400, EN = bit 0).
const AXON_ENABLE_OFFSET: usize = 0x400;
const AXON_ENABLE_EN_BIT: u32 = 1; // AXONS_ENABLE_EN_Msk

static USER_EVENT: AtomicBool = AtomicBool::new(false);

/// Power votes, mirroring Zephyr's onoff manager: the block is enabled while
/// at least one reservation holds a vote and fully power-cycled between
/// inferences. The per-inference ENABLE=0 is what clears wedged engine state
/// (e.g. when a debug session killed the firmware mid-inference) -- relying on
/// a single cycle at boot proved insufficient on hardware.
static POWER_VOTES: AtomicU32 = AtomicU32::new(0);

#[inline]
fn enable_reg() -> *mut u32 {
    (AXON_BASE_ADDR + AXON_ENABLE_OFFSET) as *mut u32
}

fn power_vote_on() {
    if POWER_VOTES.fetch_add(1, Ordering::SeqCst) == 0 {
        unsafe {
            core::ptr::write_volatile(enable_reg(), AXON_ENABLE_EN_BIT);
            bindings::nrf_axon_driver_power_on();
        }
    }
}

fn power_vote_off() {
    if POWER_VOTES.fetch_sub(1, Ordering::SeqCst) == 1 {
        unsafe {
            bindings::nrf_axon_driver_power_off();
            core::ptr::write_volatile(enable_reg(), 0);
        }
    }
}

/// One-time bring-up, mirroring Zephyr's `nrf_axon_platform_init`: enable the
/// block, init the driver, wire the IRQ, then power the block back OFF. Each
/// inference powers it on/off via the reservation hooks below. Returns the
/// driver result code (0 == success).
///
/// RRAM note: the Zephyr platform votes to keep RRAM in standby
/// (`nrf_sys_event_register`) so the engine can read model constants during
/// inference. After reset RRAM is active (code runs from it), so no action is
/// needed for a simple app. Only if you enter low-power modes that power-gate
/// RRAM must you hold it in standby across inference.
pub fn init() -> i32 {
    unsafe {
        // Power-cycle first: probe-rs reflash is only a soft reset, so engine
        // state survives from a previous (possibly killed mid-inference)
        // session. ENABLE=0 resets it.
        core::ptr::write_volatile(enable_reg(), 0);
        cortex_m::asm::delay(64);
        core::ptr::write_volatile(enable_reg(), AXON_ENABLE_EN_BIT);

        let r = bindings::nrf_axon_driver_init(AXON_BASE_ADDR as *mut c_void);
        if r.0 != 0 {
            return r.0;
        }

        // Zephyr's ordering: driver_init, then IRQ_CONNECT + irq_enable. A
        // stale IRQ delivered into an uninitialized driver generates a
        // spurious user event and shifts every infer_sync one completion
        // early. No unpend: a pending bring-up event must be serviced by the
        // (now initialized) driver, not discarded.
        cortex_m::peripheral::NVIC::unmask(AxonsIrq);
        USER_EVENT.store(false, Ordering::SeqCst);

        // Like Zephyr: leave the block off until the first reservation.
        core::ptr::write_volatile(enable_reg(), 0);
    }
    0
}

// --- Interrupt masking: lightweight critical sections for the driver. ---

#[no_mangle]
pub extern "C" fn nrf_axon_platform_disable_interrupts() -> u32 {
    let was_enabled = cortex_m::register::primask::read().is_active();
    cortex_m::interrupt::disable();
    was_enabled as u32
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_restore_interrupts(restore_value: u32) {
    if restore_value != 0 {
        // SAFETY: only re-enables if interrupts were enabled before the matching
        // disable call, preserving nesting.
        unsafe { cortex_m::interrupt::enable() };
    }
}

// --- Hardware reservation: single owner, so always granted. Reservations
// carry the power votes (Zephyr: reserve -> onoff request -> axon_power_on),
// which power-cycles the engine around every inference.

#[no_mangle]
pub extern "C" fn nrf_axon_platform_reserve_for_user() -> bool {
    power_vote_on();
    true
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_free_reservation_from_user() {
    power_vote_off();
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_reserve_for_driver() -> bool {
    power_vote_on();
    true
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_free_reservation_from_driver() {
    power_vote_off();
}

// --- Event signalling. ---

#[no_mangle]
pub extern "C" fn nrf_axon_platform_generate_user_event() {
    USER_EVENT.store(true, Ordering::SeqCst);
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_wait_for_user_event() {
    // Model inference blocks in NRF_AXON_SYNC_MODE_BLOCKING_EVENT
    // (nrf_axon_nn_infer.c): completion normally arrives via the AXONS IRQ
    // (handler in main.rs -> nrf_axon_handle_interrupt -> generate_driver_event
    // -> process_driver_event inline -> generate_user_event). Belt and braces
    // for stale peripheral state across soft resets: also poll the handler
    // here -- it clears the event at its source, is benign when nothing is
    // pending, and the ISR racing it is harmless now that both run the same
    // initialized-driver path.
    while !USER_EVENT.swap(false, Ordering::SeqCst) {
        unsafe {
            bindings::nrf_axon_handle_interrupt();
        }
    }
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_generate_driver_event() {
    // Bare-metal: process inline rather than signalling a driver thread.
    unsafe { bindings::nrf_axon_process_driver_event() };
}

// --- Timing (used only by the driver's profiling/test paths). ---

#[no_mangle]
pub extern "C" fn nrf_axon_platform_get_clk_hz() -> u32 {
    // CPU/Axon clock; used to convert profiling ticks. Adjust if you change clocks.
    128_000_000
}

#[no_mangle]
pub extern "C" fn nrf_axon_platform_get_ticks() -> u32 {
    cortex_m::peripheral::DWT::cycle_count()
}
