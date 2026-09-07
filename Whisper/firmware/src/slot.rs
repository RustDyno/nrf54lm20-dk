//! The runtime-loaded Axon model slot.
//!
//! The host streams a per-layer blob (produced by tools/make-blob.sh) into
//! the SLOT memory region, then issues CMD_RUN_NPU. A blob is the generated
//! Axon model header compiled and linked at SLOT_BASE against this firmware's
//! ELF, so every embedded pointer (command buffer, weights, interlayer
//! addresses) is already absolute and correct; the blob starts with a header
//! naming the descriptor.

use crate::bindings;

/// Must match memory.x's SLOT region and tools/slot.ld.
pub const SLOT_BASE: usize = 0x2004_B000;
pub const SLOT_BYTES: usize = 208 * 1024;

/// First word of every blob (tools/make-blob.sh writes it): "LAYR".
pub const SLOT_MAGIC: u32 = 0x4C41_5952;

#[repr(C)]
struct SlotHeader {
    magic: u32,
    model: *const bindings::nrf_axon_nn_compiled_model_s,
}

/// Cycles spent in the driver's model validation, for the phase profile
/// (a fixed cost per run that the decode's 1400 tiny runs pay too).
pub static mut VALIDATE_CYCLES: u64 = 0;

/// Validate and run the blob currently in the slot. `input`/`output` are
/// absolute arena addresses (0 = the model's own interlayer locations, per
/// the driver's NULL contract).
pub unsafe fn run(input: u32, output: u32, name: &str) -> i32 {
    run_at(SLOT_BASE, input, output, name)
}

/// The blob's declared interlayer use in bytes (from its descriptor), or
/// None if no valid blob header sits at `base`. Decode parks tables in
/// the interlayer above what its blobs touch and checks this.
pub unsafe fn interlayer_needed(base: usize) -> Option<u32> {
    let hdr = &*(base as *const SlotHeader);
    if hdr.magic != SLOT_MAGIC {
        return None;
    }
    let model = hdr.model as usize;
    if model < base || model >= SLOT_BASE + SLOT_BYTES {
        return None;
    }
    Some((*hdr.model).interlayer_buffer_needed)
}

/// Same for a blob linked and loaded at `base` inside the slot (the
/// per-token decoder blobs live at the top of it, app.rs DEC_BASE).
pub unsafe fn run_at(base: usize, input: u32, output: u32, name: &str) -> i32 {
    let hdr = &*(base as *const SlotHeader);
    if hdr.magic != SLOT_MAGIC {
        return -101;
    }
    let model = hdr.model;
    if (model as usize) < base || (model as usize) >= SLOT_BASE + SLOT_BYTES {
        return -102;
    }
    let t0 = cortex_m::peripheral::DWT::cycle_count();
    let rc = bindings::nrf_axon_nn_model_validate(model);
    VALIDATE_CYCLES += cortex_m::peripheral::DWT::cycle_count().wrapping_sub(t0) as u64;
    #[cfg(feature = "npu-log")]
    rtt_target::rprintln!("npu {}: validate rc={} infer...", name, rc.0);
    if rc.0 != 0 {
        rtt_target::rprintln!("npu {}: validate rc={}", name, rc.0);
        return rc.0;
    }
    crate::crumb2(0x201); // entering infer_sync
    let input = if input == 0 {
        core::ptr::null()
    } else {
        input as *const i8
    };
    let output = if output == 0 {
        core::ptr::null_mut()
    } else {
        output as *mut i8
    };
    let rc = bindings::nrf_axon_nn_model_infer_sync(model, input, output).0;
    crate::crumb2(0x202); // infer_sync returned
    #[cfg(feature = "npu-log")]
    rtt_target::rprintln!("npu {}: infer rc={}", name, rc);
    #[cfg(not(feature = "npu-log"))]
    if rc != 0 {
        rtt_target::rprintln!("npu {}: infer rc={}", name, rc);
    }
    rc
}
