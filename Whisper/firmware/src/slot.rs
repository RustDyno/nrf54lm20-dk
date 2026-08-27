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

/// Validate and run the blob currently in the slot. `input`/`output` are
/// absolute arena addresses (0 = the model's own interlayer locations, per
/// the driver's NULL contract).
pub unsafe fn run(input: u32, output: u32, name: &str) -> i32 {
    let hdr = &*(SLOT_BASE as *const SlotHeader);
    if hdr.magic != SLOT_MAGIC {
        return -101;
    }
    let model = hdr.model;
    if (model as usize) < SLOT_BASE || (model as usize) >= SLOT_BASE + SLOT_BYTES {
        return -102;
    }
    let rc = bindings::nrf_axon_nn_model_validate(model);
    if VERBOSE {
        rtt_target::rprintln!("npu {}: validate rc={} infer...", name, rc.0);
    }
    if rc.0 != 0 {
        rtt_target::rprintln!("npu {}: validate FAILED rc={}", name, rc.0);
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
    if VERBOSE {
        rtt_target::rprintln!("npu: infer rc={}", rc);
    } else if rc != 0 {
        rtt_target::rprintln!("npu {}: infer FAILED rc={}", name, rc);
    }
    rc
}

/// Per-inference chatter (two RTT lines per validate/infer pair, hundreds
/// per utterance) drowned everything else out of the log. Quiet by default
/// now that bring-up is done; failures always print.
pub const VERBOSE: bool = false;
