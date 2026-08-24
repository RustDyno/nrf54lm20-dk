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

use cortex_m_rt::entry;
use panic_halt as _;
use rtt_target::{rprintln, rtt_init, ChannelMode};

mod bindings;
mod kernels;
mod libm_shims;
mod platform;
mod slot;

use kernels::Quant;

// Storage backing the C `extern uint32_t nrf_axon_interlayer_buffer[]` and
// `nrf_axon_psum_buffer[]`. Sizes must match the -D defines in build.rs.
const INTERLAYER_BUFFER_BYTES: usize = 147456;
const PSUM_BUFFER_BYTES: usize = 16384;

#[no_mangle]
pub static mut nrf_axon_interlayer_buffer: [u32; INTERLAYER_BUFFER_BYTES / 4] =
    [0; INTERLAYER_BUFFER_BYTES / 4];

#[no_mangle]
pub static mut nrf_axon_psum_buffer: [u32; PSUM_BUFFER_BYTES / 4] = [0; PSUM_BUFFER_BYTES / 4];

/// Activation / parameter arena. The host owns the layout: every mailbox
/// command carries absolute addresses that the host computed from this
/// symbol's ELF address plus its own allocation plan.
pub const ARENA_BYTES: usize = 100 * 1024;

#[no_mangle]
pub static mut ARENA: [u8; ARENA_BYTES] = [0; ARENA_BYTES];

// --- Device interrupt vector table (AXONS IRQ 86; see ../../npu/src/main.rs).

const AXONS_IRQN: usize = 86;

unsafe extern "C" fn default_irq_handler() {
    loop {
        cortex_m::asm::bkpt();
    }
}

unsafe extern "C" fn axons_irq_handler() {
    bindings::nrf_axon_handle_interrupt();
}

const fn vector_table() -> [unsafe extern "C" fn(); AXONS_IRQN + 1] {
    let mut t = [default_irq_handler as unsafe extern "C" fn(); AXONS_IRQN + 1];
    t[AXONS_IRQN] = axons_irq_handler;
    t
}

#[no_mangle]
#[link_section = ".vector_table.interrupts"]
pub static __INTERRUPTS: [unsafe extern "C" fn(); AXONS_IRQN + 1] = vector_table();

// --- Mailbox protocol ---------------------------------------------------------
// Host: write `args` + `cmd`, then increment `cmd_seq`. Firmware: on
// cmd_seq != ack_seq, execute, write `status`, then set ack_seq = cmd_seq.

pub const MAILBOX_MAGIC: u32 = 0x4C41_5952; // "LAYR"

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
const CMD_MATMUL_B: u32 = 5; // args: a, b, acc, m, k, n, za
const CMD_SOFTMAX: u32 = 6; // args: acc, dst, rows, cols, mult(f32 bits)
const CMD_REQUANT: u32 = 7; // args: acc, dst, len, mult(f32), scale(f32), zp
const CMD_LN: u32 = 8; // args: param block addr (LnParams)
const CMD_ADD16: u32 = 9; // args: param block addr (AddParams)
const CMD_ADDPOS: u32 = 10; // args: param block addr (AddPosParams)
const CMD_LOGITS_MAX: u32 = 11; // args: acc, mults, idx, len, state

/// Host-written parameter block for CMD_LN (all addresses absolute).
#[repr(C)]
struct LnParams {
    src: u32,
    dst: u32,
    rows: u32,
    cols: u32,
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
        CMD_RUN_NPU => slot::run(a[0], a[1]),
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
        CMD_MATMUL_B => {
            kernels::matmul_i8_b(
                sl(a[0], a[3] * a[4]),
                a[6] as i32,
                sl(a[1], a[4] * a[5]),
                sl_mut(a[2], a[3] * a[5]),
                a[3] as usize,
                a[4] as usize,
                a[5] as usize,
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
            kernels::ln_i16_to_i8(
                sl(p.src, p.rows * p.cols),
                p.sq,
                sl(p.gamma, p.cols),
                sl(p.beta, p.cols),
                sl_mut(p.dst, p.rows * p.cols),
                p.dq,
                p.cols as usize,
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

#[entry]
fn main() -> ! {
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

    let rc = platform::init();
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

    let mb = core::ptr::addr_of_mut!(MAILBOX);
    unsafe {
        if rc == 0 {
            core::ptr::write_volatile(core::ptr::addr_of_mut!((*mb).magic), MAILBOX_MAGIC);
        }
        loop {
            let seq = core::ptr::read_volatile(core::ptr::addr_of!((*mb).cmd_seq));
            if seq == core::ptr::read_volatile(core::ptr::addr_of!((*mb).ack_seq)) {
                continue;
            }
            let cmd = core::ptr::read_volatile(core::ptr::addr_of!((*mb).cmd));
            let args = core::ptr::read_volatile(core::ptr::addr_of!((*mb).args));
            let status = dispatch(cmd, &args);
            core::ptr::write_volatile(core::ptr::addr_of_mut!((*mb).status), status);
            core::ptr::write_volatile(core::ptr::addr_of_mut!((*mb).ack_seq), seq);
        }
    }
}
