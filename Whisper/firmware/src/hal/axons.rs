//! The Axon NPU register block (AXONS), written the way chiptool generates
//! nrf-pac: the block is absent from the public SVD, so the PAC has neither
//! a peripheral nor an interrupt for it. Base addresses and the ENABLE
//! register come from the nRF54LM20B MDK (`NRF_AXONS_S_BASE`,
//! `AXONS_ENABLE_EN_Msk`, `AXONS_IRQn`); everything else in the block is
//! driven by Nordic's pre-compiled driver.
//!
//! An SVD patch for nrf-pac (svd-patches, svdtools syntax) that would make
//! this generated code redundant:
//!
//! ```yaml
//! _add:
//!   AXONS_S:
//!     description: Axon neural processing unit
//!     baseAddress: 0x50056000
//!     addressBlock: { offset: 0, size: 0x1000, usage: registers }
//!     interrupts: { AXONS: { description: AXONS, value: 86 } }
//!     registers:
//!       ENABLE:
//!         addressOffset: 0x400
//!         description: Enable the block
//!         fields: { EN: { bitOffset: 0, bitWidth: 1 } }
//!   AXONS_NS:
//!     derivedFrom: AXONS_S
//!     baseAddress: 0x40056000
//! ```

#![allow(dead_code)]

use embassy_nrf::pac::common::{Reg, RW};

/// AXONS interrupt line (`AXONS_IRQn`).
pub const IRQ: u16 = 86;

/// Axon neural processing unit.
#[derive(Copy, Clone, Eq, PartialEq)]
pub struct Axons {
    ptr: *mut u8,
}

unsafe impl Send for Axons {}
unsafe impl Sync for Axons {}

impl Axons {
    #[inline(always)]
    pub const unsafe fn from_ptr(ptr: *mut ()) -> Self {
        Self { ptr: ptr as _ }
    }

    #[inline(always)]
    pub const fn as_ptr(&self) -> *mut () {
        self.ptr as _
    }

    /// Enable the block.
    #[inline(always)]
    pub const fn enable(self) -> Reg<regs::Enable, RW> {
        unsafe { Reg::from_ptr(self.ptr.wrapping_add(0x0400usize) as _) }
    }
}

pub mod regs {
    /// Enable the block.
    #[repr(transparent)]
    #[derive(Copy, Clone, Eq, PartialEq)]
    pub struct Enable(pub u32);

    impl Enable {
        #[inline(always)]
        pub const fn en(&self) -> bool {
            let val = (self.0 >> 0usize) & 0x01;
            val != 0
        }

        #[inline(always)]
        pub const fn set_en(&mut self, val: bool) {
            self.0 = (self.0 & !(0x01 << 0usize)) | (((val as u32) & 0x01) << 0usize);
        }
    }

    impl Default for Enable {
        #[inline(always)]
        fn default() -> Enable {
            Enable(0)
        }
    }
}

/// AXONS, non-secure alias.
pub const AXONS_NS: Axons = unsafe { Axons::from_ptr(0x4005_6000usize as _) };
/// AXONS, secure alias. The core boots secure without SPU setup, so this is
/// the one the firmware uses.
pub const AXONS_S: Axons = unsafe { Axons::from_ptr(0x5005_6000usize as _) };
pub const AXONS: Axons = AXONS_S;
