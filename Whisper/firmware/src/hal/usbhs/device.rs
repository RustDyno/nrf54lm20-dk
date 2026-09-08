//! USBHS in device mode: endpoint 0 and the numbered endpoints, polled.
//!
//! Buffer DMA, no interrupts: GINTSTS and the per-endpoint DOEPINT/DIEPINT
//! registers are read directly. The driver owns the register side of the
//! bus events and endpoint transfers; the control-request handling, class
//! descriptors and framing above the pipes belong to the caller.
//!
//! Endpoint register files: IN at 0x900, OUT at 0xB00, stride 0x20 per
//! endpoint. Endpoints 1..15 share one layout, which the SVD spells out
//! as fifteen copies (DIEPCTL1, DIEPCTL2, ...); the numbered accessors
//! here are the endpoint-1 register types at the endpoint's own address.
//! Endpoint 0 differs (a 2-bit MPS enum, a 7-bit XferSize) and keeps its
//! own accessors.

use embassy_nrf::pac;
use embassy_nrf::pac::common::{Reg, RW};
use embassy_nrf::pac::usbhscore::regs;
use embassy_nrf::pac::usbhscore::vals::{
    Devspd, Diepctl0Mps, Diepctl1Eptype, Diepctl1Txfnum, Dmaen, Doepctl1Eptype, Enumspd,
    GrstctlTxfnum, Hbstlen, Supcnt,
};
use embassy_nrf::Peri;

use super::{delay_ms, wait_for, Error, Instance, Platform};

/// Endpoint 0's packet size, fixed by the driver (the value advertised at
/// either speed).
pub const EP0_MPS: usize = 64;

/// FIFO carve-up in 32-bit words: the shared RX FIFO and one TX FIFO per
/// IN endpoint, endpoint 0 first. The endpoint-info block the databook
/// wants above every FIFO is placed after the last one.
#[derive(Clone, Copy)]
pub struct Config {
    pub rx_fifo_words: u32,
    pub tx_fifo_words: [u32; 3],
}

impl Default for Config {
    fn default() -> Self {
        // The shared RX FIFO needs (4*ctrl_eps + 6) + (MPS/4 + 1)*packets +
        // 2*out_eps + 1; 640 words covers four 512-byte packets in flight
        // with room to spare.
        Self {
            rx_fifo_words: 640,
            tx_fifo_words: [256, 1024, 64],
        }
    }
}

/// The speed the host enumerated us at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speed {
    High,
    Full,
}

/// A bus-level event serviced by [`Device::poll_bus`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BusEvent {
    None,
    /// USB reset from the host: the address is back to 0 and every
    /// endpoint is deconfigured; the caller re-arms its SETUP landing zone.
    Reset,
    /// Speed enumeration done; the caller re-arms its SETUP landing zone.
    Enumerated(Speed),
}

/// What endpoint 0 OUT has delivered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ep0Out {
    None,
    /// A SETUP packet is in the landing zone.
    Setup,
    /// A data or status stage completed.
    TransferComplete,
}

/// What an armed endpoint transfer is doing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EndpointStatus {
    Busy,
    /// Done; for OUT, the number of bytes received.
    Complete(usize),
    AhbError,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Out,
    In,
}

/// Registers of one numbered IN endpoint.
struct InEp {
    ctl: Reg<regs::Diepctl1, RW>,
    int: Reg<regs::Diepint1, RW>,
    tsiz: Reg<regs::Dieptsiz1, RW>,
    dma: Reg<u32, RW>,
    txfsts: Reg<regs::Dtxfsts1, RW>,
}

/// Registers of one numbered OUT endpoint.
struct OutEp {
    ctl: Reg<regs::Doepctl1, RW>,
    int: Reg<regs::Doepint1, RW>,
    tsiz: Reg<regs::Doeptsiz1, RW>,
    dma: Reg<u32, RW>,
}

/// USBHS device.
pub struct Device<'d> {
    p: Platform<'d>,
}

impl<'d> Device<'d> {
    /// Take the peripheral for device use. Nothing is touched until
    /// [`Device::power_up`].
    pub fn new_blocking<T: Instance>(usb: Peri<'d, T>) -> Self {
        Self { p: Platform::new(usb) }
    }

    fn core(&self) -> pac::usbhscore::Usbhscore {
        self.p.core
    }

    fn in_ep(&self, n: u8) -> InEp {
        assert!((1..16).contains(&n));
        let base = self.core().as_ptr() as *mut u8;
        let ep = base.wrapping_add(0x900 + n as usize * 0x20);
        unsafe {
            InEp {
                ctl: Reg::from_ptr(ep as *mut _),
                int: Reg::from_ptr(ep.wrapping_add(0x08) as *mut _),
                tsiz: Reg::from_ptr(ep.wrapping_add(0x10) as *mut _),
                dma: Reg::from_ptr(ep.wrapping_add(0x14) as *mut _),
                txfsts: Reg::from_ptr(ep.wrapping_add(0x18) as *mut _),
            }
        }
    }

    fn out_ep(&self, n: u8) -> OutEp {
        assert!((1..16).contains(&n));
        let base = self.core().as_ptr() as *mut u8;
        let ep = base.wrapping_add(0xB00 + n as usize * 0x20);
        unsafe {
            OutEp {
                ctl: Reg::from_ptr(ep as *mut _),
                int: Reg::from_ptr(ep.wrapping_add(0x08) as *mut _),
                tsiz: Reg::from_ptr(ep.wrapping_add(0x10) as *mut _),
                dma: Reg::from_ptr(ep.wrapping_add(0x14) as *mut _),
            }
        }
    }

    /// Power, clocks, core reset, device mode, FIFOs and masks. The bus is
    /// held off (soft disconnect) until [`Device::attach`], so the host's
    /// first reset finds a device that can answer.
    pub fn power_up(&mut self, config: Config) -> Result<(), Error> {
        self.p.power_up()?;
        // On a refusal the core stays powered; the caller powers down.
        self.p.force_mode(false)?;
        let core = self.core();
        core.dctl().modify(|w| w.set_sftdiscon(true));
        core.gahbcfg().write(|w| {
            w.set_dmaen(Dmaen::Dmamode);
            w.set_hbstlen(Hbstlen::Word16orincr4);
        });
        core.dcfg().write(|w| w.set_devspd(Devspd::Usbhs20));

        // Written as whole words (depth in the high half, start address
        // in the low): the SVD declares these size and start fields as 10
        // bits, so the typed setters silently turn a 1024-word FIFO into a
        // zero-word one (hardware-observed: DTXFSTS = 0, every IN transfer
        // times out), while the core takes the value.
        let rx = config.rx_fifo_words;
        let [tx0, tx1, tx2] = config.tx_fifo_words;
        core.grxfsiz().write_value(regs::Grxfsiz(rx));
        core.gnptxfsiz().write_value(regs::Gnptxfsiz((tx0 << 16) | rx));
        // dieptxf(0) is DIEPTXF1 (EP1 IN), dieptxf(1) is DIEPTXF2.
        core.dieptxf(0).write_value(regs::Dieptxf((tx1 << 16) | (rx + tx0)));
        core.dieptxf(1).write_value(regs::Dieptxf((tx2 << 16) | (rx + tx0 + tx1)));
        // The databook requires the endpoint-info block to sit above every
        // FIFO. Only that base address is ours to set -- the low half is
        // the core's own total-size value, so read-modify-write it.
        let epinfo = rx + tx0 + tx1 + tx2;
        core.gdfifocfg().modify(|w| w.set_epinfobaseaddr(epinfo as u16));
        // Clock gating off: a gated PHY/core clock silently swallows
        // transfers.
        core.pcgcctl().write_value(regs::Pcgcctl(0));

        self.p.flush_fifos();

        // Interrupt lines stay masked at the NVIC; these masks only gate
        // the status bits this driver polls.
        core.diepmsk().write_value(regs::Diepmsk(0));
        core.doepmsk().write_value(regs::Doepmsk(0));
        core.daintmsk().write_value(regs::Daintmsk(0));
        core.gintsts().write(|w| w.0 = !0);
        core.dctl().modify(|w| w.set_pwronprgdone(true));
        Ok(())
    }

    pub fn power_down(&mut self) {
        self.p.power_down();
    }

    /// Connect to the bus. Held off for the USB 2.0 debounce time first:
    /// a reset that leaves VBUS up (a reflash, say) can otherwise present
    /// so brief a gap that the host never notices, keeps its old view of
    /// the device, never issues a bus reset, and enumeration simply never
    /// happens -- the device then sits in DSTS.SuspSts with no USBRst
    /// forever.
    pub fn attach(&mut self) {
        delay_ms(150);
        self.core().dctl().modify(|w| w.set_sftdiscon(false));
    }

    pub fn detach(&mut self) {
        self.core().dctl().modify(|w| w.set_sftdiscon(true));
    }

    /// Service the bus-level interrupts the driver owns. Deliberately
    /// touches NO endpoint: the endpoint transfers own their DIEPINT/DOEPINT
    /// bits and would lose completions to a poll that cleared them.
    pub fn poll_bus(&mut self) -> BusEvent {
        let core = self.core();
        let g = core.gintsts().read();
        if g.usbrst() {
            core.gintsts().write(|w| w.set_usbrst(true));
            core.dctl().modify(|w| w.set_cgoutnak(true));
            core.dcfg().modify(|w| w.set_devaddr(0));
            return BusEvent::Reset;
        }
        if g.enumdone() {
            core.gintsts().write(|w| w.set_enumdone(true));
            let speed = match core.dsts().read().enumspd() {
                Enumspd::Hs3060 => Speed::High,
                _ => Speed::Full,
            };
            // EP0 MPS is an enum there (0 = 64 B), which is what we
            // advertise at either speed, so the reset value already suits.
            core.diepctl0().modify(|w| w.set_mps(Diepctl0Mps::Bytes64));
            core.dctl().modify(|w| w.set_cgnpinnak(true));
            return BusEvent::Enumerated(speed);
        }
        BusEvent::None
    }

    pub fn set_address(&mut self, address: u8) {
        self.core().dcfg().modify(|w| w.set_devaddr(address));
    }

    // --- endpoint 0 ----------------------------------------------------

    /// Arm endpoint 0 OUT to receive SETUP packets into `buf` (64 bytes,
    /// word aligned). SUPCnt = 3 lets the core stack up to three
    /// back-to-back SETUP packets without an intervening re-arm.
    pub fn ep0_arm_setup(&mut self, buf: *mut u8) {
        let core = self.core();
        core.doepdma0().write_value(buf as u32);
        core.doeptsiz0().write(|w| {
            w.set_supcnt(Supcnt::Threepacket);
            w.set_pktcnt(true);
            w.set_xfersize(24);
        });
        core.doepctl0().modify(|w| {
            w.set_epena(true);
            w.set_cnak(true);
        });
    }

    /// Arm endpoint 0 OUT for a data or status stage of at most one packet
    /// into `buf` (word aligned, EP0_MPS bytes of room).
    pub fn ep0_arm_out(&mut self, buf: *mut u8, len: usize) {
        let core = self.core();
        core.doepdma0().write_value(buf as u32);
        core.doeptsiz0().write(|w| {
            w.set_pktcnt(true);
            w.set_xfersize(len as u8);
        });
        core.doepctl0().modify(|w| {
            w.set_epena(true);
            w.set_cnak(true);
        });
    }

    /// What endpoint 0 OUT delivered since the last call; the event is
    /// cleared.
    pub fn ep0_out_event(&mut self) -> Ep0Out {
        let core = self.core();
        let o = core.doepint0().read();
        if o.setup() {
            core.doepint0().write(|w| {
                w.set_setup(true);
                w.set_xfercompl(true);
            });
            Ep0Out::Setup
        } else if o.xfercompl() {
            core.doepint0().write(|w| w.set_xfercompl(true));
            Ep0Out::TransferComplete
        } else {
            Ep0Out::None
        }
    }

    /// Wait up to `ms` for an armed endpoint 0 OUT stage to complete.
    pub fn ep0_out_wait(&mut self, ms: u32) -> bool {
        let core = self.core();
        if wait_for(|| core.doepint0().read().xfercompl(), ms) {
            core.doepint0().write(|w| w.set_xfercompl(true));
            true
        } else {
            false
        }
    }

    /// Send `len` bytes (at most three packets: PktCnt is 2 bits on
    /// endpoint 0) from `buf` (word aligned) and wait up to `ms` for the
    /// core to hand them over. `len` = 0 sends the zero-length status
    /// packet.
    pub fn ep0_in(&mut self, buf: *const u8, len: usize, ms: u32) -> bool {
        let core = self.core();
        let pkts = if len == 0 { 1 } else { len.div_ceil(EP0_MPS) } as u8;
        core.diepdma0().write_value(buf as u32);
        core.dieptsiz0().write(|w| {
            w.set_pktcnt(pkts);
            w.set_xfersize(len as u8);
        });
        core.diepctl0().modify(|w| {
            w.set_epena(true);
            w.set_cnak(true);
        });
        let ok = wait_for(|| core.diepint0().read().xfercompl(), ms);
        core.diepint0().write(|w| w.set_xfercompl(true));
        ok
    }

    /// STALL both directions of endpoint 0 (the answer to an unsupported
    /// request).
    pub fn ep0_stall(&mut self) {
        let core = self.core();
        core.diepctl0().modify(|w| w.set_stall(true));
        core.doepctl0().modify(|w| w.set_stall(true));
    }

    // --- numbered endpoints -----------------------------------------------

    /// Activate a bulk endpoint in `direction` with its data toggle at
    /// DATA0. An IN endpoint transmits from TX FIFO `n`.
    pub fn configure_bulk(&mut self, n: u8, direction: Direction, max_packet_size: u16) {
        match direction {
            Direction::In => self.in_ep(n).ctl.write(|w| {
                w.set_mps(max_packet_size);
                w.set_usbactep(true);
                w.set_eptype(Diepctl1Eptype::Bulk);
                w.set_txfnum(Diepctl1Txfnum::from_bits(n));
                w.set_setd0pid(true);
                w.set_snak(true);
            }),
            Direction::Out => self.out_ep(n).ctl.write(|w| {
                w.set_mps(max_packet_size);
                w.set_usbactep(true);
                w.set_eptype(Doepctl1Eptype::Bulk);
                w.set_setd0pid(true);
                w.set_snak(true);
            }),
        }
    }

    /// Activate an interrupt IN endpoint (transmitting from TX FIFO `n`).
    pub fn configure_interrupt_in(&mut self, n: u8, max_packet_size: u16) {
        self.in_ep(n).ctl.write(|w| {
            w.set_mps(max_packet_size);
            w.set_usbactep(true);
            w.set_eptype(Diepctl1Eptype::Interrup);
            w.set_txfnum(Diepctl1Txfnum::from_bits(n));
            w.set_setd0pid(true);
            w.set_snak(true);
        });
    }

    /// Arm an IN transfer of `len` bytes (at most 1023 packets) from the
    /// word-aligned buffer at `buf`. A length that is not a whole number of
    /// packets ends with a short packet.
    pub fn in_arm(&mut self, n: u8, max_packet_size: usize, buf: u32, len: usize) {
        let pkts = if len == 0 { 1 } else { len.div_ceil(max_packet_size) } as u16;
        // The endpoint DMA reads a buffer the CPU has just written; make
        // those stores visible before the transfer is armed.
        cortex_m::asm::dmb();
        let ep = self.in_ep(n);
        ep.int.write(|w| w.0 = !0);
        ep.dma.write_value(buf);
        ep.tsiz.write(|w| {
            w.set_pktcnt(pkts);
            w.set_xfersize(len as u32);
        });
        ep.ctl.modify(|w| {
            w.set_epena(true);
            w.set_cnak(true);
        });
    }

    /// Completion check for [`Device::in_arm`]; a completion clears its flag.
    pub fn in_status(&mut self, n: u8) -> EndpointStatus {
        let ep = self.in_ep(n);
        let i = ep.int.read();
        if i.xfercompl() {
            ep.int.write(|w| w.set_xfercompl(true));
            EndpointStatus::Complete(0)
        } else if i.ahberr() {
            EndpointStatus::AhbError
        } else {
            EndpointStatus::Busy
        }
    }

    /// Arm an OUT transfer into the word-aligned buffer at `buf` with `cap`
    /// bytes of room (rounded down to whole packets, at least one).
    /// Returns the number of bytes the transfer was programmed for.
    pub fn out_arm(&mut self, n: u8, max_packet_size: usize, buf: u32, cap: usize) -> usize {
        let pkts = (cap / max_packet_size).max(1) as u16;
        let want = pkts as usize * max_packet_size;
        let ep = self.out_ep(n);
        ep.int.write(|w| w.0 = !0);
        ep.dma.write_value(buf);
        ep.tsiz.write(|w| {
            w.set_pktcnt(pkts);
            w.set_xfersize(want as u32);
        });
        ep.ctl.modify(|w| {
            w.set_epena(true);
            w.set_cnak(true);
        });
        want
    }

    /// Completion check for [`Device::out_arm`], reporting the bytes
    /// received (the core leaves the shortfall as the residual XferSize; a
    /// short packet ends the transfer early).
    pub fn out_status(&mut self, n: u8, want: usize) -> EndpointStatus {
        let ep = self.out_ep(n);
        let i = ep.int.read();
        if i.xfercompl() {
            ep.int.write(|w| w.set_xfercompl(true));
            let left = ep.tsiz.read().xfersize() as usize;
            EndpointStatus::Complete(want - left)
        } else if i.ahberr() {
            EndpointStatus::AhbError
        } else {
            EndpointStatus::Busy
        }
    }

    /// Bytes of the running OUT transfer not yet written (DOEPTSIZ.XferSize
    /// counts the programmed run down per packet).
    pub fn out_remaining(&self, n: u8) -> usize {
        self.out_ep(n).tsiz.read().xfersize() as usize
    }

    /// Disable an endpoint that is still armed, so the next transfer's
    /// programming is not ignored. A transfer that times out leaves EPENA
    /// set, and the databook's disable ceremony -- global NAK, then EPDis,
    /// then a TX FIFO flush for IN -- is the only way back. Without this a
    /// single timeout wedges the link until the board is reset.
    pub fn abort(&mut self, n: u8, direction: Direction) {
        let core = self.core();
        match direction {
            Direction::In => {
                let ep = self.in_ep(n);
                if !ep.ctl.read().epena() {
                    return;
                }
                core.dctl().modify(|w| w.set_sgnpinnak(true));
                wait_for(|| core.gintsts().read().ginnakeff(), 10);
                ep.ctl.modify(|w| {
                    w.set_epdis(true);
                    w.set_snak(true);
                });
                wait_for(|| ep.int.read().epdisbld(), 10);
                ep.int.write(|w| w.0 = !0);
                // Stale packets in the endpoint's TxFIFO would go out ahead
                // of the next transfer's first packet.
                core.grstctl().write(|w| {
                    w.set_txfflsh(true);
                    w.set_txfnum(GrstctlTxfnum::from_bits(n));
                });
                wait_for(|| !core.grstctl().read().txfflsh(), 10);
                core.dctl().modify(|w| w.set_cgnpinnak(true));
            }
            Direction::Out => {
                let ep = self.out_ep(n);
                if !ep.ctl.read().epena() {
                    return;
                }
                core.dctl().modify(|w| w.set_sgoutnak(true));
                wait_for(|| core.gintsts().read().goutnakeff(), 10);
                ep.ctl.modify(|w| {
                    w.set_epdis(true);
                    w.set_snak(true);
                });
                wait_for(|| ep.int.read().epdisbld(), 10);
                ep.int.write(|w| w.0 = !0);
                core.dctl().modify(|w| w.set_cgoutnak(true));
            }
        }
    }

    /// Raw register snapshot for a diagnostic line: everything that decides
    /// whether a packet on endpoint `n` can actually move.
    pub fn snapshot(&self, n: u8) -> Snapshot {
        let core = self.core();
        let iep = self.in_ep(n);
        let oep = self.out_ep(n);
        Snapshot {
            gintsts: core.gintsts().read().0,
            dsts: core.dsts().read().0,
            dctl: core.dctl().read().0,
            ghwcfg3: core.ghwcfg3().read().0,
            gdfifocfg: core.gdfifocfg().read().0,
            grxfsiz: core.grxfsiz().read().0,
            gnptxfsiz: core.gnptxfsiz().read().0,
            dieptxf1: core.dieptxf(0).read().0,
            dieptxf2: core.dieptxf(1).read().0,
            diepctl: iep.ctl.read().0,
            dieptsiz: iep.tsiz.read().0,
            diepint: iep.int.read().0,
            dtxfsts: iep.txfsts.read().0,
            doepctl: oep.ctl.read().0,
            doeptsiz: oep.tsiz.read().0,
            doepint: oep.int.read().0,
        }
    }
}

/// See [`Device::snapshot`].
#[derive(Debug, Clone, Copy)]
pub struct Snapshot {
    pub gintsts: u32,
    pub dsts: u32,
    pub dctl: u32,
    pub ghwcfg3: u32,
    pub gdfifocfg: u32,
    pub grxfsiz: u32,
    pub gnptxfsiz: u32,
    pub dieptxf1: u32,
    pub dieptxf2: u32,
    pub diepctl: u32,
    pub dieptsiz: u32,
    pub diepint: u32,
    pub dtxfsts: u32,
    pub doepctl: u32,
    pub doeptsiz: u32,
    pub doepint: u32,
}

impl<'d> Drop for Device<'d> {
    fn drop(&mut self) {
        self.p.power_down();
    }
}
