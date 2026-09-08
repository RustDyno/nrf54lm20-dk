//! USBHS in host mode: the root port and one host channel, polled.
//!
//! Every transfer runs on HOST CHANNEL 0 ONLY: a strictly sequential
//! transport (bulk-only mass storage) needs no more, one channel
//! reprogrammed per transfer is enough, and it sidesteps the HC[n]
//! address-stride ambiguity in the datasheet (0x18 per entry where
//! Synopsys hardware canonically decodes 0x20; the SVD says 0x1C -- only
//! channel 0's base of 0x500 is beyond doubt). The core retries NAKs in
//! hardware (buffer DMA mode), so a transfer either completes, errors, or
//! hits the caller's deadline.

use embassy_nrf::pac;
use embassy_nrf::pac::usbhscore::regs;
use embassy_nrf::pac::usbhscore::vals::{
    Avalidovval, CharEptype, Dmaen, Ec, Epdir, Epnum, Fslspclksel, Fslssupp, Hbstlen, Pid as PacPid,
    Prtspd, Vbvalidovval,
};
use embassy_nrf::Peri;

use super::{delay_ms, elapsed_ms, now, wait_for, Error, Instance, Platform};

/// The speed the port enumerated at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Speed {
    High,
    Full,
    Low,
}

/// Packet ID for the next transaction (HCTSIZ.Pid).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pid {
    Data0,
    Data2,
    Data1,
    /// Host side only: the SVD's device-flavored enum names this MDATA.
    Setup,
}

impl Pid {
    fn to_bits(self) -> u8 {
        match self {
            Pid::Data0 => 0,
            Pid::Data2 => 1,
            Pid::Data1 => 2,
            Pid::Setup => 3,
        }
    }

    fn from_bits(v: u8) -> Self {
        match v & 3 {
            0 => Pid::Data0,
            1 => Pid::Data2,
            2 => Pid::Data1,
            _ => Pid::Setup,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    Out,
    In,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum EndpointType {
    Control,
    Isochronous,
    Bulk,
    Interrupt,
}

/// The endpoint a transfer addresses.
#[derive(Debug, Clone, Copy)]
pub struct Endpoint {
    pub device_address: u8,
    pub number: u8,
    pub direction: Direction,
    pub ep_type: EndpointType,
    pub max_packet_size: u16,
}

/// How a transfer ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum TransferError {
    /// The caller's deadline passed (device NAKing forever, or gone); the
    /// channel has been halted.
    Timeout,
    Stall,
    Transaction,
    Babble,
    DataToggle,
    Ahb,
    FrameOverrun,
    Unknown,
}

/// A transfer armed on the channel, for split-phase use.
#[derive(Clone, Copy)]
pub struct Transfer {
    chr: regs::Char,
    start: u32,
}

impl Transfer {
    /// A placeholder for state that has no transfer in flight yet.
    pub const NONE: Transfer = Transfer {
        chr: regs::Char(0),
        start: 0,
    };
}

/// USBHS host.
pub struct Host<'d> {
    p: Platform<'d>,
}

impl<'d> Host<'d> {
    /// Take the peripheral for host use. Nothing is touched until
    /// [`Host::power_up`].
    pub fn new_blocking<T: Instance>(usb: Peri<'d, T>) -> Self {
        Self { p: Platform::new(usb) }
    }

    /// Power, clocks, core reset, host mode, FIFOs. Leaves the port
    /// unpowered.
    pub fn power_up(&mut self) -> Result<(), Error> {
        self.p.power_up()?;
        let core = self.p.core;
        // GOTGCTL session overrides: the host state machine wants A-session
        // and VBUS valid from the PHY; force both so port power does not
        // depend on how the wrapper routes the VBUS comparator in host
        // mode. Set before the mode switch (the force bit survives the
        // reset per the datasheet, but ordering this way needs no such
        // trust).
        core.gusbcfg().modify(|w| w.set_forcehstmode(true));
        core.gotgctl().modify(|w| {
            w.set_vbvalidoven(true);
            w.set_vbvalidovval(Vbvalidovval::Set1);
            w.set_avalidoven(true);
            w.set_avalidovval(Avalidovval::Value1);
        });
        // On a refusal the core stays powered so the caller can read
        // [`Host::otg_mode`] before [`Host::power_down`].
        self.p.force_mode(true)?;

        // Buffer DMA, INCR4 bursts; global interrupt output stays masked.
        core.gahbcfg().write(|w| {
            w.set_dmaen(Dmaen::Dmamode);
            w.set_hbstlen(Hbstlen::Word16orincr4);
        });
        // FIFO carve-up in 32-bit words (12160 available): RX 1024,
        // non-periodic TX 512, periodic TX 256 (unused, must still fit).
        // Written as whole words (depth in the high half, start address
        // in the low): the SVD declares these size and start fields as 10
        // bits, so the typed setters silently turn 1024 into 0, while the
        // core takes 1024-word values (hardware-observed: this carve-up
        // runs).
        core.grxfsiz().write_value(regs::Grxfsiz(0x400));
        core.gnptxfsiz().write_value(regs::Gnptxfsiz(0x200 << 16 | 0x400));
        core.hptxfsiz().write_value(regs::Hptxfsiz(0x100 << 16 | 0x600));
        self.p.flush_fifos();
        // HCFG reset state is right for the internal UTMI HS PHY
        // (FSLSPclkSel = 30/60 MHz, FS/LS-only support off, buffer DMA).
        core.hcfg().modify(|w| {
            w.set_fslspclksel(Fslspclksel::Clk3060);
            w.set_fslssupp(Fslssupp::Hsfsls);
        });
        Ok(())
    }

    pub fn power_down(&mut self) {
        self.p.power_down();
    }

    /// The core's hardwired OTG mode (GHWCFG2.OTGMODE) and the whole
    /// register, for a bring-up log line.
    pub fn otg_mode(&self) -> (u8, u32) {
        let hw = self.p.core.ghwcfg2().read();
        (hw.otgmode().to_bits(), hw.0)
    }

    /// Read-modify-write HPRT with its write-1-to-clear bits masked out
    /// (PRTENA included: writing 1 DISABLES the port), then apply `f`.
    fn hprt_rmw(&self, f: impl FnOnce(&mut regs::Hprt)) {
        let mut v = self.p.core.hprt().read();
        v.set_prtconndet(false);
        v.set_prtena(false);
        v.set_prtenchng(false);
        v.set_prtovrcurrchng(false);
        f(&mut v);
        self.p.core.hprt().write_value(v);
    }

    pub fn port_power(&mut self, on: bool) {
        self.hprt_rmw(|w| w.set_prtpwr(on));
    }

    /// Wait up to `ms` for a device on the port; on success the connect
    /// event is cleared.
    pub fn wait_connect(&mut self, ms: u32) -> bool {
        let core = self.p.core;
        if !wait_for(|| core.hprt().read().prtconnsts(), ms) {
            return false;
        }
        self.hprt_rmw(|w| w.set_prtconndet(true));
        true
    }

    /// Reset the port (60 ms) and wait for it to enable; returns the
    /// negotiated speed.
    pub fn port_reset(&mut self) -> Result<Speed, Error> {
        let core = self.p.core;
        self.hprt_rmw(|w| w.set_prtrst(true));
        delay_ms(60);
        self.hprt_rmw(|w| w.set_prtrst(false));
        if !wait_for(|| core.hprt().read().prtena(), 100) {
            return Err(Error::ResetTimeout);
        }
        let hprt = core.hprt().read();
        self.hprt_rmw(|w| {
            w.set_prtenchng(true);
            w.set_prtconndet(true);
        });
        Ok(match hprt.prtspd() {
            Prtspd::Highspd => Speed::High,
            Prtspd::Fullspd => Speed::Full,
            _ => Speed::Low,
        })
    }

    fn hc0(&self) -> pac::usbhscore::Hc {
        self.p.core.hc(0)
    }

    /// HCCHAR value for one endpoint: multi-count 1, channel not yet
    /// enabled.
    fn chr(ep: &Endpoint) -> regs::Char {
        let mut c = regs::Char(0);
        c.set_mps(ep.max_packet_size);
        c.set_epnum(Epnum::from_bits(ep.number));
        c.set_epdir(match ep.direction {
            Direction::In => Epdir::In,
            Direction::Out => Epdir::Out,
        });
        c.set_eptype(match ep.ep_type {
            EndpointType::Control => CharEptype::Ctrl,
            EndpointType::Isochronous => CharEptype::Isoc,
            EndpointType::Bulk => CharEptype::Bulk,
            EndpointType::Interrupt => CharEptype::Interr,
        });
        c.set_ec(Ec::Transone);
        c.set_devaddr(ep.device_address);
        c
    }

    /// One transfer on channel 0 (buffer DMA), blocking. For IN, `len`
    /// must be a multiple of the endpoint's packet size and the buffer at
    /// `buf` must have room for all of it; a short packet from the device
    /// completes the transfer early. Returns the next PID the endpoint
    /// expects (the data toggle).
    pub fn transfer(
        &mut self,
        ep: &Endpoint,
        pid: Pid,
        buf: u32,
        len: usize,
        timeout_ms: u32,
    ) -> Result<Pid, TransferError> {
        let t = self.start_transfer(ep, pid, buf, len);
        loop {
            if let Some(r) = self.poll_transfer(&t, timeout_ms) {
                return r;
            }
        }
    }

    /// Program and enable channel 0; the core runs the transfer on its own
    /// from here (buffer DMA), so the CPU is free until
    /// [`Host::poll_transfer`] says the channel halted.
    pub fn start_transfer(&mut self, ep: &Endpoint, pid: Pid, buf: u32, len: usize) -> Transfer {
        let mps = ep.max_packet_size as usize;
        let pkts = if len == 0 { 1 } else { len.div_ceil(mps) as u16 };
        let hc = self.hc0();
        hc.int().write(|w| w.0 = 0xFFFF_FFFF);
        hc.intmsk().write_value(regs::Intmsk(0)); // polled: no propagation
        hc.tsiz().write(|w| {
            w.set_xfersize(len as u32);
            w.set_pktcnt(pkts);
            w.set_pid(PacPid::from_bits(pid.to_bits()));
        });
        hc.dma().write_value(buf);
        let chr = Self::chr(ep);
        let mut c = chr;
        c.set_chena(true);
        hc.char().write_value(c);
        Transfer { chr, start: now() }
    }

    /// Non-blocking completion check for an armed transfer: None while it
    /// is still running and within `timeout_ms` of its start.
    pub fn poll_transfer(&mut self, t: &Transfer, timeout_ms: u32) -> Option<Result<Pid, TransferError>> {
        let hc = self.hc0();
        let ints = hc.int().read();
        if !ints.chhltd() {
            if !elapsed_ms(t.start, timeout_ms) {
                return None;
            }
            // Deadline: request a halt and give the core a moment to
            // return the channel.
            let mut c = t.chr;
            c.set_chena(true);
            c.set_chdis(true);
            hc.char().write_value(c);
            wait_for(|| hc.int().read().chhltd(), 5);
            return Some(Err(TransferError::Timeout));
        }
        if ints.xfercompl() {
            return Some(Ok(Pid::from_bits(hc.tsiz().read().pid().to_bits())));
        }
        Some(Err(if ints.stall() {
            TransferError::Stall
        } else if ints.xacterr() {
            TransferError::Transaction
        } else if ints.bblerr() {
            TransferError::Babble
        } else if ints.datatglerr() {
            TransferError::DataToggle
        } else if ints.ahberr() {
            TransferError::Ahb
        } else if ints.frmovrun() {
            TransferError::FrameOverrun
        } else {
            TransferError::Unknown
        }))
    }

    /// Bytes the running transfer has not yet moved (HCTSIZ.XferSize
    /// counts down per packet written out of the RxFIFO).
    pub fn bytes_remaining(&self) -> usize {
        self.hc0().tsiz().read().xfersize() as usize
    }
}

impl<'d> Drop for Host<'d> {
    fn drop(&mut self) {
        self.p.power_down();
    }
}
