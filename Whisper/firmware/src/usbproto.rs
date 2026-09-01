//! USB protocol builders/parsers for the host mass-storage path: control
//! setup packets, configuration-descriptor walking, bulk-only-transport
//! CBW/CSW framing, and the SCSI command blocks a USB stick needs.
//!
//! No hardware access in this file: tools/usbcheck compiles it on the host
//! and replays real captured descriptors and framing vectors through it.

/// Standard requests used during enumeration.
pub const REQ_GET_DESCRIPTOR: u8 = 6;
pub const REQ_SET_ADDRESS: u8 = 5;
pub const REQ_SET_CONFIGURATION: u8 = 9;
pub const REQ_CLEAR_FEATURE: u8 = 1;

pub const DESC_DEVICE: u16 = 1 << 8;
pub const DESC_CONFIG: u16 = 2 << 8;

pub const FEATURE_ENDPOINT_HALT: u16 = 0;

/// Build the 8-byte SETUP packet (USB 2.0, 9.3).
pub fn setup(bm_req_type: u8, b_request: u8, w_value: u16, w_index: u16, w_length: u16) -> [u8; 8] {
    [
        bm_req_type,
        b_request,
        w_value as u8,
        (w_value >> 8) as u8,
        w_index as u8,
        (w_index >> 8) as u8,
        w_length as u8,
        (w_length >> 8) as u8,
    ]
}

/// What enumeration needs to know about the stick's mass-storage function.
#[derive(Clone, Copy, Default, PartialEq, Debug)]
pub struct MscIface {
    pub cfg_value: u8,
    pub ifnum: u8,
    /// Endpoint numbers with the direction bit stripped.
    pub ep_in: u8,
    pub ep_out: u8,
    pub mps_in: u16,
    pub mps_out: u16,
}

/// Walk a configuration descriptor (the full wTotalLength read) and find
/// the first mass-storage bulk-only interface (class 8, subclass 6,
/// protocol 0x50, alternate setting 0) with both bulk endpoints.
/// Composite devices interleave other interfaces; endpoints are assigned
/// to the interface descriptor that precedes them.
pub fn parse_config(d: &[u8]) -> Result<MscIface, i32> {
    if d.len() < 9 || d[1] != 2 {
        return Err(-630);
    }
    let mut m = MscIface {
        cfg_value: d[5],
        ..Default::default()
    };
    let mut in_msc = false;
    let mut p = d[0] as usize;
    while p + 2 <= d.len() {
        let len = d[p] as usize;
        if len < 2 || p + len > d.len() {
            return Err(-631); // malformed descriptor chain
        }
        match d[p + 1] {
            4 if len >= 9 => {
                // interface: bInterfaceNumber, bAlternateSetting, class/sub/proto
                if in_msc && m.ep_in != 0 && m.ep_out != 0 {
                    break; // already complete; a later interface ends the scan
                }
                in_msc = d[p + 3] == 0
                    && d[p + 5] == 0x08
                    && d[p + 6] == 0x06
                    && d[p + 7] == 0x50;
                if in_msc {
                    m.ifnum = d[p + 2];
                    m.ep_in = 0;
                    m.ep_out = 0;
                }
            }
            5 if len >= 7 && in_msc => {
                // endpoint: bEndpointAddress, bmAttributes, wMaxPacketSize
                let addr = d[p + 2];
                let bulk = d[p + 3] & 0x03 == 0x02;
                let mps = u16::from_le_bytes([d[p + 4], d[p + 5]]) & 0x7FF;
                if bulk && addr & 0x80 != 0 && m.ep_in == 0 {
                    m.ep_in = addr & 0x0F;
                    m.mps_in = mps;
                } else if bulk && addr & 0x80 == 0 && m.ep_out == 0 {
                    m.ep_out = addr & 0x0F;
                    m.mps_out = mps;
                }
            }
            _ => {}
        }
        p += len;
    }
    if m.ep_in == 0 || m.ep_out == 0 {
        return Err(-632); // no bulk-only mass-storage interface found
    }
    Ok(m)
}

// --- Bulk-only transport (USB MSC BOT 1.0) -----------------------------------

pub const CBW_LEN: usize = 31;
pub const CSW_LEN: usize = 13;
const CBW_SIG: u32 = 0x4342_5355; // "USBC"
const CSW_SIG: u32 = 0x5342_5355; // "USBS"

/// Fill a 31-byte command block wrapper. `dir_in` is the data-phase
/// direction (ignored by devices when `data_len` is 0).
pub fn build_cbw(buf: &mut [u8; 32], tag: u32, data_len: u32, dir_in: bool, cb: &[u8]) {
    buf.fill(0);
    buf[0..4].copy_from_slice(&CBW_SIG.to_le_bytes());
    buf[4..8].copy_from_slice(&tag.to_le_bytes());
    buf[8..12].copy_from_slice(&data_len.to_le_bytes());
    buf[12] = if dir_in { 0x80 } else { 0x00 };
    buf[13] = 0; // LUN 0
    buf[14] = cb.len() as u8;
    buf[15..15 + cb.len()].copy_from_slice(cb);
}

/// Validate a command status wrapper against the CBW tag. Returns the
/// status byte: 0 passed, 1 command failed (read the sense data), 2 phase
/// error (reset recovery required).
pub fn check_csw(raw: &[u8], tag: u32) -> Result<u8, i32> {
    if raw.len() < CSW_LEN {
        return Err(-633);
    }
    if u32::from_le_bytes(raw[0..4].try_into().unwrap()) != CSW_SIG {
        return Err(-634);
    }
    if u32::from_le_bytes(raw[4..8].try_into().unwrap()) != tag {
        return Err(-635);
    }
    Ok(raw[12])
}

// --- SCSI command blocks (SBC-3 subset every stick implements) ---------------

pub fn cdb_test_unit_ready() -> [u8; 6] {
    [0; 6]
}

pub fn cdb_request_sense(alloc: u8) -> [u8; 6] {
    [0x03, 0, 0, 0, alloc, 0]
}

pub fn cdb_inquiry(alloc: u8) -> [u8; 6] {
    [0x12, 0, 0, 0, alloc, 0]
}

pub fn cdb_read_capacity10() -> [u8; 10] {
    [0x25, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

pub fn cdb_read10(lba: u32, count: u16) -> [u8; 10] {
    let l = lba.to_be_bytes();
    let c = count.to_be_bytes();
    [0x28, 0, l[0], l[1], l[2], l[3], 0, c[0], c[1], 0]
}

pub fn cdb_write10(lba: u32, count: u16) -> [u8; 10] {
    let l = lba.to_be_bytes();
    let c = count.to_be_bytes();
    [0x2A, 0, l[0], l[1], l[2], l[3], 0, c[0], c[1], 0]
}
