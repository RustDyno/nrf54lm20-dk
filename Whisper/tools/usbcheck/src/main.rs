//! Host-side check of the firmware USB protocol code: compiles the real
//! usbproto.rs and replays configuration descriptors (typical stick,
//! full-speed stick, composite device, alternate settings, malformed
//! chains) plus CBW/CSW framing and SCSI CDB golden bytes through it.

// Firmware-only items (enumeration constants) are unused here.
#[allow(dead_code)]
#[path = "../../../firmware/src/usbproto.rs"]
mod proto;

use proto::MscIface;

fn ep(addr: u8, attr: u8, mps: u16) -> [u8; 7] {
    [7, 5, addr, attr, mps as u8, (mps >> 8) as u8, 0]
}

fn iface(num: u8, alt: u8, neps: u8, class: u8, sub: u8, prot: u8) -> [u8; 9] {
    [9, 4, num, alt, neps, class, sub, prot, 0]
}

fn cfg_header(total: u16, value: u8) -> [u8; 9] {
    [9, 2, total as u8, (total >> 8) as u8, 1, value, 0, 0x80, 50]
}

fn build(parts: &[&[u8]]) -> Vec<u8> {
    let mut v: Vec<u8> = parts.iter().flat_map(|p| p.iter().copied()).collect();
    let total = v.len() as u16;
    v[2] = total as u8;
    v[3] = (total >> 8) as u8;
    v
}

fn main() {
    let mut checks = 0;
    let mut check = |name: &str, ok: bool| {
        assert!(ok, "FAILED: {name}");
        checks += 1;
    };

    // 1. Typical high-speed stick: one interface, bulk IN 0x81 / OUT 0x02.
    let hs = build(&[
        &cfg_header(0, 1),
        &iface(0, 0, 2, 0x08, 0x06, 0x50),
        &ep(0x81, 0x02, 512),
        &ep(0x02, 0x02, 512),
    ]);
    check(
        "hs stick",
        proto::parse_config(&hs)
            == Ok(MscIface {
                cfg_value: 1,
                ifnum: 0,
                ep_in: 1,
                ep_out: 2,
                mps_in: 512,
                mps_out: 512,
            }),
    );

    // 2. Full-speed stick, OUT endpoint listed first, different numbers.
    let fs = build(&[
        &cfg_header(0, 1),
        &iface(0, 0, 2, 0x08, 0x06, 0x50),
        &ep(0x02, 0x02, 64),
        &ep(0x84, 0x02, 64),
    ]);
    let m = proto::parse_config(&fs).unwrap();
    check("fs stick", m.ep_in == 4 && m.ep_out == 2 && m.mps_in == 64);

    // 3. Composite: HID interface with an interrupt endpoint first, MSC
    // second; the HID endpoint must not be claimed.
    let composite = build(&[
        &cfg_header(0, 2),
        &iface(0, 0, 1, 0x03, 0x00, 0x00),
        &ep(0x83, 0x03, 8),
        &iface(1, 0, 2, 0x08, 0x06, 0x50),
        &ep(0x81, 0x02, 512),
        &ep(0x02, 0x02, 512),
    ]);
    let m = proto::parse_config(&composite).unwrap();
    check(
        "composite",
        m.cfg_value == 2 && m.ifnum == 1 && m.ep_in == 1 && m.ep_out == 2,
    );

    // 4. Alternate setting 1 of the MSC interface follows alt 0: its
    // endpoints must be ignored (alt 0 already complete ends the scan).
    let alts = build(&[
        &cfg_header(0, 1),
        &iface(0, 0, 2, 0x08, 0x06, 0x50),
        &ep(0x81, 0x02, 512),
        &ep(0x02, 0x02, 512),
        &iface(0, 1, 2, 0x08, 0x06, 0x50),
        &ep(0x85, 0x02, 64),
        &ep(0x06, 0x02, 64),
    ]);
    let m = proto::parse_config(&alts).unwrap();
    check("alt settings", m.ep_in == 1 && m.ep_out == 2 && m.mps_in == 512);

    // 5. Extra class-specific descriptors between interface and endpoints
    // (real sticks put a few there) are skipped by length.
    let extra = build(&[
        &cfg_header(0, 1),
        &iface(0, 0, 2, 0x08, 0x06, 0x50),
        &[4, 0x24, 0, 0], // class-specific filler
        &ep(0x81, 0x02, 512),
        &ep(0x02, 0x02, 512),
    ]);
    check("class filler", proto::parse_config(&extra).is_ok());

    // 6. No MSC interface -> -632; truncated chain -> -631; not a config
    // descriptor -> -630.
    let nomsc = build(&[
        &cfg_header(0, 1),
        &iface(0, 0, 1, 0x03, 0x00, 0x00),
        &ep(0x83, 0x03, 8),
    ]);
    check("no msc", proto::parse_config(&nomsc) == Err(-632));
    let mut trunc = hs.clone();
    trunc[9] = 200; // interface descriptor length claims to run past the buffer
    check("truncated", proto::parse_config(&trunc) == Err(-631));
    check("not a config", proto::parse_config(&[9, 1, 0, 0, 0, 0, 0, 0, 0]) == Err(-630));

    // 7. CBW golden bytes (BOT 1.0, 5.1): READ(10) of 8 blocks at LBA 2.
    let mut cbw = [0u8; 32];
    proto::build_cbw(&mut cbw, 0xDEAD_BEEF, 4096, true, &proto::cdb_read10(2, 8));
    check(
        "cbw",
        cbw[..31]
            == [
                0x55, 0x53, 0x42, 0x43, // "USBC"
                0xEF, 0xBE, 0xAD, 0xDE, // tag LE
                0x00, 0x10, 0x00, 0x00, // 4096 LE
                0x80, 0x00, 0x0A, // IN, LUN 0, 10-byte CDB
                0x28, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x08, 0x00, // READ(10)
                0, 0, 0, 0, 0, 0,
            ],
    );

    // 8. CSW parsing: pass, wrong signature, wrong tag, command failed.
    let mut csw = [0u8; 13];
    csw[0..4].copy_from_slice(b"USBS");
    csw[4..8].copy_from_slice(&0xDEAD_BEEFu32.to_le_bytes());
    check("csw pass", proto::check_csw(&csw, 0xDEAD_BEEF) == Ok(0));
    csw[12] = 1;
    check("csw fail status", proto::check_csw(&csw, 0xDEAD_BEEF) == Ok(1));
    check("csw bad tag", proto::check_csw(&csw, 1) == Err(-635));
    csw[0] = b'X';
    check("csw bad sig", proto::check_csw(&csw, 0xDEAD_BEEF) == Err(-634));
    check("csw short", proto::check_csw(&csw[..12], 0xDEAD_BEEF) == Err(-633));

    // 9. SCSI CDB golden bytes.
    check(
        "write10",
        proto::cdb_write10(0x0102_0304, 0x0506)
            == [0x2A, 0, 0x01, 0x02, 0x03, 0x04, 0, 0x05, 0x06, 0],
    );
    check("tur", proto::cdb_test_unit_ready() == [0; 6]);
    check("sense", proto::cdb_request_sense(18) == [0x03, 0, 0, 0, 18, 0]);
    check("capacity", proto::cdb_read_capacity10()[0] == 0x25);

    // 10. Setup packet golden bytes: GET_DESCRIPTOR(config, 64).
    check(
        "setup",
        proto::setup(0x80, proto::REQ_GET_DESCRIPTOR, proto::DESC_CONFIG, 0, 64)
            == [0x80, 6, 0, 2, 0, 0, 64, 0],
    );

    println!("usbcheck: {checks} checks match the firmware protocol code");
}
