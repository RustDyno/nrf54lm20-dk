//! Cross-language check of the 4-bit weight coding: compiles the real
//! firmware unpacker (firmware/src/q4.rs) and replays the vectors that
//! model/quant4.py emitted (run `pixi run python quant4.py` first).

#[path = "../../../firmware/src/q4.rs"]
mod q4;

fn main() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../model/out/q4vec.bin");
    let v = std::fs::read(path).expect("q4vec.bin (run model/quant4.py)");
    let mut p = 0usize;
    let mut rd = |p: &mut usize| -> usize {
        let x = u32::from_le_bytes(v[*p..*p + 4].try_into().unwrap());
        *p += 4;
        x as usize
    };
    let count = rd(&mut p);
    for i in 0..count {
        let n = rd(&mut p);
        let amax = &v[p..p + n / q4::G];
        p += n / q4::G;
        let nibs = &v[p..p + n / 2];
        p += n / 2;
        let expect = &v[p..p + n];
        p += n;
        let mut dst = vec![0i8; n];
        q4::unpack(amax, nibs, &mut dst);
        let got: &[u8] =
            unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const u8, n) };
        assert_eq!(got, expect, "vector {i}");
    }
    assert_eq!(p, v.len());
    println!("q4check: {count} vectors match the firmware unpacker bit for bit");
}
