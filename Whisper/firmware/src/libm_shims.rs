//! libm symbols required by the vendored C op-extensions, satisfied from the
//! pure-Rust `libm` crate so we never link newlib. The C side calls these by
//! their standard C names.

#[no_mangle]
pub extern "C" fn expf(x: f32) -> f32 {
    libm::expf(x)
}

#[no_mangle]
pub extern "C" fn exp(x: f64) -> f64 {
    libm::exp(x)
}

#[no_mangle]
pub extern "C" fn roundf(x: f32) -> f32 {
    libm::roundf(x)
}

#[no_mangle]
pub extern "C" fn round(x: f64) -> f64 {
    libm::round(x)
}
