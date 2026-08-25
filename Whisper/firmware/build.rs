use std::env;
use std::path::PathBuf;

// Axon buffer sizes (bytes). Measured needs across ALL 116 submodels:
// FC tiles 24576, conv1 29856, conv2 tile 58112, decoder token blobs less;
// psum 0 everywhere. Sized to the maximum plus slack -- the freed RAM holds
// the standalone decoder's self-attention KV cache. Keep in sync with
// src/main.rs AND tools/make-blob.sh (blobs bake the static_assert).
const INTERLAYER_BUFFER_SIZE: &str = "65536";
const PSUM_BUFFER_SIZE: &str = "4096";

// Nordic driver blob + open C wrappers + headers, shared with the npu/ crate
// (vendor/ there is populated from the sdk-edge-ai add-on; see npu/README.md).
const VENDOR: &str = "../../npu/vendor";

fn main() {
    let crate_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let vendor = crate_dir.join(VENDOR);
    let vendor_inc = vendor.join("include");
    let vendor_drv = vendor.join("include/drivers");

    // Make memory.x available to the linker (cortex-m-rt's link.x includes it).
    std::fs::copy("memory.x", out.join("memory.x")).unwrap();
    // cortex-m-rt's "device" feature INCLUDEs a device.x (normally from a PAC).
    // Our vector table in main.rs is self-contained, so an empty one is fine.
    std::fs::write(
        out.join("device.x"),
        "/* interrupt vectors resolved via __INTERRUPTS in src/main.rs */\n",
    )
    .unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");
    // Build id: printed at boot and embedded in the SD image (fwid asset)
    // so a stale card announces itself instead of crashing mysteriously.
    let build_id = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();
    println!("cargo:rustc-env=BUILD_ID={build_id:x}");
    println!("cargo:rerun-if-changed=src");

    // Compile the open-source Axon driver wrappers (high-level inference API +
    // CPU op extensions) plus the variadic-printf glue. No model is linked at
    // build time: models arrive at runtime as slot blobs.
    cc::Build::new()
        .compiler("arm-none-eabi-gcc")
        .flag("-ffreestanding")
        .opt_level_str("s")
        .include(&vendor_inc)
        .include(&vendor_drv)
        .define("NRF_AXON_INTERLAYER_BUFFER_SIZE", INTERLAYER_BUFFER_SIZE)
        .define("NRF_AXON_PSUM_BUFFER_SIZE", PSUM_BUFFER_SIZE)
        .file(vendor.join("src/nrf_axon_nn_infer.c"))
        .file(vendor.join("src/nrf_axon_nn_op_extensions.c"))
        .file("csrc/glue.c")
        .compile("axon_src");

    // Link Nordic's pre-compiled low-level driver blob.
    println!(
        "cargo:rustc-link-search=native={}",
        vendor.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=nrf-axon-driver-internal-fpu");

    // Generate Rust bindings for the API we call.
    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg(format!("-I{}", vendor_inc.display()))
        .clang_arg(format!("-I{}", vendor_drv.display()))
        .clang_arg("--target=thumbv8m.main-none-eabihf")
        .clang_arg("-ffreestanding")
        .clang_arg(format!(
            "-DNRF_AXON_INTERLAYER_BUFFER_SIZE={INTERLAYER_BUFFER_SIZE}"
        ))
        .clang_arg(format!("-DNRF_AXON_PSUM_BUFFER_SIZE={PSUM_BUFFER_SIZE}"))
        .use_core()
        .default_enum_style(bindgen::EnumVariation::NewType {
            is_bitfield: false,
            is_global: false,
        })
        .allowlist_function("nrf_axon_.*")
        .allowlist_type("nrf_axon_.*")
        .allowlist_var("NRF_AXON_[A-Z].*")
        .generate()
        .expect("bindgen failed to generate Axon bindings");
    bindings
        .write_to_file(out.join("bindings.rs"))
        .expect("failed to write bindings.rs");

    println!("cargo:rerun-if-changed=wrapper.h");
    println!("cargo:rerun-if-changed=csrc/glue.c");
}
