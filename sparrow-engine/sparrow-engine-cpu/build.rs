fn main() {
    // -----------------------------------------------------------------------
    // FFI binding generation (only when `ffi` feature is active)
    // -----------------------------------------------------------------------
    #[cfg(feature = "ffi")]
    {
        // cbindgen: generate sparrow_engine.h for C/C++ consumers
        let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
        let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();
        cbindgen::Builder::new()
            .with_crate(&crate_dir)
            .with_config(config)
            .with_language(cbindgen::Language::C)
            .generate()
            .expect("cbindgen failed to generate sparrow_engine.h")
            .write_to_file("sparrow_engine.h");

        // csbindgen: generate NativeMethods.g.cs for C# P/Invoke consumers
        csbindgen::Builder::default()
            .input_extern_file("src/ffi.rs")
            .csharp_dll_name("sparrow_engine")
            .csharp_namespace("SparrowEngine.Native")
            .csharp_class_name("NativeMethods")
            .generate_csharp_file("NativeMethods.g.cs")
            .expect("csbindgen failed to generate NativeMethods.g.cs");

        // Phase 3.8 Phase A M8 + v2 N10 closure: post-build copy of generated
        // headers into the workspace `sparrow-engine/include/` directory so consumer
        // C/C++/C# tooling can find them at a stable, crate-independent path.
        // mtime-only idempotency check (cheaper than SHA256, sufficient for
        // race-safety in a single-implementer Phase A workflow). Errors are
        // logged via `eprintln!` and dropped — generation already wrote the
        // files into `sparrow-engine-cpu/`, so a copy failure does not break the build.
        {
            use std::path::PathBuf;
            let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
            let workspace_root = PathBuf::from(&manifest_dir).parent().unwrap().to_path_buf();
            let include_dir = workspace_root.join("include");
            let _ = std::fs::create_dir_all(&include_dir);

            for filename in &["sparrow_engine.h", "NativeMethods.g.cs"] {
                let src = PathBuf::from(&manifest_dir).join(filename);
                let dst = include_dir.join(filename);
                let needs_copy = match (std::fs::metadata(&src), std::fs::metadata(&dst)) {
                    (Ok(sm), Ok(dm)) => sm.modified().ok() > dm.modified().ok(),
                    (Ok(_), Err(_)) => true,
                    _ => false,
                };
                if needs_copy {
                    if let Err(e) = std::fs::copy(&src, &dst) {
                        eprintln!(
                            "warning: failed to copy {} to {}: {}",
                            src.display(),
                            dst.display(),
                            e
                        );
                    }
                }
            }
        }
        println!("cargo:rerun-if-changed=sparrow_engine.h");
        println!("cargo:rerun-if-changed=NativeMethods.g.cs");
    }

    // -----------------------------------------------------------------------
    // Symbol visibility for cdylib builds (only when ffi feature produces cdylib)
    // -----------------------------------------------------------------------
    #[cfg(feature = "ffi")]
    {
        let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();

        if target_os == "linux" {
            let map_path = format!("{}/exports.map", manifest_dir);
            println!(
                "cargo:rustc-cdylib-link-arg=-Wl,--version-script={}",
                map_path
            );
        } else if target_os == "windows" {
            let def_path = format!("{}/exports.def", manifest_dir);
            let target_env = std::env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();
            if target_env == "msvc" {
                println!("cargo:rustc-cdylib-link-arg=/DEF:{}", def_path);
            }
        }
    }

    // Rerun if these files change
    println!("cargo:rerun-if-changed=exports.map");
    println!("cargo:rerun-if-changed=exports.def");
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
}
