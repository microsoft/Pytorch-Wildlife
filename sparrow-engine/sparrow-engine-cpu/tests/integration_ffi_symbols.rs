//
// Phase 3.8 Phase A S7 closure: when `--features ffi` is on, the cdylib must
// expose all 34 symbols listed in `exports.def`. Without `--features ffi` the
// cdylib still builds but emits zero `sparrow_engine_*` symbols (the `sparrow_engine_*; local: *;`
// filter in `exports.map` plus the absence of `pub mod ffi` produce that).
//
// Two test approaches:
//   1. Compile-time link smoke test — references 5 symbols through the Rust
//      `ffi` module, so the test binary fails to compile if those exports
//      disappear from the rlib's `pub fn ffi::*` surface.
//   2. nm shell-out — reads exports.def, runs `nm -D --defined-only` on
//      `target/release/libsparrow_engine.so`, and asserts every `sparrow_engine_*` symbol is
//      present. SKIPS (no fail) when the cdylib hasn't been built.
//
// Both gated `#[cfg(feature = "ffi")]` so they only compile when the feature
// is on (the FFI module is feature-gated; without it, `sparrow_engine::ffi` doesn't
// exist).

#![cfg(feature = "ffi")]

// -----------------------------------------------------------------------------
// Test 1: link-smoke — 5 sample FFI exports must be reachable through
// `sparrow_engine::ffi`. We don't CALL them (that requires Engine + ORT), only
// verify the Rust compiler can resolve the symbols at compile time.
// -----------------------------------------------------------------------------

#[test]
fn ffi_link_smoke_for_sample_symbols() {
    // We name 5 of the 34 symbols. If any disappear from `sparrow-engine-cpu/src/ffi.rs`
    // (e.g., a refactor accidentally drops `#[no_mangle]` or `pub`), this test
    // fails to compile. We pin a function-pointer reference so the compiler has
    // a reason to resolve them.
    use sparrow_engine::ffi::{
        sparrow_engine_engine_free, sparrow_engine_engine_new, sparrow_engine_free_string,
        sparrow_engine_health, sparrow_engine_last_error,
    };

    let p1 = sparrow_engine_engine_new as *const ();
    let p2 = sparrow_engine_engine_free as *const ();
    let p3 = sparrow_engine_last_error as *const ();
    let p4 = sparrow_engine_free_string as *const ();
    let p5 = sparrow_engine_health as *const ();
    assert!(!p1.is_null());
    assert!(!p2.is_null());
    assert!(!p3.is_null());
    assert!(!p4.is_null());
    assert!(!p5.is_null());
}

// -----------------------------------------------------------------------------
// Test 2: cdylib symbol surface — `nm` shell-out against libsparrow_engine.so. SKIPS
// gracefully when the cdylib hasn't been built (so plain `cargo test` doesn't
// fail; the test only does load-bearing work after `cargo build --release
// --features ffi`).
// -----------------------------------------------------------------------------
//
// IMPORTANT: this test is OS-gated. Linux/macOS use `nm`; Windows uses
// `dumpbin /EXPORTS`. We only implement the Linux path since Phase A is
// Linux-only per the implementation plan; document the Windows TODO inline.

#[cfg(target_os = "linux")]
#[test]
fn cdylib_exports_match_exports_def() {
    use std::path::PathBuf;
    use std::process::Command;

    // Locate the cdylib relative to the workspace target dir. CARGO_MANIFEST_DIR
    // points to sparrow-engine/sparrow-engine-cpu/; the workspace target is one level up.
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_root = manifest_dir.parent().expect("workspace root");
    let cdylib = workspace_root
        .join("target")
        .join("release")
        .join("libsparrow_engine.so");

    if !cdylib.exists() {
        eprintln!(
            "SKIP cdylib_exports_match_exports_def: {:?} not found. \
             Run `cargo build --release --features ffi` from the workspace \
             root before running this test.",
            cdylib
        );
        return;
    }

    // Read exports.def and extract the sparrow_engine_* symbol names.
    let def_path = manifest_dir.join("exports.def");
    let def_content = std::fs::read_to_string(&def_path)
        .unwrap_or_else(|e| panic!("failed to read {:?}: {}", def_path, e));
    let expected: Vec<String> = def_content
        .lines()
        .map(|l| l.trim())
        .filter(|l| l.starts_with("sparrow_engine_"))
        .map(|l| l.to_string())
        .collect();
    assert!(
        !expected.is_empty(),
        "exports.def parsed to zero sparrow_engine_* lines — wrong file?"
    );

    // Run `nm -D --defined-only <cdylib>` and grep for ` T sparrow_engine_`.
    let nm_out = Command::new("nm")
        .args(&["-D", "--defined-only"])
        .arg(&cdylib)
        .output()
        .expect("`nm` not found on PATH — cannot verify exports");
    assert!(
        nm_out.status.success(),
        "nm failed: stderr = {}",
        String::from_utf8_lossy(&nm_out.stderr)
    );
    let stdout = String::from_utf8_lossy(&nm_out.stdout);

    // Each line is `<addr> T <name>` for a public text symbol.
    let actual: std::collections::HashSet<String> = stdout
        .lines()
        .filter_map(|l| {
            let mut parts = l.split_whitespace();
            let _addr = parts.next()?;
            let kind = parts.next()?;
            let name = parts.next()?;
            if kind == "T" && name.starts_with("sparrow_engine_") {
                Some(name.to_string())
            } else {
                None
            }
        })
        .collect();

    // Assert every expected symbol is in the cdylib.
    let mut missing: Vec<&String> = expected.iter().filter(|s| !actual.contains(*s)).collect();
    missing.sort();
    assert!(
        missing.is_empty(),
        "{} symbol(s) declared in exports.def but missing from {:?}: {:?}",
        missing.len(),
        cdylib,
        missing
    );

    // Sanity: count matches (34 per Phase C FFI V2 audio inventory).
    assert_eq!(
        expected.len(),
        34,
        "exports.def line count drifted from Phase C baseline (was 34, now {})",
        expected.len()
    );
}

#[cfg(not(target_os = "linux"))]
#[test]
fn cdylib_exports_match_exports_def() {
    eprintln!(
        "SKIP cdylib_exports_match_exports_def: only implemented for Linux. \
         Windows would use `dumpbin /EXPORTS`; macOS would use `nm -gU`. \
         TODO when those targets are added to Phase A."
    );
}
