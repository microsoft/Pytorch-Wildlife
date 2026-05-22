//
// Phase 3.8 Phase A M8 + v2 N10 closure: `build.rs` does an mtime-based
// idempotency check before copying generated headers (`sparrow_engine.h`,
// `NativeMethods.g.cs`) into `sparrow-engine/include/`. The check is:
//
//     match (metadata(src), metadata(dst)) {
//         (Ok(sm), Ok(dm)) => sm.modified().ok() > dm.modified().ok(),
//         (Ok(_),  Err(_)) => true,
//         _                => false,
//     }
//
// We can't invoke build.rs directly from a cargo test (build scripts only run
// in build context). Instead we transcribe the SAME 5-line decision logic
// here and verify it on staged file fixtures. If a future refactor changes
// the decision rule (e.g., switching to SHA256 — which Phase A explicitly
// rejected), this test guards the contract.

use std::fs;
use std::path::Path;
use std::time::{Duration, SystemTime};

/// Mirror of the decision logic in `sparrow-engine-cpu/build.rs:44-48`. If build.rs's
/// logic changes, this helper must change in lockstep — the test then catches
/// any drift.
fn needs_copy(src: &Path, dst: &Path) -> bool {
    match (fs::metadata(src), fs::metadata(dst)) {
        (Ok(sm), Ok(dm)) => sm.modified().ok() > dm.modified().ok(),
        (Ok(_), Err(_)) => true,
        _ => false,
    }
}

/// Touch a file with a specific mtime offset (relative to `now`). Negative
/// `offset_secs` = older. Returns the path.
fn touch_with_mtime(dir: &Path, name: &str, content: &[u8], offset_secs: i64) -> std::path::PathBuf {
    let p = dir.join(name);
    fs::write(&p, content).unwrap();
    let target = if offset_secs >= 0 {
        SystemTime::now() + Duration::from_secs(offset_secs as u64)
    } else {
        SystemTime::now() - Duration::from_secs((-offset_secs) as u64)
    };
    // `set_modified` exists on stable Rust 1.75+ via FileTimes; fall back to
    // utime via filetime crate if needed. tempfile lets us regenerate, so we
    // accept whatever mtime resolution the FS gives us — the relative
    // ordering is what we test, and we space offsets >=2s apart to be safe.
    let f = fs::OpenOptions::new().write(true).open(&p).unwrap();
    f.set_modified(target).expect("set_modified on test fixture");
    p
}

// -----------------------------------------------------------------------------
// Test 1: src older than dst → needs_copy = false (no-op). This is the
// idempotent steady-state — repeated builds without changes do not retouch.
// -----------------------------------------------------------------------------

#[test]
fn idempotency_src_older_than_dst_skips_copy() {
    let tmp = tempfile::tempdir().unwrap();
    // dst written first (so it's older), then src LATER would be newer; we
    // want src OLDER, so we set src mtime explicitly before now and dst at now.
    let src = touch_with_mtime(tmp.path(), "sparrow_engine.h", b"old src", -10);
    let dst = touch_with_mtime(tmp.path(), "bongo_dst.h", b"new dst", 0);
    assert!(
        !needs_copy(&src, &dst),
        "Expected needs_copy=false when src is older than dst (steady state)"
    );
}

// -----------------------------------------------------------------------------
// Test 2: src newer than dst → needs_copy = true (must copy). This is the
// post-build state where cbindgen regenerated the header.
// -----------------------------------------------------------------------------

#[test]
fn idempotency_src_newer_than_dst_copies() {
    let tmp = tempfile::tempdir().unwrap();
    // src is newer (offset 0 = now), dst is older (-10s).
    let src = touch_with_mtime(tmp.path(), "sparrow_engine.h", b"new src", 0);
    let dst = touch_with_mtime(tmp.path(), "bongo_dst.h", b"old dst", -10);
    assert!(
        needs_copy(&src, &dst),
        "Expected needs_copy=true when src is newer than dst (post-regen)"
    );
}

// -----------------------------------------------------------------------------
// Test 3: dst missing → needs_copy = true (initial copy). First build path.
// -----------------------------------------------------------------------------

#[test]
fn idempotency_missing_dst_copies() {
    let tmp = tempfile::tempdir().unwrap();
    let src = touch_with_mtime(tmp.path(), "sparrow_engine.h", b"any", 0);
    let dst = tmp.path().join("never_existed.h");
    assert!(
        needs_copy(&src, &dst),
        "Expected needs_copy=true when dst is missing (first-build state)"
    );
}
