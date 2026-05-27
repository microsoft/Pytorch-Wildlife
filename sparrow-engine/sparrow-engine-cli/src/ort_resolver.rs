//! Self-contained ORT dylib discovery for the `spe` / `spe-gpu` CLI tarballs (RP-4).
//!
//! The CLI binaries are built with `ort/load-dynamic` (matching the Python wheel
//! per RP-3 / 2026-05-23), which means the `ort` crate `dlopen`s
//! `libonnxruntime` at runtime rather than DT_NEEDED-linking it at process
//! load. With no env override `ort` falls back to a bare platform name
//! (`libonnxruntime.so` / `libonnxruntime.dylib` / `onnxruntime.dll`); ORT's
//! own release tarballs don't ship the unversioned symlink, so an
//! unaided startup would dlopen-fail.
//!
//! RP-4 (Path B, tarball CLI) ships the CLI as:
//!
//! ```text
//! spe-<version>-<platform>/
//! ├── bin/spe(.exe)                              ← this binary
//! ├── lib/libonnxruntime.so.X.Y.Z                ← bundled
//! │  (or libonnxruntime.X.Y.Z.dylib / onnxruntime.dll)
//! │  GPU adds libonnxruntime_providers_cuda.so + _providers_shared.so
//! └── ...
//! ```
//!
//! At startup we resolve the bundle root from `current_exe()`, locate the
//! versioned ORT dylib in `<bundle_root>/lib/`, and set `ORT_DYLIB_PATH`
//! before any engine call. For the GPU flavor on Linux we also prepend
//! `<bundle_root>/lib` to `LD_LIBRARY_PATH` so ORT can `dlopen` its
//! CUDA provider sidecar next to `libonnxruntime.so`.
//!
//! Silent fallthrough on every error path: if `current_exe()` fails, if the
//! tarball layout isn't present (e.g. `cargo run` from source, or a system
//! package that uses a different layout), or if `ORT_DYLIB_PATH` is already
//! set by the user, this function is a no-op and the existing env-var /
//! `ort` crate defaults remain authoritative.
//!
//! Port of `_discover_ort_dylib()` from
//! `sparrow-engine-python/python/sparrow_engine/__init__.py:149-201` (RP-3).

use std::cmp::Ordering;
use std::env;
#[cfg(all(feature = "gpu", target_os = "linux"))]
use std::ffi::OsString;
use std::path::{Path, PathBuf};

/// Set `ORT_DYLIB_PATH` (and on Linux GPU bundles, prepend `LD_LIBRARY_PATH`)
/// when the binary is running from inside an RP-4 tarball layout. Idempotent
/// and side-effect-free if no tarball layout is detected.
pub fn init_ort_env() {
    // Respect existing user override.
    if env::var_os("ORT_DYLIB_PATH").is_some_and(|v| !v.is_empty()) {
        return;
    }

    let Some(lib_dir) = find_bundle_lib_dir() else {
        return;
    };

    if let Some(ort_dylib) = find_ort_dylib(&lib_dir) {
        // env::set_var is safe in Rust edition 2021 (no unsafe block
        // required); the 2024 edition marks it unsafe with the contract
        // "no other thread is observing the environment". We satisfy that
        // contract by calling this at the very top of main() before
        // tracing init, before clap parsing, before any tokio runtime,
        // and before any engine crate call — no other Rust thread exists
        // at this point. When this crate migrates to edition 2024,
        // wrap these calls in `unsafe { ... }` and the comment becomes
        // the safety invariant.
        env::set_var("ORT_DYLIB_PATH", &ort_dylib);
    }

    // GPU on Linux needs ORT to find its provider sidecars
    // (libonnxruntime_providers_cuda.so / _providers_shared.so) next to
    // libonnxruntime.so. ORT loads these via dlopen with a bare name, so
    // they must be on LD_LIBRARY_PATH. CPU bundles have no providers and
    // skip this step. macOS arm64 + Windows have no CUDA path in our
    // build matrix, so this is Linux-only.
    #[cfg(all(feature = "gpu", target_os = "linux"))]
    prepend_library_path(&lib_dir);
}

/// Walk up from `current_exe()` to locate the tarball's `lib/` directory.
///
/// Expected layout: `<root>/bin/<exe>` ↔ `<root>/lib/`.
/// Returns `None` if the layout doesn't match (dev `cargo run`, system
/// package that scatters bin and lib differently, etc.).
fn find_bundle_lib_dir() -> Option<PathBuf> {
    let exe = env::current_exe().ok()?;
    // Canonicalize to follow symlinks (e.g. brew's `bin/spe` → `libexec/.../spe`).
    let exe = exe.canonicalize().ok()?;
    let bin_dir = exe.parent()?;
    let root = bin_dir.parent()?;
    let lib_dir = root.join("lib");
    lib_dir.is_dir().then_some(lib_dir)
}

/// Find the highest-versioned `libonnxruntime` in `lib_dir` for the current
/// platform. Returns `None` if no matching file exists.
fn find_ort_dylib(lib_dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<OrtDylibCandidate> = Vec::new();
    let entries = std::fs::read_dir(lib_dir).ok()?;
    for entry in entries.flatten() {
        if let Some(candidate) = ort_dylib_candidate(entry.path()) {
            candidates.push(candidate);
        }
    }
    candidates.sort_by(compare_ort_candidates);
    candidates.pop().map(|candidate| candidate.path)
}

#[derive(Debug)]
struct OrtDylibCandidate {
    path: PathBuf,
    version: Option<Vec<u32>>,
    is_symlink: bool,
}

fn ort_dylib_candidate(path: PathBuf) -> Option<OrtDylibCandidate> {
    let name = path.file_name().and_then(|n| n.to_str())?;
    let version = parse_ort_dylib_version(name)?;
    let is_symlink = path
        .symlink_metadata()
        .map(|m| m.is_symlink())
        .unwrap_or(false);
    if !path.is_file() {
        return None;
    }
    Some(OrtDylibCandidate {
        path,
        version,
        is_symlink,
    })
}

fn compare_ort_candidates(a: &OrtDylibCandidate, b: &OrtDylibCandidate) -> Ordering {
    match (&a.version, &b.version) {
        (Some(a_version), Some(b_version)) => a_version.cmp(b_version),
        (Some(_), None) => Ordering::Greater,
        (None, Some(_)) => Ordering::Less,
        (None, None) => Ordering::Equal,
    }
    .then_with(|| (!a.is_symlink).cmp(&(!b.is_symlink)))
    .then_with(|| a.path.cmp(&b.path))
}

/// Platform-specific filename prefixes/suffixes for ORT's shared library.
#[cfg(test)]
fn matches_ort_dylib_name(name: &str) -> bool {
    parse_ort_dylib_version(name).is_some()
}

fn parse_ort_dylib_version(name: &str) -> Option<Option<Vec<u32>>> {
    if cfg!(target_os = "windows") {
        return name.eq_ignore_ascii_case("onnxruntime.dll").then_some(None);
    }
    if cfg!(target_os = "macos") {
        if name == "libonnxruntime.dylib" {
            return Some(None);
        }
        let version = name
            .strip_prefix("libonnxruntime.")?
            .strip_suffix(".dylib")?;
        return parse_numeric_version(version).map(Some);
    }
    if name == "libonnxruntime.so" {
        return Some(None);
    }
    let version = name.strip_prefix("libonnxruntime.so.")?;
    parse_numeric_version(version).map(Some)
}

fn parse_numeric_version(version: &str) -> Option<Vec<u32>> {
    let mut parts = Vec::new();
    for part in version.split('.') {
        if part.is_empty() || !part.bytes().all(|b| b.is_ascii_digit()) {
            return None;
        }
        parts.push(part.parse().ok()?);
    }
    (!parts.is_empty()).then_some(parts)
}

/// Prepend `lib_dir` to `LD_LIBRARY_PATH` (GPU + Linux only). Idempotent —
/// only prepends if `lib_dir` is not already the first segment.
#[cfg(all(feature = "gpu", target_os = "linux"))]
fn prepend_library_path(lib_dir: &Path) {
    let existing = env::var_os("LD_LIBRARY_PATH").unwrap_or_default();
    if existing.is_empty() {
        // See ORT_DYLIB_PATH set_var safety comment above.
        env::set_var("LD_LIBRARY_PATH", lib_dir.as_os_str());
        return;
    }
    // Skip if lib_dir is already first.
    if let Some(first) = env::split_paths(&existing).next() {
        if first == lib_dir {
            return;
        }
    }
    let mut combined: OsString = lib_dir.as_os_str().to_owned();
    combined.push(":");
    combined.push(&existing);
    // See ORT_DYLIB_PATH set_var safety comment above.
    env::set_var("LD_LIBRARY_PATH", &combined);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_versioned_linux_dylib() {
        let cases_match = [
            "libonnxruntime.so",
            "libonnxruntime.so.1",
            "libonnxruntime.so.1.25.1",
            "libonnxruntime.so.1.26.0",
        ];
        let cases_no_match = [
            "libonnxruntime.so.bak",
            "libonnxruntime.so.old",
            "libonnxruntime_providers_cuda.so",
            "libonnxruntime_providers_shared.so",
            "libfoo.so",
            "onnxruntime.dll",
        ];
        for c in cases_match {
            // The function answers per-cfg; on Linux all 4 above should match.
            // On non-Linux builds the test still compiles; assertions only
            // fire on the right platform. Skip otherwise.
            if cfg!(target_os = "linux") {
                assert!(matches_ort_dylib_name(c), "expected match for {c}");
            }
        }
        for c in cases_no_match {
            if cfg!(target_os = "linux") {
                assert!(!matches_ort_dylib_name(c), "expected no match for {c}");
            }
        }
    }

    #[test]
    fn matches_macos_dylib() {
        let cases_match = [
            "libonnxruntime.dylib",
            "libonnxruntime.1.dylib",
            "libonnxruntime.1.25.1.dylib",
        ];
        let cases_no_match = ["libonnxruntime.so", "libfoo.dylib", "onnxruntime.dll"];
        if cfg!(target_os = "macos") {
            for c in cases_match {
                assert!(matches_ort_dylib_name(c), "expected match for {c}");
            }
            for c in cases_no_match {
                assert!(!matches_ort_dylib_name(c), "expected no match for {c}");
            }
        }
    }

    #[test]
    fn matches_windows_dll() {
        if cfg!(target_os = "windows") {
            assert!(matches_ort_dylib_name("onnxruntime.dll"));
            assert!(matches_ort_dylib_name("OnnxRuntime.DLL")); // case-insensitive
            assert!(!matches_ort_dylib_name("libonnxruntime.so"));
            assert!(!matches_ort_dylib_name("libonnxruntime.dylib"));
        }
    }

    #[test]
    fn fallthrough_when_no_lib_dir() {
        // Synthesize a tempdir with `bin/fake-spe` but NO `lib/` dir. The
        // resolver should return None.
        let tmp = tempfile::tempdir().unwrap();
        let bin = tmp.path().join("bin");
        std::fs::create_dir(&bin).unwrap();
        let fake_exe = bin.join("fake-spe");
        std::fs::write(&fake_exe, b"").unwrap();

        // Can't easily override `current_exe()` in a unit test, so exercise
        // the lower-level path via find_ort_dylib instead. Confirms the
        // "no candidates" branch returns None.
        assert!(find_ort_dylib(tmp.path()).is_none());
    }

    #[test]
    fn picks_highest_versioned_dylib() {
        if !cfg!(target_os = "linux") {
            return; // glob shape is platform-specific
        }
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path();
        std::fs::write(lib.join("libonnxruntime.so.1.9.0"), b"").unwrap();
        std::fs::write(lib.join("libonnxruntime.so.1.25.0"), b"").unwrap();
        std::fs::write(lib.join("libonnxruntime.so.1.25.1"), b"").unwrap();
        std::fs::write(lib.join("libonnxruntime_providers_cuda.so"), b"").unwrap();

        let picked = find_ort_dylib(lib).expect("expected a candidate");
        assert!(
            picked
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with("1.25.1"),
            "expected highest version, got {}",
            picked.display()
        );
    }

    #[test]
    fn rejects_malformed_linux_dylib_names_and_directories() {
        if !cfg!(target_os = "linux") {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path();
        std::fs::write(lib.join("libonnxruntime.so.1.25.1"), b"").unwrap();
        std::fs::write(lib.join("libonnxruntime.so.bak"), b"").unwrap();
        std::fs::create_dir(lib.join("libonnxruntime.so.99.0.0")).unwrap();

        let picked = find_ort_dylib(lib).expect("expected a candidate");
        assert!(
            picked
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .ends_with("1.25.1"),
            "expected valid numeric dylib, got {}",
            picked.display()
        );
    }

    #[cfg(unix)]
    #[test]
    fn accepts_valid_symlink_and_rejects_dangling_symlink() {
        if !cfg!(target_os = "linux") {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let lib = tmp.path();
        let target = lib.join("actual-onnxruntime");
        std::fs::write(&target, b"").unwrap();
        std::os::unix::fs::symlink(&target, lib.join("libonnxruntime.so.1.25.1")).unwrap();
        std::os::unix::fs::symlink(lib.join("missing"), lib.join("libonnxruntime.so.9.99.0"))
            .unwrap();

        let picked = find_ort_dylib(lib).expect("expected a valid symlink candidate");
        assert_eq!(
            picked.file_name().unwrap().to_str().unwrap(),
            "libonnxruntime.so.1.25.1"
        );
    }
}
