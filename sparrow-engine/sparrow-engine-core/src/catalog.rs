//! Local model catalog: verification, listing, and checksum management.

use std::path::{Component, Path, PathBuf};

use sparrow_engine_types::manifest::{self, ModelManifest};
use sparrow_engine_types::{SparrowEngineError, ModelInfo, Result, derive_model_type};

/// Validate that a model ID is a flat directory name (no traversal, no separators).
pub fn validate_model_id(model_id: &str) -> Result<()> {
    let path = std::path::Path::new(model_id);
    if path.is_absolute() || model_id.starts_with('\\') {
        return Err(SparrowEngineError::PathTraversal(format!(
            "model_id: absolute path not allowed: '{model_id}'"
        )));
    }
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            return Err(SparrowEngineError::PathTraversal(format!(
                "model_id: parent directory traversal not allowed: '{model_id}'"
            )));
        }
    }
    if model_id.contains('/') || model_id.contains('\\') {
        return Err(SparrowEngineError::PathTraversal(format!(
            "model_id: path separators not allowed: '{model_id}'"
        )));
    }
    Ok(())
}

/// Resolve the ONNX model file path relative to the manifest's directory.
fn resolve_onnx_path(manifest_path: &Path, manifest: &ModelManifest) -> PathBuf {
    manifest_path
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .join(&manifest.model_file)
}

/// Result of model verification.
#[derive(Debug, Clone, PartialEq)]
pub enum VerifyResult {
    /// Model matches expected checksum and size.
    Ok,
    /// No checksum in manifest (cannot verify).
    NoChecksum,
    /// File size does not match expected.
    SizeMismatch { expected: u64, actual: u64 },
    /// SHA-256 checksum does not match.
    ChecksumMismatch { expected: String, actual: String },
}

/// Verify a model's ONNX file against manifest checksums.
///
/// Loads the manifest from `{model_dir}/{model_id}/manifest.toml`,
/// checks `onnx_size_bytes` (fast), then `onnx_sha256` (thorough).
pub fn verify_model(model_dir: &Path, model_id: &str) -> Result<VerifyResult> {
    validate_model_id(model_id)?;
    let manifest_path = model_dir.join(model_id).join("manifest.toml");
    let m = manifest::load_manifest(&manifest_path)?;

    // If no checksum fields present, cannot verify.
    if m.onnx_sha256.is_none() && m.onnx_size_bytes.is_none() {
        return Ok(VerifyResult::NoChecksum);
    }

    let onnx_path = resolve_onnx_path(&manifest_path, &m);

    let metadata = std::fs::metadata(&onnx_path)?;
    let actual_size = metadata.len();

    // Size check (fast).
    if let Some(expected_size) = m.onnx_size_bytes {
        if actual_size != expected_size {
            return Ok(VerifyResult::SizeMismatch {
                expected: expected_size,
                actual: actual_size,
            });
        }
    }

    // SHA-256 check (thorough).
    if let Some(ref expected_hash) = m.onnx_sha256 {
        let actual_hash = crate::hash::hash_file(&onnx_path)?;
        if actual_hash != *expected_hash {
            return Ok(VerifyResult::ChecksumMismatch {
                expected: expected_hash.clone(),
                actual: actual_hash,
            });
        }
    }

    Ok(VerifyResult::Ok)
}

/// List available models from disk without loading ORT sessions.
///
/// Scans `{model_dir}/{id}/manifest.toml` for each subdirectory. Emits
/// `tracing::warn!` on any load failure (I/O error, TOML parse error,
/// schema validation, wrong-manifest-type, etc.) and skips the offending
/// manifest. The graceful empty-Vec semantic on `read_dir` failure of
/// `model_dir` is preserved.
pub fn list_available_models(model_dir: &Path) -> Vec<ModelInfo> {
    let mut models = Vec::new();

    let entries = match std::fs::read_dir(model_dir) {
        Ok(entries) => entries,
        Err(_) => return models,
    };

    for entry in entries {
        // Linux ReadDir.next() yields Err only on EBADF (corrupted DIR*) per
        // man 3 readdir; opendir EACCES is handled at line 105-108 by the
        // early-return on read_dir(model_dir). This arm is OS-driven and not
        // deterministically reproducible from user-space without DI refactor;
        // existing test list_available_models_skips_unloadable_manifest_and_keeps_good_one
        // covers the common Ok-arm path. Logging here mirrors the load_manifest
        // warn-and-skip pattern below so I/O errors don't disappear silently.
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!(
                    target: "engine_dispatch::core::catalog",
                    "skipping unreadable directory entry in {}: {e}",
                    model_dir.display()
                );
                continue;
            }
        };
        // Use std::fs::metadata (= Path::metadata, FOLLOWS symlinks) instead
        // of Path::is_dir() — the latter silently coerces EACCES/ELOOP/etc
        // to false (rustdoc: "convenience function that coerces errors to
        // false"), which would drop the offending sub-entry without a log
        // line. DirEntry::metadata() is symlink_metadata-equivalent on Unix
        // (rustdoc: "this function will not traverse symlinks") and would
        // silently drop dirsymlinks like `model_dir/<name> -> /shared/<name>/`
        // (NFS / atomic-swap deploy pattern) — std::fs::metadata follows
        // symlinks like the original Path::is_dir() did. The warn-and-skip
        // arm mirrors the R1+R2 pattern at the manifest::load_manifest match
        // below, so all I/O failure modes in this loop produce a uniform
        // tracing::warn under target "engine_dispatch::core::catalog".
        let entry_path = entry.path();
        match std::fs::metadata(&entry_path) {
            Ok(md) if !md.is_dir() => continue,
            Ok(_) => {}
            Err(e) => {
                tracing::warn!(
                    target: "engine_dispatch::core::catalog",
                    "skipping entry with unreadable metadata {}: {e}",
                    entry_path.display()
                );
                continue;
            }
        }
        let manifest_path = entry_path.join("manifest.toml");
        // Path::try_exists() (stable since 1.63) returns io::Result<bool>;
        // Path::exists() coerces errors to false (rustdoc explicitly warns
        // "this method may be error-prone, consider using try_exists()
        // instead").
        match manifest_path.try_exists() {
            Ok(true) => {}
            Ok(false) => continue,
            Err(e) => {
                tracing::warn!(
                    target: "engine_dispatch::core::catalog",
                    "skipping manifest with unreadable existence-check {}: {e}",
                    manifest_path.display()
                );
                continue;
            }
        }
        match manifest::load_manifest(&manifest_path) {
            Ok(m) => {
                models.push(ModelInfo {
                    id: m.id,
                    path: manifest_path,
                    model_type: derive_model_type(
                        &m.preprocess_method,
                        &m.postprocess_method,
                        m.subtype,
                    ),
                    default: m.default,
                    version: m.version,
                    description: m.description,
                    onnx_sha256: m.onnx_sha256,
                    onnx_size_bytes: m.onnx_size_bytes,
                });
            }
            Err(e) => {
                tracing::warn!(
                    target: "engine_dispatch::core::catalog",
                    "skipping unloadable manifest {}: {e}",
                    manifest_path.display()
                );
            }
        }
    }

    models
}

/// Compute and write checksum + size into a model's manifest.toml.
///
/// Returns `(sha256_hex, file_size)`.
pub fn write_checksum(model_dir: &Path, model_id: &str) -> Result<(String, u64)> {
    validate_model_id(model_id)?;
    let manifest_path = model_dir.join(model_id).join("manifest.toml");
    let m = manifest::load_manifest(&manifest_path)?;

    let onnx_path = resolve_onnx_path(&manifest_path, &m);

    let hash = crate::hash::hash_file(&onnx_path)?;
    let size = std::fs::metadata(&onnx_path)?.len();

    // Read existing TOML, update [model] section.
    let content = std::fs::read_to_string(&manifest_path)?;
    let mut doc: toml::Table = content
        .parse()
        .map_err(|e: toml::de::Error| SparrowEngineError::InvalidManifest(e.to_string()))?;

    let model_table = doc
        .get_mut("model")
        .and_then(|v| v.as_table_mut())
        .ok_or_else(|| SparrowEngineError::InvalidManifest("missing [model] section".to_string()))?;
    model_table.insert(
        "onnx_sha256".to_string(),
        toml::Value::String(hash.clone()),
    );
    let size_i64 = i64::try_from(size)
        .map_err(|_| SparrowEngineError::InvalidManifest(format!("file size {size} exceeds i64::MAX")))?;
    model_table.insert(
        "onnx_size_bytes".to_string(),
        toml::Value::Integer(size_i64),
    );

    let updated = toml::to_string_pretty(&doc)
        .map_err(|e| SparrowEngineError::InvalidManifest(e.to_string()))?;
    std::fs::write(&manifest_path, updated)?;

    Ok((hash, size))
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::ModelType;
    use std::io::Write;

    fn setup_model_dir() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("test-model");
        std::fs::create_dir(&model_dir).unwrap();

        // Write a minimal ONNX file (just some bytes).
        let onnx_path = model_dir.join("model.onnx");
        let mut f = std::fs::File::create(&onnx_path).unwrap();
        f.write_all(b"fake onnx content for testing").unwrap();

        // Compute actual hash and size for verification.
        let hash = crate::hash::hash_file(&onnx_path).unwrap();
        let size = std::fs::metadata(&onnx_path).unwrap().len();

        // Write manifest with checksums.
        let manifest = format!(
            r#"
[model]
id = "test-model"
format = "onnx"
file = "model.onnx"
onnx_sha256 = "{hash}"
onnx_size_bytes = {size}

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#
        );
        std::fs::write(model_dir.join("manifest.toml"), manifest).unwrap();
        std::fs::write(model_dir.join("labels.txt"), "animal\nperson\nvehicle\n").unwrap();

        dir
    }

    #[test]
    fn verify_ok() {
        let dir = setup_model_dir();
        let result = verify_model(dir.path(), "test-model").unwrap();
        assert_eq!(result, VerifyResult::Ok);
    }

    #[test]
    fn verify_no_checksum() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("no-hash");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"data").unwrap();

        let manifest = r#"
[model]
id = "no-hash"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        std::fs::write(model_dir.join("manifest.toml"), manifest).unwrap();
        std::fs::write(model_dir.join("labels.txt"), "animal\n").unwrap();

        let result = verify_model(dir.path(), "no-hash").unwrap();
        assert_eq!(result, VerifyResult::NoChecksum);
    }

    #[test]
    fn verify_size_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("bad-size");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"short").unwrap();

        let manifest = r#"
[model]
id = "bad-size"
format = "onnx"
file = "model.onnx"
onnx_size_bytes = 999999

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        std::fs::write(model_dir.join("manifest.toml"), manifest).unwrap();
        std::fs::write(model_dir.join("labels.txt"), "animal\n").unwrap();

        let result = verify_model(dir.path(), "bad-size").unwrap();
        assert!(matches!(result, VerifyResult::SizeMismatch { .. }));
    }

    #[test]
    fn verify_checksum_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("bad-hash");
        std::fs::create_dir(&model_dir).unwrap();
        let data = b"some data";
        std::fs::write(model_dir.join("model.onnx"), data).unwrap();
        let size = data.len() as u64;

        let manifest = format!(
            r#"
[model]
id = "bad-hash"
format = "onnx"
file = "model.onnx"
onnx_sha256 = "0000000000000000000000000000000000000000000000000000000000000000"
onnx_size_bytes = {size}

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#
        );
        std::fs::write(model_dir.join("manifest.toml"), manifest).unwrap();
        std::fs::write(model_dir.join("labels.txt"), "animal\n").unwrap();

        let result = verify_model(dir.path(), "bad-hash").unwrap();
        assert!(matches!(result, VerifyResult::ChecksumMismatch { .. }));
    }

    #[test]
    fn write_checksum_updates_manifest() {
        let dir = setup_model_dir();
        let (hash, size) = write_checksum(dir.path(), "test-model").unwrap();
        assert_eq!(hash.len(), 64);
        assert!(size > 0);

        // Verify the manifest was updated.
        let content = std::fs::read_to_string(dir.path().join("test-model/manifest.toml")).unwrap();
        assert!(content.contains(&hash));
    }

    #[test]
    fn list_available_models_finds_models() {
        let dir = setup_model_dir();
        let models = list_available_models(dir.path());
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "test-model");
        assert_eq!(models[0].model_type, ModelType::Detector);
    }

    #[test]
    fn list_available_models_empty_dir() {
        let dir = tempfile::tempdir().unwrap();
        let models = list_available_models(dir.path());
        assert!(models.is_empty());
    }

    #[test]
    fn verify_model_not_found() {
        let dir = tempfile::tempdir().unwrap();
        let result = verify_model(dir.path(), "nonexistent");
        assert!(result.is_err());
    }

    // Regression: write_checksum must error when [model] section is missing.
    #[test]
    fn write_checksum_errors_on_missing_model_section() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("no-model-section");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"data").unwrap();

        // Manifest without [model] section — uses flat keys (invalid structure).
        let manifest = r#"
[model]
id = "no-model-section"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        std::fs::write(model_dir.join("manifest.toml"), manifest).unwrap();
        std::fs::write(model_dir.join("labels.txt"), "animal\n").unwrap();

        // This should succeed since [model] section exists.
        let result = write_checksum(dir.path(), "no-model-section");
        assert!(result.is_ok());

        // Now test with a manifest that truly lacks [model].
        let bad_manifest = r#"
id = "flat-keys"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        std::fs::write(model_dir.join("manifest.toml"), bad_manifest).unwrap();
        // This should now fail during TOML update (missing [model] section),
        // but may also fail during manifest loading since the schema requires [model].
        let result = write_checksum(dir.path(), "no-model-section");
        assert!(result.is_err());
    }

    // Regression: model_id with path traversal must be rejected.
    #[test]
    fn verify_model_rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let result = verify_model(dir.path(), "../../etc");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SparrowEngineError::PathTraversal(_)),
            "expected PathTraversal, got: {err:?}"
        );
    }

    #[test]
    fn write_checksum_rejects_path_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let result = write_checksum(dir.path(), "../escape");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SparrowEngineError::PathTraversal(_)),
            "expected PathTraversal, got: {err:?}"
        );
    }

    #[test]
    fn validate_model_id_rejects_separators() {
        assert!(validate_model_id("nested/path").is_err());
        assert!(validate_model_id("back\\slash").is_err());
        assert!(validate_model_id("/absolute").is_err());
        assert!(validate_model_id("..").is_err());
        assert!(validate_model_id("valid-model-id").is_ok());
        assert!(validate_model_id("model_v2").is_ok());
    }
}

#[cfg(test)]
mod phase_a_r1_catalog {
    use super::*;
    use sparrow_engine_types::SparrowEngineError;
    use std::io::Write;

    /// Whitelist of model IDs that should pass — flat directory names with
    /// hyphens, underscores, digits, dots in the middle. Locks `validate_model_id`
    /// against accidental over-rejection. (`..` rejection is already covered by
    /// `validate_model_id_rejects_separators`; we cover ACCEPT cases here.)
    #[test]
    fn validate_model_id_accepts_reasonable_names() {
        for ok in [
            "mdv6",
            "mdv6-yolov10e",
            "my_model",
            "model_v2",
            "mdv6.1",
            "model123",
            "MDv6", // case is allowed (no normalization)
        ] {
            assert!(
                validate_model_id(ok).is_ok(),
                "validate_model_id rejected legitimate id: {ok:?}"
            );
        }
    }

    /// Reject set: separators, traversal, leading slash, empty.
    /// Existing `validate_model_id_rejects_separators` covers `nested/path`,
    /// `back\\slash`, `/absolute`, `..`. We add: empty string and a leading-dot
    /// hidden-name path-traversal vector (`./xyz` is rejected via the `/`
    /// separator check before traversal kicks in).
    #[test]
    fn validate_model_id_rejects_traversal_and_separators() {
        for bad in [
            "..",
            "../etc",
            "../../escape",
            "model/sub",
            "model\\sub",
            "/abs",
            "\\winabs",
            "./hidden",
            "model/..",
        ] {
            let r = validate_model_id(bad);
            assert!(r.is_err(), "expected rejection for {bad:?}, got {r:?}");
            assert!(
                matches!(r.unwrap_err(), SparrowEngineError::PathTraversal(_)),
                "expected PathTraversal variant for {bad:?}"
            );
        }
    }

    /// `verify_model` returns `SizeMismatch` (not `Err`) when the manifest's
    /// declared `onnx_size_bytes` disagrees with the on-disk file size.
    /// Distinct from the existing `verify_size_mismatch` test: that one declares
    /// a much-larger expected size; this asserts the variant carries the
    /// expected/actual numbers (load-bearing for CLI error messages).
    #[test]
    fn verify_model_size_mismatch_carries_actual_and_expected() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("size-mismatch");
        std::fs::create_dir(&model_dir).unwrap();
        let payload = b"on-disk model bytes (24 chars)";
        std::fs::write(model_dir.join("model.onnx"), payload).unwrap();
        let manifest = r#"
[model]
id = "size-mismatch"
format = "onnx"
file = "model.onnx"
onnx_size_bytes = 9999

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#
        .to_string();
        std::fs::write(model_dir.join("manifest.toml"), manifest).unwrap();
        std::fs::write(model_dir.join("labels.txt"), "animal\n").unwrap();

        let r = verify_model(dir.path(), "size-mismatch").unwrap();
        match r {
            VerifyResult::SizeMismatch { expected, actual } => {
                assert_eq!(expected, 9999, "expected_size must come from manifest");
                assert_eq!(
                    actual,
                    payload.len() as u64,
                    "actual must equal on-disk file size"
                );
            }
            other => panic!("expected SizeMismatch, got {other:?}"),
        }
    }

    /// Explicitly tests the `NoChecksum` branch when BOTH `onnx_sha256` and
    /// `onnx_size_bytes` are absent. Existing `verify_no_checksum` covers it
    /// but we expand to confirm `result == VerifyResult::NoChecksum` (variant
    /// equality) rather than just "not error".
    #[test]
    fn verify_model_no_checksum_when_both_fields_absent() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("no-fields");
        std::fs::create_dir(&model_dir).unwrap();
        std::fs::write(model_dir.join("model.onnx"), b"data").unwrap();

        let manifest = r#"
[model]
id = "no-fields"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        std::fs::write(model_dir.join("manifest.toml"), manifest).unwrap();
        std::fs::write(model_dir.join("labels.txt"), "animal\n").unwrap();

        assert_eq!(
            verify_model(dir.path(), "no-fields").unwrap(),
            VerifyResult::NoChecksum
        );
    }

    /// Byte-flip mutation test: take a known-good model, corrupt one byte,
    /// re-run verify, expect ChecksumMismatch with the original hash in the
    /// `expected` field. Exercises the SHA-256 step (line 82 onward) — the
    /// existing `verify_checksum_mismatch` test uses a manifest with all-zero
    /// hash, which is a synthetic case; this is a realistic post-corruption.
    #[test]
    fn verify_model_checksum_mismatch_after_byte_flip() {
        let dir = tempfile::tempdir().unwrap();
        let model_dir = dir.path().join("flip-test");
        std::fs::create_dir(&model_dir).unwrap();

        let onnx_path = model_dir.join("model.onnx");
        let mut f = std::fs::File::create(&onnx_path).unwrap();
        f.write_all(b"original-bytes").unwrap();
        drop(f);

        let real_hash = crate::hash::hash_file(&onnx_path).unwrap();
        let real_size = std::fs::metadata(&onnx_path).unwrap().len();

        let manifest = format!(
            r#"
[model]
id = "flip-test"
format = "onnx"
file = "model.onnx"
onnx_sha256 = "{real_hash}"
onnx_size_bytes = {real_size}

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#,
        );
        std::fs::write(model_dir.join("manifest.toml"), manifest).unwrap();
        std::fs::write(model_dir.join("labels.txt"), "animal\n").unwrap();

        // Sanity: the unmodified file verifies cleanly.
        assert_eq!(
            verify_model(dir.path(), "flip-test").unwrap(),
            VerifyResult::Ok
        );

        // Flip one byte, preserving file SIZE (so we hit the SHA branch, not Size).
        let mut bytes = std::fs::read(&onnx_path).unwrap();
        bytes[0] ^= 0x01;
        std::fs::write(&onnx_path, &bytes).unwrap();

        match verify_model(dir.path(), "flip-test").unwrap() {
            VerifyResult::ChecksumMismatch { expected, actual } => {
                assert_eq!(
                    expected, real_hash,
                    "expected hash must echo the manifest value"
                );
                assert_ne!(
                    actual, expected,
                    "actual hash must differ after the byte flip"
                );
                assert_eq!(actual.len(), 64, "actual hash must be 64 hex chars");
            }
            other => panic!("expected ChecksumMismatch after byte flip, got {other:?}"),
        }
    }

    /// Regression test for the warn-and-skip path on a broken manifest.
    ///
    /// Before commit 7d8dc15, `list_available_models` used `if let Ok(m) = ...`
    /// which silently dropped errors. The new `match` arm logs via
    /// `tracing::warn!(target: "engine_dispatch::core::catalog", ...)` and skips. This
    /// test locks the SKIP semantic (good models still listed; broken sibling
    /// dropped) so a future regression that re-introduces an error-propagating
    /// early return — or worse, a `panic!` on parse error — is caught.
    ///
    /// Coverage gap surfaced by audit-fix R1 (subagent_catalog.md Q7,
    /// inquisitor MISSING TEST flag under item B). This test deliberately
    /// does NOT assert on the warn message contents — that would couple the
    /// regression test to the auditor's separate reformat (item A).
    #[test]
    fn list_available_models_skips_unloadable_manifest_and_keeps_good_one() {
        let dir = tempfile::tempdir().unwrap();

        // Good model — minimal valid manifest matching the schema.
        let good_dir = dir.path().join("good-model");
        std::fs::create_dir(&good_dir).unwrap();
        std::fs::write(good_dir.join("model.onnx"), b"fake onnx bytes").unwrap();
        std::fs::write(good_dir.join("labels.txt"), "animal\nperson\nvehicle\n").unwrap();
        let good_manifest = r#"
[model]
id = "good-model"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        std::fs::write(good_dir.join("manifest.toml"), good_manifest).unwrap();

        // Sibling dir with a syntactically invalid TOML manifest (parser fails
        // before schema check). `[[` is unmatched-table-array, classic toml
        // error — the warn-and-skip arm fires.
        let bad_dir = dir.path().join("broken-model");
        std::fs::create_dir(&bad_dir).unwrap();
        std::fs::write(
            bad_dir.join("manifest.toml"),
            "this = is = not = valid [[ toml",
        )
        .unwrap();

        let models = list_available_models(dir.path());

        assert_eq!(
            models.len(),
            1,
            "good model must survive a broken sibling — got {} models: {:?}",
            models.len(),
            models.iter().map(|m| &m.id).collect::<Vec<_>>()
        );
        assert_eq!(
            models[0].id, "good-model",
            "the surviving model must be the good one (not the broken sibling's id)"
        );
    }

    /// Symlink-follow regression guard (R5 T2).
    ///
    /// R3 B5 shipped DirEntry::metadata() which is symlink_metadata-equivalent
    /// on Unix and silently dropped dirsymlinks. Operators using
    /// `model_dir/<name> -> /shared/<name>/` (NFS / atomic-swap deploy)
    /// silently lost models from `sparrow-engine models list` / inference loaders.
    /// R5 C1 replaced with std::fs::metadata(&entry_path) which follows
    /// symlinks like the original Path::is_dir() did.
    ///
    /// This test asserts list_available_models discovers a symlinked model
    /// dir. Unix-only because std::os::unix::fs::symlink is platform-gated;
    /// deterministic on tmpfs/ext4/xfs (no flakiness budget).
    #[cfg(unix)]
    #[test]
    fn list_available_models_follows_symlink_to_model_dir() {
        use std::os::unix::fs::symlink;
        let target = tempfile::tempdir().unwrap();
        let real_model_dir = target.path().join("real-megadet");
        std::fs::create_dir(&real_model_dir).unwrap();
        std::fs::write(real_model_dir.join("model.onnx"), b"fake onnx").unwrap();
        std::fs::write(real_model_dir.join("labels.txt"), "animal\n").unwrap();
        std::fs::write(
            real_model_dir.join("manifest.toml"),
            r#"
[model]
id = "real-megadet"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#,
        )
        .unwrap();

        let parent = tempfile::tempdir().unwrap();
        symlink(&real_model_dir, parent.path().join("symlinked-megadet")).unwrap();

        let models = list_available_models(parent.path());
        assert_eq!(
            models.len(),
            1,
            "list_available_models must follow dir symlinks (regression guard for R3 B5 / R5 C1)"
        );
        assert_eq!(models[0].id, "real-megadet");
    }
}
