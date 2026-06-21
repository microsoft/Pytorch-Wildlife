//! TensorRT engine cache helpers.

use std::path::{Path, PathBuf};
use std::time::SystemTime;

use serde::Serialize;
use sha2::{Digest, Sha256};

pub const TRT_CACHE_ENV: &str = "SPARROW_ENGINE_TRT_CACHE_DIR";

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TrtCacheKeyInput {
    pub onnx_sha256: String,
    pub manifest_sha256: String,
    pub ort_version: String,
    pub trt_version: String,
    pub cuda_version: String,
    pub gpu_identity: String,
    pub profile_shapes_json: String,
    pub precision: String,
    pub builder_optimization_level: u8,
    pub engine_hw_compatible: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrtCacheKey {
    pub full_hash: String,
    pub short_key: String,
}

pub fn trt_cache_key(input: &TrtCacheKeyInput) -> TrtCacheKey {
    let encoded = serde_json::to_vec(input).expect("TrtCacheKeyInput serialization is infallible");
    let full_hash = hex_sha256(&encoded);
    let short_key = full_hash[..16].to_string();
    TrtCacheKey {
        full_hash,
        short_key,
    }
}

pub fn hex_sha256(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    format!("{:x}", hasher.finalize())
}

pub fn trt_cache_root_from_env(env_override: Option<&str>) -> PathBuf {
    if let Some(path) = env_override.filter(|s| !s.trim().is_empty()) {
        return PathBuf::from(path);
    }
    default_trt_cache_root()
}

pub fn trt_cache_dir(root: &Path, key: &TrtCacheKey) -> PathBuf {
    root.join(&key.short_key)
}

pub fn prepare_trt_cache_dir(dir: &Path, full_hash: &str) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    // TODO(RP-24 follow-up): honor a future --rebuild-trt-cache CLI flag by
    // clearing this key directory before ORT's TensorRT EP sees it.
    std::fs::write(
        dir.join("sparrow-cache-key.json"),
        format!("{{\"full_hash\":\"{full_hash}\"}}\n"),
    )
}

pub fn cache_file_stale(cache_file_mtime: Option<SystemTime>, onnx_mtime: SystemTime) -> bool {
    match cache_file_mtime {
        Some(cache_time) => cache_time < onnx_mtime,
        None => false,
    }
}

#[cfg(target_os = "windows")]
fn default_trt_cache_root() -> PathBuf {
    std::env::var_os("LOCALAPPDATA")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join("sparrow-engine")
        .join("trt-cache")
}

#[cfg(not(target_os = "windows"))]
fn default_trt_cache_root() -> PathBuf {
    std::env::var_os("XDG_CACHE_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".cache")))
        .unwrap_or_else(|| PathBuf::from("."))
        .join("sparrow-engine")
        .join("trt-cache")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_input() -> TrtCacheKeyInput {
        TrtCacheKeyInput {
            onnx_sha256: "onnx".into(),
            manifest_sha256: "manifest".into(),
            ort_version: "ort".into(),
            trt_version: "trt".into(),
            cuda_version: "cuda".into(),
            gpu_identity: "sm89-RTX".into(),
            profile_shapes_json: "{}".into(),
            precision: "fp16".into(),
            builder_optimization_level: 3,
            engine_hw_compatible: false,
        }
    }

    #[test]
    fn trt_cache_key_is_deterministic_and_short_key_is_16_hex() {
        let a = trt_cache_key(&sample_input());
        let b = trt_cache_key(&sample_input());
        assert_eq!(a, b);
        assert_eq!(a.short_key.len(), 16);
        assert!(a.full_hash.starts_with(&a.short_key));
    }

    #[test]
    fn trt_cache_key_changes_when_input_changes() {
        let mut changed = sample_input();
        let original = trt_cache_key(&changed);
        changed.precision = "fp32".into();
        assert_ne!(original, trt_cache_key(&changed));
    }

    #[test]
    fn trt_cache_key_changes_when_trt_engine_settings_change() {
        let mut changed = sample_input();
        let original = trt_cache_key(&changed);

        changed.builder_optimization_level = 4;
        assert_ne!(original, trt_cache_key(&changed));

        changed = sample_input();
        changed.engine_hw_compatible = true;
        assert_ne!(original, trt_cache_key(&changed));
    }

    #[test]
    fn trt_cache_key_changes_when_gpu_identity_changes() {
        let mut changed = sample_input();
        let original = trt_cache_key(&changed);
        changed.gpu_identity = "sm75-T4".into();
        assert_ne!(original, trt_cache_key(&changed));
    }

    #[test]
    fn trt_cache_root_honors_env_override() {
        assert_eq!(
            trt_cache_root_from_env(Some("project-cache/trt")),
            PathBuf::from("project-cache/trt")
        );
    }

    #[test]
    fn cache_file_stale_follows_mtime_order() {
        let onnx = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(20);
        let old_cache = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(10);
        let new_cache = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(30);
        assert!(cache_file_stale(Some(old_cache), onnx));
        assert!(!cache_file_stale(Some(new_cache), onnx));
        assert!(!cache_file_stale(None, onnx));
    }
}
