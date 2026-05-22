//! Engine configuration: device, threading, model directory.
//!
//! Surgically extracted from the legacy monolithic engine crate for Phase 3.8 Phase A.
//! Pure POD: `Device` + thread counts + path. R2 Phase 2 peer-convergence
//! concession (3-way agreement; concrete + dep-direction-clean). Phase B can
//! wrap or extend this with `GpuEngineConfig` without breaking the rlib API.

use std::path::PathBuf;

use crate::device::Device;

/// Engine configuration. Thread pool settings are immutable after creation
/// (ORT constraint — `OrtEnv` configures thread pools once per process).
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Compute device: CPU or CUDA with device index.
    pub device: Device,
    /// ORT inter-op parallelism threads. Default: 1.
    pub inter_threads: u32,
    /// ORT intra-op parallelism threads. Default: 4 (CPU) / 1 (GPU).
    pub intra_threads: u32,
    /// Base directory for ID-based model resolution (`{model_dir}/{id}/manifest.toml`).
    pub model_dir: PathBuf,
}

impl EngineConfig {
    /// Create config with defaults for the given device and model directory.
    ///
    /// Thread defaults:
    /// - `Device::Cpu` / `Device::Auto`: intra-threads = min(available_parallelism, 8).
    ///   Auto uses the same CPU-optimized default because ORT thread pools are
    ///   configured before EP resolution — if CUDA EP fails and falls back to CPU,
    ///   single-threaded inference would be a severe performance regression.
    ///   Extra threads are harmless on GPU (ORT CPU thread pool is lazy-init).
    /// - `Device::Cuda`: intra-threads = 1 (GPU kernels don't use CPU intra-threads).
    pub fn new(device: Device, model_dir: impl Into<PathBuf>) -> Self {
        let cpu_threads = std::thread::available_parallelism()
            .map(|n| (n.get() as u32).min(8))
            .unwrap_or(4);
        let intra = match &device {
            Device::Cpu | Device::Auto => cpu_threads,
            Device::Cuda(_) => 1,
        };
        Self {
            device,
            inter_threads: 1,
            intra_threads: intra,
            model_dir: model_dir.into(),
        }
    }
}

#[cfg(test)]
mod phase_a_r1_engine_config_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    #[test]
    fn cuda_device_uses_one_intra_thread() {
        let cfg = EngineConfig::new(Device::Cuda(0), "/tmp");
        assert_eq!(
            cfg.intra_threads, 1,
            "GPU configs must use 1 intra-thread; ORT GPU kernels don't use CPU intra threads"
        );
    }

    #[test]
    fn cuda_device_with_nonzero_index_still_uses_one_intra_thread() {
        let cfg = EngineConfig::new(Device::Cuda(7), "/tmp");
        assert_eq!(cfg.intra_threads, 1);
    }

    #[test]
    fn cpu_device_caps_intra_threads_at_eight_and_at_least_one() {
        let cfg = EngineConfig::new(Device::Cpu, "/tmp");
        assert!(
            cfg.intra_threads >= 1,
            "intra_threads must be >= 1, got {}",
            cfg.intra_threads
        );
        assert!(
            cfg.intra_threads <= 8,
            "intra_threads must be capped at 8, got {}",
            cfg.intra_threads
        );
    }

    #[test]
    fn auto_device_matches_cpu_intra_threads() {
        // Doc rationale: Auto must use CPU-optimized defaults so that if CUDA EP
        // fails and falls back to CPU, we don't end up single-threaded.
        let cpu_cfg = EngineConfig::new(Device::Cpu, "/tmp");
        let auto_cfg = EngineConfig::new(Device::Auto, "/tmp");
        assert_eq!(
            cpu_cfg.intra_threads, auto_cfg.intra_threads,
            "Auto must inherit CPU intra-thread default for safe fallback"
        );
    }

    #[test]
    fn inter_threads_always_one_regardless_of_device() {
        // ORT inter-op = 1 is a project-wide invariant in EngineConfig::new.
        for d in [Device::Auto, Device::Cpu, Device::Cuda(0), Device::Cuda(3)] {
            let cfg = EngineConfig::new(d.clone(), "/tmp");
            assert_eq!(cfg.inter_threads, 1, "inter_threads != 1 for {d:?}");
        }
    }

    #[test]
    fn model_dir_accepts_str_string_pathbuf_and_path_ref() {
        // model_dir is `impl Into<PathBuf>` — verify all common call sites work.
        let from_str = EngineConfig::new(Device::Cpu, "/tmp/m");
        let from_string = EngineConfig::new(Device::Cpu, String::from("/tmp/m"));
        let from_pathbuf = EngineConfig::new(Device::Cpu, PathBuf::from("/tmp/m"));
        let from_pathref = EngineConfig::new(Device::Cpu, Path::new("/tmp/m").to_path_buf());

        let expected = PathBuf::from("/tmp/m");
        assert_eq!(from_str.model_dir, expected);
        assert_eq!(from_string.model_dir, expected);
        assert_eq!(from_pathbuf.model_dir, expected);
        assert_eq!(from_pathref.model_dir, expected);
    }

    #[test]
    fn engine_config_clone_preserves_all_fields() {
        // Clone derive sanity — important since EngineConfig is passed around
        // by value at FFI/PyO3 boundaries.
        let original = EngineConfig::new(Device::Cuda(2), "/tmp/m");
        let cloned = original.clone();
        assert_eq!(cloned.device, original.device);
        assert_eq!(cloned.inter_threads, original.inter_threads);
        assert_eq!(cloned.intra_threads, original.intra_threads);
        assert_eq!(cloned.model_dir, original.model_dir);
    }
}
