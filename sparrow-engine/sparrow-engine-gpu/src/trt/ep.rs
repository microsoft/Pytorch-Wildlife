//! TensorRT execution-provider policy helpers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cudarc::driver::CudaContext;
use ort::ep::cuda::ConvAlgorithmSearch;
use ort::ep::ExecutionProviderDispatch;
use serde::Serialize;
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{TrtConfig, TrtPrecision};

use crate::trt::cache::{
    cache_file_stale, hex_sha256, prepare_trt_cache_dir, trt_cache_dir, trt_cache_key,
    trt_cache_root_from_env, TrtCacheKeyInput, TRT_CACHE_ENV,
};

const CUDA_VERSION_FOR_CACHE: &str = "cuda-12080";
const ORT_VERSION_FOR_CACHE: &str = "ort-2.0.0-rc.12-api-24";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct GpuIdentity {
    pub(crate) name: String,
    pub(crate) sm_major: i32,
    pub(crate) sm_minor: i32,
}

impl GpuIdentity {
    pub(crate) fn from_context(ctx: &Arc<CudaContext>) -> Result<Self> {
        let name = ctx
            .name()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.name: {e}")))?;
        let (sm_major, sm_minor) = ctx
            .compute_capability()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.compute_capability: {e}")))?;
        Ok(Self {
            name,
            sm_major,
            sm_minor,
        })
    }

    fn cache_identity(&self) -> String {
        format!("{}-sm{}.{}", self.name, self.sm_major, self.sm_minor)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TrtProviderKind {
    TensorRt,
    Cuda,
    Cpu,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum TrtPolicyDecision {
    EnvDisabled,
    UnsupportedSm,
    NotOptedIn,
    TensorRtEnabled,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TrtProviderPlan {
    pub(crate) decision: TrtPolicyDecision,
    pub(crate) providers: Vec<TrtProviderKind>,
}

#[derive(Debug, Clone)]
pub(crate) struct CudaEpConfig {
    pub(crate) device_id: i32,
    pub(crate) compute_stream: Option<*mut ()>,
    pub(crate) conv_algorithm_search: Option<ConvAlgorithmSearch>,
}

impl CudaEpConfig {
    pub(crate) const fn new(device_id: i32) -> Self {
        Self {
            device_id,
            compute_stream: None,
            conv_algorithm_search: None,
        }
    }

    pub(crate) const fn with_compute_stream(mut self, stream: *mut ()) -> Self {
        self.compute_stream = Some(stream);
        self
    }

    pub(crate) fn with_conv_algorithm_search(mut self, search: ConvAlgorithmSearch) -> Self {
        self.conv_algorithm_search = Some(search);
        self
    }
}

pub(crate) struct TrtEpBuilder<'a> {
    model_id: &'a str,
    trt: Option<&'a TrtConfig>,
    gpu: &'a GpuIdentity,
    cuda: CudaEpConfig,
    onnx_path: &'a Path,
    manifest_cache_material: &'a str,
}

impl<'a> TrtEpBuilder<'a> {
    pub(crate) fn new(
        model_id: &'a str,
        trt: Option<&'a TrtConfig>,
        gpu: &'a GpuIdentity,
        cuda: CudaEpConfig,
        onnx_path: &'a Path,
        manifest_cache_material: &'a str,
    ) -> Self {
        Self {
            model_id,
            trt,
            gpu,
            cuda,
            onnx_path,
            manifest_cache_material,
        }
    }

    pub(crate) fn execution_providers(&self) -> Result<Vec<ExecutionProviderDispatch>> {
        let env_disabled =
            trt_disabled_env_is_set(std::env::var("SPARROW_ENGINE_TRT_DISABLE").ok().as_deref());
        let libs_probe = find_tensorrt_runtime();
        let plan = decide_trt_provider_order(
            self.trt,
            self.gpu.sm_major,
            self.gpu.sm_minor,
            env_disabled,
            libs_probe.present,
            self.model_id,
            &self.gpu.name,
        )?;

        match plan.decision {
            TrtPolicyDecision::EnvDisabled => {
                tracing::info!("TRT disabled via SPARROW_ENGINE_TRT_DISABLE");
            }
            TrtPolicyDecision::UnsupportedSm => {
                tracing::warn!(
                    "GPU {} is SM {}.{}; TensorRT requires SM 7.5+, using CUDA EP",
                    self.gpu.name,
                    self.gpu.sm_major,
                    self.gpu.sm_minor
                );
            }
            TrtPolicyDecision::NotOptedIn => {
                tracing::info!(model_id = self.model_id, "TRT not opted in by manifest");
            }
            TrtPolicyDecision::TensorRtEnabled => {
                let config = self.trt.expect("TensorRT plan requires config");
                let cache = self.cache_dir(config, libs_probe.version.as_deref())?;
                tracing::info!(
                    model_id = self.model_id,
                    precision = ?config.precision,
                    builder_optimization_level = config.builder_optimization_level,
                    engine_hw_compatible = config.engine_hw_compatible,
                    cache_dir = %cache.display(),
                    fallback = "CUDA→CPU",
                    "TRT EP registered"
                );
            }
        }

        let mut providers = Vec::with_capacity(plan.providers.len());
        for provider in plan.providers {
            match provider {
                TrtProviderKind::TensorRt => {
                    let config = self.trt.expect("TensorRT provider requires config");
                    let cache_dir = self.cache_dir(config, libs_probe.version.as_deref())?;
                    providers.push(
                        self.build_trt_provider(config, &cache_dir)
                            .error_on_failure(),
                    );
                }
                TrtProviderKind::Cuda => {
                    providers.push(self.build_cuda_provider().error_on_failure())
                }
                TrtProviderKind::Cpu => providers.push(ort::ep::CPU::default().build()),
            }
        }
        Ok(providers)
    }

    fn build_cuda_provider(&self) -> ExecutionProviderDispatch {
        let mut cuda = ort::ep::CUDA::default().with_device_id(self.cuda.device_id);
        if let Some(search) = self.cuda.conv_algorithm_search.clone() {
            cuda = cuda.with_conv_algorithm_search(search);
        }
        if let Some(stream) = self.cuda.compute_stream {
            // SAFETY: callers pass a CUDA stream owned by the model/session object.
            // The stream outlives the ORT session just like the pre-RP-24 CUDA-only
            // audio path did.
            cuda = unsafe { cuda.with_compute_stream(stream) };
        }
        cuda.build()
    }

    fn build_trt_provider(
        &self,
        config: &TrtConfig,
        cache_dir: &Path,
    ) -> ExecutionProviderDispatch {
        let mut trt = ort::ep::TensorRT::default()
            .with_device_id(self.cuda.device_id)
            .with_engine_cache(true)
            .with_engine_cache_path(cache_dir.display().to_string())
            .with_timing_cache(true)
            .with_timing_cache_path(cache_dir.display().to_string())
            .with_builder_optimization_level(config.builder_optimization_level)
            .with_engine_hw_compatible(config.engine_hw_compatible);

        match config.precision {
            TrtPrecision::Fp32 => {}
            TrtPrecision::Fp16 => trt = trt.with_fp16(true),
            TrtPrecision::Int8 => trt = trt.with_int8(true),
        }
        if let Some(stream) = self.cuda.compute_stream {
            // SAFETY: same lifetime rule as `build_cuda_provider`.
            trt = unsafe { trt.with_compute_stream(stream) };
        }
        if let Some(shapes) = format_profile_shapes(config.profile_min.as_ref()) {
            trt = trt.with_profile_min_shapes(shapes);
        }
        if let Some(shapes) = format_profile_shapes(config.profile_opt.as_ref()) {
            trt = trt.with_profile_opt_shapes(shapes);
        }
        if let Some(shapes) = format_profile_shapes(config.profile_max.as_ref()) {
            trt = trt.with_profile_max_shapes(shapes);
        }
        trt.build()
    }

    fn cache_dir(&self, config: &TrtConfig, trt_version: Option<&str>) -> Result<PathBuf> {
        let onnx_bytes = std::fs::read(self.onnx_path).map_err(SparrowEngineError::Io)?;
        let onnx_hash = hex_sha256(&onnx_bytes);
        let manifest_hash = hex_sha256(self.manifest_cache_material.as_bytes());
        let profile_shapes_json = serde_json::to_string(&ProfileShapesForKey {
            min: &config.profile_min,
            opt: &config.profile_opt,
            max: &config.profile_max,
        })?;
        let key = trt_cache_key(&TrtCacheKeyInput {
            onnx_sha256: onnx_hash,
            manifest_sha256: manifest_hash,
            ort_version: ORT_VERSION_FOR_CACHE.to_string(),
            trt_version: trt_version.unwrap_or("unknown").to_string(),
            cuda_version: CUDA_VERSION_FOR_CACHE.to_string(),
            gpu_identity: self.gpu.cache_identity(),
            profile_shapes_json,
            precision: format!("{:?}", config.precision).to_ascii_lowercase(),
        });
        let root = trt_cache_root_from_env(std::env::var(TRT_CACHE_ENV).ok().as_deref());
        let dir = trt_cache_dir(&root, &key);

        let onnx_mtime = std::fs::metadata(self.onnx_path)
            .and_then(|m| m.modified())
            .map_err(SparrowEngineError::Io)?;
        if cache_dir_has_stale_entries(&dir, onnx_mtime)? {
            let _ = std::fs::remove_dir_all(&dir);
        }
        prepare_trt_cache_dir(&dir, &key.full_hash)?;
        Ok(dir)
    }
}

#[derive(Serialize)]
struct ProfileShapesForKey<'a> {
    min: &'a Option<BTreeMap<String, Vec<i64>>>,
    opt: &'a Option<BTreeMap<String, Vec<i64>>>,
    max: &'a Option<BTreeMap<String, Vec<i64>>>,
}

pub(crate) fn sm_supports_trt(major: i32, minor: i32) -> bool {
    major > 7 || (major == 7 && minor >= 5)
}

pub(crate) fn trt_disabled_env_is_set(value: Option<&str>) -> bool {
    value.is_some_and(|v| !v.trim().is_empty())
}

pub(crate) fn decide_trt_provider_order(
    trt: Option<&TrtConfig>,
    sm_major: i32,
    sm_minor: i32,
    env_disabled: bool,
    trt_libs_present: bool,
    model_id: &str,
    _gpu_name: &str,
) -> Result<TrtProviderPlan> {
    if env_disabled {
        return Ok(TrtProviderPlan {
            decision: TrtPolicyDecision::EnvDisabled,
            providers: vec![TrtProviderKind::Cuda, TrtProviderKind::Cpu],
        });
    }
    if !sm_supports_trt(sm_major, sm_minor) {
        return Ok(TrtProviderPlan {
            decision: TrtPolicyDecision::UnsupportedSm,
            providers: vec![TrtProviderKind::Cuda, TrtProviderKind::Cpu],
        });
    }
    let Some(_config) = trt.filter(|config| config.enabled) else {
        return Ok(TrtProviderPlan {
            decision: TrtPolicyDecision::NotOptedIn,
            providers: vec![TrtProviderKind::Cuda, TrtProviderKind::Cpu],
        });
    };
    if !trt_libs_present {
        return Err(SparrowEngineError::TrtRuntimeMissing(format!(
            "Model {model_id} requires TensorRT but libnvinfer was not found. Install TensorRT 10.x (see docs), or set SPARROW_ENGINE_TRT_DISABLE=1 to run on the CUDA EP."
        )));
    }
    Ok(TrtProviderPlan {
        decision: TrtPolicyDecision::TensorRtEnabled,
        providers: vec![
            TrtProviderKind::TensorRt,
            TrtProviderKind::Cuda,
            TrtProviderKind::Cpu,
        ],
    })
}

fn format_profile_shapes(shapes: Option<&BTreeMap<String, Vec<i64>>>) -> Option<String> {
    let shapes = shapes?;
    if shapes.is_empty() {
        return None;
    }
    Some(
        shapes
            .iter()
            .map(|(name, dims)| {
                let dims = dims
                    .iter()
                    .map(i64::to_string)
                    .collect::<Vec<_>>()
                    .join("x");
                format!("{name}:{dims}")
            })
            .collect::<Vec<_>>()
            .join(","),
    )
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TrtLibProbe {
    present: bool,
    version: Option<String>,
}

fn find_tensorrt_runtime() -> TrtLibProbe {
    let mut dirs: Vec<PathBuf> = std::env::var_os("LD_LIBRARY_PATH")
        .map(|paths| std::env::split_paths(&paths).collect())
        .unwrap_or_default();
    dirs.extend([
        PathBuf::from("/usr/lib/x86_64-linux-gnu"),
        PathBuf::from("/usr/local/lib"),
        PathBuf::from("/usr/lib"),
    ]);
    find_tensorrt_runtime_in_dirs(&dirs)
}

fn find_tensorrt_runtime_in_dirs(dirs: &[PathBuf]) -> TrtLibProbe {
    for dir in dirs {
        for name in ["libnvinfer.so.10", "libnvinfer.so"] {
            let candidate = dir.join(name);
            if candidate.exists() {
                return TrtLibProbe {
                    present: true,
                    version: Some(name.trim_start_matches("libnvinfer.so.").to_string()),
                };
            }
        }
    }
    TrtLibProbe {
        present: false,
        version: None,
    }
}

fn cache_dir_has_stale_entries(dir: &Path, onnx_mtime: std::time::SystemTime) -> Result<bool> {
    if !dir.exists() {
        return Ok(false);
    }
    for entry in std::fs::read_dir(dir).map_err(SparrowEngineError::Io)? {
        let entry = entry.map_err(SparrowEngineError::Io)?;
        let mtime = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .map_err(SparrowEngineError::Io)?;
        if cache_file_stale(Some(mtime), onnx_mtime) {
            return Ok(true);
        }
    }
    Ok(false)
}

pub(crate) fn manifest_cache_material(
    manifest: &sparrow_engine_types::manifest::ModelManifest,
) -> String {
    format!(
        "preprocess={:?};postprocess={:?};precision={:?}",
        manifest.preprocess_method, manifest.postprocess_method, manifest.precision
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{TrtConfig, TrtPrecision};

    fn enabled_trt() -> TrtConfig {
        TrtConfig {
            enabled: true,
            precision: TrtPrecision::Fp16,
            builder_optimization_level: 3,
            engine_hw_compatible: false,
            profile_min: None,
            profile_opt: None,
            profile_max: None,
        }
    }

    #[test]
    fn trt_decision_env_disable_wins() {
        let plan =
            decide_trt_provider_order(Some(&enabled_trt()), 8, 9, true, false, "m", "gpu").unwrap();
        assert_eq!(plan.decision, TrtPolicyDecision::EnvDisabled);
        assert_eq!(
            plan.providers,
            vec![TrtProviderKind::Cuda, TrtProviderKind::Cpu]
        );
    }

    #[test]
    fn trt_decision_sm70_uses_cuda() {
        let plan = decide_trt_provider_order(Some(&enabled_trt()), 7, 0, false, false, "m", "V100")
            .unwrap();
        assert_eq!(plan.decision, TrtPolicyDecision::UnsupportedSm);
        assert_eq!(
            plan.providers,
            vec![TrtProviderKind::Cuda, TrtProviderKind::Cpu]
        );
    }

    #[test]
    fn trt_decision_not_opted_in_uses_cuda() {
        let plan = decide_trt_provider_order(None, 8, 9, false, false, "m", "gpu").unwrap();
        assert_eq!(plan.decision, TrtPolicyDecision::NotOptedIn);
        assert_eq!(
            plan.providers,
            vec![TrtProviderKind::Cuda, TrtProviderKind::Cpu]
        );
    }

    #[test]
    fn trt_decision_enabled_with_libs_uses_trt_first() {
        let plan =
            decide_trt_provider_order(Some(&enabled_trt()), 8, 9, false, true, "m", "gpu").unwrap();
        assert_eq!(plan.decision, TrtPolicyDecision::TensorRtEnabled);
        assert_eq!(
            plan.providers,
            vec![
                TrtProviderKind::TensorRt,
                TrtProviderKind::Cuda,
                TrtProviderKind::Cpu
            ]
        );
    }

    #[test]
    fn trt_decision_enabled_missing_libs_returns_actionable_error() {
        let err =
            decide_trt_provider_order(Some(&enabled_trt()), 8, 9, false, false, "model-a", "gpu")
                .unwrap_err();
        assert!(matches!(err, SparrowEngineError::TrtRuntimeMissing(_)));
        let message = err.to_string();
        assert!(message.contains("Model model-a requires TensorRT"));
        assert!(message.contains("SPARROW_ENGINE_TRT_DISABLE=1"));
    }

    #[test]
    fn profile_shapes_are_ort_formatted_in_key_order() {
        let shapes = BTreeMap::from([
            ("audio".to_string(), vec![1, 1, 224, 90]),
            ("image".to_string(), vec![1, 3, 640, 640]),
        ]);
        assert_eq!(
            format_profile_shapes(Some(&shapes)).unwrap(),
            "audio:1x1x224x90,image:1x3x640x640"
        );
    }
}
