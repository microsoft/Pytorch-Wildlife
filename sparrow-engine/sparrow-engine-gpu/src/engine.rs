//! Phase 3.8 Phase C Wave 1 — `sparrow-engine-gpu::Engine` dispatch glue.
//!
//! `sparrow-engine-gpu`'s [`Engine`] mirrors `sparrow_engine_cpu::engine::Engine`'s public
//! surface so consumer crates (`sparrow-engine-server`, `sparrow-engine-cli`,
//! `sparrow-engine-python`) can swap between flavors via compile-time feature
//! dispatch. The `SparrowEngineApi` trait insertion (`final_design.md §3`
//! footnote) stays deferred to Phase B; Wave 1 keeps the concrete struct.
//!
//! # Dispatch shape
//!
//! Each loaded model is wrapped in a [`LoadedModelInner`] enum that holds
//! the per-model GPU pipeline ([`crate::models::yolo::YoloModel`],
//! [`crate::models::classifier::ClassifierModel`],
//! [`crate::models::tiled::TiledModel`],
//! [`crate::models::audio::AudioModel`]). [`Engine::load_model`]
//! dispatches on `derive_model_type(&preprocess, &postprocess, subtype)`
//! to the right per-model `load`. Free functions
//! `sparrow_engine_gpu::detect::detect`, `sparrow_engine_gpu::classify::classify`,
//! `sparrow_engine_gpu::detect_audio::detect_audio`,
//! `sparrow_engine_gpu::pipeline::run_pipeline` accept a [`ModelHandle`] /
//! [`Engine`] and route to the right inner variant.
//!
//! # Engine-shared GPU primitives
//!
//! [`EngineInner`] owns the CUDA primitives that the per-model paths
//! borrow on each call:
//!
//! - `letterbox: LetterboxKernel` — used by [`crate::detect::detect`] for
//!   YOLO models.
//! - `center_crop: CenterCropKernel` — held for forward compat (today's
//!   `ClassifierModel::classify` argument is unused, see its docstring).
//! - `resize: ResizeKernel` — used by [`crate::classify::classify`].
//! - `decoder: Mutex<JpegDecoder>` — used by [`crate::classify::classify`]
//!   to amortise nvjpeg handle creation across calls. (Yolo + Tiled
//!   already cache their own decoders inside the model struct.)
//!
//! Each free fn reaches `EngineInner` via
//! `handle.engine_ref.upgrade().ok_or(SparrowEngineError::EngineFreed)?`,
//! mirroring `sparrow_engine_cpu`'s `Weak<EngineInner>` pattern.
//!
//! # Singleton
//!
//! Mirrors `sparrow_engine_cpu`'s discipline: one [`Engine`] per process, claimed
//! atomically via [`ENGINE_EXISTS`]. The `sparrow-engine-cpu` and `sparrow-engine-gpu`
//! singletons are presently disjoint AtomicBools; Phase C consumer crates
//! pick exactly one engine flavor at compile time, so two flavors can
//! never co-exist in the same process.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use cudarc::driver::CudaContext;
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{self, ModelManifest, PipelineManifest};
use sparrow_engine_types::{derive_model_type, ModelInfo, ModelType};

// Phase 3.8 Phase C Wave 4b: re-export `Device` + `EngineConfig` at the
// `engine::*` path to mirror `sparrow_engine_cpu::engine::{Device, EngineConfig}`
// (`sparrow-engine-cpu/src/engine.rs:28`). Required so consumers (the cdylib FFI
// in `src/ffi.rs`, the sparrow-engine-python dispatch shim, and integration tests)
// can write `engine_dispatch::engine::{Device, EngineConfig}` symmetrically.
pub use sparrow_engine_types::{Device, EngineConfig};

use crate::kernels::center_crop::CenterCropKernel;
use crate::kernels::letterbox::LetterboxKernel;
use crate::kernels::resize::ResizeKernel;
use crate::models::audio::AudioModel;
use crate::models::audio_raw::RawAudioModel;
use crate::models::classifier::{ClassifierModel, JpegDecoder};
use crate::models::tiled::TiledModel;
use crate::models::yolo::YoloModel;

// ---------------------------------------------------------------------------
// Singleton guard
// ---------------------------------------------------------------------------

/// Process-global flag: true if a `sparrow-engine-gpu` Engine instance exists.
///
/// `sparrow-engine-cpu` and `sparrow-engine-gpu` share the ORT singleton in spirit (ORT
/// Environment is process-global, regardless of which EP is active), so
/// downstream consumers treat the two engines as mutually exclusive even
/// though the AtomicBools live in different crates. The Phase C consumer
/// wiring picks one flavor at compile time, so two flavors can never be
/// instantiated simultaneously.
static ENGINE_EXISTS: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// LoadedModel + LoadedModelInner
// ---------------------------------------------------------------------------

/// Per-loaded-model inner state — one variant per GPU model family.
///
/// `Audio` is boxed because [`crate::models::audio::AudioModel`] is the
/// largest variant by ~5× (audio carries an `AudioOrtSession`, cached
/// cuFFT plan map, mel filterbank uploads, and a workspace mutex);
/// inlining it would force every loaded image model to pay the audio-
/// sized stack/heap layout. `clippy::large_enum_variant` flags the
/// disparity, so we box the heaviest variant.
pub(crate) enum LoadedModelInner {
    Yolo(YoloModel),
    Classifier(ClassifierModel),
    Tiled(TiledModel),
    Audio(Box<AudioModel>),
    /// Phase D round 2 B-08: raw-audio classifiers (Perch 2 / perch-v2)
    /// whose ONNX consumes raw f32 samples directly with no mel pipeline.
    /// Held inline (not boxed) because `RawAudioModel` is small (single
    /// Mutex<Session> + ~50 bytes of params) — the
    /// `large_enum_variant` lint that motivated boxing `Audio` does not
    /// apply.
    AudioRaw(RawAudioModel),
}

// SAFETY: every per-model type is `Send + Sync` (each declares its own
// `unsafe impl Send for X` / `unsafe impl Sync for X` in `models/*.rs`).
unsafe impl Send for LoadedModelInner {}
unsafe impl Sync for LoadedModelInner {}

/// One loaded model: dispatch enum + manifest snapshot + label table +
/// liveness flag + manifest path. Cheaply cloneable via `Arc`.
pub(crate) struct LoadedModel {
    pub(crate) manifest: Arc<ModelManifest>,
    pub(crate) labels: Arc<Vec<String>>,
    pub(crate) path: PathBuf,
    pub(crate) active: Arc<AtomicBool>,
    pub(crate) inner: LoadedModelInner,
    /// Unix-millis timestamp of the last `get_model_handle` lookup. Mirrors
    /// `sparrow-engine-cpu`'s `LoadedModel::last_used`. Used by `reap_idle_models` to
    /// identify auto-unload candidates.
    pub(crate) last_used: Arc<AtomicU64>,
}

impl LoadedModel {
    /// Derived model type — used by `loaded_models()` / `model_info()`.
    pub(crate) fn model_type(&self) -> ModelType {
        derive_model_type(
            &self.manifest.preprocess_method,
            &self.manifest.postprocess_method,
            self.manifest.subtype,
        )
    }

    /// Build a [`ModelInfo`] snapshot for `loaded_models()` / `model_info()`.
    /// Single source of truth for the `LoadedModel → ModelInfo` field copy
    /// so adding a new manifest field touches exactly one site.
    pub(crate) fn to_model_info(&self) -> ModelInfo {
        ModelInfo {
            id: self.manifest.id.clone(),
            path: self.path.clone(),
            model_type: self.model_type(),
            default: self.manifest.default,
            version: self.manifest.version.clone(),
            description: self.manifest.description.clone(),
            onnx_sha256: self.manifest.onnx_sha256.clone(),
            onnx_size_bytes: self.manifest.onnx_size_bytes,
        }
    }
}

// SAFETY: every field is itself Send+Sync (POD / Arc / AtomicBool /
// LoadedModelInner above).
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

// ---------------------------------------------------------------------------
// EngineInner
// ---------------------------------------------------------------------------

/// Engine-wide shared state behind `Arc`. Every [`ModelHandle`] holds a
/// [`Weak`] back-pointer so it can detect post-`Drop` use without keeping
/// the engine alive.
pub(crate) struct EngineInner {
    /// CUDA context for the active GPU. Always device 0 today;
    /// multi-GPU support is a future-Phase concern.
    pub(crate) ctx: Arc<CudaContext>,
    /// Resolved device after construction. `Auto` always picks `Cuda(0)`
    /// inside `sparrow-engine-gpu` because the crate only loads when GPU is the
    /// chosen flavor (Phase C consumer wiring decides that upstream).
    pub(crate) resolved_device: Device,
    /// Engine config snapshot.
    pub(crate) config: EngineConfig,
    /// Compiled CUDA letterbox kernel. Used by YOLO dispatch.
    pub(crate) letterbox: LetterboxKernel,
    /// Compiled CUDA center-crop kernel. Held for forward compat; today's
    /// `ClassifierModel::classify` accepts but does not use it (see its
    /// docstring).
    #[allow(dead_code)]
    pub(crate) center_crop: CenterCropKernel,
    /// Compiled CUDA resize kernel. Used by classifier dispatch.
    pub(crate) resize: ResizeKernel,
    /// Cached nvjpeg decoder. Used by classifier dispatch (Yolo + Tiled
    /// already carry their own decoder behind a private `Mutex`).
    pub(crate) decoder: Mutex<JpegDecoder>,
}

// SAFETY: every field is itself Send+Sync (CudaContext is Send+Sync;
// kernels wrap cudarc CudaFunction = Send+Sync; JpegDecoder is wrapped
// in Mutex; POD scalars).
unsafe impl Send for EngineInner {}
unsafe impl Sync for EngineInner {}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The singleton GPU inference engine.
///
/// Only one [`Engine`] may exist per process (ORT Environment is
/// process-global). A second [`Engine::new`] returns
/// [`SparrowEngineError::EngineAlreadyExists`].
pub struct Engine {
    pub(crate) inner: Arc<EngineInner>,
    /// Loaded model handles, keyed by model ID.
    pub(crate) models: RwLock<HashMap<String, Arc<LoadedModel>>>,
    /// Registered pipeline configs, keyed by pipeline ID.
    pub(crate) pipelines: Mutex<HashMap<String, PipelineManifest>>,
    /// Serializes first-load operations to prevent TOCTOU double-load
    /// race in [`Engine::get_or_load_model`]. Mirrors `sparrow-engine-cpu`.
    loading_lock: Mutex<()>,
}

unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

/// Opaque handle to a loaded model.
///
/// Holds an `Arc<LoadedModel>` snapshot so dispatch is safe even after
/// the model is replaced or unloaded. Cheap to clone.
#[derive(Clone)]
pub struct ModelHandle {
    /// Weak reference back to the engine. Fails to upgrade if engine is
    /// dropped.
    pub(crate) engine_ref: Weak<EngineInner>,
    /// Set to false when the model is unloaded.
    pub(crate) active: Arc<AtomicBool>,
    /// Pinned snapshot of the loaded model.
    pub(crate) inner: Arc<LoadedModel>,
    /// Model ID from the manifest.
    model_id: String,
}

unsafe impl Send for ModelHandle {}
unsafe impl Sync for ModelHandle {}

impl std::fmt::Debug for ModelHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ModelHandle")
            .field("model_id", &self.model_id)
            .field("active", &self.active.load(Ordering::Relaxed))
            .field("engine_alive", &self.engine_ref.upgrade().is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Engine implementation
// ---------------------------------------------------------------------------

impl Engine {
    /// Create the singleton GPU engine.
    ///
    /// Initializes a CUDA context on the configured device, compiles the
    /// shared preprocess kernels via NVRTC, builds an engine-level
    /// nvjpeg decoder, and claims the singleton slot.
    pub fn new(config: EngineConfig) -> Result<Self> {
        if ENGINE_EXISTS.swap(true, Ordering::SeqCst) {
            return Err(SparrowEngineError::EngineAlreadyExists);
        }

        // Resolve device. `sparrow-engine-gpu` always lands on `Cuda(_)`; explicit
        // indices pass through unchanged for forward compat. One match
        // produces the ordinal directly; `resolved_device` is then
        // built from it. Exhaustive over `Device` (no wildcard arm) so
        // adding a variant later forces an explicit decision here.
        let ordinal: u32 = match &config.device {
            Device::Auto | Device::Cpu => 0,
            Device::Cuda(n) => *n,
        };
        let resolved_device = Device::Cuda(ordinal);

        // Build engine-shared CUDA primitives. On any failure release
        // the singleton slot before propagating the error.
        let init = move || -> Result<EngineInner> {
            let ctx = CudaContext::new(ordinal as usize).map_err(|e| {
                SparrowEngineError::Ort(format!("CudaContext::new({ordinal}): {e}"))
            })?;
            let letterbox = LetterboxKernel::new(&ctx)?;
            let center_crop = CenterCropKernel::new(&ctx)?;
            let resize = ResizeKernel::new(&ctx)?;
            let decoder = JpegDecoder::new(&ctx)?;
            Ok(EngineInner {
                ctx,
                resolved_device,
                config,
                letterbox,
                center_crop,
                resize,
                decoder: Mutex::new(decoder),
            })
        };
        let inner = init().inspect_err(|_e| {
            ENGINE_EXISTS.store(false, Ordering::SeqCst);
        })?;

        Ok(Engine {
            inner: Arc::new(inner),
            models: RwLock::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            loading_lock: Mutex::new(()),
        })
    }

    /// Borrow the CUDA context. Used by Wave 2/3/4 module wiring; not
    /// part of the public surface that sparrow-engine-cli/python/server consume.
    #[doc(hidden)]
    pub fn cuda_context(&self) -> &Arc<CudaContext> {
        &self.inner.ctx
    }

    /// Returns the resolved device for this engine.
    pub fn active_device(&self) -> &Device {
        &self.inner.resolved_device
    }

    /// Get the engine config.
    pub fn config(&self) -> &EngineConfig {
        &self.inner.config
    }

    // -----------------------------------------------------------------
    // Model loading + unloading
    // -----------------------------------------------------------------

    /// Load a model from a manifest path. Dispatches on the manifest's
    /// `model_type` (derived from preprocess + postprocess + subtype) to
    /// the right per-model GPU pipeline.
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<ModelHandle> {
        let manifest_path = path.as_ref();
        let manifest_owned = manifest::load_manifest(manifest_path)?;

        // Flavor-strict: the gpu flavor runs ONNX models via ORT. The shared loader
        // now also accepts `tflite` manifests (for the mobile LiteRT flavor); reject
        // a non-ONNX format here with a clear error. Mirrors sparrow-engine-cpu.
        if manifest_owned.format != "onnx" {
            return Err(SparrowEngineError::UnsupportedFormat {
                format: manifest_owned.format.clone(),
            });
        }
        let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        let model_id = manifest_owned.id.clone();

        // Load labels (optional — audio binary detector has none).
        let labels = match (&manifest_owned.label_file, &manifest_owned.label_format) {
            (Some(file), Some(fmt)) => {
                let label_path = manifest_dir.join(file);
                manifest::load_labels(&label_path, fmt)?
            }
            _ => Vec::new(),
        };

        // Dispatch on model type.
        let model_type = derive_model_type(
            &manifest_owned.preprocess_method,
            &manifest_owned.postprocess_method,
            manifest_owned.subtype,
        );
        let inner = match model_type {
            ModelType::Detector | ModelType::OverheadDetector => {
                // Tiled vs single-shot dispatch lives inside the GPU
                // model layer: tiled-strategy manifests build a
                // `TiledModel`; single-strategy + heatmap-peaks
                // postprocess (overhead) likewise route through
                // `TiledModel` if the manifest declares
                // `inference.strategy = tiled`. Single-shot YOLO
                // manifests build a `YoloModel`.
                match manifest_owned.inference_strategy {
                    manifest::InferenceStrategy::Tiled { .. } => LoadedModelInner::Tiled(
                        TiledModel::load(&self.inner.ctx, &manifest_owned, manifest_dir)?,
                    ),
                    manifest::InferenceStrategy::Single => LoadedModelInner::Yolo(YoloModel::load(
                        &self.inner.ctx,
                        &manifest_owned,
                        manifest_dir,
                    )?),
                    manifest::InferenceStrategy::SlidingWindow { .. } => {
                        return Err(SparrowEngineError::InvalidManifest(format!(
                            "manifest '{}': sliding_window strategy is reserved for audio models, but model_type = {:?}",
                            manifest_owned.id, model_type
                        )));
                    }
                }
            }
            ModelType::Classifier => LoadedModelInner::Classifier(ClassifierModel::load(
                &self.inner.ctx,
                &manifest_owned,
                manifest_dir,
            )?),
            ModelType::AudioDetector | ModelType::AudioClassifier => {
                // Phase D round 2 B-08: dispatch on preprocess variant.
                // RawAudio (Perch 2) bypasses the mel pipeline entirely;
                // MelSpectrogram is the existing AudioModel path.
                match manifest_owned.preprocess_method {
                    manifest::PreprocessMethod::RawAudio { .. } => {
                        LoadedModelInner::AudioRaw(RawAudioModel::load_from_manifest(
                            &self.inner.ctx,
                            &manifest_owned,
                            manifest_dir,
                        )?)
                    }
                    _ => LoadedModelInner::Audio(Box::new(AudioModel::load_from_manifest(
                        &self.inner.ctx,
                        &manifest_owned,
                        manifest_dir,
                    )?)),
                }
            }
        };

        let manifest = Arc::new(manifest_owned);
        let labels = Arc::new(labels);
        let active = Arc::new(AtomicBool::new(true));
        let last_used = Arc::new(AtomicU64::new(now_millis()));
        let loaded = Arc::new(LoadedModel {
            manifest: Arc::clone(&manifest),
            labels: Arc::clone(&labels),
            path: manifest_path.to_path_buf(),
            active,
            inner,
            last_used,
        });

        // Insert into the model map. If same ID exists, mark it inactive
        // first (mirrors `sparrow_engine_cpu::Engine::load_model`).
        {
            let mut models = self
                .models
                .write()
                .map_err(|_| SparrowEngineError::Ort("models lock poisoned".into()))?;
            if let Some(old) = models.get(&model_id) {
                old.active.store(false, Ordering::Release);
            }
            models.insert(model_id.clone(), Arc::clone(&loaded));
        }

        Ok(ModelHandle::from_loaded(&self.inner, model_id, loaded))
    }

    /// Load a model by ID. Resolves `{model_dir}/{id}/manifest.toml`.
    pub fn load_model_by_id(&self, id: &str) -> Result<ModelHandle> {
        sparrow_engine_core::catalog::validate_model_id(id)?;
        let manifest_path = self.inner.config.model_dir.join(id).join("manifest.toml");
        self.load_model(manifest_path)
    }

    /// Unload a model. The handle's `active` flag is set to false and
    /// the model is removed from the engine's map. Mirrors `sparrow_engine_cpu`'s
    /// TOCTOU-safe pattern (compare_exchange + Arc::ptr_eq).
    pub fn unload_model(&self, handle: &ModelHandle) -> Result<()> {
        if handle.engine_ref.upgrade().is_none() {
            return Err(SparrowEngineError::EngineFreed);
        }
        if handle
            .active
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(SparrowEngineError::ModelUnloaded);
        }
        let mut models = self
            .models
            .write()
            .map_err(|_| SparrowEngineError::Ort("models lock poisoned".into()))?;
        if let Some(entry) = models.get(&handle.model_id) {
            if Arc::ptr_eq(&entry.active, &handle.active) {
                models.remove(&handle.model_id);
            }
        }
        Ok(())
    }

    /// Unload an idle model by its ID. Used by the background reaper task in
    /// sparrow-engine-server. Returns `Ok(true)` if a model was unloaded, `Ok(false)`
    /// if the id is not currently loaded (idempotent — silent no-op). Mirrors
    /// `sparrow-engine-cpu`'s implementation.
    pub fn unload_model_by_id(&self, model_id: &str) -> Result<bool> {
        let mut models = self
            .models
            .write()
            .map_err(|_| SparrowEngineError::Ort("models lock poisoned".into()))?;
        match models.remove(model_id) {
            Some(entry) => {
                entry.active.store(false, Ordering::Release);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn unload_idle_snapshot(
        &self,
        model_id: &str,
        snapshot_last_used: u64,
        snapshot_active: &Arc<AtomicBool>,
        now: u64,
        idle_threshold_millis: u64,
    ) -> Result<bool> {
        let mut models = self
            .models
            .write()
            .map_err(|_| SparrowEngineError::Ort("models lock poisoned".into()))?;
        let should_remove = match models.get(model_id) {
            Some(entry) => {
                let current_last_used = entry.last_used.load(Ordering::Relaxed);
                if !reaper_snapshot_still_matches(
                    snapshot_active,
                    &entry.active,
                    snapshot_last_used,
                    current_last_used,
                    now,
                    idle_threshold_millis,
                ) {
                    false
                } else {
                    entry.active.store(false, Ordering::Release);
                    true
                }
            }
            None => false,
        };
        if should_remove {
            models.remove(model_id);
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Reap idle models: unload anything whose `last_used` is older than
    /// `idle_threshold_millis`, EXCEPT the `keep_last_n` most-recently-used.
    /// Returns the list of unloaded model IDs (for logging by the caller).
    /// Mirrors `sparrow-engine-cpu::Engine::reap_idle_models`.
    pub fn reap_idle_models(&self, idle_threshold_millis: u64, keep_last_n: usize) -> Vec<String> {
        let now = now_millis();
        let snapshot: Vec<(String, u64, Arc<AtomicBool>)> = {
            let models = match self.models.read() {
                Ok(m) => m,
                Err(_) => return Vec::new(),
            };
            models
                .iter()
                .filter(|(_, m)| m.active.load(Ordering::Acquire))
                .map(|(id, m)| {
                    (
                        id.clone(),
                        m.last_used.load(Ordering::Relaxed),
                        Arc::clone(&m.active),
                    )
                })
                .collect()
        };
        if snapshot.is_empty() {
            return Vec::new();
        }
        let mut sorted = snapshot;
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        let mut unloaded = Vec::new();
        for (id, last_used, active) in sorted.into_iter().skip(keep_last_n) {
            if let Ok(true) =
                self.unload_idle_snapshot(&id, last_used, &active, now, idle_threshold_millis)
            {
                unloaded.push(id);
            }
        }
        unloaded
    }

    // -----------------------------------------------------------------
    // Model lookup
    // -----------------------------------------------------------------

    /// Look up a loaded model handle by model ID. Returns `None` if not
    /// loaded or unloaded. Touches `last_used` on every successful lookup
    /// (mirrors `sparrow-engine-cpu`) so the background reaper task can decide
    /// auto-unload candidates.
    pub fn get_model_handle(&self, model_id: &str) -> Option<ModelHandle> {
        let models = match self.models.read() {
            Ok(models) => models,
            Err(_) => {
                tracing::error!("models lock poisoned while looking up model handle");
                return None;
            }
        };
        models.get(model_id).and_then(|m| {
            if m.active.load(Ordering::Acquire) {
                touch_last_used(&m.last_used);
                Some(ModelHandle::from_loaded(
                    &self.inner,
                    model_id.to_string(),
                    Arc::clone(m),
                ))
            } else {
                None
            }
        })
    }

    /// Look up multiple model handles atomically under a single read
    /// lock. Returns `(found_handles, missing_ids)`.
    pub fn get_model_handles(&self, ids: &[&str]) -> (Vec<ModelHandle>, Vec<String>) {
        let models = match self.models.read() {
            Ok(models) => models,
            Err(_) => {
                tracing::error!("models lock poisoned while looking up model handles");
                return (Vec::new(), ids.iter().map(|id| (*id).to_string()).collect());
            }
        };
        let mut found = Vec::with_capacity(ids.len());
        let mut missing = Vec::new();
        for &id in ids {
            match models.get(id) {
                Some(m) if m.active.load(Ordering::Acquire) => {
                    touch_last_used(&m.last_used);
                    found.push(ModelHandle::from_loaded(
                        &self.inner,
                        id.to_string(),
                        Arc::clone(m),
                    ));
                }
                _ => missing.push(id.to_string()),
            }
        }
        (found, missing)
    }

    /// Lazy model loading: return cached handle if loaded, otherwise
    /// load by ID. Double-checked locking via `loading_lock` prevents
    /// duplicate session creation.
    pub fn get_or_load_model(&self, model_id: &str) -> Result<ModelHandle> {
        if let Some(handle) = self.get_model_handle(model_id) {
            return Ok(handle);
        }
        let _guard = self
            .loading_lock
            .lock()
            .map_err(|_| SparrowEngineError::Ort("loading_lock poisoned".into()))?;
        if let Some(handle) = self.get_model_handle(model_id) {
            return Ok(handle);
        }
        self.load_model_by_id(model_id)
    }

    /// List all loaded models. Mirrors `sparrow_engine_cpu::Engine::loaded_models`.
    pub fn loaded_models(&self) -> Vec<ModelInfo> {
        let models = match self.models.read() {
            Ok(models) => models,
            Err(_) => {
                tracing::error!("models lock poisoned while listing models");
                return Vec::new();
            }
        };
        models
            .values()
            .filter(|m| m.active.load(Ordering::Acquire))
            .map(|m| m.to_model_info())
            .collect()
    }

    /// Scan model_dir for available models without loading them.
    pub fn list_available_models(&self) -> Vec<ModelInfo> {
        sparrow_engine_core::catalog::list_available_models(&self.inner.config.model_dir)
    }

    /// Look up info for a model by ID. Checks loaded models first, then
    /// falls back to the on-disk catalog.
    pub fn model_info(&self, id: &str) -> Result<ModelInfo> {
        // Loaded path.
        if let Some(handle) = self.get_model_handle(id) {
            return Ok(handle.inner.to_model_info());
        }
        // On-disk fallback. `SparrowEngineError` has no dedicated `ModelNotFound`
        // variant; we surface a `ManifestNotFound` pointing at the
        // expected on-disk path so the consumer error message names the
        // resolution path that failed.
        sparrow_engine_core::catalog::list_available_models(&self.inner.config.model_dir)
            .into_iter()
            .find(|info| info.id == id)
            .ok_or_else(|| {
                SparrowEngineError::ManifestNotFound(
                    self.inner.config.model_dir.join(id).join("manifest.toml"),
                )
            })
    }

    /// Resolve the default model ID for a given model type. Resolution
    /// order: env var override (type-validated against the catalog) → manifest
    /// `default = true` → unique-of-type. If the env-var value resolves to a
    /// model whose `model_type` differs from the requested type, a
    /// `tracing::warn!` is emitted and resolution falls through to the scan.
    pub fn resolve_default_model(&self, model_type: ModelType) -> Option<String> {
        let available = self.list_available_models();
        let env_var = match model_type {
            ModelType::Detector | ModelType::OverheadDetector => "SPARROW_ENGINE_DEFAULT_DETECTOR",
            ModelType::Classifier => "SPARROW_ENGINE_DEFAULT_CLASSIFIER",
            ModelType::AudioDetector => "SPARROW_ENGINE_DEFAULT_AUDIO_DETECTOR",
            ModelType::AudioClassifier => "SPARROW_ENGINE_DEFAULT_AUDIO_CLASSIFIER",
        };
        if let Ok(val) = std::env::var(env_var) {
            if !val.is_empty() {
                match available.iter().find(|m| m.id == val) {
                    Some(info) if info.model_type != model_type => {
                        tracing::warn!(
                            env_var = env_var,
                            requested = ?model_type,
                            resolved = ?info.model_type,
                            id = %val,
                            "env var resolved to a model whose type does not match the requested type; \
                             falling through to manifest scan",
                        );
                    }
                    _ => return Some(val),
                }
            }
        }
        let matching: Vec<&ModelInfo> = available
            .iter()
            .filter(|m| m.model_type == model_type)
            .collect();
        for m in &matching {
            if m.default {
                return Some(m.id.clone());
            }
        }
        if matching.len() == 1 {
            return Some(matching[0].id.clone());
        }
        None
    }

    // -----------------------------------------------------------------
    // Pipeline registration
    // -----------------------------------------------------------------

    /// Register a pipeline config from a manifest path.
    pub fn load_pipeline(&self, path: impl AsRef<Path>) -> Result<()> {
        let pipeline = manifest::load_pipeline_manifest(path.as_ref())?;
        self.register_pipeline_manifest(pipeline)
    }

    /// Register an already-validated pipeline manifest in memory.
    pub fn register_pipeline_manifest(&self, pipeline: PipelineManifest) -> Result<()> {
        let pipeline_id = pipeline.id.clone();
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SparrowEngineError::Ort("pipelines lock poisoned".into()))?;
        pipelines.insert(pipeline_id, pipeline);
        Ok(())
    }

    /// Register a pipeline config by ID.
    pub fn load_pipeline_by_id(&self, id: &str) -> Result<()> {
        let pipeline_path = self.inner.config.model_dir.join(id).join("pipeline.toml");
        self.load_pipeline(pipeline_path)
    }

    /// Unregister a pipeline config.
    pub fn unload_pipeline(&self, pipeline_id: &str) -> Result<()> {
        let mut pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SparrowEngineError::Ort("pipelines lock poisoned".into()))?;
        if pipelines.remove(pipeline_id).is_none() {
            return Err(SparrowEngineError::PipelineNotFound {
                id: pipeline_id.to_string(),
            });
        }
        Ok(())
    }

    /// Look up a registered pipeline config by ID.
    pub fn get_pipeline(&self, pipeline_id: &str) -> Result<PipelineManifest> {
        let pipelines = self
            .pipelines
            .lock()
            .map_err(|_| SparrowEngineError::Ort("pipelines lock poisoned".into()))?;
        pipelines
            .get(pipeline_id)
            .cloned()
            .ok_or_else(|| SparrowEngineError::PipelineNotFound {
                id: pipeline_id.to_string(),
            })
    }

    /// List all registered pipelines.
    pub fn loaded_pipelines(&self) -> Vec<PipelineManifest> {
        let pipelines = match self.pipelines.lock() {
            Ok(pipelines) => pipelines,
            Err(_) => {
                tracing::error!("pipelines lock poisoned while listing pipelines");
                return Vec::new();
            }
        };
        pipelines.values().cloned().collect()
    }

    /// Run a loaded pipeline on an image. Convenience wrapper around
    /// [`crate::pipeline::run_pipeline`].
    pub fn run_pipeline(
        &self,
        pipeline_id: &str,
        image: &sparrow_engine_types::ImageInput,
        detect_opts: &sparrow_engine_types::DetectOpts,
        classify_opts: &sparrow_engine_types::ClassifyOpts,
    ) -> Result<sparrow_engine_types::PipelineResult> {
        crate::pipeline::run_pipeline(self, pipeline_id, image, detect_opts, classify_opts)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // Mirrors `sparrow_engine_cpu::Engine::drop` (MT-17 mitigation): mark
        // every loaded model inactive so stale handles see `ModelUnloaded`
        // rather than reach into a freed session. Then LEAK the loaded
        // sessions (and the EngineInner Arc below) to avoid running `Drop` on
        // cudarc/ORT primitives during glibc `_dl_fini` (the pykeio/ort #564
        // class of teardown bug).
        //
        // RP-24 manual test (2026-06-20): the ORT TensorRT EP's session
        // teardown is far more fragile than the CUDA EP's — dropping a
        // TRT-backed session during `_dl_fini` SIGABRTs ~50% of the time with
        // "corrupted double-linked list", AFTER a fully correct inference.
        // `take` + `forget` leaks the session map so the TRT engines are never
        // torn down at process exit. Benign: the process is exiting (CLI) or
        // the `Engine` is a process-lifetime singleton (server) — the OS
        // reclaims at exit. Per-model runtime `unload_model` still drops
        // sessions normally (outside `_dl_fini`), so live eviction is
        // unaffected; only the final teardown leaks.
        if let Ok(mut models) = self.models.write() {
            for model in models.values() {
                model.active.store(false, Ordering::Release);
            }
            std::mem::forget(std::mem::take(&mut *models));
        }
        if let Ok(mut pipelines) = self.pipelines.lock() {
            pipelines.clear();
        }
        std::mem::forget(Arc::clone(&self.inner));
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
    }
}

// ---------------------------------------------------------------------------
// ModelHandle implementation
// ---------------------------------------------------------------------------

/// Current wall-clock unix-millis. Saturates to 0 if the system clock is
/// before the unix epoch (essentially impossible — but `unwrap` would panic).
/// Mirrors `sparrow-engine-cpu::engine::now_millis`.
pub(crate) fn now_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn touch_last_used(last_used: &AtomicU64) {
    let now = now_millis();
    let mut observed = last_used.load(Ordering::Relaxed);
    loop {
        let next = now.max(observed.saturating_add(1));
        match last_used.compare_exchange_weak(observed, next, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(actual) => observed = actual,
        }
    }
}

fn reaper_snapshot_still_matches(
    snapshot_active: &Arc<AtomicBool>,
    current_active: &Arc<AtomicBool>,
    snapshot_last_used: u64,
    current_last_used: u64,
    now: u64,
    idle_threshold_millis: u64,
) -> bool {
    Arc::ptr_eq(current_active, snapshot_active)
        && current_last_used == snapshot_last_used
        && now.saturating_sub(current_last_used) >= idle_threshold_millis
}

impl ModelHandle {
    /// Build a fresh handle from a pinned [`LoadedModel`] entry.
    ///
    /// Single source of truth for the handle ctor shape: a `Weak` back to
    /// the engine, an `Arc::clone` of the loaded model's `active` flag,
    /// and the pinned `Arc<LoadedModel>` snapshot.
    pub(crate) fn from_loaded(
        engine_inner: &Arc<EngineInner>,
        model_id: String,
        loaded: Arc<LoadedModel>,
    ) -> Self {
        Self {
            engine_ref: Arc::downgrade(engine_inner),
            active: Arc::clone(&loaded.active),
            inner: loaded,
            model_id,
        }
    }

    /// Check that this handle is still valid (model not unloaded,
    /// engine not freed).
    pub(crate) fn check_valid(&self) -> Result<()> {
        if self.engine_ref.upgrade().is_none() {
            return Err(SparrowEngineError::EngineFreed);
        }
        if !self.active.load(Ordering::Acquire) {
            return Err(SparrowEngineError::ModelUnloaded);
        }
        Ok(())
    }

    /// Pin the inner LoadedModel snapshot. Validates first. The
    /// returned `Arc<LoadedModel>` is safe to hold across model
    /// replace / unload events.
    pub(crate) fn pin_inner(&self) -> Result<Arc<LoadedModel>> {
        self.check_valid()?;
        Ok(Arc::clone(&self.inner))
    }

    /// Get the model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the manifest snapshot.
    pub fn manifest(&self) -> &Arc<ModelManifest> {
        &self.inner.manifest
    }

    /// Get the label table (or empty for binary detectors).
    pub fn labels(&self) -> &Arc<Vec<String>> {
        &self.inner.labels
    }

    /// Returns the model type derived from the manifest.
    pub fn model_type(&self) -> ModelType {
        self.inner.model_type()
    }

    /// Get the audio preprocessing config from the manifest, if this
    /// model uses mel-spectrogram preprocessing.
    pub fn audio_preprocess_config(
        &self,
    ) -> Option<sparrow_engine_core::preprocess_audio::AudioPreprocessConfig> {
        sparrow_engine_core::preprocess_audio::AudioPreprocessConfig::from_manifest(
            &self.inner.manifest.preprocess_method,
        )
    }

    /// Get the manifest-declared confidence threshold, if any.
    pub fn audio_confidence_threshold(&self) -> Option<f32> {
        self.inner.manifest.confidence_threshold
    }

    /// Get the audio inference window + stride from the manifest, if
    /// this model uses sliding-window inference.
    pub fn audio_window_stride(&self) -> Option<(f32, f32)> {
        match self.inner.manifest.inference_strategy {
            sparrow_engine_types::manifest::InferenceStrategy::SlidingWindow {
                segment_duration_s,
                segment_stride_s,
            } => Some((segment_duration_s, segment_stride_s)),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use std::path::PathBuf;

    fn dummy_model_dir() -> PathBuf {
        PathBuf::from("/tmp/bongo_gpu_test_models_nonexistent")
    }

    /// Helper: skip a test cleanly when no GPU is available. Mirrors
    /// the gating used by other `sparrow-engine-gpu` integration tests.
    fn cuda_available() -> bool {
        CudaContext::new(0).is_ok()
    }

    fn dummy_pipeline_manifest(id: &str) -> PipelineManifest {
        PipelineManifest {
            id: id.to_string(),
            steps: vec![
                manifest::PipelineStep {
                    role: manifest::PipelineRole::Detector,
                    model: "detector-model".to_string(),
                },
                manifest::PipelineStep {
                    role: manifest::PipelineRole::Classifier,
                    model: "classifier-model".to_string(),
                },
            ],
        }
    }

    fn same_pipeline_steps(a: &PipelineManifest, b: &PipelineManifest) -> bool {
        a.steps.len() == b.steps.len()
            && a.steps
                .iter()
                .zip(&b.steps)
                .all(|(a, b)| a.role == b.role && a.model == b.model)
    }

    #[test]
    #[serial]
    fn singleton_enforcement_no_gpu_safe() {
        if !cuda_available() {
            eprintln!("singleton_enforcement: no CUDA, skipping");
            return;
        }
        // Reset global state for test isolation.
        ENGINE_EXISTS.store(false, Ordering::SeqCst);

        let config = EngineConfig::new(Device::Auto, dummy_model_dir());
        let engine = Engine::new(config.clone()).expect("first engine");
        let res2 = Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir()));
        assert!(
            matches!(res2, Err(SparrowEngineError::EngineAlreadyExists)),
            "second engine must fail with EngineAlreadyExists"
        );
        // Drop the Ok-arm engine if any (shouldn't be), then drop the
        // first engine and verify a third construction succeeds.
        drop(res2);
        drop(engine);
        let engine3 = Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir()))
            .expect("engine after drop");
        drop(engine3);
    }

    #[test]
    #[serial]
    fn loaded_models_empty_on_new_engine() {
        if !cuda_available() {
            eprintln!("loaded_models_empty_on_new_engine: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Auto, dummy_model_dir());
        let engine = Engine::new(config).expect("engine");
        assert!(engine.loaded_models().is_empty());
        drop(engine);
    }

    #[test]
    #[serial]
    fn unload_pipeline_not_found() {
        if !cuda_available() {
            eprintln!("unload_pipeline_not_found: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        let err = engine.unload_pipeline("nonexistent").unwrap_err();
        assert!(matches!(err, SparrowEngineError::PipelineNotFound { .. }));
        drop(engine);
    }

    #[test]
    #[serial]
    fn get_pipeline_not_found() {
        if !cuda_available() {
            eprintln!("get_pipeline_not_found: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        let err = engine.get_pipeline("nonexistent").unwrap_err();
        assert!(matches!(err, SparrowEngineError::PipelineNotFound { .. }));
        drop(engine);
    }

    #[test]
    #[serial]
    fn register_pipeline_manifest_round_trips_and_unloads() {
        if !cuda_available() {
            eprintln!("register_pipeline_manifest_round_trips_and_unloads: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        let manifest = dummy_pipeline_manifest("runtime-alias");

        engine.register_pipeline_manifest(manifest.clone()).unwrap();
        let registered = engine.get_pipeline("runtime-alias").unwrap();
        assert_eq!(registered.id, manifest.id);
        assert!(same_pipeline_steps(&registered, &manifest));

        engine.unload_pipeline("runtime-alias").unwrap();
        let err = engine.get_pipeline("runtime-alias").unwrap_err();
        assert!(matches!(err, SparrowEngineError::PipelineNotFound { .. }));
        drop(engine);
    }

    #[test]
    #[serial]
    fn get_model_handle_not_found() {
        if !cuda_available() {
            eprintln!("get_model_handle_not_found: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        assert!(engine.get_model_handle("nonexistent").is_none());
        drop(engine);
    }

    #[test]
    #[serial]
    fn active_device_resolves_auto() {
        if !cuda_available() {
            eprintln!("active_device_resolves_auto: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        assert!(matches!(engine.active_device(), Device::Cuda(_)));
        drop(engine);
    }

    #[test]
    #[serial]
    fn list_available_models_empty_for_nonexistent_dir() {
        if !cuda_available() {
            eprintln!("list_available_models_empty: no CUDA, skipping");
            return;
        }
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let engine =
            Engine::new(EngineConfig::new(Device::Auto, dummy_model_dir())).expect("engine");
        assert!(engine.list_available_models().is_empty());
        drop(engine);
    }

    #[test]
    fn touch_last_used_increments_when_clock_has_not_advanced() {
        let last_used = AtomicU64::new(now_millis());
        let before = last_used.load(Ordering::Relaxed);
        touch_last_used(&last_used);
        let after = last_used.load(Ordering::Relaxed);
        assert!(
            after > before,
            "same-millisecond touches must still advance last_used; before={before}, after={after}"
        );
    }

    #[test]
    fn reaper_snapshot_match_rejects_touched_entry() {
        let active = Arc::new(AtomicBool::new(true));
        assert!(!reaper_snapshot_still_matches(
            &active, &active, 100, 101, 2_000, 1_000,
        ));
    }

    #[test]
    fn reaper_snapshot_match_rejects_replacement_generation() {
        let snapshot_active = Arc::new(AtomicBool::new(true));
        let current_active = Arc::new(AtomicBool::new(true));
        assert!(!reaper_snapshot_still_matches(
            &snapshot_active,
            &current_active,
            100,
            100,
            2_000,
            1_000,
        ));
    }

    #[test]
    fn reaper_snapshot_match_accepts_stale_same_generation() {
        let active = Arc::new(AtomicBool::new(true));
        assert!(reaper_snapshot_still_matches(
            &active, &active, 100, 100, 2_000, 1_000,
        ));
    }
}
