//! ORT singleton engine, session management, model loading.
//!
//! ALL `ort` crate usage is isolated in this file. No `ort` types appear in
//! the public API — only sparrow-engine types. Other modules access ORT sessions
//! through `ModelHandle::pin_session()` which returns `Arc<ort::Session>`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{SystemTime, UNIX_EPOCH};

use ort::session::Session;

use sparrow_engine_types::manifest::{
    self, ModelManifest, PipelineManifest, PostprocessMethod, PreprocessMethod,
};
use sparrow_engine_types::{derive_model_type, ModelInfo, ModelType, Result, SparrowEngineError};

// Phase 3.8 Phase A back-compat re-exports: consumers historically imported
// `sparrow_engine::engine::Device` and `sparrow_engine::engine::EngineConfig`
// via the `[lib] name = "sparrow_engine"` cdylib after the R2 rename.
// After the Phase A crate split these types live in `sparrow-engine-types` and are
// re-exported from sparrow-engine-cpu's crate root via the glob in `lib.rs`, but the
// `engine::*` path also needs to keep working. Re-exporting here preserves the
// public API surface across Phase A without forcing consumer-crate updates.
pub use sparrow_engine_types::{Device, EngineConfig};

// ---------------------------------------------------------------------------
// Singleton guard
// ---------------------------------------------------------------------------

/// Process-global flag: true if an Engine instance exists.
static ENGINE_EXISTS: AtomicBool = AtomicBool::new(false);

// ---------------------------------------------------------------------------
// Internal types
// ---------------------------------------------------------------------------

/// A loaded ORT session with its manifest and label data.
pub(crate) struct LoadedModel {
    session: Arc<Mutex<Session>>,
    manifest: Arc<ModelManifest>,
    labels: Arc<Vec<String>>,
    active: Arc<AtomicBool>,
    path: PathBuf,
    /// Unix-millis timestamp of the last `get_model_handle` lookup (touched
    /// once per inference HTTP request, before the actual work runs). The
    /// background reaper (`reap_idle_models`) compares this to the configured
    /// idle timeout when deciding which models to auto-unload. Shared with
    /// every `ModelHandle` via `Arc<AtomicU64>` so a touch is one relaxed store.
    pub(crate) last_used: Arc<AtomicU64>,
}

// Safety: Session is behind Mutex. All other fields are Arc-wrapped or plain data.
unsafe impl Send for LoadedModel {}
unsafe impl Sync for LoadedModel {}

/// Internal engine state behind Arc for shared ownership.
pub(crate) struct EngineInner {
    /// Session builder template — cloned per `load_model` call so each session
    /// inherits the same EP and thread config.
    session_builder: Mutex<ort::session::builder::SessionBuilder>,
    /// Engine configuration snapshot.
    pub(crate) config: EngineConfig,
    /// Device after resolving `Device::Auto` via ORT EP availability check.
    resolved_device: Device,
}

// Safety: SessionBuilder is behind Mutex — only one thread accesses it at a time.
unsafe impl Send for EngineInner {}
unsafe impl Sync for EngineInner {}

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// The singleton inference engine.
///
/// Only one `Engine` may exist per process (ORT Environment is process-global).
/// A second `Engine::new()` returns [`SparrowEngineError::EngineAlreadyExists`].
pub struct Engine {
    pub(crate) inner: Arc<EngineInner>,
    /// Loaded model handles, keyed by model ID.
    pub(crate) models: RwLock<HashMap<String, LoadedModel>>,
    /// Registered pipeline configs, keyed by pipeline ID.
    pub(crate) pipelines: Mutex<HashMap<String, PipelineManifest>>,
    /// Serializes first-load operations to prevent TOCTOU double-load race.
    /// Coarse-grained (all model IDs share one lock) because `session_builder`
    /// is already globally serialized — per-model locks would add complexity
    /// for zero throughput gain.
    loading_lock: Mutex<()>,
}

// Safety: All non-Send/Sync ORT types (SessionBuilder, Session) are wrapped
// behind std::sync::Mutex. Engine's public API is designed for concurrent use.
// The !Send/!Sync on ORT types comes from raw pointers in ort-sys bindings,
// not from actual thread-safety violations.
unsafe impl Send for Engine {}
unsafe impl Sync for Engine {}

/// Opaque handle to a loaded model.
///
/// Holds a pinned `Arc<Mutex<Session>>` — inference uses this session directly without
/// looking up from the engine's model map. Cheaply cloneable (all fields are Arc).
/// Safe to use across threads.
#[derive(Clone)]
pub struct ModelHandle {
    /// Weak reference back to the engine. Fails to upgrade if engine is dropped.
    pub(crate) engine_ref: Weak<EngineInner>,
    /// Set to false when the model is unloaded. Checked before every inference call.
    pub(crate) active: Arc<AtomicBool>,
    /// The ORT session for this model.
    pub(crate) session: Arc<Mutex<Session>>,
    /// Parsed and validated model manifest.
    pub(crate) manifest: Arc<ModelManifest>,
    /// Ordered label names. Index = label_id.
    pub(crate) labels: Arc<Vec<String>>,
    /// Model ID from the manifest.
    model_id: String,
}

// Safety: All ORT types in ModelHandle (Session) are behind Mutex. Weak<EngineInner>
// is safe because EngineInner is Send+Sync (above). All other fields are Arc-wrapped
// atomics or plain data.
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
    /// Create the singleton engine.
    ///
    /// Initializes the ORT environment, configures execution providers and
    /// thread pools. Returns `EngineAlreadyExists` if an engine already exists.
    pub fn new(config: EngineConfig) -> Result<Self> {
        // Atomically claim the singleton slot.
        if ENGINE_EXISTS.swap(true, Ordering::SeqCst) {
            return Err(SparrowEngineError::EngineAlreadyExists);
        }

        // Resolve Auto to a concrete device before building the session.
        let resolved_device = Self::resolve_device(&config.device);

        // Build the ORT session builder template. On failure release the
        // singleton flag before propagating; mirrors sparrow-engine-gpu's
        // `Engine::new` `init().inspect_err(...)?` pattern for cross-flavor
        // consistency on the singleton-release side effect.
        let builder = Self::create_session_builder(&config)
            .inspect_err(|_| ENGINE_EXISTS.store(false, Ordering::SeqCst))?;

        // EngineInner is !Send+!Sync (ort SessionBuilder is !Send). Arc is still
        // correct here: shared ownership between Engine and ModelHandles. Thread
        // safety is provided by the Mutex wrapping SessionBuilder, not by Arc.
        #[allow(clippy::arc_with_non_send_sync)]
        Ok(Engine {
            inner: Arc::new(EngineInner {
                session_builder: Mutex::new(builder),
                config,
                resolved_device,
            }),
            models: RwLock::new(HashMap::new()),
            pipelines: Mutex::new(HashMap::new()),
            loading_lock: Mutex::new(()),
        })
    }

    /// Resolve `Device::Auto` and `Device::Cuda(_)` to `Device::Cpu`.
    ///
    /// Phase 4.1 MT-4.1-2: sparrow-engine-cpu is strictly CPU at the binary level; the
    /// `ort` crate is built without the `cuda` feature, so CUDA EP is not
    /// reachable from this flavor regardless of which `libonnxruntime.so.1` the
    /// dynamic linker resolves at runtime. Mirrors sparrow-engine-gpu's strict-GPU
    /// coercion (`Device::Auto | Cpu => Cuda(0)` in sparrow-engine-gpu/src/engine.rs).
    /// `Device::Cuda(_)` requests are silently coerced to `Cpu` (with a
    /// `tracing::warn!`) rather than rejected, so existing callers that pass
    /// `Cuda(n)` from a flavor-agnostic config keep working — they just get
    /// CPU inference. Use the `sparrow-engine-gpu` flavor for actual CUDA inference.
    fn resolve_device(device: &Device) -> Device {
        match device {
            Device::Auto | Device::Cpu => Device::Cpu,
            Device::Cuda(n) => {
                tracing::warn!(
                    requested_cuda_device = *n,
                    "sparrow-engine-cpu flavor cannot use CUDA; coercing Device::Cuda({n}) to Device::Cpu. \
                     Install / link the sparrow-engine-gpu flavor for CUDA inference."
                );
                Device::Cpu
            }
        }
    }

    /// Build the ORT `SessionBuilder` template with EP and thread config.
    fn create_session_builder(
        config: &EngineConfig,
    ) -> Result<ort::session::builder::SessionBuilder> {
        use ort::session::builder::GraphOptimizationLevel;

        let builder = Session::builder().map_err(ort_err)?;

        // Enable all graph optimizations (constant folding, node fusions, layout).
        // ORT default is already All, but set explicitly for clarity and to match
        // the Python benchmark config (ORT_ENABLE_ALL).
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| SparrowEngineError::Ort(e.to_string()))?;

        // Configure thread pools.
        let builder = builder
            .with_intra_threads(config.intra_threads as usize)
            .map_err(|e| SparrowEngineError::Ort(e.to_string()))?;
        let builder = builder
            .with_inter_threads(config.inter_threads as usize)
            .map_err(|e| SparrowEngineError::Ort(e.to_string()))?;

        // Phase 4.1 MT-4.1-2: sparrow-engine-cpu is strictly CPU; CUDA EP is not
        // compiled in (`ort` without "cuda" feature). All Device variants
        // resolve to CPU EP. `resolve_device()` already coerced Auto/Cuda
        // to Cpu before this point — match arm covers all three for safety.
        let builder = match &config.device {
            Device::Auto | Device::Cpu | Device::Cuda(_) => builder
                .with_execution_providers([ort::ep::CPU::default().build()])
                .map_err(|e| SparrowEngineError::Ort(e.to_string()))?,
        };

        Ok(builder)
    }

    /// Load a model from a manifest path.
    ///
    /// Parses the TOML manifest, creates an ORT session, validates the output
    /// shape against the declared postprocessing method, loads labels, and
    /// returns an opaque [`ModelHandle`].
    ///
    /// If a model with the same ID is already loaded, it is replaced. In-flight
    /// inference on the old session completes safely via `Arc<Mutex<Session>>` refcounting.
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<ModelHandle> {
        let manifest_path = path.as_ref();

        // Parse and validate manifest (handles file existence check, format
        // validation, tiled field validation, label path traversal check).
        let manifest = manifest::load_manifest(manifest_path)?;

        // Flavor-strict: the cpu/gpu flavors run ONNX models via ORT. The shared
        // loader now also accepts `tflite` manifests (for the mobile LiteRT
        // flavor); reject a non-ONNX format here with a clear error rather than
        // letting ORT fail to parse a `.tflite` file. Mirrors the Device::Auto
        // flavor-strict coercion contract.
        if manifest.format != "onnx" {
            return Err(SparrowEngineError::UnsupportedFormat {
                format: manifest.format.clone(),
            });
        }

        let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));

        // Resolve ONNX file path (relative to manifest directory). When the
        // manifest opts into FP16 inference (Phase 3.8), use the FP16-converted
        // model file; manifest validation already ensured `model_file_fp16` is
        // present whenever `precision == Fp16`.
        let onnx_path = match manifest.precision {
            manifest::Precision::Fp32 => manifest_dir.join(&manifest.model_file),
            manifest::Precision::Int8 => manifest_dir.join(&manifest.model_file),
            manifest::Precision::Fp16 => manifest_dir.join(
                manifest
                    .model_file_fp16
                    .as_ref()
                    .expect("manifest validation guarantees file_fp16 when precision = Fp16"),
            ),
        };

        // Load labels (optional — binary detectors like audio bird detector have none).
        let labels = match (&manifest.label_file, &manifest.label_format) {
            (Some(file), Some(fmt)) => {
                let label_path = manifest_dir.join(file);
                manifest::load_labels(&label_path, fmt)?
            }
            _ => Vec::new(),
        };

        // Create ORT session from cloned template builder.
        let session = {
            let mut builder = self
                .inner
                .session_builder
                .lock()
                .expect("session_builder lock poisoned")
                .clone();
            builder.commit_from_file(&onnx_path).map_err(ort_err)?
        };

        // Validate output shape vs postprocessing method.
        validate_output_shape(&session, &manifest)?;

        let session = Arc::new(Mutex::new(session));
        let active = Arc::new(AtomicBool::new(true));
        let manifest = Arc::new(manifest);
        let labels = Arc::new(labels);
        let model_id = manifest.id.clone();
        let last_used = Arc::new(AtomicU64::new(now_millis()));

        // Store in the model map. If same ID exists, replace it.
        {
            let mut models = self.models.write().expect("models lock poisoned");
            if let Some(old) = models.get(&model_id) {
                // Mark old model as inactive so stale handles get ModelUnloaded.
                old.active.store(false, Ordering::Release);
            }
            models.insert(
                model_id.clone(),
                LoadedModel {
                    session: Arc::clone(&session),
                    manifest: Arc::clone(&manifest),
                    labels: Arc::clone(&labels),
                    active: Arc::clone(&active),
                    path: manifest_path.to_path_buf(),
                    last_used: Arc::clone(&last_used),
                },
            );
        }

        Ok(ModelHandle {
            engine_ref: Arc::downgrade(&self.inner),
            active,
            session,
            manifest,
            labels,
            model_id,
        })
    }

    /// Load a model by ID. Resolves `{model_dir}/{id}/manifest.toml`.
    pub fn load_model_by_id(&self, id: &str) -> Result<ModelHandle> {
        crate::catalog::validate_model_id(id)?;
        let manifest_path = self.inner.config.model_dir.join(id).join("manifest.toml");
        self.load_model(manifest_path)
    }

    /// Unload a model. The handle's `active` flag is set to false and the model
    /// is removed from the engine's map. In-flight inference on the old session
    /// completes safely via `Arc<Mutex<Session>>` refcounting.
    ///
    /// Uses `compare_exchange` on the active flag as an atomic gate so only one
    /// caller wins. Before removing from the HashMap, verifies via `Arc::ptr_eq`
    /// that the map entry is the *same generation* — a replacement model (same
    /// ID, different `Arc<AtomicBool>`) survives the unload.
    pub fn unload_model(&self, handle: &ModelHandle) -> Result<()> {
        // Check engine is still alive.
        if handle.engine_ref.upgrade().is_none() {
            return Err(SparrowEngineError::EngineFreed);
        }

        // Atomically claim the unload — only one caller wins.
        // compare_exchange(true→false) fails if already inactive (double-unload).
        if handle
            .active
            .compare_exchange(true, false, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return Err(SparrowEngineError::ModelUnloaded);
        }

        // Remove from map only if the entry is the same generation.
        // A concurrent load_model with the same ID may have replaced the entry
        // between our CAS and this write lock — Arc::ptr_eq guards against
        // deleting the replacement.
        let mut models = self.models.write().expect("models lock poisoned");
        if let Some(entry) = models.get(&handle.model_id) {
            if Arc::ptr_eq(&entry.active, &handle.active) {
                models.remove(&handle.model_id);
            }
        }

        Ok(())
    }

    /// Unload an idle model by its ID. Used by the background reaper task in
    /// sparrow-engine-server. Returns `Ok(true)` if a model was unloaded, `Ok(false)`
    /// if the id is not currently loaded (idempotent — silent no-op). In-flight
    /// inference holds an `Arc<Mutex<Session>>` and completes safely; the next
    /// inference request lazy-reloads via `get_or_load_model`.
    pub fn unload_model_by_id(&self, model_id: &str) -> Result<bool> {
        let mut models = self.models.write().expect("models lock poisoned");
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
    ) -> bool {
        let mut models = self.models.write().expect("models lock poisoned");
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
            true
        } else {
            false
        }
    }

    /// Reap idle models: unload anything whose `last_used` is older than
    /// `idle_threshold_millis`, EXCEPT the `keep_last_n` most-recently-used.
    /// Returns the list of unloaded model IDs (for logging by the caller).
    ///
    /// Called periodically by the sparrow-engine-server background reaper task.
    /// Cheap: single read lock to snapshot, then per-eviction write lock
    /// inside `unload_idle_snapshot`.
    pub fn reap_idle_models(&self, idle_threshold_millis: u64, keep_last_n: usize) -> Vec<String> {
        let now = now_millis();
        // Snapshot (id, last_used, generation) under read lock.
        let snapshot: Vec<(String, u64, Arc<AtomicBool>)> = {
            let models = self.models.read().expect("models lock poisoned");
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
        // Sort by last_used desc (most-recently-used first). Ties broken by
        // id ascending for determinism.
        let mut sorted = snapshot;
        sorted.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
        // The top `keep_last_n` are protected regardless of idle age.
        // The rest are candidates if they're stale enough.
        let mut unloaded = Vec::new();
        for (id, last_used, active) in sorted.into_iter().skip(keep_last_n) {
            if self.unload_idle_snapshot(&id, last_used, &active, now, idle_threshold_millis) {
                unloaded.push(id);
            }
        }
        unloaded
    }

    /// Register a pipeline config from a manifest path.
    ///
    /// Does NOT load the referenced models — they must be loaded separately.
    pub fn load_pipeline(&self, path: impl AsRef<Path>) -> Result<()> {
        let pipeline = manifest::load_pipeline_manifest(path.as_ref())?;
        self.register_pipeline_manifest(pipeline)
    }

    /// Register an already-validated pipeline manifest in memory.
    pub fn register_pipeline_manifest(&self, pipeline: PipelineManifest) -> Result<()> {
        let pipeline_id = pipeline.id.clone();
        let mut pipelines = self.pipelines.lock().expect("pipelines lock poisoned");
        pipelines.insert(pipeline_id, pipeline);
        Ok(())
    }

    /// Register a pipeline config by ID. Resolves `{model_dir}/{id}/pipeline.toml`.
    pub fn load_pipeline_by_id(&self, id: &str) -> Result<()> {
        let pipeline_path = self.inner.config.model_dir.join(id).join("pipeline.toml");
        self.load_pipeline(pipeline_path)
    }

    /// Unregister a pipeline config.
    pub fn unload_pipeline(&self, pipeline_id: &str) -> Result<()> {
        let mut pipelines = self.pipelines.lock().expect("pipelines lock poisoned");
        if pipelines.remove(pipeline_id).is_none() {
            return Err(SparrowEngineError::PipelineNotFound {
                id: pipeline_id.to_string(),
            });
        }
        Ok(())
    }

    /// Look up a registered pipeline config by ID.
    pub fn get_pipeline(&self, pipeline_id: &str) -> Result<PipelineManifest> {
        let pipelines = self.pipelines.lock().expect("pipelines lock poisoned");
        pipelines
            .get(pipeline_id)
            .cloned()
            .ok_or_else(|| SparrowEngineError::PipelineNotFound {
                id: pipeline_id.to_string(),
            })
    }

    /// Look up a loaded model handle by model ID.
    ///
    /// Used by pipeline orchestration to resolve model references at runtime.
    /// Returns `None` if the model is not loaded or has been unloaded.
    ///
    /// Touches the model's `last_used` timestamp on every successful lookup.
    /// The background idle reaper (`reap_idle_models`) reads this to decide
    /// which models are eligible for auto-unload.
    pub fn get_model_handle(&self, model_id: &str) -> Option<ModelHandle> {
        let models = self.models.read().expect("models lock poisoned");
        models.get(model_id).and_then(|m| {
            if m.active.load(Ordering::Acquire) {
                touch_last_used(&m.last_used);
                Some(ModelHandle::from_loaded(
                    &self.inner,
                    model_id.to_string(),
                    m,
                ))
            } else {
                None
            }
        })
    }

    /// Look up multiple model handles atomically under a single read lock.
    ///
    /// Returns `(found_handles, missing_ids)`. Holds one read lock on
    /// `self.models` for the entire lookup, so the set of returned handles
    /// is a consistent snapshot — no model can be replaced or removed between
    /// individual lookups. Used by pipeline orchestration for atomic session
    /// pinning.
    pub fn get_model_handles(&self, ids: &[&str]) -> (Vec<ModelHandle>, Vec<String>) {
        let models = self.models.read().expect("models lock poisoned");
        let mut found = Vec::with_capacity(ids.len());
        let mut missing = Vec::new();

        for &id in ids {
            match models.get(id) {
                Some(m) if m.active.load(Ordering::Acquire) => {
                    touch_last_used(&m.last_used);
                    found.push(ModelHandle::from_loaded(&self.inner, id.to_string(), m));
                }
                _ => {
                    missing.push(id.to_string());
                }
            }
        }

        (found, missing)
    }

    /// List all loaded models (ID, path, type).
    pub fn loaded_models(&self) -> Vec<ModelInfo> {
        let models = self.models.read().expect("models lock poisoned");
        models
            .values()
            .filter(|m| m.active.load(Ordering::Acquire))
            .map(|m| ModelInfo {
                id: m.manifest.id.clone(),
                path: m.path.clone(),
                model_type: derive_model_type(
                    &m.manifest.preprocess_method,
                    &m.manifest.postprocess_method,
                    m.manifest.subtype,
                ),
                default: m.manifest.default,
                version: m.manifest.version.clone(),
                description: m.manifest.description.clone(),
                onnx_sha256: m.manifest.onnx_sha256.clone(),
                onnx_size_bytes: m.manifest.onnx_size_bytes,
            })
            .collect()
    }

    /// Lazy model loading: return cached handle if loaded, otherwise load by ID.
    ///
    /// Resolves `{model_dir}/{model_id}/manifest.toml` on first access.
    /// Uses double-checked locking to prevent TOCTOU race where two concurrent
    /// callers both see "not loaded" and both create ORT sessions. The second
    /// load would replace the first, invalidating the first caller's handle.
    pub fn get_or_load_model(&self, model_id: &str) -> Result<ModelHandle> {
        // Fast path: model already loaded (read lock only).
        if let Some(handle) = self.get_model_handle(model_id) {
            return Ok(handle);
        }
        // Slow path: serialize first-loads to prevent duplicate session creation.
        let _guard = self
            .loading_lock
            .lock()
            .map_err(|_| SparrowEngineError::Ort("loading_lock poisoned".into()))?;
        // Re-check after acquiring lock — another thread may have loaded it.
        if let Some(handle) = self.get_model_handle(model_id) {
            return Ok(handle);
        }
        self.load_model_by_id(model_id)
    }

    /// Scan model_dir for available models without loading them.
    ///
    /// Reads `{model_dir}/{id}/manifest.toml` for each subdirectory, parses
    /// the manifest header (id, model_type, default), but does NOT create ORT sessions.
    pub fn list_available_models(&self) -> Vec<ModelInfo> {
        crate::catalog::list_available_models(&self.inner.config.model_dir)
    }

    /// Resolve the default model ID for a given model type.
    ///
    /// Resolution order:
    /// 1. Environment variable (`SPARROW_ENGINE_DEFAULT_DETECTOR`, `SPARROW_ENGINE_DEFAULT_CLASSIFIER`,
    ///    `SPARROW_ENGINE_DEFAULT_AUDIO_DETECTOR`, `SPARROW_ENGINE_DEFAULT_AUDIO_CLASSIFIER`).
    ///    If the env-var value resolves to a known model whose `model_type` differs
    ///    from the requested type, a `tracing::warn!` is emitted and resolution falls
    ///    through to (2). Unknown IDs are returned unchanged so downstream
    ///    "model not found" errors fire as before.
    /// 2. Manifest with `default = true` matching the requested type
    /// 3. If exactly one model of the requested type exists, use it
    /// 4. `None` otherwise (ambiguous — caller must specify)
    pub fn resolve_default_model(&self, model_type: ModelType) -> Option<String> {
        let available = self.list_available_models();

        // 1. Check env var override.
        let env_var = match model_type {
            // Standard and overhead detectors share the detector env var, but
            // the resolved ID is type-validated below: an env var pointing at
            // an OverheadDetector when a Detector was requested falls through
            // to the manifest scan rather than silently widening.
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

        // 2. Scan available models.
        let matching: Vec<&ModelInfo> = available
            .iter()
            .filter(|m| m.model_type == model_type)
            .collect();

        // Check for manifest default.
        for m in &matching {
            if m.default {
                return Some(m.id.clone());
            }
        }

        // 3. If exactly one model of this type, use it.
        if matching.len() == 1 {
            return Some(matching[0].id.clone());
        }

        None
    }

    /// List all registered pipelines.
    pub fn loaded_pipelines(&self) -> Vec<PipelineManifest> {
        let pipelines = self.pipelines.lock().expect("pipelines lock poisoned");
        pipelines.values().cloned().collect()
    }

    /// Get the engine config.
    pub fn config(&self) -> &EngineConfig {
        &self.inner.config
    }

    /// Returns the resolved device for this engine.
    ///
    /// If the engine was created with `Device::Auto`, this returns the
    /// concrete device selected (`Cpu` or `Cuda(n)`), never `Auto`.
    ///
    /// **Known limitation**: this is a best-effort report based on compile-time
    /// EP availability, not runtime GPU probing. On a system where ORT was
    /// compiled with CUDA EP support but no physical GPU is present (or CUDA
    /// drivers are missing), this may return `Cuda(0)` while actual inference
    /// runs on CPU via ORT's silent EP fallback. Similarly, an explicit
    /// `Device::Cuda(n)` passes through without runtime validation.
    /// The authoritative device is determined by ORT at session creation time,
    /// which is after this value is resolved.
    pub fn active_device(&self) -> &Device {
        &self.inner.resolved_device
    }

    /// Run a loaded pipeline on an image. Convenience wrapper around `pipeline::run_pipeline`.
    pub fn run_pipeline(
        &self,
        pipeline_id: &str,
        image: &crate::ImageInput,
        detect_opts: &crate::DetectOpts,
        classify_opts: &crate::ClassifyOpts,
    ) -> crate::error::Result<crate::PipelineResult> {
        crate::pipeline::run_pipeline(self, pipeline_id, image, detect_opts, classify_opts)
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        // MT-17 mitigation — see docs/bugs.md and
        // https://github.com/pykeio/ort/issues/564 (maintainer confirmed this
        // class of teardown bug is not fixable in `ort`).
        //
        // Two stacked fixes:
        //
        //   (a) Clear sessions eagerly under a write lock. Leaving
        //       `Arc<Mutex<Session>>` entries inside the HashMap until
        //       Rust's field-drop sweep ran was letting CUDA EP sessions
        //       drop *after* Arc<EngineInner>'s SessionBuilder template,
        //       which observationally halved but did not eliminate the
        //       "corrupted double-linked list" SIGABRT at process exit.
        //
        //   (b) Leak `Arc<EngineInner>` so `SessionBuilder::drop` does
        //       not fire during glibc `_dl_fini` after `main()` returns.
        //       ORT's `SessionBuilder` retains EP hooks that reach into
        //       `libonnxruntime_providers_cuda.so`; if that shared
        //       object is finalized first, the builder drop reads freed
        //       memory. pykeio/ort already keeps its `Environment` as a
        //       static singleton (discussion #280), so leaking our small
        //       `EngineInner` is symmetric — one struct per `Engine`
        //       instance, benign for CLI and library consumers.
        if let Ok(mut models) = self.models.write() {
            for model in models.values() {
                model.active.store(false, Ordering::Release);
            }
            models.clear();
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
    /// Create a handle from a loaded model entry.
    pub(crate) fn from_loaded(
        inner: &Arc<EngineInner>,
        model_id: String,
        loaded: &LoadedModel,
    ) -> Self {
        ModelHandle {
            engine_ref: Arc::downgrade(inner),
            active: Arc::clone(&loaded.active),
            session: Arc::clone(&loaded.session),
            manifest: Arc::clone(&loaded.manifest),
            labels: Arc::clone(&loaded.labels),
            model_id,
        }
    }

    /// Check that this handle is still valid (model not unloaded, engine not freed).
    pub(crate) fn check_valid(&self) -> Result<()> {
        if self.engine_ref.upgrade().is_none() {
            return Err(SparrowEngineError::EngineFreed);
        }
        if !self.active.load(Ordering::Acquire) {
            return Err(SparrowEngineError::ModelUnloaded);
        }
        Ok(())
    }

    /// Pin the session: clone `Arc<Mutex<Session>>` for snapshot isolation.
    ///
    /// Checks handle validity first. The returned `Arc<Mutex<Session>>` is the
    /// snapshot — safe to use even if the model is replaced or unloaded
    /// after this call.
    pub(crate) fn pin_session(&self) -> Result<Arc<Mutex<Session>>> {
        self.check_valid()?;
        Ok(Arc::clone(&self.session))
    }

    /// Get the model ID.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Get the manifest snapshot.
    ///
    /// Phase 3.8 Phase C Wave 4 NS-1 closure: mirrors
    /// `sparrow_engine_gpu::ModelHandle::manifest` so consumer crates with compile-time
    /// engine dispatch (`sparrow-engine-server` / `sparrow-engine-cli` / `sparrow-engine-python`) see the
    /// same public accessor surface on either flavor.
    pub fn manifest(&self) -> &Arc<ModelManifest> {
        &self.manifest
    }

    /// Get the label table (or empty for binary detectors).
    ///
    /// Phase 3.8 Phase C Wave 4 NS-1 closure: mirrors
    /// `sparrow_engine_gpu::ModelHandle::labels`.
    pub fn labels(&self) -> &Arc<Vec<String>> {
        &self.labels
    }

    /// Get the audio preprocessing config from the manifest, if this model
    /// uses mel-spectrogram preprocessing. Returns `None` for image models.
    ///
    /// Used by `spe detect-audio --visualize` to render a real mel
    /// spectrogram backdrop from the same parameters the model was trained on.
    pub fn audio_preprocess_config(
        &self,
    ) -> Option<crate::preprocess_audio::AudioPreprocessConfig> {
        crate::preprocess_audio::AudioPreprocessConfig::from_manifest(
            &self.manifest.preprocess_method,
        )
    }

    /// Get the manifest-declared confidence threshold, if any.
    ///
    /// Used by `spe detect-audio --visualize` to recover the user's intended
    /// output filter when running internal inference at threshold=0 (so the
    /// heatmap layer sees the full per-window confidence distribution while
    /// the JSON / merged-range output still respects the manifest default).
    pub fn audio_confidence_threshold(&self) -> Option<f32> {
        self.manifest.confidence_threshold
    }

    /// Get the audio inference window + stride from the manifest, if this
    /// model uses sliding-window inference. Returns `(window_s, stride_s)`
    /// or `None` for non-sliding-window models (image / single-shot audio).
    ///
    /// Used by `spe detect-audio --visualize` so the slot resolution and
    /// merge stride aren't hardcoded to one model's parameters.
    pub fn audio_window_stride(&self) -> Option<(f32, f32)> {
        match self.manifest.inference_strategy {
            crate::manifest::InferenceStrategy::SlidingWindow {
                segment_duration_s,
                segment_stride_s,
            } => Some((segment_duration_s, segment_stride_s)),
            _ => None,
        }
    }

    /// Returns the model type based on preprocessing + postprocessing method + subtype.
    pub fn model_type(&self) -> ModelType {
        derive_model_type(
            &self.manifest.preprocess_method,
            &self.manifest.postprocess_method,
            self.manifest.subtype,
        )
    }
}

// ---------------------------------------------------------------------------
// Output shape validation
// ---------------------------------------------------------------------------

/// Validate that the ONNX model's output shape matches the declared
/// postprocessing method. Rejects non-conforming models at load time.
fn validate_output_shape(session: &Session, manifest: &ModelManifest) -> Result<()> {
    let method = &manifest.postprocess_method;
    let outputs = session.outputs();

    if outputs.is_empty() {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: manifest.id.clone(),
            shape: "no outputs".to_string(),
            method: method.as_str().to_string(),
        });
    }

    let output_names: Vec<&str> = outputs.iter().map(|output| output.name()).collect();
    let output_index = select_validation_output_index(
        &output_names,
        &manifest.preprocess_method,
        method,
        &manifest.id,
    )?;
    let output = &outputs[output_index];
    let shape = output_shape_dims(output);
    validate_output_dims(&shape, &manifest.id, method)
}

fn select_validation_output_index(
    output_names: &[&str],
    preprocess: &PreprocessMethod,
    method: &PostprocessMethod,
    model_id: &str,
) -> Result<usize> {
    if output_names.is_empty() {
        return Err(SparrowEngineError::OutputShapeMismatch {
            id: model_id.to_string(),
            shape: "no outputs".to_string(),
            method: method.as_str().to_string(),
        });
    }
    if output_names.len() == 1 {
        return Ok(0);
    }
    if matches!(
        (preprocess, method),
        (
            PreprocessMethod::RawAudio { .. },
            PostprocessMethod::Softmax
        )
    ) {
        return output_names
            .iter()
            .position(|name| *name == "label")
            .ok_or_else(|| SparrowEngineError::OutputShapeMismatch {
                id: model_id.to_string(),
                shape: format!(
                    "multi-output RawAudio+Softmax missing required output named 'label'; outputs [{}]",
                    output_names.join(", ")
                ),
                method: method.as_str().to_string(),
            });
    }
    Ok(0)
}

fn validate_output_dims(shape: &[i64], model_id: &str, method: &PostprocessMethod) -> Result<()> {
    let shape_str = format_shape(shape);
    let method_str = method.as_str().to_string();

    match method {
        PostprocessMethod::YoloE2e => {
            // Expected: [N, 6] or [1, N, 6] where N is dynamic (-1) or
            // positive.
            if shape.len() != 2 && shape.len() != 3 {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: model_id.to_string(),
                    shape: shape_str,
                    method: method_str,
                });
            }
            let last_dim = shape[shape.len() - 1];
            // last_dim == 6 for conforming models; -1 is dynamic (acceptable).
            if last_dim != 6 && last_dim != -1 {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: model_id.to_string(),
                    shape: shape_str,
                    method: method.as_str().to_string(),
                });
            }
            match shape {
                [n, _] if *n == -1 || *n > 0 => {}
                [batch, n, _] if (*batch == 1 || *batch == -1) && (*n == -1 || *n > 0) => {}
                _ => {
                    return Err(SparrowEngineError::OutputShapeMismatch {
                        id: model_id.to_string(),
                        shape: shape_str,
                        method: method_str,
                    });
                }
            }
        }
        PostprocessMethod::MegadetV5a { .. } => {
            // Expected: [N, 5+C] or [1, N, 5+C] where N is dynamic (-1) or
            // positive.
            if shape.len() != 2 && shape.len() != 3 {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: model_id.to_string(),
                    shape: shape_str,
                    method: method_str,
                });
            }
            let last_dim = shape[shape.len() - 1];
            // Runtime MegaDet postprocess accepts any `[N, 5+C]` with `C > 0`
            // and tolerates `labels.len() != num_classes` via `unknown_<id>`
            // fallback, so CPU load-time shape validation must not pin the
            // static last dimension to a fixed class count.
            if last_dim != -1 && last_dim <= 5 {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: model_id.to_string(),
                    shape: shape_str,
                    method: method.as_str().to_string(),
                });
            }
            match shape {
                [n, _] if *n == -1 || *n > 0 => {}
                [batch, n, _] if (*batch == 1 || *batch == -1) && (*n == -1 || *n > 0) => {}
                _ => {
                    return Err(SparrowEngineError::OutputShapeMismatch {
                        id: model_id.to_string(),
                        shape: shape_str,
                        method: method_str,
                    });
                }
            }
        }
        PostprocessMethod::HeatmapPeaks { .. } => {
            // Expected: spatial output with rank >= 3 (batch, channels, H, W) or
            // (batch, H, W, channels). We check rank >= 3.
            if shape.len() < 3 {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: model_id.to_string(),
                    shape: shape_str,
                    method: method_str,
                });
            }
        }
        PostprocessMethod::Softmax => {
            // Expected: [1, num_classes] or [batch, num_classes] (rank 2).
            // Also accept rank 1 [num_classes] for some models.
            //
            // NOTE: This rejects rank > 2, which makes the 3D squeeze branch
            // in classify.rs (ndim == 3) unreachable. If this validation is
            // ever relaxed to allow rank 3, that branch will become live.
            if shape.is_empty() || shape.len() > 2 {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: model_id.to_string(),
                    shape: shape_str,
                    method: method_str,
                });
            }
        }
        PostprocessMethod::Sigmoid { .. } => {
            // Expected: [1, 1] or [batch, 1] (binary output). Accept rank 1 or 2.
            if shape.is_empty() || shape.len() > 2 {
                return Err(SparrowEngineError::OutputShapeMismatch {
                    id: model_id.to_string(),
                    shape: shape_str,
                    method: method_str,
                });
            }
        }
    }

    Ok(())
}

/// Extract dimension vector from an output outlet.
fn output_shape_dims(outlet: &ort::value::Outlet) -> Vec<i64> {
    match outlet.dtype() {
        ort::value::ValueType::Tensor { shape, .. } => shape.iter().copied().collect(),
        _ => vec![],
    }
}

/// Format a shape vector for error messages.
fn format_shape(dims: &[i64]) -> String {
    let parts: Vec<String> = dims
        .iter()
        .map(|d| {
            if *d < 0 {
                "?".to_string()
            } else {
                d.to_string()
            }
        })
        .collect();
    format!("[{}]", parts.join(", "))
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// `derive_model_type` was hoisted to `sparrow-engine-types/src/model_type.rs` as part
// of the Phase 3.8 Phase A crate split (C2 closure). Imported at top via
// `use sparrow_engine_types::derive_model_type;`.

/// Convert an `ort::Error` (with or without recovery payload) to `SparrowEngineError::Ort`.
pub(crate) fn ort_err<R: std::fmt::Display>(e: R) -> SparrowEngineError {
    SparrowEngineError::Ort(e.to_string())
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
        PathBuf::from("/tmp/bongo_test_models_nonexistent")
    }

    fn megadet_v5a_method() -> PostprocessMethod {
        PostprocessMethod::MegadetV5a {
            iou_threshold: 0.45,
        }
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
    fn singleton_enforcement() {
        // Reset global state for test isolation.
        ENGINE_EXISTS.store(false, Ordering::SeqCst);

        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config.clone());
        assert!(engine.is_ok(), "First engine should succeed");

        let config2 = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine2 = Engine::new(config2);
        assert!(
            matches!(engine2, Err(SparrowEngineError::EngineAlreadyExists)),
            "Second engine should fail with EngineAlreadyExists"
        );

        // Drop first engine, should allow creating a new one.
        drop(engine);
        let config3 = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine3 = Engine::new(config3);
        assert!(engine3.is_ok(), "Engine after drop should succeed");

        // Clean up.
        drop(engine3);
    }

    #[test]
    #[serial]
    fn config_defaults_cpu() {
        let config = EngineConfig::new(Device::Cpu, "/tmp");
        assert_eq!(config.inter_threads, 1);
        // CPU default: min(available_parallelism, 8), at least 1.
        let expected = std::thread::available_parallelism()
            .map(|n| (n.get() as u32).min(8))
            .unwrap_or(4);
        assert_eq!(config.intra_threads, expected);
        assert!(config.intra_threads >= 1 && config.intra_threads <= 8);
    }

    #[test]
    #[serial]
    fn config_defaults_auto() {
        let config = EngineConfig::new(Device::Auto, "/tmp");
        assert_eq!(config.inter_threads, 1);
        // Auto uses same CPU-optimized default (CUDA EP fallback safety).
        let expected = std::thread::available_parallelism()
            .map(|n| (n.get() as u32).min(8))
            .unwrap_or(4);
        assert_eq!(config.intra_threads, expected);
    }

    #[test]
    #[serial]
    fn config_defaults_gpu() {
        let config = EngineConfig::new(Device::Cuda(0), "/tmp");
        assert_eq!(config.inter_threads, 1);
        assert_eq!(config.intra_threads, 1);
    }

    #[test]
    #[serial]
    fn format_shape_display() {
        assert_eq!(format_shape(&[1, -1, 6]), "[1, ?, 6]");
        assert_eq!(format_shape(&[1, 3]), "[1, 3]");
        assert_eq!(format_shape(&[]), "[]");
    }

    fn raw_audio_preprocess() -> PreprocessMethod {
        PreprocessMethod::RawAudio {
            sample_rate: 32_000,
            window_samples: 160_000,
            pass_orig_sample_rate: false,
        }
    }

    #[test]
    #[serial]
    fn select_validation_output_index_keeps_single_output_raw_softmax() {
        assert_eq!(
            select_validation_output_index(
                &["embedding"],
                &raw_audio_preprocess(),
                &PostprocessMethod::Softmax,
                "model",
            )
            .unwrap(),
            0
        );
    }

    #[test]
    #[serial]
    fn select_validation_output_index_uses_label_for_multi_output_raw_softmax() {
        assert_eq!(
            select_validation_output_index(
                &["embedding", "spatial_embedding", "spectrogram", "label"],
                &raw_audio_preprocess(),
                &PostprocessMethod::Softmax,
                "perch2",
            )
            .unwrap(),
            3
        );
    }

    #[test]
    #[serial]
    fn select_validation_output_index_rejects_multi_output_raw_softmax_without_label() {
        let err = select_validation_output_index(
            &["embedding", "spectrogram"],
            &raw_audio_preprocess(),
            &PostprocessMethod::Softmax,
            "bad-audio",
        )
        .unwrap_err();

        match err {
            SparrowEngineError::OutputShapeMismatch { shape, .. } => {
                assert!(shape.contains("missing required output named 'label'"));
                assert!(shape.contains("embedding"));
                assert!(shape.contains("spectrogram"));
            }
            other => panic!("unexpected error: {other:?}"),
        }
    }

    #[test]
    #[serial]
    fn select_validation_output_index_keeps_first_output_for_non_raw_softmax() {
        assert_eq!(
            select_validation_output_index(
                &["scores", "aux"],
                &PreprocessMethod::Resize,
                &PostprocessMethod::Softmax,
                "image-classifier",
            )
            .unwrap(),
            0
        );
    }

    #[test]
    #[serial]
    fn model_type_from_preprocess_postprocess() {
        use crate::types::ModelSubtype;
        let std_sub = ModelSubtype::Standard;
        // Vision models with Standard subtype.
        assert_eq!(
            derive_model_type(
                &PreprocessMethod::Letterbox,
                &PostprocessMethod::Softmax,
                std_sub,
            ),
            ModelType::Classifier
        );
        assert_eq!(
            derive_model_type(
                &PreprocessMethod::Letterbox,
                &PostprocessMethod::YoloE2e,
                std_sub,
            ),
            ModelType::Detector
        );
        assert_eq!(
            derive_model_type(
                &PreprocessMethod::Letterbox,
                &PostprocessMethod::MegadetV5a {
                    iou_threshold: 0.45
                },
                std_sub,
            ),
            ModelType::Detector
        );
        assert_eq!(
            derive_model_type(
                &PreprocessMethod::Resize,
                &PostprocessMethod::HeatmapPeaks {
                    peak_threshold: 0.1,
                    adaptive: true,
                    point_to_box_half_size: 10,
                },
                std_sub,
            ),
            ModelType::Detector
        );
        // Overhead subtype promotes Detector → OverheadDetector (Phase 3.5 S3).
        assert_eq!(
            derive_model_type(
                &PreprocessMethod::Resize,
                &PostprocessMethod::HeatmapPeaks {
                    peak_threshold: 0.1,
                    adaptive: true,
                    point_to_box_half_size: 10,
                },
                ModelSubtype::Overhead,
            ),
            ModelType::OverheadDetector
        );
        // Overhead does NOT promote non-detector types.
        assert_eq!(
            derive_model_type(
                &PreprocessMethod::Letterbox,
                &PostprocessMethod::Softmax,
                ModelSubtype::Overhead,
            ),
            ModelType::Classifier,
            "Overhead hint must be ignored for classifiers"
        );
        // Audio models.
        let mel = PreprocessMethod::MelSpectrogram {
            sample_rate: 48000,
            n_fft: 1024,
            hop_length: 512,
            n_mels: 224,
            fmin: 0.0,
            fmax: 24000.0,
            top_db: 80.0,
            window: "hann_symmetric".to_string(),
            mel_scale: "slaney".to_string(),
            filter_norm: "slaney".to_string(),
            fill_highfreq: false,
        };
        assert_eq!(
            derive_model_type(
                &mel,
                &PostprocessMethod::Sigmoid {
                    confidence_threshold: 0.5
                },
                std_sub,
            ),
            ModelType::AudioDetector
        );
        assert_eq!(
            derive_model_type(&mel, &PostprocessMethod::Softmax, std_sub),
            ModelType::AudioClassifier,
            "RP-39: MelSpectrogram + Softmax (mel-input ecotype re-export) is an AudioClassifier"
        );
        // Overhead does NOT promote audio detectors.
        assert_eq!(
            derive_model_type(
                &mel,
                &PostprocessMethod::Sigmoid {
                    confidence_threshold: 0.5
                },
                ModelSubtype::Overhead,
            ),
            ModelType::AudioDetector,
            "Overhead hint must be ignored for audio detectors"
        );
    }

    #[test]
    #[serial]
    fn loaded_models_empty_on_new_engine() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();
        assert!(engine.loaded_models().is_empty());
        drop(engine);
    }

    #[test]
    #[serial]
    fn unload_pipeline_not_found() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();
        let err = engine.unload_pipeline("nonexistent").unwrap_err();
        assert!(matches!(err, SparrowEngineError::PipelineNotFound { .. }));
        drop(engine);
    }

    #[test]
    #[serial]
    fn get_pipeline_not_found() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();
        let err = engine.get_pipeline("nonexistent").unwrap_err();
        assert!(matches!(err, SparrowEngineError::PipelineNotFound { .. }));
        drop(engine);
    }

    #[test]
    #[serial]
    fn register_pipeline_manifest_round_trips_and_unloads() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();
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
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();
        assert!(engine.get_model_handle("nonexistent").is_none());
        drop(engine);
    }

    #[test]
    #[serial]
    fn load_model_manifest_not_found() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();
        let err = engine
            .load_model("/nonexistent/path/manifest.toml")
            .unwrap_err();
        assert!(matches!(err, SparrowEngineError::ManifestNotFound(_)));
        drop(engine);
    }

    // -----------------------------------------------------------------------
    // Helpers for map-level tests (inject LoadedModel directly, no real ONNX)
    // -----------------------------------------------------------------------

    /// Create a dummy ORT session from the minimal identity ONNX fixture.
    /// Requires /tmp/bongo_test_identity.onnx (76 bytes, created by test setup).
    fn dummy_session() -> Arc<Mutex<Session>> {
        let session = Session::builder()
            .unwrap()
            .commit_from_file("/tmp/bongo_test_identity.onnx")
            .unwrap();
        Arc::new(Mutex::new(session))
    }

    fn dummy_manifest(id: &str) -> Arc<ModelManifest> {
        Arc::new(ModelManifest {
            id: id.to_string(),
            format: "onnx".to_string(),
            model_file: "model.onnx".to_string(),
            model_file_fp16: None,
            preprocess_method: manifest::PreprocessMethod::Letterbox,
            input_size: Some([640, 640]),
            layout: Some(manifest::Layout::Nchw),
            normalization: Some(manifest::Normalization::Unit),
            pad_value: Some(114.0),
            channel_order: Some(manifest::ChannelOrder::Rgb),
            precision: manifest::Precision::Fp32,
            inference_strategy: manifest::InferenceStrategy::Single,
            trt: None,
            postprocess_method: PostprocessMethod::Softmax,
            confidence_threshold: None,
            label_file: Some("labels.txt".to_string()),
            label_format: Some(manifest::LabelFormat::OnePerLine),
            default: false,
            subtype: crate::types::ModelSubtype::Standard,
            onnx_sha256: None,
            onnx_size_bytes: None,
            version: None,
            description: None,
            provenance: None,
            drift_reference: None,
        })
    }

    /// Insert a synthetic LoadedModel into the engine's map and return
    /// a corresponding ModelHandle. Uses a real (minimal) ORT session.
    fn inject_model(engine: &Engine, id: &str) -> ModelHandle {
        let session = dummy_session();
        let active = Arc::new(AtomicBool::new(true));
        let manifest = dummy_manifest(id);
        let labels = Arc::new(vec!["class0".to_string()]);
        let last_used = Arc::new(AtomicU64::new(now_millis()));

        let loaded = LoadedModel {
            session: Arc::clone(&session),
            manifest: Arc::clone(&manifest),
            labels: Arc::clone(&labels),
            active: Arc::clone(&active),
            path: PathBuf::from("/tmp/fake_manifest.toml"),
            last_used,
        };

        engine
            .models
            .write()
            .expect("models lock poisoned")
            .insert(id.to_string(), loaded);

        ModelHandle {
            engine_ref: Arc::downgrade(&engine.inner),
            active,
            session,
            manifest,
            labels,
            model_id: id.to_string(),
        }
    }

    // -----------------------------------------------------------------------
    // Idle-unload reaper (Phase 4.2 follow-up)
    // -----------------------------------------------------------------------

    #[test]
    #[serial]
    fn reaper_unloads_stale_and_keeps_recent() {
        // Load three models: ancient + stale + fresh. With keep_last_n=1 and
        // idle threshold 1s, expect ancient + stale unloaded; fresh protected
        // by the keep-last-N policy.
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let _ancient = inject_model(&engine, "ancient");
        let _stale = inject_model(&engine, "stale");
        let _fresh = inject_model(&engine, "fresh");

        let now = now_millis();
        // Reach into the model map and rewrite last_used to simulate ages.
        {
            let models = engine.models.read().expect("models lock");
            models["ancient"]
                .last_used
                .store(now.saturating_sub(10 * 60 * 1000), Ordering::Relaxed);
            models["stale"]
                .last_used
                .store(now.saturating_sub(5_000), Ordering::Relaxed);
            models["fresh"].last_used.store(now, Ordering::Relaxed);
        }

        let unloaded = engine.reap_idle_models(1_000, 1);
        assert!(unloaded.iter().all(|id| id != "fresh"));
        assert_eq!(
            unloaded
                .iter()
                .filter(|id| *id == "stale" || *id == "ancient")
                .count(),
            2,
            "ancient + stale should both be unloaded; got {unloaded:?}"
        );
        assert!(engine.get_model_handle("fresh").is_some());
        assert!(engine.get_model_handle("stale").is_none());
        assert!(engine.get_model_handle("ancient").is_none());

        drop(engine);
    }

    #[test]
    #[serial]
    fn reaper_keeps_last_n_protects_even_if_idle() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let _a = inject_model(&engine, "a");
        let _b = inject_model(&engine, "b");
        let _c = inject_model(&engine, "c");

        let base = now_millis().saturating_sub(60_000);
        {
            let models = engine.models.read().expect("models lock");
            models["a"].last_used.store(base, Ordering::Relaxed);
            models["b"].last_used.store(base + 1_000, Ordering::Relaxed);
            models["c"].last_used.store(base + 2_000, Ordering::Relaxed);
        }

        let unloaded = engine.reap_idle_models(1_000, 2);
        assert_eq!(unloaded, vec!["a".to_string()]);
        assert!(engine.get_model_handle("a").is_none());
        assert!(engine.get_model_handle("b").is_some());
        assert!(engine.get_model_handle("c").is_some());

        drop(engine);
    }

    #[test]
    #[serial]
    fn reaper_noop_when_all_recent() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let _a = inject_model(&engine, "a");
        let _b = inject_model(&engine, "b");

        let unloaded = engine.reap_idle_models(60_000, 1);
        assert!(unloaded.is_empty(), "no stale models should be unloaded");
        assert!(engine.get_model_handle("a").is_some());
        assert!(engine.get_model_handle("b").is_some());

        drop(engine);
    }

    #[test]
    #[serial]
    fn get_model_handle_updates_last_used() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let _h = inject_model(&engine, "m");
        {
            let models = engine.models.read().expect("models lock");
            models["m"]
                .last_used
                .store(now_millis().saturating_sub(60_000), Ordering::Relaxed);
        }

        let before = now_millis();
        let _handle = engine.get_model_handle("m").expect("model still loaded");

        let observed = {
            let models = engine.models.read().expect("models lock");
            models["m"].last_used.load(Ordering::Relaxed)
        };
        assert!(
            observed >= before,
            "last_used should advance to the current logical tick; before={before}, observed={observed}"
        );

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
    #[serial]
    fn reaper_keeps_model_touched_after_snapshot() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let _handle = inject_model(&engine, "foo");
        let snapshot = {
            let models = engine.models.read().expect("models lock");
            let entry = &models["foo"];
            entry
                .last_used
                .store(now_millis().saturating_sub(60_000), Ordering::Relaxed);
            (
                entry.last_used.load(Ordering::Relaxed),
                Arc::clone(&entry.active),
            )
        };

        let _ = engine
            .get_model_handle("foo")
            .expect("touch should keep model loaded");

        let unloaded =
            engine.unload_idle_snapshot("foo", snapshot.0, &snapshot.1, now_millis(), 1_000);
        assert!(
            !unloaded,
            "touched model must survive a stale reaper snapshot"
        );
        assert!(engine.get_model_handle("foo").is_some());

        drop(engine);
    }

    #[test]
    #[serial]
    fn reaper_keeps_replacement_generation_after_stale_snapshot() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let handle_a = inject_model(&engine, "foo");
        let snapshot = {
            let models = engine.models.read().expect("models lock");
            let entry = &models["foo"];
            entry
                .last_used
                .store(now_millis().saturating_sub(60_000), Ordering::Relaxed);
            (
                entry.last_used.load(Ordering::Relaxed),
                Arc::clone(&entry.active),
            )
        };

        let handle_b = inject_model(&engine, "foo");
        handle_a.active.store(false, Ordering::Release);

        let unloaded =
            engine.unload_idle_snapshot("foo", snapshot.0, &snapshot.1, now_millis(), 1_000);
        assert!(
            !unloaded,
            "reaper must not remove a fresh replacement generation"
        );
        let lookup = engine
            .get_model_handle("foo")
            .expect("replacement model must remain loaded");
        assert!(Arc::ptr_eq(&lookup.active, &handle_b.active));

        drop(engine);
    }

    #[test]
    #[serial]
    fn get_or_load_reports_poisoned_loading_lock() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _guard = engine.loading_lock.lock().expect("loading lock");
            panic!("poison loading lock for test");
        }));

        let err = engine
            .get_or_load_model("foo")
            .expect_err("poisoned loading lock must return an error");
        assert!(
            matches!(err, SparrowEngineError::Ort(msg) if msg.contains("loading_lock poisoned"))
        );

        drop(engine);
    }

    // -----------------------------------------------------------------------
    // M5: unload_model TOCTOU race fix
    // -----------------------------------------------------------------------

    #[test]
    #[serial]
    fn test_unload_does_not_remove_replacement() {
        // Scenario: load model A, replace with model B (same ID), then
        // unload using old handle A. Model B must survive in the map.
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        // Insert model A under ID "foo".
        let handle_a = inject_model(&engine, "foo");

        // Replace with model B under same ID "foo".
        let handle_b = inject_model(&engine, "foo");

        // handle_a's active flag was NOT automatically cleared by inject_model
        // (unlike load_model which marks old active=false). Simulate what
        // load_model does: mark handle_a inactive.
        handle_a.active.store(false, Ordering::Release);

        // Attempt unload with old handle_a — should fail (already inactive)
        // and must NOT remove handle_b from the map.
        let result = engine.unload_model(&handle_a);
        assert!(
            matches!(result, Err(SparrowEngineError::ModelUnloaded)),
            "unload of already-inactive handle should return ModelUnloaded"
        );

        // Verify model B is still in the map.
        let lookup = engine.get_model_handle("foo");
        assert!(
            lookup.is_some(),
            "replacement model must still be in the map"
        );
        assert!(
            Arc::ptr_eq(&lookup.unwrap().active, &handle_b.active),
            "map entry must be handle_b, not handle_a"
        );

        drop(engine);
    }

    // -----------------------------------------------------------------------
    // Phase 4.2 lazy-load contract: get_or_load_model fast-path
    // -----------------------------------------------------------------------

    #[test]
    #[serial]
    fn get_or_load_returns_cached_handle_after_first_load() {
        // FFI `sparrow_engine_load_model_by_id`, CLI `cmd_detect`/`cmd_classify`/
        // `cmd_detect_audio`/`cmd_pipeline`, Python `detect`/`classify`/
        // `detect_audio`/`pipeline`, and HTTP `/v1/models/load` + the
        // 4 inference endpoints all route through `Engine::get_or_load_model`.
        // Contract: a repeat call must return a handle pointing at the SAME
        // `LoadedModel` (no ORT session re-creation, no map replacement).
        // Pre-Phase-4.2, the FFI + CLI used `load_model_by_id`, which
        // unconditionally re-loaded and invalidated prior handles.
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let h1 = inject_model(&engine, "foo");
        let h2 = engine
            .get_or_load_model("foo")
            .expect("fast-path get_or_load on already-loaded model");

        assert!(
            Arc::ptr_eq(h1.manifest(), h2.manifest()),
            "fast-path get_or_load must return cached manifest, not a re-load"
        );
        assert!(
            Arc::ptr_eq(&h1.active, &h2.active),
            "fast-path get_or_load must reuse the same active flag"
        );

        drop(engine);
    }

    // -----------------------------------------------------------------------
    // M4: get_model_handles atomic batch lookup
    // -----------------------------------------------------------------------

    #[test]
    #[serial]
    fn test_get_model_handles_all_found() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let _h1 = inject_model(&engine, "alpha");
        let _h2 = inject_model(&engine, "beta");

        let (found, missing) = engine.get_model_handles(&["alpha", "beta"]);
        assert_eq!(found.len(), 2, "both models should be found");
        assert!(missing.is_empty(), "no models should be missing");
        assert_eq!(found[0].model_id(), "alpha");
        assert_eq!(found[1].model_id(), "beta");

        drop(engine);
    }

    #[test]
    #[serial]
    fn test_get_model_handles_reports_missing() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let _h1 = inject_model(&engine, "alpha");

        let (found, missing) = engine.get_model_handles(&["alpha", "ghost"]);
        assert_eq!(found.len(), 1, "only alpha should be found");
        assert_eq!(found[0].model_id(), "alpha");
        assert_eq!(missing, vec!["ghost".to_string()]);

        drop(engine);
    }

    // -----------------------------------------------------------------------
    // active_device resolves Auto
    // -----------------------------------------------------------------------

    #[test]
    #[serial]
    fn test_active_device_auto_resolves_to_cpu() {
        // Phase 4.1 MT-4.1-2: sparrow-engine-cpu is strictly CPU; Auto resolves to Cpu.
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Auto, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        assert_eq!(
            *engine.active_device(),
            Device::Cpu,
            "sparrow-engine-cpu Auto must resolve to Cpu (CUDA EP not compiled in)"
        );

        drop(engine);
    }

    #[test]
    #[serial]
    fn test_active_device_cpu_stays_cpu() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        assert_eq!(
            *engine.active_device(),
            Device::Cpu,
            "explicit Cpu must remain Cpu"
        );

        drop(engine);
    }

    #[test]
    #[serial]
    fn test_explicit_cuda_coerces_to_cpu() {
        // Phase 4.1 MT-4.1-2: sparrow-engine-cpu silently coerces Device::Cuda(n) to
        // Device::Cpu (with a tracing::warn!). Use sparrow-engine-gpu for actual CUDA
        // inference.
        ENGINE_EXISTS.store(false, Ordering::SeqCst);

        let resolved = Engine::resolve_device(&Device::Cuda(0));
        assert_eq!(
            resolved,
            Device::Cpu,
            "sparrow-engine-cpu must coerce Device::Cuda(0) to Device::Cpu (CUDA EP not compiled in)"
        );

        let resolved_n = Engine::resolve_device(&Device::Cuda(3));
        assert_eq!(
            resolved_n,
            Device::Cpu,
            "sparrow-engine-cpu must coerce Device::Cuda(3) to Device::Cpu (CUDA EP not compiled in)"
        );

        // No engine created — resolve_device is a pure function, no singleton needed.
    }

    #[test]
    #[serial]
    fn test_resolve_device_auto_returns_cpu() {
        // sparrow-engine-cpu's resolve_device(Auto) deterministically returns Cpu —
        // CUDA EP support is not compiled in, so no runtime probing happens.
        assert_eq!(
            Engine::resolve_device(&Device::Auto),
            Device::Cpu,
            "sparrow-engine-cpu resolve_device(Auto) must always return Cpu"
        );
    }

    // -----------------------------------------------------------------------
    // NS-1: ModelHandle::manifest() + ModelHandle::labels() accessors
    //
    // Phase 3.8 Phase C audit-fix R1 NS1-MINOR-1: the NS-1 carry-forward
    // commit `956af03` added two accessors mirroring sparrow_engine_gpu's signatures
    // but shipped no test. Verifies (a) the accessors return references
    // tied to &self (not temp-aliased), (b) the inner Arc is shared with
    // the engine's loaded-model record (no clone), (c) the manifest field
    // round-trips the original id, (d) labels surfaces the populated
    // label vector for classifier-style models.
    // -----------------------------------------------------------------------

    #[test]
    #[serial]
    fn test_model_handle_manifest_accessor() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let handle = inject_model(&engine, "ns1_manifest_model");

        let manifest_ref: &Arc<ModelManifest> = handle.manifest();
        assert_eq!(
            manifest_ref.id, "ns1_manifest_model",
            "manifest().id must round-trip the model id used to construct the handle"
        );
        assert_eq!(
            manifest_ref.format, "onnx",
            "manifest().format must surface the dummy_manifest field"
        );
        assert_eq!(
            manifest_ref.input_size,
            Some([640, 640]),
            "manifest().input_size must surface the dummy_manifest field"
        );
        // manifest() must return a reference to the SAME Arc the engine's
        // loaded-model record holds (no clone). Verify by inspecting
        // strong_count after grabbing the loaded entry: the Arc is shared
        // between the LoadedModel record AND the ModelHandle, so count >= 2.
        let strong = Arc::strong_count(manifest_ref);
        assert!(
            strong >= 2,
            "manifest() Arc strong_count must be >= 2 (handle + engine map share); got {strong}"
        );

        drop(engine);
    }

    #[test]
    fn validate_output_dims_accepts_megadet_static_last_dim_above_five() {
        validate_output_dims(&[1, 8400, 8], "mdv5a", &megadet_v5a_method())
            .expect("megadet static last_dim > 5 should be accepted");
    }

    #[test]
    fn validate_output_dims_accepts_rank_two_megadet_static_last_dim_above_five() {
        validate_output_dims(&[8400, 8], "mdv5a-rank-two", &megadet_v5a_method())
            .expect("rank-2 megadet static last_dim > 5 should be accepted");
    }

    #[test]
    fn validate_output_dims_accepts_rank_two_yolo_e2e() {
        validate_output_dims(&[8400, 6], "yolo-rank-two", &PostprocessMethod::YoloE2e)
            .expect("rank-2 yolo_e2e [N, 6] should be accepted");
    }

    #[test]
    fn validate_output_dims_rejects_rank_two_yolo_e2e_wrong_last() {
        let err = validate_output_dims(
            &[8400, 5],
            "bad-yolo-rank-two-last",
            &PostprocessMethod::YoloE2e,
        )
        .expect_err("rank-2 yolo_e2e [N, 5] must be rejected");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));
    }

    #[test]
    fn validate_output_dims_accepts_rank_three_yolo_e2e() {
        validate_output_dims(
            &[1, 8400, 6],
            "yolo-rank-three",
            &PostprocessMethod::YoloE2e,
        )
        .expect("rank-3 yolo_e2e [1, N, 6] should be accepted");
    }

    #[test]
    fn validate_output_dims_keeps_yolo_strict_at_six() {
        let err = validate_output_dims(&[1, 8400, 8], "bad-yolo-last", &PostprocessMethod::YoloE2e)
            .expect_err("yolo_e2e static last_dim > 6 must still be rejected");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));
    }

    #[test]
    fn validate_output_dims_rejects_megadet_static_last_dim_at_or_below_five() {
        let err = validate_output_dims(&[1, 8400, 5], "bad-last-5", &megadet_v5a_method())
            .expect_err("megadet static last_dim == 5 must be rejected");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));

        let err = validate_output_dims(&[1, 8400, 4], "bad-last-4", &megadet_v5a_method())
            .expect_err("megadet static last_dim < 5 must be rejected");
        assert!(matches!(
            err,
            SparrowEngineError::OutputShapeMismatch { .. }
        ));
    }

    #[test]
    #[serial]
    fn test_model_handle_labels_accessor() {
        ENGINE_EXISTS.store(false, Ordering::SeqCst);
        let config = EngineConfig::new(Device::Cpu, dummy_model_dir());
        let engine = Engine::new(config).unwrap();

        let handle = inject_model(&engine, "ns1_labels_model");

        let labels_ref: &Arc<Vec<String>> = handle.labels();
        assert_eq!(
            labels_ref.len(),
            1,
            "labels() must surface the dummy 1-class label vector"
        );
        assert_eq!(
            labels_ref[0], "class0",
            "labels()[0] must round-trip the dummy class name"
        );
        let strong = Arc::strong_count(labels_ref);
        assert!(
            strong >= 2,
            "labels() Arc strong_count must be >= 2 (handle + engine map share); got {strong}"
        );

        drop(engine);
    }

    #[test]
    #[serial]
    fn resolve_default_model_strict_env_var_falls_through_on_type_mismatch() {
        // RP-8: SPARROW_ENGINE_DEFAULT_DETECTOR pointing at an OverheadDetector
        // when a Detector is requested must fall through to the manifest scan
        // (with a tracing::warn!), not silently widen the env-var result.
        ENGINE_EXISTS.store(false, Ordering::SeqCst);

        let dir = tempfile::tempdir().unwrap();

        // Standard Detector manifest (letterbox + yolo_e2e).
        let std_dir = dir.path().join("std-det");
        std::fs::create_dir(&std_dir).unwrap();
        std::fs::write(
            std_dir.join("manifest.toml"),
            r#"
[model]
id = "std-det"
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
        std::fs::write(std_dir.join("labels.txt"), "animal\n").unwrap();

        // OverheadDetector manifest (resize + heatmap_peaks + subtype = "overhead").
        let ovh_dir = dir.path().join("ovh-det");
        std::fs::create_dir(&ovh_dir).unwrap();
        std::fs::write(
            ovh_dir.join("manifest.toml"),
            r#"
[model]
id = "ovh-det"
format = "onnx"
file = "model.onnx"
subtype = "overhead"

[preprocessing]
method = "resize"
input_size = [512, 512]
layout = "nchw"
normalization = "imagenet"

[inference]
strategy = "tiled"
tile_size = [512, 512]
tile_overlap = 0

[postprocessing]
method = "heatmap_peaks"
peak_threshold = 0.2
adaptive = false
point_to_box_half_size = 10

[labels]
file = "labels.txt"
format = "name_index_csv"
"#,
        )
        .unwrap();
        std::fs::write(ovh_dir.join("labels.txt"), "0,animal\n").unwrap();

        let config = EngineConfig::new(Device::Cpu, dir.path().to_path_buf());
        let engine = Engine::new(config).unwrap();

        // Sanity: scan picks up both, with the expected model_types.
        let available = engine.list_available_models();
        assert_eq!(available.len(), 2, "scan must find both manifests");
        let by_id: std::collections::HashMap<_, _> = available
            .iter()
            .map(|m| (m.id.as_str(), m.model_type))
            .collect();
        assert_eq!(by_id["std-det"], ModelType::Detector);
        assert_eq!(by_id["ovh-det"], ModelType::OverheadDetector);

        // Clear inherited env state from other tests.
        std::env::remove_var("SPARROW_ENGINE_DEFAULT_DETECTOR");

        // Case 1: env points at OverheadDetector but caller asks for Detector
        // → falls through to scan, returns the unique standard Detector.
        std::env::set_var("SPARROW_ENGINE_DEFAULT_DETECTOR", "ovh-det");
        assert_eq!(
            engine.resolve_default_model(ModelType::Detector),
            Some("std-det".to_string()),
            "env-var widening must NOT bypass type check; falls through to scan",
        );

        // Case 2: same env var, caller asks for OverheadDetector → match.
        assert_eq!(
            engine.resolve_default_model(ModelType::OverheadDetector),
            Some("ovh-det".to_string()),
            "env-var resolves cleanly when requested type matches",
        );

        // Case 3: env points at the standard Detector, caller asks for Detector.
        std::env::set_var("SPARROW_ENGINE_DEFAULT_DETECTOR", "std-det");
        assert_eq!(
            engine.resolve_default_model(ModelType::Detector),
            Some("std-det".to_string()),
        );

        // Case 4: env points at an unknown ID — preserve existing behavior
        // (return the value so downstream "model not found" surfaces the typo).
        std::env::set_var("SPARROW_ENGINE_DEFAULT_DETECTOR", "no-such-model");
        assert_eq!(
            engine.resolve_default_model(ModelType::Detector),
            Some("no-such-model".to_string()),
            "unknown env-var IDs pass through unchanged for downstream error surface",
        );

        // Cleanup.
        std::env::remove_var("SPARROW_ENGINE_DEFAULT_DETECTOR");
        drop(engine);
    }
}
