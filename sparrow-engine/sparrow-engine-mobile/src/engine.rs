//! Manifest-driven mobile inference engine (LiteRT/TFLite backend).
//!
//! RP-25-FU-1: the generic, manifest-driven peer of `sparrow-engine-cpu::Engine`
//! and `sparrow-engine-gpu::Engine`, on the LiteRT backend. It replaces the
//! hardcoded 5-export orca cascade with a model catalog the engine loads by id,
//! generic single-model audio detection, and a config-described audio cascade
//! ([`crate::pipeline`]) — the orca cascade is now a `pipeline.toml`, not C code.
//!
//! ## Threading contract (single-threaded / thread-affine)
//!
//! LiteRT compiled models are `&mut`-invoked and the runtime is `Rc`-based, so
//! the engine is **not** thread-safe. Create, use, AND free one `Engine` on a
//! single thread. The engine records its creating thread; the inference and
//! model/pipeline operations actively reject calls from any other thread with a
//! clear error. Teardown (`engine_free` / `unload_model`) assumes the contract
//! is honored — calling it from another thread while the owner thread is mid-call
//! is undefined behaviour (a non-atomic `Rc`/`Weak` refcount race), the same
//! hazard any `!Send` handle carries; it is not separately re-checked because a
//! `void` free cannot surface an error. (JNI / water-sparrow consume from a
//! single inference thread, so this is honored in practice.)
//!
//! ## Image inference (deferred — RP-42)
//!
//! [`MobileModel::detect`] / [`MobileModel::classify`] are exposed for ABI stability but
//! return a clear error: no mobile (`.tflite`) image model is onboarded yet. The
//! image preprocessing + decode path lands with the first converted image model
//! in RP-42 (the ONNX→TFLite conversion + onboarding task).

use std::cell::RefCell;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::rc::{Rc, Weak};
use std::sync::Arc;
use std::thread::ThreadId;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};

use sparrow_engine_core::catalog;
use sparrow_engine_core::preprocess_audio::{
    compute_segment_offsets, load_audio_at_sample_rate, mel_spectrogram, segment_time_range,
    AudioPreprocessConfig, MelFilterbank,
};
use sparrow_engine_types::manifest::{
    self, InferenceStrategy, ModelManifest, PostprocessMethod,
};
use sparrow_engine_types::derive_model_type;
use sparrow_engine_types::types::{
    AudioClass, AudioDetectOpts, AudioDetectResult, AudioInput, AudioSegment, ModelInfo, ModelType,
};
pub use sparrow_engine_types::EngineConfig;

use crate::cascade::{nchw_mel_to_nhwc_le_bytes, sigmoid, softmax};
use crate::sys::LiteRtElementType;
use crate::tflite::{LiteRtBackend, LiteRtRuntime};

/// Default number of top classes returned per segment by a multi-class audio
/// classifier (mirrors the cpu flavor's `DEFAULT_AUDIO_CLASSIFIER_TOP_K`).
const DEFAULT_AUDIO_CLASSIFIER_TOP_K: usize = 5;

/// One model loaded into the LiteRT runtime, with its manifest + labels.
///
/// `backend` is `RefCell`-wrapped because [`LiteRtBackend::invoke_single`] takes
/// `&mut self`; the engine is thread-affine so the borrow is always single-thread.
pub(crate) struct LoadedModel {
    pub(crate) id: String,
    pub(crate) backend: RefCell<LiteRtBackend>,
    pub(crate) manifest: Arc<ModelManifest>,
    pub(crate) labels: Arc<Vec<String>>,
    pub(crate) model_type: ModelType,
}

/// Shared engine state. Held by the public [`Engine`] (strong) and by every
/// [`MobileModel`] handle (weak), so a freed engine invalidates live handles
/// instead of dangling.
pub(crate) struct EngineInner {
    runtime: LiteRtRuntime,
    model_dir: PathBuf,
    num_threads: usize,
    owner_thread: ThreadId,
    models: RefCell<HashMap<String, Rc<LoadedModel>>>,
    pipelines: RefCell<HashMap<String, Rc<crate::pipeline::MobilePipeline>>>,
}

impl EngineInner {
    /// Reject any call from a thread other than the one that created the engine.
    pub(crate) fn check_thread(&self) -> Result<()> {
        if std::thread::current().id() != self.owner_thread {
            bail!(
                "sparrow-engine-mobile Engine is single-threaded: it was created on a different \
                 thread and must only be used from that thread"
            );
        }
        Ok(())
    }

    pub(crate) fn model_dir(&self) -> &Path {
        &self.model_dir
    }

    /// Get an already-loaded model by id, if present.
    pub(crate) fn get_model(&self, id: &str) -> Option<Rc<LoadedModel>> {
        self.models.borrow().get(id).cloned()
    }

    /// Load a model by id (idempotent: returns the existing handle if loaded).
    pub(crate) fn load_model(&self, id: &str) -> Result<Rc<LoadedModel>> {
        self.check_thread()?;
        if let Some(existing) = self.get_model(id) {
            return Ok(existing);
        }
        let loaded = Rc::new(self.load_model_uncached(id)?);
        self.models
            .borrow_mut()
            .insert(id.to_string(), Rc::clone(&loaded));
        Ok(loaded)
    }

    /// Load a model from its manifest without touching the cache.
    fn load_model_uncached(&self, id: &str) -> Result<LoadedModel> {
        catalog::validate_model_id(id).map_err(|e| anyhow!("{e}"))?;
        let manifest_path = self.model_dir.join(id).join("manifest.toml");
        let manifest = manifest::load_manifest(&manifest_path)
            .map_err(|e| anyhow!("load manifest {}: {e}", manifest_path.display()))?;

        // Flavor-strict: the mobile flavor's LiteRT backend loads `.tflite` only.
        // The shared loader also accepts ONNX (for the cpu/gpu ORT flavors); reject
        // a non-TFLite format here with a clear error.
        if manifest.format != "tflite" {
            bail!(
                "model '{id}' has format '{}', but the mobile (LiteRT) flavor loads only 'tflite' \
                 models; use the cpu/gpu flavor for ONNX models",
                manifest.format
            );
        }

        let manifest_dir = manifest_path.parent().unwrap_or_else(|| Path::new("."));
        // TFLite bakes precision into the single `file`; there is no fp32/fp16 file
        // pair (unlike ONNX), so always load `model_file`.
        let model_path = manifest_dir.join(&manifest.model_file);

        let labels = match (&manifest.label_file, &manifest.label_format) {
            (Some(file), Some(fmt)) => {
                let label_path = manifest_dir.join(file);
                manifest::load_labels(&label_path, fmt).map_err(|e| anyhow!("{e}"))?
            }
            _ => Vec::new(),
        };

        let backend = self
            .runtime
            .load(&model_path, self.num_threads)
            .with_context(|| format!("load tflite model {}", model_path.display()))?;

        let model_type = derive_model_type(
            &manifest.preprocess_method,
            &manifest.postprocess_method,
            manifest.subtype,
        );

        Ok(LoadedModel {
            id: id.to_string(),
            backend: RefCell::new(backend),
            manifest: Arc::new(manifest),
            labels: Arc::new(labels),
            model_type,
        })
    }

    /// Remove a model from the cache (no-op if not loaded).
    pub(crate) fn unload_model(&self, id: &str) -> Result<()> {
        self.check_thread()?;
        self.models.borrow_mut().remove(id);
        Ok(())
    }

    /// All models discoverable on disk in the model dir (loaded or not).
    pub(crate) fn list_models(&self) -> Vec<ModelInfo> {
        catalog::list_available_models(&self.model_dir)
    }

    pub(crate) fn pipelines(&self) -> &RefCell<HashMap<String, Rc<crate::pipeline::MobilePipeline>>> {
        &self.pipelines
    }

    /// Generic single-model audio detection (mel detector or mel/raw classifier).
    pub(crate) fn detect_audio(
        &self,
        model: &LoadedModel,
        audio: &AudioInput,
        opts: &AudioDetectOpts,
    ) -> Result<AudioDetectResult> {
        self.check_thread()?;
        let start = Instant::now();

        let config = AudioPreprocessConfig::from_manifest(&model.manifest.preprocess_method)
            .ok_or_else(|| {
                anyhow!(
                    "model '{}' is not an audio model (preprocess method '{}')",
                    model.id,
                    model.manifest.preprocess_method.as_str()
                )
            })?;
        config.validate().map_err(|e| anyhow!("{e}"))?;

        let (segment_duration_s, segment_stride_s) =
            window_params(&model.manifest, opts).ok_or_else(|| {
                anyhow!(
                    "model '{}' has no sliding-window inference strategy; audio detection requires one",
                    model.id
                )
            })?;

        let target_sr = config.sample_rate;
        let audio_samples = load_audio_at_sample_rate(audio, target_sr).map_err(|e| anyhow!("{e}"))?;
        let total = audio_samples.data.len();
        let duration_s = total as f32 / target_sr as f32;

        let segment_samples = (segment_duration_s * target_sr as f32).round() as usize;
        let stride_samples = ((segment_stride_s * target_sr as f32).round() as usize).max(1);
        if segment_samples == 0 {
            bail!("segment_duration_s resolves to zero samples for model '{}'", model.id);
        }
        let filterbank = MelFilterbank::new(&config).map_err(|e| anyhow!("{e}"))?;

        let is_detector = matches!(
            model.manifest.postprocess_method,
            PostprocessMethod::Sigmoid { .. }
        );
        let threshold = resolve_detector_threshold(&model.manifest, opts);

        let mut backend = model.backend.borrow_mut();
        let mut segments = Vec::new();
        // The mel's `orig_sample_rate` is the input's ORIGINAL rate (before the
        // whole-buffer resample to `target_sr`), matching the proven cascade —
        // it drives `fill_highfreq` (mel bins above the original Nyquist). For
        // already-target-rate input (the deployed path) it equals `target_sr`.
        let orig_sr = audio_samples.orig_sample_rate;
        for offset in compute_segment_offsets(total, segment_samples, stride_samples) {
            let logits = run_mel_segment(
                &mut backend,
                &audio_samples.data,
                offset,
                segment_samples,
                orig_sr,
                &config,
                &filterbank,
            )?;
            let (start_time_s, end_time_s) =
                segment_time_range(offset, segment_samples, total, target_sr);

            if is_detector {
                let logit = *logits.first().context("detector returned no logit")?;
                let confidence = sigmoid(logit);
                if confidence >= threshold {
                    segments.push(AudioSegment {
                        start_time_s,
                        end_time_s,
                        confidence,
                        classes: detector_classes(&model.labels, confidence),
                    });
                }
            } else {
                // Multi-class classifier: emit every window with top-K classes.
                let probs = softmax(&logits);
                let classes = top_k_classes(&probs, &model.labels, DEFAULT_AUDIO_CLASSIFIER_TOP_K);
                let confidence = classes.first().map(|c| c.probability).unwrap_or(0.0);
                segments.push(AudioSegment {
                    start_time_s,
                    end_time_s,
                    confidence,
                    classes,
                });
            }
        }

        Ok(AudioDetectResult {
            segments,
            duration_s,
            sample_rate: target_sr,
            processing_time_ms: start.elapsed().as_secs_f32() * 1000.0,
        })
    }
}

/// Public manifest-driven mobile engine. Cheap to clone (`Rc` to shared state).
///
/// `Rc` (not `Arc`) because the engine is thread-affine — it is never shared
/// across threads, so atomic refcounting would be wasted overhead.
#[derive(Clone)]
pub struct Engine {
    inner: Rc<EngineInner>,
}

impl Engine {
    /// Create an engine over a model catalog directory.
    ///
    /// `config.intra_threads` sets the LiteRT CPU inference thread count
    /// (`0` = LiteRT default). `config.device` / `inter_threads` are ignored: the
    /// mobile flavor runs the LiteRT CPU backend only (flavor-strict).
    pub fn new(config: EngineConfig) -> Result<Self> {
        let runtime = LiteRtRuntime::new().context("create LiteRT runtime")?;
        let inner = EngineInner {
            runtime,
            model_dir: config.model_dir,
            num_threads: config.intra_threads as usize,
            owner_thread: std::thread::current().id(),
            models: RefCell::new(HashMap::new()),
            pipelines: RefCell::new(HashMap::new()),
        };
        Ok(Self {
            inner: Rc::new(inner),
        })
    }

    /// Load a model by catalog id; returns a handle usable for inference.
    pub fn load_model_by_id(&self, id: &str) -> Result<MobileModel> {
        let loaded = self.inner.load_model(id)?;
        Ok(MobileModel {
            inner: Rc::downgrade(&self.inner),
            model_id: loaded.id.clone(),
        })
    }

    /// Unload a model by id.
    pub fn unload_model_by_id(&self, id: &str) -> Result<()> {
        self.inner.unload_model(id)
    }

    /// All models discoverable in the model directory.
    pub fn list_models(&self) -> Result<Vec<ModelInfo>> {
        self.inner.check_thread()?;
        Ok(self.inner.list_models())
    }

    /// Load a pipeline (audio cascade) by catalog id.
    pub fn load_pipeline_by_id(&self, id: &str) -> Result<()> {
        crate::pipeline::load_pipeline_by_id(&self.inner, id)
    }

    /// Run a loaded audio-cascade pipeline over an audio input (file or samples).
    pub fn run_pipeline(
        &self,
        pipeline_id: &str,
        input: &AudioInput,
        opts: &crate::pipeline::CascadeOpts,
    ) -> Result<crate::pipeline::CascadeResult> {
        crate::pipeline::run_pipeline(&self.inner, pipeline_id, input, opts)
    }

    /// Unload a pipeline by id (its stage models stay loaded; unload them
    /// separately if desired).
    pub fn unload_pipeline(&self, pipeline_id: &str) -> Result<()> {
        self.inner.check_thread()?;
        self.inner.pipelines().borrow_mut().remove(pipeline_id);
        Ok(())
    }
}

/// A handle to one loaded model. Holds a weak reference to the engine: a freed
/// or thread-foreign engine, or an unloaded model, surfaces as a clear error
/// instead of a dangling pointer.
pub struct MobileModel {
    inner: Weak<EngineInner>,
    model_id: String,
}

impl MobileModel {
    fn resolve(&self) -> Result<(Rc<EngineInner>, Rc<LoadedModel>)> {
        let inner = self
            .inner
            .upgrade()
            .ok_or_else(|| anyhow!("engine has been freed"))?;
        inner.check_thread()?;
        let loaded = inner
            .get_model(&self.model_id)
            .ok_or_else(|| anyhow!("model '{}' has been unloaded", self.model_id))?;
        Ok((inner, loaded))
    }

    /// The model id this handle refers to.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Run audio detection with this model.
    pub fn detect_audio(
        &self,
        audio: &AudioInput,
        opts: &AudioDetectOpts,
    ) -> Result<AudioDetectResult> {
        let (inner, loaded) = self.resolve()?;
        inner.detect_audio(&loaded, audio, opts)
    }

    /// Image detection — exposed for ABI stability, not yet available on mobile.
    pub fn detect(&self) -> Result<()> {
        Err(image_not_supported())
    }

    /// Image classification — exposed for ABI stability, not yet available on mobile.
    pub fn classify(&self) -> Result<()> {
        Err(image_not_supported())
    }

    /// Unload the model this handle refers to.
    pub fn unload(&self) -> Result<()> {
        let inner = self
            .inner
            .upgrade()
            .ok_or_else(|| anyhow!("engine has been freed"))?;
        inner.unload_model(&self.model_id)
    }
}

/// The RP-42 image-deferral message shared by `detect` / `classify` (Rust + FFI surfaces).
pub(crate) const IMAGE_UNSUPPORTED_MSG: &str =
    "image inference (detect/classify) is not yet available in the mobile (LiteRT) flavor: no \
     mobile (.tflite) image model is onboarded. It will be enabled by RP-42 (the ONNX→TFLite \
     conversion + onboarding task).";

/// The RP-42 image-deferral error shared by `detect` / `classify`.
pub(crate) fn image_not_supported() -> anyhow::Error {
    anyhow!(IMAGE_UNSUPPORTED_MSG)
}

/// Resolve sliding-window (duration, stride) from manifest, overridable by opts.
fn window_params(manifest: &ModelManifest, opts: &AudioDetectOpts) -> Option<(f32, f32)> {
    let (mut duration, mut stride) = match manifest.inference_strategy {
        InferenceStrategy::SlidingWindow {
            segment_duration_s,
            segment_stride_s,
        } => (segment_duration_s, segment_stride_s),
        _ => return None,
    };
    if let Some(d) = opts.segment_duration_s {
        duration = d;
    }
    if let Some(s) = opts.stride_s {
        stride = s;
    }
    Some((duration, stride))
}

/// Resolve a detector confidence threshold (manifest default, opts override).
/// Classifiers (softmax) emit every window, so this is used only by detectors.
fn resolve_detector_threshold(manifest: &ModelManifest, opts: &AudioDetectOpts) -> f32 {
    let default = match &manifest.postprocess_method {
        PostprocessMethod::Sigmoid {
            confidence_threshold,
        } => *confidence_threshold,
        _ => manifest.confidence_threshold.unwrap_or(0.5),
    };
    opts.confidence_threshold.unwrap_or(default)
}

/// Compute the dB-mel for one window and route it to a single-input model.
///
/// The window `[offset, offset+segment_samples)` is truncated/zero-padded to
/// exactly `segment_samples` so the fixed-shape mel matches the model's input.
/// Mirrors the proven `cascade::orca_mel_spectrogram` segment handling.
pub(crate) fn run_mel_segment(
    backend: &mut LiteRtBackend,
    samples: &[f32],
    offset: usize,
    segment_samples: usize,
    sample_rate: u32,
    config: &AudioPreprocessConfig,
    filterbank: &MelFilterbank,
) -> Result<Vec<f32>> {
    let bytes = mel_bytes_for_segment(samples, offset, segment_samples, sample_rate, config, filterbank)?;
    let outputs = backend.invoke_single(bytes, LiteRtElementType::kLiteRtElementTypeFloat32)?;
    outputs
        .into_iter()
        .next()
        .context("model returned no output tensor")
}

/// Compute the little-endian dB-mel input bytes for one window (no inference).
///
/// Shared by single-model [`EngineInner::detect_audio`] and the two-stage audio
/// cascade ([`crate::pipeline`]), which computes the mel **once** and feeds it to
/// both stages (the share-one-front-end optimization that matters on the Pi).
pub(crate) fn mel_bytes_for_segment(
    samples: &[f32],
    offset: usize,
    segment_samples: usize,
    sample_rate: u32,
    config: &AudioPreprocessConfig,
    filterbank: &MelFilterbank,
) -> Result<Vec<u8>> {
    let end = (offset + segment_samples).min(samples.len());
    let mut segment = samples[offset..end].to_vec();
    segment.resize(segment_samples, 0.0);
    let mel =
        mel_spectrogram(&segment, sample_rate, config, filterbank).map_err(|e| anyhow!("{e}"))?;
    nchw_mel_to_nhwc_le_bytes(&mel)
}

/// Build the `classes` vec for a binary detector window (1 entry when a labels
/// file is present, else empty — matches the cpu flavor's binary convention).
fn detector_classes(labels: &[String], confidence: f32) -> Vec<AudioClass> {
    if labels.is_empty() {
        Vec::new()
    } else {
        vec![AudioClass {
            class_idx: 0,
            label: labels.first().cloned(),
            probability: confidence,
        }]
    }
}

/// Top-K classes (descending probability) for a multi-class classifier window.
fn top_k_classes(probs: &[f32], labels: &[String], k: usize) -> Vec<AudioClass> {
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| probs[b].total_cmp(&probs[a]));
    idx.into_iter()
        .take(k)
        .map(|i| AudioClass {
            class_idx: i as u32,
            label: labels.get(i).cloned(),
            probability: probs[i],
        })
        .collect()
}
