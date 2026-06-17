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
//! ## Image inference (RP-42)
//!
//! [`MobileModel::detect`] runs single-shot image detection for `yolo_e2e`
//! detectors (the MegaDetector family) — decode → letterbox (NHWC) → LiteRT
//! invoke → shared [`sparrow_engine_core::postprocess::yolo_e2e`]. Landed in
//! RP-42 with the first ONNX→TFLite-converted image model.
//! [`MobileModel::classify`] is still exposed for ABI stability but returns a
//! clear error until a mobile (`.tflite`) classifier model is onboarded.

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
    self, InferenceStrategy, ModelManifest, Normalization, PostprocessMethod,
};
use sparrow_engine_types::derive_model_type;
use sparrow_engine_types::types::{
    AudioClass, AudioDetectOpts, AudioDetectResult, AudioInput, AudioSegment, DetectOpts,
    DetectResult, ImageInput, ModelInfo, ModelType,
};
pub use sparrow_engine_types::EngineConfig;

use ndarray::ArrayView2;
use sparrow_engine_core::postprocess;
use sparrow_engine_core::preprocess::decode_to_rgb;

use crate::cascade::{nchw_mel_to_nhwc_le_bytes, sigmoid, softmax};
use crate::preprocess::{f32_slice_to_le_bytes, letterbox_nhwc};
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

    /// Run single-shot image detection with a `yolo_e2e` detector model.
    ///
    /// decode → letterbox (NHWC f32) → LiteRT invoke → shared
    /// [`postprocess::try_yolo_e2e`] (which undoes the letterbox via the
    /// [`sparrow_engine_types::PreprocessMeta`] returned by [`letterbox_nhwc`]).
    pub(crate) fn detect(
        &self,
        model: &LoadedModel,
        image: &ImageInput,
        opts: &DetectOpts,
    ) -> Result<DetectResult> {
        self.check_thread()?;
        let start = Instant::now();

        // Flavor scope: the mobile image path implements `yolo_e2e` single-shot
        // detection (MegaDetector family). Other postprocess methods
        // (`megadet_v5a`, `heatmap_peaks`) and classification are not yet ported.
        if !matches!(model.manifest.postprocess_method, PostprocessMethod::YoloE2e) {
            bail!(
                "model '{}' uses postprocess '{:?}', which the mobile flavor does not yet support \
                 for image detection (only yolo_e2e is implemented)",
                model.id,
                model.manifest.postprocess_method
            );
        }

        // Preprocess: decode → letterbox → NHWC f32. A yolo_e2e model is always an
        // image model, so the image-only manifest fields must be present.
        let input_size = model.manifest.input_size.ok_or_else(|| {
            anyhow!("model '{}' has no input_size; not an image model", model.id)
        })?;
        let normalization = model.manifest.normalization.unwrap_or(Normalization::Unit);
        let channel_order = model.manifest.channel_order.unwrap_or_default();
        let pad_value = model.manifest.pad_value.unwrap_or(0.0);

        let rgb = decode_to_rgb(image).map_err(|e| anyhow!("{e}"))?;
        let (nhwc, meta) = letterbox_nhwc(
            &rgb,
            input_size[0],
            input_size[1],
            pad_value,
            normalization,
            channel_order,
        );

        // Inference: single-input NHWC f32 little-endian bytes.
        let bytes = f32_slice_to_le_bytes(&nhwc);
        let outputs = model
            .backend
            .borrow_mut()
            .invoke_single(bytes, LiteRtElementType::kLiteRtElementTypeFloat32)
            .map_err(|e| anyhow!("{e}"))?;
        let flat = outputs
            .first()
            .ok_or_else(|| anyhow!("model '{}' returned no output tensor", model.id))?;

        // yolo_e2e output is a flat row-major `[N, 6]` block (N = top-K candidates).
        if flat.is_empty() || flat.len() % 6 != 0 {
            bail!(
                "model '{}' yolo_e2e output length {} is not a positive multiple of 6",
                model.id,
                flat.len()
            );
        }
        let rows = flat.len() / 6;
        let view = ArrayView2::from_shape((rows, 6), flat)
            .map_err(|e| anyhow!("reshape yolo_e2e output to [{rows}, 6]: {e}"))?;

        let detections = postprocess::try_yolo_e2e(
            &view,
            &model.labels,
            opts,
            &meta,
            model.manifest.confidence_threshold.unwrap_or(0.2),
        )
        .map_err(|e| anyhow!("{e}"))?;

        Ok(DetectResult {
            detections,
            image_width: meta.original_width,
            image_height: meta.original_height,
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

    /// Run single-shot image detection with this model (`yolo_e2e` detectors).
    pub fn detect(&self, image: &ImageInput, opts: &DetectOpts) -> Result<DetectResult> {
        let (inner, loaded) = self.resolve()?;
        inner.detect(&loaded, image, opts)
    }

    /// Image classification — exposed for ABI stability, not yet available on mobile.
    pub fn classify(&self) -> Result<()> {
        Err(classify_not_supported())
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

/// Classification-deferral message (Rust + FFI surfaces). Image *detection*
/// (`detect`) is implemented as of RP-42; only `classify` remains deferred until
/// a mobile (`.tflite`) classifier model is onboarded.
pub(crate) const CLASSIFY_UNSUPPORTED_MSG: &str =
    "image classification is not yet available in the mobile (LiteRT) flavor: no mobile \
     (.tflite) classifier model is onboarded. Image detection (detect) is available as of RP-42.";

/// The classification-deferral error used by `MobileModel::classify` and the FFI.
pub(crate) fn classify_not_supported() -> anyhow::Error {
    anyhow!(CLASSIFY_UNSUPPORTED_MSG)
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
