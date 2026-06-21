//! GPU detection inference: single-shot and tiled paths.
//!
//! Mirrors `sparrow_engine_cpu::detect`'s surface so consumers can swap between
//! flavors via compile-time feature dispatch (Phase C). The top-level
//! [`detect`] / [`detect_batch`] free fns take a [`ModelHandle`] and
//! route to the right per-model GPU pipeline:
//!
//! - YOLO E2E (single-shot, [`InferenceStrategy::Single`]) →
//!   [`crate::models::yolo::YoloModel::detect`].
//! - Tiled detection ([`InferenceStrategy::Tiled`], also covers
//!   `OverheadDetector` subtypes like HerdNet and OWL-T) →
//!   [`crate::models::tiled::TiledModel::detect_tiled`].
//! - Classifiers + audio models are rejected up-front via
//!   [`validate_vision_detector`] (mirrors sparrow-engine-cpu).
//!
//! Engine-level CUDA primitives (letterbox kernel) are reached via
//! `handle.engine_ref.upgrade()`. The kernel is engine-shared (one per
//! process) so we do not pay re-compile cost per call.

use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{ModelManifest, PostprocessMethod, PreprocessMethod};
use sparrow_engine_types::types::{DetectOpts, DetectResult, ImageInput};

use crate::engine::{LoadedModelInner, ModelHandle};

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that a manifest represents a vision detection model (not a
/// classifier, not audio). Mirrors `sparrow_engine_cpu::detect::validate_vision_detector`.
pub(crate) fn validate_vision_detector(manifest: &ModelManifest) -> Result<()> {
    if matches!(manifest.postprocess_method, PostprocessMethod::Softmax) {
        return Err(SparrowEngineError::NotADetector {
            id: manifest.id.clone(),
            method: manifest.postprocess_method.as_str().to_string(),
        });
    }
    if matches!(
        manifest.preprocess_method,
        PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. }
    ) {
        return Err(SparrowEngineError::IsAudioModel {
            id: manifest.id.clone(),
            method: manifest.preprocess_method.as_str().to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run detection inference on a single image.
///
/// Validates model type, dispatches to the right per-model GPU pipeline.
///
/// # Errors
/// - [`SparrowEngineError::NotADetector`] if the model is a classifier.
/// - [`SparrowEngineError::IsAudioModel`] if the model is audio.
/// - [`SparrowEngineError::ModelUnloaded`] / [`SparrowEngineError::EngineFreed`] if the
///   handle is invalid.
/// - [`SparrowEngineError::Ort`] on GPU pipeline / ORT runtime errors.
pub fn detect(handle: &ModelHandle, image: &ImageInput, opts: &DetectOpts) -> Result<DetectResult> {
    let inner = handle.pin_inner()?;
    validate_vision_detector(&inner.manifest)?;
    let _ = sparrow_engine_core::postprocess::resolve_confidence_threshold(
        opts.confidence_threshold,
        inner.manifest.confidence_threshold.unwrap_or(0.0),
    )?;

    let engine_inner = handle
        .engine_ref
        .upgrade()
        .ok_or(SparrowEngineError::EngineFreed)?;

    match &inner.inner {
        LoadedModelInner::Yolo(model) => {
            model.detect(&engine_inner.ctx, &engine_inner.letterbox, image, opts)
        }
        LoadedModelInner::Tiled(model) => model.detect_tiled(&engine_inner.ctx, image, opts),
        LoadedModelInner::Classifier(_) => Err(SparrowEngineError::NotADetector {
            id: inner.manifest.id.clone(),
            method: inner.manifest.postprocess_method.as_str().to_string(),
        }),
        LoadedModelInner::Audio(_) | LoadedModelInner::AudioRaw(_) => {
            Err(SparrowEngineError::IsAudioModel {
                id: inner.manifest.id.clone(),
                method: inner.manifest.preprocess_method.as_str().to_string(),
            })
        }
    }
}

/// Run detection on multiple images, invoking `on_result` after each
/// image's detections are ready.
///
/// `batch_size` is accepted for parity with `sparrow_engine_cpu::detect::detect_batch`
/// but the GPU path runs per-image dispatch in Wave 1. The batched ORT
/// path (`YoloModel::detect_batch_pipelined`) stays `pub(crate)` for now;
/// surfacing it is a Phase C bench-driven follow-up. The CPU's batch
/// fallback for tiled models is unnecessary here because the GPU dispatch
/// is already per-image.
///
/// Phase 3.8 Phase C W1 audit-fix R2 (I-S5): the SIGNATURE preserves the
/// CPU surface but the SEMANTICS diverge — sparrow-engine-cpu uses `batch_size`
/// to drive batched ORT inference (`ndarray::concatenate`), sparrow-engine-gpu
/// silently falls back to per-image dispatch. A `tracing::debug!` is
/// emitted when the caller passes a non-trivial `batch_size` so the
/// runtime divergence surfaces under `RUST_LOG=sparrow_engine_gpu=debug`.
#[allow(clippy::type_complexity)]
pub fn detect_batch(
    handle: &ModelHandle,
    images: &[ImageInput],
    opts: &DetectOpts,
    batch_size: usize,
    mut on_result: Option<&mut dyn FnMut(usize, &DetectResult)>,
) -> Result<Vec<DetectResult>> {
    // I-S5: surface the cpu/gpu semantic divergence at runtime.
    if batch_size != 0 && batch_size != 1 {
        tracing::debug!(
            batch_size,
            "sparrow-engine-gpu detect_batch ignores batch_size; using per-image dispatch"
        );
    }

    // Validate up-front so we fail fast on a wrong-model-type batch.
    let inner = handle.pin_inner()?;
    validate_vision_detector(&inner.manifest)?;
    let _ = sparrow_engine_core::postprocess::resolve_confidence_threshold(
        opts.confidence_threshold,
        inner.manifest.confidence_threshold.unwrap_or(0.0),
    )?;
    drop(inner);

    let mut results = Vec::with_capacity(images.len());
    for (i, image) in images.iter().enumerate() {
        let r = detect(handle, image, opts)?;
        if let Some(ref mut cb) = on_result {
            cb(i, &r);
        }
        results.push(r);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Tests (compile-only — exercise the routing on bad model types).
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{
        InferenceStrategy, Layout, ModelManifest, Normalization, Precision,
    };
    use sparrow_engine_types::types::ModelSubtype;

    fn fake_audio_manifest() -> ModelManifest {
        ModelManifest {
            id: "fake_audio".into(),
            format: "onnx".into(),
            model_file: "model.onnx".into(),
            model_file_fp16: None,
            preprocess_method: PreprocessMethod::MelSpectrogram {
                sample_rate: 48000,
                n_fft: 1024,
                hop_length: 512,
                n_mels: 64,
                fmin: 0.0,
                fmax: 24000.0,
                top_db: 80.0,
                window: "hann_symmetric".into(),
                mel_scale: "slaney".into(),
                filter_norm: "slaney".into(),
                fill_highfreq: false,
            },
            input_size: None,
            layout: Some(Layout::Nchw),
            normalization: Some(Normalization::Unit),
            pad_value: None,
            channel_order: None,
            precision: Precision::Fp32,
            inference_strategy: InferenceStrategy::SlidingWindow {
                segment_duration_s: 3.0,
                segment_stride_s: 1.5,
            },
            trt: None,
            postprocess_method: PostprocessMethod::Sigmoid {
                confidence_threshold: 0.5,
            },
            confidence_threshold: Some(0.5),
            label_file: None,
            label_format: None,
            default: false,
            subtype: ModelSubtype::Standard,
            onnx_sha256: None,
            onnx_size_bytes: None,
            version: None,
            description: None,
            provenance: None,
            drift_reference: None,
        }
    }

    #[test]
    fn validate_vision_detector_rejects_audio_manifest() {
        let m = fake_audio_manifest();
        let err = validate_vision_detector(&m).unwrap_err();
        assert!(matches!(err, SparrowEngineError::IsAudioModel { .. }));
    }

    #[test]
    fn validate_vision_detector_no_longer_rejects_sliding_window() {
        // F7 — engine.rs:364-368 already catches contradictory SlidingWindow
        // + non-audio manifests at load time with InvalidManifest. The check
        // previously at detect.rs:50-55 was unreachable; removing aligns gpu
        // with sparrow-engine-cpu's validate_vision_detector shape.
        //
        // Build a manifest that has SlidingWindow strategy but ALSO has
        // letterbox preprocess + yolo_e2e postprocess (i.e., a vision
        // detector shape). validate_vision_detector should now return Ok(())
        // because the SlidingWindow check is gone and no other check fires.
        let m = ModelManifest {
            id: "fake_letterbox_sliding".into(),
            format: "onnx".into(),
            model_file: "model.onnx".into(),
            model_file_fp16: None,
            preprocess_method: PreprocessMethod::Letterbox,
            input_size: Some([640, 640]),
            layout: Some(Layout::Nchw),
            normalization: Some(Normalization::Unit),
            pad_value: Some(114.0),
            channel_order: None,
            precision: Precision::Fp32,
            inference_strategy: InferenceStrategy::SlidingWindow {
                segment_duration_s: 3.0,
                segment_stride_s: 1.5,
            },
            trt: None,
            postprocess_method: PostprocessMethod::YoloE2e,
            confidence_threshold: Some(0.3),
            label_file: None,
            label_format: None,
            default: false,
            subtype: ModelSubtype::Standard,
            onnx_sha256: None,
            onnx_size_bytes: None,
            version: None,
            description: None,
            provenance: None,
            drift_reference: None,
        };
        assert!(validate_vision_detector(&m).is_ok());
    }
}
