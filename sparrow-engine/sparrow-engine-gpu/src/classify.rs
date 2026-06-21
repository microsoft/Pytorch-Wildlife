//! GPU classification inference.
//!
//! Mirrors `sparrow_engine_cpu::classify`'s surface. The top-level [`classify`]
//! free fn takes a [`ModelHandle`] and routes to
//! [`crate::models::classifier::ClassifierModel::classify`], borrowing
//! the engine-shared [`crate::kernels::resize::ResizeKernel`],
//! [`crate::kernels::center_crop::CenterCropKernel`] and a cached
//! [`crate::models::classifier::JpegDecoder`] from `EngineInner`.

use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{ModelManifest, PostprocessMethod, PreprocessMethod};
use sparrow_engine_types::types::{ClassifyOpts, ClassifyResult, ImageInput};

use crate::engine::{LoadedModelInner, ModelHandle};

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that a manifest represents a vision classification model
/// (not a detector, not audio). Mirrors
/// `sparrow_engine_cpu::classify::validate_vision_classifier`.
pub(crate) fn validate_vision_classifier(manifest: &ModelManifest) -> Result<()> {
    if matches!(
        manifest.preprocess_method,
        PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. }
    ) {
        return Err(SparrowEngineError::IsAudioModel {
            id: manifest.id.clone(),
            method: manifest.preprocess_method.as_str().to_string(),
        });
    }
    if !matches!(manifest.postprocess_method, PostprocessMethod::Softmax) {
        return Err(SparrowEngineError::NotAClassifier {
            id: manifest.id.clone(),
            method: manifest.postprocess_method.as_str().to_string(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run classification inference on a single image.
///
/// # Errors
/// - [`SparrowEngineError::NotAClassifier`] if the model is not a softmax
///   classifier.
/// - [`SparrowEngineError::IsAudioModel`] if the model is audio.
/// - [`SparrowEngineError::ModelUnloaded`] / [`SparrowEngineError::EngineFreed`] if the
///   handle is invalid.
/// - [`SparrowEngineError::Ort`] on GPU pipeline / ORT runtime errors.
pub fn classify(
    handle: &ModelHandle,
    image: &ImageInput,
    opts: &ClassifyOpts,
) -> Result<ClassifyResult> {
    let inner = handle.pin_inner()?;
    validate_vision_classifier(&inner.manifest)?;

    let engine_inner = handle
        .engine_ref
        .upgrade()
        .ok_or(SparrowEngineError::EngineFreed)?;

    match &inner.inner {
        LoadedModelInner::Classifier(model) => {
            let mut decoder_guard = engine_inner
                .decoder
                .lock()
                .map_err(|_| SparrowEngineError::Ort("engine JpegDecoder lock poisoned".into()))?;
            model.classify(
                &engine_inner.ctx,
                &engine_inner.center_crop,
                &engine_inner.resize,
                &mut decoder_guard,
                image,
                opts,
            )
        }
        LoadedModelInner::Yolo(_) | LoadedModelInner::Tiled(_) => {
            Err(SparrowEngineError::NotAClassifier {
                id: inner.manifest.id.clone(),
                method: inner.manifest.postprocess_method.as_str().to_string(),
            })
        }
        LoadedModelInner::Audio(_) | LoadedModelInner::AudioRaw(_) => {
            Err(SparrowEngineError::IsAudioModel {
                id: inner.manifest.id.clone(),
                method: inner.manifest.preprocess_method.as_str().to_string(),
            })
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{
        InferenceStrategy, Layout, ModelManifest, Normalization, Precision,
    };
    use sparrow_engine_types::types::ModelSubtype;

    fn yolo_like_manifest() -> ModelManifest {
        ModelManifest {
            id: "fake_yolo".into(),
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
            inference_strategy: InferenceStrategy::Single,
            trt: None,
            postprocess_method: PostprocessMethod::YoloE2e,
            confidence_threshold: None,
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
    fn validate_vision_classifier_rejects_yolo_postprocess() {
        let m = yolo_like_manifest();
        let err = validate_vision_classifier(&m).unwrap_err();
        assert!(matches!(err, SparrowEngineError::NotAClassifier { .. }));
    }
}
