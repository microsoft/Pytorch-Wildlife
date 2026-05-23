//! Classification inference.
//!
//! Orchestrates: preprocess -> ORT session.run -> softmax postprocess.

use std::time::Instant;

use ndarray::{ArrayView2, ArrayViewD};
use ort::value::TensorRef;

use crate::detect::preprocess_config_from_manifest;
use crate::engine::ModelHandle;
use crate::error::{SparrowEngineError, Result};
use crate::manifest::{ModelManifest, PostprocessMethod, PreprocessMethod};
use crate::postprocess;
use crate::preprocess;
use crate::types::{ClassifyOpts, ClassifyResult, ImageInput};

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that a manifest represents a vision classification model (not a detector, not audio).
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
/// Validates model type, preprocesses, runs ORT, applies softmax, and returns
/// top-k classifications.
///
/// # Errors
/// - `NotAClassifier` if the model's postprocessing method is not `softmax`
/// - `ModelUnloaded` if the handle has been invalidated — also surfaces if the
///   engine itself has been dropped (post-S1 MT-17 mitigation: `Drop for Engine`
///   in `engine.rs` leaks `Arc<EngineInner>` so `Weak::upgrade()` keeps
///   succeeding; the signal the handle actually sees is the per-model `active`
///   flag that `Drop` clears before releasing sessions — see `docs/bugs.md`
///   MT-17 for the full rationale).
/// - `EngineFreed` reserved for pre-Drop paths (e.g. `Engine::unload_model`).
/// - `Ort` on ORT runtime errors
pub fn classify(
    handle: &ModelHandle,
    image: &ImageInput,
    opts: &ClassifyOpts,
) -> Result<ClassifyResult> {
    let start = Instant::now();

    // 1. Validate model type: reject audio and non-classifier models.
    let manifest = &handle.manifest;
    validate_vision_classifier(manifest)?;

    // 2. Pin session (checks handle validity: active flag + engine weak ref).
    let session = handle.pin_session()?;
    let labels = &*handle.labels;

    // 3. Preprocess.
    let config = preprocess_config_from_manifest(manifest)?;
    let prep = preprocess::preprocess(image, &config)?;
    let original_width = prep.meta.original_width;
    let original_height = prep.meta.original_height;

    // 4. Run ORT.
    let input_value = TensorRef::from_array_view(&prep.tensor).map_err(crate::engine::ort_err)?;

    // Lock session for exclusive ORT access. The guard must outlive `outputs`
    // because `SessionOutputs` borrows from the session.
    let mut guard = session
        .lock()
        .map_err(|_| SparrowEngineError::Ort("classifier session lock poisoned".into()))?;
    let outputs = guard
        .run(ort::inputs![input_value])
        .map_err(crate::engine::ort_err)?;

    if outputs.len() == 0 {
        return Err(SparrowEngineError::Ort(
            "classifier session returned no outputs".to_string(),
        ));
    }

    // 5. Postprocess: extract logits and apply softmax.
    let output_view: ArrayViewD<'_, f32> = outputs[0]
        .try_extract_array::<f32>()
        .map_err(crate::engine::ort_err)?;

    let ndim = output_view.ndim();

    // Logits expected as [1, num_classes] or [batch, num_classes].
    let view_2d: ArrayView2<f32> = if ndim == 2 {
        output_view
            .into_dimensionality::<ndarray::Ix2>()
            .map_err(crate::engine::ort_err)?
    } else if ndim == 1 {
        // [num_classes] -> reshape to [1, num_classes]
        let len = output_view.len();
        output_view
            .into_shape_with_order((1, len))
            .map_err(crate::engine::ort_err)?
    } else {
        // Engine validation (`validate_output_shape`) rejects softmax models
        // with rank > 2 at load time. If we reach here, either validation was
        // bypassed or a new output rank was introduced without updating this
        // function.
        return Err(SparrowEngineError::Ort(format!(
            "Unexpected classifier output rank {ndim}; expected 1 or 2. \
             Softmax models must produce rank-2 output (engine rejects rank > 2 \
             at load time).",
        )));
    };

    let classifications = postprocess::try_softmax(&view_2d, labels, opts)?;
    drop(outputs);
    drop(guard);

    let elapsed = start.elapsed();

    // 6. Return result.
    Ok(ClassifyResult {
        classifications,
        image_width: original_width,
        image_height: original_height,
        processing_time_ms: elapsed.as_secs_f32() * 1000.0,
    })
}
