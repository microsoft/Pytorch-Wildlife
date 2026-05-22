//! Multi-model pipeline orchestration: detect → crop → classify.
//!
//! Looks up the pipeline config, pins ALL model sessions at entry,
//! runs the detector, crops each detection from the original image, runs
//! classifier(s) on each crop, and assembles the final `PipelineResult`.

use std::collections::HashMap;
use std::time::Instant;

use crate::classify;
use crate::detect;
use crate::engine::{Engine, ModelHandle};
use crate::error::{SparrowEngineError, Result};
use crate::manifest::{PipelineManifest, PipelineRole};
use sparrow_engine_core::pipeline_compat::validate_pipeline_compat;

use crate::types::{
    ClassifyOpts, DetectOpts, ImageInput, ModelInfo, PipelineDetection, PipelineResult,
};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run a multi-model pipeline: detect → crop → classify.
///
/// 1. Looks up the pipeline config from the engine's registered pipelines.
/// 2. Pins ALL referenced model sessions at entry (see `pin_all_sessions` caveat).
/// 3. Runs the detector step.
/// 4. For each detection above threshold, crops the region from the original image.
/// 5. Runs classifier step(s) on each crop.
/// 6. Assembles and returns the combined result.
///
/// # Errors
/// - `PipelineNotFound` if the pipeline ID is not registered
/// - `PipelineMissingModels` if any referenced model is not loaded
/// - All errors from `detect()` and `classify()`
pub fn run_pipeline(
    engine: &Engine,
    pipeline_id: &str,
    image: &ImageInput,
    detect_opts: &DetectOpts,
    classify_opts: &ClassifyOpts,
) -> Result<PipelineResult> {
    let start = Instant::now();

    // 1. Look up pipeline config.
    let pipeline_config = engine.get_pipeline(pipeline_id)?;

    // 2. Pin ALL sessions atomically: collect handles for every referenced model.
    let pinned = pin_all_sessions(engine, pipeline_id, &pipeline_config)?;

    // 3. Find the detector step and run detection.
    let detector_model_id = pipeline_config
        .steps
        .iter()
        .find(|s| s.role == PipelineRole::Detector)
        .map(|s| &s.model)
        .ok_or_else(|| {
            SparrowEngineError::InvalidPipeline(format!(
                "Pipeline '{pipeline_id}' has no detector step (load_pipeline_manifest \
             should have rejected this)",
            ))
        })?;

    let detector_handle = &pinned[detector_model_id.as_str()];
    let detect_result = detect::detect(detector_handle, image, detect_opts)?;

    // 4. Decode the original image for cropping.
    // We need the original image pixels to crop detection regions.
    let decoded = detect::decode_image(image)?;
    let (orig_w, orig_h) = (decoded.width(), decoded.height());

    // Collect classifier model IDs from pipeline steps.
    let classifier_model_ids: Vec<&str> = pipeline_config
        .steps
        .iter()
        .filter(|s| s.role == PipelineRole::Classifier)
        .map(|s| s.model.as_str())
        .collect();

    // 5. For each detection, crop and classify.
    let mut pipeline_detections = Vec::with_capacity(detect_result.detections.len());

    for detection in &detect_result.detections {
        // Crop the detection region from the original image.
        // Returns None for degenerate crops (< MIN_CROP_SIZE in either axis).
        let crop = crop_detection(&decoded, &detection.bbox, orig_w, orig_h);

        // Run each classifier on the crop. For MVP, take the first classifier's result.
        // Skip classification entirely for degenerate crops.
        let classification = match crop {
            Some(crop_image) if !classifier_model_ids.is_empty() => {
                let classifier_handle = &pinned[classifier_model_ids[0]];
                match classify::classify(classifier_handle, &crop_image, classify_opts) {
                    Ok(cls_result) => cls_result.classifications.into_iter().next(),
                    Err(_) => None, // Classification failure on a crop is non-fatal.
                }
            }
            _ => None, // No classifier, or degenerate crop — skip classification.
        };

        pipeline_detections.push(PipelineDetection {
            detection: detection.clone(),
            classification,
        });
    }

    let elapsed = start.elapsed();

    // 6. Assemble result.
    Ok(PipelineResult {
        pipeline_id: pipeline_id.to_string(),
        detections: pipeline_detections,
        image_width: detect_result.image_width,
        image_height: detect_result.image_height,
        processing_time_ms: elapsed.as_secs_f32() * 1000.0,
    })
}

/// Run an ad-hoc pipeline: detect → crop → classify without pre-defined TOML config.
///
/// Auto-loads models by ID if not already loaded (via `Engine::get_or_load_model`).
/// Reuses the same crop + classify logic as `run_pipeline()`.
///
/// # Errors
/// - All errors from `Engine::get_or_load_model` (manifest not found, ORT session, etc.)
/// - All errors from `detect()` and `classify()`
pub fn run_pipeline_adhoc(
    engine: &Engine,
    image: &ImageInput,
    detector_id: &str,
    classifier_id: &str,
    detect_opts: &DetectOpts,
    classify_opts: &ClassifyOpts,
) -> Result<PipelineResult> {
    let start = Instant::now();

    // 1. Validate compatibility before any lazy ORT session creation.
    validate_adhoc_model_types(&engine.list_available_models(), detector_id, classifier_id)?;

    // 2. Auto-load models if needed.
    let detector_handle = engine.get_or_load_model(detector_id)?;
    let classifier_handle = engine.get_or_load_model(classifier_id)?;

    // 3. Run detection.
    let detect_result = detect::detect(&detector_handle, image, detect_opts)?;

    // 3. Decode the original image for cropping.
    let decoded = detect::decode_image(image)?;
    let (orig_w, orig_h) = (decoded.width(), decoded.height());

    // 4. For each detection, crop and classify.
    let mut pipeline_detections = Vec::with_capacity(detect_result.detections.len());

    for detection in &detect_result.detections {
        let crop = crop_detection(&decoded, &detection.bbox, orig_w, orig_h);

        let classification = match crop {
            Some(crop_image) => {
                match classify::classify(&classifier_handle, &crop_image, classify_opts) {
                    Ok(cls_result) => cls_result.classifications.into_iter().next(),
                    Err(_) => None,
                }
            }
            None => None,
        };

        pipeline_detections.push(PipelineDetection {
            detection: detection.clone(),
            classification,
        });
    }

    let elapsed = start.elapsed();

    // 5. Assemble result with synthetic pipeline ID.
    Ok(PipelineResult {
        pipeline_id: format!("adhoc:{detector_id}+{classifier_id}"),
        detections: pipeline_detections,
        image_width: detect_result.image_width,
        image_height: detect_result.image_height,
        processing_time_ms: elapsed.as_secs_f32() * 1000.0,
    })
}

fn validate_adhoc_model_types(
    available: &[ModelInfo],
    detector_id: &str,
    classifier_id: &str,
) -> Result<()> {
    let detector_type = available
        .iter()
        .find(|m| m.id == detector_id)
        .map(|m| m.model_type);
    let classifier_type = available
        .iter()
        .find(|m| m.id == classifier_id)
        .map(|m| m.model_type);

    match (detector_type, classifier_type) {
        (Some(detector), Some(classifier)) => {
            validate_pipeline_compat(Some(detector), Some(classifier))
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Session pinning
// ---------------------------------------------------------------------------

/// Pin all model sessions referenced by the pipeline.
///
/// Uses `Engine::get_model_handles()` for atomic batch lookup: a single
/// read lock across all models ensures a consistent snapshot (no model
/// can be replaced or removed between individual lookups).
///
/// If any referenced model is missing or unloaded, returns an error listing
/// ALL missing models (not just the first).
fn pin_all_sessions(
    engine: &Engine,
    pipeline_id: &str,
    config: &PipelineManifest,
) -> Result<HashMap<String, ModelHandle>> {
    let model_ids: Vec<&str> = config.steps.iter().map(|s| s.model.as_str()).collect();

    // Atomic batch lookup: single read lock across all models.
    let (found, missing_ids) = engine.get_model_handles(&model_ids);

    if !missing_ids.is_empty() {
        return Err(SparrowEngineError::PipelineMissingModels {
            id: pipeline_id.to_string(),
            missing: missing_ids.join(", "),
        });
    }

    // Build HashMap keyed by model ID for step lookups.
    let pinned: HashMap<String, ModelHandle> = found
        .into_iter()
        .map(|h| (h.model_id().to_string(), h))
        .collect();

    Ok(pinned)
}

// ---------------------------------------------------------------------------
// Cropping
// ---------------------------------------------------------------------------

/// Minimum crop dimension (pixels). Crops smaller than this in either axis
/// are considered degenerate and skipped for classification.
const MIN_CROP_SIZE: u32 = 2;

/// Crop a detection region from the original image.
///
/// Given a detection bbox (normalized [0,1]) and the decoded image,
/// returns `Some(ImageInput::Raw { .. })` with the crop pixels, or `None`
/// if the crop is degenerate (smaller than [`MIN_CROP_SIZE`] in either axis).
///
/// Degenerate crops (near-zero area) produce garbage classifications, so
/// callers should set classification to `None` when this returns `None`.
fn crop_detection(
    img: &image::DynamicImage,
    bbox: &crate::types::BBox,
    img_w: u32,
    img_h: u32,
) -> Option<ImageInput> {
    // Convert normalized coords to pixel coords.
    let x1 = (bbox.x_min * img_w as f32).round() as u32;
    let y1 = (bbox.y_min * img_h as f32).round() as u32;
    let x2 = (bbox.x_max * img_w as f32).round() as u32;
    let y2 = (bbox.y_max * img_h as f32).round() as u32;

    // Clamp to image bounds.
    let x1 = x1.min(img_w);
    let y1 = y1.min(img_h);
    let x2 = x2.min(img_w).max(x1);
    let y2 = y2.min(img_h).max(y1);

    let crop_w = x2 - x1;
    let crop_h = y2 - y1;

    // Skip degenerate crops — they produce garbage classifications.
    if crop_w < MIN_CROP_SIZE || crop_h < MIN_CROP_SIZE {
        return None;
    }

    // Crop the region.
    let cropped = img.crop_imm(x1, y1, crop_w, crop_h);
    let rgb = cropped.to_rgb8();

    // Return as raw pixel buffer (avoids JPEG encode/decode round-trip).
    let width = rgb.width();
    let height = rgb.height();
    Some(ImageInput::Raw {
        stride: width * 3,
        width,
        height,
        data: rgb.into_raw(),
        format: crate::types::PixelFormat::Rgb,
    })
}

// Image decoding reuses `detect::decode_image` — see M1 fix (no duplication).

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ModelType;
    use std::path::PathBuf;

    fn info(id: &str, model_type: ModelType) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            path: PathBuf::from(format!("/models/{id}/manifest.toml")),
            model_type,
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        }
    }

    #[test]
    fn adhoc_compat_rejects_known_incompatible_pair_before_load() {
        let available = vec![
            info("owl-t", ModelType::OverheadDetector),
            info("speciesnet-crop", ModelType::Classifier),
        ];
        let err = validate_adhoc_model_types(&available, "owl-t", "speciesnet-crop").unwrap_err();
        match err {
            SparrowEngineError::IncompatiblePipeline { reason, .. } => {
                assert!(
                    reason.contains("point detection"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected IncompatiblePipeline, got {other:?}"),
        }
    }

    #[test]
    fn adhoc_compat_defers_unknown_ids_to_load_path() {
        let available = vec![info("speciesnet-crop", ModelType::Classifier)];
        validate_adhoc_model_types(&available, "missing", "speciesnet-crop").unwrap();
    }
}

// ---------------------------------------------------------------------------
// Integration tests needed (require ORT session creation)
// ---------------------------------------------------------------------------
// Integration test needed: pipeline skips classification for degenerate crops
