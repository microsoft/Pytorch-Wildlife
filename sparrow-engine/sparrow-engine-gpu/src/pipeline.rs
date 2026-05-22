//! Multi-model GPU pipeline orchestration: detect → crop → classify.
//!
//! Mirrors `sparrow_engine_cpu::pipeline`'s surface and shape. The detect /
//! classify steps route through `crate::detect::detect` and
//! `crate::classify::classify`, which dispatch to the per-model GPU
//! pipelines. Image cropping happens on CPU (the original image is
//! decoded once via the `image` crate, then cropped to a raw RGB
//! buffer fed back into `classify`); this matches sparrow-engine-cpu's behaviour
//! and avoids re-uploading every crop to the GPU.

use std::collections::HashMap;
use std::time::Instant;

use sparrow_engine_core::pipeline_compat::validate_pipeline_compat;
use sparrow_engine_types::error::{SparrowEngineError, Result};
use sparrow_engine_types::manifest::{PipelineManifest, PipelineRole};
use sparrow_engine_types::types::{
    BBox, ClassifyOpts, DetectOpts, ImageInput, ModelInfo, PipelineDetection, PipelineResult,
    PixelFormat,
};

use crate::classify;
use crate::detect;
use crate::engine::{Engine, ModelHandle};

/// Minimum crop dimension (pixels). Crops smaller than this in either
/// axis are considered degenerate and skipped for classification.
const MIN_CROP_SIZE: u32 = 2;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run a multi-model pipeline: detect → crop → classify.
///
/// 1. Looks up the pipeline config from the engine's registered pipelines.
/// 2. Pins ALL referenced model sessions at entry (atomic snapshot).
/// 3. Runs the detector step.
/// 4. For each detection above threshold, crops the region from the
///    original image.
/// 5. Runs classifier step(s) on each crop.
/// 6. Assembles and returns the combined result.
///
/// # Errors
/// - [`SparrowEngineError::PipelineNotFound`] if the pipeline ID is not registered.
/// - [`SparrowEngineError::PipelineMissingModels`] if any referenced model is not
///   loaded.
/// - All errors from [`detect::detect`] and [`classify::classify`].
pub fn run_pipeline(
    engine: &Engine,
    pipeline_id: &str,
    image: &ImageInput,
    detect_opts: &DetectOpts,
    classify_opts: &ClassifyOpts,
) -> Result<PipelineResult> {
    let start = Instant::now();

    let pipeline_config = engine.get_pipeline(pipeline_id)?;
    let pinned = pin_all_sessions(engine, pipeline_id, &pipeline_config)?;

    // Resolve detector step.
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

    let decoded = decode_image(image)?;
    let (orig_w, orig_h) = (decoded.width(), decoded.height());

    // Collect classifier model IDs from pipeline steps.
    let classifier_model_ids: Vec<&str> = pipeline_config
        .steps
        .iter()
        .filter(|s| s.role == PipelineRole::Classifier)
        .map(|s| s.model.as_str())
        .collect();

    let mut pipeline_detections = Vec::with_capacity(detect_result.detections.len());
    for detection in &detect_result.detections {
        let crop = crop_detection(&decoded, &detection.bbox, orig_w, orig_h);
        let classification = match crop {
            Some(crop_image) if !classifier_model_ids.is_empty() => {
                let classifier_handle = &pinned[classifier_model_ids[0]];
                match classify::classify(classifier_handle, &crop_image, classify_opts) {
                    Ok(cls_result) => cls_result.classifications.into_iter().next(),
                    Err(_) => None, // classification failure on a crop is non-fatal.
                }
            }
            _ => None,
        };
        pipeline_detections.push(PipelineDetection {
            detection: detection.clone(),
            classification,
        });
    }

    let elapsed = start.elapsed();
    Ok(PipelineResult {
        pipeline_id: pipeline_id.to_string(),
        detections: pipeline_detections,
        image_width: detect_result.image_width,
        image_height: detect_result.image_height,
        processing_time_ms: elapsed.as_secs_f32() * 1000.0,
    })
}

/// Run an ad-hoc pipeline: detect → crop → classify without pre-defined
/// TOML config. Auto-loads models by ID.
pub fn run_pipeline_adhoc(
    engine: &Engine,
    image: &ImageInput,
    detector_id: &str,
    classifier_id: &str,
    detect_opts: &DetectOpts,
    classify_opts: &ClassifyOpts,
) -> Result<PipelineResult> {
    let start = Instant::now();

    validate_adhoc_model_types(&engine.list_available_models(), detector_id, classifier_id)?;

    let detector_handle = engine.get_or_load_model(detector_id)?;
    let classifier_handle = engine.get_or_load_model(classifier_id)?;

    let detect_result = detect::detect(&detector_handle, image, detect_opts)?;
    let decoded = decode_image(image)?;
    let (orig_w, orig_h) = (decoded.width(), decoded.height());

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

/// Pin all model sessions referenced by the pipeline using the engine's
/// atomic batch lookup. Mirrors `sparrow_engine_cpu::pipeline::pin_all_sessions`.
fn pin_all_sessions(
    engine: &Engine,
    pipeline_id: &str,
    config: &PipelineManifest,
) -> Result<HashMap<String, ModelHandle>> {
    let model_ids: Vec<&str> = config.steps.iter().map(|s| s.model.as_str()).collect();
    let (found, missing_ids) = engine.get_model_handles(&model_ids);
    if !missing_ids.is_empty() {
        return Err(SparrowEngineError::PipelineMissingModels {
            id: pipeline_id.to_string(),
            missing: missing_ids.join(", "),
        });
    }
    Ok(found
        .into_iter()
        .map(|h| (h.model_id().to_string(), h))
        .collect())
}

// ---------------------------------------------------------------------------
// Image decode + crop (pure CPU helpers, identical to sparrow-engine-cpu).
// ---------------------------------------------------------------------------

/// Decode the original image to a `DynamicImage` for cropping.
///
/// Lives in this module rather than `crate::detect` because the GPU
/// detect path does not need a CPU-side decoded image (it goes through
/// nvjpeg + GPU letterbox). Pipeline does need the original pixels for
/// cropping, so we keep the entry-point here.
///
/// Phase 3.8 Phase C W1 audit-fix R2 (CR-1): delegates to
/// [`sparrow_engine_core::preprocess::decode_to_rgb`] for the actual decode, so
/// `sparrow-engine-cpu` and `sparrow-engine-gpu` share one byte-identical implementation
/// (subsumes reviewer F1-F4 error-variant fixes — sparrow-engine-core uses
/// [`SparrowEngineError::ImageDecode`] / [`SparrowEngineError::ImageFileNotFound`] /
/// [`SparrowEngineError::InvalidStride`] correctly).
pub(crate) fn decode_image(image: &ImageInput) -> Result<image::DynamicImage> {
    let rgb = sparrow_engine_core::preprocess::decode_to_rgb(image)?;
    Ok(image::DynamicImage::ImageRgb8(rgb))
}

/// Crop a detection region from the original image. Returns
/// `Some(ImageInput::Raw { .. })` with the crop pixels, or `None` if the
/// crop is degenerate.
fn crop_detection(
    img: &image::DynamicImage,
    bbox: &BBox,
    img_w: u32,
    img_h: u32,
) -> Option<ImageInput> {
    let x1 = (bbox.x_min * img_w as f32).round() as u32;
    let y1 = (bbox.y_min * img_h as f32).round() as u32;
    let x2 = (bbox.x_max * img_w as f32).round() as u32;
    let y2 = (bbox.y_max * img_h as f32).round() as u32;

    let x1 = x1.min(img_w);
    let y1 = y1.min(img_h);
    let x2 = x2.min(img_w).max(x1);
    let y2 = y2.min(img_h).max(y1);

    let crop_w = x2 - x1;
    let crop_h = y2 - y1;

    if crop_w < MIN_CROP_SIZE || crop_h < MIN_CROP_SIZE {
        return None;
    }

    let cropped = img.crop_imm(x1, y1, crop_w, crop_h);
    let rgb = cropped.to_rgb8();
    let width = rgb.width();
    let height = rgb.height();
    Some(ImageInput::Raw {
        stride: width * 3,
        width,
        height,
        data: rgb.into_raw(),
        format: PixelFormat::Rgb,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Note: `raw_rgb_round_trip` / `raw_bgr_swaps_channels` moved to
    // `sparrow-engine-core/src/preprocess.rs::tests` per Phase 3.8 Phase C W1
    // audit-fix R2 CR-1 (decode_to_rgb hoist). The `decode_image`
    // delegate is exercised transitively by every test that runs the
    // pipeline path.

    /// Helper: build a `DynamicImage` from a tight RGB pixel buffer via
    /// the public `decode_image` entry point (replaces the removed
    /// `raw_to_dynamic_image` private helper).
    fn dyn_from_rgb(pixels: Vec<u8>, width: u32, height: u32) -> image::DynamicImage {
        let img = ImageInput::Raw {
            data: pixels,
            width,
            height,
            stride: width * 3,
            format: PixelFormat::Rgb,
        };
        decode_image(&img).expect("decode_image")
    }

    #[test]
    fn crop_rejects_degenerate() {
        // Construct a 4x4 image and a bbox that maps to <2 pixels.
        let pixels: Vec<u8> = vec![128; 4 * 4 * 3];
        let img = dyn_from_rgb(pixels, 4, 4);
        let bbox = BBox {
            x_min: 0.0,
            y_min: 0.0,
            x_max: 0.1,
            y_max: 0.1,
        };
        let result = crop_detection(&img, &bbox, 4, 4);
        assert!(result.is_none());
    }

    #[test]
    fn crop_keeps_normal_bbox() {
        let pixels: Vec<u8> = vec![200; 8 * 8 * 3];
        let img = dyn_from_rgb(pixels, 8, 8);
        let bbox = BBox {
            x_min: 0.25,
            y_min: 0.25,
            x_max: 0.75,
            y_max: 0.75,
        };
        let result = crop_detection(&img, &bbox, 8, 8).expect("non-degenerate");
        if let ImageInput::Raw {
            width,
            height,
            format,
            ..
        } = result
        {
            assert_eq!(width, 4);
            assert_eq!(height, 4);
            assert_eq!(format, PixelFormat::Rgb);
        } else {
            panic!("expected ImageInput::Raw");
        }
    }

    fn info(id: &str, model_type: sparrow_engine_types::ModelType) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            path: std::path::PathBuf::from(format!("/models/{id}/manifest.toml")),
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
            info("owl-t", sparrow_engine_types::ModelType::OverheadDetector),
            info("speciesnet-crop", sparrow_engine_types::ModelType::Classifier),
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
        let available = vec![info("speciesnet-crop", sparrow_engine_types::ModelType::Classifier)];
        validate_adhoc_model_types(&available, "missing", "speciesnet-crop").unwrap();
    }
}
