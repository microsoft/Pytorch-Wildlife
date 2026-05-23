//! Detection inference: single-shot and tiled paths.
//!
//! Orchestrates: preprocess -> ORT session.run -> postprocess.
//! For tiled inference, splits the image into tiles, runs each through the
//! same pinned session, then finds peaks per-tile with coordinate mapping.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use ndarray::{ArrayView2, ArrayView4, ArrayViewD, Axis};
use ort::session::Session;
use ort::value::TensorRef;

use crate::engine::ModelHandle;
use crate::error::{SparrowEngineError, Result};
use crate::manifest::{InferenceStrategy, ModelManifest, PostprocessMethod, PreprocessMethod};
use crate::postprocess::{self, HeatmapConfig};
use crate::preprocess::PreprocessMeta;
use crate::preprocess::{self, PreprocessConfig};
use crate::types::{DetectOpts, DetectResult, ImageInput};

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that a manifest represents a vision detection model (not a classifier, not audio).
pub(crate) fn validate_vision_detector(manifest: &ModelManifest) -> Result<()> {
    if matches!(manifest.postprocess_method, PostprocessMethod::Softmax) {
        return Err(SparrowEngineError::NotADetector {
            id: manifest.id.clone(),
            method: "softmax".to_string(),
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
/// Validates model type, preprocesses, runs ORT, and postprocesses.
/// For tiled models, splits the image into tiles and assembles results.
///
/// # Errors
/// - `NotADetector` if the model's postprocessing method is `softmax`
/// - `ModelUnloaded` if the handle has been invalidated — also surfaces if the
///   engine itself has been dropped (post-S1 MT-17 mitigation: `Drop for Engine`
///   in `engine.rs` leaks `Arc<EngineInner>` so `Weak::upgrade()` keeps
///   succeeding; the signal the handle actually sees is the per-model `active`
///   flag that `Drop` clears before releasing sessions — see `docs/bugs.md`
///   MT-17 for the full rationale).
/// - `EngineFreed` reserved for pre-Drop paths (e.g. `Engine::unload_model`).
/// - `Ort` on ORT runtime errors
pub fn detect(handle: &ModelHandle, image: &ImageInput, opts: &DetectOpts) -> Result<DetectResult> {
    let start = Instant::now();

    // 1. Validate model type: reject classifiers and audio models.
    let manifest = &handle.manifest;
    validate_vision_detector(manifest)?;
    let _ = postprocess::resolve_confidence_threshold(
        opts.confidence_threshold,
        manifest.confidence_threshold.unwrap_or(0.0),
    )?;

    // 2. Pin session (checks handle validity: active flag + engine weak ref).
    let session = handle.pin_session()?;
    let labels = &*handle.labels;

    // 3. Dispatch based on inference strategy.
    let (detections, orig_w, orig_h) = match manifest.inference_strategy {
        InferenceStrategy::Single => detect_single(&session, manifest, labels, image, opts)?,
        InferenceStrategy::Tiled {
            tile_size,
            tile_overlap,
        } => detect_tiled(
            &session,
            manifest,
            labels,
            image,
            opts,
            tile_size,
            tile_overlap,
        )?,
        InferenceStrategy::SlidingWindow { .. } => {
            return Err(SparrowEngineError::IsAudioModel {
                id: manifest.id.clone(),
                method: manifest.preprocess_method.as_str().to_string(),
            });
        }
    };

    let elapsed = start.elapsed();

    Ok(DetectResult {
        detections,
        image_width: orig_w,
        image_height: orig_h,
        processing_time_ms: elapsed.as_secs_f32() * 1000.0,
    })
}

// ---------------------------------------------------------------------------
// Batch detection
// ---------------------------------------------------------------------------

/// Default batch size for image batch detection.
const DEFAULT_IMAGE_BATCH_SIZE: usize = 4;

/// Run detection on multiple images in batches. Each batch runs as a single ORT call
/// with a stacked [N, C, H, W] tensor for higher GPU throughput.
///
/// Only supports `Single` inference strategy. Tiled models fall back to per-image.
/// Invokes `on_result` after each image's detections are ready (for progress updates).
#[allow(clippy::type_complexity)]
pub fn detect_batch(
    handle: &ModelHandle,
    images: &[ImageInput],
    opts: &DetectOpts,
    batch_size: usize,
    mut on_result: Option<&mut dyn FnMut(usize, &DetectResult)>,
) -> Result<Vec<DetectResult>> {
    let start = Instant::now();

    let manifest = &handle.manifest;
    validate_vision_detector(manifest)?;
    let _ = postprocess::resolve_confidence_threshold(
        opts.confidence_threshold,
        manifest.confidence_threshold.unwrap_or(0.0),
    )?;

    // For tiled models, fall back to per-image detection.
    if !matches!(manifest.inference_strategy, InferenceStrategy::Single) {
        let mut results = Vec::with_capacity(images.len());
        for (i, image) in images.iter().enumerate() {
            let r = detect(handle, image, opts)?;
            if let Some(ref mut cb) = on_result {
                cb(i, &r);
            }
            results.push(r);
        }
        return Ok(results);
    }

    let session = handle.pin_session()?;
    let labels = &*handle.labels;
    let config = preprocess_config_from_manifest(manifest)?;
    let default_threshold = manifest.confidence_threshold;
    let bs = if batch_size == 0 {
        DEFAULT_IMAGE_BATCH_SIZE
    } else {
        batch_size
    };

    let mut all_results = Vec::with_capacity(images.len());

    for chunk_start in (0..images.len()).step_by(bs) {
        let chunk_end = (chunk_start + bs).min(images.len());
        let chunk = &images[chunk_start..chunk_end];
        let chunk_len = chunk.len();

        // 1. Preprocess each image.
        let mut preps: Vec<preprocess::PreprocessResult> = Vec::with_capacity(chunk_len);
        for image in chunk {
            preps.push(preprocess::preprocess(image, &config)?);
        }

        // 2. Stack tensors into [N, C, H, W].
        let views: Vec<ndarray::ArrayViewD<'_, f32>> =
            preps.iter().map(|p| p.tensor.view().into_dyn()).collect();
        let batch_tensor = ndarray::concatenate(ndarray::Axis(0), &views)
            .map_err(|e| SparrowEngineError::Ort(format!("batch concatenation failed: {e}")))?;

        // 3. Try batched ORT call. If model doesn't support batch > 1, fall back to per-image.
        let batch_ok = if chunk_len > 1 {
            // Try batched inference — clone output before releasing session lock.
            let batch_output: Option<ndarray::ArrayD<f32>> =
                (|| -> Option<ndarray::ArrayD<f32>> {
                    let input_value = TensorRef::from_array_view(&batch_tensor).ok()?;
                    let mut guard = session.lock().ok()?;
                    let outputs = guard.run(ort::inputs![input_value]).ok()?;
                    if outputs.len() == 0 {
                        return None;
                    }
                    let output_view = outputs[0].try_extract_array::<f32>().ok()?;
                    let owned = output_view.to_owned();
                    drop(outputs);
                    Some(owned)
                })();

            if let Some(output_owned) = batch_output {
                let shape: Vec<usize> = output_owned.shape().to_vec();

                // Validate batch output is rank-3 [N, ...] with correct batch size.
                if shape.len() != 3 || shape[0] != chunk_len {
                    return Err(SparrowEngineError::Ort(format!(
                        "Unexpected batched output shape: {shape:?} for batch {chunk_len}"
                    )));
                }

                for (i, prep) in preps.iter().enumerate() {
                    let meta = &prep.meta;
                    let per_image = output_owned
                        .view()
                        .index_axis(Axis(0), i)
                        .into_dimensionality::<ndarray::Ix2>()
                        .map_err(crate::engine::ort_err)?
                        .to_owned();
                    let detections = match &manifest.postprocess_method {
                        PostprocessMethod::YoloE2e => postprocess::try_yolo_e2e(
                            &per_image.view(),
                            labels,
                            opts,
                            meta,
                            default_threshold.unwrap_or(0.2),
                        )?,
                        PostprocessMethod::MegadetV5a { iou_threshold } => {
                            postprocess::try_megadet_v5a(
                                &per_image.view(),
                                labels,
                                opts,
                                meta,
                                default_threshold.unwrap_or(0.1),
                                *iou_threshold,
                            )?
                        }
                        // HeatmapPeaks: tiled only (falls back above).
                        // Sigmoid: audio only (rejected at entry).
                        // Softmax: rejected at entry.
                        _ => {
                            return Err(SparrowEngineError::Ort(format!(
                                "Batch detection not supported for {:?}",
                                manifest.postprocess_method,
                            )))
                        }
                    };
                    let elapsed = start.elapsed();
                    let result = DetectResult {
                        detections,
                        image_width: meta.original_width,
                        image_height: meta.original_height,
                        processing_time_ms: elapsed.as_secs_f32() * 1000.0,
                    };
                    if let Some(ref mut cb) = on_result {
                        cb(chunk_start + i, &result);
                    }
                    all_results.push(result);
                }
                true
            } else {
                false // Model doesn't support batching — fall back
            }
        } else {
            false // Single image, just use per-image path
        };

        // Fallback: process each image individually
        if !batch_ok {
            for (i, prep) in preps.into_iter().enumerate() {
                let input_value =
                    TensorRef::from_array_view(&prep.tensor).map_err(crate::engine::ort_err)?;
                let mut guard = session
                    .lock()
                    .map_err(|_| SparrowEngineError::Ort("detector session lock poisoned".into()))?;
                let outputs = guard
                    .run(ort::inputs![input_value])
                    .map_err(crate::engine::ort_err)?;
                let detections =
                    dispatch_postprocess(&outputs, labels, opts, &prep.meta, manifest)?;
                drop(outputs);
                drop(guard);

                let elapsed = start.elapsed();
                let result = DetectResult {
                    detections,
                    image_width: prep.meta.original_width,
                    image_height: prep.meta.original_height,
                    processing_time_ms: elapsed.as_secs_f32() * 1000.0,
                };
                if let Some(ref mut cb) = on_result {
                    cb(chunk_start + i, &result);
                }
                all_results.push(result);
            }
        }
    }

    Ok(all_results)
}

// ---------------------------------------------------------------------------
// Single-shot detection
// ---------------------------------------------------------------------------

/// Run single-shot detection: preprocess full image -> one ORT call -> postprocess.
///
/// Returns `(detections, original_width, original_height)`.
fn detect_single(
    session: &Arc<Mutex<Session>>,
    manifest: &crate::manifest::ModelManifest,
    labels: &[String],
    image: &ImageInput,
    opts: &DetectOpts,
) -> Result<(Vec<crate::types::Detection>, u32, u32)> {
    // Preprocess
    let config = preprocess_config_from_manifest(manifest)?;
    let prep = preprocess::preprocess(image, &config)?;
    let meta = prep.meta;
    let orig_w = meta.original_width;
    let orig_h = meta.original_height;

    // Create ORT input tensor from ndarray reference.
    let input_value = TensorRef::from_array_view(&prep.tensor).map_err(crate::engine::ort_err)?;

    // Lock session for exclusive ORT access. The guard must outlive `outputs`
    // because `SessionOutputs` borrows from the session.
    let mut guard = session
        .lock()
        .map_err(|_| SparrowEngineError::Ort("detector session lock poisoned".into()))?;
    let outputs = guard
        .run(ort::inputs![input_value])
        .map_err(crate::engine::ort_err)?;

    // Postprocess based on method (outputs borrow from session via guard).
    let detections = dispatch_postprocess(&outputs, labels, opts, &meta, manifest)?;
    drop(outputs);
    drop(guard);

    Ok((detections, orig_w, orig_h))
}

// ---------------------------------------------------------------------------
// Tiled detection
// ---------------------------------------------------------------------------

/// Run tiled detection: split image -> run each tile -> find peaks per tile -> map to full image.
///
/// Currently only supports heatmap-based models (HerdNet). Tiled detection
/// for YOLO/MegaDet models is undefined and out of MVP scope.
///
/// Processes each tile independently (matching the golden reference script):
/// peaks are found in each tile's heatmap, then mapped to full-image pixel
/// coordinates accounting for the heatmap-to-pixel scale factor.
///
/// Returns `(detections, original_width, original_height)`.
fn detect_tiled(
    session: &Arc<Mutex<Session>>,
    manifest: &crate::manifest::ModelManifest,
    labels: &[String],
    image: &ImageInput,
    opts: &DetectOpts,
    tile_size: [u32; 2],
    tile_overlap: u32,
) -> Result<(Vec<crate::types::Detection>, u32, u32)> {
    use crate::types::{BBox, Detection};

    // Validate tiled inference is only used with heatmap models.
    let heatmap_config = match &manifest.postprocess_method {
        PostprocessMethod::HeatmapPeaks {
            peak_threshold,
            adaptive,
            point_to_box_half_size,
        } => HeatmapConfig {
            peak_threshold: *peak_threshold,
            adaptive: *adaptive,
            point_to_box_half_size: *point_to_box_half_size,
        },
        other => {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "Tiled inference is only supported for heatmap_peaks models, \
                 got postprocessing method: {other:?}",
            )));
        }
    };

    // Decode the original image to get dimensions and pixel data for cropping.
    let decoded = decode_image(image)?;
    let (img_w, img_h) = (decoded.width(), decoded.height());

    let tw = tile_size[0];
    let th = tile_size[1];
    let stride_x = tw.saturating_sub(tile_overlap).max(1);
    let stride_y = th.saturating_sub(tile_overlap).max(1);

    let config = preprocess_config_from_manifest(manifest)?;

    let threshold = opts
        .confidence_threshold
        .unwrap_or(heatmap_config.peak_threshold);
    let half = heatmap_config.point_to_box_half_size as f32;
    let img_wf = img_w as f32;
    let img_hf = img_h as f32;

    let mut all_detections = Vec::new();

    // Iterate over tiles, process peaks per-tile.
    let mut y = 0u32;
    while y < img_h {
        let mut x = 0u32;
        while x < img_w {
            // Crop tile from original image.
            let crop_w = tw.min(img_w - x);
            let crop_h = th.min(img_h - y);
            let tile_img = decoded.crop_imm(x, y, crop_w, crop_h);

            // Pad edge tiles to tile_size with black pixels (matching the
            // golden script, which pads the full image to tile grid boundaries).
            // Without padding, resize_direct() stretches partial tiles,
            // producing different model inputs than the reference.
            let tile_rgb = if crop_w < tw || crop_h < th {
                use image::GenericImage;
                let mut padded = image::RgbImage::new(tw, th);
                padded
                    .copy_from(&tile_img.to_rgb8(), 0, 0)
                    .expect("crop fits within padded tile");
                padded
            } else {
                tile_img.to_rgb8()
            };

            // Convert tile to ImageInput for preprocessing.
            let tile_input = ImageInput::Raw {
                stride: tw * 3,
                width: tw,
                height: th,
                data: tile_rgb.into_raw(),
                format: crate::types::PixelFormat::Rgb,
            };

            // Preprocess the tile.
            let prep = preprocess::preprocess(&tile_input, &config)?;

            // Create ORT input and run on the SAME pinned session.
            let input_value =
                TensorRef::from_array_view(&prep.tensor).map_err(crate::engine::ort_err)?;

            let mut guard = session
                .lock()
                .map_err(|_| SparrowEngineError::Ort("detector session lock poisoned".into()))?;
            let outputs = guard
                .run(ort::inputs![input_value])
                .map_err(crate::engine::ort_err)?;

            if outputs.len() == 0 {
                return Err(SparrowEngineError::Ort(
                    "tiled detector session returned no outputs".to_string(),
                ));
            }

            // Extract heatmap outputs.
            // Dual-output models (HerdNet): output[0] = loc_map, output[1] = cls_map
            // Single-output models (OWL-T): output[0] = heatmap only (single class)
            let loc_view: ArrayViewD<'_, f32> = outputs[0]
                .try_extract_array::<f32>()
                .map_err(crate::engine::ort_err)?;
            let has_cls = outputs.len() > 1;
            let cls_4d = if has_cls {
                let cls_view: ArrayViewD<'_, f32> = outputs[1]
                    .try_extract_array::<f32>()
                    .map_err(crate::engine::ort_err)?;
                Some(
                    cls_view
                        .into_dimensionality::<ndarray::Ix4>()
                        .map_err(crate::engine::ort_err)?
                        .to_owned(),
                )
            } else {
                None
            };

            let loc_4d = loc_view
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(crate::engine::ort_err)?
                .to_owned();

            if let Some(ref cls) = cls_4d {
                postprocess::validate_heatmap_maps(
                    &loc_4d.view(),
                    Some(&cls.view()),
                    "tiled detector",
                )?;
            } else {
                postprocess::validate_heatmap_maps(&loc_4d.view(), None, "tiled detector")?;
            }

            drop(outputs);
            drop(guard);

            // --- Per-tile peak finding ---
            let loc_h = loc_4d.shape()[2];
            let loc_w = loc_4d.shape()[3];

            // Scale from heatmap coordinates to tile pixel coordinates.
            let scale_x = tw as f32 / loc_w as f32;
            let scale_y = th as f32 / loc_h as f32;

            let loc_view4 = loc_4d.view();

            // For adaptive thresholding on single-output models (OWL-T style):
            // threshold = max(peak_threshold, tile_max * peak_threshold, 0.1)
            // For sigmoid-bounded heatmaps ([0,1]), tile_max <= 1.0 so the
            // adaptive term never exceeds peak_threshold — effectively a no-op.
            // The scaling only activates for unbounded heatmaps (raw logits > 1.0).
            // See owl_adaptive_threshold() doc comment for details.
            let effective_base_threshold = if !has_cls && heatmap_config.adaptive {
                let tile_max = loc_4d.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                postprocess::owl_adaptive_threshold(threshold, tile_max)
            } else {
                threshold
            };

            for py in 0..loc_h {
                for px in 0..loc_w {
                    let val = loc_4d[[0, 0, py, px]];
                    if val < effective_base_threshold {
                        continue;
                    }

                    // 8-connected local maximum check (with plateau tie-breaking).
                    if !postprocess::is_local_maximum(&loc_view4, py, px, loc_h, loc_w) {
                        continue;
                    }

                    // Classify: if dual-output, use cls_map. If single-output, class=0, score=1.
                    let (class_id, confidence) = if let Some(ref cls) = cls_4d {
                        let cls_h = cls.shape()[2];
                        let cls_w = cls.shape()[3];
                        let num_classes = cls.shape()[1];

                        let cy = if cls_h == loc_h {
                            py
                        } else {
                            (py * cls_h / loc_h).min(cls_h - 1)
                        };
                        let cx = if cls_w == loc_w {
                            px
                        } else {
                            (px * cls_w / loc_w).min(cls_w - 1)
                        };

                        let mut best_id = 0usize;
                        let mut best_val = f32::NEG_INFINITY;
                        for c in 0..num_classes {
                            let v = cls[[0, c, cy, cx]];
                            if v > best_val {
                                best_val = v;
                                best_id = c;
                            }
                        }
                        (best_id, val * best_val)
                    } else {
                        // Single-output: heatmap value IS the confidence, class = 0
                        (0usize, val)
                    };

                    if confidence < threshold {
                        continue;
                    }

                    // Map heatmap position to full-image pixel coordinates.
                    let full_px = x as f32 + px as f32 * scale_x;
                    let full_py = y as f32 + py as f32 * scale_y;

                    // Point to bbox in [0,1] coords relative to original image.
                    // half (point_to_box_half_size) is in pixel coordinates.
                    let x_min = ((full_px - half) / img_wf).clamp(0.0, 1.0);
                    let y_min = ((full_py - half) / img_hf).clamp(0.0, 1.0);
                    let x_max = ((full_px + half) / img_wf).clamp(0.0, 1.0);
                    let y_max = ((full_py + half) / img_hf).clamp(0.0, 1.0);

                    let label = postprocess::label_for_id(labels, class_id as u32);

                    all_detections.push(Detection {
                        bbox: BBox {
                            x_min,
                            y_min,
                            x_max,
                            y_max,
                        },
                        label,
                        label_id: class_id as u32,
                        confidence,
                    });
                }
            }

            x += stride_x;
        }
        y += stride_y;
    }

    all_detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Deduplicate detections from overlapping tiles. When tile_overlap > 0,
    // the same peak near tile boundaries is detected in multiple tiles.
    // Suppress lower-confidence duplicates whose bbox centers are within
    // 2 * point_to_box_half_size pixels of a higher-confidence detection.
    if tile_overlap > 0 {
        deduplicate_tiled(&mut all_detections, img_wf, img_hf, half * 2.0);
    }

    postprocess::apply_max_detections(&mut all_detections, opts.max_detections);

    Ok((all_detections, img_w, img_h))
}

// ---------------------------------------------------------------------------
// Postprocess dispatch (single-shot path)
// ---------------------------------------------------------------------------

/// Dispatch to the correct postprocessor based on manifest method.
fn dispatch_postprocess(
    outputs: &ort::session::SessionOutputs,
    labels: &[String],
    opts: &DetectOpts,
    meta: &PreprocessMeta,
    manifest: &crate::manifest::ModelManifest,
) -> Result<Vec<crate::types::Detection>> {
    let default_threshold = manifest.confidence_threshold;

    match &manifest.postprocess_method {
        PostprocessMethod::YoloE2e => {
            if outputs.len() == 0 {
                return Err(SparrowEngineError::Ort(
                    "yolo_e2e session returned no outputs".to_string(),
                ));
            }
            let output_view: ArrayViewD<'_, f32> = outputs[0]
                .try_extract_array::<f32>()
                .map_err(crate::engine::ort_err)?;

            let shape = output_view.shape();
            // yolo_e2e expects [N, 6] -- handle [N, 6] or [batch, N, 6].
            let view_2d: ArrayView2<f32> = if shape.len() == 2 {
                output_view
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(crate::engine::ort_err)?
            } else if shape.len() == 3 {
                // [batch, N, 6] -> squeeze batch dim
                let squeezed = output_view.index_axis(Axis(0), 0);
                squeezed
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(crate::engine::ort_err)?
            } else {
                return Err(SparrowEngineError::Ort(format!(
                    "Unexpected yolo_e2e output shape: {shape:?}",
                )));
            };

            postprocess::try_yolo_e2e(
                &view_2d,
                labels,
                opts,
                meta,
                default_threshold.unwrap_or(0.2),
            )
        }
        PostprocessMethod::MegadetV5a { iou_threshold } => {
            if outputs.len() == 0 {
                return Err(SparrowEngineError::Ort(
                    "megadet_v5a session returned no outputs".to_string(),
                ));
            }
            let output_view: ArrayViewD<'_, f32> = outputs[0]
                .try_extract_array::<f32>()
                .map_err(crate::engine::ort_err)?;

            let shape = output_view.shape();
            let view_2d: ArrayView2<f32> = if shape.len() == 2 {
                output_view
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(crate::engine::ort_err)?
            } else if shape.len() == 3 {
                let squeezed = output_view.index_axis(Axis(0), 0);
                squeezed
                    .into_dimensionality::<ndarray::Ix2>()
                    .map_err(crate::engine::ort_err)?
            } else {
                return Err(SparrowEngineError::Ort(format!(
                    "Unexpected megadet_v5a output shape: {shape:?}",
                )));
            };

            postprocess::try_megadet_v5a(
                &view_2d,
                labels,
                opts,
                meta,
                default_threshold.unwrap_or(0.1),
                *iou_threshold,
            )
        }
        PostprocessMethod::HeatmapPeaks {
            peak_threshold,
            adaptive,
            point_to_box_half_size,
        } => {
            // Single-output heatmap models (e.g., OWL-T) require tiled inference.
            // If reached here via single-shot path, return an error instead of
            // panicking on missing outputs[1].
            if outputs.len() < 2 {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "Single-output heatmap model requires tiled inference strategy. \
                     Model has {} output(s); heatmap_peaks single-shot needs 2 \
                     (loc_map + cls_map).",
                    outputs.len(),
                )));
            }

            let loc_view: ArrayViewD<'_, f32> = outputs[0]
                .try_extract_array::<f32>()
                .map_err(crate::engine::ort_err)?;
            let cls_view: ArrayViewD<'_, f32> = outputs[1]
                .try_extract_array::<f32>()
                .map_err(crate::engine::ort_err)?;

            let loc_4d: ArrayView4<f32> = loc_view
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(crate::engine::ort_err)?;
            let cls_4d: ArrayView4<f32> = cls_view
                .into_dimensionality::<ndarray::Ix4>()
                .map_err(crate::engine::ort_err)?;

            let config = HeatmapConfig {
                peak_threshold: *peak_threshold,
                adaptive: *adaptive,
                point_to_box_half_size: *point_to_box_half_size,
            };

            postprocess::try_heatmap_peaks(&loc_4d, &cls_4d, labels, opts, &config)
        }
        PostprocessMethod::Softmax => {
            // This branch is unreachable because we check for Softmax at entry.
            unreachable!("Softmax models are rejected at the start of detect()");
        }
        PostprocessMethod::Sigmoid { .. } => Err(SparrowEngineError::IsAudioModel {
            id: manifest.id.clone(),
            method: manifest.postprocess_method.as_str().to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a `PreprocessConfig` from a `ModelManifest`.
///
/// Only valid for image models. Audio models will have `None` for image fields;
/// callers must guard against this (detect/classify reject audio models at entry).
pub(crate) fn preprocess_config_from_manifest(
    manifest: &crate::manifest::ModelManifest,
) -> Result<PreprocessConfig> {
    Ok(PreprocessConfig {
        method: manifest.preprocess_method.clone(),
        input_size: manifest
            .input_size
            .ok_or_else(|| SparrowEngineError::NotAnAudioModel {
                id: manifest.id.clone(),
                method: format!("{:?}", manifest.preprocess_method),
            })?,
        layout: manifest.layout.ok_or_else(|| SparrowEngineError::NotAnAudioModel {
            id: manifest.id.clone(),
            method: format!("{:?}", manifest.preprocess_method),
        })?,
        normalization: manifest
            .normalization
            .ok_or_else(|| SparrowEngineError::NotAnAudioModel {
                id: manifest.id.clone(),
                method: format!("{:?}", manifest.preprocess_method),
            })?,
        pad_value: manifest.pad_value.unwrap_or(0.0),
        channel_order: manifest.channel_order.unwrap_or_default(),
    })
}

/// Deduplicate detections from overlapping tiles by center proximity.
///
/// When tiles overlap, the same peak near a tile boundary is detected in
/// multiple tiles, producing near-duplicate detections. This function
/// suppresses lower-confidence duplicates whose bbox centers are within
/// `radius_pixels` of a higher-confidence detection (in full-image pixel
/// coordinates).
///
/// Expects `detections` sorted by confidence descending (greedy suppression).
pub(crate) fn deduplicate_tiled(
    detections: &mut Vec<crate::types::Detection>,
    img_w: f32,
    img_h: f32,
    radius_pixels: f32,
) {
    if detections.len() <= 1 {
        return;
    }
    let r_sq = radius_pixels * radius_pixels;
    let mut keep = vec![true; detections.len()];
    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        let ci_x = (detections[i].bbox.x_min + detections[i].bbox.x_max) * 0.5 * img_w;
        let ci_y = (detections[i].bbox.y_min + detections[i].bbox.y_max) * 0.5 * img_h;
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            let cj_x = (detections[j].bbox.x_min + detections[j].bbox.x_max) * 0.5 * img_w;
            let cj_y = (detections[j].bbox.y_min + detections[j].bbox.y_max) * 0.5 * img_h;
            let dist_sq = (ci_x - cj_x).powi(2) + (ci_y - cj_y).powi(2);
            if dist_sq < r_sq {
                keep[j] = false;
            }
        }
    }
    let mut idx = 0;
    detections.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Decode an `ImageInput` into a `DynamicImage`.
///
/// Delegates to `preprocess::decode_to_rgb` for consistent decoding logic
/// (including buffer-length validation for Raw inputs).
pub(crate) fn decode_image(image: &ImageInput) -> Result<image::DynamicImage> {
    let rgb = crate::preprocess::decode_to_rgb(image)?;
    Ok(image::DynamicImage::ImageRgb8(rgb))
}

// ---------------------------------------------------------------------------
// Integration tests needed (require ORT session creation)
// ---------------------------------------------------------------------------
// Integration test needed: tiled detection produces correct spatial output
// Integration test needed: concurrent detect calls on same model

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::deduplicate_tiled;
    use crate::types::{BBox, Detection};

    /// Helper: create a Detection at the given normalized bbox center with a
    /// fixed half-size (for 6000x4000 image, half=10px → bbox width = 20/6000).
    fn make_det(center_x_px: f32, center_y_px: f32, confidence: f32) -> Detection {
        let img_w = 6000.0_f32;
        let img_h = 4000.0_f32;
        let half = 10.0_f32;
        Detection {
            bbox: BBox {
                x_min: (center_x_px - half) / img_w,
                y_min: (center_y_px - half) / img_h,
                x_max: (center_x_px + half) / img_w,
                y_max: (center_y_px + half) / img_h,
            },
            label: "animal".to_string(),
            label_id: 0,
            confidence,
        }
    }

    #[test]
    fn test_dedup_exact_duplicates() {
        // Two detections at the exact same pixel position; higher confidence kept.
        let mut dets = vec![make_det(100.0, 200.0, 0.9), make_det(100.0, 200.0, 0.7)];
        deduplicate_tiled(&mut dets, 6000.0, 4000.0, 20.0);
        assert_eq!(dets.len(), 1, "Exact duplicates should collapse to 1");
        assert!(
            (dets[0].confidence - 0.9).abs() < 1e-6,
            "Higher confidence kept"
        );
    }

    #[test]
    fn test_dedup_near_duplicates() {
        // Two detections within radius (5px apart < 20px radius); higher confidence kept.
        let mut dets = vec![make_det(100.0, 200.0, 0.8), make_det(105.0, 200.0, 0.6)];
        deduplicate_tiled(&mut dets, 6000.0, 4000.0, 20.0);
        assert_eq!(
            dets.len(),
            1,
            "Near-duplicates within radius should collapse"
        );
        assert!(
            (dets[0].confidence - 0.8).abs() < 1e-6,
            "Higher confidence kept"
        );
    }

    #[test]
    fn test_dedup_distinct_detections() {
        // Two detections far apart (500px > 20px radius); both kept.
        let mut dets = vec![make_det(100.0, 200.0, 0.9), make_det(600.0, 200.0, 0.7)];
        deduplicate_tiled(&mut dets, 6000.0, 4000.0, 20.0);
        assert_eq!(dets.len(), 2, "Distinct detections should both be kept");
    }

    #[test]
    fn test_dedup_empty() {
        let mut dets: Vec<Detection> = vec![];
        deduplicate_tiled(&mut dets, 6000.0, 4000.0, 20.0);
        assert!(dets.is_empty(), "Empty input should produce empty output");
    }

    #[test]
    fn test_dedup_single() {
        let mut dets = vec![make_det(100.0, 200.0, 0.5)];
        deduplicate_tiled(&mut dets, 6000.0, 4000.0, 20.0);
        assert_eq!(dets.len(), 1, "Single detection should be unchanged");
    }
}
