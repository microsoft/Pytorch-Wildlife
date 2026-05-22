//! Postprocessing: confidence filter, softmax, heatmap peak finding, bbox normalization.
//!
//! NMS is in the ONNX graph, never in sparrow-engine. These functions only handle:
//! - Confidence filtering
//! - Softmax for classification
//! - Heatmap peak finding (local maxima)
//! - Letterbox denormalization + [0,1] normalization

use ndarray::{ArrayView2, ArrayView4, Axis};

use sparrow_engine_types::{
    BBox, Classification, ClassifyOpts, DetectOpts, Detection, PreprocessMeta, Result,
    SparrowEngineError,
};

/// Sort detections by confidence (descending) and cap to max_detections if specified.
///
/// `pub(crate)` — internal helper. The other `pub` postprocess functions are flipped
/// for cross-crate use by `sparrow-engine-cpu` (label_for_id, apply_max_detections,
/// owl_adaptive_threshold, is_local_maximum); this one has no cross-crate caller.
pub(crate) fn sort_desc_and_cap(detections: &mut Vec<Detection>, max_detections: Option<u32>) {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    apply_max_detections(detections, max_detections);
}

/// Resolve and validate a confidence threshold before inference output is interpreted.
pub fn resolve_confidence_threshold(
    override_threshold: Option<f32>,
    default_threshold: f32,
) -> Result<f32> {
    let threshold = override_threshold.unwrap_or(default_threshold);
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err(SparrowEngineError::Ort(format!(
            "confidence threshold must be finite and in [0.0, 1.0], got {threshold}"
        )));
    }
    Ok(threshold)
}

/// Validate letterbox metadata before using it to normalize model-space boxes.
pub fn validate_preprocess_meta(meta: &PreprocessMeta) -> Result<()> {
    if meta.original_width == 0 || meta.original_height == 0 {
        return Err(SparrowEngineError::Ort(format!(
            "preprocess metadata has zero original dimensions: {}x{}",
            meta.original_width, meta.original_height
        )));
    }
    if !meta.scale.is_finite() || meta.scale <= 0.0 {
        return Err(SparrowEngineError::Ort(format!(
            "preprocess metadata scale must be finite and positive, got {}",
            meta.scale
        )));
    }
    if !meta.pad_x.is_finite() || !meta.pad_y.is_finite() {
        return Err(SparrowEngineError::Ort(format!(
            "preprocess metadata padding must be finite, got ({}, {})",
            meta.pad_x, meta.pad_y
        )));
    }
    Ok(())
}

/// Validate non-empty heatmap tensors before code maps coordinates or indexes channels.
pub fn validate_heatmap_maps(
    loc_map: &ArrayView4<f32>,
    cls_map: Option<&ArrayView4<f32>>,
    method: &str,
) -> Result<()> {
    let loc_shape = loc_map.shape();
    if loc_shape[0] != 1 || loc_shape[1] != 1 || loc_shape[2] == 0 || loc_shape[3] == 0 {
        return Err(SparrowEngineError::Ort(format!(
            "{method} expects loc_map shape [1, 1, H, W] with non-empty H/W, got {loc_shape:?}"
        )));
    }
    if !loc_map.iter().all(|v| v.is_finite()) {
        return Err(SparrowEngineError::Ort(format!(
            "{method} loc_map contains non-finite values"
        )));
    }
    if let Some(cls_map) = cls_map {
        let cls_shape = cls_map.shape();
        if cls_shape[0] != 1 || cls_shape[1] == 0 || cls_shape[2] == 0 || cls_shape[3] == 0 {
            return Err(SparrowEngineError::Ort(format!(
                "{method} expects cls_map shape [1, C, H, W] with non-empty C/H/W, got {cls_shape:?}"
            )));
        }
        if !cls_map.iter().all(|v| v.is_finite()) {
            return Err(SparrowEngineError::Ort(format!(
                "{method} cls_map contains non-finite values"
            )));
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Heatmap config
// ---------------------------------------------------------------------------

/// Configuration for heatmap peak finding (HerdNet-style models).
#[derive(Debug, Clone, Copy)]
pub struct HeatmapConfig {
    /// Minimum value for a pixel to be considered a peak.
    pub peak_threshold: f32,
    /// If true, use adaptive thresholding (local-mean based).
    pub adaptive: bool,
    /// Half-size of the bounding box generated around each peak point.
    pub point_to_box_half_size: u32,
}

// ---------------------------------------------------------------------------
// Detection postprocessors
// ---------------------------------------------------------------------------

/// Postprocess YOLO end-to-end output (NMS already applied in ONNX graph).
///
/// `output` shape: `[N, 6]` — columns: x1, y1, x2, y2, confidence, class_id.
/// Coordinates are in model-input pixel space (letterboxed).
pub fn yolo_e2e(
    output: &ArrayView2<f32>,
    labels: &[String],
    opts: &DetectOpts,
    meta: &PreprocessMeta,
    default_threshold: f32,
) -> Vec<Detection> {
    try_yolo_e2e(output, labels, opts, meta, default_threshold).unwrap_or_default()
}

/// Fallible YOLO E2E postprocessor used by runtime inference paths.
pub fn try_yolo_e2e(
    output: &ArrayView2<f32>,
    labels: &[String],
    opts: &DetectOpts,
    meta: &PreprocessMeta,
    default_threshold: f32,
) -> Result<Vec<Detection>> {
    let threshold = resolve_confidence_threshold(opts.confidence_threshold, default_threshold)?;
    validate_preprocess_meta(meta)?;
    let ncols = output.ncols();
    if ncols != 6 {
        return Err(SparrowEngineError::Ort(format!(
            "yolo_e2e expects exactly [N, 6], got {ncols} columns"
        )));
    }

    let mut detections = Vec::new();

    for row in output.rows() {
        if !row.iter().all(|v| v.is_finite()) {
            return Err(SparrowEngineError::Ort(
                "yolo_e2e output contains non-finite values".to_string(),
            ));
        }
        let confidence = row[4];
        if !(0.0..=1.0).contains(&confidence) {
            return Err(SparrowEngineError::Ort(
                "yolo_e2e output contains scores outside [0.0, 1.0]".to_string(),
            ));
        }

        // Skip below-threshold rows BEFORE bbox geometry validation.
        // YOLOv10e exports many low-confidence TopK candidates whose boxes
        // can lie outside the original image; after letterbox-undo and clamp
        // they may end up degenerate (x_min == x_max). Those rows are already
        // discarded by the confidence gate; their post-clamp geometry is
        // irrelevant. High-confidence degenerate boxes still signal a model
        // fault and are rejected below.
        if confidence < threshold {
            continue;
        }

        let bbox = denormalize_and_normalize(row[0], row[1], row[2], row[3], meta);
        if bbox.x_min >= bbox.x_max || bbox.y_min >= bbox.y_max {
            return Err(SparrowEngineError::Ort(
                "yolo_e2e output contains degenerate normalized boxes".to_string(),
            ));
        }

        if row[5] < 0.0 {
            return Err(SparrowEngineError::Ort(format!(
                "yolo_e2e class_id must be non-negative, got {}",
                row[5]
            )));
        }
        let class_id = row[5] as u32;
        let label = label_for_id(labels, class_id);

        detections.push(Detection {
            bbox,
            label,
            label_id: class_id,
            confidence,
        });
    }

    // Sort descending by confidence for deterministic cap behavior.
    sort_desc_and_cap(&mut detections, opts.max_detections);
    Ok(detections)
}

/// Postprocess MegaDetector v5a output (objectness * class_scores).
///
/// `output` shape: `[N, 5+num_classes]` — columns: cx, cy, w, h, objectness, class_scores...
/// Coordinates are in model-input pixel space (letterboxed), CENTER-format
/// (cx, cy, w, h) per YOLOv5's raw decoded head — NMS is NOT in graph for v5,
/// so this postprocessor runs greedy class-aware NMS internally.
pub fn megadet_v5a(
    output: &ArrayView2<f32>,
    labels: &[String],
    opts: &DetectOpts,
    meta: &PreprocessMeta,
    default_threshold: f32,
    iou_threshold: f32,
) -> Vec<Detection> {
    try_megadet_v5a(output, labels, opts, meta, default_threshold, iou_threshold)
        .unwrap_or_default()
}

/// Fallible MegaDetector v5a postprocessor used by runtime inference paths.
pub fn try_megadet_v5a(
    output: &ArrayView2<f32>,
    labels: &[String],
    opts: &DetectOpts,
    meta: &PreprocessMeta,
    default_threshold: f32,
    iou_threshold: f32,
) -> Result<Vec<Detection>> {
    let threshold = resolve_confidence_threshold(opts.confidence_threshold, default_threshold)?;
    validate_preprocess_meta(meta)?;
    if !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold) {
        return Err(SparrowEngineError::Ort(format!(
            "megadet_v5a iou_threshold must be finite and in [0.0, 1.0], got {iou_threshold}"
        )));
    }
    let ncols = output.ncols();
    if ncols <= 5 {
        return Err(SparrowEngineError::Ort(format!(
            "megadet_v5a expects [N, 5+C], got {ncols} columns"
        )));
    }

    let num_classes = ncols - 5;
    let mut detections = Vec::new();

    for row in output.rows() {
        if !row.iter().all(|v| v.is_finite()) {
            return Err(SparrowEngineError::Ort(
                "megadet_v5a output contains non-finite values".to_string(),
            ));
        }
        let objectness = row[4];

        // Find argmax and max value among class scores.
        let (class_id, max_class_score) = argmax_slice(&row, 5, 5 + num_classes);
        if !(0.0..=1.0).contains(&objectness) || !(0.0..=1.0).contains(&max_class_score) {
            return Err(SparrowEngineError::Ort(
                "megadet_v5a output contains scores outside [0.0, 1.0]".to_string(),
            ));
        }

        // YOLOv5 raw rows are (cx, cy, w, h) in model-input pixel space —
        // convert to (x1, y1, x2, y2) before letterbox de-normalize.
        let (cx, cy, w, h) = (row[0], row[1], row[2], row[3]);
        if w <= 0.0 || h <= 0.0 {
            return Err(SparrowEngineError::Ort(
                "megadet_v5a output contains non-positive box size".to_string(),
            ));
        }

        // Compute confidence and skip below-threshold rows BEFORE bbox
        // geometry validation. Rationale mirrors try_yolo_e2e: low-confidence
        // candidates whose boxes lie outside the original image clamp to
        // degenerate shapes; those rows are already discarded by the gate.
        // High-confidence degenerate boxes still signal a model fault and
        // are rejected below.
        let confidence = objectness * max_class_score;
        if confidence < threshold {
            continue;
        }

        let half_w = w * 0.5;
        let half_h = h * 0.5;
        let bbox =
            denormalize_and_normalize(cx - half_w, cy - half_h, cx + half_w, cy + half_h, meta);
        if bbox.x_min >= bbox.x_max || bbox.y_min >= bbox.y_max {
            return Err(SparrowEngineError::Ort(
                "megadet_v5a output contains degenerate normalized boxes".to_string(),
            ));
        }

        let label = label_for_id(labels, class_id as u32);
        detections.push(Detection {
            bbox,
            label,
            label_id: class_id as u32,
            confidence,
        });
    }

    // Greedy class-aware NMS on normalized [0,1] bboxes.
    let kept = nms(detections, iou_threshold);
    let mut kept = kept;
    sort_desc_and_cap(&mut kept, opts.max_detections);
    Ok(kept)
}

/// Greedy class-aware non-max suppression on already-confidence-sorted detections.
/// `iou_threshold` ∈ [0,1]: suppress B if `IoU(A, B) >= iou_threshold` and A has
/// higher confidence and same `label_id`. Operates on normalized [0,1] bboxes.
fn nms(mut detections: Vec<Detection>, iou_threshold: f32) -> Vec<Detection> {
    detections.sort_by(|a, b| {
        b.confidence
            .partial_cmp(&a.confidence)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut kept: Vec<Detection> = Vec::with_capacity(detections.len());
    for det in detections {
        let suppress = kept
            .iter()
            .any(|k| k.label_id == det.label_id && bbox_iou(&k.bbox, &det.bbox) >= iou_threshold);
        if !suppress {
            kept.push(det);
        }
    }
    kept
}

/// IoU of two normalized-[0,1] bboxes. Returns 0.0 on degenerate boxes.
fn bbox_iou(a: &sparrow_engine_types::BBox, b: &sparrow_engine_types::BBox) -> f32 {
    let x1 = a.x_min.max(b.x_min);
    let y1 = a.y_min.max(b.y_min);
    let x2 = a.x_max.min(b.x_max);
    let y2 = a.y_max.min(b.y_max);
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let area_a = (a.x_max - a.x_min).max(0.0) * (a.y_max - a.y_min).max(0.0);
    let area_b = (b.x_max - b.x_min).max(0.0) * (b.y_max - b.y_min).max(0.0);
    let union = area_a + area_b - inter;
    if union <= 0.0 {
        0.0
    } else {
        inter / union
    }
}

/// Find peaks in a location heatmap and classify using a class heatmap.
///
/// `loc_map` shape: `[batch, 1, H, W]` — location confidence heatmap.
/// `cls_map` shape: `[batch, num_classes, H, W]` — per-class confidence heatmaps.
pub fn heatmap_peaks(
    loc_map: &ArrayView4<f32>,
    cls_map: &ArrayView4<f32>,
    labels: &[String],
    opts: &DetectOpts,
    config: &HeatmapConfig,
) -> Vec<Detection> {
    try_heatmap_peaks(loc_map, cls_map, labels, opts, config).unwrap_or_default()
}

/// Fallible heatmap postprocessor used by runtime inference paths.
pub fn try_heatmap_peaks(
    loc_map: &ArrayView4<f32>,
    cls_map: &ArrayView4<f32>,
    labels: &[String],
    opts: &DetectOpts,
    config: &HeatmapConfig,
) -> Result<Vec<Detection>> {
    let threshold = resolve_confidence_threshold(opts.confidence_threshold, config.peak_threshold)?;
    validate_heatmap_maps(loc_map, Some(cls_map), "heatmap_peaks")?;
    if !config.peak_threshold.is_finite() || !(0.0..=1.0).contains(&config.peak_threshold) {
        return Err(SparrowEngineError::Ort(format!(
            "heatmap_peaks peak_threshold must be finite and in [0.0, 1.0], got {}",
            config.peak_threshold
        )));
    }

    let loc_shape = loc_map.shape();
    let cls_shape = cls_map.shape();
    let h = loc_shape[2];
    let w = loc_shape[3];
    if cls_shape[2] < h || cls_shape[3] < w {
        return Err(SparrowEngineError::Ort(format!(
            "heatmap_peaks expects cls_map spatial dims >= loc_map dims, got loc_map {loc_shape:?}, cls_map {cls_shape:?}"
        )));
    }
    let num_classes = cls_shape[1];

    let mut detections = Vec::new();

    // Iterate over spatial positions looking for local maxima.
    for y in 0..h {
        for x in 0..w {
            let val = loc_map[[0, 0, y, x]];

            let effective_threshold = if config.adaptive {
                adaptive_threshold(loc_map, y, x, h, w, threshold)
            } else {
                threshold
            };

            if val < effective_threshold {
                continue;
            }

            // Check 8-connected neighborhood: pixel must be >= all neighbors.
            if !is_local_maximum(loc_map, y, x, h, w) {
                continue;
            }

            // Classify: argmax over class dimension at this spatial location.
            // loc_map and cls_map may have different spatial resolutions
            // (e.g., HerdNet: loc=256x256, cls=16x16). Map coordinates proportionally.
            let cls_h = cls_map.shape()[2];
            let cls_w = cls_map.shape()[3];
            let cls_y = if cls_h == h {
                y
            } else {
                (y * cls_h / h).min(cls_h - 1)
            };
            let cls_x = if cls_w == w {
                x
            } else {
                (x * cls_w / w).min(cls_w - 1)
            };

            let (class_id, class_score) = {
                let mut best_id = 0usize;
                let mut best_val = f32::NEG_INFINITY;
                for c in 0..num_classes {
                    let v = cls_map[[0, c, cls_y, cls_x]];
                    if v > best_val {
                        best_val = v;
                        best_id = c;
                    }
                }
                (best_id, best_val)
            };

            let confidence = val * class_score;
            if confidence < threshold {
                continue;
            }

            // Convert point to bbox in normalized [0,1] coordinates.
            let half = config.point_to_box_half_size as f32;
            let hf = h as f32;
            let wf = w as f32;

            let x_min = ((x as f32 - half) / wf).clamp(0.0, 1.0);
            let y_min = ((y as f32 - half) / hf).clamp(0.0, 1.0);
            let x_max = ((x as f32 + half) / wf).clamp(0.0, 1.0);
            let y_max = ((y as f32 + half) / hf).clamp(0.0, 1.0);

            let label = label_for_id(labels, class_id as u32);

            detections.push(Detection {
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

    sort_desc_and_cap(&mut detections, opts.max_detections);
    Ok(detections)
}

// ---------------------------------------------------------------------------
// Classification postprocessor
// ---------------------------------------------------------------------------

/// Compute softmax over logits and return top-k classifications.
///
/// `logits` shape: `[1, num_classes]` or `[batch, num_classes]`.
/// Only the first batch element is processed.
pub fn softmax(
    logits: &ArrayView2<f32>,
    labels: &[String],
    opts: &ClassifyOpts,
) -> Vec<Classification> {
    try_softmax(logits, labels, opts).unwrap_or_default()
}

/// Fallible softmax postprocessor used by runtime inference paths.
pub fn try_softmax(
    logits: &ArrayView2<f32>,
    labels: &[String],
    opts: &ClassifyOpts,
) -> Result<Vec<Classification>> {
    if logits.nrows() == 0 || logits.ncols() == 0 {
        return Err(SparrowEngineError::Ort(format!(
            "softmax expects non-empty [N, C] logits, got {:?}",
            logits.shape()
        )));
    }
    let top_k = opts.top_k.unwrap_or(1) as usize;
    if top_k == 0 {
        return Err(SparrowEngineError::Ort(
            "softmax top_k must be >= 1".to_string(),
        ));
    }
    let row = logits.index_axis(Axis(0), 0);
    if !row.iter().all(|v| v.is_finite()) {
        return Err(SparrowEngineError::Ort(
            "softmax logits contain non-finite values".to_string(),
        ));
    }
    let num_classes = row.len();

    // Numerically stable softmax: subtract max before exp.
    let max_val = row.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    if !max_val.is_finite() {
        return Err(SparrowEngineError::Ort(
            "softmax max logit is non-finite".to_string(),
        ));
    }
    let exps: Vec<f32> = row.iter().map(|&v| (v - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if !sum.is_finite() || sum <= 0.0 {
        return Err(SparrowEngineError::Ort(format!(
            "softmax denominator must be finite and positive, got {sum}"
        )));
    }

    // Build (index, probability) pairs.
    let mut scored: Vec<(usize, f32)> = exps
        .iter()
        .enumerate()
        .map(|(i, &e)| (i, e / sum))
        .collect();

    // Sort descending by probability.
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Take top_k, capped by actual class count.
    let k = top_k.min(num_classes);

    Ok(scored[..k]
        .iter()
        .map(|&(idx, prob)| Classification {
            label: label_for_id(labels, idx as u32),
            label_id: idx as u32,
            confidence: prob,
        })
        .collect())
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Undo letterbox transform and normalize bbox to [0,1] relative to original image.
///
/// Input coords are in model-input pixel space (with letterbox padding).
/// Output coords are normalized [0,1] relative to the original image dimensions.
/// Assumes callers use letterbox or resize-with-padding preprocessing.
/// Do NOT use with `resize_direct` metadata (scale=1.0, pad=0.0 are dummy values).
fn denormalize_and_normalize(x1: f32, y1: f32, x2: f32, y2: f32, meta: &PreprocessMeta) -> BBox {
    // Step 1: Remove letterbox padding.
    let x1 = x1 - meta.pad_x;
    let y1 = y1 - meta.pad_y;
    let x2 = x2 - meta.pad_x;
    let y2 = y2 - meta.pad_y;

    // Step 2: Undo the scale to get original-pixel coordinates.
    let x1 = x1 / meta.scale;
    let y1 = y1 / meta.scale;
    let x2 = x2 / meta.scale;
    let y2 = y2 / meta.scale;

    // Step 3: Normalize to [0,1] by dividing by original dimensions.
    let ow = meta.original_width as f32;
    let oh = meta.original_height as f32;

    BBox {
        x_min: (x1 / ow).clamp(0.0, 1.0),
        y_min: (y1 / oh).clamp(0.0, 1.0),
        x_max: (x2 / ow).clamp(0.0, 1.0),
        y_max: (y2 / oh).clamp(0.0, 1.0),
    }
}

/// Find the argmax and max value in a slice of an ndarray row.
fn argmax_slice(row: &ndarray::ArrayView1<f32>, start: usize, end: usize) -> (usize, f32) {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for i in start..end {
        let v = row[i];
        if v > best_val {
            best_val = v;
            best_idx = i - start;
        }
    }
    (best_idx, best_val)
}

/// Look up label by id, falling back to "unknown_<id>" if out of range.
pub fn label_for_id(labels: &[String], id: u32) -> String {
    labels
        .get(id as usize)
        .cloned()
        .unwrap_or_else(|| format!("unknown_{}", id))
}

/// Truncate detections to max_detections if set.
pub fn apply_max_detections(detections: &mut Vec<Detection>, max: Option<u32>) {
    if let Some(cap) = max {
        detections.truncate(cap as usize);
    }
}

/// Check if position (y, x) is a local maximum in the 8-connected neighborhood.
///
/// Plateau tie-breaking: for neighbors to the south and east (dy > 0, or dy == 0 && dx > 0),
/// we require strict `>` (current pixel must be strictly greater than those neighbors).
/// For north/west neighbors, equal values are allowed. This ensures that in a plateau of
/// equal values, only the bottom-right pixel is reported as the peak.
pub fn is_local_maximum(map: &ArrayView4<f32>, y: usize, x: usize, h: usize, w: usize) -> bool {
    let shape = map.shape();
    if shape[0] == 0
        || shape[1] == 0
        || shape[2] < h
        || shape[3] < w
        || h == 0
        || w == 0
        || y >= h
        || x >= w
    {
        return false;
    }
    let val = map[[0, 0, y, x]];

    for dy in -1i32..=1 {
        for dx in -1i32..=1 {
            if dy == 0 && dx == 0 {
                continue;
            }
            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                let neighbor = map[[0, 0, ny as usize, nx as usize]];
                // South/east neighbors: current must be strictly greater (>= means fail).
                // North/west neighbors: only fail if neighbor is strictly greater.
                let is_south_or_east = dy > 0 || (dy == 0 && dx > 0);
                if is_south_or_east {
                    if neighbor >= val {
                        return false;
                    }
                } else if neighbor > val {
                    return false;
                }
            }
        }
    }
    true
}

/// Compute OWL-T style adaptive threshold for single-output heatmap models.
///
/// `threshold = max(peak_threshold, global_max * peak_threshold, 0.1)`
///
/// This prevents noisy low-activation heatmaps from producing false detections:
/// - `global_max * peak_threshold` scales the threshold relative to the strongest activation
/// - Floor of 0.1 prevents near-zero thresholds when the model produces very low activations
///
/// **Note on [0,1] heatmaps (sigmoid-bounded models like OWL-T):** When the
/// model's heatmap values are in [0,1], `global_max <= 1.0`, so
/// `global_max * peak_threshold <= peak_threshold`. The adaptive term never
/// exceeds the base threshold, and the formula simplifies to
/// `max(peak_threshold, 0.1)`. For OWL-T with `peak_threshold = 0.2`, the
/// result is always 0.2 — the adaptive behavior is effectively a no-op.
/// The adaptive scaling only activates for models that output unbounded
/// heatmaps (raw logits > 1.0). The no-op outcome on sigmoid-bounded
/// outputs is the intended behavior — a fixed peak threshold is the
/// correct discrimination rule when the score range is already
/// constrained to [0,1].
pub fn owl_adaptive_threshold(peak_threshold: f32, global_max: f32) -> f32 {
    let adaptive_t = global_max * peak_threshold;
    peak_threshold.max(adaptive_t).max(0.1)
}

/// Compute adaptive threshold as the mean of a 5x5 local region around (y, x).
fn adaptive_threshold(
    map: &ArrayView4<f32>,
    y: usize,
    x: usize,
    h: usize,
    w: usize,
    base_threshold: f32,
) -> f32 {
    let radius: i32 = 2; // 5x5 window
    let mut sum = 0.0f32;
    let mut count = 0u32;

    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let ny = y as i32 + dy;
            let nx = x as i32 + dx;
            if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32 {
                sum += map[[0, 0, ny as usize, nx as usize]];
                count += 1;
            }
        }
    }

    let local_mean = sum / count as f32;
    // Adaptive threshold: use whichever is higher — base threshold or local mean.
    base_threshold.max(local_mean)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array4};

    fn test_labels() -> Vec<String> {
        vec!["animal".into(), "person".into(), "vehicle".into()]
    }

    fn identity_meta(w: u32, h: u32) -> PreprocessMeta {
        // No padding, no scale — coordinates already in original pixel space.
        PreprocessMeta {
            original_width: w,
            original_height: h,
            scale: 1.0,
            pad_x: 0.0,
            pad_y: 0.0,
        }
    }

    #[test]
    fn try_postprocessors_reject_malformed_shapes() {
        let opts = DetectOpts::default();
        let meta = identity_meta(100, 100);
        let yolo_bad = ndarray::Array2::<f32>::zeros((1, 5));
        assert!(try_yolo_e2e(&yolo_bad.view(), &test_labels(), &opts, &meta, 0.5).is_err());

        let md_bad = ndarray::Array2::<f32>::zeros((1, 5));
        assert!(try_megadet_v5a(&md_bad.view(), &test_labels(), &opts, &meta, 0.5, 0.45).is_err());

        let cls_opts = ClassifyOpts { top_k: Some(1) };
        let logits_bad = ndarray::Array2::<f32>::zeros((0, 3));
        assert!(try_softmax(&logits_bad.view(), &test_labels(), &cls_opts).is_err());

        let loc = Array4::<f32>::zeros((0, 1, 1, 1));
        let cls = Array4::<f32>::zeros((1, 1, 1, 1));
        let cfg = HeatmapConfig {
            peak_threshold: 0.5,
            adaptive: false,
            point_to_box_half_size: 2,
        };
        assert!(try_heatmap_peaks(&loc.view(), &cls.view(), &test_labels(), &opts, &cfg).is_err());
    }

    #[test]
    fn validate_heatmap_maps_rejects_ignored_batches_and_channels() {
        let loc_extra_batch = Array4::<f32>::zeros((2, 1, 2, 2));
        assert!(validate_heatmap_maps(&loc_extra_batch.view(), None, "test").is_err());

        let loc_extra_channel = Array4::<f32>::zeros((1, 2, 2, 2));
        assert!(validate_heatmap_maps(&loc_extra_channel.view(), None, "test").is_err());

        let loc = Array4::<f32>::zeros((1, 1, 2, 2));
        let cls_extra_batch = Array4::<f32>::zeros((2, 1, 2, 2));
        assert!(validate_heatmap_maps(&loc.view(), Some(&cls_extra_batch.view()), "test").is_err());

        let cls_empty_class = Array4::<f32>::zeros((1, 0, 2, 2));
        assert!(validate_heatmap_maps(&loc.view(), Some(&cls_empty_class.view()), "test").is_err());
    }

    #[test]
    fn test_yolo_e2e_filters_by_threshold() {
        // Two detections: one above threshold, one below.
        let data = array![
            [10.0, 20.0, 90.0, 80.0, 0.9, 0.0],
            [10.0, 20.0, 90.0, 80.0, 0.1, 1.0],
        ];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();

        let dets = yolo_e2e(&data.view(), &test_labels(), &opts, &meta, 0.5);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].label, "animal");
        assert!((dets[0].confidence - 0.9).abs() < 1e-5);
    }

    #[test]
    fn test_yolo_e2e_normalizes_coords() {
        let data = array![[50.0, 25.0, 150.0, 75.0, 0.8, 0.0]];
        let meta = identity_meta(200, 100);
        let opts = DetectOpts::default();

        let dets = yolo_e2e(&data.view(), &test_labels(), &opts, &meta, 0.1);
        assert_eq!(dets.len(), 1);
        assert!((dets[0].bbox.x_min - 0.25).abs() < 1e-5);
        assert!((dets[0].bbox.y_min - 0.25).abs() < 1e-5);
        assert!((dets[0].bbox.x_max - 0.75).abs() < 1e-5);
        assert!((dets[0].bbox.y_max - 0.75).abs() < 1e-5);
    }

    #[test]
    fn test_yolo_e2e_max_detections() {
        let data = array![
            [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],
            [0.0, 0.0, 10.0, 10.0, 0.8, 1.0],
            [0.0, 0.0, 10.0, 10.0, 0.7, 2.0],
        ];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts {
            max_detections: Some(2),
            ..Default::default()
        };

        let dets = yolo_e2e(&data.view(), &test_labels(), &opts, &meta, 0.1);
        assert_eq!(dets.len(), 2);
        // Highest confidence first.
        assert!((dets[0].confidence - 0.9).abs() < 1e-5);
        assert!((dets[1].confidence - 0.8).abs() < 1e-5);
    }

    #[test]
    fn test_yolo_e2e_letterbox_undo() {
        // Simulate letterbox: scale=0.5, pad_x=10, pad_y=0 on a 640x640 input.
        let meta = PreprocessMeta {
            original_width: 1280,
            original_height: 1280,
            scale: 0.5,
            pad_x: 10.0,
            pad_y: 0.0,
        };
        // A box at (10, 0, 330, 640) in model space.
        // After undo: ((10-10)/0.5, (0-0)/0.5, (330-10)/0.5, (640-0)/0.5) = (0, 0, 640, 1280)
        // Normalized: (0/1280, 0/1280, 640/1280, 1280/1280) = (0, 0, 0.5, 1.0)
        let data = array![[10.0, 0.0, 330.0, 640.0, 0.95, 0.0]];
        let opts = DetectOpts::default();

        let dets = yolo_e2e(&data.view(), &test_labels(), &opts, &meta, 0.1);
        assert_eq!(dets.len(), 1);
        assert!((dets[0].bbox.x_min - 0.0).abs() < 1e-4);
        assert!((dets[0].bbox.y_min - 0.0).abs() < 1e-4);
        assert!((dets[0].bbox.x_max - 0.5).abs() < 1e-4);
        assert!((dets[0].bbox.y_max - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_try_yolo_e2e_rejects_scores_outside_unit_interval() {
        let data = array![[10.0, 20.0, 90.0, 80.0, 1.2, 0.0]];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();
        let err = try_yolo_e2e(&data.view(), &test_labels(), &opts, &meta, 0.5)
            .expect_err("out-of-range scores must fail the whole batch");
        assert!(
            matches!(err, SparrowEngineError::Ort(msg) if msg.contains("scores outside [0.0, 1.0]"))
        );
    }

    #[test]
    fn test_try_yolo_e2e_silently_skips_low_confidence_degenerate_normalized_boxes() {
        // YOLOv10e exports many low-confidence TopK candidates whose boxes
        // can clamp to degenerate (x_min == x_max) shapes after letterbox-
        // undo + image-bound clamp. Below the confidence threshold they are
        // already discarded; their post-clamp geometry must not error the
        // whole batch. Coords (110, 10, 120, 20) over a 100×100 identity
        // letterbox normalize to (1.1, 0.1, 1.2, 0.2) → clamp → (1.0, 0.1,
        // 1.0, 0.2) — x_min == x_max → degenerate. Confidence 0.1 < threshold
        // 0.5, so this row is skipped.
        let data = array![[110.0, 10.0, 120.0, 20.0, 0.1, 0.0]];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();
        let dets = try_yolo_e2e(&data.view(), &test_labels(), &opts, &meta, 0.5)
            .expect("low-confidence degenerate boxes must be silently skipped");
        assert!(
            dets.is_empty(),
            "below-threshold rows must not emit detections"
        );
    }

    #[test]
    fn test_try_yolo_e2e_rejects_high_confidence_degenerate_normalized_boxes() {
        // Same coords as the silently-skips test above, but confidence 0.9 is
        // above threshold 0.5. A high-confidence degenerate box signals a
        // genuine model fault and must error the whole batch (REV-009 intent
        // preserved for rows that affect user-visible detections).
        let data = array![[110.0, 10.0, 120.0, 20.0, 0.9, 0.0]];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();
        let err = try_yolo_e2e(&data.view(), &test_labels(), &opts, &meta, 0.5)
            .expect_err("high-confidence degenerate normalized boxes must fail the whole batch");
        assert!(
            matches!(err, SparrowEngineError::Ort(msg) if msg.contains("degenerate normalized boxes"))
        );
    }

    #[test]
    fn test_megadet_v5a_confidence_product() {
        // objectness=0.8, class_scores=[0.5, 0.9, 0.1] -> conf = 0.8*0.9 = 0.72, class=1
        // Coordinates (10, 10, 90, 90) are interpreted as (cx, cy, w, h) per the
        // YOLOv5 raw-output convention; bbox sanity is asserted in the new
        // cxcywh test below — here we only assert the confidence-product +
        // label-argmax invariants.
        let data = array![[10.0, 10.0, 90.0, 90.0, 0.8, 0.5, 0.9, 0.1]];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();

        let dets = megadet_v5a(&data.view(), &test_labels(), &opts, &meta, 0.1, 0.45);
        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].label, "person");
        assert_eq!(dets[0].label_id, 1);
        assert!((dets[0].confidence - 0.72).abs() < 1e-5);
    }

    #[test]
    fn test_megadet_v5a_cxcywh_to_xyxy_and_nms() {
        // Two near-duplicate anchors at cx=50, cy=50, w=20, h=20 (same class)
        // — NMS should collapse to one detection. Their xyxy in 100×100 input
        // is (40, 40, 60, 60); after identity letterbox + /100 normalization,
        // the bbox should be (0.4, 0.4, 0.6, 0.6).
        // Third row is a different class (class 2 = "vehicle"); must SURVIVE.
        let data = array![
            [50.0, 50.0, 20.0, 20.0, 0.9, 0.1, 0.9, 0.1],
            [51.0, 51.0, 20.0, 20.0, 0.9, 0.1, 0.85, 0.1], // overlaps row 0 (same class 1)
            [50.0, 50.0, 20.0, 20.0, 0.9, 0.1, 0.1, 0.9],  // different class (2)
        ];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();
        let dets = megadet_v5a(&data.view(), &test_labels(), &opts, &meta, 0.05, 0.45);

        // NMS keeps the highest-confidence box per class.
        assert_eq!(dets.len(), 2, "NMS should collapse same-class overlap");
        // Highest-conf detection is class 1 (person) with confidence 0.81.
        assert_eq!(dets[0].label_id, 1);
        assert!((dets[0].confidence - 0.81).abs() < 1e-5);
        // bbox should be (cx-w/2, cy-h/2, cx+w/2, cy+h/2) / 100 = (0.4, 0.4, 0.6, 0.6).
        assert!((dets[0].bbox.x_min - 0.4).abs() < 1e-4);
        assert!((dets[0].bbox.y_min - 0.4).abs() < 1e-4);
        assert!((dets[0].bbox.x_max - 0.6).abs() < 1e-4);
        assert!((dets[0].bbox.y_max - 0.6).abs() < 1e-4);
        // Second detection is the surviving class-2 box.
        assert_eq!(dets[1].label_id, 2);
    }

    #[test]
    fn test_megadet_v5a_rejects_scores_outside_unit_interval() {
        let data = array![
            [50.0, 50.0, 20.0, 20.0, 1.2, 0.1, 0.9, 0.1],
            [50.0, 50.0, 20.0, 20.0, 0.9, 0.1, 0.9, 0.1],
        ];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();
        let err = try_megadet_v5a(&data.view(), &test_labels(), &opts, &meta, 0.05, 0.45)
            .expect_err("out-of-range scores must fail the whole batch");
        assert!(
            matches!(err, SparrowEngineError::Ort(msg) if msg.contains("scores outside [0.0, 1.0]"))
        );
    }

    #[test]
    fn test_megadet_v5a_rejects_nonpositive_box_geometry() {
        let data = array![[50.0, 50.0, -20.0, 20.0, 0.9, 0.1, 0.9, 0.1]];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();
        let err = try_megadet_v5a(&data.view(), &test_labels(), &opts, &meta, 0.05, 0.45)
            .expect_err("non-positive box size must fail the whole batch");
        assert!(
            matches!(err, SparrowEngineError::Ort(msg) if msg.contains("non-positive box size"))
        );
    }

    #[test]
    fn test_megadet_v5a_rejects_low_confidence_nonpositive_box_geometry() {
        let data = array![[50.0, 50.0, -20.0, 20.0, 0.1, 0.1, 0.2, 0.1]];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();
        let err = try_megadet_v5a(&data.view(), &test_labels(), &opts, &meta, 0.05, 0.45)
            .expect_err("low-confidence malformed boxes must still fail the whole batch");
        assert!(
            matches!(err, SparrowEngineError::Ort(msg) if msg.contains("non-positive box size"))
        );
    }

    #[test]
    fn test_megadet_v5a_silently_skips_low_confidence_degenerate_normalized_box() {
        // Parallel to test_try_yolo_e2e_silently_skips_low_confidence_*:
        // (cx, cy, w, h) = (-10, 50, 5, 20) → xyxy (-12.5, 40, -7.5, 60). On
        // an identity 100×100 letterbox, /100 → (-0.125, 0.4, -0.075, 0.6),
        // clamp → (0.0, 0.4, 0.0, 0.6) → x_min == x_max → degenerate.
        // Confidence = 0.1 * 0.2 = 0.02 < threshold 0.05, so this row is
        // skipped before bbox geometry validation. Raw-shape sanity (w > 0,
        // scores in [0,1]) still holds and is checked before the gate.
        let data = array![[-10.0, 50.0, 5.0, 20.0, 0.1, 0.1, 0.2, 0.1]];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();
        let dets = try_megadet_v5a(&data.view(), &test_labels(), &opts, &meta, 0.05, 0.45)
            .expect("low-confidence degenerate boxes must be silently skipped");
        assert!(
            dets.is_empty(),
            "below-threshold rows must not emit detections"
        );
    }

    #[test]
    fn test_megadet_v5a_rejects_high_confidence_degenerate_normalized_box() {
        // Same xyxy-degenerate geometry, but objectness × max_class = 0.9 *
        // 0.9 = 0.81 >= threshold 0.05. High-confidence degenerate signals a
        // model fault and must error the whole batch (REV-009 intent
        // preserved for above-threshold rows).
        let data = array![[-10.0, 50.0, 5.0, 20.0, 0.9, 0.1, 0.9, 0.1]];
        let meta = identity_meta(100, 100);
        let opts = DetectOpts::default();
        let err = try_megadet_v5a(&data.view(), &test_labels(), &opts, &meta, 0.05, 0.45)
            .expect_err("high-confidence degenerate normalized boxes must fail the whole batch");
        assert!(
            matches!(err, SparrowEngineError::Ort(msg) if msg.contains("degenerate normalized boxes"))
        );
    }

    #[test]
    fn test_softmax_basic() {
        let logits = array![[2.0, 1.0, 0.0]];
        let labels = test_labels();
        let opts = ClassifyOpts { top_k: Some(2) };

        let results = softmax(&logits.view(), &labels, &opts);
        assert_eq!(results.len(), 2);
        // Class 0 should have highest probability.
        assert_eq!(results[0].label, "animal");
        assert!(results[0].confidence > results[1].confidence);
        // Probabilities should sum close to 1 across all classes.
        let total: f32 = results.iter().map(|r| r.confidence).sum::<f32>();
        // Only top_k=2, so not quite 1.0, but each should be valid.
        assert!(total > 0.5);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large logits that would overflow naive exp().
        let logits = array![[1000.0, 999.0, 998.0]];
        let labels = test_labels();
        let opts = ClassifyOpts { top_k: Some(1) };

        let results = softmax(&logits.view(), &labels, &opts);
        assert_eq!(results.len(), 1);
        assert!(results[0].confidence.is_finite());
        assert_eq!(results[0].label, "animal");
    }

    #[test]
    fn test_softmax_default_top_k() {
        let logits = array![[1.0, 2.0, 3.0]];
        let labels = test_labels();
        let opts = ClassifyOpts::default(); // top_k = None -> 1

        let results = softmax(&logits.view(), &labels, &opts);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].label, "vehicle"); // class 2 has highest logit
    }

    #[test]
    fn test_heatmap_peaks_basic() {
        // 5x5 location map with a single peak at (2, 2).
        let mut loc_data = Array4::<f32>::zeros((1, 1, 5, 5));
        loc_data[[0, 0, 2, 2]] = 0.9;
        loc_data[[0, 0, 2, 1]] = 0.3;
        loc_data[[0, 0, 1, 2]] = 0.3;

        // Class map: 2 classes, class 1 dominant at (2, 2).
        let mut cls_data = Array4::<f32>::zeros((1, 2, 5, 5));
        cls_data[[0, 0, 2, 2]] = 0.2;
        cls_data[[0, 1, 2, 2]] = 0.8;

        let labels = vec!["cat".into(), "dog".into()];
        let opts = DetectOpts::default();
        let config = HeatmapConfig {
            peak_threshold: 0.1,
            adaptive: false,
            point_to_box_half_size: 1,
        };

        let dets = heatmap_peaks(&loc_data.view(), &cls_data.view(), &labels, &opts, &config);

        assert_eq!(dets.len(), 1);
        assert_eq!(dets[0].label, "dog");
        assert!((dets[0].confidence - 0.72).abs() < 1e-5); // 0.9 * 0.8
    }

    #[test]
    fn test_label_for_id_fallback() {
        let labels = vec!["a".into(), "b".into()];
        assert_eq!(label_for_id(&labels, 0), "a");
        assert_eq!(label_for_id(&labels, 5), "unknown_5");
    }

    #[test]
    fn test_denormalize_clamps() {
        let meta = identity_meta(100, 100);
        // Coordinates slightly outside image bounds.
        let bbox = denormalize_and_normalize(-5.0, -3.0, 105.0, 103.0, &meta);
        assert_eq!(bbox.x_min, 0.0);
        assert_eq!(bbox.y_min, 0.0);
        assert_eq!(bbox.x_max, 1.0);
        assert_eq!(bbox.y_max, 1.0);
    }

    #[test]
    fn test_softmax_uniform_distribution() {
        // All equal logits should produce uniform probabilities.
        let logits = array![[1.0, 1.0, 1.0]];
        let labels = test_labels();
        let opts = ClassifyOpts { top_k: Some(3) };
        let results = softmax(&logits.view(), &labels, &opts);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(
                (r.confidence - 1.0 / 3.0).abs() < 1e-5,
                "Expected ~0.333, got {}",
                r.confidence
            );
        }
    }

    // -----------------------------------------------------------------------
    // OWL-T single-output heatmap tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_owl_adaptive_threshold_basic() {
        // global_max = 1.0, peak_threshold = 0.2 → max(0.2, 0.2, 0.1) = 0.2
        assert!((owl_adaptive_threshold(0.2, 1.0) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_owl_adaptive_threshold_scales_with_max() {
        // global_max = 5.0, peak_threshold = 0.2 → max(0.2, 1.0, 0.1) = 1.0
        assert!((owl_adaptive_threshold(0.2, 5.0) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_owl_adaptive_threshold_floor() {
        // global_max = 0.01, peak_threshold = 0.02 → max(0.02, 0.0002, 0.1) = 0.1
        // The 0.1 floor prevents near-zero thresholds on quiet heatmaps.
        assert!((owl_adaptive_threshold(0.02, 0.01) - 0.1).abs() < 1e-6);
    }

    #[test]
    fn test_owl_adaptive_threshold_zero_max() {
        // global_max = 0.0 → max(0.2, 0.0, 0.1) = 0.2
        assert!((owl_adaptive_threshold(0.2, 0.0) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_owl_adaptive_threshold_negative_max() {
        // global_max = -1.0 → max(0.2, -0.2, 0.1) = 0.2
        assert!((owl_adaptive_threshold(0.2, -1.0) - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_single_output_heatmap_peak_confidence() {
        // Simulate single-output model: confidence = heatmap value directly (no cls multiplication).
        // In detect_tiled, single-output gives: (class_id=0, confidence=val).
        // Verify that is_local_maximum + threshold check works for this scenario.
        let mut loc = Array4::<f32>::zeros((1, 1, 5, 5));
        loc[[0, 0, 2, 2]] = 0.8; // Peak
        loc[[0, 0, 2, 1]] = 0.3; // Neighbor
        loc[[0, 0, 1, 2]] = 0.3; // Neighbor

        // The peak at (2,2) should be a local maximum
        assert!(is_local_maximum(&loc.view(), 2, 2, 5, 5));
        // For single-output: confidence = val = 0.8 (no cls multiplication)
        let val = loc[[0, 0, 2, 2]];
        assert!((val - 0.8).abs() < 1e-6);
        // With OWL-T adaptive threshold: max(0.2, 0.8*0.2, 0.1) = 0.2
        let threshold = owl_adaptive_threshold(0.2, 0.8);
        assert!((threshold - 0.2).abs() < 1e-6);
        assert!(val >= threshold, "Peak should pass adaptive threshold");
    }

    #[test]
    fn test_single_output_low_activation_filtered() {
        // All heatmap values very low — OWL-T adaptive floor should reject them.
        let mut loc = Array4::<f32>::zeros((1, 1, 5, 5));
        loc[[0, 0, 2, 2]] = 0.05; // Weak peak

        let global_max = 0.05;
        // Adaptive threshold: max(0.2, 0.05*0.2, 0.1) = 0.2
        let threshold = owl_adaptive_threshold(0.2, global_max);
        assert!((threshold - 0.2).abs() < 1e-6);
        // The weak peak (0.05) should be below threshold (0.2)
        assert!(
            loc[[0, 0, 2, 2]] < threshold,
            "Weak peak should be filtered"
        );
    }

    #[test]
    fn test_heatmap_boundary_peaks() {
        // Peaks at corners (0,0) and (4,4) of a 5x5 heatmap.
        let mut loc = Array4::<f32>::zeros((1, 1, 5, 5));
        loc[[0, 0, 0, 0]] = 0.9;
        loc[[0, 0, 4, 4]] = 0.85;

        let mut cls = Array4::<f32>::zeros((1, 1, 5, 5));
        cls[[0, 0, 0, 0]] = 0.8;
        cls[[0, 0, 4, 4]] = 0.7;

        let labels = vec!["cat".into()];
        let opts = DetectOpts::default();
        let config = HeatmapConfig {
            peak_threshold: 0.1,
            adaptive: false,
            point_to_box_half_size: 1,
        };

        let dets = heatmap_peaks(&loc.view(), &cls.view(), &labels, &opts, &config);

        assert_eq!(
            dets.len(),
            2,
            "Expected 2 boundary peaks, got {}",
            dets.len()
        );
        // Top-left peak: bbox clamped at (0,0)
        let top_left = dets
            .iter()
            .find(|d| d.bbox.x_min == 0.0 && d.bbox.y_min == 0.0);
        assert!(top_left.is_some(), "Expected peak at top-left corner");
        // Bottom-right peak: bbox extends to (x_max, y_max) near 1.0
        let bot_right = dets
            .iter()
            .find(|d| d.bbox.x_max == 1.0 && d.bbox.y_max == 1.0);
        assert!(bot_right.is_some(), "Expected peak at bottom-right corner");
    }

    #[test]
    fn test_plateau_single_detection() {
        // 5x5 heatmap with a 3x3 plateau of equal values in the center.
        // With tie-breaking, only the bottom-right pixel of the plateau should be the peak.
        let mut loc = Array4::<f32>::zeros((1, 1, 5, 5));
        for y in 1..4 {
            for x in 1..4 {
                loc[[0, 0, y, x]] = 0.9;
            }
        }

        let mut cls = Array4::<f32>::zeros((1, 1, 5, 5));
        for y in 0..5 {
            for x in 0..5 {
                cls[[0, 0, y, x]] = 0.8;
            }
        }

        let labels = vec!["cat".into()];
        let opts = DetectOpts::default();
        let config = HeatmapConfig {
            peak_threshold: 0.1,
            adaptive: false,
            point_to_box_half_size: 1,
        };

        let dets = heatmap_peaks(&loc.view(), &cls.view(), &labels, &opts, &config);

        assert_eq!(
            dets.len(),
            1,
            "Plateau should produce exactly 1 detection, got {}",
            dets.len()
        );
    }
}

#[cfg(test)]
mod phase_a_r1_postprocess {
    use super::*;
    use ndarray::{array, Array4};
    use sparrow_engine_types::{BBox, Detection};

    fn det(conf: f32) -> Detection {
        Detection {
            bbox: BBox {
                x_min: 0.0,
                y_min: 0.0,
                x_max: 0.5,
                y_max: 0.5,
            },
            label: "x".into(),
            label_id: 0,
            confidence: conf,
        }
    }

    /// NaN confidence in `sort_desc_and_cap` must not panic. The sort uses
    /// `partial_cmp(...).unwrap_or(Ordering::Equal)` (postprocess.rs:17–19)
    /// which absorbs NaN by treating it as equal to its peer. Pin: no panic,
    /// non-NaN values are still sorted descending.
    #[test]
    fn sort_desc_and_cap_with_nan_does_not_panic() {
        let mut v = vec![det(f32::NAN), det(0.5), det(0.9)];
        sort_desc_and_cap(&mut v, None);
        // NaN may end up anywhere relative to non-NaNs (Ordering::Equal), but
        // the two non-NaNs must be in correct descending order whenever both
        // appear adjacent. Filter out NaN and check ordering.
        let non_nan: Vec<f32> = v
            .iter()
            .filter(|d| !d.confidence.is_nan())
            .map(|d| d.confidence)
            .collect();
        assert_eq!(non_nan.len(), 2, "two non-NaN values must remain");
        assert!(
            non_nan[0] >= non_nan[1],
            "non-NaN entries must be sorted descending: {non_nan:?}"
        );
    }

    /// `cap = Some(0)` → empty Vec (truncate to length 0). The
    /// `apply_max_detections` line `detections.truncate(cap as usize)` becomes
    /// `truncate(0)` which is a documented Vec method that empties the vector.
    #[test]
    fn sort_desc_and_cap_zero_cap_yields_empty() {
        let mut v = vec![det(0.9), det(0.8), det(0.7)];
        sort_desc_and_cap(&mut v, Some(0));
        assert!(v.is_empty(), "cap=0 must truncate to zero elements");
    }

    /// `cap = None` → no truncation. Pins the early-return branch in
    /// `apply_max_detections` (line 338: `if let Some(cap) = max`).
    #[test]
    fn sort_desc_and_cap_none_cap_keeps_all() {
        let mut v = vec![det(0.5), det(0.9), det(0.7)];
        sort_desc_and_cap(&mut v, None);
        assert_eq!(v.len(), 3, "cap=None must preserve all detections");
        // Order check — descending.
        assert!((v[0].confidence - 0.9).abs() < 1e-6);
        assert!((v[1].confidence - 0.7).abs() < 1e-6);
        assert!((v[2].confidence - 0.5).abs() < 1e-6);
    }

    /// `apply_max_detections` with cap > current length is a no-op (Vec::truncate
    /// silently does nothing if `cap >= len`).
    #[test]
    fn apply_max_detections_cap_larger_than_len_is_noop() {
        let mut v = vec![det(0.9), det(0.8)];
        apply_max_detections(&mut v, Some(100));
        assert_eq!(v.len(), 2);
    }

    /// Uniform logits → uniform softmax probabilities ≈ 1/N. The existing
    /// `test_softmax_uniform_distribution` covers `top_k=3` over 3 classes; we
    /// expand to lock the EXACT 1/3 value (1e-6 tolerance) AND assert ALL
    /// probabilities sum to 1.0 with a hard `< 1e-5` gate; the existing test
    /// asserts only loosely (`> 0.5`).
    #[test]
    fn softmax_uniform_logits_produces_uniform_distribution() {
        let logits = array![[1.0, 1.0, 1.0]];
        let labels = vec!["a".into(), "b".into(), "c".into()];
        let opts = sparrow_engine_types::ClassifyOpts { top_k: Some(3) };
        let r = softmax(&logits.view(), &labels, &opts);
        assert_eq!(r.len(), 3);
        let total: f32 = r.iter().map(|c| c.confidence).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "softmax probabilities must sum to ~1.0, got {total}"
        );
        for c in &r {
            assert!(
                (c.confidence - 1.0 / 3.0).abs() < 1e-6,
                "uniform logits => uniform 1/3 probability, got {}",
                c.confidence
            );
        }
    }

    /// Numerical stability: very large logits must not overflow `exp`. The
    /// stable form subtracts the max before `exp` (postprocess.rs:252), so even
    /// `[1e6, 1e6 - 1, 1e6 - 2]` computes `exp(0)`, `exp(-1)`, `exp(-2)` —
    /// all finite. Existing `test_softmax_numerical_stability` covers logits at
    /// 1000.0; we widen by orders of magnitude.
    #[test]
    fn softmax_extreme_logits_stay_finite() {
        let logits = array![[1.0e6, 1.0e6 - 1.0, 1.0e6 - 2.0]];
        let labels = vec!["a".into(), "b".into(), "c".into()];
        let opts = sparrow_engine_types::ClassifyOpts { top_k: Some(3) };
        let r = softmax(&logits.view(), &labels, &opts);
        for c in &r {
            assert!(
                c.confidence.is_finite(),
                "softmax at logits=1e6 must stay finite, got {} for {}",
                c.confidence,
                c.label
            );
            assert!(c.confidence >= 0.0 && c.confidence <= 1.0);
        }
    }

    /// `is_local_maximum` corner-pixel boundary: peak at (0,0) sees only 3
    /// in-bounds neighbours (right, down, down-right). Out-of-bounds neighbours
    /// are skipped, so the corner remains a valid local max if it's strictly
    /// greater than each in-bounds neighbour. Pins the boundary check at
    /// `if ny >= 0 && ny < h as i32 && nx >= 0 && nx < w as i32`.
    #[test]
    fn is_local_maximum_corner_pixel_works() {
        let mut m = Array4::<f32>::zeros((1, 1, 5, 5));
        m[[0, 0, 0, 0]] = 0.9;
        m[[0, 0, 0, 1]] = 0.1;
        m[[0, 0, 1, 0]] = 0.1;
        m[[0, 0, 1, 1]] = 0.1;
        assert!(
            is_local_maximum(&m.view(), 0, 0, 5, 5),
            "corner pixel must be reportable as a local maximum"
        );
    }

    /// 2x2 plateau: all four pixels equal. Tie-breaking rule (south/east
    /// strict-greater, north/west non-strict-greater) should leave EXACTLY ONE
    /// peak — the bottom-right of the plateau — when the comparison fires.
    /// For (0,0) (top-left of plateau): the south neighbour at (1,0) ties, so
    /// the south-strict rule returns false. For (1,1) (bottom-right): all
    /// in-bounds neighbours are north/west and equal (allowed), so it returns
    /// true. This pins the plateau-tie behaviour for 2×2 plateaus, which the
    /// existing `test_plateau_single_detection` only covers for 3×3 plateaus.
    #[test]
    fn is_local_maximum_2x2_plateau_only_bottom_right_wins() {
        let mut m = Array4::<f32>::zeros((1, 1, 4, 4));
        for y in 1..3 {
            for x in 1..3 {
                m[[0, 0, y, x]] = 0.5;
            }
        }
        // (1,1) top-left of plateau — south neighbour (2,1) ties → fails strict.
        assert!(
            !is_local_maximum(&m.view(), 1, 1, 4, 4),
            "top-left of plateau must NOT be a peak (south neighbour ties)"
        );
        // (2,2) bottom-right of plateau — all in-bounds neighbours are
        // north/west (relative to it) and equal, which is allowed.
        assert!(
            is_local_maximum(&m.view(), 2, 2, 4, 4),
            "bottom-right of plateau must be the single peak"
        );
    }

    /// `owl_adaptive_threshold(t, m)` formula: `max(t, t*m, 0.1)`. Existing
    /// tests cover several specific numeric pairs. This pins the *symbolic*
    /// shape of the formula via three derived properties: the function never
    /// returns less than 0.1, never less than `peak_threshold`, and is
    /// monotonic non-decreasing in `global_max` once `m * t > t`.
    #[test]
    fn owl_adaptive_threshold_formula_invariants() {
        // Floor: never below 0.1.
        for (t, m) in [(0.0, 0.0), (0.0, 1.0), (0.05, 0.05)] {
            let r = owl_adaptive_threshold(t, m);
            assert!(
                r >= 0.1 - 1e-6,
                "threshold must be ≥ 0.1, got {r} for t={t}, m={m}"
            );
        }
        // Floor on peak_threshold: never below `t` itself.
        for t in [0.2, 0.5, 0.8, 1.0] {
            let r = owl_adaptive_threshold(t, 0.0);
            assert!(
                r >= t - 1e-6,
                "threshold must be ≥ peak_threshold {t}, got {r}"
            );
        }
        // Monotone in `global_max` for m > 1: increasing m must not decrease
        // the result.
        let r1 = owl_adaptive_threshold(0.2, 2.0);
        let r2 = owl_adaptive_threshold(0.2, 5.0);
        assert!(
            r2 >= r1 - 1e-6,
            "increasing global_max must not decrease threshold: {r1} → {r2}"
        );
    }
}
