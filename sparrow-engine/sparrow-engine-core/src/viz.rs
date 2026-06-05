//! Visualization: bounding box rendering and audio confidence heatmaps.
//!
//! Manual pixel ops on `image::RgbaImage`. No `imageproc` dependency.
//! All result types normalize to `BboxAnnotation` before rendering.

use std::collections::HashMap;

use image::{DynamicImage, RgbaImage};

use sparrow_engine_types::{
    AudioDetectResult, ClassifyResult, DetectResult, ModelType, PipelineResult,
};

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/// A single annotation to render.
#[derive(Debug, Clone)]
pub struct BboxAnnotation {
    /// Normalized [0,1] xyxy.
    pub bbox: [f32; 4],
    pub label: String,
    pub confidence: f32,
}

/// Rendering options for bounding box visualization.
///
/// The `model_type` field controls dispatch (Phase 3.5 S3, MT-9 fix):
/// - `ModelType::OverheadDetector` → draw a filled circle at the bbox centroid.
/// - Any other `ModelType` → draw a bounding rectangle.
///
/// Before S3 this was dispatched by comparing bbox pixel-size against
/// `point_radius * 2`, which false-negatived for overhead models on
/// high-resolution images. The explicit `model_type` removes that ambiguity.
#[derive(Debug, Clone)]
pub struct RenderOpts {
    pub line_width: u32,
    pub colors: HashMap<String, [u8; 3]>,
    pub min_confidence: f32,
    /// Fill alpha for bbox interior (0.0-1.0). None = no fill.
    pub fill_alpha: Option<f32>,
    /// For overhead dot models: confidence threshold below which dots are skipped.
    pub point_threshold: f32,
    /// Radius of filled circles for point / overhead-dot detections.
    pub point_radius: u32,
    /// Dispatch hint: `OverheadDetector` → dot path; anything else → bbox path.
    /// Default `Detector` preserves pre-S3 behaviour for callers that do not
    /// set this (MDv6, DeepFaune, and other bbox models are unaffected).
    pub model_type: ModelType,
    /// Render `"{label} {conf:.2}"` text above each bbox. Default `false`
    /// for clean overlays. Glyph rasterizer (`ab_glyph`) and bundled DejaVu
    /// Sans font are always linked — the toggle is runtime, not compile-time
    /// (post-Phase-3.7 lift; pre-lift this was the `viz-text` Cargo feature).
    /// Has no effect on the `OverheadDetector` dot path.
    pub show_labels: bool,
}

impl Default for RenderOpts {
    fn default() -> Self {
        Self {
            line_width: 2,
            colors: HashMap::new(),
            min_confidence: 0.0,
            fill_alpha: None,
            point_threshold: 0.0,
            point_radius: 4,
            model_type: ModelType::Detector,
            show_labels: false,
        }
    }
}

/// Options for audio confidence heatmap rendering.
#[derive(Debug, Clone)]
pub struct HeatmapOpts {
    pub emphasis: Emphasis,
    pub colormap: Colormap,
    pub blur_passes: u32,
    pub blur_radius: Option<u32>,
    pub alpha_min: u8,
    pub alpha_max: u8,
    pub alpha_gamma: f32,
    pub threshold: f32,
}

impl Default for HeatmapOpts {
    fn default() -> Self {
        Self {
            emphasis: Emphasis::Pow(1.0),
            colormap: Colormap::Inferno,
            blur_passes: 3,
            blur_radius: None,
            alpha_min: 0,
            alpha_max: 200,
            alpha_gamma: 1.0,
            threshold: 0.0,
        }
    }
}

/// Options governing which layers `render_audio_layers` emits and how
/// the heatmap layer is rendered.
#[derive(Debug, Clone)]
pub struct AudioLayersOpts {
    /// If true, layer 03 uses a smoothed inferno heatmap (Gaussian-style blur).
    /// If false, layer 03 mirrors layer 02 (raw per-slot pattern).
    pub smooth: bool,
    /// If true, emits an extra layer "02_segments_windows" between 02 and 03,
    /// showing each inference window staggered across lanes.
    pub show_windows: bool,
    /// Inference window length (seconds). Used for window-lane stagger count.
    pub window_s: f32,
    /// Inference stride length (seconds). Used for slot resolution + lane count.
    pub stride_s: f32,
}

impl Default for AudioLayersOpts {
    fn default() -> Self {
        Self {
            smooth: false,
            show_windows: false,
            window_s: 1.0,
            stride_s: 0.3,
        }
    }
}

const MAX_AUDIO_VIZ_SLOTS: usize = 1_000_000;
const MAX_AUDIO_WINDOW_LANES: u32 = 128;

fn bounded_window_lanes(window_s: f32, stride_s: f32) -> u32 {
    if !window_s.is_finite()
        || window_s <= 0.0
        || !stride_s.is_finite()
        || stride_s <= 0.0
    {
        return 1;
    }
    let lanes = (window_s / stride_s).ceil();
    if !lanes.is_finite() || lanes <= 1.0 {
        1
    } else {
        (lanes as u32).min(MAX_AUDIO_WINDOW_LANES)
    }
}

fn valid_time_span(start_time_s: f32, end_time_s: f32) -> bool {
    start_time_s.is_finite() && end_time_s.is_finite() && end_time_s > start_time_s
}

fn valid_audio_segment(seg: &sparrow_engine_types::AudioSegment) -> bool {
    valid_time_span(seg.start_time_s, seg.end_time_s) && seg.confidence.is_finite()
}

fn valid_audio_range(range: &sparrow_engine_types::AudioRange) -> bool {
    valid_time_span(range.start_time_s, range.end_time_s) && range.max_confidence.is_finite()
}

fn validate_heatmap_opts(opts: &HeatmapOpts) -> std::result::Result<(), String> {
    if !opts.threshold.is_finite() || !(0.0..=1.0).contains(&opts.threshold) {
        return Err(format!(
            "threshold must be finite and in [0,1], got {}",
            opts.threshold
        ));
    }
    if !opts.alpha_gamma.is_finite() || opts.alpha_gamma <= 0.0 {
        return Err(format!(
            "alpha_gamma must be finite and > 0, got {}",
            opts.alpha_gamma
        ));
    }
    match &opts.emphasis {
        Emphasis::Pow(p) if !p.is_finite() || *p <= 0.0 => Err(format!(
            "Pow emphasis exponent must be finite and > 0, got {p}"
        )),
        Emphasis::Sigmoid { midpoint, steepness } => {
            if !midpoint.is_finite() || !steepness.is_finite() || *steepness <= 0.0 {
                Err(format!(
                    "Sigmoid emphasis midpoint must be finite and steepness finite > 0, got midpoint={midpoint} steepness={steepness}"
                ))
            } else {
                Ok(())
            }
        }
        _ => Ok(()),
    }
}

/// Emphasis curve for heatmap values.
#[derive(Debug, Clone)]
pub enum Emphasis {
    Pow(f32),
    Sigmoid { midpoint: f32, steepness: f32 },
}

/// Available colormaps.
#[derive(Debug, Clone)]
pub enum Colormap {
    Inferno,
}

// ---------------------------------------------------------------------------
// Default colors
// ---------------------------------------------------------------------------

fn default_color(label: &str) -> [u8; 3] {
    match label.to_lowercase().as_str() {
        "animal" => [0, 255, 0],
        "person" | "human" => [255, 0, 0],
        "vehicle" | "car" => [0, 0, 255],
        _ => [255, 255, 0],
    }
}

fn resolve_color(label: &str, colors: &HashMap<String, [u8; 3]>) -> [u8; 3] {
    colors.get(label).copied().unwrap_or_else(|| default_color(label))
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

/// Convert DetectResult to annotations.
pub fn detections_to_annotations(result: &DetectResult) -> Vec<BboxAnnotation> {
    result
        .detections
        .iter()
        .map(|d| BboxAnnotation {
            bbox: [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max],
            label: d.label.clone(),
            confidence: d.confidence,
        })
        .collect()
}

/// Convert ClassifyResult to a single full-image annotation.
pub fn classifications_to_annotations(result: &ClassifyResult) -> Vec<BboxAnnotation> {
    result
        .classifications
        .first()
        .map(|c| BboxAnnotation {
            bbox: [0.0, 0.0, 1.0, 1.0],
            label: c.label.clone(),
            confidence: c.confidence,
        })
        .into_iter()
        .collect()
}

/// Convert PipelineResult to annotations (uses classification label if available).
pub fn pipeline_to_annotations(result: &PipelineResult) -> Vec<BboxAnnotation> {
    result
        .detections
        .iter()
        .map(|pd| {
            let label = if let Some(ref cls) = pd.classification {
                cls.label.clone()
            } else {
                pd.detection.label.clone()
            };
            BboxAnnotation {
                bbox: [
                    pd.detection.bbox.x_min,
                    pd.detection.bbox.y_min,
                    pd.detection.bbox.x_max,
                    pd.detection.bbox.y_max,
                ],
                label,
                confidence: pd.detection.confidence,
            }
        })
        .collect()
}

/// Convert AudioDetectResult segments to time-domain annotations.
pub fn audio_segments_to_annotations(result: &AudioDetectResult) -> Vec<BboxAnnotation> {
    let dur = if result.duration_s > 0.0 {
        result.duration_s
    } else {
        1.0
    };
    result
        .segments
        .iter()
        .map(|s| BboxAnnotation {
            bbox: [s.start_time_s / dur, 0.0, s.end_time_s / dur, 1.0],
            label: "detected".to_string(),
            confidence: s.confidence,
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Drawing primitives
// ---------------------------------------------------------------------------

fn draw_rect(img: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32, color: [u8; 3], width: u32) {
    let (w, h) = img.dimensions();
    let x0 = x0.min(w.saturating_sub(1));
    let x1 = x1.min(w.saturating_sub(1));
    let y0 = y0.min(h.saturating_sub(1));
    let y1 = y1.min(h.saturating_sub(1));

    for t in 0..width {
        // Top and bottom edges
        for x in x0..=x1 {
            if y0 + t < h {
                img.put_pixel(x, y0 + t, image::Rgba([color[0], color[1], color[2], 255]));
            }
            if y1 >= t && y1 - t < h {
                img.put_pixel(x, y1 - t, image::Rgba([color[0], color[1], color[2], 255]));
            }
        }
        // Left and right edges
        for y in y0..=y1 {
            if x0 + t < w {
                img.put_pixel(x0 + t, y, image::Rgba([color[0], color[1], color[2], 255]));
            }
            if x1 >= t && x1 - t < w {
                img.put_pixel(x1 - t, y, image::Rgba([color[0], color[1], color[2], 255]));
            }
        }
    }
}

fn draw_filled_circle(img: &mut RgbaImage, cx: i32, cy: i32, radius: u32, color: [u8; 3]) {
    let (w, h) = img.dimensions();
    let r = radius as i32;
    for dy in -r..=r {
        for dx in -r..=r {
            if dx * dx + dy * dy <= r * r {
                let px = cx + dx;
                let py = cy + dy;
                if px >= 0 && px < w as i32 && py >= 0 && py < h as i32 {
                    img.put_pixel(px as u32, py as u32, image::Rgba([color[0], color[1], color[2], 255]));
                }
            }
        }
    }
}

fn draw_filled_rect_alpha(img: &mut RgbaImage, x0: u32, y0: u32, x1: u32, y1: u32, color: [u8; 3], alpha: u8) {
    let (w, h) = img.dimensions();
    let x1 = x1.min(w.saturating_sub(1));
    let y1 = y1.min(h.saturating_sub(1));
    let a = alpha as f32 / 255.0;
    let inv_a = 1.0 - a;

    for y in y0..=y1 {
        for x in x0..=x1 {
            if x < w && y < h {
                let existing = img.get_pixel(x, y);
                let r = (color[0] as f32 * a + existing[0] as f32 * inv_a).round() as u8;
                let g = (color[1] as f32 * a + existing[1] as f32 * inv_a).round() as u8;
                let b = (color[2] as f32 * a + existing[2] as f32 * inv_a).round() as u8;
                img.put_pixel(x, y, image::Rgba([r, g, b, 255]));
            }
        }
    }
}

// ---------------------------------------------------------------------------
// render()
// ---------------------------------------------------------------------------

/// Render bounding box annotations onto an image.
pub fn render(image: &DynamicImage, annotations: &[BboxAnnotation], opts: &RenderOpts) -> DynamicImage {
    let mut canvas = image.to_rgba8();
    let (w, h) = canvas.dimensions();

    for ann in annotations {
        if ann.confidence < opts.min_confidence {
            continue;
        }

        // Skip annotations with non-finite bbox coordinates (NaN / ±Inf). Clamping
        // NaN through `.clamp()` would still yield NaN, and `f32 as u32` saturates
        // (+Inf → u32::MAX, NaN → 0), which otherwise produces off-screen point
        // draws or zero-area rects at origin.
        if !ann.bbox[0].is_finite() || !ann.bbox[1].is_finite()
            || !ann.bbox[2].is_finite() || !ann.bbox[3].is_finite()
        {
            continue;
        }

        // Clamp finite-but-out-of-range values to [0,1] before the f32→u32 cast.
        // Values like 1e30 survive the is_finite gate but overflow on multiply
        // (1e30 * w = +Inf → u32::MAX, then (px0+px1)/2 can overflow in point path).
        let bx0 = ann.bbox[0].clamp(0.0, 1.0);
        let by0 = ann.bbox[1].clamp(0.0, 1.0);
        let bx1 = ann.bbox[2].clamp(0.0, 1.0);
        let by1 = ann.bbox[3].clamp(0.0, 1.0);

        let color = resolve_color(&ann.label, &opts.colors);
        let px0 = (bx0 * w as f32).round() as u32;
        let py0 = (by0 * h as f32).round() as u32;
        let px1 = (bx1 * w as f32).round() as u32;
        let py1 = (by1 * h as f32).round() as u32;

        // Dispatch on ModelType, not bbox pixel size (Phase 3.5 S3, MT-9 fix).
        // Pre-S3 used `bbox_w <= point_radius*2 && bbox_h <= point_radius*2`,
        // which produced false-bbox renders for overhead models at high
        // resolution (a 10px-wide centroid is > point_radius*2 at 4K+).
        match opts.model_type {
            ModelType::OverheadDetector => {
                if ann.confidence >= opts.point_threshold {
                    // `px0 + px1` can reach 2 * w (up to ~8192 for 4K inputs) —
                    // well within u32 range, but use u64 midpoint for safety.
                    let cx = ((px0 as u64 + px1 as u64) / 2) as i32;
                    let cy = ((py0 as u64 + py1 as u64) / 2) as i32;
                    draw_filled_circle(&mut canvas, cx, cy, opts.point_radius, color);
                }
            }
            // Standard / classifier / audio model types all take the bbox path.
            // Audio and classifier paths already produce full-image or
            // time-extent rectangles that render correctly as bboxes.
            ModelType::Detector
            | ModelType::Classifier
            | ModelType::AudioDetector
            | ModelType::AudioClassifier => {
                if let Some(alpha) = opts.fill_alpha {
                    let a = (alpha * 255.0).clamp(0.0, 255.0) as u8;
                    draw_filled_rect_alpha(&mut canvas, px0, py0, px1, py1, color, a);
                }
                draw_rect(&mut canvas, px0, py0, px1, py1, color, opts.line_width);
                if opts.show_labels {
                    text::render_text_label(
                        &mut canvas,
                        px0 as i32,
                        py0 as i32,
                        &ann.label,
                        ann.confidence,
                        color,
                    );
                }
            }
        }
    }

    DynamicImage::ImageRgba8(canvas)
}

// ---------------------------------------------------------------------------
// Inferno colormap
// ---------------------------------------------------------------------------

fn inferno(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    // 5-point piecewise linear interpolation.
    let stops: [(f32, [f32; 3]); 5] = [
        (0.0, [0.0, 0.0, 4.0]),
        (0.25, [87.0, 16.0, 110.0]),
        (0.5, [188.0, 55.0, 84.0]),
        (0.75, [249.0, 142.0, 9.0]),
        (1.0, [252.0, 255.0, 164.0]),
    ];

    for i in 0..stops.len() - 1 {
        let (t0, c0) = stops[i];
        let (t1, c1) = stops[i + 1];
        if t >= t0 && t <= t1 {
            let f = (t - t0) / (t1 - t0);
            return [
                (c0[0] + (c1[0] - c0[0]) * f).round() as u8,
                (c0[1] + (c1[1] - c0[1]) * f).round() as u8,
                (c0[2] + (c1[2] - c0[2]) * f).round() as u8,
            ];
        }
    }

    [252, 255, 164]
}

// ---------------------------------------------------------------------------
// render_audio_heatmap()
// ---------------------------------------------------------------------------

/// Compose the audio-detection visualization layers from a spectrogram backdrop,
/// segment list, and optional merged ranges.
///
/// Returns a Vec of (layer_name, image) in render order. Layer names match the
/// CLI's `spe detect-audio --visualize` output filename stems:
///   - "01_spec"               — raw spectrogram, no overlays
///   - "02_segments"           — discrete per-slot confidence (no blur)
///   - "02_segments_windows"   — only if opts.show_windows
///   - "03_heatmap"            — smoothed inferno heatmap if opts.smooth, else
///     identical to 02_segments
///   - "04_full"               — 03_heatmap + cyan range bars, only if ranges Some
pub fn render_audio_layers(
    spec: &DynamicImage,
    segments: &[sparrow_engine_types::AudioSegment],
    ranges: Option<&[sparrow_engine_types::AudioRange]>,
    duration_s: f32,
    opts: &AudioLayersOpts,
) -> Vec<(&'static str, DynamicImage)> {
    let mut layers = Vec::with_capacity(
        3 + usize::from(opts.show_windows) + usize::from(ranges.is_some()),
    );

    layers.push(("01_spec", spec.clone()));

    let diag_segments = segments_to_overlap_mean_slots(segments, duration_s, opts.stride_s);
    let segments_opts = HeatmapOpts {
        blur_passes: 0,
        blur_radius: Some(0),
        alpha_max: 200,
        ..HeatmapOpts::default()
    };
    let segments_img = render_audio_heatmap(spec, &diag_segments, duration_s, &segments_opts);
    layers.push(("02_segments", segments_img.clone()));

    if opts.show_windows {
        let window_opts = WindowLanesOpts {
            n_lanes: bounded_window_lanes(opts.window_s, opts.stride_s),
            ..WindowLanesOpts::default()
        };
        let segments_windows_img =
            render_window_lanes(&segments_img, segments, duration_s, &window_opts);
        layers.push(("02_segments_windows", segments_windows_img));
    }

    let heatmap_img = if opts.smooth {
        let heatmap_opts = HeatmapOpts {
            blur_passes: 3,
            blur_radius: None,
            alpha_max: 200,
            ..HeatmapOpts::default()
        };
        render_audio_heatmap(spec, &diag_segments, duration_s, &heatmap_opts)
    } else {
        segments_img
    };
    layers.push(("03_heatmap", heatmap_img.clone()));

    if let Some(rs) = ranges {
        let full_img =
            render_range_overlay(&heatmap_img, rs, duration_s, &RangeOverlayOpts::default());
        layers.push(("04_full", full_img));
    }

    layers
}

/// Compute per-slot **overlap-mean** confidence at step-size (stride)
/// resolution. For a window=1.0 s / stride=0.3 s model, every 0.3 s slot
/// is contained in ~3-4 sliding-window segments. The mean over those
/// overlapping segments is a denser, more representative confidence at
/// step-size resolution than the single window starting at the slot.
///
/// Each output `AudioSegment` covers exactly one stride-width slot, so
/// when fed to `render_audio_heatmap` the per-pixel `max` aggregation is
/// effectively a no-op (each pixel covered by exactly one input segment).
/// The rendered heat array reads the mean confidence directly.
///
/// Slots with zero overlapping windows (gaps at the timeline tail beyond
/// the last full window) are dropped so the renderer skips them.
pub fn segments_to_overlap_mean_slots(
    segments: &[sparrow_engine_types::AudioSegment],
    duration_s: f32,
    step_s: f32,
) -> Vec<sparrow_engine_types::AudioSegment> {
    if segments.is_empty()
        || !duration_s.is_finite()
        || duration_s <= 0.0
        || !step_s.is_finite()
        || step_s <= 0.0
    {
        return vec![];
    }
    let n_slots_f = (duration_s / step_s).ceil();
    if !n_slots_f.is_finite() || n_slots_f <= 0.0 || n_slots_f > MAX_AUDIO_VIZ_SLOTS as f32 {
        return vec![];
    }
    let n_slots = n_slots_f as usize;
    let mut sums = vec![0.0f32; n_slots];
    let mut counts = vec![0u32; n_slots];
    for seg in segments {
        if !valid_audio_segment(seg) {
            continue;
        }
        let slot_lo = ((seg.start_time_s / step_s).floor() as usize).min(n_slots);
        let slot_hi = ((seg.end_time_s / step_s).ceil() as usize).min(n_slots);
        for slot in slot_lo..slot_hi {
            sums[slot] += seg.confidence;
            counts[slot] += 1;
        }
    }
    let mut result = Vec::with_capacity(n_slots);
    for slot in 0..n_slots {
        if counts[slot] == 0 {
            continue;
        }
        let mean = sums[slot] / counts[slot] as f32;
        result.push(sparrow_engine_types::AudioSegment {
            start_time_s: slot as f32 * step_s,
            end_time_s: ((slot + 1) as f32 * step_s).min(duration_s),
            confidence: mean,
            classes: Vec::new(),
        });
    }
    result
}

/// Render an audio confidence heatmap overlaid on a spectrogram image.
pub fn render_audio_heatmap(
    spectrogram: &DynamicImage,
    segments: &[sparrow_engine_types::AudioSegment],
    duration_s: f32,
    opts: &HeatmapOpts,
) -> DynamicImage {
    let base = spectrogram.to_rgba8();
    let (w, h) = base.dimensions();
    if w == 0 || h == 0 || !duration_s.is_finite() || duration_s <= 0.0 {
        return DynamicImage::ImageRgba8(base);
    }

    // Guard against misconfigured HeatmapOpts. alpha_min > alpha_max would invert
    // the per-pixel alpha ramp (stronger confidence → more transparent), which is
    // the opposite of the intended semantics. Loud in dev (debug_assert), safe
    // in release (no-op overlay = return the spectrogram unchanged).
    if opts.alpha_min > opts.alpha_max {
        debug_assert!(
            false,
            "HeatmapOpts: alpha_min ({}) > alpha_max ({}) inverts the alpha ramp",
            opts.alpha_min, opts.alpha_max
        );
        return DynamicImage::ImageRgba8(base);
    }
    if let Err(msg) = validate_heatmap_opts(opts) {
        debug_assert!(false, "HeatmapOpts: {msg}");
        return DynamicImage::ImageRgba8(base);
    }

    // Accumulate confidence into a 1D array (width of the image).
    let mut heat = vec![0.0f32; w as usize];
    for seg in segments {
        if !valid_audio_segment(seg) || seg.confidence < opts.threshold {
            continue;
        }
        let x_start = ((seg.start_time_s / duration_s) * w as f32).round() as usize;
        let x_start = x_start.min(w as usize);
        let x_end = ((seg.end_time_s / duration_s) * w as f32).round() as usize;
        let x_end = x_end.min(w as usize);
        if x_start >= x_end {
            continue;
        }
        for val in &mut heat[x_start..x_end] {
            *val = val.max(seg.confidence);
        }
    }

    // Apply emphasis.
    for v in &mut heat {
        *v = match &opts.emphasis {
            Emphasis::Pow(p) => v.powf(*p),
            Emphasis::Sigmoid { midpoint, steepness } => {
                1.0 / (1.0 + (-(* v - midpoint) * steepness).exp())
            }
        };
    }

    // Triple box blur for approximated Gaussian.
    let radius = opts.blur_radius.unwrap_or(w / 50);
    for _ in 0..opts.blur_passes {
        box_blur_1d(&mut heat, radius as usize);
    }

    // Heat values are already on the absolute [0, 1] scale (sigmoid output
    // from the audio detector). No max-normalization: per-image rescaling
    // would amplify silence-floor noise on quiet clips and is not needed
    // when the input distribution is already bounded.

    // Composite heatmap onto the base image.
    let mut canvas = base;
    for y in 0..h {
        for x in 0..w {
            let t = heat[x as usize];
            if t < 0.01 {
                continue;
            }
            let color = inferno(t);
            let alpha_range = opts.alpha_max as f32 - opts.alpha_min as f32;
            let alpha_f = (t.powf(opts.alpha_gamma) * alpha_range
                + opts.alpha_min as f32)
                / 255.0;
            let alpha_f = alpha_f.clamp(0.0, 1.0);
            let inv = 1.0 - alpha_f;
            let existing = canvas.get_pixel(x, y);
            let r = (color[0] as f32 * alpha_f + existing[0] as f32 * inv) as u8;
            let g = (color[1] as f32 * alpha_f + existing[1] as f32 * inv) as u8;
            let b = (color[2] as f32 * alpha_f + existing[2] as f32 * inv) as u8;
            canvas.put_pixel(x, y, image::Rgba([r, g, b, 255]));
        }
    }

    DynamicImage::ImageRgba8(canvas)
}

/// Render a real mel spectrogram of an audio file as a grayscale image.
///
/// Decodes the audio, computes the full-clip mel spectrogram (using the same
/// DSP path as `detect_audio` inference), and renders it as a grayscale image
/// suitable for use as a `render_audio_heatmap` backdrop.
///
/// Image dimensions: width = number of STFT frames, height = `config.n_mels`.
/// Y orientation: low frequencies at the bottom (image y = height-1), high
/// frequencies at the top (image y = 0) — standard spectrogram convention.
/// Pixel value: dB-scale mel energy, mapped from `[-config.top_db, 0]` to
/// `[0, 255]` (silence ≈ black, full-scale ≈ white).
///
/// Used by `spe detect-audio --visualize` to give the heatmap a real audio
/// backdrop instead of a synthetic gray gradient.
pub fn render_mel_spectrogram(
    audio_path: &std::path::Path,
    config: &crate::preprocess_audio::AudioPreprocessConfig,
) -> sparrow_engine_types::Result<DynamicImage> {
    use crate::preprocess_audio::{load_audio, mel_spectrogram, MelFilterbank};
    use sparrow_engine_types::AudioInput;

    let samples = load_audio(&AudioInput::FilePath(audio_path.to_path_buf()), config)?;
    let filterbank = MelFilterbank::new(config)?;
    let mel_tensor = mel_spectrogram(&samples.data, samples.orig_sample_rate, config, &filterbank)?;

    // Tensor shape: [1, 1, n_mels, n_frames]. Flatten to a slice for indexing.
    let shape = mel_tensor.shape();
    let n_mels = shape[2];
    let n_frames = shape[3];
    let mel = mel_tensor
        .as_slice()
        .ok_or_else(|| sparrow_engine_types::SparrowEngineError::AudioPreprocess(
            "mel tensor not contiguous".into(),
        ))?;

    let top_db = config.top_db.max(1e-3);
    let mut img = RgbaImage::new(n_frames as u32, n_mels as u32);
    for m in 0..n_mels {
        // Image y=0 is top; place low-freq mel rows (m=0) at the bottom.
        let image_y = (n_mels - 1 - m) as u32;
        for t in 0..n_frames {
            let v = mel[m * n_frames + t];
            // mel values are in dB scale clamped to [-top_db, 0]. Map to [0, 1]
            // then scale to 0..255. Use a small gamma boost so quiet detail
            // (e.g., bird calls in lower-energy bins) reads more clearly.
            let normalized = ((v + top_db) / top_db).clamp(0.0, 1.0);
            let gamma_boosted = normalized.powf(0.7);
            let gray = (gamma_boosted * 255.0).round() as u8;
            img.put_pixel(t as u32, image_y, image::Rgba([gray, gray, gray, 255]));
        }
    }

    Ok(DynamicImage::ImageRgba8(img))
}

/// Options for `render_range_overlay`. All in pixel units / 0–255 alpha.
#[derive(Debug, Clone)]
pub struct RangeOverlayOpts {
    /// RGB color of vertical bars and bottom band.
    pub bar_color: [u8; 3],
    /// Alpha of vertical bars (0–255). Higher = more opaque.
    pub bar_alpha: u8,
    /// Width of each vertical bar in pixels.
    pub bar_width_px: u32,
    /// Height of the bottom confidence band in pixels.
    pub band_height_px: u32,
}

impl Default for RangeOverlayOpts {
    fn default() -> Self {
        Self {
            bar_color: [0, 255, 255], // cyan — high contrast against inferno colormap
            bar_alpha: 220,
            bar_width_px: 2,
            band_height_px: 6,
        }
    }
}

/// Overlay merged-range markers on top of a rendered audio heatmap.
///
/// For each range, draws:
/// - Vertical bars at `start_time_s` and `end_time_s` (full image height)
/// - Horizontal band along the bottom between the two bars, with band alpha
///   scaled by `max_confidence`
///
/// Use this on top of `render_audio_heatmap` to verify visually that merged
/// ranges line up with the underlying per-window confidence heatmap.
pub fn render_range_overlay(
    heatmap: &DynamicImage,
    ranges: &[sparrow_engine_types::AudioRange],
    duration_s: f32,
    opts: &RangeOverlayOpts,
) -> DynamicImage {
    let mut canvas = heatmap.to_rgba8();
    let (w, h) = canvas.dimensions();
    if w == 0 || h == 0 || !duration_s.is_finite() || duration_s <= 0.0 || ranges.is_empty() {
        return DynamicImage::ImageRgba8(canvas);
    }

    let bar_color = opts.bar_color;
    let bar_alpha = opts.bar_alpha as f32 / 255.0;
    let band_h = opts.band_height_px.min(h);
    let band_y0 = h.saturating_sub(band_h);

    let time_to_edge_x = |t: f32| -> u32 {
        ((t / duration_s) * w as f32).round().clamp(0.0, w as f32) as u32
    };
    let time_to_bar_x = |t: f32| -> u32 { time_to_edge_x(t).min(w.saturating_sub(1)) };

    let blend_pixel = |canvas: &mut image::RgbaImage, x: u32, y: u32, alpha: f32| {
        if x >= w || y >= h {
            return;
        }
        let inv = 1.0 - alpha;
        let existing = canvas.get_pixel(x, y);
        let r = (bar_color[0] as f32 * alpha + existing[0] as f32 * inv) as u8;
        let g = (bar_color[1] as f32 * alpha + existing[1] as f32 * inv) as u8;
        let b = (bar_color[2] as f32 * alpha + existing[2] as f32 * inv) as u8;
        canvas.put_pixel(x, y, image::Rgba([r, g, b, 255]));
    };

    let draw_v_bar = |canvas: &mut image::RgbaImage, x_center: u32| {
        let half = opts.bar_width_px / 2;
        let x0 = x_center.saturating_sub(half);
        let x1 = (x_center + half + 1).min(w);
        for x in x0..x1 {
            for y in 0..h {
                blend_pixel(canvas, x, y, bar_alpha);
            }
        }
    };

    for range in ranges {
        if !valid_audio_range(range) {
            continue;
        }
        let x_start = time_to_edge_x(range.start_time_s);
        let x_end = time_to_edge_x(range.end_time_s);
        draw_v_bar(&mut canvas, time_to_bar_x(range.start_time_s));
        draw_v_bar(&mut canvas, time_to_bar_x(range.end_time_s));

        // Bottom band: alpha scales with max_confidence so high-confidence
        // ranges stand out and low-confidence ones fade.
        let band_alpha = (range.max_confidence.clamp(0.0, 1.0) * bar_alpha).clamp(0.0, 1.0);
        let xa = x_start.min(x_end);
        let xb = x_start.max(x_end);
        for x in xa..xb {
            for y in band_y0..h {
                blend_pixel(&mut canvas, x, y, band_alpha);
            }
        }
    }

    DynamicImage::ImageRgba8(canvas)
}

/// Options for `render_window_lanes`. Pixel units.
#[derive(Debug, Clone)]
pub struct WindowLanesOpts {
    /// Number of lanes — typically `ceil(window_duration_s / stride_s)` so
    /// adjacent overlapping windows render in different lanes.
    pub n_lanes: u32,
    /// Height of each window line in pixels.
    pub line_height_px: u32,
    /// Vertical gap between consecutive lanes.
    pub lane_gap_px: u32,
    /// Gap between the base image and the lanes band.
    pub gap_to_base_px: u32,
    /// RGB color of the gap row(s).
    pub gap_color: [u8; 3],
}

impl Default for WindowLanesOpts {
    fn default() -> Self {
        Self {
            n_lanes: 4,
            line_height_px: 2,
            lane_gap_px: 2,
            gap_to_base_px: 1,
            gap_color: [0, 0, 0],
        }
    }
}

/// Append a "window lanes" band below `base`. Each sliding-window segment is
/// rendered as a thin horizontal line spanning its `[start_time_s,
/// end_time_s]` x range, coloured by `inferno(seg.confidence)`. Windows
/// stagger across `n_lanes` lanes by `idx % n_lanes` so adjacent overlapping
/// windows occupy different lanes — with `n_lanes = ceil(window_s /
/// stride_s)`, each lane carries non-overlapping windows by construction.
///
/// The full window duration is preserved on the time axis (each line is as
/// wide as the window covers), and the band is below the spectrogram so the
/// time axis aligns. Confidence reads off the line colour.
///
/// Pass raw `result.segments` (one per inference call) — NOT the overlap-mean
/// diagnostic slots.
pub fn render_window_lanes(
    base: &DynamicImage,
    segments: &[sparrow_engine_types::AudioSegment],
    duration_s: f32,
    opts: &WindowLanesOpts,
) -> DynamicImage {
    let base_rgba = base.to_rgba8();
    let (w, base_h) = base_rgba.dimensions();
    if w == 0 || base_h == 0 {
        return DynamicImage::ImageRgba8(base_rgba);
    }
    let n_lanes = opts.n_lanes.clamp(1, MAX_AUDIO_WINDOW_LANES);
    let band_h = n_lanes
        .saturating_mul(opts.line_height_px)
        .saturating_add(n_lanes.saturating_sub(1).saturating_mul(opts.lane_gap_px));
    let total_h = base_h
        .saturating_add(opts.gap_to_base_px)
        .saturating_add(band_h);

    let mut canvas = RgbaImage::from_pixel(w, total_h, image::Rgba([0, 0, 0, 255]));
    for y in 0..base_h {
        for x in 0..w {
            canvas.put_pixel(x, y, *base_rgba.get_pixel(x, y));
        }
    }
    let gap_rgba = image::Rgba([
        opts.gap_color[0],
        opts.gap_color[1],
        opts.gap_color[2],
        255,
    ]);
    for y in base_h..base_h.saturating_add(opts.gap_to_base_px).min(total_h) {
        for x in 0..w {
            canvas.put_pixel(x, y, gap_rgba);
        }
    }

    if !duration_s.is_finite() || duration_s <= 0.0 || segments.is_empty() {
        return DynamicImage::ImageRgba8(canvas);
    }

    let band_y0 = base_h.saturating_add(opts.gap_to_base_px);
    let lane_pitch = opts.line_height_px.saturating_add(opts.lane_gap_px);
    let time_to_x = |t: f32| -> u32 {
        ((t / duration_s) * w as f32).round().clamp(0.0, w as f32) as u32
    };

    for (idx, seg) in segments.iter().enumerate() {
        if !valid_audio_segment(seg) {
            continue;
        }
        let lane = (idx as u32) % n_lanes;
        let y0 = band_y0.saturating_add(lane.saturating_mul(lane_pitch));
        let y1 = y0.saturating_add(opts.line_height_px).min(total_h);
        if y0 >= total_h {
            continue;
        }
        let x0 = time_to_x(seg.start_time_s);
        let x1 = time_to_x(seg.end_time_s).min(w);
        if x0 >= x1 {
            continue;
        }
        let [r, g, b] = inferno(seg.confidence.clamp(0.0, 1.0));
        let pixel = image::Rgba([r, g, b, 255]);
        for x in x0..x1 {
            for y in y0..y1 {
                canvas.put_pixel(x, y, pixel);
            }
        }
    }

    DynamicImage::ImageRgba8(canvas)
}

fn box_blur_1d(data: &mut [f32], radius: usize) {
    if data.is_empty() || radius == 0 {
        return;
    }
    let len = data.len();
    let mut output = vec![0.0f32; len];

    // Simple correct blur: average the window [max(0, i-radius), min(len, i+radius+1))
    // for each position. O(n*radius) but trivial for heatmap widths (< 5K).
    for (i, slot) in output.iter_mut().enumerate() {
        let left = i.saturating_sub(radius);
        let right = (i + radius + 1).min(len);
        let sum: f32 = data[left..right].iter().sum();
        *slot = sum / (right - left) as f32;
    }

    data.copy_from_slice(&output);
}

// ---------------------------------------------------------------------------
// Text labels (Phase 3.5 S4, lifted to runtime in Phase 3.7)
// ---------------------------------------------------------------------------

mod text {
    //! Glyph rasterization for the `RenderOpts.show_labels` runtime toggle.
    //! Embeds a DejaVu Sans TTF at compile time and draws
    //! `"{label} {conf:.2}"` above the bbox via per-pixel coverage alpha
    //! blending. Private module; the only public entry is
    //! `render_text_label`, called once per annotation from `render()`'s
    //! bbox branch when `show_labels` is true.

    use std::sync::OnceLock;

    use ab_glyph::{Font, FontRef, PxScale, ScaleFont};
    use image::RgbaImage;

    const FONT_BYTES: &[u8] = include_bytes!("../assets/fonts/DejaVuSans.ttf");
    const FONT_PX: f32 = 14.0;
    const PADDING_PX: i32 = 2;

    /// Parsed once per process. Previously re-parsed on every annotation —
    /// O(N) work for N bboxes. Tables are now parsed on first call; every
    /// subsequent `render_text_label` reads `&'static FontRef<'static>`.
    static FONT: OnceLock<FontRef<'static>> = OnceLock::new();

    fn font() -> &'static FontRef<'static> {
        FONT.get_or_init(|| {
            FontRef::try_from_slice(FONT_BYTES)
                .expect("bundled DejaVu Sans failed to parse — build bug")
        })
    }

    /// Draw `{label} {conf:.2}` above the given bbox top-left.
    ///
    /// Baseline sits `PADDING_PX` above the bbox top; if that would clip off
    /// the image, the baseline is placed just below the bbox top edge instead.
    /// Off-canvas glyphs are clipped pixel-by-pixel (no allocation escape).
    pub fn render_text_label(
        image: &mut RgbaImage,
        bbox_x0: i32,
        bbox_y0: i32,
        label: &str,
        confidence: f32,
        color: [u8; 3],
    ) {
        let text = format!("{} {:.2}", label, confidence);
        let font = font();
        let scale = PxScale::from(FONT_PX);
        let scaled = font.as_scaled(scale);
        let ascent = scaled.ascent();

        let mut pen_y = bbox_y0 as f32 - PADDING_PX as f32;
        if pen_y - ascent < 0.0 {
            pen_y = bbox_y0 as f32 + ascent + PADDING_PX as f32;
        }

        let mut pen_x = bbox_x0 as f32;
        let (iw, ih) = image.dimensions();

        for ch in text.chars() {
            let glyph_id = font.glyph_id(ch);
            let glyph = glyph_id
                .with_scale_and_position(scale, ab_glyph::point(pen_x, pen_y));
            if let Some(outlined) = font.outline_glyph(glyph) {
                let bounds = outlined.px_bounds();
                outlined.draw(|x, y, cov| {
                    let px = bounds.min.x as i32 + x as i32;
                    let py = bounds.min.y as i32 + y as i32;
                    if px < 0 || py < 0 || px >= iw as i32 || py >= ih as i32 {
                        return;
                    }
                    let pixel = image.get_pixel_mut(px as u32, py as u32);
                    let cov = cov.clamp(0.0, 1.0);
                    let inv = 1.0 - cov;
                    pixel.0[0] = (pixel.0[0] as f32 * inv + color[0] as f32 * cov).round() as u8;
                    pixel.0[1] = (pixel.0[1] as f32 * inv + color[1] as f32 * cov).round() as u8;
                    pixel.0[2] = (pixel.0[2] as f32 * inv + color[2] as f32 * cov).round() as u8;
                    pixel.0[3] = 255;
                });
            }
            pen_x += scaled.h_advance(glyph_id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::{AudioSegment, BBox, Detection};

    #[test]
    fn detections_to_annotations_preserves_data() {
        let result = DetectResult {
            detections: vec![Detection {
                bbox: BBox {
                    x_min: 0.1,
                    y_min: 0.2,
                    x_max: 0.5,
                    y_max: 0.6,
                },
                label: "animal".to_string(),
                label_id: 0,
                confidence: 0.9,
            }],
            image_width: 640,
            image_height: 480,
            processing_time_ms: 10.0,
        };
        let anns = detections_to_annotations(&result);
        assert_eq!(anns.len(), 1);
        assert_eq!(anns[0].label, "animal");
        assert!((anns[0].bbox[0] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn render_produces_same_dimensions() {
        let img = DynamicImage::new_rgb8(100, 80);
        let ann = BboxAnnotation {
            bbox: [0.1, 0.1, 0.5, 0.5],
            label: "animal".to_string(),
            confidence: 0.9,
        };
        let result = render(&img, &[ann], &RenderOpts::default());
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 80);
    }

    #[test]
    fn render_empty_annotations() {
        let img = DynamicImage::new_rgb8(100, 80);
        let result = render(&img, &[], &RenderOpts::default());
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 80);
    }

    // Regression (V1): non-finite bbox coords must be skipped, not cast to u32.
    // +Inf saturates to u32::MAX, causing off-screen point draws and u32 overflow
    // in (px0+px1)/2 (debug panic, release wrap). Test both +Inf and NaN paths.
    #[test]
    fn render_skips_nonfinite_bbox() {
        let img = DynamicImage::new_rgb8(100, 80);
        let bad_annotations = vec![
            BboxAnnotation {
                bbox: [f32::INFINITY, 0.1, 0.5, 0.5],
                label: "bad".to_string(),
                confidence: 0.9,
            },
            BboxAnnotation {
                bbox: [0.1, f32::NAN, 0.5, 0.5],
                label: "bad".to_string(),
                confidence: 0.9,
            },
            BboxAnnotation {
                bbox: [0.1, 0.1, f32::NEG_INFINITY, 0.5],
                label: "bad".to_string(),
                confidence: 0.9,
            },
        ];
        // Should not panic in debug mode (would panic on u32 overflow otherwise).
        let result = render(&img, &bad_annotations, &RenderOpts::default());
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 80);

        // Output must match empty-annotation render (nothing drawn).
        let baseline = render(&img, &[], &RenderOpts::default());
        assert_eq!(result.to_rgba8().as_raw(), baseline.to_rgba8().as_raw());
    }

    // Regression (V1 completeness): finite-but-extreme bbox coords must clamp to
    // [0,1] before the f32→u32 cast. Values like 1e30 pass is_finite, but
    // 1e30 * w overflows f32 to +Inf → u32::MAX, causing (px0+px1)/2 to overflow
    // u32 in the point path (debug panic, release wrap).
    #[test]
    fn render_clamps_extreme_finite_bbox() {
        let img = DynamicImage::new_rgb8(100, 80);
        let extreme = vec![BboxAnnotation {
            bbox: [1e30, 0.0, 0.5, 0.5],
            label: "extreme".to_string(),
            confidence: 0.9,
        }];
        // Must not panic. Clamped x_min=1.0 produces px0=100, and px1=50, so
        // (px0+px1)/2 = 75 — no overflow.
        let result = render(&img, &extreme, &RenderOpts::default());
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 80);

        // Verify no out-of-canvas draws by constructing the expected image:
        // x_min clamped to 1.0 (→ px0=100 = image right edge), so the rect is
        // degenerate on the right edge. The draw routines use saturating_sub
        // and bounds-check internally, so no panic and clipped output.
        let expected = vec![BboxAnnotation {
            bbox: [1.0, 0.0, 0.5, 0.5],
            label: "extreme".to_string(),
            confidence: 0.9,
        }];
        let reference = render(&img, &expected, &RenderOpts::default());
        assert_eq!(result.to_rgba8().as_raw(), reference.to_rgba8().as_raw());
    }

    #[test]
    fn inferno_endpoints() {
        let low = inferno(0.0);
        assert_eq!(low, [0, 0, 4]);
        let high = inferno(1.0);
        assert_eq!(high, [252, 255, 164]);
    }

    // Phase 3.5 W4 R1 reviewer regression: the audio_heatmap_e2e `saw_warm`
    // check in `tests/audio_heatmap_e2e.rs` relies on `R >= G` to detect warm
    // pixels, but at the exact inferno peak [252,255,164], G is 3 LSB greater
    // than R. The e2e test still passes because blur smearing produces
    // abundant t < 0.993 pixels (where R > G). This test documents the peak
    // asymmetry so a future change to the inferno colormap that inverts the
    // R-vs-G relationship at t=1.0 (for example, swapping to a palette whose
    // endpoint is [R>=G]) is flagged immediately and the e2e comment is
    // updated to match.
    #[test]
    fn inferno_peak_has_g_gt_r_by_small_margin() {
        let high = inferno(1.0);
        assert!(
            high[1] > high[0],
            "inferno peak no longer has G > R; update \
             tests/audio_heatmap_e2e.rs saw_warm comment — got {:?}",
            high
        );
        assert!(
            (high[1] as i16 - high[0] as i16) <= 5,
            "inferno peak G-R gap > 5 LSB; the blur-smeared warm-tone \
             assumption in the e2e saw_warm check may no longer hold — got {:?}",
            high
        );
    }

    #[test]
    fn heatmap_empty_segments() {
        let spec = DynamicImage::new_rgb8(200, 100);
        let result = render_audio_heatmap(&spec, &[], 10.0, &HeatmapOpts::default());
        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }

    // Regression (N5): inverted alpha range (alpha_min > alpha_max) is a
    // configuration bug. In debug builds, debug_assert! must fire loud so devs
    // catch the misconfig early. In release builds, the function must safely
    // return the input unchanged (no-op overlay) rather than silently invert
    // the alpha ramp. The `#[cfg(debug_assertions)]` gate below is required
    // because `debug_assert!` compiles to a no-op in release, so the panic
    // wouldn't fire and `#[should_panic]` would fail under `cargo test
    // --release`.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "alpha_min")]
    fn heatmap_inverted_alpha_panics_in_debug() {
        let spec = DynamicImage::new_rgb8(200, 100);
        let segments = vec![AudioSegment {
            start_time_s: 0.0,
            end_time_s: 5.0,
            confidence: 0.9,
            classes: Vec::new(),
        }];
        let bad_opts = HeatmapOpts {
            alpha_min: 200,
            alpha_max: 50,
            ..HeatmapOpts::default()
        };
        // In debug builds this panics at the debug_assert. In release the call
        // would return the input unchanged (tested separately via
        // #[cfg(not(debug_assertions))]).
        let _ = render_audio_heatmap(&spec, &segments, 10.0, &bad_opts);
    }

    // N5 release-path coverage. The `#[cfg(not(debug_assertions))]` gate means
    // this test is silently skipped under plain `cargo test` (debug) — no
    // "ignored" marker, no output. CI MUST run `cargo test --release` to
    // exercise it; otherwise a regression in the release early-return would go
    // undetected because the debug-only test above covers only the panic path.
    #[cfg(not(debug_assertions))]
    #[test]
    fn heatmap_inverted_alpha_returns_input_in_release() {
        let spec = DynamicImage::new_rgb8(200, 100);
        let segments = vec![AudioSegment {
            start_time_s: 0.0,
            end_time_s: 5.0,
            confidence: 0.9,
            classes: Vec::new(),
        }];
        let bad_opts = HeatmapOpts {
            alpha_min: 200,
            alpha_max: 50,
            ..HeatmapOpts::default()
        };
        let result = render_audio_heatmap(&spec, &segments, 10.0, &bad_opts);
        // No overlay applied — output equals input.
        assert_eq!(result.to_rgba8().as_raw(), spec.to_rgba8().as_raw());
    }

    #[test]
    fn heatmap_with_segment() {
        let spec = DynamicImage::new_rgb8(200, 100);
        let segments = vec![AudioSegment {
            start_time_s: 0.0,
            end_time_s: 5.0,
            confidence: 0.9,
            classes: Vec::new(),
        }];
        let result = render_audio_heatmap(&spec, &segments, 10.0, &HeatmapOpts::default());
        assert_eq!(result.width(), 200);
    }

    #[test]
    fn audio_segments_to_annotations_normalizes_time() {
        let result = AudioDetectResult {
            segments: vec![AudioSegment {
                start_time_s: 2.0,
                end_time_s: 4.0,
                confidence: 0.8,
                classes: Vec::new(),
            }],
            duration_s: 10.0,
            sample_rate: 48000,
            processing_time_ms: 5.0,
        };
        let anns = audio_segments_to_annotations(&result);
        assert_eq!(anns.len(), 1);
        assert!((anns[0].bbox[0] - 0.2).abs() < 1e-6);
        assert!((anns[0].bbox[2] - 0.4).abs() < 1e-6);
    }

    // Regression: segment with start_time > duration should not panic.
    #[test]
    fn heatmap_segment_beyond_duration_no_panic() {
        let spec = DynamicImage::new_rgb8(100, 50);
        let segments = vec![AudioSegment {
            start_time_s: 15.0, // beyond duration
            end_time_s: 20.0,
            confidence: 0.9,
            classes: Vec::new(),
        }];
        let result = render_audio_heatmap(&spec, &segments, 10.0, &HeatmapOpts::default());
        assert_eq!(result.width(), 100);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Pow emphasis exponent must be finite and > 0")]
    fn heatmap_invalid_pow_panics_in_debug() {
        let spec = DynamicImage::new_rgb8(100, 50);
        let segments = vec![AudioSegment {
            start_time_s: 0.0,
            end_time_s: 1.0,
            confidence: 0.0,
            classes: Vec::new(),
        }];
        let opts = HeatmapOpts {
            emphasis: Emphasis::Pow(-1.0),
            ..HeatmapOpts::default()
        };
        let _ = render_audio_heatmap(&spec, &segments, 10.0, &opts);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn heatmap_invalid_pow_returns_input_in_release() {
        let spec = DynamicImage::new_rgb8(100, 50);
        let baseline = spec.to_rgba8();
        let segments = vec![AudioSegment {
            start_time_s: 0.0,
            end_time_s: 1.0,
            confidence: 0.0,
            classes: Vec::new(),
        }];
        let opts = HeatmapOpts {
            emphasis: Emphasis::Pow(-1.0),
            ..HeatmapOpts::default()
        };
        let result = render_audio_heatmap(&spec, &segments, 10.0, &opts);
        assert_eq!(result.to_rgba8().as_raw(), baseline.as_raw());
    }

    // Note: the old "alpha_max < alpha_min should not panic" regression was
    // replaced by heatmap_inverted_alpha_panics_in_debug (+ the release-gated
    // companion). The new semantic is "loud in dev, safe in release" — silent
    // acceptance masks configuration bugs.

    // ----- range overlay tests -----

    fn make_range(start: f32, end: f32, conf: f32) -> sparrow_engine_types::AudioRange {
        sparrow_engine_types::AudioRange {
            start_time_s: start,
            end_time_s: end,
            max_confidence: conf,
            class: None,
        }
    }

    #[test]
    fn range_overlay_preserves_dimensions() {
        let img = DynamicImage::new_rgb8(200, 100);
        let result = render_range_overlay(
            &img,
            &[make_range(1.0, 3.0, 0.8)],
            10.0,
            &RangeOverlayOpts::default(),
        );
        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
    }

    #[test]
    fn range_overlay_empty_ranges_is_noop() {
        let img = DynamicImage::new_rgb8(50, 30);
        let baseline = img.to_rgba8();
        let result = render_range_overlay(&img, &[], 10.0, &RangeOverlayOpts::default());
        assert_eq!(result.to_rgba8().as_raw(), baseline.as_raw(),
            "empty ranges must produce a pixel-identical output");
    }

    #[test]
    fn range_overlay_zero_duration_is_noop() {
        let img = DynamicImage::new_rgb8(50, 30);
        let baseline = img.to_rgba8();
        let result = render_range_overlay(
            &img,
            &[make_range(0.0, 1.0, 0.5)],
            0.0,
            &RangeOverlayOpts::default(),
        );
        assert_eq!(result.to_rgba8().as_raw(), baseline.as_raw());
    }

    #[test]
    fn range_overlay_draws_at_correct_x() {
        // 100px wide / 10s duration → 10 px/s. Range at 4-6s should land
        // approximately at x=40 and x=60.
        let img = DynamicImage::new_rgb8(100, 20);
        let opts = RangeOverlayOpts {
            bar_width_px: 1,
            ..RangeOverlayOpts::default()
        };
        let result = render_range_overlay(&img, &[make_range(4.0, 6.0, 1.0)], 10.0, &opts);
        let result = result.to_rgba8();
        // The black source has [0,0,0]. The overlay color is cyan [0,255,255].
        // After alpha blend with bar_alpha=220/255, cyan G channel should be ~220.
        // Sample at the start bar (x=40, y=10):
        let pixel_at_start = result.get_pixel(40, 10);
        assert!(
            pixel_at_start[1] > 100,
            "expected blended cyan at start bar (x=40), got {:?}",
            pixel_at_start
        );
        // Sample mid-range, top half (no band there) — should be untouched (black):
        let pixel_mid_top = result.get_pixel(50, 5);
        assert_eq!(pixel_mid_top[1], 0, "mid-top should be untouched, got {:?}", pixel_mid_top);
        // Sample mid-range, bottom band — should be blended cyan:
        let pixel_mid_bottom = result.get_pixel(50, 18);
        assert!(
            pixel_mid_bottom[1] > 100,
            "expected blended band at bottom (x=50, y=18), got {:?}",
            pixel_mid_bottom
        );
    }

    #[test]
    fn range_overlay_draws_end_bar_at_duration() {
        let img = DynamicImage::new_rgb8(100, 20);
        let opts = RangeOverlayOpts {
            bar_width_px: 1,
            ..RangeOverlayOpts::default()
        };
        let result = render_range_overlay(&img, &[make_range(8.0, 10.0, 1.0)], 10.0, &opts);
        let result = result.to_rgba8();
        let pixel_at_end = result.get_pixel(99, 10);
        assert!(
            pixel_at_end[1] > 100,
            "expected blended cyan at final drawable column, got {:?}",
            pixel_at_end
        );
        let pixel_mid_bottom = result.get_pixel(95, 18);
        assert!(
            pixel_mid_bottom[1] > 100,
            "bottom band should still include the final range, got {:?}",
            pixel_mid_bottom
        );
    }

    #[test]
    fn range_overlay_band_alpha_scales_with_confidence() {
        // High-confidence range should produce a more opaque band than
        // low-confidence one (more cyan = larger G channel value).
        let img = DynamicImage::new_rgb8(200, 30);
        let high = render_range_overlay(
            &img,
            &[make_range(2.0, 4.0, 1.0)],
            10.0,
            &RangeOverlayOpts::default(),
        );
        let low = render_range_overlay(
            &img,
            &[make_range(2.0, 4.0, 0.1)],
            10.0,
            &RangeOverlayOpts::default(),
        );
        // Sample mid-range, bottom band (x=60, y=27 — well inside band):
        let g_high = high.to_rgba8().get_pixel(60, 27)[1];
        let g_low = low.to_rgba8().get_pixel(60, 27)[1];
        assert!(
            g_high > g_low,
            "expected high-conf band brighter than low-conf: g_high={g_high}, g_low={g_low}"
        );
    }

    #[test]
    fn range_overlay_clamps_out_of_bounds_range() {
        // A range whose end_time_s exceeds duration should be clamped to image
        // width without panicking.
        let img = DynamicImage::new_rgb8(100, 20);
        let result = render_range_overlay(
            &img,
            &[make_range(8.0, 15.0, 0.9)], // end beyond duration=10
            10.0,
            &RangeOverlayOpts::default(),
        );
        assert_eq!(result.width(), 100);
    }

    // ----- window lanes tests -----

    fn make_seg_full(start: f32, end: f32, conf: f32) -> sparrow_engine_types::AudioSegment {
        sparrow_engine_types::AudioSegment {
            start_time_s: start,
            end_time_s: end,
            confidence: conf,
            classes: Vec::new(),
        }
    }

    #[test]
    fn window_lanes_extends_height() {
        let img = DynamicImage::new_rgb8(200, 100);
        let opts = WindowLanesOpts {
            n_lanes: 4,
            line_height_px: 2,
            lane_gap_px: 2,
            gap_to_base_px: 1,
            ..WindowLanesOpts::default()
        };
        let result = render_window_lanes(
            &img,
            &[make_seg_full(0.0, 1.0, 0.5)],
            10.0,
            &opts,
        );
        assert_eq!(result.width(), 200);
        // band = 4*2 + 3*2 = 14, gap = 1 → total = 100 + 1 + 14 = 115
        assert_eq!(result.height(), 115);
    }

    #[test]
    fn window_lanes_preserves_base_pixels() {
        let mut img = RgbaImage::new(50, 30);
        for y in 0..30 {
            for x in 0..50 {
                img.put_pixel(x, y, image::Rgba([10, 20, 30, 255]));
            }
        }
        let dyn_img = DynamicImage::ImageRgba8(img);
        let result = render_window_lanes(
            &dyn_img,
            &[make_seg_full(0.0, 1.0, 1.0)],
            10.0,
            &WindowLanesOpts::default(),
        );
        let result = result.to_rgba8();
        let p = result.get_pixel(25, 15);
        assert_eq!([p[0], p[1], p[2]], [10, 20, 30],
            "base pixel at (25,15) was overwritten, got {:?}", p);
    }

    #[test]
    fn window_lanes_empty_segments_paints_only_base_and_gap() {
        let img = DynamicImage::new_rgb8(80, 20);
        let result = render_window_lanes(&img, &[], 10.0, &WindowLanesOpts::default());
        let result = result.to_rgba8();
        // Lanes band rows should be black (canvas init) since no segments.
        let p_lane = result.get_pixel(40, 25);
        assert_eq!([p_lane[0], p_lane[1], p_lane[2]], [0, 0, 0],
            "empty lanes band should be black, got {:?}", p_lane);
    }

    #[test]
    fn window_lanes_idx_modulo_lane_assignment_with_inferno_color() {
        // 200px / 10s = 20 px/s. window=1.0s, stride=0.3s → 4 lanes.
        // base_h = 40, gap_to_base = 1 → band starts at y=41.
        // line_height = 2, lane_gap = 2, lane_pitch = 4.
        // Lane 0 → y=[41..43], lane 1 → y=[45..47], lane 2 → y=[49..51], lane 3 → y=[53..55].
        let img = DynamicImage::new_rgb8(200, 40);
        let opts = WindowLanesOpts {
            n_lanes: 4,
            line_height_px: 2,
            lane_gap_px: 2,
            gap_to_base_px: 1,
            ..WindowLanesOpts::default()
        };
        let segments = vec![
            make_seg_full(0.0, 1.0, 1.0), // lane 0, x=[0,20], bright inferno
            make_seg_full(0.3, 1.3, 0.0), // lane 1, x=[6,26], dark inferno
        ];
        let result = render_window_lanes(&img, &segments, 10.0, &opts);
        let result = result.to_rgba8();

        // Lane 0 line (y=42, x=10): high-confidence segment → bright inferno.
        let p_lane0 = result.get_pixel(10, 42);
        assert!(p_lane0[1] > 200,
            "expected bright inferno at lane 0, got {:?}", p_lane0);

        // Lane 1 line (y=46, x=10): low-confidence segment → dark inferno.
        let p_lane1 = result.get_pixel(10, 46);
        assert!(p_lane1[0] < 30 && p_lane1[1] < 30,
            "expected dark inferno at lane 1, got {:?}", p_lane1);

        // Gap row between lanes (y=44) should be black (canvas background).
        let p_gap = result.get_pixel(10, 44);
        assert_eq!([p_gap[0], p_gap[1], p_gap[2]], [0, 0, 0],
            "lane gap should be black, got {:?}", p_gap);
    }

    // Regression: box_blur_1d symmetry — uniform input stays uniform.
    #[test]
    fn box_blur_1d_uniform_input() {
        let mut data = vec![1.0f32; 20];
        box_blur_1d(&mut data, 3);
        // All values should remain approximately 1.0 (uniform input is a blur fixpoint).
        for (i, v) in data.iter().enumerate() {
            assert!(
                (*v - 1.0).abs() < 0.01,
                "box_blur_1d[{i}] = {v}, expected ~1.0"
            );
        }
    }

    // Regression: box_blur_1d output should be centered (no shift).
    // A delta peak at index 10 becomes a plateau after blur — verify center of mass.
    #[test]
    fn box_blur_1d_peak_stays_centered() {
        let len = 21;
        let mut data = vec![0.0f32; len];
        data[10] = 1.0; // peak at center
        box_blur_1d(&mut data, 2);
        // Center of mass should be at or near index 10.
        let total: f32 = data.iter().sum();
        let center_of_mass: f32 = data
            .iter()
            .enumerate()
            .map(|(i, v)| i as f32 * v)
            .sum::<f32>()
            / total;
        assert!(
            (center_of_mass - 10.0).abs() < 0.5,
            "center of mass shifted to {center_of_mass:.2}, expected ~10.0"
        );
    }

    // -- Phase 3.5 S3 (MT-9): ModelType-driven dispatch --

    /// Count how many pixels in a canvas have been painted a specific RGB color.
    fn count_rgb_pixels(canvas: &RgbaImage, target: [u8; 3]) -> usize {
        canvas
            .pixels()
            .filter(|p| p.0[0] == target[0] && p.0[1] == target[1] && p.0[2] == target[2])
            .count()
    }

    // OverheadDetector dispatch draws a filled circle (dot) at the bbox centroid.
    // Same annotation rendered as Detector would produce a rectangle outline.
    #[test]
    fn render_dispatches_dot_for_overhead_detector() {
        let img = DynamicImage::new_rgb8(200, 200);
        let ann = BboxAnnotation {
            // 20px-wide centroid: larger than 2*point_radius (8px), so the old
            // pixel-size heuristic would have misclassified this as a bbox.
            bbox: [0.45, 0.45, 0.55, 0.55],
            label: "animal".to_string(),
            confidence: 0.9,
        };
        let opts = RenderOpts {
            model_type: ModelType::OverheadDetector,
            point_radius: 4,
            point_threshold: 0.0,
            ..Default::default()
        };
        let result = render(&img, &[ann], &opts);
        let canvas = result.to_rgba8();
        let green = count_rgb_pixels(&canvas, [0, 255, 0]);
        // A radius-4 circle has ~49 pixels (pi * r^2 rounded up for the mask).
        // Exact bbox-outline rendering would paint far more (perimeter of a
        // 20×20 rectangle at line_width 2 = ~80 px at minimum).
        assert!(green > 0, "OverheadDetector must draw a visible dot");
        assert!(
            green <= 81,
            "OverheadDetector draws a dot, not a rectangle outline; got {green} green px"
        );
    }

    // Detector dispatch draws a bounding rectangle even when the bbox is tiny
    // (regression lock on the removed pixel-size heuristic).
    #[test]
    fn render_dispatches_bbox_for_detector_even_for_tiny_bbox() {
        let img = DynamicImage::new_rgb8(200, 200);
        let ann = BboxAnnotation {
            // 4×4 box — smaller than the old `point_radius * 2 = 8` threshold.
            // Pre-S3 this would have rendered as a dot even for a standard
            // detector; S3 must render it as a bbox outline because
            // model_type=Detector.
            bbox: [0.50, 0.50, 0.52, 0.52],
            label: "animal".to_string(),
            confidence: 0.9,
        };
        let opts = RenderOpts {
            model_type: ModelType::Detector,
            point_radius: 4,
            ..Default::default()
        };
        let result = render(&img, std::slice::from_ref(&ann), &opts);
        let canvas = result.to_rgba8();
        let green = count_rgb_pixels(&canvas, [0, 255, 0]);
        assert!(green > 0, "Detector must draw at least some green pixels");
        // A filled 4-radius circle would paint ~49 pixels of uniform coverage.
        // A 4×4 bbox outline at line_width 2 paints the full 4×4 square = 16 px.
        // The critical regression: result must NOT equal the dot-only render.
        let dot_opts = RenderOpts {
            model_type: ModelType::OverheadDetector,
            point_radius: 4,
            ..Default::default()
        };
        let dot_result = render(&img, &[ann], &dot_opts);
        assert_ne!(
            canvas.as_raw(),
            dot_result.to_rgba8().as_raw(),
            "Detector dispatch must differ from OverheadDetector dispatch on the same tiny bbox"
        );
    }

    // Sub-threshold overhead detections render nothing (dot-path respects
    // point_threshold).
    #[test]
    fn render_overhead_dot_respects_point_threshold() {
        let img = DynamicImage::new_rgb8(100, 100);
        let ann = BboxAnnotation {
            bbox: [0.4, 0.4, 0.6, 0.6],
            label: "animal".to_string(),
            confidence: 0.3,
        };
        let opts = RenderOpts {
            model_type: ModelType::OverheadDetector,
            point_threshold: 0.5, // well above the 0.3 confidence
            ..Default::default()
        };
        let result = render(&img, &[ann], &opts);
        // Nothing drawn → output equals a black canvas of matching size.
        let baseline = render(&img, &[], &opts);
        assert_eq!(
            result.to_rgba8().as_raw(),
            baseline.to_rgba8().as_raw(),
            "Sub-threshold overhead detection must render no dot"
        );
    }

    // Regression lock: the pixel-size heuristic must be gone. A detector
    // annotation with a centroid-sized bbox at high canvas resolution used to
    // trigger dot rendering; now it must render as a bbox.
    #[test]
    fn render_no_pixel_size_heuristic_regression() {
        // 4K canvas, 0.5% bbox — 20×20 px. At pre-S3 defaults (point_radius=4),
        // 2*point_radius = 8, so bbox_w=20 > 8 triggered the bbox path; but at
        // point_radius=16 the threshold was 32, so bbox_w=20 <= 32 triggered
        // the dot path. Demonstrate that model_type=Detector ignores
        // point_radius entirely for dispatch now.
        let img = DynamicImage::new_rgb8(3840, 2160);
        let ann = BboxAnnotation {
            bbox: [0.498, 0.498, 0.503, 0.503],
            label: "animal".to_string(),
            confidence: 0.9,
        };
        let detector_opts = RenderOpts {
            model_type: ModelType::Detector,
            point_radius: 100, // would trivially cover the bbox under old heuristic
            ..Default::default()
        };
        let overhead_opts = RenderOpts {
            model_type: ModelType::OverheadDetector,
            point_radius: 100,
            ..Default::default()
        };
        let det = render(&img, std::slice::from_ref(&ann), &detector_opts);
        let ovh = render(&img, std::slice::from_ref(&ann), &overhead_opts);
        assert_ne!(
            det.to_rgba8().as_raw(),
            ovh.to_rgba8().as_raw(),
            "ModelType dispatch must be the sole signal; point_radius must not steal the bbox path"
        );
    }

    /// `show_labels=true` writes glyphs into the band above the bbox.
    ///
    /// Avoids binary-exact golden match (subpixel rasterizer drift). Asserts
    /// the labeled render differs from the default-opts (labels-off) render
    /// in the band above the bbox top.
    #[test]
    fn viz_text_labels_modify_pixels_above_bbox() {
        let img = DynamicImage::new_rgb8(400, 300);
        let ann = BboxAnnotation {
            bbox: [0.25, 0.50, 0.75, 0.80],
            label: "animal".to_string(),
            confidence: 0.87,
        };
        let opts = RenderOpts {
            show_labels: true,
            ..Default::default()
        };

        let labeled = render(&img, std::slice::from_ref(&ann), &opts).to_rgba8();

        let mut changed = 0usize;
        for y in 130..150 {
            for x in 100..300 {
                let p = labeled.get_pixel(x, y);
                if p.0 != [0, 0, 0, 255] {
                    changed += 1;
                }
            }
        }
        assert!(
            changed > 20,
            "show_labels=true should write label glyphs into the band above the bbox; got {changed} non-black pixels",
        );
    }

    /// `show_labels=false` (default) leaves the band above the bbox untouched.
    /// Companion to the test above — guards against accidentally always
    /// rendering labels after the runtime-flag lift.
    #[test]
    fn viz_text_labels_off_leaves_band_clean() {
        let img = DynamicImage::new_rgb8(400, 300);
        let ann = BboxAnnotation {
            bbox: [0.25, 0.50, 0.75, 0.80],
            label: "animal".to_string(),
            confidence: 0.87,
        };
        let opts = RenderOpts::default();
        assert!(
            !opts.show_labels,
            "RenderOpts::default() must keep show_labels off"
        );

        let unlabeled = render(&img, std::slice::from_ref(&ann), &opts).to_rgba8();

        let mut changed = 0usize;
        for y in 130..150 {
            for x in 100..300 {
                let p = unlabeled.get_pixel(x, y);
                if p.0 != [0, 0, 0, 255] {
                    changed += 1;
                }
            }
        }
        assert_eq!(
            changed, 0,
            "show_labels=false must NOT touch pixels above the bbox; got {changed} non-black pixels",
        );
    }
}

#[cfg(test)]
mod phase_a_r1_viz {
    use super::*;
    use sparrow_engine_types::{BBox, DetectResult, Detection, ModelType};

    /// `render` is byte-identical for two identical calls. Pins determinism
    /// across the entire pixel pipeline (resolve_color, draw_rect, alpha
    /// blend) and protects against accidental introduction of HashMap
    /// iteration order, time-based jitter, or threading non-determinism.
    #[test]
    fn render_is_byte_deterministic() {
        let img = image::DynamicImage::new_rgb8(80, 60);
        let anns = vec![
            BboxAnnotation {
                bbox: [0.1, 0.1, 0.4, 0.4],
                label: "animal".into(),
                confidence: 0.9,
            },
            BboxAnnotation {
                bbox: [0.5, 0.5, 0.9, 0.9],
                label: "person".into(),
                confidence: 0.8,
            },
        ];
        let opts = RenderOpts::default();
        let a = render(&img, &anns, &opts).to_rgba8();
        let b = render(&img, &anns, &opts).to_rgba8();
        assert_eq!(
            a.as_raw(),
            b.as_raw(),
            "render() must be byte-deterministic for identical input + opts"
        );
    }

    /// `detections_to_annotations` preserves bbox values, label, and
    /// confidence verbatim (no clamping, no rounding). The existing
    /// `detections_to_annotations_preserves_data` checks one detection +
    /// one bbox component; this widens to all 4 bbox components and to a
    /// non-trivial confidence value.
    #[test]
    fn detections_to_annotations_preserves_all_bbox_fields() {
        let res = DetectResult {
            detections: vec![Detection {
                bbox: BBox {
                    x_min: 0.123,
                    y_min: 0.456,
                    x_max: 0.789,
                    y_max: 0.012,
                },
                label: "fox".into(),
                label_id: 7,
                confidence: 0.42,
            }],
            image_width: 100,
            image_height: 100,
            processing_time_ms: 1.0,
        };
        let anns = detections_to_annotations(&res);
        assert_eq!(anns.len(), 1);
        let a = &anns[0];
        assert!((a.bbox[0] - 0.123).abs() < 1e-6);
        assert!((a.bbox[1] - 0.456).abs() < 1e-6);
        assert!((a.bbox[2] - 0.789).abs() < 1e-6);
        assert!((a.bbox[3] - 0.012).abs() < 1e-6);
        assert_eq!(a.label, "fox");
        assert!((a.confidence - 0.42).abs() < 1e-6);
    }

    /// `RenderOpts::default()` produces sane values: line_width > 0 (else no
    /// rectangle is drawn), point_radius > 0 (else circle path no-ops), no
    /// fill_alpha (None), default model_type is Detector (not OverheadDetector,
    /// which would silently switch dispatch for callers that omit the field).
    #[test]
    fn render_opts_default_is_sane() {
        let d = RenderOpts::default();
        assert!(d.line_width > 0, "line_width must be > 0 to draw outlines");
        assert!(d.point_radius > 0, "point_radius must be > 0 to draw dots");
        assert!(d.fill_alpha.is_none(), "default must not fill bboxes");
        assert!(matches!(d.model_type, ModelType::Detector));
        assert!(!d.show_labels, "default must keep label glyphs off");
        assert!(d.colors.is_empty(), "default colors map must be empty");
    }
}
