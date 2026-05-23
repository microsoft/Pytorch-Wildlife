//! Core types for the sparrow-engine public API.
//!
//! All bounding boxes are normalized [0,1] at the public API boundary.
//! Consumers convert to pixels at display time: `pixel_x = bbox.x_min * image_width`.

use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Bounding box
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in normalized [0,1] coordinates, xyxy format.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

/// A single detection result.
#[derive(Debug, Clone)]
pub struct Detection {
    pub bbox: BBox,
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
}

/// Full detection output from a single `detect()` call.
#[derive(Debug, Clone)]
pub struct DetectResult {
    pub detections: Vec<Detection>,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

/// A single classification result (one class prediction).
#[derive(Debug, Clone)]
pub struct Classification {
    pub label: String,
    pub label_id: u32,
    pub confidence: f32,
}

/// Full classification output from a single `classify()` call.
#[derive(Debug, Clone)]
pub struct ClassifyResult {
    pub classifications: Vec<Classification>,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// A detection with an optional classification attached (from pipeline).
#[derive(Debug, Clone)]
pub struct PipelineDetection {
    pub detection: Detection,
    pub classification: Option<Classification>,
}

/// Full pipeline output from `run_pipeline()`.
#[derive(Debug, Clone)]
pub struct PipelineResult {
    pub pipeline_id: String,
    pub detections: Vec<PipelineDetection>,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
}

// ---------------------------------------------------------------------------
// Inference options
// ---------------------------------------------------------------------------

/// Options for detection inference. All fields optional (None = use manifest default).
#[derive(Debug, Clone, Default)]
pub struct DetectOpts {
    /// Override minimum confidence threshold.
    pub confidence_threshold: Option<f32>,
    /// Cap output count. None = unlimited.
    pub max_detections: Option<u32>,
}

/// Options for classification inference. All fields optional.
#[derive(Debug, Clone, Default)]
pub struct ClassifyOpts {
    /// Number of top classifications to return. None = 1.
    pub top_k: Option<u32>,
}

// ---------------------------------------------------------------------------
// Image input
// ---------------------------------------------------------------------------

/// Pixel format for raw image buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PixelFormat {
    /// 3 channels, standard.
    Rgb = 0,
    /// 4 channels, alpha ignored.
    Rgba = 1,
    /// 4 channels, blue-first (Windows Bitmap default: Format32bppArgb).
    Bgra = 2,
    /// 3 channels, blue-first (Windows Bitmap: Format24bppRgb).
    Bgr = 3,
}

/// Image input — one of three forms consumers can provide.
#[derive(Debug, Clone)]
pub enum ImageInput {
    /// JPEG/PNG encoded byte buffer (most common).
    Encoded(Vec<u8>),
    /// Path to an image file on disk.
    FilePath(PathBuf),
    /// Pre-decoded raw pixel buffer.
    Raw {
        data: Vec<u8>,
        width: u32,
        height: u32,
        stride: u32,
        format: PixelFormat,
    },
}

// ---------------------------------------------------------------------------
// Audio input
// ---------------------------------------------------------------------------

/// Audio input — WAV file or raw samples.
#[derive(Debug, Clone)]
pub enum AudioInput {
    /// Path to a WAV file on disk.
    FilePath(PathBuf),
    /// Pre-decoded raw samples (mono f32 [-1,1]).
    Samples { data: Vec<f32>, sample_rate: u32 },
}

// ---------------------------------------------------------------------------
// Audio detection
// ---------------------------------------------------------------------------

/// A single classification slot inside an `AudioSegment`. Phase 4.2+ unified
/// audio model: every audio segment carries a top-K list of `AudioClass`
/// entries (K=1 for binary detectors like MD_AudioBirds_V1, K≥1 for
/// multi-class classifiers like Perch 2).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioClass {
    /// Index into `manifest.labels` (0-based).
    pub class_idx: u32,
    /// Resolved label string from `labels.txt`. `None` when the model has no
    /// labels file (e.g. legacy binary detectors that pre-date label files).
    pub label: Option<String>,
    /// Softmax probability (for classifiers) or sigmoid confidence (for
    /// binary detectors). Always in `[0, 1]`.
    pub probability: f32,
}

/// A single detected audio segment.
///
/// `confidence` is the top-class probability and is preserved for backward
/// compatibility with all existing readers; it equals `classes[0].probability`
/// when `classes` is non-empty. `classes` carries the full top-K list (sorted
/// descending by probability) for multi-class classifiers; for binary
/// detectors `classes` is a 1-entry vec or empty (when no labels file is
/// present).
#[derive(Debug, Clone, PartialEq)]
pub struct AudioSegment {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub confidence: f32,
    /// Top-K class candidates for this segment, sorted by probability desc.
    /// Empty for legacy binary detectors with no labels file.
    pub classes: Vec<AudioClass>,
}

/// Full audio detection output from a single `detect_audio()` call.
#[derive(Debug, Clone)]
pub struct AudioDetectResult {
    pub segments: Vec<AudioSegment>,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
}

/// Merged-segment range output from `detect_audio::merge_segments`.
///
/// `class` carries the resolved label string when class-aware merging is in
/// effect (multi-class classifiers). For binary detectors with no labels file
/// it is `None`.
///
/// Phase 3.8 Phase A note: this lived in the legacy audio-detection module but
/// was hoisted to `sparrow-engine-types` (Commit 2 widening) because `sparrow-engine-core`'s
/// `viz::render_range_overlay` consumes it in its public API and sparrow-engine-core
/// cannot reach into sparrow-engine-cpu (dep-direction violation). Pure POD; no
/// behavior change. `sparrow-engine-cpu::detect_audio` re-exports it so the
/// `engine_dispatch::detect_audio::AudioRange` consumer path keeps resolving.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioRange {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub max_confidence: f32,
    pub class: Option<String>,
}

/// Options for audio detection inference. All fields optional (None = use manifest default).
#[derive(Debug, Clone, Default)]
pub struct AudioDetectOpts {
    /// Override minimum confidence threshold.
    pub confidence_threshold: Option<f32>,
    /// Override segment duration in seconds.
    pub segment_duration_s: Option<f32>,
    /// Override segment stride in seconds.
    pub stride_s: Option<f32>,
}

// ---------------------------------------------------------------------------
// Model info (for Engine::loaded_models)
// ---------------------------------------------------------------------------

/// Summary info about a model (loaded or available on disk).
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub id: String,
    pub path: PathBuf,
    pub model_type: ModelType,
    /// Whether this model is the default for its type (from manifest `default = true`).
    pub default: bool,
    pub version: Option<String>,
    pub description: Option<String>,
    pub onnx_sha256: Option<String>,
    pub onnx_size_bytes: Option<u64>,
}

/// Inferred model type based on preprocessing + postprocessing method + subtype.
///
/// `OverheadDetector` is distinguished from `Detector` by the manifest's
/// `[model].subtype = "overhead"` hint. Visualization dispatches on this
/// variant to draw a dot at the bbox centroid instead of a rectangle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelType {
    Detector,
    /// Point-detection model rendered as a dot at the bbox centroid
    /// (e.g., HerdNet, OWL-T). Distinguished by manifest `subtype = "overhead"`.
    OverheadDetector,
    Classifier,
    AudioDetector,
    AudioClassifier,
}

impl ModelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelType::Detector => "detector",
            ModelType::OverheadDetector => "overhead_detector",
            ModelType::Classifier => "classifier",
            ModelType::AudioDetector => "audio_detector",
            ModelType::AudioClassifier => "audio_classifier",
        }
    }
}

impl std::fmt::Display for ModelType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Rendering / behaviour hint from the TOML `[model].subtype` field.
///
/// - `Standard` (default): bounding-box detectors (MDv6, DeepFaune).
/// - `Overhead`: point / overhead-dot detectors (HerdNet, OWL-T). Viz renders
///   a dot at the bbox centroid instead of a rectangle.
///
/// The enum is intentionally minimal; future additions (`Segmentation`, etc.)
/// belong here. Classifier and audio models ignore this hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ModelSubtype {
    #[default]
    Standard,
    Overhead,
}

impl ModelSubtype {
    pub fn as_str(&self) -> &'static str {
        match self {
            ModelSubtype::Standard => "standard",
            ModelSubtype::Overhead => "overhead",
        }
    }
}

impl std::fmt::Display for ModelSubtype {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod phase_a_r1_types_tests {
    use super::*;

    #[test]
    fn model_type_as_str_table_driven() {
        // Lock in the public surface strings consumed by Sparrow Studio + CLI.
        let table: &[(ModelType, &str)] = &[
            (ModelType::Detector, "detector"),
            (ModelType::OverheadDetector, "overhead_detector"),
            (ModelType::Classifier, "classifier"),
            (ModelType::AudioDetector, "audio_detector"),
            (ModelType::AudioClassifier, "audio_classifier"),
        ];
        for (mt, expected) in table {
            assert_eq!(mt.as_str(), *expected, "as_str mismatch for {mt:?}");
        }
    }

    #[test]
    fn model_type_display_matches_as_str() {
        for mt in [
            ModelType::Detector,
            ModelType::OverheadDetector,
            ModelType::Classifier,
            ModelType::AudioDetector,
            ModelType::AudioClassifier,
        ] {
            assert_eq!(
                mt.to_string(),
                mt.as_str(),
                "Display impl must equal as_str() for {mt:?}"
            );
        }
    }

    #[test]
    fn model_subtype_as_str_and_display_table() {
        assert_eq!(ModelSubtype::Standard.as_str(), "standard");
        assert_eq!(ModelSubtype::Overhead.as_str(), "overhead");
        assert_eq!(ModelSubtype::Standard.to_string(), "standard");
        assert_eq!(ModelSubtype::Overhead.to_string(), "overhead");
    }

    #[test]
    fn model_subtype_default_is_standard() {
        let s: ModelSubtype = Default::default();
        assert_eq!(s, ModelSubtype::Standard);
    }

    #[test]
    fn bbox_partial_eq_round_trip() {
        let a = BBox {
            x_min: 0.1,
            y_min: 0.2,
            x_max: 0.3,
            y_max: 0.4,
        };
        let b = BBox {
            x_min: 0.1,
            y_min: 0.2,
            x_max: 0.3,
            y_max: 0.4,
        };
        let c = BBox {
            x_min: 0.1,
            y_min: 0.2,
            x_max: 0.3,
            y_max: 0.5, // differs
        };
        assert_eq!(a, b);
        assert_ne!(a, c);
        // Copy semantics — bbox is small, must be Copy.
        let d = a;
        let _e = a; // both readable after copy.
        assert_eq!(d, a);
    }

    #[test]
    fn detection_clone_preserves_all_fields() {
        let d = Detection {
            bbox: BBox {
                x_min: 0.0,
                y_min: 0.0,
                x_max: 1.0,
                y_max: 1.0,
            },
            label: "animal".to_string(),
            label_id: 1,
            confidence: 0.987,
        };
        let cloned = d.clone();
        assert_eq!(cloned.bbox, d.bbox);
        assert_eq!(cloned.label, d.label);
        assert_eq!(cloned.label_id, d.label_id);
        assert!(
            (cloned.confidence - d.confidence).abs() < f32::EPSILON,
            "confidence diverged: {} vs {}",
            cloned.confidence,
            d.confidence
        );
    }

    #[test]
    fn pixel_format_discriminants_are_stable_for_ffi() {
        // Sparrow Local relies on these numeric values across the FFI boundary.
        // Changing them silently would break Windows BGRA decode paths.
        assert_eq!(PixelFormat::Rgb as u32, 0);
        assert_eq!(PixelFormat::Rgba as u32, 1);
        assert_eq!(PixelFormat::Bgra as u32, 2);
        assert_eq!(PixelFormat::Bgr as u32, 3);
    }

    #[test]
    fn audio_segment_partial_eq_and_clone() {
        let a = AudioSegment {
            start_time_s: 0.0,
            end_time_s: 3.0,
            confidence: 0.9,
            classes: vec![AudioClass {
                class_idx: 0,
                label: Some("bird".to_string()),
                probability: 0.9,
            }],
        };
        let b = a.clone();
        assert_eq!(a, b);
    }
}
