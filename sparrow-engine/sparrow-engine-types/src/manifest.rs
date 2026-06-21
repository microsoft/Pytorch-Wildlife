//! TOML manifest parsing and validation for model and pipeline manifests.
//!
//! Model manifests drive preprocessing, inference, and postprocessing.
//! Pipeline manifests define multi-model workflows (detect → classify).
//!
//! All file paths in manifests are relative to the manifest directory.

use std::collections::BTreeMap;
use std::path::{Component, Path};

use serde::{Deserialize, Serialize};

use crate::drift_metrics::DriftReference;
use crate::error::{Result, SparrowEngineError};
use crate::types::ModelSubtype;

// ---------------------------------------------------------------------------
// Public enums
// ---------------------------------------------------------------------------

/// Preprocessing method: how input is transformed before inference.
#[derive(Debug, Clone, PartialEq)]
pub enum PreprocessMethod {
    /// Resize preserving aspect ratio, pad to target size with `pad_value`.
    Letterbox,
    /// Direct resize to target size (distorts aspect ratio).
    Resize,
    /// Mel spectrogram for audio models.
    MelSpectrogram {
        sample_rate: u32,
        n_fft: u32,
        hop_length: u32,
        n_mels: u32,
        fmin: f32,
        fmax: f32,
        top_db: f32,
        window: String,
        mel_scale: String,
        filter_norm: String,
        /// Opt-in high-frequency mel-band fill for upsampled inputs.
        ///
        /// When `true` AND the engine resampled the input upward (orig_sr <
        /// `sample_rate`), the engine replaces mel bins whose center
        /// frequency lies above `orig_sr/2 - 2500 Hz` with the 10th-percentile
        /// dB value of the valid (below-boundary) bins, then clamps the whole
        /// spectrogram to `[-top_db, +20.0]`. Mirrors PytorchWildlife
        /// `compute_mel_spectrograms_gpu(fill_highfreq=True, ...)` (RP-27,
        /// 2026-06-01). Default `false` preserves md-audiobirds-v1 behavior.
        fill_highfreq: bool,
    },
    /// Raw audio windowing for audio models whose mel front-end is in-graph
    /// (e.g., Perch 2). Decode + resample to `sample_rate`, then slice into
    /// fixed-size `window_samples`-long windows (no STFT, no filterbank).
    RawAudio {
        sample_rate: u32,
        window_samples: u32,
        /// Opt-in: when true, engine passes a second ONNX input
        /// `orig_sample_rate [1] int64` carrying the original (pre-resample)
        /// sample rate. Used by in-graph fill_highfreq passes that need to
        /// know whether the audio was upsampled and where the original
        /// Nyquist sat. Default false preserves Perch 2 / single-input
        /// RawAudio behavior (RP-27 Part 2, 2026-06-05).
        pass_orig_sample_rate: bool,
    },
}

/// Tensor layout expected by the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    /// Batch × Channels × Height × Width.
    Nchw,
    /// Batch × Height × Width × Channels.
    Nhwc,
}

/// Normalization applied to pixel values after resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Normalization {
    /// Scale to [0, 1] (divide by 255).
    Unit,
    /// ImageNet mean/std normalization.
    Imagenet,
    /// No normalization (raw 0–255).
    None,
}

/// Channel order expected by the model on the input tensor.
///
/// Models trained via Ultralytics (YOLOv5/v8/v10 family — MDv6, DeepFaune)
/// expect **BGR** because OpenCV's default is BGR. Models trained via
/// torchvision / classic CNN pipelines expect **RGB**. Bongo decodes images
/// to RGB internally; when `Bgr` is specified, the channels are swapped
/// before tensor construction.
///
/// Default: `Rgb` (preserves pre-3.8 sparrow-engine behaviour for manifests without
/// the field). YOLO-family manifests should explicitly set `channel_order = "bgr"`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ChannelOrder {
    /// R, G, B — torchvision / classic CNN convention.
    #[default]
    Rgb,
    /// B, G, R — OpenCV / Ultralytics convention.
    Bgr,
}

/// Inference precision: tensor data type used inside the ONNX graph.
///
/// FP32 is the default (preserves pre-3.8 behaviour). FP16 requires:
/// - A FP16-converted ONNX model file specified via `[model] file_fp16`
/// - Tensor Cores on the GPU (sm_80+ for fast FP16; sm_75 RTX 20-series works
///   but slower; pre-Volta has no Tensor Cores and FP16 may be slower than FP32)
///
/// ORT's `transformers.float16` converter with `keep_io_types=True` keeps the
/// model's input/output as FP32, so sparrow-engine's preprocess + postprocess code is
/// unchanged when switching precision — only the model file differs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Precision {
    /// 32-bit float — sparrow-engine's default.
    #[default]
    Fp32,
    /// 16-bit float — ~1.7x faster on Tensor Cores, ≤0.5% IoU drop.
    Fp16,
    /// 8-bit integer (quantized). TFLite/LiteRT (mobile) only — precision is
    /// baked into the single `.tflite` file, so the mobile loader uses `file`
    /// directly. Rejected for `format = "onnx"` (cpu/gpu) at manifest parse.
    Int8,
}

/// TensorRT inference precision for the GPU flavor's per-model TRT opt-in.
///
/// This is independent from `[inference] precision`: the existing precision
/// selects the model artifact (`file` vs `file_fp16`), while this value configures
/// TensorRT engine building for manifests that opt into `[inference.trt]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TrtPrecision {
    /// Build a 32-bit float TensorRT engine.
    Fp32,
    /// Build a 16-bit float TensorRT engine.
    #[default]
    Fp16,
    /// Build an 8-bit integer TensorRT engine.
    Int8,
}

/// Inference strategy: single-shot, tiled, or sliding window.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InferenceStrategy {
    /// One `session.run()` on the full preprocessed image.
    Single,
    /// Split image into tiles, run each, aggregate outputs.
    Tiled {
        tile_size: [u32; 2],
        tile_overlap: u32,
    },
    /// Sliding window over audio segments.
    SlidingWindow {
        segment_duration_s: f32,
        segment_stride_s: f32,
    },
}

/// Optional `[inference.trt]` TensorRT settings for the GPU flavor.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrtConfig {
    /// Opt into TensorRT for this model. Missing field defaults to false.
    #[serde(default)]
    pub enabled: bool,
    /// TensorRT builder precision. Missing field defaults to FP16.
    #[serde(default)]
    pub precision: TrtPrecision,
    /// TensorRT builder optimization level. Valid range is 1..=5.
    #[serde(default = "default_trt_builder_optimization_level")]
    pub builder_optimization_level: u8,
    /// False = SM-specific engine; true = SM-portable hardware-compatible mode.
    #[serde(default)]
    pub engine_hw_compatible: bool,
    /// Minimum dynamic-shape profile dimensions, keyed by ONNX input tensor name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_min: Option<BTreeMap<String, Vec<i64>>>,
    /// Optimal dynamic-shape profile dimensions, keyed by ONNX input tensor name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_opt: Option<BTreeMap<String, Vec<i64>>>,
    /// Maximum dynamic-shape profile dimensions, keyed by ONNX input tensor name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub profile_max: Option<BTreeMap<String, Vec<i64>>>,
}

/// Postprocessing method: how raw model output becomes detections/classifications.
#[derive(Debug, Clone, PartialEq)]
pub enum PostprocessMethod {
    /// YOLO end-to-end (NMS in ONNX graph). Confidence filter + bbox normalization.
    YoloE2e,
    /// MegaDetector v5a. Confidence filter + class scoring + bbox normalization.
    MegadetV5a {
        /// IoU threshold for non-max suppression (NMS not in graph for v5).
        iou_threshold: f32,
    },
    /// Heatmap peak finding (HerdNet). Point-to-box conversion.
    HeatmapPeaks {
        peak_threshold: f32,
        adaptive: bool,
        point_to_box_half_size: u32,
    },
    /// Softmax → argmax → label lookup (classifiers).
    Softmax,
    /// Sigmoid activation for binary audio detection.
    Sigmoid { confidence_threshold: f32 },
}

/// Label file format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LabelFormat {
    /// One label name per line. Line number (0-based) = label ID.
    OnePerLine,
    /// Each line: `name,index` (e.g., `animal,0`).
    NameIndexCsv,
    /// Each line: `index,name` (e.g., `0,animal`).
    IndexNameCsv,
}

impl PreprocessMethod {
    /// Return a static string name for error messages and diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            PreprocessMethod::Letterbox => "letterbox",
            PreprocessMethod::Resize => "resize",
            PreprocessMethod::MelSpectrogram { .. } => "mel_spectrogram",
            PreprocessMethod::RawAudio { .. } => "raw_audio",
        }
    }

    /// True for any audio preprocessing method. Used to gate manifest field
    /// requirements (image fields like `input_size`/`layout`/`normalization`
    /// are not required for audio models).
    pub fn is_audio(&self) -> bool {
        matches!(
            self,
            PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. }
        )
    }
}

impl PostprocessMethod {
    /// Return a static string name for error messages and diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            PostprocessMethod::YoloE2e => "yolo_e2e",
            PostprocessMethod::MegadetV5a { .. } => "megadet_v5a",
            PostprocessMethod::HeatmapPeaks { .. } => "heatmap_peaks",
            PostprocessMethod::Softmax => "softmax",
            PostprocessMethod::Sigmoid { .. } => "sigmoid",
        }
    }
}

// ---------------------------------------------------------------------------
// Public structs — parsed and validated manifest data
// ---------------------------------------------------------------------------

/// A fully validated model manifest.
#[derive(Debug, Clone)]
pub struct ModelManifest {
    pub id: String,
    pub format: String,
    pub model_file: String,

    pub preprocess_method: PreprocessMethod,
    /// Image-only: target [width, height]. None for audio models.
    pub input_size: Option<[u32; 2]>,
    /// Image-only: tensor layout. None for audio models.
    pub layout: Option<Layout>,
    /// Image-only: pixel normalization. None for audio models.
    pub normalization: Option<Normalization>,
    /// Image-only: letterbox pad value. None for audio models.
    pub pad_value: Option<f32>,
    /// Image-only: channel order expected by the model. None for audio models.
    /// Defaults to `Rgb` (preserves pre-3.8 behaviour) when manifest field absent.
    pub channel_order: Option<ChannelOrder>,

    /// Inference precision: FP32 (default) or FP16. When `Fp16`, the engine
    /// loads `model_file_fp16` instead of `model_file`. Phase 3.8 fix.
    pub precision: Precision,
    /// Optional path to FP16-converted ONNX file (relative to manifest dir).
    /// Required when `precision = Fp16`. Created via `tools/convert_fp16.py`.
    pub model_file_fp16: Option<String>,

    pub inference_strategy: InferenceStrategy,
    /// Optional per-model TensorRT settings from `[inference.trt]`.
    /// `None` preserves backward compatibility for manifests without the section.
    pub trt: Option<TrtConfig>,

    pub postprocess_method: PostprocessMethod,
    pub confidence_threshold: Option<f32>,

    /// Label file path (relative to manifest dir). None for binary detectors.
    pub label_file: Option<String>,
    /// Label file format. None when label_file is None.
    pub label_format: Option<LabelFormat>,

    /// Whether this model is the default for its type.
    pub default: bool,

    /// Rendering / behaviour hint from `[model].subtype`.
    ///
    /// `Standard` (default, bbox rendering) when absent; `Overhead` triggers
    /// centroid-dot rendering in `viz::render` (MT-9 fix, Phase 3.5 S3).
    /// Backward-compatible: missing field → `Standard`.
    pub subtype: ModelSubtype,

    pub onnx_sha256: Option<String>,
    pub onnx_size_bytes: Option<u64>,
    pub version: Option<String>,
    pub description: Option<String>,

    /// Optional `[provenance]` section. None when the manifest omits it.
    /// Phase 4 — sparrow-engine round-trips these values for sibling-repo joins
    /// without interpreting them.
    pub provenance: Option<ProvenanceRecord>,

    /// Optional `[drift_reference]` section (Phase 4 W4). Reference class
    /// distribution against which per-request `DriftMetrics::class_distribution_psi`
    /// is computed. `None` ⇒ PSI is `None` in every request's drift snapshot.
    pub drift_reference: Option<DriftReference>,
}

/// A single step in a pipeline.
#[derive(Debug, Clone)]
pub struct PipelineStep {
    pub role: PipelineRole,
    pub model: String,
}

/// Role of a pipeline step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineRole {
    Detector,
    Classifier,
}

/// A fully validated pipeline manifest.
#[derive(Debug, Clone)]
pub struct PipelineManifest {
    pub id: String,
    pub steps: Vec<PipelineStep>,
}

/// Optional `[provenance]` section on a model manifest — three pointer fields
/// that link a deployed sparrow-engine model back to its training artefacts in the
/// (eventual) sibling repos `bongo-fine-tuning` and `sparrow-data`.
///
/// Phase 4 (Phase 3.7 Track A folded the v4-era "Phase 5a" pointer fields here,
/// 2026-04-30 user directive). Bongo only round-trips these fields; the values
/// are opaque to the engine. They surface in `InferenceLogRecord` (W2) so
/// downstream `sparrow-data` can join inference rows to training artefacts
/// without sparrow-engine gaining any sibling-repo coupling.
///
/// All fields are optional. Manifests without `[provenance]` load unchanged.
/// `Serialize` is derived because `InferenceLogRecord` embeds the same struct
/// on the wire — keeps a single canonical type, no parallel definitions.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceRecord {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_dataset_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_experiment_id: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub training_repo_commit: Option<String>,
}

// ---------------------------------------------------------------------------
// Raw TOML deserialization types (private)
// ---------------------------------------------------------------------------

/// Top-level raw TOML for model manifests.
#[derive(Deserialize)]
struct RawModelToml {
    model: RawModel,
    preprocessing: RawPreprocessing,
    inference: RawInference,
    postprocessing: RawPostprocessing,
    /// Optional: binary detectors (e.g., audio bird detector) have no labels.
    labels: Option<RawLabels>,
    /// Optional `[provenance]` pointer fields (Phase 4).
    #[serde(default)]
    provenance: Option<RawProvenance>,
    /// Optional `[drift_reference]` section (Phase 4 W4).
    #[serde(default)]
    drift_reference: Option<RawDriftReference>,
}

/// Raw TOML mirror of `DriftReference`. Inline `class_distribution` map
/// stays as `BTreeMap<String, f32>` so the parser preserves operator-supplied
/// frequency values exactly (no rescaling).
#[derive(Deserialize, Default)]
struct RawDriftReference {
    #[serde(default)]
    class_distribution: std::collections::BTreeMap<String, f32>,
}

/// Raw TOML mirror of `ProvenanceRecord`. Each field is `#[serde(default)]`
/// so missing entries become `None` instead of failing the parse.
#[derive(Deserialize, Default)]
struct RawProvenance {
    #[serde(default)]
    training_dataset_id: Option<String>,
    #[serde(default)]
    training_experiment_id: Option<String>,
    #[serde(default)]
    training_repo_commit: Option<String>,
}

#[derive(Deserialize)]
struct RawModel {
    id: String,
    format: String,
    file: String,
    /// Optional FP16-converted ONNX file (Phase 3.8). Used when
    /// `[inference] precision = "fp16"`.
    #[serde(default)]
    file_fp16: Option<String>,
    #[serde(default)]
    default: bool,
    /// Rendering / behaviour hint. Accepts "standard" | "overhead". Missing
    /// field defaults to "standard" for backward compatibility with pre-3.5
    /// manifests. Added in Phase 3.5 S3 (item #3, MT-9 fix).
    #[serde(default)]
    subtype: Option<String>,
    #[serde(default)]
    onnx_sha256: Option<String>,
    #[serde(default)]
    onnx_size_bytes: Option<u64>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    description: Option<String>,
}

#[derive(Deserialize)]
struct RawPreprocessing {
    method: String,
    // Image-specific fields (required for vision, absent for audio).
    input_size: Option<[u32; 2]>,
    layout: Option<String>,
    normalization: Option<String>,
    #[serde(default)]
    pad_value: Option<f32>,
    /// Channel order: "rgb" (default) | "bgr". Phase 3.8 fix for YOLO-family
    /// models trained via Ultralytics (which use BGR per cv2 default).
    #[serde(default)]
    channel_order: Option<String>,
    // Audio-specific fields (required for mel_spectrogram, absent for vision).
    sample_rate: Option<u32>,
    n_fft: Option<u32>,
    hop_length: Option<u32>,
    n_mels: Option<u32>,
    fmin: Option<f32>,
    fmax: Option<f32>,
    top_db: Option<f32>,
    window: Option<String>,
    mel_scale: Option<String>,
    filter_norm: Option<String>,
    // Raw-audio-specific fields (required for raw_audio).
    /// Number of samples per inference window (= segment_duration_s × sample_rate).
    /// Required for `raw_audio`. For Perch 2: 160000 = 5 s × 32 kHz.
    window_samples: Option<u32>,
    /// RawAudio-only opt-in (RP-27 Part 2, 2026-06-05): when true, the
    /// engine passes a second ONNX input `orig_sample_rate [1] int64`
    /// alongside the audio tensor so the model can apply in-graph
    /// fill_highfreq.
    #[serde(default)]
    pass_orig_sample_rate: Option<bool>,
    /// Opt-in high-frequency fill for mel_spectrogram preprocess (RP-27).
    /// Defaults to `false` (md-audiobirds-v1 behavior). When `true` and the
    /// engine resamples upward, mel bins above `orig_sr/2 - 2500 Hz` are
    /// replaced with the 10th-percentile dB of valid bins. Ignored for
    /// non-mel preprocess methods.
    #[serde(default)]
    fill_highfreq: Option<bool>,
}

#[derive(Deserialize)]
struct RawInference {
    strategy: String,
    /// Inference precision: "fp32" (default) | "fp16". Phase 3.8.
    #[serde(default)]
    precision: Option<String>,
    // Tiled fields.
    tile_size: Option<[u32; 2]>,
    tile_overlap: Option<u32>,
    // Sliding window fields.
    segment_duration_s: Option<f32>,
    segment_stride_s: Option<f32>,
    /// Optional `[inference.trt]` nested table.
    #[serde(default)]
    trt: Option<TrtConfig>,
}

#[derive(Deserialize)]
struct RawPostprocessing {
    method: String,
    confidence_threshold: Option<f32>,
    iou_threshold: Option<f32>,
    peak_threshold: Option<f32>,
    adaptive: Option<bool>,
    point_to_box_half_size: Option<u32>,
}

#[derive(Deserialize)]
struct RawLabels {
    file: String,
    format: String,
}

/// Top-level raw TOML for pipeline manifests.
#[derive(Deserialize)]
struct RawPipelineToml {
    pipeline: RawPipeline,
    /// Present if this is actually a model manifest (used for discrimination).
    model: Option<toml::Value>,
}

#[derive(Deserialize)]
struct RawPipeline {
    id: String,
    steps: Vec<RawPipelineStep>,
}

#[derive(Deserialize)]
struct RawPipelineStep {
    role: String,
    model: String,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Parse and validate a model manifest from a TOML file.
///
/// # Errors
/// - `ManifestNotFound` if the file doesn't exist
/// - `TomlParse` if the TOML is malformed
/// - `WrongManifestType` if the file contains a `[pipeline]` section
/// - `UnsupportedFormat` if `format` is not "onnx" or "tflite"
/// - `MissingTiledFields` if `strategy = "tiled"` but `tile_size`/`tile_overlap` missing
/// - `PathTraversal` if any file path contains `..` components or is absolute
/// - `InvalidManifest` for other validation failures
pub fn load_manifest(path: &Path) -> Result<ModelManifest> {
    if !path.exists() {
        return Err(SparrowEngineError::ManifestNotFound(path.to_path_buf()));
    }

    let content = std::fs::read_to_string(path)?;

    // Discrimination: check for [pipeline] section before strict model parse.
    // A pipeline manifest won't parse as RawModelToml (missing [model]), so
    // check via loose Table parse first.
    if let Ok(table) = content.parse::<toml::Table>() {
        if table.contains_key("pipeline") {
            return Err(SparrowEngineError::WrongManifestType);
        }
    }

    let raw: RawModelToml = toml::from_str(&content)?;

    // -- Validate format. Both ONNX (cpu/gpu ORT flavors) and TFLite (mobile
    // LiteRT flavor) are accepted at this shared layer; each flavor's engine
    // rejects formats its own backend cannot load (flavor-strict — mirrors the
    // Device::Auto coercion contract). This keeps one manifest schema across all
    // flavors while preventing, e.g., an ORT engine from trying to load a .tflite.
    if raw.model.format != "onnx" && raw.model.format != "tflite" {
        return Err(SparrowEngineError::UnsupportedFormat {
            format: raw.model.format,
        });
    }

    // -- Validate id and file are non-empty --
    if raw.model.id.is_empty() {
        return Err(SparrowEngineError::InvalidManifest(
            "model id must not be empty".to_string(),
        ));
    }
    if raw.model.file.is_empty() {
        return Err(SparrowEngineError::InvalidManifest(
            "model file must not be empty".to_string(),
        ));
    }

    // -- Parse preprocessing --
    let is_audio = matches!(
        raw.preprocessing.method.as_str(),
        "mel_spectrogram" | "raw_audio"
    );

    let preprocess_method = match raw.preprocessing.method.as_str() {
        "letterbox" => PreprocessMethod::Letterbox,
        "resize" => PreprocessMethod::Resize,
        "raw_audio" => {
            let raw_err = |name: &str| {
                SparrowEngineError::InvalidManifest(format!("raw_audio requires '{name}' field"))
            };
            PreprocessMethod::RawAudio {
                sample_rate: raw
                    .preprocessing
                    .sample_rate
                    .ok_or_else(|| raw_err("sample_rate"))?,
                window_samples: raw
                    .preprocessing
                    .window_samples
                    .ok_or_else(|| raw_err("window_samples"))?,
                pass_orig_sample_rate: raw.preprocessing.pass_orig_sample_rate.unwrap_or(false),
            }
        }
        "mel_spectrogram" => {
            let mel_err = |name: &str| {
                SparrowEngineError::InvalidManifest(format!(
                    "mel_spectrogram requires '{name}' field"
                ))
            };
            PreprocessMethod::MelSpectrogram {
                sample_rate: raw
                    .preprocessing
                    .sample_rate
                    .ok_or_else(|| mel_err("sample_rate"))?,
                n_fft: raw.preprocessing.n_fft.ok_or_else(|| mel_err("n_fft"))?,
                hop_length: raw
                    .preprocessing
                    .hop_length
                    .ok_or_else(|| mel_err("hop_length"))?,
                n_mels: raw.preprocessing.n_mels.ok_or_else(|| mel_err("n_mels"))?,
                fmin: raw.preprocessing.fmin.ok_or_else(|| mel_err("fmin"))?,
                fmax: raw.preprocessing.fmax.ok_or_else(|| mel_err("fmax"))?,
                top_db: raw.preprocessing.top_db.ok_or_else(|| mel_err("top_db"))?,
                window: raw.preprocessing.window.ok_or_else(|| mel_err("window"))?,
                mel_scale: raw
                    .preprocessing
                    .mel_scale
                    .ok_or_else(|| mel_err("mel_scale"))?,
                filter_norm: raw
                    .preprocessing
                    .filter_norm
                    .ok_or_else(|| mel_err("filter_norm"))?,
                fill_highfreq: raw.preprocessing.fill_highfreq.unwrap_or(false),
            }
        }
        other => {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "Unknown preprocessing method: '{other}'"
            )))
        }
    };

    // -- Validate audio numeric fields (prevent division by zero in DSP pipeline) --
    if let PreprocessMethod::MelSpectrogram {
        sample_rate,
        n_fft,
        hop_length,
        n_mels,
        fmin,
        fmax,
        window,
        mel_scale,
        filter_norm,
        ..
    } = &preprocess_method
    {
        if *sample_rate == 0 {
            return Err(SparrowEngineError::InvalidManifest(
                "sample_rate must be > 0".to_string(),
            ));
        }
        if *n_fft < 2 {
            return Err(SparrowEngineError::InvalidManifest(
                "n_fft must be >= 2".to_string(),
            ));
        }
        if !n_fft.is_power_of_two() {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "n_fft must be a power of 2 (got {n_fft}); realfft requires power-of-2 input"
            )));
        }
        if *hop_length == 0 {
            return Err(SparrowEngineError::InvalidManifest(
                "hop_length must be > 0".to_string(),
            ));
        }
        if *n_mels == 0 {
            return Err(SparrowEngineError::InvalidManifest(
                "n_mels must be > 0".to_string(),
            ));
        }
        if fmax <= fmin {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "fmax ({fmax}) must be > fmin ({fmin})"
            )));
        }
        let nyquist = *sample_rate as f32 / 2.0;
        if *fmax > nyquist {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "fmax ({fmax}) exceeds Nyquist frequency ({nyquist}) for sample_rate {sample_rate}"
            )));
        }
        // Validate DSP algorithm fields: only supported values are accepted.
        // The preprocessing hardcodes symmetric Hann window, Slaney mel scale,
        // and Slaney filter normalization. Reject anything else to prevent
        // silent correctness bugs where the manifest specifies an algorithm
        // but preprocessing ignores it.
        //
        // Phase 3.8 Step 2 Wave 0a (F0.8 corrective fix, 2026-05-04): switched
        // accepted values from "htk" + "area" to "slaney" + "slaney" to match
        // `MD_AudioBirds_V1` training (PW Bioacoustics). Loading an old
        // manifest with the pre-fix values fails parsing — a deliberate
        // tripwire so any out-of-tree manifest copies are caught at load time.
        if window != "hann_symmetric" {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "unsupported window '{}'; only 'hann_symmetric' is implemented",
                window
            )));
        }
        if mel_scale != "slaney" {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "unsupported mel_scale '{}'; only 'slaney' is implemented \
                 (phase 3.8 step 2 wave 0a switched from 'htk' to 'slaney' \
                 to match MD_AudioBirds_V1 training; update the manifest)",
                mel_scale
            )));
        }
        if filter_norm != "slaney" {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "unsupported filter_norm '{}'; only 'slaney' is implemented \
                 (phase 3.8 step 2 wave 0a switched from 'area' to 'slaney' \
                 to match MD_AudioBirds_V1 training; update the manifest)",
                filter_norm
            )));
        }
    }

    // -- Validate raw-audio numeric fields --
    if let PreprocessMethod::RawAudio {
        sample_rate,
        window_samples,
        ..
    } = &preprocess_method
    {
        if *sample_rate == 0 {
            return Err(SparrowEngineError::InvalidManifest(
                "sample_rate must be > 0".to_string(),
            ));
        }
        if *window_samples == 0 {
            return Err(SparrowEngineError::InvalidManifest(
                "window_samples must be > 0".to_string(),
            ));
        }
    }

    // -- Parse image-specific fields (required for vision, absent for audio) --
    let (input_size, layout, normalization, pad_value, channel_order) = if is_audio {
        (None, None, None, None, None)
    } else {
        let input_size = raw.preprocessing.input_size.ok_or_else(|| {
            SparrowEngineError::InvalidManifest(
                "image models require 'input_size' field".to_string(),
            )
        })?;
        let layout_str = raw.preprocessing.layout.as_deref().ok_or_else(|| {
            SparrowEngineError::InvalidManifest("image models require 'layout' field".to_string())
        })?;
        let norm_str = raw.preprocessing.normalization.as_deref().ok_or_else(|| {
            SparrowEngineError::InvalidManifest(
                "image models require 'normalization' field".to_string(),
            )
        })?;

        let layout = match layout_str {
            "nchw" => Layout::Nchw,
            // NHWC is rejected for ONNX models: ORT CUDA EP has known SafeInt
            // overflow bugs in Conv with NHWC + dynamic shapes (ORT issues
            // #27912, #12288). TFLite models, by contrast, are natively NHWC
            // (the channels-last TensorFlow convention) and the mobile flavor's
            // LiteRT backend consumes NHWC directly — so NHWC is permitted for
            // `format = "tflite"`. Convert ONNX inputs with
            // `python -m tf2onnx.convert --inputs-as-nchw <input> ...` or
            // `onnx-simplifier` before onboarding.
            "nhwc" if raw.model.format == "tflite" => Layout::Nhwc,
            "nhwc" => {
                return Err(SparrowEngineError::InvalidManifest(
                    "layout 'nhwc' is not supported for ONNX models: ORT requires NCHW. \
                     Convert with `tf2onnx --inputs-as-nchw` or onnx-simplifier before \
                     onboarding. See ORT issues #27912 / #12288 for the NHWC Conv bug. \
                     (NHWC is permitted only for format = \"tflite\".)"
                        .to_string(),
                ))
            }
            other => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "Unknown layout: '{other}' (expected 'nchw' or 'nhwc')"
                )))
            }
        };

        let normalization = match norm_str {
            "unit" => Normalization::Unit,
            "imagenet" => Normalization::Imagenet,
            "none" => Normalization::None,
            other => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "Unknown normalization: '{other}'"
                )))
            }
        };

        // Validate input_size > 0.
        if input_size[0] == 0 || input_size[1] == 0 {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "input_size dimensions must be > 0, got {:?}",
                input_size
            )));
        }

        // Channel order: optional, defaults to RGB (preserves pre-3.8 behaviour
        // for manifests without the field).
        let channel_order = match raw.preprocessing.channel_order.as_deref() {
            None => ChannelOrder::Rgb,
            Some("rgb") => ChannelOrder::Rgb,
            Some("bgr") => ChannelOrder::Bgr,
            Some(other) => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "Unknown channel_order: '{other}' (expected 'rgb' or 'bgr')"
                )))
            }
        };

        (
            Some(input_size),
            Some(layout),
            Some(normalization),
            Some(raw.preprocessing.pad_value.unwrap_or(0.0)),
            Some(channel_order),
        )
    };

    // -- Parse inference strategy --
    let inference_strategy = match raw.inference.strategy.as_str() {
        "single" => InferenceStrategy::Single,
        "tiled" => {
            let tile_size = raw
                .inference
                .tile_size
                .ok_or(SparrowEngineError::MissingTiledFields)?;
            let tile_overlap = raw
                .inference
                .tile_overlap
                .ok_or(SparrowEngineError::MissingTiledFields)?;
            InferenceStrategy::Tiled {
                tile_size,
                tile_overlap,
            }
        }
        "sliding_window" => {
            let segment_duration_s = raw.inference.segment_duration_s.ok_or_else(|| {
                SparrowEngineError::InvalidManifest(
                    "sliding_window requires 'segment_duration_s' field".to_string(),
                )
            })?;
            let segment_stride_s = raw.inference.segment_stride_s.ok_or_else(|| {
                SparrowEngineError::InvalidManifest(
                    "sliding_window requires 'segment_stride_s' field".to_string(),
                )
            })?;
            if !segment_duration_s.is_finite() || segment_duration_s <= 0.0 {
                return Err(SparrowEngineError::InvalidManifest(
                    "segment_duration_s must be finite and > 0".to_string(),
                ));
            }
            if !segment_stride_s.is_finite() || segment_stride_s <= 0.0 {
                return Err(SparrowEngineError::InvalidManifest(
                    "segment_stride_s must be finite and > 0".to_string(),
                ));
            }
            InferenceStrategy::SlidingWindow {
                segment_duration_s,
                segment_stride_s,
            }
        }
        other => {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "Unknown inference strategy: '{other}'"
            )))
        }
    };

    if is_audio && !matches!(inference_strategy, InferenceStrategy::SlidingWindow { .. }) {
        return Err(SparrowEngineError::InvalidManifest(
            "audio preprocessing requires inference strategy 'sliding_window'".to_string(),
        ));
    }

    if let (
        PreprocessMethod::RawAudio {
            sample_rate,
            window_samples,
            ..
        },
        InferenceStrategy::SlidingWindow {
            segment_duration_s, ..
        },
    ) = (&preprocess_method, inference_strategy)
    {
        let expected = (segment_duration_s * (*sample_rate as f32)).round() as i64;
        let actual = *window_samples as i64;
        if (expected - actual).abs() > 1 {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "window_samples ({actual}) does not match segment_duration_s × sample_rate \
                 ({segment_duration_s} × {sample_rate} = {expected}); allowed tolerance is ±1 sample"
            )));
        }
    }

    // -- Parse precision (Phase 3.8: FP16 support; int8 is tflite/mobile-only) --
    let precision = match raw.inference.precision.as_deref() {
        None | Some("fp32") => Precision::Fp32,
        Some("fp16") => Precision::Fp16,
        Some("int8") => Precision::Int8,
        Some(other) => {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "Unknown precision: '{other}' (expected 'fp32', 'fp16', or 'int8')"
            )))
        }
    };
    // int8 is supported only by the mobile TFLite/LiteRT flavor (the .tflite bakes
    // the quantization into its single `file`). ONNX (cpu/gpu) has no int8 path
    // wired, so reject int8 + onnx at parse rather than fail obscurely at load.
    if raw.model.format == "onnx" && precision == Precision::Int8 {
        return Err(SparrowEngineError::InvalidManifest(
            "precision = 'int8' is only supported for tflite (mobile) models, not onnx".to_string(),
        ));
    }
    // ONNX fp16 uses a separate fp16-converted file (`file_fp16`), with `file`
    // holding the fp32 original (Phase 3.8). TFLite artifacts bake precision into
    // the single `file` (there is no fp32/fp16 file pair), so the mobile LiteRT
    // flavor loads `file` directly and does not require `file_fp16`.
    if raw.model.format == "onnx" && precision == Precision::Fp16 && raw.model.file_fp16.is_none() {
        return Err(SparrowEngineError::InvalidManifest(
            "precision = 'fp16' requires [model] file_fp16 to be set".to_string(),
        ));
    }
    if let Some(fp16_path) = &raw.model.file_fp16 {
        reject_unsafe_path(fp16_path, "fp16 model file")?;
    }

    let trt = raw.inference.trt;
    validate_trt_config(&trt)?;

    // -- Parse postprocessing method --
    let postprocess_method = match raw.postprocessing.method.as_str() {
        "yolo_e2e" => PostprocessMethod::YoloE2e,
        "megadet_v5a" => {
            let iou_threshold = raw.postprocessing.iou_threshold.unwrap_or(0.45);
            if !iou_threshold.is_finite() || !(0.0..=1.0).contains(&iou_threshold) {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "megadet_v5a iou_threshold must be finite and in [0.0, 1.0], got {iou_threshold}"
                )));
            }
            PostprocessMethod::MegadetV5a { iou_threshold }
        }
        "heatmap_peaks" => {
            let peak_threshold = raw.postprocessing.peak_threshold.ok_or_else(|| {
                SparrowEngineError::InvalidManifest(
                    "heatmap_peaks requires 'peak_threshold' field".to_string(),
                )
            })?;
            let adaptive = raw.postprocessing.adaptive.ok_or_else(|| {
                SparrowEngineError::InvalidManifest(
                    "heatmap_peaks requires 'adaptive' field".to_string(),
                )
            })?;
            let point_to_box_half_size =
                raw.postprocessing.point_to_box_half_size.ok_or_else(|| {
                    SparrowEngineError::InvalidManifest(
                        "heatmap_peaks requires 'point_to_box_half_size' field".to_string(),
                    )
                })?;
            PostprocessMethod::HeatmapPeaks {
                peak_threshold,
                adaptive,
                point_to_box_half_size,
            }
        }
        "softmax" => PostprocessMethod::Softmax,
        "sigmoid" => {
            let confidence_threshold =
                raw.postprocessing.confidence_threshold.ok_or_else(|| {
                    SparrowEngineError::InvalidManifest(
                        "sigmoid requires 'confidence_threshold' field".to_string(),
                    )
                })?;
            PostprocessMethod::Sigmoid {
                confidence_threshold,
            }
        }
        other => {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "Unknown postprocessing method: '{other}'"
            )))
        }
    };

    if is_audio {
        match (&preprocess_method, &postprocess_method) {
            (PreprocessMethod::MelSpectrogram { .. }, PostprocessMethod::Sigmoid { .. })
            | (PreprocessMethod::MelSpectrogram { .. }, PostprocessMethod::Softmax)
            | (PreprocessMethod::RawAudio { .. }, PostprocessMethod::Softmax) => {}
            _ => {
                return Err(SparrowEngineError::InvalidManifest(format!(
                    "unsupported audio preprocess/postprocess combination: preprocessing method '{}' with postprocessing method '{}'",
                    raw.preprocessing.method,
                    raw.postprocessing.method
                )));
            }
        }
    }

    // -- Parse labels (optional for binary detectors) --
    let (label_file, label_format) = if let Some(ref labels) = raw.labels {
        let fmt = parse_label_format(&labels.format)?;
        (Some(labels.file.clone()), Some(fmt))
    } else {
        (None, None)
    };

    // -- Validate tile dimensions when tiled --
    if let InferenceStrategy::Tiled {
        tile_size,
        tile_overlap,
    } = inference_strategy
    {
        if tile_size[0] == 0 || tile_size[1] == 0 {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "tile_size dimensions must be > 0, got {:?}",
                tile_size
            )));
        }
        let min_tile_dim = tile_size[0].min(tile_size[1]);
        if tile_overlap >= min_tile_dim {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "tile_overlap ({tile_overlap}) must be < min(tile_size) ({min_tile_dim})"
            )));
        }
    }

    // -- H1: tiled + heatmap_peaks requires tile_size == input_size --
    if let InferenceStrategy::Tiled { tile_size, .. } = inference_strategy {
        if matches!(postprocess_method, PostprocessMethod::HeatmapPeaks { .. })
            && Some(tile_size) != input_size
        {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "tiled + heatmap_peaks requires tile_size == input_size, got tile_size={:?} input_size={:?}",
                tile_size, input_size
            )));
        }
    }

    // -- H2: yolo_e2e and megadet_v5a require letterbox --
    if matches!(
        postprocess_method,
        PostprocessMethod::YoloE2e | PostprocessMethod::MegadetV5a { .. }
    ) && preprocess_method != PreprocessMethod::Letterbox
    {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "postprocessing method '{}' requires preprocessing method 'letterbox'",
            raw.postprocessing.method
        )));
    }

    // -- Validate paths: no traversal or absolute paths --
    reject_unsafe_path(&raw.model.file, "model file")?;
    if let Some(ref lf) = label_file {
        reject_unsafe_path(lf, "label file")?;
    }

    // -- Parse subtype (Phase 3.5 S3, MT-9 fix) --
    // Missing field → Standard (backward-compat with pre-3.5 manifests).
    let subtype = match raw.model.subtype.as_deref() {
        None => ModelSubtype::Standard,
        Some("standard") => ModelSubtype::Standard,
        Some("overhead") => ModelSubtype::Overhead,
        Some(other) => {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "Unknown model subtype: '{other}' (expected 'standard' or 'overhead')"
            )));
        }
    };

    // -- Round-trip optional [provenance] section (Phase 4 W1) --
    let provenance = raw.provenance.map(|p| ProvenanceRecord {
        training_dataset_id: p.training_dataset_id,
        training_experiment_id: p.training_experiment_id,
        training_repo_commit: p.training_repo_commit,
    });

    // -- Round-trip optional [drift_reference] section (Phase 4 W4) --
    let drift_reference = raw.drift_reference.map(|d| DriftReference {
        class_distribution: d.class_distribution,
    });

    Ok(ModelManifest {
        id: raw.model.id,
        format: raw.model.format,
        model_file: raw.model.file,
        model_file_fp16: raw.model.file_fp16,
        preprocess_method,
        input_size,
        layout,
        normalization,
        pad_value,
        channel_order,
        precision,
        inference_strategy,
        trt,
        postprocess_method,
        confidence_threshold: raw.postprocessing.confidence_threshold,
        label_file,
        label_format,
        default: raw.model.default,
        subtype,
        onnx_sha256: raw.model.onnx_sha256,
        onnx_size_bytes: raw.model.onnx_size_bytes,
        version: raw.model.version,
        description: raw.model.description,
        provenance,
        drift_reference,
    })
}

/// Parse and validate a pipeline manifest from a TOML file.
///
/// # Errors
/// - `ManifestNotFound` if the file doesn't exist
/// - `TomlParse` if the TOML is malformed
/// - `WrongPipelineType` if the file contains a `[model]` section
/// - `InvalidPipeline` if there is not exactly one detector step
pub fn load_pipeline_manifest(path: &Path) -> Result<PipelineManifest> {
    if !path.exists() {
        return Err(SparrowEngineError::ManifestNotFound(path.to_path_buf()));
    }

    let content = std::fs::read_to_string(path)?;

    // Discrimination: reject if this is a model manifest.
    if let Ok(table) = content.parse::<toml::Table>() {
        if table.contains_key("model") {
            return Err(SparrowEngineError::WrongPipelineType);
        }
    }

    let raw: RawPipelineToml = toml::from_str(&content)?;

    // Double-check discrimination via the parsed model field.
    if raw.model.is_some() {
        return Err(SparrowEngineError::WrongPipelineType);
    }

    // Parse steps.
    let mut steps = Vec::with_capacity(raw.pipeline.steps.len());
    let mut detector_count = 0u32;

    for raw_step in &raw.pipeline.steps {
        let role = match raw_step.role.as_str() {
            "detector" => {
                detector_count += 1;
                PipelineRole::Detector
            }
            "classifier" => PipelineRole::Classifier,
            other => {
                return Err(SparrowEngineError::InvalidPipeline(format!(
                    "Unknown pipeline step role: '{other}'"
                )))
            }
        };

        steps.push(PipelineStep {
            role,
            model: raw_step.model.clone(),
        });
    }

    // Validate: exactly one detector step.
    if detector_count != 1 {
        return Err(SparrowEngineError::InvalidPipeline(format!(
            "Pipeline must have exactly one detector step, found {detector_count}"
        )));
    }

    Ok(PipelineManifest {
        id: raw.pipeline.id,
        steps,
    })
}

/// Parse a label file into a Vec<String> where index = label_id.
///
/// # Formats
/// - `OnePerLine`: each line is a label name, index = line number (0-based)
/// - `NameIndexCsv`: each line is `name,index`
/// - `IndexNameCsv`: each line is `index,name`
///
/// # Errors
/// - `LabelFileNotFound` if the file doesn't exist
/// - `InvalidLabelFormat` if any line cannot be parsed
pub fn load_labels(path: &Path, format: &LabelFormat) -> Result<Vec<String>> {
    if !path.exists() {
        return Err(SparrowEngineError::LabelFileNotFound(path.to_path_buf()));
    }

    let content = std::fs::read_to_string(path)?;

    match format {
        LabelFormat::OnePerLine => {
            let labels: Vec<String> = content
                .lines()
                .filter(|line| !line.is_empty())
                .map(|line| line.trim().to_string())
                .collect();
            Ok(labels)
        }
        LabelFormat::NameIndexCsv => parse_csv_labels(&content, false, path),
        LabelFormat::IndexNameCsv => parse_csv_labels(&content, true, path),
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

fn default_trt_builder_optimization_level() -> u8 {
    3
}

fn validate_trt_config(trt: &Option<TrtConfig>) -> Result<()> {
    let Some(trt) = trt else {
        return Ok(());
    };

    if !(1..=5).contains(&trt.builder_optimization_level) {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "inference.trt.builder_optimization_level must be in 1..=5, got {}",
            trt.builder_optimization_level
        )));
    }

    if trt.profile_min.is_some() || trt.profile_opt.is_some() || trt.profile_max.is_some() {
        let Some(profile_min) = &trt.profile_min else {
            return Err(SparrowEngineError::InvalidManifest(
                "inference.trt profiles must set profile_min, profile_opt, and profile_max together"
                    .to_string(),
            ));
        };
        let Some(profile_opt) = &trt.profile_opt else {
            return Err(SparrowEngineError::InvalidManifest(
                "inference.trt profiles must set profile_min, profile_opt, and profile_max together"
                    .to_string(),
            ));
        };
        let Some(profile_max) = &trt.profile_max else {
            return Err(SparrowEngineError::InvalidManifest(
                "inference.trt profiles must set profile_min, profile_opt, and profile_max together"
                    .to_string(),
            ));
        };

        if !profile_min.keys().eq(profile_opt.keys()) || !profile_min.keys().eq(profile_max.keys())
        {
            return Err(SparrowEngineError::InvalidManifest(
                "inference.trt profile_min/profile_opt/profile_max must have identical input keys"
                    .to_string(),
            ));
        }
    }

    Ok(())
}

/// Reject paths containing parent-directory components (`..`) or absolute prefixes.
///
/// Uses `Path::components()` so filenames like `model..v2.onnx` pass cleanly.
fn reject_unsafe_path(p: &str, field: &str) -> Result<()> {
    let path = Path::new(p);

    // Reject absolute paths (Unix `/…` or Windows `C:\…`, `\\…`).
    if path.is_absolute() || p.starts_with('\\') {
        return Err(SparrowEngineError::PathTraversal(format!(
            "{field}: absolute path not allowed: '{p}'"
        )));
    }

    // Reject any `..` component (but allow `..` inside filenames).
    for component in path.components() {
        if matches!(component, Component::ParentDir) {
            return Err(SparrowEngineError::PathTraversal(format!(
                "{field}: parent directory traversal not allowed: '{p}'"
            )));
        }
    }

    Ok(())
}

/// Parse a label format string from TOML.
fn parse_label_format(s: &str) -> Result<LabelFormat> {
    match s {
        "one_per_line" => Ok(LabelFormat::OnePerLine),
        "name_index_csv" => Ok(LabelFormat::NameIndexCsv),
        "index_name_csv" => Ok(LabelFormat::IndexNameCsv),
        other => Err(SparrowEngineError::InvalidManifest(format!(
            "Unknown label format: '{other}'"
        ))),
    }
}

/// Parse CSV label files. `index_first` = true for `index,name`, false for `name,index`.
fn parse_csv_labels(content: &str, index_first: bool, path: &Path) -> Result<Vec<String>> {
    let mut entries: Vec<(usize, String)> = Vec::new();

    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let parts: Vec<&str> = line.splitn(2, ',').collect();
        if parts.len() != 2 {
            return Err(SparrowEngineError::InvalidLabelFormat(format!(
                "{}:{}: expected 'name,index' or 'index,name', got '{line}'",
                path.display(),
                line_num + 1
            )));
        }

        let (name_part, index_part) = if index_first {
            (parts[1].trim(), parts[0].trim())
        } else {
            (parts[0].trim(), parts[1].trim())
        };

        let index: usize = index_part.parse().map_err(|_| {
            SparrowEngineError::InvalidLabelFormat(format!(
                "{}:{}: cannot parse index '{index_part}' as integer",
                path.display(),
                line_num + 1
            ))
        })?;

        entries.push((index, name_part.to_string()));
    }

    if entries.is_empty() {
        return Ok(Vec::new());
    }

    // Build Vec<String> indexed by label ID.
    let max_index = entries.iter().map(|(i, _)| *i).max().unwrap_or(0);
    let mut labels = vec![String::new(); max_index + 1];

    for (index, name) in entries {
        labels[index] = name;
    }

    Ok(labels)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn write_temp_file(name: &str, content: &str) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let file_path = dir.path().join(name);
        let mut f = std::fs::File::create(&file_path).unwrap();
        f.write_all(content.as_bytes()).unwrap();
        dir
    }

    // -- Model manifest tests --

    #[test]
    fn test_load_valid_single_shot_manifest() {
        let toml = r#"
[model]
id = "megadetector-v6"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [1280, 1280]
layout = "nchw"
normalization = "unit"
pad_value = 0.447

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();

        assert_eq!(manifest.id, "megadetector-v6");
        assert_eq!(manifest.format, "onnx");
        assert_eq!(manifest.preprocess_method, PreprocessMethod::Letterbox);
        assert_eq!(manifest.input_size, Some([1280, 1280]));
        assert_eq!(manifest.layout, Some(Layout::Nchw));
        assert_eq!(manifest.normalization, Some(Normalization::Unit));
        assert!((manifest.pad_value.unwrap() - 0.447).abs() < 1e-6);
        assert_eq!(manifest.inference_strategy, InferenceStrategy::Single);
        assert!(matches!(
            manifest.postprocess_method,
            PostprocessMethod::YoloE2e
        ));
        assert_eq!(manifest.confidence_threshold, Some(0.2));
        assert_eq!(manifest.label_format, Some(LabelFormat::OnePerLine));
        assert_eq!(manifest.trt, None);
    }

    #[test]
    fn trt_config_round_trips() {
        let toml = r#"
enabled = true
precision = "fp16"
builder_optimization_level = 3
engine_hw_compatible = false

[profile_min]
audio = [1, 1, 224, 90]

[profile_opt]
audio = [1, 1, 224, 90]

[profile_max]
audio = [1, 1, 224, 90]
"#;
        let trt: TrtConfig = toml::from_str(toml).expect("valid TRT config should parse");
        let serialized = toml::to_string(&trt).expect("TRT config should serialize");
        let reparsed: TrtConfig =
            toml::from_str(&serialized).expect("serialized TRT config should reparse");

        assert_eq!(reparsed, trt);
        assert!(trt.enabled);
        assert_eq!(trt.precision, TrtPrecision::Fp16);
        assert_eq!(trt.builder_optimization_level, 3);
        assert_eq!(trt.engine_hw_compatible, false);
        assert_eq!(
            trt.profile_min
                .as_ref()
                .and_then(|profiles| profiles.get("audio")),
            Some(&vec![1, 1, 224, 90])
        );
    }

    #[test]
    fn test_manifest_without_trt_section_keeps_trt_none() {
        let toml = make_model_toml(&[]);
        let dir = write_temp_file("manifest.toml", &toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();

        assert_eq!(manifest.trt, None);
    }

    #[test]
    fn test_trt_precision_values_parse_and_invalid_errors() {
        for (value, expected) in [
            ("fp16", TrtPrecision::Fp16),
            ("fp32", TrtPrecision::Fp32),
            ("int8", TrtPrecision::Int8),
        ] {
            let toml = format!("enabled = true\nprecision = \"{value}\"\n");
            let trt: TrtConfig = toml::from_str(&toml).unwrap();
            assert_eq!(trt.precision, expected);
        }

        let err = toml::from_str::<TrtConfig>("enabled = true\nprecision = \"bf16\"\n")
            .expect_err("invalid TRT precision should fail deserialization");
        assert!(err.to_string().contains("unknown variant"));
    }

    #[test]
    fn test_trt_profile_tables_must_be_all_or_none_when_enabled() {
        let mut toml = make_model_toml(&[]);
        toml.push_str(
            r#"
[inference.trt]
enabled = true

[inference.trt.profile_min]
audio = [1, 1, 224, 90]
"#,
        );
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(
            err.to_string().contains("profile_min")
                && err.to_string().contains("profile_opt")
                && err.to_string().contains("profile_max"),
            "error should name the required profile tables: {err}"
        );
    }

    #[test]
    fn test_trt_profile_tables_must_be_all_or_none_even_when_disabled() {
        let mut toml = make_model_toml(&[]);
        toml.push_str(
            r#"
[inference.trt]
enabled = false

[inference.trt.profile_min]
audio = [1, 1, 224, 90]
"#,
        );
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("profile_opt"));
    }

    #[test]
    fn test_trt_profile_tables_must_have_identical_keys() {
        let mut toml = make_model_toml(&[]);
        toml.push_str(
            r#"
[inference.trt]
enabled = true

[inference.trt.profile_min]
audio = [1, 1, 224, 90]

[inference.trt.profile_opt]
image = [1, 3, 640, 640]

[inference.trt.profile_max]
audio = [32, 1, 224, 90]
"#,
        );
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("identical input keys"));
    }

    #[test]
    fn test_trt_builder_optimization_level_must_be_one_through_five() {
        let mut toml = make_model_toml(&[]);
        toml.push_str(
            r#"
[inference.trt]
enabled = true
builder_optimization_level = 6
"#,
        );
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("builder_optimization_level"));
    }

    #[test]
    fn test_trt_builder_optimization_level_rejects_zero() {
        let mut toml = make_model_toml(&[]);
        toml.push_str(
            r#"
[inference.trt]
enabled = true
builder_optimization_level = 0
"#,
        );
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("builder_optimization_level"));
    }

    #[test]
    fn test_real_shipped_manifest_still_parses_without_trt() {
        let manifest_path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .expect("types crate should live under the workspace root")
            .join("models/audiobirds.toml");
        let manifest =
            load_manifest(&manifest_path).expect("real audiobirds manifest should parse");

        assert_eq!(manifest.id, "md-audiobirds-v1");
        assert_eq!(manifest.trt, None);
    }

    #[test]
    fn test_load_tiled_manifest() {
        let toml = r#"
[model]
id = "herdnet-v1"
format = "onnx"
file = "herdnet.onnx"

[preprocessing]
method = "resize"
input_size = [512, 512]
layout = "nchw"
normalization = "imagenet"

[inference]
strategy = "tiled"
tile_size = [512, 512]
tile_overlap = 0

[postprocessing]
method = "heatmap_peaks"
peak_threshold = 0.1
adaptive = true
point_to_box_half_size = 10

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();

        assert!(matches!(
            manifest.inference_strategy,
            InferenceStrategy::Tiled {
                tile_size: [512, 512],
                tile_overlap: 0
            }
        ));
        assert!(matches!(
            manifest.postprocess_method,
            PostprocessMethod::HeatmapPeaks {
                adaptive: true,
                point_to_box_half_size: 10,
                ..
            }
        ));
    }

    #[test]
    fn test_unsupported_format() {
        let toml = r#"
[model]
id = "test"
format = "coreml"
file = "model.mlmodel"

[preprocessing]
method = "resize"
input_size = [224, 224]
layout = "nchw"
normalization = "none"

[inference]
strategy = "single"

[postprocessing]
method = "softmax"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::UnsupportedFormat { .. }));
    }

    #[test]
    fn test_tflite_fp16_audio_cascade_manifests_load() {
        // End-to-end of the three tflite relaxations (format + fp16-without-
        // file_fp16 + mel+softmax), mirroring the real orca cascade manifests
        // (sparrow-engine-models-v0.6.0). Stage 1 detector = mel + sigmoid;
        // stage 2 ecotype = mel + softmax. Both are fp16 with a single `file`.
        let detector = r#"
[model]
id = "orca-detector-fp16-tflite"
format = "tflite"
file = "orca-detector-fp16.tflite"

[preprocessing]
method = "mel_spectrogram"
sample_rate = 24000
n_fft = 1024
hop_length = 128
n_mels = 256
fmin = 200
fmax = 12000
top_db = 80
window = "hann_symmetric"
mel_scale = "slaney"
filter_norm = "slaney"
fill_highfreq = true

[inference]
strategy = "sliding_window"
segment_duration_s = 3.0
segment_stride_s = 1.5
precision = "fp16"

[postprocessing]
method = "sigmoid"
confidence_threshold = 0.5
"#;
        let ecotype = r#"
[model]
id = "orca-ecotype-melinput-fp16-tflite"
format = "tflite"
file = "orca-ecotype-melinput-fp16.tflite"

[preprocessing]
method = "mel_spectrogram"
sample_rate = 24000
n_fft = 1024
hop_length = 128
n_mels = 256
fmin = 200
fmax = 12000
top_db = 80
window = "hann_symmetric"
mel_scale = "slaney"
filter_norm = "slaney"
fill_highfreq = true

[inference]
strategy = "sliding_window"
segment_duration_s = 3.0
segment_stride_s = 3.0
precision = "fp16"

[postprocessing]
method = "softmax"
"#;
        let d_dir = write_temp_file("manifest.toml", detector);
        let det = load_manifest(&d_dir.path().join("manifest.toml"))
            .expect("orca detector tflite manifest should load");
        assert_eq!(det.format, "tflite");
        assert_eq!(det.precision, Precision::Fp16);
        assert!(det.model_file_fp16.is_none());
        assert_eq!(
            crate::model_type::derive_model_type(
                &det.preprocess_method,
                &det.postprocess_method,
                det.subtype,
            ),
            crate::types::ModelType::AudioDetector,
        );

        let e_dir = write_temp_file("manifest.toml", ecotype);
        let eco = load_manifest(&e_dir.path().join("manifest.toml"))
            .expect("orca ecotype mel-input tflite manifest should load");
        assert_eq!(eco.format, "tflite");
        assert_eq!(eco.precision, Precision::Fp16);
        assert_eq!(
            crate::model_type::derive_model_type(
                &eco.preprocess_method,
                &eco.postprocess_method,
                eco.subtype,
            ),
            crate::types::ModelType::AudioClassifier,
        );
    }

    #[test]
    fn test_nhwc_layout_allowed_for_tflite_image_model() {
        // RP-42: the mobile (LiteRT) flavor onboards NHWC `.tflite` image
        // detectors (TensorFlow's channels-last convention). NHWC is permitted
        // for `format = "tflite"` but still rejected for ONNX (ORT CUDA EP Conv
        // bug, issues #27912 / #12288).
        let tflite_nhwc = r#"
[model]
id = "mdv6-tflite"
format = "tflite"
file = "1/model.tflite"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nhwc"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2

[labels]
file = "labels.txt"
format = "name_index_csv"
"#;
        let dir = write_temp_file("manifest.toml", tflite_nhwc);
        let m = load_manifest(&dir.path().join("manifest.toml"))
            .expect("nhwc tflite image manifest should load");
        assert_eq!(m.format, "tflite");
        assert_eq!(m.layout, Some(Layout::Nhwc));

        // The same NHWC layout on an ONNX model must still be rejected.
        let onnx_nhwc = tflite_nhwc
            .replace("format = \"tflite\"", "format = \"onnx\"")
            .replace("file = \"1/model.tflite\"", "file = \"1/model.onnx\"");
        let dir2 = write_temp_file("manifest.toml", &onnx_nhwc);
        let err = load_manifest(&dir2.path().join("manifest.toml")).unwrap_err();
        assert!(
            matches!(err, SparrowEngineError::InvalidManifest(ref s) if s.contains("nhwc")),
            "ONNX + nhwc must be rejected, got: {err:?}"
        );
    }

    #[test]
    fn test_tflite_format_accepted() {
        // The mobile (LiteRT) flavor onboards `.tflite` models. The shared loader
        // accepts the format; flavor-specific backends reject what they can't load.
        let toml = r#"
[model]
id = "test-tflite"
format = "tflite"
file = "model.tflite"

[preprocessing]
method = "resize"
input_size = [224, 224]
layout = "nchw"
normalization = "none"

[inference]
strategy = "single"

[postprocessing]
method = "softmax"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml"))
            .expect("tflite format must be accepted by the shared loader");
        assert_eq!(manifest.format, "tflite");
        assert_eq!(manifest.model_file, "model.tflite");
    }

    #[test]
    fn test_missing_tiled_fields() {
        let toml = r#"
[model]
id = "test"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [512, 512]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "tiled"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::MissingTiledFields));
    }

    #[test]
    fn test_label_path_traversal() {
        let toml = r#"
[model]
id = "test"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "resize"
input_size = [224, 224]
layout = "nchw"
normalization = "none"

[inference]
strategy = "single"

[postprocessing]
method = "softmax"

[labels]
file = "../../../etc/passwd"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::PathTraversal(_)));
    }

    #[test]
    fn test_wrong_manifest_type() {
        let toml = r#"
[pipeline]
id = "test-pipeline"

[[pipeline.steps]]
role = "detector"
model = "megadet"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::WrongManifestType));
    }

    // -- Pipeline manifest tests --

    #[test]
    fn test_load_valid_pipeline() {
        let toml = r#"
[pipeline]
id = "megadet-deepfaune"

[[pipeline.steps]]
role = "detector"
model = "megadetector-v6-yolov9c"

[[pipeline.steps]]
role = "classifier"
model = "deepfaune-v1"
crop_from = "detector"
"#;
        let dir = write_temp_file("pipeline.toml", toml);
        let pipeline = load_pipeline_manifest(&dir.path().join("pipeline.toml")).unwrap();

        assert_eq!(pipeline.id, "megadet-deepfaune");
        assert_eq!(pipeline.steps.len(), 2);
        assert_eq!(pipeline.steps[0].role, PipelineRole::Detector);
        assert_eq!(pipeline.steps[0].model, "megadetector-v6-yolov9c");
        assert_eq!(pipeline.steps[1].role, PipelineRole::Classifier);
    }

    #[test]
    fn test_pipeline_no_detector() {
        let toml = r#"
[pipeline]
id = "bad"

[[pipeline.steps]]
role = "classifier"
model = "deepfaune-v1"
crop_from = "detector"
"#;
        let dir = write_temp_file("pipeline.toml", toml);
        let err = load_pipeline_manifest(&dir.path().join("pipeline.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidPipeline(_)));
    }

    #[test]
    fn test_pipeline_two_detectors() {
        let toml = r#"
[pipeline]
id = "bad"

[[pipeline.steps]]
role = "detector"
model = "det1"

[[pipeline.steps]]
role = "detector"
model = "det2"
"#;
        let dir = write_temp_file("pipeline.toml", toml);
        let err = load_pipeline_manifest(&dir.path().join("pipeline.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidPipeline(_)));
    }

    #[test]
    fn test_wrong_pipeline_type() {
        let toml = r#"
[model]
id = "test"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "resize"
input_size = [224, 224]
layout = "nchw"
normalization = "none"

[inference]
strategy = "single"

[postprocessing]
method = "softmax"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let err = load_pipeline_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::WrongPipelineType));
    }

    // -- Label loading tests --

    #[test]
    fn test_load_labels_one_per_line() {
        let content = "animal\nperson\nvehicle\n";
        let dir = write_temp_file("labels.txt", content);
        let labels = load_labels(&dir.path().join("labels.txt"), &LabelFormat::OnePerLine).unwrap();
        assert_eq!(labels, vec!["animal", "person", "vehicle"]);
    }

    #[test]
    fn test_load_labels_name_index_csv() {
        let content = "animal,0\nperson,1\ncar,2\n";
        let dir = write_temp_file("labels.txt", content);
        let labels =
            load_labels(&dir.path().join("labels.txt"), &LabelFormat::NameIndexCsv).unwrap();
        assert_eq!(labels, vec!["animal", "person", "car"]);
    }

    #[test]
    fn test_load_labels_index_name_csv() {
        let content = "0,animal\n1,person\n2,car\n";
        let dir = write_temp_file("labels.txt", content);
        let labels =
            load_labels(&dir.path().join("labels.txt"), &LabelFormat::IndexNameCsv).unwrap();
        assert_eq!(labels, vec!["animal", "person", "car"]);
    }

    #[test]
    fn test_load_labels_name_index_csv_sparse() {
        let content = "cat,0\ndog,3\n";
        let dir = write_temp_file("labels.txt", content);
        let labels =
            load_labels(&dir.path().join("labels.txt"), &LabelFormat::NameIndexCsv).unwrap();
        assert_eq!(labels.len(), 4);
        assert_eq!(labels[0], "cat");
        assert_eq!(labels[1], "");
        assert_eq!(labels[3], "dog");
    }

    #[test]
    fn test_load_real_label_files() {
        // Test against real label files from the test_files directory.
        let test_dir = Path::new("/home/miao/repos/PW_refactor/test_files/onnx");

        // MDV6 labels: name_index_csv format (animal,0 / person,1 / car,2)
        let mdv6_path = test_dir.join("models_MDV6-yolov10-e_labels.txt");
        if mdv6_path.exists() {
            let labels = load_labels(&mdv6_path, &LabelFormat::NameIndexCsv).unwrap();
            assert_eq!(labels[0], "animal");
            assert_eq!(labels[1], "person");
            assert_eq!(labels[2], "car");
        }

        // HerdNet labels: name_index_csv format
        let herdnet_path = test_dir.join("models_HerdNet_General_Dataset_2022_labels.txt");
        if herdnet_path.exists() {
            let labels = load_labels(&herdnet_path, &LabelFormat::NameIndexCsv).unwrap();
            assert_eq!(labels[0], "background");
            assert_eq!(labels[1], "topi");
            assert_eq!(labels.len(), 7);
        }
    }

    #[test]
    fn test_label_file_not_found() {
        let err = load_labels(
            Path::new("/nonexistent/labels.txt"),
            &LabelFormat::OnePerLine,
        )
        .unwrap_err();
        assert!(matches!(err, SparrowEngineError::LabelFileNotFound(_)));
    }

    #[test]
    fn test_manifest_not_found() {
        let err = load_manifest(Path::new("/nonexistent/manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::ManifestNotFound(_)));
    }

    #[test]
    fn test_softmax_classifier_manifest() {
        let toml = r#"
[model]
id = "deepfaune-v1"
format = "onnx"
file = "deepfaune_v1.onnx"

[preprocessing]
method = "resize"
input_size = [224, 224]
layout = "nchw"
normalization = "imagenet"

[inference]
strategy = "single"

[postprocessing]
method = "softmax"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();

        assert_eq!(manifest.postprocess_method, PostprocessMethod::Softmax);
        assert_eq!(manifest.confidence_threshold, None);
        assert_eq!(manifest.normalization, Some(Normalization::Imagenet));
    }

    #[test]
    fn test_pad_value_defaults_to_zero() {
        let toml = r#"
[model]
id = "test"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert_eq!(manifest.pad_value, Some(0.0));
    }

    // -- Round 1 review fix tests --

    /// Helper: build a minimal valid model TOML with overrideable fields.
    fn make_model_toml(overrides: &[(&str, &str)]) -> String {
        let mut id = r#""test""#.to_string();
        let mut format = r#""onnx""#.to_string();
        let mut file = r#""model.onnx""#.to_string();
        let mut method = r#""letterbox""#.to_string();
        let mut input_size = "[640, 640]".to_string();
        let mut strategy = r#""single""#.to_string();
        let mut tile_size = String::new();
        let mut tile_overlap = String::new();
        let mut postmethod = r#""yolo_e2e""#.to_string();
        let mut post_extra = String::new();
        let mut label_file = r#""labels.txt""#.to_string();
        let mut label_format = r#""one_per_line""#.to_string();

        for &(k, v) in overrides {
            match k {
                "id" => id = v.to_string(),
                "format" => format = v.to_string(),
                "file" => file = v.to_string(),
                "method" => method = v.to_string(),
                "input_size" => input_size = v.to_string(),
                "strategy" => strategy = v.to_string(),
                "tile_size" => tile_size = format!("tile_size = {v}"),
                "tile_overlap" => tile_overlap = format!("tile_overlap = {v}"),
                "postmethod" => postmethod = v.to_string(),
                "post_extra" => post_extra = v.to_string(),
                "label_file" => label_file = v.to_string(),
                "label_format" => label_format = v.to_string(),
                _ => panic!("unknown override key: {k}"),
            }
        }

        format!(
            r#"
[model]
id = {id}
format = {format}
file = {file}

[preprocessing]
method = {method}
input_size = {input_size}
layout = "nchw"
normalization = "unit"

[inference]
strategy = {strategy}
{tile_size}
{tile_overlap}

[postprocessing]
method = {postmethod}
{post_extra}

[labels]
file = {label_file}
format = {label_format}
"#
        )
    }

    #[test]
    fn test_empty_model_id() {
        let toml = make_model_toml(&[("id", r#""""#)]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("id"));
    }

    #[test]
    fn test_empty_model_file() {
        let toml = make_model_toml(&[("file", r#""""#)]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("file"));
    }

    #[test]
    fn test_zero_input_size() {
        let toml = make_model_toml(&[("input_size", "[0, 640]")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("input_size"));
    }

    #[test]
    fn test_zero_tile_size() {
        let toml = make_model_toml(&[
            ("strategy", r#""tiled""#),
            ("tile_size", "[0, 0]"),
            ("tile_overlap", "0"),
            ("method", r#""resize""#),
            ("postmethod", r#""softmax""#),
        ]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("tile_size"));
    }

    #[test]
    fn test_tile_overlap_exceeds_tile_size() {
        let toml = make_model_toml(&[
            ("strategy", r#""tiled""#),
            ("tile_size", "[512, 512]"),
            ("tile_overlap", "512"),
            ("input_size", "[512, 512]"),
            ("method", r#""resize""#),
            ("postmethod", r#""softmax""#),
        ]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("tile_overlap"));
    }

    #[test]
    fn test_path_traversal_model_file() {
        let toml = make_model_toml(&[("file", r#""../../etc/model.onnx""#)]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::PathTraversal(_)));
    }

    #[test]
    fn test_absolute_path_label() {
        let toml = make_model_toml(&[("label_file", r#""/etc/passwd""#)]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::PathTraversal(_)));
    }

    #[test]
    fn test_legitimate_double_dot_filename() {
        let toml = make_model_toml(&[("file", r#""model..v2.onnx""#)]);
        let dir = write_temp_file("manifest.toml", &toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert_eq!(manifest.model_file, "model..v2.onnx");
    }

    #[test]
    fn test_tiled_tile_size_must_equal_input_size() {
        let toml = make_model_toml(&[
            ("strategy", r#""tiled""#),
            ("tile_size", "[256, 256]"),
            ("tile_overlap", "0"),
            ("input_size", "[512, 512]"),
            ("method", r#""resize""#),
            ("postmethod", r#""heatmap_peaks""#),
            (
                "post_extra",
                "peak_threshold = 0.1\nadaptive = true\npoint_to_box_half_size = 10",
            ),
        ]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("tile_size == input_size"));
    }

    #[test]
    fn test_detector_requires_letterbox() {
        let toml = make_model_toml(&[("method", r#""resize""#), ("postmethod", r#""yolo_e2e""#)]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("letterbox"));
    }

    // -- Audio manifest tests --

    /// Build a valid audio model manifest TOML with optional field overrides.
    fn make_audio_toml(overrides: &[(&str, &str)]) -> String {
        let mut sample_rate = "48000".to_string();
        let mut n_fft = "1024".to_string();
        let mut hop_length = "512".to_string();
        let mut n_mels = "224".to_string();
        let mut fmin = "0.0".to_string();
        let mut fmax = "24000.0".to_string();
        let mut top_db = "80.0".to_string();
        let mut window = r#""hann_symmetric""#.to_string();
        let mut mel_scale = r#""slaney""#.to_string();
        let mut filter_norm = r#""slaney""#.to_string();
        let mut segment_duration_s = "1.0".to_string();
        let mut segment_stride_s = "0.3".to_string();
        let mut postmethod = r#""sigmoid""#.to_string();
        let mut post_extra = "confidence_threshold = 0.5".to_string();

        for &(k, v) in overrides {
            match k {
                "sample_rate" => sample_rate = v.to_string(),
                "n_fft" => n_fft = v.to_string(),
                "hop_length" => hop_length = v.to_string(),
                "n_mels" => n_mels = v.to_string(),
                "fmin" => fmin = v.to_string(),
                "fmax" => fmax = v.to_string(),
                "top_db" => top_db = v.to_string(),
                "window" => window = v.to_string(),
                "mel_scale" => mel_scale = v.to_string(),
                "filter_norm" => filter_norm = v.to_string(),
                "segment_duration_s" => segment_duration_s = v.to_string(),
                "segment_stride_s" => segment_stride_s = v.to_string(),
                "postmethod" => postmethod = v.to_string(),
                "post_extra" => post_extra = v.to_string(),
                _ => panic!("unknown audio override key: {k}"),
            }
        }

        format!(
            r#"
[model]
id = "audio-test"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "mel_spectrogram"
sample_rate = {sample_rate}
n_fft = {n_fft}
hop_length = {hop_length}
n_mels = {n_mels}
fmin = {fmin}
fmax = {fmax}
top_db = {top_db}
window = {window}
mel_scale = {mel_scale}
filter_norm = {filter_norm}

[inference]
strategy = "sliding_window"
segment_duration_s = {segment_duration_s}
segment_stride_s = {segment_stride_s}

[postprocessing]
method = {postmethod}
{post_extra}
"#
        )
    }

    /// Build a valid RawAudio model manifest TOML with optional field overrides.
    fn make_raw_audio_toml(overrides: &[(&str, &str)]) -> String {
        let mut sample_rate = "32000".to_string();
        let mut window_samples = "160000".to_string();
        let mut strategy = r#""sliding_window""#.to_string();
        let mut inference_extra = "segment_duration_s = 5.0\nsegment_stride_s = 5.0".to_string();
        let mut postmethod = r#""softmax""#.to_string();
        let mut post_extra = "".to_string();

        for &(k, v) in overrides {
            match k {
                "sample_rate" => sample_rate = v.to_string(),
                "window_samples" => window_samples = v.to_string(),
                "strategy" => strategy = v.to_string(),
                "inference_extra" => inference_extra = v.to_string(),
                "postmethod" => postmethod = v.to_string(),
                "post_extra" => post_extra = v.to_string(),
                _ => panic!("unknown raw audio override key: {k}"),
            }
        }

        format!(
            r#"
[model]
id = "perch-test"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "raw_audio"
sample_rate = {sample_rate}
window_samples = {window_samples}

[inference]
strategy = {strategy}
{inference_extra}

[postprocessing]
method = {postmethod}
{post_extra}

[labels]
file = "labels.txt"
format = "one_per_line"
"#
        )
    }

    #[test]
    fn test_load_audio_manifest() {
        let toml = make_audio_toml(&[]);
        let dir = write_temp_file("manifest.toml", &toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();

        assert_eq!(manifest.id, "audio-test");
        assert!(matches!(
            manifest.preprocess_method,
            PreprocessMethod::MelSpectrogram {
                sample_rate: 48000,
                n_fft: 1024,
                hop_length: 512,
                n_mels: 224,
                ..
            }
        ));
        if let PreprocessMethod::MelSpectrogram {
            fmin,
            fmax,
            top_db,
            window,
            mel_scale,
            filter_norm,
            ..
        } = &manifest.preprocess_method
        {
            assert!((*fmin - 0.0).abs() < 1e-6);
            assert!((*fmax - 24000.0).abs() < 1e-6);
            assert!((*top_db - 80.0).abs() < 1e-6);
            assert_eq!(window, "hann_symmetric");
            assert_eq!(mel_scale, "slaney");
            assert_eq!(filter_norm, "slaney");
        } else {
            panic!("expected MelSpectrogram");
        }
        assert!(matches!(
            manifest.inference_strategy,
            InferenceStrategy::SlidingWindow { .. }
        ));
        assert!(matches!(
            manifest.postprocess_method,
            PostprocessMethod::Sigmoid { confidence_threshold } if (confidence_threshold - 0.5).abs() < 1e-6
        ));
        // Audio models have no image-specific fields
        assert_eq!(manifest.input_size, None);
        assert_eq!(manifest.layout, None);
        assert_eq!(manifest.normalization, None);
        assert_eq!(manifest.pad_value, None);
        // Binary detector: no labels
        assert_eq!(manifest.label_file, None);
    }

    #[test]
    fn test_load_raw_audio_manifest() {
        let toml = make_raw_audio_toml(&[]);
        let dir = write_temp_file("manifest.toml", &toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();

        assert_eq!(manifest.id, "perch-test");
        assert!(matches!(
            manifest.preprocess_method,
            PreprocessMethod::RawAudio {
                sample_rate: 32000,
                window_samples: 160000,
                ..
            }
        ));
        assert_eq!(
            manifest.inference_strategy,
            InferenceStrategy::SlidingWindow {
                segment_duration_s: 5.0,
                segment_stride_s: 5.0,
            }
        );
        assert_eq!(manifest.postprocess_method, PostprocessMethod::Softmax);
    }

    #[test]
    fn test_raw_audio_requires_sliding_window() {
        let toml = make_raw_audio_toml(&[("strategy", r#""single""#), ("inference_extra", "")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("sliding_window"));
    }

    #[test]
    fn test_raw_audio_window_samples_must_match_segment_duration() {
        let toml = make_raw_audio_toml(&[("window_samples", "159998")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("window_samples"));
    }

    #[test]
    fn test_raw_audio_window_samples_allows_one_sample_rounding() {
        let toml = make_raw_audio_toml(&[("window_samples", "160001")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert!(matches!(
            manifest.preprocess_method,
            PreprocessMethod::RawAudio {
                window_samples: 160001,
                ..
            }
        ));
    }

    #[test]
    fn test_audio_rejects_non_finite_sliding_window_fields() {
        for (field, value) in [("segment_duration_s", "inf"), ("segment_stride_s", "inf")] {
            let inference_extra = match field {
                "segment_duration_s" => {
                    format!("segment_duration_s = {value}\nsegment_stride_s = 5.0")
                }
                "segment_stride_s" => {
                    format!("segment_duration_s = 5.0\nsegment_stride_s = {value}")
                }
                _ => unreachable!(),
            };
            let overrides = [("inference_extra", inference_extra.as_str())];
            let toml = make_raw_audio_toml(&overrides);
            let dir = write_temp_file("manifest.toml", &toml);
            let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
            assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
            assert!(err.to_string().contains(field));
        }
    }

    #[test]
    fn test_audio_accepts_mel_softmax_classifier() {
        // RP-39: a mel-input multi-class audio classifier (the orca ecotype
        // mel-input re-export) must load. Previously rejected as an "unsupported
        // audio" combination. softmax needs no confidence_threshold.
        let toml = make_audio_toml(&[("postmethod", r#""softmax""#), ("post_extra", "")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml"))
            .expect("mel + softmax audio classifier should load (RP-39)");
        assert!(matches!(
            manifest.preprocess_method,
            PreprocessMethod::MelSpectrogram { .. }
        ));
        assert_eq!(manifest.postprocess_method, PostprocessMethod::Softmax);
        assert_eq!(
            crate::model_type::derive_model_type(
                &manifest.preprocess_method,
                &manifest.postprocess_method,
                manifest.subtype,
            ),
            crate::types::ModelType::AudioClassifier,
        );
    }

    #[test]
    fn test_audio_rejects_unsupported_preprocess_postprocess_pairs() {
        let cases = [
            (
                make_raw_audio_toml(&[
                    ("postmethod", r#""sigmoid""#),
                    ("post_extra", "confidence_threshold = 0.5"),
                ]),
                "unsupported audio",
            ),
            (
                make_raw_audio_toml(&[("postmethod", r#""yolo_e2e""#)]),
                "unsupported audio",
            ),
        ];
        for (toml, expected) in cases {
            let dir = write_temp_file("manifest.toml", &toml);
            let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
            assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
            assert!(
                err.to_string().contains(expected),
                "expected {expected:?} in {err}"
            );
        }
    }

    #[test]
    fn test_audio_invalid_n_fft_zero() {
        let toml = make_audio_toml(&[("n_fft", "0")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("n_fft"));
    }

    #[test]
    fn test_audio_invalid_fmax_less_than_fmin() {
        let toml = make_audio_toml(&[("fmin", "500.0"), ("fmax", "200.0")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("fmax"));
    }

    #[test]
    fn test_audio_n_fft_not_power_of_two() {
        let toml = make_audio_toml(&[("n_fft", "1000")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("power of 2"));
    }

    #[test]
    fn test_audio_invalid_stride_zero() {
        let toml = make_audio_toml(&[("segment_stride_s", "0.0")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("segment_stride_s"));
    }

    #[test]
    fn test_audio_unsupported_window() {
        let toml = make_audio_toml(&[("window", r#""blackman""#)]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("window"));
    }

    #[test]
    fn test_audio_unsupported_mel_scale() {
        // Phase 3.8 Step 2 Wave 0a (2026-05-04): "slaney" is now the only
        // supported value (was "htk"); the rejected fixture is the legacy
        // "htk" string so a stale manifest copy fails loudly.
        let toml = make_audio_toml(&[("mel_scale", r#""htk""#)]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("mel_scale"));
    }

    #[test]
    fn test_audio_unsupported_filter_norm() {
        // Phase 3.8 Step 2 Wave 0a (2026-05-04): "slaney" is now the only
        // supported value (was "area"); the rejected fixture is the legacy
        // "area" string so a stale manifest copy fails loudly.
        let toml = make_audio_toml(&[("filter_norm", r#""area""#)]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
        assert!(err.to_string().contains("filter_norm"));
    }

    #[test]
    fn test_audio_invalid_sample_rate_zero() {
        let toml = make_audio_toml(&[("sample_rate", "0")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
    }

    #[test]
    fn test_audio_invalid_hop_length_zero() {
        let toml = make_audio_toml(&[("hop_length", "0")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
    }

    #[test]
    fn test_audio_invalid_n_mels_zero() {
        let toml = make_audio_toml(&[("n_mels", "0")]);
        let dir = write_temp_file("manifest.toml", &toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        assert!(matches!(err, SparrowEngineError::InvalidManifest(_)));
    }

    // Regression (MN1): manifest parser must reject `layout = "nhwc"` with a
    // clear error pointing to the NCHW requirement + tf2onnx escape hatch.
    // ORT CUDA EP has SafeInt overflow bugs with NHWC Conv (issues #27912 /
    // #12288). See design/v4/consensus_design_revised.md.
    #[test]
    fn test_layout_nhwc_rejected_with_escape_hatch() {
        let toml = r#"
[model]
id = "nhwc-model"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nhwc"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        match err {
            SparrowEngineError::InvalidManifest(msg) => {
                assert!(msg.contains("NCHW"), "error must name NCHW: {msg}");
                assert!(
                    msg.contains("tf2onnx"),
                    "error must mention tf2onnx escape hatch: {msg}"
                );
            }
            other => panic!("expected InvalidManifest, got {other:?}"),
        }
    }

    #[test]
    fn test_layout_unknown_value_still_rejected() {
        let toml = r#"
[model]
id = "unknown-model"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "bogus"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        match err {
            SparrowEngineError::InvalidManifest(msg) => {
                assert!(msg.contains("Unknown layout"), "got: {msg}");
                assert!(
                    msg.contains("bogus"),
                    "error must echo the bad value: {msg}"
                );
            }
            other => panic!("expected InvalidManifest, got {other:?}"),
        }
    }

    // Regression (T1): Phase 3 added `version`, `description`, `onnx_sha256`,
    // `onnx_size_bytes` to `[model]` with `#[serde(default)]`. Verify
    // (1) roundtrip — fields populate when present,
    // (2) backward-compat — manifests without the fields still load (default None),
    // (3) partial — a subset of the new fields is accepted (not all-or-nothing).
    #[test]
    fn test_phase3_optional_fields_roundtrip() {
        let toml = r#"
[model]
id = "phase3-full"
format = "onnx"
file = "model.onnx"
version = "6.1.2"
description = "MegaDetector v6.1 (YOLO-V9)"
onnx_sha256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
onnx_size_bytes = 104857600

[preprocessing]
method = "letterbox"
input_size = [1280, 1280]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let m = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert_eq!(m.version.as_deref(), Some("6.1.2"));
        assert_eq!(
            m.description.as_deref(),
            Some("MegaDetector v6.1 (YOLO-V9)")
        );
        assert_eq!(
            m.onnx_sha256.as_deref(),
            Some("e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855")
        );
        assert_eq!(m.onnx_size_bytes, Some(104857600));
    }

    #[test]
    fn test_phase3_optional_fields_backward_compat() {
        // No Phase 3 fields — must default to None, not error.
        let toml = r#"
[model]
id = "legacy-model"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let m = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert!(m.version.is_none());
        assert!(m.description.is_none());
        assert!(m.onnx_sha256.is_none());
        assert!(m.onnx_size_bytes.is_none());
    }

    #[test]
    fn test_phase3_optional_fields_partial() {
        // Only some of the new fields — must accept partial population.
        let toml = r#"
[model]
id = "partial-model"
format = "onnx"
file = "model.onnx"
version = "1.0.0"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let m = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert_eq!(m.version.as_deref(), Some("1.0.0"));
        assert!(m.description.is_none());
        assert!(m.onnx_sha256.is_none());
        assert!(m.onnx_size_bytes.is_none());
    }

    // -- Phase 3.5 S3 (MT-9): subtype field tests --

    // Roundtrip: `subtype = "overhead"` parses to `ModelSubtype::Overhead`.
    #[test]
    fn test_subtype_overhead_roundtrip() {
        let toml = r#"
[model]
id = "herdnet"
format = "onnx"
file = "model.onnx"
subtype = "overhead"

[preprocessing]
method = "resize"
input_size = [512, 512]
layout = "nchw"
normalization = "imagenet"

[inference]
strategy = "tiled"
tile_size = [512, 512]
tile_overlap = 0

[postprocessing]
method = "heatmap_peaks"
peak_threshold = 0.2
adaptive = false
point_to_box_half_size = 10
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let m = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert_eq!(m.subtype, ModelSubtype::Overhead);
    }

    // Explicit `subtype = "standard"` parses to Standard.
    #[test]
    fn test_subtype_standard_explicit() {
        let toml = r#"
[model]
id = "mdv6"
format = "onnx"
file = "model.onnx"
subtype = "standard"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let m = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert_eq!(m.subtype, ModelSubtype::Standard);
    }

    // Backward compat: missing `subtype` field → Standard (no error).
    #[test]
    fn test_subtype_missing_defaults_to_standard() {
        let toml = r#"
[model]
id = "legacy"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let m = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert_eq!(m.subtype, ModelSubtype::Standard);
    }

    // Unknown subtype value must be rejected with a helpful error.
    #[test]
    fn test_subtype_unknown_value_rejected() {
        let toml = r#"
[model]
id = "bogus"
format = "onnx"
file = "model.onnx"
subtype = "segmentation"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let err = load_manifest(&dir.path().join("manifest.toml")).unwrap_err();
        match err {
            SparrowEngineError::InvalidManifest(msg) => {
                assert!(msg.contains("subtype"), "error must name subtype: {msg}");
                assert!(
                    msg.contains("segmentation"),
                    "error must echo the bad value: {msg}"
                );
                assert!(
                    msg.contains("overhead") || msg.contains("standard"),
                    "error must list accepted values: {msg}"
                );
            }
            other => panic!("expected InvalidManifest, got {other:?}"),
        }
    }

    // Canonical overhead manifests (sparrow-engine/models/herdnet.toml, owlt.toml) must
    // parse cleanly and carry `subtype = Overhead`. Guards against typo drift
    // between the canonical templates and the parser.
    #[test]
    fn test_canonical_overhead_manifests_load() {
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("models");
        for (file, id) in [
            ("herdnet.toml", "herdnet-general-2022"),
            ("owlt.toml", "owl-t"),
        ] {
            let path = manifest_dir.join(file);
            if !path.exists() {
                // Soft-skip when the repo layout differs (e.g., CI subset).
                // The file is canonical, not load-bearing for inference.
                continue;
            }
            let m = load_manifest(&path)
                .unwrap_or_else(|e| panic!("canonical manifest {file} failed to parse: {e:?}"));
            assert_eq!(
                m.subtype,
                ModelSubtype::Overhead,
                "{file} must declare subtype = overhead"
            );
            assert_eq!(m.id, id, "{file} id drift");
        }
    }

    // ----- Phase 3.8 precision (FP16) tests -----
    fn write_temp_manifest(toml: &str) -> tempfile::NamedTempFile {
        use std::io::Write;
        let mut f = tempfile::Builder::new().suffix(".toml").tempfile().unwrap();
        f.write_all(toml.as_bytes()).unwrap();
        f
    }

    #[test]
    fn test_precision_default_is_fp32() {
        let toml = r#"
[model]
id = "x"
format = "onnx"
file = "x.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"
pad_value = 0.0

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let f = write_temp_manifest(toml);
        let m = load_manifest(f.path()).unwrap();
        assert_eq!(m.precision, Precision::Fp32);
        assert_eq!(m.model_file_fp16, None);
    }

    #[test]
    fn test_precision_fp16_with_file_fp16() {
        let toml = r#"
[model]
id = "x"
format = "onnx"
file = "x.onnx"
file_fp16 = "x_fp16.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"
pad_value = 0.0

[inference]
strategy = "single"
precision = "fp16"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let f = write_temp_manifest(toml);
        let m = load_manifest(f.path()).unwrap();
        assert_eq!(m.precision, Precision::Fp16);
        assert_eq!(m.model_file_fp16.as_deref(), Some("x_fp16.onnx"));
    }

    #[test]
    fn test_precision_fp16_without_file_fp16_rejected() {
        let toml = r#"
[model]
id = "x"
format = "onnx"
file = "x.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"
pad_value = 0.0

[inference]
strategy = "single"
precision = "fp16"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let f = write_temp_manifest(toml);
        let err = load_manifest(f.path()).unwrap_err();
        assert!(
            format!("{err}").contains("file_fp16"),
            "expected file_fp16-required error, got: {err:?}"
        );
    }

    #[test]
    fn test_precision_unknown_value_rejected() {
        let toml = r#"
[model]
id = "x"
format = "onnx"
file = "x.onnx"

[preprocessing]
method = "letterbox"
input_size = [640, 640]
layout = "nchw"
normalization = "unit"
pad_value = 0.0

[inference]
strategy = "single"
precision = "bf16"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2
"#;
        let f = write_temp_manifest(toml);
        let err = load_manifest(f.path()).unwrap_err();
        assert!(
            format!("{err}").contains("Unknown precision"),
            "expected Unknown precision error, got: {err:?}"
        );
    }

    // -- Phase 4 W1: [provenance] round-trip ---------------------------------

    #[test]
    fn test_manifest_with_provenance_round_trips_all_fields() {
        let toml = r#"
[model]
id = "mdv6-r3"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [1280, 1280]
layout = "nchw"
normalization = "unit"
pad_value = 0.447

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2

[labels]
file = "labels.txt"
format = "one_per_line"

[provenance]
training_dataset_id    = "ds-2026-04-camera-trap-r1"
training_experiment_id = "exp-mdv6-fp16-r3"
training_repo_commit   = "9c4b6a3"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        let p = manifest
            .provenance
            .expect("manifest should preserve [provenance] section");
        assert_eq!(
            p.training_dataset_id.as_deref(),
            Some("ds-2026-04-camera-trap-r1")
        );
        assert_eq!(
            p.training_experiment_id.as_deref(),
            Some("exp-mdv6-fp16-r3")
        );
        assert_eq!(p.training_repo_commit.as_deref(), Some("9c4b6a3"));
    }

    // -- Phase 4 W4: [drift_reference] round-trip ---------------------------

    #[test]
    fn test_manifest_with_drift_reference_round_trips() {
        let toml = r#"
[model]
id = "mdv6"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [1280, 1280]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2

[labels]
file = "labels.txt"
format = "one_per_line"

[drift_reference.class_distribution]
animal  = 0.7
person  = 0.2
vehicle = 0.1
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        let r = manifest
            .drift_reference
            .expect("manifest should preserve [drift_reference] section");
        assert_eq!(r.class_distribution.get("animal"), Some(&0.7));
        assert_eq!(r.class_distribution.get("person"), Some(&0.2));
        assert_eq!(r.class_distribution.get("vehicle"), Some(&0.1));
        assert_eq!(r.class_distribution.len(), 3);
    }

    #[test]
    fn test_manifest_without_drift_reference_loads_with_none() {
        let toml = r#"
[model]
id = "mdv6"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [1280, 1280]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert_eq!(
            manifest.drift_reference, None,
            "missing [drift_reference] must produce None"
        );
    }

    #[test]
    fn test_manifest_without_provenance_loads_with_none() {
        // Manifests authored before Phase 4 (no [provenance] section) must
        // continue to load without error and surface `provenance = None`.
        let toml = r#"
[model]
id = "mdv6"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [1280, 1280]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2

[labels]
file = "labels.txt"
format = "one_per_line"
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        assert_eq!(
            manifest.provenance, None,
            "missing [provenance] section must produce None, not a default-filled struct"
        );
    }

    // -- Phase 4 audit-fix R1 regression tests (T-5, T-6) -------------------

    /// T-5 — Empty `[provenance]` section (header present, no fields) must
    /// distinguish from a missing section: present-with-no-values surfaces
    /// `Some(ProvenanceRecord::default())` (all fields `None`), while a
    /// missing section surfaces `None`. Pins the round-trip semantics so a
    /// future serde refactor doesn't collapse the two cases.
    #[test]
    fn test_manifest_with_empty_provenance_section_loads_as_some_default() {
        let toml = r#"
[model]
id = "mdv6"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [1280, 1280]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2

[labels]
file = "labels.txt"
format = "one_per_line"

[provenance]
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        let p = manifest
            .provenance
            .expect("[provenance] header present must yield Some, not None");
        assert_eq!(p.training_dataset_id, None);
        assert_eq!(p.training_experiment_id, None);
        assert_eq!(p.training_repo_commit, None);
        // And the type's Default impl produces the same all-None struct.
        assert_eq!(p, ProvenanceRecord::default());
    }

    /// T-6 — Empty `[drift_reference.class_distribution]` table (parent
    /// section present, no entries) must yield `Some(DriftReference {
    /// empty BTreeMap })`, not `None`. Locks the same present-vs-absent
    /// semantics for the W4 wire format.
    #[test]
    fn test_manifest_with_empty_drift_reference_class_distribution_loads_as_some_empty() {
        // TOML: section header present but no key/value entries.
        let toml = r#"
[model]
id = "mdv6"
format = "onnx"
file = "model.onnx"

[preprocessing]
method = "letterbox"
input_size = [1280, 1280]
layout = "nchw"
normalization = "unit"

[inference]
strategy = "single"

[postprocessing]
method = "yolo_e2e"
confidence_threshold = 0.2

[labels]
file = "labels.txt"
format = "one_per_line"

[drift_reference.class_distribution]
"#;
        let dir = write_temp_file("manifest.toml", toml);
        let manifest = load_manifest(&dir.path().join("manifest.toml")).unwrap();
        let r = manifest
            .drift_reference
            .expect("[drift_reference.class_distribution] header present must yield Some");
        assert!(
            r.class_distribution.is_empty(),
            "empty inline table must yield empty BTreeMap, got {} entries",
            r.class_distribution.len()
        );
    }
}
