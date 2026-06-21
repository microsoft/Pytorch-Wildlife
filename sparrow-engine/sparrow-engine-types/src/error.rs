//! Error types for the sparrow-engine workspace.

use std::path::PathBuf;

use crate::types::ModelType;

/// All errors produced by sparrow-engine.
#[derive(Debug, thiserror::Error)]
pub enum SparrowEngineError {
    // -- Engine lifecycle --
    #[error("Engine already created. ORT Environment is process-global; only one Engine allowed.")]
    EngineAlreadyExists,

    #[error("Engine has been freed")]
    EngineFreed,

    // -- Model loading --
    #[error("Model manifest not found: {0}")]
    ManifestNotFound(PathBuf),

    #[error("Invalid manifest: {0}")]
    InvalidManifest(String),

    #[error("Unsupported model format '{format}' for this engine flavor.")]
    UnsupportedFormat { format: String },

    #[error("Model '{id}' output shape {shape} does not match expected postprocessing method '{method}'.")]
    OutputShapeMismatch {
        id: String,
        shape: String,
        method: String,
    },

    #[error("Path traversal or absolute path rejected: {0}")]
    PathTraversal(String),

    #[error("Label file not found: {0}")]
    LabelFileNotFound(PathBuf),

    #[error("Invalid label file format: {0}")]
    InvalidLabelFormat(String),

    // -- Model usage --
    #[error("Model has been unloaded")]
    ModelUnloaded,

    #[error("Model '{id}' is a classifier (postprocessing: {method}). Use classify() instead.")]
    NotADetector { id: String, method: String },

    #[error("Model '{id}' is a detector (postprocessing: {method}). Use detect() instead.")]
    NotAClassifier { id: String, method: String },

    // -- Pipeline --
    #[error("Pipeline '{id}' not found")]
    PipelineNotFound { id: String },

    #[error("Pipeline '{id}' references unloaded models: {missing}")]
    PipelineMissingModels { id: String, missing: String },

    #[error("Invalid pipeline manifest: {0}")]
    InvalidPipeline(String),

    #[error("Incompatible pipeline: detector={detector:?}, classifier={classifier:?}: {reason}")]
    IncompatiblePipeline {
        detector: Option<ModelType>,
        classifier: Option<ModelType>,
        reason: &'static str,
    },

    #[error("Pipeline must contain at least one model")]
    EmptyPipeline,

    // -- Manifest validation --
    #[error("Tiled strategy requires 'tile_size' and 'tile_overlap' fields")]
    MissingTiledFields,

    #[error("Expected model manifest ([model] section), found pipeline manifest")]
    WrongManifestType,

    #[error("Expected pipeline manifest ([pipeline] section), found model manifest")]
    WrongPipelineType,

    // -- Audio --
    #[error("Failed to decode audio: {0}")]
    AudioDecode(String),

    #[error("Audio preprocessing error: {0}")]
    AudioPreprocess(String),

    #[error("Failed to resample audio: {0}")]
    Resample(String),

    #[error("Model '{id}' is not an audio model (preprocessing: {method}). Use detect_audio() for audio models.")]
    NotAnAudioModel { id: String, method: String },

    #[error("Model '{id}' is an audio model (preprocessing: {method}). Use detect() or classify() for vision models.")]
    IsAudioModel { id: String, method: String },

    // -- Image input --
    #[error("Failed to decode image: {0}")]
    ImageDecode(String),

    #[error(
        "Invalid stride: stride ({stride}) must be >= width ({width}) * bytes_per_pixel ({bpp})"
    )]
    InvalidStride { stride: u32, width: u32, bpp: u32 },

    #[error("Image file not found: {0}")]
    ImageFileNotFound(PathBuf),

    // -- nvjpeg dlopen loader (Phase E, 2026-05-25) --
    // Surfaced when the runtime libnvjpeg.so.12 load fails: library missing,
    // wrong CUDA major, or symbol missing. Holds the formatted NvjpegInitError
    // Display string which carries the full remediation text (install
    // nvidia-nvjpeg-cu12 / use the CPU wheel / override
    // SPARROW_ENGINE_NVJPEG_LIBRARY_PATH).
    #[error("nvjpeg runtime unavailable: {0}")]
    NvjpegUnavailable(String),

    // -- TensorRT runtime deps (GPU flavor) --
    #[error("{0}")]
    TrtRuntimeMissing(String),

    // -- ORT --
    #[error("ONNX Runtime error: {0}")]
    Ort(String),

    // -- General --
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("TOML parse error: {0}")]
    TomlParse(#[from] toml::de::Error),

    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Convenience alias used throughout sparrow-engine.
pub type Result<T> = std::result::Result<T, SparrowEngineError>;

#[cfg(test)]
mod phase_a_r1_error_tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn engine_already_exists_displays_singleton_message() {
        let e = SparrowEngineError::EngineAlreadyExists;
        let s = e.to_string();
        assert!(
            s.contains("Engine already created") && s.contains("ORT Environment"),
            "unexpected Display for EngineAlreadyExists: {s}"
        );
    }

    #[test]
    fn manifest_not_found_displays_path() {
        let e = SparrowEngineError::ManifestNotFound(PathBuf::from("/x/manifest.toml"));
        let s = e.to_string();
        assert!(
            s.contains("Model manifest not found") && s.contains("/x/manifest.toml"),
            "unexpected Display for ManifestNotFound: {s}"
        );
    }

    #[test]
    fn output_shape_mismatch_displays_all_three_fields() {
        let e = SparrowEngineError::OutputShapeMismatch {
            id: "mdv6".to_string(),
            shape: "[1, 84, 8400]".to_string(),
            method: "yolo_e2e".to_string(),
        };
        let s = e.to_string();
        assert!(s.contains("mdv6"), "missing id: {s}");
        assert!(s.contains("[1, 84, 8400]"), "missing shape: {s}");
        assert!(s.contains("yolo_e2e"), "missing method: {s}");
    }

    #[test]
    fn invalid_stride_displays_numeric_fields() {
        let e = SparrowEngineError::InvalidStride {
            stride: 100,
            width: 50,
            bpp: 4,
        };
        let s = e.to_string();
        assert!(
            s.contains("100") && s.contains("50") && s.contains('4'),
            "InvalidStride must include all numeric fields: {s}"
        );
    }

    #[test]
    fn ort_variant_constructable_with_owned_string() {
        // Ensure the Ort variant accepts an owned message — exercised heavily
        // by the engine crates' `?` from inference paths in Phase B.
        let e = SparrowEngineError::Ort("session run failed".to_string());
        let s = e.to_string();
        assert!(s.contains("ONNX Runtime error") && s.contains("session run failed"));
    }

    #[test]
    fn incompatible_pipeline_displays_model_types_and_reason() {
        let e = SparrowEngineError::IncompatiblePipeline {
            detector: Some(ModelType::AudioDetector),
            classifier: Some(ModelType::Classifier),
            reason: "modality mismatch",
        };
        let display = e.to_string();
        assert!(
            display.contains("Incompatible pipeline"),
            "missing prefix: {display}"
        );
        assert!(
            display.contains("AudioDetector"),
            "missing detector: {display}"
        );
        assert!(
            display.contains("Classifier"),
            "missing classifier: {display}"
        );
        assert!(
            display.contains("modality mismatch"),
            "missing reason: {display}"
        );
        let debug = format!("{e:?}");
        assert!(
            debug.contains("IncompatiblePipeline"),
            "unexpected Debug: {debug}"
        );
    }

    #[test]
    fn empty_pipeline_displays_clear_message() {
        let e = SparrowEngineError::EmptyPipeline;
        let display = e.to_string();
        assert!(
            display.contains("at least one model"),
            "unexpected Display: {display}"
        );
        let debug = format!("{e:?}");
        assert!(debug.contains("EmptyPipeline"), "unexpected Debug: {debug}");
    }

    #[test]
    fn from_io_error_via_question_mark() {
        // The #[from] on Io must let `?` lift a std::io::Error into SparrowEngineError.
        fn inner() -> Result<()> {
            let _f = std::fs::File::open("/path/that/should/not/exist/bongo_test_marker")?;
            Ok(())
        }
        let err = inner().unwrap_err();
        match err {
            SparrowEngineError::Io(_) => {}
            other => panic!("expected SparrowEngineError::Io, got {other:?}"),
        }
    }

    #[test]
    fn from_toml_de_error_via_question_mark() {
        // Construct a known-bad TOML and let `?` lift toml::de::Error.
        fn inner(s: &str) -> Result<toml::Value> {
            let v: toml::Value = toml::from_str(s)?;
            Ok(v)
        }
        // unterminated string is guaranteed to fail TOML parsing.
        let err = inner("key = \"unterminated\nstring").unwrap_err();
        match err {
            SparrowEngineError::TomlParse(_) => {}
            other => panic!("expected SparrowEngineError::TomlParse, got {other:?}"),
        }
    }

    #[test]
    fn from_serde_json_error_via_question_mark() {
        fn inner(s: &str) -> Result<serde_json::Value> {
            let v: serde_json::Value = serde_json::from_str(s)?;
            Ok(v)
        }
        let err = inner("{not valid json").unwrap_err();
        match err {
            SparrowEngineError::Json(_) => {}
            other => panic!("expected SparrowEngineError::Json, got {other:?}"),
        }
    }
}
