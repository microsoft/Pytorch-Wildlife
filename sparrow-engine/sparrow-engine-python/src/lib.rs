//! PyO3 bindings for sparrow_engine (CPU or GPU pipeline) — `_sparrow_engine_core` native module.
//!
//! Phase 3.8 Phase C Wave 4a: two wheels, both importing as Python
//! `sparrow_engine`:
//!
//! - `sparrow-engine` PyPI distribution (CPU wheel): pulls in `sparrow-engine-cpu` via
//!   `--features cpu`; runtime depends on `onnxruntime` pip package.
//! - `sparrow-engine-gpu` PyPI distribution (GPU wheel): pulls in `sparrow-engine-gpu`
//!   via `--features gpu`; runtime depends on `onnxruntime-gpu`.
//!
//! Both expose the same Python API surface; the dispatch happens at
//! Cargo feature gate time (see `engine_dispatch.rs`).
//!
//! Two-layer design:
//! - This Rust module exposes `PyEngine` + frozen result types to Python
//! - `python/sparrow_engine/__init__.py` adds input normalization and global singleton

#![allow(unexpected_cfgs)]

// Phase 3.8 Phase C Wave 4a: feature mutex enforcement. Mirrors the
// `sparrow-engine-server` and `sparrow-engine-cli` consumer crates so cargo flag
// misuse fails-loud at compile time rather than producing a wheel
// with two `sparrow_engine` candidates linked.
#[cfg(all(feature = "cpu", feature = "gpu"))]
compile_error!("sparrow-engine-python: features `cpu` and `gpu` are mutually exclusive");
#[cfg(not(any(feature = "cpu", feature = "gpu")))]
compile_error!("sparrow-engine-python: one of `cpu` or `gpu` must be enabled (default = cpu)");

mod engine_dispatch;

// Phase 3.8 Phase C audit-fix R1 I-4: alias the dispatch shim as `sparrow_engine`
// so every `sparrow_engine::*` reference in this file routes through the shim
// (which `pub use ::sparrow_engine::*;` re-exports the feature-active
// engine crate's surface). Without this alias, `sparrow_engine::*` paths resolve
// via the cargo extern-crate prelude directly to `sparrow_engine` and
// the shim is decorative; with it, the shim is the single dispatch point
// (matching the `sparrow-engine-cli` Wave 3 pattern in
// `sparrow-engine-cli/src/main.rs`).
use crate::engine_dispatch as sparrow_engine;

use std::io::Cursor;
use std::path::{Path, PathBuf};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

// Phase 3.8 Phase C Wave 4a: route Engine + opts types through the
// per-flavor dispatch shim. The `sparrow_engine` alias is set up at line 38 above
// so `sparrow_engine::SparrowEngineError`, `sparrow_engine::detect::detect`, etc. throughout this
// file route through the shim.
use crate::engine_dispatch::{
    AudioDetectOpts, AudioInput, ClassifyOpts, DetectOpts, Device, Engine, EngineConfig,
    ImageInput, ModelInfo as NativeModelInfo, ModelType,
};

// ---------------------------------------------------------------------------
// Error mapping
// ---------------------------------------------------------------------------

pyo3::create_exception!(
    _sparrow_engine_core,
    SparrowEngineError,
    pyo3::exceptions::PyException
);

fn to_pyerr(e: sparrow_engine::SparrowEngineError) -> PyErr {
    SparrowEngineError::new_err(format!("{e}"))
}

fn validate_pipeline_ids(
    engine: &Engine,
    detector_id: &str,
    classifier_id: &str,
) -> Result<(), sparrow_engine::SparrowEngineError> {
    validate_pipeline_ids_from_available(
        &engine.list_available_models(),
        detector_id,
        classifier_id,
    )
}

fn validate_pipeline_ids_from_available(
    available: &[NativeModelInfo],
    detector_id: &str,
    classifier_id: &str,
) -> Result<(), sparrow_engine::SparrowEngineError> {
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
            sparrow_engine::pipeline_compat::validate_pipeline_compat(
                Some(detector),
                Some(classifier),
            )
        }
        _ => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// Result types — frozen (immutable) PyO3 classes
// ---------------------------------------------------------------------------

/// Axis-aligned bounding box in normalized [0,1] coordinates.
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
#[derive(Clone)]
pub struct BBox {
    #[pyo3(get)]
    pub x_min: f32,
    #[pyo3(get)]
    pub y_min: f32,
    #[pyo3(get)]
    pub x_max: f32,
    #[pyo3(get)]
    pub y_max: f32,
}

#[pymethods]
impl BBox {
    /// Convert normalized [0,1] bbox to pixel coordinates.
    fn to_pixels(&self, width: u32, height: u32) -> (i64, i64, i64, i64) {
        (
            (self.x_min * width as f32).round() as i64,
            (self.y_min * height as f32).round() as i64,
            (self.x_max * width as f32).round() as i64,
            (self.y_max * height as f32).round() as i64,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "BBox(x_min={:.4}, y_min={:.4}, x_max={:.4}, y_max={:.4})",
            self.x_min, self.y_min, self.x_max, self.y_max
        )
    }
}

/// A single detection (bbox + label + confidence).
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
#[derive(Clone)]
pub struct Detection {
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub label_id: u32,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub bbox: BBox,
}

#[pymethods]
impl Detection {
    fn __repr__(&self) -> String {
        format!(
            "Detection(label='{}', confidence={:.4}, bbox={})",
            self.label,
            self.confidence,
            self.bbox.__repr__()
        )
    }
}

/// Full detection output for a single image.
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
pub struct DetectResult {
    model_type: ModelType,
    #[pyo3(get)]
    pub model_id: String,
    #[pyo3(get)]
    pub image_size: (u32, u32),
    #[pyo3(get)]
    pub processing_time_ms: f32,
    #[pyo3(get)]
    pub detections: Vec<Detection>,
}

#[pymethods]
impl DetectResult {
    fn __repr__(&self) -> String {
        format!(
            "DetectResult(model_id='{}', image_size={:?}, detections={})",
            self.model_id,
            self.image_size,
            self.detections.len()
        )
    }

    fn __len__(&self) -> usize {
        self.detections.len()
    }
}

/// A single classification prediction.
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
#[derive(Clone)]
pub struct Classification {
    #[pyo3(get)]
    pub label: String,
    #[pyo3(get)]
    pub label_id: u32,
    #[pyo3(get)]
    pub confidence: f32,
}

#[pymethods]
impl Classification {
    fn __repr__(&self) -> String {
        format!(
            "Classification(label='{}', confidence={:.4})",
            self.label, self.confidence
        )
    }
}

/// Full classification output for a single image.
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
pub struct ClassifyResult {
    #[pyo3(get)]
    pub model_id: String,
    #[pyo3(get)]
    pub image_size: (u32, u32),
    #[pyo3(get)]
    pub processing_time_ms: f32,
    #[pyo3(get)]
    pub classifications: Vec<Classification>,
}

#[pymethods]
impl ClassifyResult {
    fn __repr__(&self) -> String {
        format!(
            "ClassifyResult(model_id='{}', image_size={:?}, classifications={})",
            self.model_id,
            self.image_size,
            self.classifications.len()
        )
    }

    fn __len__(&self) -> usize {
        self.classifications.len()
    }

    /// First classification (highest confidence) or None when empty.
    /// Ergonomic accessor so callers can write `c.top1.label` instead of
    /// `c.classifications[0].label`. Added per Phase D B-11.
    #[getter]
    fn top1(&self) -> Option<Classification> {
        self.classifications.first().cloned()
    }
}

/// A detection with an optional classification (from pipeline).
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
#[derive(Clone)]
pub struct PipelineDetection {
    #[pyo3(get)]
    pub detection: Detection,
    #[pyo3(get)]
    pub classification: Option<Classification>,
}

#[pymethods]
impl PipelineDetection {
    fn __repr__(&self) -> String {
        match &self.classification {
            Some(c) => format!(
                "PipelineDetection(det='{}' {:.4}, cls='{}' {:.4})",
                self.detection.label, self.detection.confidence, c.label, c.confidence
            ),
            None => format!(
                "PipelineDetection(det='{}' {:.4}, cls=None)",
                self.detection.label, self.detection.confidence
            ),
        }
    }
}

/// Full pipeline output for a single image.
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
pub struct PipelineResult {
    #[pyo3(get)]
    pub pipeline_id: String,
    #[pyo3(get)]
    pub image_size: (u32, u32),
    #[pyo3(get)]
    pub processing_time_ms: f32,
    #[pyo3(get)]
    pub detections: Vec<PipelineDetection>,
    model_type: ModelType,
}

#[pymethods]
impl PipelineResult {
    fn __repr__(&self) -> String {
        format!(
            "PipelineResult(pipeline_id='{}', image_size={:?}, detections={})",
            self.pipeline_id,
            self.image_size,
            self.detections.len()
        )
    }

    fn __len__(&self) -> usize {
        self.detections.len()
    }
}

/// A single class entry within an AudioSegment's top-K classification output.
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
#[derive(Clone)]
pub struct AudioClass {
    #[pyo3(get)]
    pub class_idx: u32,
    #[pyo3(get)]
    pub label: Option<String>,
    #[pyo3(get)]
    pub probability: f32,
}

#[pymethods]
impl AudioClass {
    fn __repr__(&self) -> String {
        match &self.label {
            Some(label) => format!(
                "AudioClass(idx={}, label='{}', p={:.4})",
                self.class_idx, label, self.probability
            ),
            None => format!(
                "AudioClass(idx={}, label=None, p={:.4})",
                self.class_idx, self.probability
            ),
        }
    }
}

/// A single detected audio segment.
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
#[derive(Clone)]
pub struct AudioSegment {
    #[pyo3(get)]
    pub start_time_s: f32,
    #[pyo3(get)]
    pub end_time_s: f32,
    #[pyo3(get)]
    pub confidence: f32,
    #[pyo3(get)]
    pub classes: Vec<AudioClass>,
}

#[pymethods]
impl AudioSegment {
    fn __repr__(&self) -> String {
        format!(
            "AudioSegment(start={:.2}s, end={:.2}s, confidence={:.4})",
            self.start_time_s, self.end_time_s, self.confidence
        )
    }
}

/// Full audio detection output for a single audio file.
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
pub struct AudioResult {
    #[pyo3(get)]
    pub model_id: String,
    #[pyo3(get)]
    pub duration_s: f32,
    #[pyo3(get)]
    pub sample_rate: u32,
    #[pyo3(get)]
    pub window_s: f32,
    #[pyo3(get)]
    pub stride_s: f32,
    #[pyo3(get)]
    pub processing_time_ms: f32,
    #[pyo3(get)]
    pub segments: Vec<AudioSegment>,
}

#[pymethods]
impl AudioResult {
    fn __repr__(&self) -> String {
        format!(
            "AudioResult(model_id='{}', duration={:.2}s, stride={:.3}s, segments={})",
            self.model_id,
            self.duration_s,
            self.stride_s,
            self.segments.len()
        )
    }

    fn __len__(&self) -> usize {
        self.segments.len()
    }
}

/// Model metadata (id, type, default status).
#[pyclass(frozen, module = "sparrow_engine._sparrow_engine_core")]
#[derive(Clone)]
pub struct ModelInfo {
    #[pyo3(get)]
    pub id: String,
    #[pyo3(get)]
    pub model_type: String,
    /// Manifest [model].subtype hint ("standard" | "overhead"). Phase D B-10:
    /// surfaces the `[model].subtype = "overhead"` distinction (HerdNet, OWL-T)
    /// that was folded into ModelType::OverheadDetector at the native layer.
    /// Derived from `model_type` since native sparrow_engine::ModelInfo does
    /// not carry subtype directly; widening the native struct is deferred to
    /// Phase E/F (see Phase D reviewer cross-scope finding 1).
    #[pyo3(get)]
    pub subtype: String,
    #[pyo3(get)]
    pub default: bool,
    #[pyo3(get)]
    pub version: Option<String>,
    #[pyo3(get)]
    pub description: Option<String>,
    #[pyo3(get)]
    pub onnx_sha256: Option<String>,
    #[pyo3(get)]
    pub onnx_size_bytes: Option<u64>,
}

#[pymethods]
impl ModelInfo {
    fn __repr__(&self) -> String {
        format!(
            "ModelInfo(id='{}', model_type='{}', subtype='{}', default={})",
            self.id, self.model_type, self.subtype, self.default
        )
    }
}

// ---------------------------------------------------------------------------
// Type conversions: sparrow_engine native → PyO3
// ---------------------------------------------------------------------------

fn convert_bbox(b: &sparrow_engine::BBox) -> BBox {
    BBox {
        x_min: b.x_min,
        y_min: b.y_min,
        x_max: b.x_max,
        y_max: b.y_max,
    }
}

fn convert_detection(d: &sparrow_engine::Detection) -> Detection {
    Detection {
        label: d.label.clone(),
        label_id: d.label_id,
        confidence: d.confidence,
        bbox: convert_bbox(&d.bbox),
    }
}

fn convert_classification(c: &sparrow_engine::Classification) -> Classification {
    Classification {
        label: c.label.clone(),
        label_id: c.label_id,
        confidence: c.confidence,
    }
}

fn convert_audio_class(c: &sparrow_engine::AudioClass) -> AudioClass {
    AudioClass {
        class_idx: c.class_idx,
        label: c.label.clone(),
        probability: c.probability,
    }
}

fn convert_audio_segment(s: &sparrow_engine::AudioSegment) -> AudioSegment {
    AudioSegment {
        start_time_s: s.start_time_s,
        end_time_s: s.end_time_s,
        confidence: s.confidence,
        classes: s.classes.iter().map(convert_audio_class).collect(),
    }
}

fn convert_model_type(mt: ModelType) -> &'static str {
    match mt {
        ModelType::Detector => "detector",
        ModelType::OverheadDetector => "overhead_detector",
        ModelType::Classifier => "classifier",
        ModelType::AudioDetector => "audio_detector",
        ModelType::AudioClassifier => "audio_classifier",
    }
}

fn convert_model_info(m: &sparrow_engine::ModelInfo) -> ModelInfo {
    // Phase D B-10: derive subtype string from model_type. Native
    // sparrow_engine::ModelInfo does not carry the manifest subtype directly;
    // only ModelType::OverheadDetector encodes `[model].subtype = "overhead"`
    // (per `derive_model_type` in sparrow-engine-types). This mapping is
    // lossless given the current ModelSubtype = {Standard, Overhead} enum.
    // Widening the native struct is deferred (cross-scope finding 1).
    let subtype = match m.model_type {
        ModelType::OverheadDetector => "overhead",
        _ => "standard",
    }
    .to_owned();
    ModelInfo {
        id: m.id.clone(),
        model_type: convert_model_type(m.model_type).to_owned(),
        subtype,
        default: m.default,
        version: m.version.clone(),
        description: m.description.clone(),
        onnx_sha256: m.onnx_sha256.clone(),
        onnx_size_bytes: m.onnx_size_bytes,
    }
}

// ---------------------------------------------------------------------------
// Reverse conversions: PyO3 → sparrow_engine native (for viz/export)
// ---------------------------------------------------------------------------

fn pydetection_to_native(d: &Detection) -> sparrow_engine::Detection {
    sparrow_engine::Detection {
        bbox: sparrow_engine::BBox {
            x_min: d.bbox.x_min,
            y_min: d.bbox.y_min,
            x_max: d.bbox.x_max,
            y_max: d.bbox.y_max,
        },
        label: d.label.clone(),
        label_id: d.label_id,
        confidence: d.confidence,
    }
}

fn pyclassification_to_native(c: &Classification) -> sparrow_engine::Classification {
    sparrow_engine::Classification {
        label: c.label.clone(),
        label_id: c.label_id,
        confidence: c.confidence,
    }
}

fn pyaudio_class_to_native(c: &AudioClass) -> sparrow_engine::AudioClass {
    sparrow_engine::AudioClass {
        class_idx: c.class_idx,
        label: c.label.clone(),
        probability: c.probability,
    }
}

fn pyaudio_segment_to_native(s: &AudioSegment) -> sparrow_engine::AudioSegment {
    sparrow_engine::AudioSegment {
        start_time_s: s.start_time_s,
        end_time_s: s.end_time_s,
        confidence: s.confidence,
        classes: s.classes.iter().map(pyaudio_class_to_native).collect(),
    }
}

fn pyaudio_to_native(r: &AudioResult) -> sparrow_engine::AudioDetectResult {
    sparrow_engine::AudioDetectResult {
        segments: r.segments.iter().map(pyaudio_segment_to_native).collect(),
        duration_s: r.duration_s,
        sample_rate: r.sample_rate,
        processing_time_ms: r.processing_time_ms,
    }
}

fn pydetect_to_native(r: &DetectResult) -> sparrow_engine::DetectResult {
    sparrow_engine::DetectResult {
        detections: r.detections.iter().map(pydetection_to_native).collect(),
        image_width: r.image_size.0,
        image_height: r.image_size.1,
        processing_time_ms: r.processing_time_ms,
    }
}

fn pyclassify_to_native(r: &ClassifyResult) -> sparrow_engine::ClassifyResult {
    sparrow_engine::ClassifyResult {
        classifications: r
            .classifications
            .iter()
            .map(pyclassification_to_native)
            .collect(),
        image_width: r.image_size.0,
        image_height: r.image_size.1,
        processing_time_ms: r.processing_time_ms,
    }
}

fn pypipeline_to_native(r: &PipelineResult) -> sparrow_engine::PipelineResult {
    sparrow_engine::PipelineResult {
        pipeline_id: r.pipeline_id.clone(),
        detections: r
            .detections
            .iter()
            .map(|pd| sparrow_engine::PipelineDetection {
                detection: pydetection_to_native(&pd.detection),
                classification: pd.classification.as_ref().map(pyclassification_to_native),
            })
            .collect(),
        image_width: r.image_size.0,
        image_height: r.image_size.1,
        processing_time_ms: r.processing_time_ms,
    }
}

/// Compute the longest common directory prefix of a set of paths.
fn longest_common_prefix(paths: &[PathBuf]) -> PathBuf {
    if paths.is_empty() {
        return PathBuf::new();
    }
    let first = match paths[0].parent() {
        Some(p) => p.to_path_buf(),
        None => return PathBuf::new(),
    };
    let mut prefix = first;
    for path in &paths[1..] {
        let parent = path.parent().unwrap_or(Path::new(""));
        while !parent.starts_with(&prefix) {
            if !prefix.pop() {
                return PathBuf::new();
            }
        }
    }
    prefix
}

fn visualization_relative_path(input_path: &Path, common_prefix: &Path) -> PathBuf {
    let candidate = if common_prefix.as_os_str().is_empty() {
        input_path.to_path_buf()
    } else {
        input_path
            .strip_prefix(common_prefix)
            .map(Path::to_path_buf)
            .unwrap_or_else(|_| PathBuf::from(input_path.file_name().unwrap_or_default()))
    };

    if candidate.is_absolute() {
        PathBuf::from(input_path.file_name().unwrap_or_default())
    } else {
        candidate
    }
}

// ---------------------------------------------------------------------------
// Progress callback helper (S6)
// ---------------------------------------------------------------------------
//
// The 4 batch-capable inference entry points — `detect`, `classify`,
// `detect_audio`, `pipeline` — accept an optional `progress_callback` kwarg.
//
// Contract:
//
//   Called once per input file, AFTER the file's inference attempt resolves
//   (success OR failure). Arguments: `(index, total, filename)`, where
//   `index` is 0-based. The last call of a batch of N files uses
//   `index == N - 1`, so `files[index]` is always valid.
//
// If the callback raises any exception, we propagate it to Python — the batch
// is aborted and the caller sees the original exception. This matches the
// Python stdlib convention for progress hooks (e.g. `urlretrieve`) and lets
// users surface KeyboardInterrupt to halt long batches.
//
// GIL dance: the inference loop runs inside `py.allow_threads` so the GIL is
// released during ORT work. `invoke_progress` re-acquires it via
// `Python::with_gil`, calls the Python callable, and the outer `?` propagates
// any resulting PyErr back through the closure's `PyResult<Vec<_>>` return.
fn invoke_progress(
    cb: Option<&PyObject>,
    index: usize,
    total: usize,
    filename: &str,
) -> PyResult<()> {
    if let Some(cb) = cb {
        Python::with_gil(|py| -> PyResult<()> {
            cb.call1(py, (index, total, filename))?;
            Ok(())
        })?;
    }
    Ok(())
}

fn device_from_str(s: &str) -> PyResult<Device> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(Device::Auto),
        "cpu" => Ok(Device::Cpu),
        "cuda" | "cuda:0" | "gpu" => Ok(Device::Cuda(0)),
        s if s.starts_with("cuda:") => {
            let idx: u32 = s[5..]
                .parse()
                .map_err(|_| PyRuntimeError::new_err(format!("Invalid CUDA device index: {s}")))?;
            Ok(Device::Cuda(idx))
        }
        _ => Err(PyRuntimeError::new_err(format!(
            "Unknown device '{s}'. Use: auto, cpu, cuda, cuda:N"
        ))),
    }
}

// ---------------------------------------------------------------------------
// PyEngine — wraps sparrow_engine::Engine
// ---------------------------------------------------------------------------

/// ORT inference engine. Thread-safe — GIL released during inference.
#[pyclass(module = "sparrow_engine._sparrow_engine_core")]
pub struct PyEngine {
    engine: Engine,
}

#[pymethods]
impl PyEngine {
    /// Create a new engine.
    ///
    /// Args:
    ///     device: ``"auto"``, ``"cpu"``, ``"cuda"``, or ``"cuda:N"``
    ///     model_dir: Base directory containing model subdirectories
    #[new]
    fn new(device: &str, model_dir: &str) -> PyResult<Self> {
        let dev = device_from_str(device)?;
        let config = EngineConfig::new(dev, PathBuf::from(model_dir));
        let engine = Engine::new(config).map_err(to_pyerr)?;
        Ok(Self { engine })
    }

    /// Run object detection on a list of image paths.
    ///
    /// Models are auto-loaded on first use.
    #[pyo3(signature = (paths, model, threshold=None, max_detections=None, progress_callback=None))]
    fn detect(
        &self,
        py: Python<'_>,
        paths: Vec<String>,
        model: &str,
        threshold: Option<f32>,
        max_detections: Option<u32>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Vec<DetectResult>> {
        let engine = &self.engine;
        let model_id = model.to_owned();
        let opts = DetectOpts {
            confidence_threshold: threshold,
            max_detections,
        };
        let total = paths.len();

        py.allow_threads(move || {
            let handle = engine.get_or_load_model(&model_id).map_err(to_pyerr)?;
            let model_type = handle.model_type();
            let mut results = Vec::with_capacity(paths.len());
            let mut errors = 0usize;
            for (i, path) in paths.iter().enumerate() {
                let input = ImageInput::FilePath(PathBuf::from(path));
                match sparrow_engine::detect::detect(&handle, &input, &opts) {
                    Ok(r) => {
                        results.push(DetectResult {
                            model_type,
                            model_id: model_id.clone(),
                            image_size: (r.image_width, r.image_height),
                            processing_time_ms: r.processing_time_ms,
                            detections: r.detections.iter().map(convert_detection).collect(),
                        });
                    }
                    Err(e) => {
                        tracing::warn!(target: "sparrow_engine::python", "skipping {path}: {e}");
                        errors += 1;
                    }
                }
                invoke_progress(progress_callback.as_ref(), i, total, path)?;
            }
            if errors > 0 {
                tracing::warn!(target: "sparrow_engine::python", "{errors} file(s) skipped due to errors");
            }
            if errors == total && total > 0 {
                return Err(SparrowEngineError::new_err("All files failed processing."));
            }
            Ok(results)
        })
    }

    /// Run image classification on a list of image paths.
    #[pyo3(signature = (paths, model, top_k=None, progress_callback=None))]
    fn classify(
        &self,
        py: Python<'_>,
        paths: Vec<String>,
        model: &str,
        top_k: Option<u32>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Vec<ClassifyResult>> {
        let engine = &self.engine;
        let model_id = model.to_owned();
        let opts = ClassifyOpts { top_k };
        let total = paths.len();

        py.allow_threads(move || {
            let handle = engine.get_or_load_model(&model_id).map_err(to_pyerr)?;
            let mut results = Vec::with_capacity(paths.len());
            let mut errors = 0usize;
            for (i, path) in paths.iter().enumerate() {
                let input = ImageInput::FilePath(PathBuf::from(path));
                match sparrow_engine::classify::classify(&handle, &input, &opts) {
                    Ok(r) => {
                        results.push(ClassifyResult {
                            model_id: model_id.clone(),
                            image_size: (r.image_width, r.image_height),
                            processing_time_ms: r.processing_time_ms,
                            classifications: r
                                .classifications
                                .iter()
                                .map(convert_classification)
                                .collect(),
                        });
                    }
                    Err(e) => {
                        tracing::warn!(target: "sparrow_engine::python", "skipping {path}: {e}");
                        errors += 1;
                    }
                }
                invoke_progress(progress_callback.as_ref(), i, total, path)?;
            }
            if errors > 0 {
                tracing::warn!(target: "sparrow_engine::python", "{errors} file(s) skipped due to errors");
            }
            if errors == total && total > 0 {
                return Err(SparrowEngineError::new_err("All files failed processing."));
            }
            Ok(results)
        })
    }

    /// Run audio detection on a list of audio file paths.
    ///
    /// `stride_s` and `segment_duration_s` are runtime overrides for the
    /// manifest defaults. Stride is always engine-controlled. Segment
    /// duration is honored by mel-spectrogram audio models with dynamic
    /// ONNX time-axis (e.g. md-audiobirds-v1); silently ignored by
    /// raw-audio classifiers whose ONNX input is fixed-size (e.g.
    /// perch-v2's `[batch, 160000]`) — the window is an upstream
    /// architecture constraint for those models.
    #[pyo3(signature = (paths, model, threshold=None, stride_s=None, segment_duration_s=None, progress_callback=None))]
    #[allow(clippy::too_many_arguments)]
    fn detect_audio(
        &self,
        py: Python<'_>,
        paths: Vec<String>,
        model: &str,
        threshold: Option<f32>,
        stride_s: Option<f32>,
        segment_duration_s: Option<f32>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Vec<AudioResult>> {
        // Validate user overrides up-front, matching sparrow-engine-server
        // (`handlers/audio.rs`) and the CLI (`detect-audio --stride/--segment-duration`).
        if let Some(s) = stride_s {
            if !s.is_finite() || s <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "stride_s must be a finite positive number",
                ));
            }
        }
        if let Some(d) = segment_duration_s {
            if !d.is_finite() || d <= 0.0 {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "segment_duration_s must be a finite positive number",
                ));
            }
        }
        let engine = &self.engine;
        let model_id = model.to_owned();
        let opts = AudioDetectOpts {
            confidence_threshold: threshold,
            segment_duration_s,
            stride_s,
        };
        let total = paths.len();

        py.allow_threads(move || {
            let handle = engine.get_or_load_model(&model_id).map_err(to_pyerr)?;
            let (window_s, effective_stride_s) = resolve_audio_window_stride(
                handle.audio_window_stride(),
                stride_s,
                segment_duration_s,
            );
            let mut results = Vec::with_capacity(paths.len());
            let mut errors = 0usize;
            for (i, path) in paths.iter().enumerate() {
                let input = AudioInput::FilePath(PathBuf::from(path));
                match sparrow_engine::detect_audio::detect_audio(&handle, &input, &opts) {
                    Ok(r) => {
                        results.push(AudioResult {
                            model_id: model_id.clone(),
                            duration_s: r.duration_s,
                            sample_rate: r.sample_rate,
                            window_s,
                            stride_s: effective_stride_s,
                            processing_time_ms: r.processing_time_ms,
                            segments: r.segments.iter().map(convert_audio_segment).collect(),
                        });
                    }
                    Err(e) => {
                        tracing::warn!(target: "sparrow_engine::python", "skipping {path}: {e}");
                        errors += 1;
                    }
                }
                invoke_progress(progress_callback.as_ref(), i, total, path)?;
            }
            if errors > 0 {
                tracing::warn!(target: "sparrow_engine::python", "{errors} file(s) skipped due to errors");
            }
            if errors == total && total > 0 {
                return Err(SparrowEngineError::new_err("All files failed processing."));
            }
            Ok(results)
        })
    }

    /// Run ad-hoc detect+classify pipeline on a list of image paths.
    // Clippy's `too_many_arguments` default is 7; the `#[pyo3(signature = ...)]`
    // on this method intentionally exposes each Python kwarg as a separate
    // Rust argument so the Python-visible signature stays
    // `pipeline(paths, detector, classifier, threshold=None, top_k=None,
    // progress_callback=None)`. Packing the trailing kwargs into a struct
    // would break the Python contract; the lint is silenced at this boundary.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (paths, detector, classifier, threshold=None, top_k=None, progress_callback=None))]
    fn pipeline(
        &self,
        py: Python<'_>,
        paths: Vec<String>,
        detector: &str,
        classifier: &str,
        threshold: Option<f32>,
        top_k: Option<u32>,
        progress_callback: Option<PyObject>,
    ) -> PyResult<Vec<PipelineResult>> {
        let engine = &self.engine;
        let det_id = detector.to_owned();
        let cls_id = classifier.to_owned();
        let d_opts = DetectOpts {
            confidence_threshold: threshold,
            max_detections: None,
        };
        validate_pipeline_ids(engine, &det_id, &cls_id).map_err(to_pyerr)?;

        let c_opts = ClassifyOpts { top_k };
        let total = paths.len();

        py.allow_threads(move || {
            let detector_handle = engine.get_or_load_model(&det_id).map_err(to_pyerr)?;
            let detector_model_type = detector_handle.model_type();
            let _classifier_handle = engine.get_or_load_model(&cls_id).map_err(to_pyerr)?;
            let mut results = Vec::with_capacity(paths.len());
            let mut errors = 0usize;
            for (i, path) in paths.iter().enumerate() {
                let input = ImageInput::FilePath(PathBuf::from(path));
                match sparrow_engine::pipeline::run_pipeline_adhoc(
                    engine, &input, &det_id, &cls_id, &d_opts, &c_opts,
                ) {
                    Ok(r) => {
                        results.push(PipelineResult {
                            pipeline_id: r.pipeline_id.clone(),
                            image_size: (r.image_width, r.image_height),
                            processing_time_ms: r.processing_time_ms,
                            detections: r
                                .detections
                                .iter()
                                .map(|pd| PipelineDetection {
                                    detection: convert_detection(&pd.detection),
                                    classification: pd
                                        .classification
                                        .as_ref()
                                        .map(convert_classification),
                                })
                                .collect(),
                            model_type: detector_model_type,
                        });
                    }
                    Err(e) => {
                        tracing::warn!(target: "sparrow_engine::python", "skipping {path}: {e}");
                        errors += 1;
                    }
                }
                invoke_progress(progress_callback.as_ref(), i, total, path)?;
            }
            if errors > 0 {
                tracing::warn!(target: "sparrow_engine::python", "{errors} file(s) skipped due to errors");
            }
            if errors == total && total > 0 {
                return Err(SparrowEngineError::new_err("All files failed processing."));
            }
            Ok(results)
        })
    }

    /// List all available models in the model directory.
    fn list_models(&self) -> Vec<ModelInfo> {
        self.engine
            .list_available_models()
            .iter()
            .map(convert_model_info)
            .collect()
    }

    /// Get info for a specific model by ID.
    fn model_info(&self, model_id: &str) -> PyResult<ModelInfo> {
        let models = self.engine.list_available_models();
        models
            .iter()
            .find(|m| m.id == model_id)
            .map(convert_model_info)
            .ok_or_else(|| SparrowEngineError::new_err(format!("Model not found: {model_id}")))
    }

    /// Return the active compute device as a string.
    fn active_device(&self) -> String {
        self.engine.active_device().to_string()
    }

    /// Compute SHA-256 hash of a file.
    fn hash_file(&self, path: &str) -> PyResult<String> {
        hash_file(path)
    }

    /// Classify an image as day or night.
    fn day_night(&self, py: Python<'_>, path: &str) -> PyResult<PyObject> {
        day_night(py, path)
    }

    /// Verify a model's integrity against manifest checksums.
    ///
    /// Convenience wrapper that uses the engine's configured `model_dir`. Callers
    /// who already have a `PyEngine` in hand can use this; otherwise prefer the
    /// top-level `sparrow_engine.verify_model(model_id, model_dir=None)` which does not
    /// require engine initialization (see `sparrow_engine/__init__.py`).
    fn verify_model(&self, py: Python<'_>, model_id: &str) -> PyResult<PyObject> {
        let dir = &self.engine.config().model_dir;
        let result = sparrow_engine::catalog::verify_model(dir, model_id).map_err(to_pyerr)?;
        verify_result_to_dict(py, result)
    }

    fn __repr__(&self) -> String {
        format!(
            "PyEngine(device='{}', model_dir='{}')",
            self.engine.active_device(),
            self.engine.config().model_dir.display()
        )
    }
}

// ---------------------------------------------------------------------------
// Module-level functions
// ---------------------------------------------------------------------------

/// Compute SHA-256 hash of a file. No engine initialization required.
#[pyfunction]
fn hash_file(path: &str) -> PyResult<String> {
    sparrow_engine::hash::hash_file(Path::new(path)).map_err(to_pyerr)
}

/// Classify an image as day or night. No engine initialization required.
#[pyfunction]
fn day_night(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let data = std::fs::read(path)
        .map_err(|e| SparrowEngineError::new_err(format!("failed to read '{path}': {e}")))?;
    let result = sparrow_engine::daynight::day_night(&data).map_err(to_pyerr)?;
    let dict = pyo3::types::PyDict::new(py);
    let class_str = match result.classification {
        sparrow_engine::daynight::DayNight::Day => "day",
        sparrow_engine::daynight::DayNight::Night => "night",
    };
    dict.set_item("classification", class_str)?;
    dict.set_item("mean_brightness", result.mean_brightness)?;
    Ok(dict.into())
}

/// Convert a VerifyResult to a Python dict.
fn verify_result_to_dict(
    py: Python<'_>,
    result: sparrow_engine::catalog::VerifyResult,
) -> PyResult<PyObject> {
    let dict = pyo3::types::PyDict::new(py);
    match result {
        sparrow_engine::catalog::VerifyResult::Ok => {
            dict.set_item("status", "ok")?;
        }
        sparrow_engine::catalog::VerifyResult::NoChecksum => {
            dict.set_item("status", "no_checksum")?;
        }
        sparrow_engine::catalog::VerifyResult::SizeMismatch { expected, actual } => {
            dict.set_item("status", "size_mismatch")?;
            dict.set_item("expected_size", expected)?;
            dict.set_item("actual_size", actual)?;
        }
        sparrow_engine::catalog::VerifyResult::ChecksumMismatch { expected, actual } => {
            dict.set_item("status", "checksum_mismatch")?;
            dict.set_item("expected_hash", expected)?;
            dict.set_item("actual_hash", actual)?;
        }
    }
    Ok(dict.into())
}

/// Verify model integrity against manifest checksums. No engine initialization required.
#[pyfunction]
fn verify_model(py: Python<'_>, model_dir: &str, model_id: &str) -> PyResult<PyObject> {
    let dir = Path::new(model_dir);
    let result = sparrow_engine::catalog::verify_model(dir, model_id).map_err(to_pyerr)?;
    verify_result_to_dict(py, result)
}

/// Summarize detection results. No engine initialization required.
#[pyfunction]
fn summarize(py: Python<'_>, results: Vec<Py<DetectResult>>) -> PyResult<PyObject> {
    let native_results: Vec<sparrow_engine::DetectResult> = results
        .iter()
        .map(|r_py| pydetect_to_native(r_py.get()))
        .collect();

    let summary = sparrow_engine::stats::summarize_detections(&native_results);
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("total_images", summary.total_images)?;
    dict.set_item("images_with_detections", summary.images_with_detections)?;
    dict.set_item("empty_images", summary.empty_images)?;
    dict.set_item("total_detections", summary.total_detections)?;
    dict.set_item("confidence_min", summary.confidence_min)?;
    dict.set_item("confidence_max", summary.confidence_max)?;
    dict.set_item("confidence_mean", summary.confidence_mean)?;
    let per_cat = pyo3::types::PyDict::new(py);
    for (label, stats) in &summary.per_category {
        let cat_dict = pyo3::types::PyDict::new(py);
        cat_dict.set_item("count", stats.count)?;
        cat_dict.set_item("confidence_mean", stats.confidence_mean)?;
        cat_dict.set_item("confidence_min", stats.confidence_min)?;
        cat_dict.set_item("confidence_max", stats.confidence_max)?;
        per_cat.set_item(label, cat_dict)?;
    }
    dict.set_item("per_category", per_cat)?;
    Ok(dict.into())
}

/// Filename extension to use when saving a viz image encoded as `fmt`.
///
/// PS3 invariant (see reviewer commit `e473207`): `visualize()`'s format
/// selector emits only `Jpeg` or `Png`. The catch-all arm returns `"png"` as
/// a safe lossless fallback rather than `unreachable!()` — the PyO3 zero-panic
/// rule (see rust.md) forbids panicking in functions reachable from exposed
/// Python APIs, and `image::ImageFormat` is `#[non_exhaustive]` so the arm is
/// structurally required.
///
/// If PS3 is ever extended to emit a new format, update both:
///   1. The format-selection match inside `visualize()` (JPG/JPEG → Jpeg, `_` → Png).
///   2. This mapping, so the saved file's extension matches its bytes.
///
/// The companion tests in `mod tests` pin the PS3 emission set (Jpeg, Png)
/// and will need to be extended alongside the production change.
fn viz_output_extension(fmt: image::ImageFormat) -> &'static str {
    match fmt {
        image::ImageFormat::Jpeg => "jpg",
        image::ImageFormat::Png => "png",
        _ => "png",
    }
}

fn detect_visualization_model_type(result: &DetectResult) -> ModelType {
    result.model_type
}

fn pipeline_visualization_model_type(result: &PipelineResult) -> ModelType {
    result.model_type
}

fn bbox_visualization_model_type() -> ModelType {
    ModelType::Detector
}

const VISUALIZE_AUDIO_UNSUPPORTED_MESSAGE: &str =
    "visualize() does not support AudioResult — use visualize_audio() for audio";
const DEFAULT_AUDIO_WINDOW_S: f32 = 1.0;
const DEFAULT_AUDIO_STRIDE_S: f32 = 0.3;
// Keep this in sync with sparrow-engine-cli/src/main.rs VIZ_MERGE_THRESHOLD.
const VIZ_MERGE_THRESHOLD: f32 = 0.9;

fn visualize_render_opts(
    model_type: ModelType,
    show_labels: bool,
) -> sparrow_engine::viz::RenderOpts {
    sparrow_engine::viz::RenderOpts {
        model_type,
        show_labels,
        ..Default::default()
    }
}

/// Render bounding box visualizations. No engine initialization required.
///
/// # Error semantics (batch accumulation)
///
/// `visualize()` processes every `(path, result)` entry in `items` before
/// raising. Per-item failures (image decode errors, disk write failures,
/// directory creation failures) are counted but do NOT stop the batch. After
/// every item is attempted, if any failures occurred the function raises a
/// single `SparrowEngineError` summarizing the total count — the partially-rendered
/// `list[bytes]` is discarded.
///
/// # Partial disk state on failure
///
/// When `output_dir` is set and an error is eventually raised, files for
/// entries that succeeded BEFORE the failure have already been written to
/// disk. The caller sees a `SparrowEngineError` but the filesystem retains the
/// partial output. Callers that need all-or-nothing semantics must either:
/// 1. Call `visualize()` without `output_dir`, then write the returned bytes
///    atomically themselves, or
/// 2. Clean up the target directory after a raise.
///
/// This is intentional — the batch-accumulate design matches the CLI's
/// per-file error continuation (see sparrow-engine-cli `--visualize --output-dir`).
#[pyfunction]
#[pyo3(signature = (items, output_dir=None, show_labels=false))]
fn visualize(
    py: Python<'_>,
    items: Vec<(String, PyObject)>,
    output_dir: Option<String>,
    show_labels: bool,
) -> PyResult<Vec<Py<PyBytes>>> {
    // Validate output_dir upfront (spec: hard error before processing).
    if let Some(ref out_dir) = output_dir {
        let dir = Path::new(out_dir);
        std::fs::create_dir_all(dir).map_err(|e| {
            SparrowEngineError::new_err(format!(
                "cannot create output directory '{}': {e}",
                dir.display()
            ))
        })?;
    }

    let mut png_list: Vec<Py<PyBytes>> = Vec::with_capacity(items.len());
    let mut errors = 0u32;

    // Collect all input paths to compute common prefix for directory mirroring.
    let input_paths: Vec<PathBuf> = items.iter().map(|(p, _)| PathBuf::from(p)).collect();
    let common_prefix = if output_dir.is_some() {
        longest_common_prefix(&input_paths)
    } else {
        PathBuf::new()
    };

    for (path_str, result_obj) in &items {
        // Determine result type and convert to annotations.
        let (annotations, model_type) = if result_obj.bind(py).is_instance_of::<DetectResult>() {
            let r: Py<DetectResult> = result_obj.extract(py)?;
            let r = r.get();
            let native = pydetect_to_native(r);
            (
                sparrow_engine::viz::detections_to_annotations(&native),
                detect_visualization_model_type(r),
            )
        } else if result_obj.bind(py).is_instance_of::<ClassifyResult>() {
            let r: Py<ClassifyResult> = result_obj.extract(py)?;
            let r = r.get();
            let native = pyclassify_to_native(r);
            (
                sparrow_engine::viz::classifications_to_annotations(&native),
                bbox_visualization_model_type(),
            )
        } else if result_obj.bind(py).is_instance_of::<PipelineResult>() {
            let r: Py<PipelineResult> = result_obj.extract(py)?;
            let r = r.get();
            let native = pypipeline_to_native(r);
            (
                sparrow_engine::viz::pipeline_to_annotations(&native),
                pipeline_visualization_model_type(r),
            )
        } else if result_obj.bind(py).is_instance_of::<AudioResult>() {
            return Err(SparrowEngineError::new_err(
                VISUALIZE_AUDIO_UNSUPPORTED_MESSAGE,
            ));
        } else {
            return Err(SparrowEngineError::new_err(
                "visualize() expects DetectResult, ClassifyResult, or PipelineResult",
            ));
        };

        // Load image and render (release GIL for I/O + render).
        // Output format: JPEG in → JPEG out, PNG in → PNG out, unknown → PNG (lossless fallback).
        let path = path_str.clone();
        let render_result = py.allow_threads(
            move || -> std::result::Result<(Vec<u8>, image::ImageFormat), String> {
                let img = image::open(&path).map_err(|e| format!("{e}"))?;
                let opts = visualize_render_opts(model_type, show_labels);
                let rendered = sparrow_engine::viz::render(&img, &annotations, &opts);
                let fmt = match Path::new(&path)
                    .extension()
                    .and_then(|e| e.to_str())
                    .map(|e| e.to_lowercase())
                    .as_deref()
                {
                    Some("jpg") | Some("jpeg") => image::ImageFormat::Jpeg,
                    _ => image::ImageFormat::Png,
                };
                let mut buf = Vec::new();
                rendered
                    .write_to(&mut Cursor::new(&mut buf), fmt)
                    .map_err(|e| format!("{e}"))?;
                Ok((buf, fmt))
            },
        );

        match render_result {
            Ok((buf, fmt)) => {
                // Save to disk if output_dir is set.
                if let Some(ref out_dir) = output_dir {
                    let input_path = Path::new(path_str);
                    let rel = visualization_relative_path(input_path, &common_prefix);
                    let stem = rel.file_stem().unwrap_or_default().to_string_lossy();
                    // Derive output extension from the actual encoded format, not the input
                    // path extension. Post-PS3, BMP/TIFF/WEBP/unknown inputs produce PNG
                    // bytes; reusing the raw input extension would yield a MIME mismatch
                    // (e.g. `.BMP` holding PNG bytes) and mixed case with the lowercased
                    // `fmt` detection.
                    let output_ext = viz_output_extension(fmt);
                    let parent = rel.parent().unwrap_or(Path::new(""));
                    let out_path = Path::new(out_dir)
                        .join(parent)
                        .join(format!("{stem}_viz.{output_ext}"));
                    if let Some(p) = out_path.parent() {
                        if let Err(e) = std::fs::create_dir_all(p) {
                            tracing::warn!(
                                target: "sparrow_engine::python",
                                "failed to create dir {}: {e}",
                                p.display()
                            );
                        }
                    }
                    if let Err(e) = std::fs::write(&out_path, &buf) {
                        tracing::warn!(
                            target: "sparrow_engine::python",
                            "failed to save visualization for {path_str}: {e}"
                        );
                        errors += 1;
                    }
                }
                png_list.push(PyBytes::new(py, &buf).unbind());
            }
            Err(e) => {
                tracing::warn!(
                    target: "sparrow_engine::python",
                    "visualization failed for {path_str}: {e}"
                );
                errors += 1;
            }
        }
    }

    if errors > 0 {
        return Err(SparrowEngineError::new_err(format!(
            "{errors} of {} visualization(s) failed",
            items.len()
        )));
    }
    Ok(png_list)
}

fn resolve_audio_window_stride(
    manifest_window_stride: Option<(f32, f32)>,
    stride_s: Option<f32>,
    segment_duration_s: Option<f32>,
) -> (f32, f32) {
    let (manifest_window_s, manifest_stride_s) =
        manifest_window_stride.unwrap_or((DEFAULT_AUDIO_WINDOW_S, DEFAULT_AUDIO_STRIDE_S));
    (
        segment_duration_s.unwrap_or(manifest_window_s),
        stride_s.unwrap_or(manifest_stride_s),
    )
}

fn audio_result_visualization_timing(result: &AudioResult) -> (f32, f32) {
    (result.window_s, result.stride_s)
}

fn derive_audio_visualization_ranges(
    segments: &[sparrow_engine::AudioSegment],
    duration_s: f32,
    stride_s: f32,
) -> Vec<sparrow_engine::AudioRange> {
    let slots = sparrow_engine::viz::segments_to_overlap_mean_slots(segments, duration_s, stride_s);
    let high_conf_slots: Vec<sparrow_engine::AudioSegment> = slots
        .into_iter()
        .filter(|s| s.confidence >= VIZ_MERGE_THRESHOLD)
        .collect();
    sparrow_engine::detect_audio::merge_segments(&high_conf_slots, stride_s + 1e-3)
}

/// Render audio detection visualization layers for a batch.
///
/// Like `visualize()` but for `AudioResult`. Requires an initialized engine
/// so the binding can recover each result's audio preprocessing config from
/// `model_id`; each `AudioResult` carries the effective window/stride used
/// during detection.
#[pyfunction]
#[pyo3(signature = (engine, items, output_dir=None, smooth=false, show_windows=false, show_ranges=true))]
fn visualize_audio(
    py: Python<'_>,
    engine: &PyEngine,
    items: Vec<(String, PyObject)>,
    output_dir: Option<String>,
    smooth: bool,
    show_windows: bool,
    show_ranges: bool,
) -> PyResult<Vec<Vec<Py<PyBytes>>>> {
    if let Some(ref out_dir) = output_dir {
        let dir = Path::new(out_dir);
        std::fs::create_dir_all(dir).map_err(|e| {
            SparrowEngineError::new_err(format!(
                "cannot create output directory '{}': {e}",
                dir.display()
            ))
        })?;
    }

    let input_paths: Vec<PathBuf> = items.iter().map(|(p, _)| PathBuf::from(p)).collect();
    let common_prefix = if output_dir.is_some() {
        longest_common_prefix(&input_paths)
    } else {
        PathBuf::new()
    };

    let engine_ref = &engine.engine;
    let mut batch_layers: Vec<Vec<Py<PyBytes>>> = Vec::with_capacity(items.len());
    let mut errors = 0u32;

    for (path_str, result_obj) in &items {
        if !result_obj.bind(py).is_instance_of::<AudioResult>() {
            tracing::warn!(
                target: "sparrow_engine::python",
                "audio visualization failed for {path_str}: expected AudioResult"
            );
            errors += 1;
            continue;
        }

        let result_py: Py<AudioResult> = match result_obj.extract(py) {
            Ok(result) => result,
            Err(e) => {
                tracing::warn!(
                    target: "sparrow_engine::python",
                    "audio visualization failed for {path_str}: {e}"
                );
                errors += 1;
                continue;
            }
        };
        let result = result_py.get();
        let model_id = result.model_id.clone();
        let duration_s = result.duration_s;
        let (window_s, stride_s) = audio_result_visualization_timing(result);
        let native = pyaudio_to_native(result);
        let audio_path = PathBuf::from(path_str);

        let render_result = py.allow_threads(
            move || -> std::result::Result<Vec<(&'static str, Vec<u8>)>, String> {
                let handle = engine_ref
                    .get_or_load_model(&model_id)
                    .map_err(|e| format!("failed to load model '{model_id}': {e}"))?;
                let cfg = handle.audio_preprocess_config().ok_or_else(|| {
                    format!("model '{model_id}' does not expose audio preprocessing config")
                })?;
                let spec = sparrow_engine::viz::render_mel_spectrogram(&audio_path, &cfg)
                    .map_err(|e| format!("failed to render mel spectrogram: {e}"))?;
                let ranges = if show_ranges {
                    Some(derive_audio_visualization_ranges(
                        &native.segments,
                        duration_s,
                        stride_s,
                    ))
                } else {
                    None
                };
                let opts = sparrow_engine::viz::AudioLayersOpts {
                    smooth,
                    show_windows,
                    window_s,
                    stride_s,
                };
                let rendered = sparrow_engine::viz::render_audio_layers(
                    &spec,
                    &native.segments,
                    ranges.as_deref(),
                    duration_s,
                    &opts,
                );

                let mut encoded_layers = Vec::with_capacity(rendered.len());
                for (layer_name, img) in rendered {
                    let mut buf = Vec::new();
                    img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
                        .map_err(|e| format!("failed to encode {layer_name} PNG: {e}"))?;
                    encoded_layers.push((layer_name, buf));
                }
                Ok(encoded_layers)
            },
        );

        match render_result {
            Ok(encoded_layers) => {
                let mut item_failed = false;
                if let Some(ref out_dir) = output_dir {
                    let input_path = Path::new(path_str);
                    let rel = visualization_relative_path(input_path, &common_prefix);
                    let stem = rel.file_stem().unwrap_or_default().to_string_lossy();
                    let parent = rel.parent().unwrap_or(Path::new(""));
                    for (layer_name, buf) in &encoded_layers {
                        let out_path = Path::new(out_dir)
                            .join(parent)
                            .join(format!("{stem}_{layer_name}.png"));
                        if let Some(p) = out_path.parent() {
                            if let Err(e) = std::fs::create_dir_all(p) {
                                tracing::warn!(
                                    target: "sparrow_engine::python",
                                    "failed to create dir {}: {e}",
                                    p.display()
                                );
                                item_failed = true;
                                continue;
                            }
                        }
                        if let Err(e) = std::fs::write(&out_path, buf) {
                            tracing::warn!(
                                target: "sparrow_engine::python",
                                "failed to save audio visualization layer for {path_str}: {e}"
                            );
                            item_failed = true;
                        }
                    }
                }
                if item_failed {
                    errors += 1;
                }
                batch_layers.push(
                    encoded_layers
                        .into_iter()
                        .map(|(_, buf)| PyBytes::new(py, &buf).unbind())
                        .collect(),
                );
            }
            Err(e) => {
                tracing::warn!(
                    target: "sparrow_engine::python",
                    "audio visualization failed for {path_str}: {e}"
                );
                errors += 1;
            }
        }
    }

    if errors > 0 {
        return Err(SparrowEngineError::new_err(format!(
            "{errors} of {} audio visualization(s) failed",
            items.len()
        )));
    }

    Ok(batch_layers)
}

/// Export detection/pipeline results to megadet, coco, or csv. No engine initialization required.
#[pyfunction]
#[pyo3(signature = (items, format, output=None, model_id=None))]
fn export_results(
    py: Python<'_>,
    items: Vec<(String, PyObject)>,
    format: &str,
    output: Option<String>,
    model_id: Option<String>,
) -> PyResult<String> {
    // Validate format upfront. The per-format model_id requirement is enforced
    // at point-of-use in the match arm below (see "megadet" arm's ok_or_else).
    if !matches!(format, "megadet" | "coco" | "csv") {
        return Err(SparrowEngineError::new_err(format!(
            "unsupported export format '{}' — expected 'megadet', 'coco', or 'csv'",
            format
        )));
    }

    // Convert all items to native DetectResult.
    let mut native_entries: Vec<(PathBuf, sparrow_engine::DetectResult)> =
        Vec::with_capacity(items.len());

    for (path_str, result_obj) in &items {
        let path = PathBuf::from(path_str);
        if result_obj.bind(py).is_instance_of::<DetectResult>() {
            let r: Py<DetectResult> = result_obj.extract(py)?;
            let r = r.get();
            native_entries.push((path, pydetect_to_native(r)));
        } else if result_obj.bind(py).is_instance_of::<PipelineResult>() {
            let r: Py<PipelineResult> = result_obj.extract(py)?;
            let r = r.get();
            let native_pr = pypipeline_to_native(r);
            let path_ref: &Path = &path;
            let converted = sparrow_engine::export::pipeline_results_to_detect_entries(&[(
                path_ref, &native_pr,
            )]);
            for (p, dr) in converted {
                native_entries.push((p, dr));
            }
        } else {
            return Err(SparrowEngineError::new_err(
                "export() supports DetectResult and PipelineResult only — AudioResult and ClassifyResult are not exportable",
            ));
        }
    }

    let entries: Vec<(&Path, &sparrow_engine::DetectResult)> = native_entries
        .iter()
        .map(|(p, r)| (p.as_path(), r))
        .collect();

    let mut buf = Vec::new();
    match format {
        "megadet" => {
            let mid = model_id.as_deref().ok_or_else(|| {
                SparrowEngineError::new_err("model_id is required for megadet format")
            })?;
            sparrow_engine::export::to_megadet(&entries, mid, &mut buf).map_err(to_pyerr)?;
        }
        "coco" => {
            sparrow_engine::export::to_coco(&entries, &mut buf).map_err(to_pyerr)?;
        }
        "csv" => {
            sparrow_engine::export::to_csv(&entries, &mut buf).map_err(to_pyerr)?;
        }
        _ => unreachable!(),
    }

    let result_str = String::from_utf8(buf)
        .map_err(|e| SparrowEngineError::new_err(format!("export produced invalid UTF-8: {e}")))?;

    if let Some(ref output_path) = output {
        std::fs::write(output_path, &result_str).map_err(|e| {
            SparrowEngineError::new_err(format!("failed to write export to '{output_path}': {e}"))
        })?;
    }

    Ok(result_str)
}

// ---------------------------------------------------------------------------
// Test-only helpers (S6)
// ---------------------------------------------------------------------------
//
// Exposed on the `_sparrow_engine_core` module with a leading underscore — conventional
// Python marker for "private, not part of the public API". `sparrow_engine/__init__.py`
// does NOT re-export these. They exist so the bridge wiring (pyo3-log target
// mapping, progress-callback GIL dance) can be verified in CI without a
// loaded ONNX model.

/// Emit a `tracing::warn!` on target `"sparrow_engine::python"`. Used by the Python
/// test suite to verify `pyo3-log` routes the event into the Python
/// `logging.getLogger("sparrow_engine.python")` logger (a child of `"sparrow_engine"`).
#[pyfunction]
fn _emit_test_warn(msg: &str) {
    tracing::warn!(target: "sparrow_engine::python", "{msg}");
}

/// Invoke the progress-callback helper `n` times with synthetic
/// `(index, total, filename)` arguments. Exercises the same
/// `Python::with_gil` path as the real inference loops, without needing
/// ORT. If `cb` raises on any index, the exception propagates — matching
/// real-batch behavior.
#[pyfunction]
fn _invoke_test_progress_callback(cb: PyObject, n: usize) -> PyResult<()> {
    for i in 0..n {
        let filename = format!("test_{i}.jpg");
        invoke_progress(Some(&cb), i, n, &filename)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

#[pymodule]
fn _sparrow_engine_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // S6: Bridge Rust `tracing` / `log` events into Python's `logging` module.
    //
    // Rust-side stdio writes are invisible inside Jupyter kernels (PyO3 #2247),
    // so diagnostics must route through Python's logging instead.
    // `tracing::warn!(target: "sparrow_engine::python", …)` emits a `log` record via
    // the `tracing/log` feature; `pyo3_log` converts `::` in the target to `.`
    // and delivers it to `logging.getLogger("sparrow_engine.python")` — a child of
    // `logging.getLogger("sparrow_engine")`.
    //
    // `try_init` is used (not `init`) so loading `sparrow_engine` after another Rust
    // extension that already installed a `log::Log` implementation doesn't
    // panic. Losing the race is a no-op; the first extension wins.
    let _ = pyo3_log::try_init();

    // Exception type
    m.add(
        "SparrowEngineError",
        m.py().get_type::<SparrowEngineError>(),
    )?;

    // Engine
    m.add_class::<PyEngine>()?;

    // Module-level functions
    m.add_function(wrap_pyfunction!(hash_file, m)?)?;
    m.add_function(wrap_pyfunction!(day_night, m)?)?;
    m.add_function(wrap_pyfunction!(verify_model, m)?)?;
    m.add_function(wrap_pyfunction!(summarize, m)?)?;
    m.add_function(wrap_pyfunction!(visualize, m)?)?;
    m.add_function(wrap_pyfunction!(visualize_audio, m)?)?;
    m.add_function(wrap_pyfunction!(export_results, m)?)?;

    // Test-only helpers (S6) — not re-exported by sparrow_engine/__init__.py
    m.add_function(wrap_pyfunction!(_emit_test_warn, m)?)?;
    m.add_function(wrap_pyfunction!(_invoke_test_progress_callback, m)?)?;

    // Result types
    m.add_class::<BBox>()?;
    m.add_class::<Detection>()?;
    m.add_class::<DetectResult>()?;
    m.add_class::<Classification>()?;
    m.add_class::<ClassifyResult>()?;
    m.add_class::<PipelineDetection>()?;
    m.add_class::<PipelineResult>()?;
    m.add_class::<AudioClass>()?;
    m.add_class::<AudioSegment>()?;
    m.add_class::<AudioResult>()?;
    m.add_class::<ModelInfo>()?;

    Ok(())
}

// ---------------------------------------------------------------------------
// Unit tests — pure Rust logic, no Python interpreter needed
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn native_model_info(id: &str, model_type: ModelType) -> NativeModelInfo {
        NativeModelInfo {
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

    fn audio_layer_bytes(
        layers: &[(&'static str, image::DynamicImage)],
        layer_name: &str,
    ) -> Vec<u8> {
        let (_, img) = layers
            .iter()
            .find(|(name, _)| *name == layer_name)
            .unwrap_or_else(|| panic!("missing audio layer {layer_name}"));
        img.to_rgba8().into_raw()
    }

    #[test]
    fn validate_pipeline_ids_rejects_known_incompatible_pair() {
        let available = vec![
            native_model_info("owl-t", ModelType::OverheadDetector),
            native_model_info("speciesnet-crop", ModelType::Classifier),
        ];
        let err = validate_pipeline_ids_from_available(&available, "owl-t", "speciesnet-crop")
            .unwrap_err();
        match err {
            sparrow_engine::SparrowEngineError::IncompatiblePipeline { reason, .. } => {
                assert!(
                    reason.contains("point detection"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected IncompatiblePipeline, got {other:?}"),
        }
    }

    #[test]
    fn validate_pipeline_ids_defers_unknown_ids_to_load_path() {
        let available = vec![native_model_info("speciesnet-crop", ModelType::Classifier)];
        validate_pipeline_ids_from_available(&available, "missing", "speciesnet-crop").unwrap();
    }

    #[test]
    fn visualization_relative_path_avoids_absolute_output_escape() {
        let rel = visualization_relative_path(Path::new("/mnt/a/bird.wav"), Path::new(""));
        assert_eq!(rel, PathBuf::from("bird.wav"));
        assert!(!rel.is_absolute());
    }

    #[test]
    fn visualization_relative_path_preserves_relative_mirroring_without_prefix() {
        let rel = visualization_relative_path(Path::new("site-a/bird.wav"), Path::new(""));
        assert_eq!(rel, PathBuf::from("site-a/bird.wav"));
    }

    // --- device_from_str ---

    #[test]
    fn device_from_str_auto() {
        let d = device_from_str("auto").unwrap();
        assert!(matches!(d, Device::Auto));
    }

    #[test]
    fn device_from_str_cpu() {
        let d = device_from_str("cpu").unwrap();
        assert!(matches!(d, Device::Cpu));
    }

    #[test]
    fn device_from_str_gpu() {
        let d = device_from_str("gpu").unwrap();
        assert!(matches!(d, Device::Cuda(0)));
    }

    #[test]
    fn device_from_str_cuda() {
        let d = device_from_str("cuda").unwrap();
        assert!(matches!(d, Device::Cuda(0)));
    }

    #[test]
    fn device_from_str_cuda_0() {
        let d = device_from_str("cuda:0").unwrap();
        assert!(matches!(d, Device::Cuda(0)));
    }

    #[test]
    fn device_from_str_cuda_2() {
        let d = device_from_str("cuda:2").unwrap();
        assert!(matches!(d, Device::Cuda(2)));
    }

    #[test]
    fn device_from_str_case_insensitive() {
        assert!(matches!(device_from_str("AUTO").unwrap(), Device::Auto));
        assert!(matches!(device_from_str("CPU").unwrap(), Device::Cpu));
        assert!(matches!(device_from_str("CUDA").unwrap(), Device::Cuda(0)));
        assert!(matches!(
            device_from_str("Cuda:1").unwrap(),
            Device::Cuda(1)
        ));
    }

    #[test]
    fn device_from_str_invalid() {
        assert!(device_from_str("invalid").is_err());
        assert!(device_from_str("tpu").is_err());
        assert!(device_from_str("").is_err());
    }

    #[test]
    fn device_from_str_cuda_bad_index() {
        assert!(device_from_str("cuda:abc").is_err());
        assert!(device_from_str("cuda:").is_err());
        assert!(device_from_str("cuda:-1").is_err());
    }

    // --- Device::Display (canonical impl in sparrow-engine-types) ---

    #[test]
    fn device_to_string_auto() {
        assert_eq!(Device::Auto.to_string(), "auto");
    }

    #[test]
    fn device_to_string_cpu() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
    }

    #[test]
    fn device_to_string_cuda_0() {
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
    }

    #[test]
    fn device_to_string_cuda_3() {
        assert_eq!(Device::Cuda(3).to_string(), "cuda:3");
    }

    // --- convert_bbox ---

    #[test]
    fn convert_bbox_preserves_values() {
        let src = sparrow_engine::BBox {
            x_min: 0.1,
            y_min: 0.2,
            x_max: 0.8,
            y_max: 0.9,
        };
        let dst = convert_bbox(&src);
        assert_eq!(dst.x_min, 0.1);
        assert_eq!(dst.y_min, 0.2);
        assert_eq!(dst.x_max, 0.8);
        assert_eq!(dst.y_max, 0.9);
    }

    // --- convert_detection ---

    #[test]
    fn convert_detection_maps_all_fields() {
        let src = sparrow_engine::Detection {
            bbox: sparrow_engine::BBox {
                x_min: 0.0,
                y_min: 0.1,
                x_max: 0.5,
                y_max: 0.6,
            },
            label: "animal".to_owned(),
            label_id: 1,
            confidence: 0.95,
        };
        let dst = convert_detection(&src);
        assert_eq!(dst.label, "animal");
        assert_eq!(dst.label_id, 1);
        assert_eq!(dst.confidence, 0.95);
        assert_eq!(dst.bbox.x_min, 0.0);
        assert_eq!(dst.bbox.y_max, 0.6);
    }

    // --- convert_classification ---

    #[test]
    fn convert_classification_maps_all_fields() {
        let src = sparrow_engine::Classification {
            label: "deer".to_owned(),
            label_id: 42,
            confidence: 0.87,
        };
        let dst = convert_classification(&src);
        assert_eq!(dst.label, "deer");
        assert_eq!(dst.label_id, 42);
        assert_eq!(dst.confidence, 0.87);
    }

    // --- convert_audio_segment ---

    #[test]
    fn convert_audio_segment_maps_classes() {
        let src = sparrow_engine::AudioSegment {
            start_time_s: 1.0,
            end_time_s: 2.5,
            confidence: 0.91,
            classes: vec![sparrow_engine::AudioClass {
                class_idx: 7,
                label: Some("sparrow".to_owned()),
                probability: 0.91,
            }],
        };
        let dst = convert_audio_segment(&src);
        assert_eq!(dst.start_time_s, 1.0);
        assert_eq!(dst.end_time_s, 2.5);
        assert_eq!(dst.confidence, 0.91);
        assert_eq!(dst.classes.len(), 1);
        assert_eq!(dst.classes[0].class_idx, 7);
        assert_eq!(dst.classes[0].label.as_deref(), Some("sparrow"));
        assert_eq!(dst.classes[0].probability, 0.91);
    }

    #[test]
    fn convert_audio_segment_preserves_top_k_order_and_none_labels() {
        let src = sparrow_engine::AudioSegment {
            start_time_s: 0.0,
            end_time_s: 5.0,
            confidence: 0.7,
            classes: vec![
                sparrow_engine::AudioClass {
                    class_idx: 10,
                    label: Some("sparrow".to_owned()),
                    probability: 0.7,
                },
                sparrow_engine::AudioClass {
                    class_idx: 11,
                    label: None,
                    probability: 0.2,
                },
                sparrow_engine::AudioClass {
                    class_idx: 12,
                    label: Some("warbler".to_owned()),
                    probability: 0.1,
                },
            ],
        };

        let dst = convert_audio_segment(&src);

        assert_eq!(dst.classes.len(), 3);
        assert_eq!(dst.confidence, dst.classes[0].probability);
        assert_eq!(dst.classes[0].class_idx, 10);
        assert_eq!(dst.classes[0].label.as_deref(), Some("sparrow"));
        assert_eq!(dst.classes[1].class_idx, 11);
        assert_eq!(dst.classes[1].label, None);
        assert_eq!(dst.classes[1].probability, 0.2);
        assert_eq!(dst.classes[2].class_idx, 12);
        assert_eq!(dst.classes[2].label.as_deref(), Some("warbler"));
    }

    #[test]
    fn convert_audio_segment_preserves_empty_classes() {
        let src = sparrow_engine::AudioSegment {
            start_time_s: 0.0,
            end_time_s: 1.0,
            confidence: 0.0,
            classes: Vec::new(),
        };

        let dst = convert_audio_segment(&src);

        assert_eq!(dst.start_time_s, 0.0);
        assert_eq!(dst.end_time_s, 1.0);
        assert_eq!(dst.confidence, 0.0);
        assert!(dst.classes.is_empty());
    }

    #[test]
    fn pyaudio_to_native_preserves_audio_result_fields() {
        let src = AudioResult {
            model_id: "md-audiobirds-v1".to_owned(),
            duration_s: 3.2,
            sample_rate: 48_000,
            window_s: 1.0,
            stride_s: 0.3,
            processing_time_ms: 12.5,
            segments: vec![AudioSegment {
                start_time_s: 0.3,
                end_time_s: 1.3,
                confidence: 0.82,
                classes: vec![AudioClass {
                    class_idx: 4,
                    label: Some("sparrow".to_owned()),
                    probability: 0.82,
                }],
            }],
        };

        let dst = pyaudio_to_native(&src);

        assert_eq!(dst.duration_s, 3.2);
        assert_eq!(dst.sample_rate, 48_000);
        assert_eq!(dst.processing_time_ms, 12.5);
        assert_eq!(audio_result_visualization_timing(&src), (1.0, 0.3));
        assert_eq!(dst.segments.len(), 1);
        assert_eq!(dst.segments[0].start_time_s, 0.3);
        assert_eq!(dst.segments[0].classes[0].label.as_deref(), Some("sparrow"));
    }

    #[test]
    fn visualize_audio_unsupported_message_names_new_function() {
        assert!(VISUALIZE_AUDIO_UNSUPPORTED_MESSAGE.contains("visualize_audio()"));
    }

    #[test]
    fn resolve_audio_window_stride_applies_runtime_overrides() {
        assert_eq!(
            resolve_audio_window_stride(Some((1.0, 0.3)), None, None),
            (1.0, 0.3)
        );
        assert_eq!(
            resolve_audio_window_stride(Some((1.0, 0.3)), Some(0.1), Some(0.7)),
            (0.7, 0.1)
        );
        assert_eq!(
            resolve_audio_window_stride(None, Some(0.2), None),
            (DEFAULT_AUDIO_WINDOW_S, 0.2)
        );
    }

    #[test]
    fn audio_visualization_uses_result_timing_metadata() {
        let result = AudioResult {
            model_id: "md-audiobirds-v1".to_owned(),
            duration_s: 2.0,
            sample_rate: 48_000,
            window_s: 0.7,
            stride_s: 0.1,
            processing_time_ms: 0.0,
            segments: vec![
                AudioSegment {
                    start_time_s: 0.0,
                    end_time_s: 1.0,
                    confidence: 1.0,
                    classes: Vec::new(),
                },
                AudioSegment {
                    start_time_s: 0.5,
                    end_time_s: 1.5,
                    confidence: 0.0,
                    classes: Vec::new(),
                },
                AudioSegment {
                    start_time_s: 1.0,
                    end_time_s: 2.0,
                    confidence: 0.0,
                    classes: Vec::new(),
                },
            ],
        };
        let (window_s, stride_s) = audio_result_visualization_timing(&result);
        assert_eq!((window_s, stride_s), (0.7, 0.1));

        let native = pyaudio_to_native(&result);
        let override_ranges =
            derive_audio_visualization_ranges(&native.segments, result.duration_s, stride_s);
        let manifest_default_ranges = derive_audio_visualization_ranges(
            &native.segments,
            result.duration_s,
            DEFAULT_AUDIO_STRIDE_S,
        );
        assert_ne!(override_ranges, manifest_default_ranges);
        assert_eq!(override_ranges[0].end_time_s, stride_s * 5.0);

        let spec = image::DynamicImage::new_rgb8(80, 16);
        let override_opts = sparrow_engine::viz::AudioLayersOpts {
            smooth: false,
            show_windows: true,
            window_s,
            stride_s,
        };
        let manifest_opts = sparrow_engine::viz::AudioLayersOpts {
            smooth: false,
            show_windows: true,
            window_s: DEFAULT_AUDIO_WINDOW_S,
            stride_s: DEFAULT_AUDIO_STRIDE_S,
        };
        let override_layers = sparrow_engine::viz::render_audio_layers(
            &spec,
            &native.segments,
            Some(&override_ranges),
            result.duration_s,
            &override_opts,
        );
        let manifest_layers = sparrow_engine::viz::render_audio_layers(
            &spec,
            &native.segments,
            Some(&manifest_default_ranges),
            result.duration_s,
            &manifest_opts,
        );

        assert_ne!(
            audio_layer_bytes(&override_layers, "04_full"),
            audio_layer_bytes(&manifest_layers, "04_full"),
            "range overlay must use AudioResult.stride_s, not the manifest default"
        );
        let override_windows = override_layers
            .iter()
            .find(|(name, _)| *name == "02_segments_windows")
            .expect("missing override windows layer")
            .1
            .height();
        let manifest_windows = manifest_layers
            .iter()
            .find(|(name, _)| *name == "02_segments_windows")
            .expect("missing manifest windows layer")
            .1
            .height();
        assert_ne!(
            override_windows, manifest_windows,
            "window lanes must use AudioResult.window_s/stride_s, not manifest defaults"
        );
    }

    #[test]
    fn audio_visualization_ranges_match_cli_slot_pipeline() {
        let spec = image::DynamicImage::new_rgb8(100, 20);
        let duration_s = 2.0;
        let stride_s = 0.5;
        let segments = vec![
            sparrow_engine::AudioSegment {
                start_time_s: 0.0,
                end_time_s: 1.0,
                confidence: 1.0,
                classes: Vec::new(),
            },
            sparrow_engine::AudioSegment {
                start_time_s: 0.5,
                end_time_s: 1.5,
                confidence: 0.0,
                classes: Vec::new(),
            },
            sparrow_engine::AudioSegment {
                start_time_s: 1.0,
                end_time_s: 2.0,
                confidence: 0.0,
                classes: Vec::new(),
            },
        ];

        let cli_slots =
            sparrow_engine::viz::segments_to_overlap_mean_slots(&segments, duration_s, stride_s);
        let cli_high_conf_slots: Vec<sparrow_engine::AudioSegment> = cli_slots
            .into_iter()
            .filter(|s| s.confidence >= VIZ_MERGE_THRESHOLD)
            .collect();
        let cli_ranges =
            sparrow_engine::detect_audio::merge_segments(&cli_high_conf_slots, stride_s + 1e-3);
        let binding_ranges = derive_audio_visualization_ranges(&segments, duration_s, stride_s);
        let raw_merge_ranges =
            sparrow_engine::detect_audio::merge_segments(&segments, stride_s + 1e-3);

        assert_eq!(binding_ranges, cli_ranges);
        assert_eq!(binding_ranges.len(), 1);
        assert_eq!(binding_ranges[0].start_time_s, 0.0);
        assert_eq!(binding_ranges[0].end_time_s, stride_s);
        assert_ne!(raw_merge_ranges, cli_ranges);

        let opts = sparrow_engine::viz::AudioLayersOpts {
            smooth: false,
            show_windows: false,
            window_s: 1.0,
            stride_s,
        };
        let cli_layers = sparrow_engine::viz::render_audio_layers(
            &spec,
            &segments,
            Some(&cli_ranges),
            duration_s,
            &opts,
        );
        let binding_layers = sparrow_engine::viz::render_audio_layers(
            &spec,
            &segments,
            Some(&binding_ranges),
            duration_s,
            &opts,
        );
        let raw_layers = sparrow_engine::viz::render_audio_layers(
            &spec,
            &segments,
            Some(&raw_merge_ranges),
            duration_s,
            &opts,
        );

        assert_eq!(
            audio_layer_bytes(&binding_layers, "04_full"),
            audio_layer_bytes(&cli_layers, "04_full")
        );
        assert_ne!(
            audio_layer_bytes(&raw_layers, "04_full"),
            audio_layer_bytes(&cli_layers, "04_full"),
            "raw segment merging should not silently match the CLI-shaped overlay"
        );
    }

    #[test]
    fn render_audio_layers_returns_expected_layer_counts() {
        let spec = image::DynamicImage::new_rgb8(120, 64);
        let segments = vec![sparrow_engine::AudioSegment {
            start_time_s: 0.0,
            end_time_s: 1.0,
            confidence: 0.91,
            classes: Vec::new(),
        }];
        let ranges = vec![sparrow_engine::AudioRange {
            start_time_s: 0.0,
            end_time_s: 1.0,
            max_confidence: 0.91,
            class: None,
        }];

        let base_opts = sparrow_engine::viz::AudioLayersOpts {
            smooth: false,
            show_windows: false,
            window_s: 1.0,
            stride_s: 0.3,
        };
        let base =
            sparrow_engine::viz::render_audio_layers(&spec, &segments, None, 2.0, &base_opts);
        assert_eq!(base.len(), 3);
        assert_eq!(
            base.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
            vec!["01_spec", "02_segments", "03_heatmap"]
        );

        let with_ranges = sparrow_engine::viz::render_audio_layers(
            &spec,
            &segments,
            Some(&ranges),
            2.0,
            &base_opts,
        );
        assert_eq!(with_ranges.len(), 4);
        assert_eq!(with_ranges.last().map(|(name, _)| *name), Some("04_full"));

        let window_opts = sparrow_engine::viz::AudioLayersOpts {
            show_windows: true,
            ..base_opts
        };
        let with_windows_and_ranges = sparrow_engine::viz::render_audio_layers(
            &spec,
            &segments,
            Some(&ranges),
            2.0,
            &window_opts,
        );
        assert_eq!(with_windows_and_ranges.len(), 5);
        assert_eq!(
            with_windows_and_ranges
                .iter()
                .map(|(name, _)| *name)
                .collect::<Vec<_>>(),
            vec![
                "01_spec",
                "02_segments",
                "02_segments_windows",
                "03_heatmap",
                "04_full",
            ]
        );
    }

    // --- convert_model_type ---

    #[test]
    fn convert_model_type_all_variants() {
        assert_eq!(convert_model_type(ModelType::Detector), "detector");
        assert_eq!(
            convert_model_type(ModelType::OverheadDetector),
            "overhead_detector"
        );
        assert_eq!(convert_model_type(ModelType::Classifier), "classifier");
        assert_eq!(
            convert_model_type(ModelType::AudioDetector),
            "audio_detector"
        );
        assert_eq!(
            convert_model_type(ModelType::AudioClassifier),
            "audio_classifier"
        );
    }

    // --- convert_model_info ---

    #[test]
    fn convert_model_info_maps_fields() {
        let src = sparrow_engine::ModelInfo {
            id: "mdv6".to_owned(),
            path: PathBuf::from("/models/mdv6"),
            model_type: ModelType::Detector,
            default: true,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        };
        let dst = convert_model_info(&src);
        assert_eq!(dst.id, "mdv6");
        assert_eq!(dst.model_type, "detector");
        assert!(dst.default);
    }

    #[test]
    fn convert_model_info_non_default() {
        let src = sparrow_engine::ModelInfo {
            id: "speciesnet".to_owned(),
            path: PathBuf::from("/models/speciesnet"),
            model_type: ModelType::Classifier,
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        };
        let dst = convert_model_info(&src);
        assert_eq!(dst.id, "speciesnet");
        assert_eq!(dst.model_type, "classifier");
        assert!(!dst.default);
    }

    #[test]
    fn visualize_detect_result_uses_detect_model_type() {
        let result = DetectResult {
            model_type: ModelType::OverheadDetector,
            model_id: "owl-t".to_owned(),
            image_size: (32, 32),
            processing_time_ms: 0.0,
            detections: Vec::new(),
        };
        let opts = visualize_render_opts(detect_visualization_model_type(&result), false);
        assert_eq!(opts.model_type, ModelType::OverheadDetector);
        assert!(!opts.show_labels);
    }

    #[test]
    fn visualize_pipeline_result_uses_detector_model_type() {
        let result = PipelineResult {
            pipeline_id: "overhead-pipeline".to_owned(),
            image_size: (32, 32),
            processing_time_ms: 0.0,
            detections: Vec::new(),
            model_type: ModelType::OverheadDetector,
        };
        let opts = visualize_render_opts(pipeline_visualization_model_type(&result), false);
        assert_eq!(opts.model_type, ModelType::OverheadDetector);
        assert!(!opts.show_labels);
    }

    #[test]
    fn visualize_bbox_result_types_keep_bbox_dispatch() {
        let opts = visualize_render_opts(bbox_visualization_model_type(), true);
        assert_eq!(opts.model_type, ModelType::Detector);
        assert!(opts.show_labels);
    }

    // --- device_from_str → Device::Display round-trip ---

    #[test]
    fn device_roundtrip() {
        for input in &["auto", "cpu", "cuda:0", "cuda:2"] {
            let dev = device_from_str(input).unwrap();
            let s = dev.to_string();
            assert_eq!(&s, *input);
        }
    }

    // --- viz_output_extension (PS4 regression — inquisitor F2) ---
    //
    // Pins the PS3 emission set (Jpeg, Png) to its on-disk extension. If PS3
    // (reviewer commit `e473207`) is ever extended to emit a new format, the
    // production match in `visualize()` AND `viz_output_extension` must both
    // be updated. The `_` arm covers `image::ImageFormat`'s `#[non_exhaustive]`
    // marker — production keeps a silent `"png"` fallback (safer than a
    // panic through the PyO3 boundary); the extension-new-format test below
    // serves as the trigger to re-examine the mapping.

    #[test]
    fn viz_output_extension_jpeg_yields_jpg() {
        assert_eq!(viz_output_extension(image::ImageFormat::Jpeg), "jpg");
    }

    #[test]
    fn viz_output_extension_png_yields_png() {
        assert_eq!(viz_output_extension(image::ImageFormat::Png), "png");
    }

    #[test]
    fn viz_output_extension_is_always_lowercase() {
        // Post-PS4 guarantee: the saved filename's extension is lowercase
        // regardless of the input path's case. `fake.JPG` → `fmt=Jpeg` →
        // extension `"jpg"` (not `"JPG"`).
        let ext_jpeg = viz_output_extension(image::ImageFormat::Jpeg);
        let ext_png = viz_output_extension(image::ImageFormat::Png);
        assert_eq!(ext_jpeg, ext_jpeg.to_lowercase());
        assert_eq!(ext_png, ext_png.to_lowercase());
    }

    #[test]
    fn viz_output_extension_unknown_format_falls_back_to_png() {
        // WARNING — documentation anchor, NOT an active drift guard.
        //
        // This test asserts the current fallback behavior (`_ => "png"`) for
        // formats that PS3 does NOT currently emit. If PS3 is later extended
        // to emit any of these variants (e.g. `Some("bmp") => Bmp`) and
        // `viz_output_extension` is NOT updated to match, this test STILL
        // PASSES — the helper hits the same `_ => "png"` arm, the assertion
        // still holds, and the silent MIME mismatch (BMP bytes in `.png`
        // file) reaches users unnoticed.
        //
        // To convert this into an active drift guard, parametrize over every
        // `image::ImageFormat` variant PS3 can actually produce and assert
        // the exact MIME correspondence — drift would then fail at compile
        // time via the `#[non_exhaustive]` match. Tracked as a Phase 3.5
        // improvement (kept light for R1 per inquisitor agreement).
        //
        // Active guards remain tests #1 (Jpeg → "jpg"), #2 (Png → "png"),
        // and #3 (always-lowercase). Those catch the common drift modes.
        assert_eq!(viz_output_extension(image::ImageFormat::Bmp), "png");
        assert_eq!(viz_output_extension(image::ImageFormat::Tiff), "png");
        assert_eq!(viz_output_extension(image::ImageFormat::WebP), "png");
    }
}
