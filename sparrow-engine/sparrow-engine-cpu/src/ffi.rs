//! C FFI boundary layer for sparrow-engine-cpu.
//!
//! All items in this module are gated behind `#[cfg(feature = "ffi")]`.
//! Design principles:
//!   - Opaque handles: Engine and Model are `*mut c_void` — consumers cannot inspect.
//!   - Transparent results: Detection/BBox/Classification are `#[repr(C)]` structs.
//!   - Allocator discipline: sparrow-engine-cpu allocates, sparrow-engine-cpu frees. Every `_new()` has `_free()`.
//!   - Panic safety: every FFI export wraps body in `std::panic::catch_unwind`.
//!   - Thread-local errors: `sparrow_engine_last_error()` returns per-thread error string.
//!   - No reserved fields: structs are immutable once shipped. New fields = `_v2` function.

use crate::engine::{Device, Engine, EngineConfig, ModelHandle};
use crate::types::{
    AudioDetectOpts, AudioDetectResult, AudioInput, ClassifyOpts, ClassifyResult, DetectOpts,
    DetectResult, ImageInput, PipelineResult, PixelFormat,
};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::ptr;

// ===========================================================================
// Thread-local error (errno pattern)
// ===========================================================================

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: String) {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = CString::new(msg).ok();
    });
}

fn clear_last_error() {
    LAST_ERROR.with(|cell| {
        *cell.borrow_mut() = None;
    });
}

// ===========================================================================
// Opaque handle types
// ===========================================================================

/// Opaque engine handle. Consumers must not inspect or dereference.
pub type SparrowEngine = c_void;
/// Opaque model handle. Consumers must not inspect or dereference.
pub type SparrowEngineModel = c_void;

// ===========================================================================
// C-compatible structs
// ===========================================================================

/// Axis-aligned bounding box, normalized [0,1], xyxy format.
#[repr(C)]
pub struct SparrowEngineBBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

/// Single detection result. `label` pointer valid until `sparrow_engine_detections_free()`.
#[repr(C)]
pub struct SparrowEngineDetection {
    pub bbox: SparrowEngineBBox,
    pub label: *const c_char,
    pub label_id: u32,
    pub confidence: f32,
}

/// Detection result set from a single `sparrow_engine_detect()` / `sparrow_engine_detect_raw()` call.
#[repr(C)]
pub struct SparrowEngineDetections {
    pub data: *const SparrowEngineDetection,
    pub len: usize,
    pub image_width: u32,
    pub image_height: u32,
}

/// Single classification prediction.
#[repr(C)]
pub struct SparrowEngineClassification {
    pub label: *const c_char,
    pub label_id: u32,
    pub confidence: f32,
}

/// Classification result from a single `sparrow_engine_classify()` call.
#[repr(C)]
pub struct SparrowEngineClassifyResult {
    pub label: *const c_char,
    pub label_id: u32,
    pub confidence: f32,
    pub top_results: *const SparrowEngineClassification,
    pub top_results_len: usize,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
}

/// A pipeline detection: detection + optional classification.
#[repr(C)]
pub struct SparrowEnginePipelineDetection {
    pub detection: SparrowEngineDetection,
    pub has_classification: bool,
    pub classification: SparrowEngineClassification,
}

/// Pipeline result from `sparrow_engine_run_pipeline()`.
#[repr(C)]
pub struct SparrowEnginePipelineResult {
    pub pipeline_id: *const c_char,
    pub data: *const SparrowEnginePipelineDetection,
    pub len: usize,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
}

/// Detection inference options. Zero = use default.
#[repr(C)]
pub struct SparrowEngineDetectOpts {
    pub confidence_threshold: f32,
    pub max_detections: u32,
}

/// Classification inference options. Zero = use default.
#[repr(C)]
pub struct SparrowEngineClassifyOpts {
    pub top_k: u32,
}

/// A single detected audio segment.
#[repr(C)]
pub struct SparrowEngineAudioSegment {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub confidence: f32,
}

/// Audio detection result from `sparrow_engine_detect_audio`.
#[repr(C)]
pub struct SparrowEngineAudioResult {
    pub data: *const SparrowEngineAudioSegment,
    pub len: usize,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
}

/// V2 (Perch 2 + future multi-class classifiers): per-class entry for top-K output.
/// `label` is a borrowed pointer into the result's CString arena; valid until the
/// SparrowEngineAudioResult_v2 is freed via sparrow_engine_audio_result_v2_free.
/// `label` may be null when the model has no label for this index.
#[repr(C)]
pub struct SparrowEngineAudioClass {
    pub class_idx: u32,
    pub label: *const c_char,
    pub probability: f32,
}

/// V2 audio segment: same V1 fields plus a top-K classes array.
/// `classes` is a borrowed pointer into the result; valid for the lifetime of the result.
#[repr(C)]
pub struct SparrowEngineAudioSegment_v2 {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub confidence: f32,
    pub classes: *const SparrowEngineAudioClass,
    pub classes_len: usize,
}

/// V2 audio detection result. Free with sparrow_engine_audio_result_v2_free.
#[repr(C)]
pub struct SparrowEngineAudioResult_v2 {
    pub data: *const SparrowEngineAudioSegment_v2,
    pub len: usize,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
}

/// Audio detection options. Zero = use manifest default.
#[repr(C)]
pub struct SparrowEngineAudioDetectOpts {
    pub confidence_threshold: f32,
    pub segment_duration_s: f32,
    pub segment_stride_s: f32,
}

/// Pixel format code for raw image buffers.
///
/// Valid values are 0 = RGB, 1 = RGBA, 2 = BGRA, 3 = BGR.
pub type SparrowEnginePixelFormat = u32;
pub const SPARROW_ENGINE_PIXEL_FORMAT_RGB: SparrowEnginePixelFormat = 0;
pub const SPARROW_ENGINE_PIXEL_FORMAT_RGBA: SparrowEnginePixelFormat = 1;
pub const SPARROW_ENGINE_PIXEL_FORMAT_BGRA: SparrowEnginePixelFormat = 2;
pub const SPARROW_ENGINE_PIXEL_FORMAT_BGR: SparrowEnginePixelFormat = 3;

// ===========================================================================
// Conversion helpers
// ===========================================================================

/// Convert C detect options to Rust. NULL pointer → all defaults.
unsafe fn detect_opts_from_c(opts: *const SparrowEngineDetectOpts) -> DetectOpts {
    if opts.is_null() {
        return DetectOpts::default();
    }
    let o = &*opts;
    DetectOpts {
        confidence_threshold: if o.confidence_threshold == 0.0 {
            None
        } else {
            Some(o.confidence_threshold)
        },
        max_detections: if o.max_detections == 0 {
            None
        } else {
            Some(o.max_detections)
        },
    }
}

/// Convert C classify options to Rust. NULL pointer → all defaults.
unsafe fn classify_opts_from_c(opts: *const SparrowEngineClassifyOpts) -> ClassifyOpts {
    if opts.is_null() {
        return ClassifyOpts::default();
    }
    let o = &*opts;
    ClassifyOpts {
        top_k: if o.top_k == 0 { None } else { Some(o.top_k) },
    }
}

/// Convert a raw C pixel-format code to the Rust `PixelFormat`.
fn pixel_format_from_c(fmt: SparrowEnginePixelFormat) -> Result<PixelFormat, String> {
    match fmt {
        SPARROW_ENGINE_PIXEL_FORMAT_RGB => Ok(PixelFormat::Rgb),
        SPARROW_ENGINE_PIXEL_FORMAT_RGBA => Ok(PixelFormat::Rgba),
        SPARROW_ENGINE_PIXEL_FORMAT_BGRA => Ok(PixelFormat::Bgra),
        SPARROW_ENGINE_PIXEL_FORMAT_BGR => Ok(PixelFormat::Bgr),
        other => Err(format!(
            "unsupported pixel format {other}; expected 0=RGB, 1=RGBA, 2=BGRA, or 3=BGR"
        )),
    }
}

/// Convert C audio detect options to Rust. NULL pointer → all defaults.
unsafe fn audio_detect_opts_from_c(opts: *const SparrowEngineAudioDetectOpts) -> AudioDetectOpts {
    if opts.is_null() {
        return AudioDetectOpts::default();
    }
    let o = &*opts;
    AudioDetectOpts {
        confidence_threshold: if o.confidence_threshold == 0.0 {
            None
        } else {
            Some(o.confidence_threshold)
        },
        segment_duration_s: if o.segment_duration_s == 0.0 {
            None
        } else {
            Some(o.segment_duration_s)
        },
        stride_s: if o.segment_stride_s == 0.0 {
            None
        } else {
            Some(o.segment_stride_s)
        },
    }
}

/// Leak a `String` into a `*const c_char`. Caller must free with `sparrow_engine_free_string`.
fn string_to_c(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(_) => ptr::null_mut(),
    }
}

/// Read a `*const c_char` into a `&str`. Returns `Err` on null or invalid UTF-8.
unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("null string pointer".to_string());
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|e| format!("invalid UTF-8: {e}"))
}

// ---------------------------------------------------------------------------
// DetectResult → SparrowEngineDetections
// ---------------------------------------------------------------------------

/// Backing storage for a SparrowEngineDetections that keeps CStrings alive.
struct DetectionsOwned {
    detections: Vec<SparrowEngineDetection>,
    /// CStrings whose pointers are lent to `detections[i].label`.
    _labels: Vec<CString>,
}

fn detect_result_to_c(result: DetectResult) -> *mut SparrowEngineDetections {
    let mut labels: Vec<CString> = Vec::with_capacity(result.detections.len());
    let mut c_dets: Vec<SparrowEngineDetection> = Vec::with_capacity(result.detections.len());

    for det in &result.detections {
        let label_c = CString::new(det.label.replace('\0', "")).unwrap_or_default();
        let label_ptr = label_c.as_ptr();
        labels.push(label_c);
        c_dets.push(SparrowEngineDetection {
            bbox: SparrowEngineBBox {
                x_min: det.bbox.x_min,
                y_min: det.bbox.y_min,
                x_max: det.bbox.x_max,
                y_max: det.bbox.y_max,
            },
            label: label_ptr,
            label_id: det.label_id,
            confidence: det.confidence,
        });
    }

    let owned = DetectionsOwned {
        detections: c_dets,
        _labels: labels,
    };

    // Combine header + backing storage in one allocation so _free() recovers both.
    // `header` fields are set after owned is in place to get stable pointers.
    let mut combined = Box::new(DetectionsWithOwner {
        header: SparrowEngineDetections {
            data: ptr::null(),
            len: owned.detections.len(),
            image_width: result.image_width,
            image_height: result.image_height,
        },
        _owner: owned,
    });
    // Now that _owner is at its final heap location, point header.data at it.
    combined.header.data = if combined._owner.detections.is_empty() {
        ptr::null()
    } else {
        combined._owner.detections.as_ptr()
    };

    let ptr = Box::into_raw(combined);
    // Return pointer to the header, which is the first field.
    ptr as *mut SparrowEngineDetections
}

#[repr(C)]
struct DetectionsWithOwner {
    header: SparrowEngineDetections,
    _owner: DetectionsOwned,
}

// ---------------------------------------------------------------------------
// ClassifyResult → SparrowEngineClassifyResult
// ---------------------------------------------------------------------------

struct ClassifyResultOwned {
    top_results: Vec<SparrowEngineClassification>,
    _labels: Vec<CString>,
    _top1_label: CString,
}

fn classify_result_to_c(result: ClassifyResult) -> *mut SparrowEngineClassifyResult {
    let mut labels: Vec<CString> = Vec::with_capacity(result.classifications.len());
    let mut c_cls: Vec<SparrowEngineClassification> =
        Vec::with_capacity(result.classifications.len());

    for cls in &result.classifications {
        let label_c = CString::new(cls.label.replace('\0', "")).unwrap_or_default();
        let label_ptr = label_c.as_ptr();
        labels.push(label_c);
        c_cls.push(SparrowEngineClassification {
            label: label_ptr,
            label_id: cls.label_id,
            confidence: cls.confidence,
        });
    }

    // Top-1 is first element (highest confidence).
    let (top1_label, top1_id, top1_conf) = if let Some(first) = result.classifications.first() {
        (
            CString::new(first.label.replace('\0', "")).unwrap_or_default(),
            first.label_id,
            first.confidence,
        )
    } else {
        (CString::default(), 0, 0.0)
    };

    let top_results_len = c_cls.len();
    let owned = ClassifyResultOwned {
        top_results: c_cls,
        _labels: labels,
        _top1_label: top1_label,
    };

    let mut combined = Box::new(ClassifyResultWithOwner {
        header: SparrowEngineClassifyResult {
            label: ptr::null(),
            label_id: top1_id,
            confidence: top1_conf,
            top_results: ptr::null(),
            top_results_len,
            image_width: result.image_width,
            image_height: result.image_height,
            processing_time_ms: result.processing_time_ms,
        },
        _owner: owned,
    });
    // Point at stable heap locations.
    combined.header.label = combined._owner._top1_label.as_ptr();
    combined.header.top_results = if combined._owner.top_results.is_empty() {
        ptr::null()
    } else {
        combined._owner.top_results.as_ptr()
    };

    let ptr = Box::into_raw(combined);
    ptr as *mut SparrowEngineClassifyResult
}

#[repr(C)]
struct ClassifyResultWithOwner {
    header: SparrowEngineClassifyResult,
    _owner: ClassifyResultOwned,
}

// ---------------------------------------------------------------------------
// PipelineResult → SparrowEnginePipelineResult
// ---------------------------------------------------------------------------

struct PipelineResultOwned {
    data: Vec<SparrowEnginePipelineDetection>,
    _labels: Vec<CString>,
    _pipeline_id: CString,
}

fn pipeline_result_to_c(result: PipelineResult) -> *mut SparrowEnginePipelineResult {
    let mut labels: Vec<CString> = Vec::new();
    let mut c_pipe: Vec<SparrowEnginePipelineDetection> =
        Vec::with_capacity(result.detections.len());

    for pd in &result.detections {
        // Detection label
        let det_label_c = CString::new(pd.detection.label.replace('\0', "")).unwrap_or_default();
        let det_label_ptr = det_label_c.as_ptr();
        labels.push(det_label_c);

        let (has_cls, cls) = if let Some(ref cls) = pd.classification {
            let cls_label_c = CString::new(cls.label.replace('\0', "")).unwrap_or_default();
            let cls_label_ptr = cls_label_c.as_ptr();
            labels.push(cls_label_c);
            (
                true,
                SparrowEngineClassification {
                    label: cls_label_ptr,
                    label_id: cls.label_id,
                    confidence: cls.confidence,
                },
            )
        } else {
            (
                false,
                SparrowEngineClassification {
                    label: ptr::null(),
                    label_id: 0,
                    confidence: 0.0,
                },
            )
        };

        c_pipe.push(SparrowEnginePipelineDetection {
            detection: SparrowEngineDetection {
                bbox: SparrowEngineBBox {
                    x_min: pd.detection.bbox.x_min,
                    y_min: pd.detection.bbox.y_min,
                    x_max: pd.detection.bbox.x_max,
                    y_max: pd.detection.bbox.y_max,
                },
                label: det_label_ptr,
                label_id: pd.detection.label_id,
                confidence: pd.detection.confidence,
            },
            has_classification: has_cls,
            classification: cls,
        });
    }

    let pipeline_id_c = CString::new(result.pipeline_id.replace('\0', "")).unwrap_or_default();
    let data_len = c_pipe.len();

    let owned = PipelineResultOwned {
        data: c_pipe,
        _labels: labels,
        _pipeline_id: pipeline_id_c,
    };

    let mut combined = Box::new(PipelineResultWithOwner {
        header: SparrowEnginePipelineResult {
            pipeline_id: ptr::null(),
            data: ptr::null(),
            len: data_len,
            image_width: result.image_width,
            image_height: result.image_height,
            processing_time_ms: result.processing_time_ms,
        },
        _owner: owned,
    });
    // Point at stable heap locations.
    combined.header.pipeline_id = combined._owner._pipeline_id.as_ptr();
    combined.header.data = if combined._owner.data.is_empty() {
        ptr::null()
    } else {
        combined._owner.data.as_ptr()
    };

    let ptr = Box::into_raw(combined);
    ptr as *mut SparrowEnginePipelineResult
}

#[repr(C)]
struct PipelineResultWithOwner {
    header: SparrowEnginePipelineResult,
    _owner: PipelineResultOwned,
}

// ---------------------------------------------------------------------------
// AudioDetectResult → SparrowEngineAudioResult
// ---------------------------------------------------------------------------

struct AudioResultOwned {
    segments: Vec<SparrowEngineAudioSegment>,
}

fn audio_result_to_c(result: AudioDetectResult) -> *mut SparrowEngineAudioResult {
    let c_segments: Vec<SparrowEngineAudioSegment> = result
        .segments
        .iter()
        .map(|s| SparrowEngineAudioSegment {
            start_time_s: s.start_time_s,
            end_time_s: s.end_time_s,
            confidence: s.confidence,
        })
        .collect();

    let owned = AudioResultOwned {
        segments: c_segments,
    };

    let mut combined = Box::new(AudioResultWithOwner {
        header: SparrowEngineAudioResult {
            data: ptr::null(),
            len: owned.segments.len(),
            duration_s: result.duration_s,
            sample_rate: result.sample_rate,
            processing_time_ms: result.processing_time_ms,
        },
        _owner: owned,
    });
    // Point at stable heap location when non-empty; expose null for len=0.
    combined.header.data = if combined._owner.segments.is_empty() {
        ptr::null()
    } else {
        combined._owner.segments.as_ptr()
    };

    let ptr = Box::into_raw(combined);
    ptr as *mut SparrowEngineAudioResult
}

#[repr(C)]
struct AudioResultWithOwner {
    header: SparrowEngineAudioResult,
    _owner: AudioResultOwned,
}

struct AudioResultV2Owned {
    _labels: Vec<CString>,
    _classes: Vec<SparrowEngineAudioClass>,
    segments: Vec<SparrowEngineAudioSegment_v2>,
}

fn audio_result_v2_to_c(result: AudioDetectResult) -> *mut SparrowEngineAudioResult_v2 {
    let total_class_count = result.segments.iter().map(|s| s.classes.len()).sum();

    let mut labels = Vec::new();
    let mut label_indices = Vec::with_capacity(result.segments.len());
    for segment in &result.segments {
        let mut segment_label_indices = Vec::with_capacity(segment.classes.len());
        for class in &segment.classes {
            if let Some(label) = &class.label {
                labels.push(CString::new(label.replace('\0', "")).unwrap_or_default());
                segment_label_indices.push(Some(labels.len() - 1));
            } else {
                segment_label_indices.push(None);
            }
        }
        label_indices.push(segment_label_indices);
    }

    let mut classes = Vec::with_capacity(total_class_count);
    for (segment_idx, segment) in result.segments.iter().enumerate() {
        for (class_idx, class) in segment.classes.iter().enumerate() {
            let label = label_indices[segment_idx][class_idx]
                .map(|idx| labels[idx].as_ptr())
                .unwrap_or(ptr::null());
            classes.push(SparrowEngineAudioClass {
                class_idx: class.class_idx,
                label,
                probability: class.probability,
            });
        }
    }

    let mut segments = Vec::with_capacity(result.segments.len());
    let mut class_offset = 0;
    for segment in &result.segments {
        let classes_len = segment.classes.len();
        let classes_ptr = if classes_len == 0 {
            ptr::null()
        } else {
            // The class arena is fully built before segment pointers are taken.
            unsafe { classes.as_ptr().add(class_offset) }
        };
        segments.push(SparrowEngineAudioSegment_v2 {
            start_time_s: segment.start_time_s,
            end_time_s: segment.end_time_s,
            confidence: segment.confidence,
            classes: classes_ptr,
            classes_len,
        });
        class_offset += classes_len;
    }

    let owned = AudioResultV2Owned {
        _labels: labels,
        _classes: classes,
        segments,
    };

    let mut combined = Box::new(AudioResultV2WithOwner {
        header: SparrowEngineAudioResult_v2 {
            data: ptr::null(),
            len: owned.segments.len(),
            duration_s: result.duration_s,
            sample_rate: result.sample_rate,
            processing_time_ms: result.processing_time_ms,
        },
        _owner: owned,
    });
    combined.header.data = if combined._owner.segments.is_empty() {
        ptr::null()
    } else {
        combined._owner.segments.as_ptr()
    };

    let ptr = Box::into_raw(combined);
    ptr as *mut SparrowEngineAudioResult_v2
}

#[repr(C)]
struct AudioResultV2WithOwner {
    header: SparrowEngineAudioResult_v2,
    _owner: AudioResultV2Owned,
}

// ===========================================================================
// Config JSON parsing
// ===========================================================================

/// Intermediate struct for JSON config deserialization.
#[derive(serde::Deserialize)]
struct ConfigJson {
    #[serde(default = "default_device_str")]
    device: String,
    #[serde(default = "default_inter")]
    inter_threads: u32,
    #[serde(default = "default_intra")]
    intra_threads: u32,
    model_dir: String,
}

fn default_device_str() -> String {
    "auto".to_string()
}
fn default_inter() -> u32 {
    1
}
fn default_intra() -> u32 {
    0 // 0 = auto-detect (resolved in parse_config_json based on device)
}

fn parse_device(s: &str) -> Result<Device, String> {
    s.parse()
}

fn parse_config_json(json_str: &str) -> Result<EngineConfig, String> {
    let cfg: ConfigJson =
        serde_json::from_str(json_str).map_err(|e| format!("invalid config JSON: {e}"))?;
    let device = parse_device(&cfg.device)?;
    // Apply device-aware default: 0 means "auto" (4 for CPU, 1 for GPU).
    let intra = if cfg.intra_threads > 0 {
        cfg.intra_threads
    } else {
        match &device {
            Device::Cpu | Device::Auto => 4,
            Device::Cuda(_) => 1,
        }
    };

    Ok(EngineConfig {
        device,
        inter_threads: cfg.inter_threads,
        intra_threads: intra,
        model_dir: PathBuf::from(cfg.model_dir),
    })
}

// ===========================================================================
// FFI exports
// ===========================================================================

// ---------------------------------------------------------------------------
// Lifecycle
// ---------------------------------------------------------------------------

/// Create a new engine from a JSON config string. Returns null on error.
///
/// # Safety
/// `config_json` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_engine_new(
    config_json: *const c_char,
) -> *mut SparrowEngine {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngine, String> {
            let json_str = cstr_to_str(config_json)?;
            let config = parse_config_json(json_str)?;
            let engine = Engine::new(config).map_err(|e| e.to_string())?;
            Ok(Box::into_raw(Box::new(engine)) as *mut SparrowEngine)
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_engine_new".to_string());
            ptr::null_mut()
        }
    }
}

/// Free an engine. All models loaded through this engine become invalid.
///
/// # Safety
/// `engine` must be a pointer returned by `sparrow_engine_engine_new`, or null (no-op).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_engine_free(engine: *mut SparrowEngine) {
    clear_last_error();
    if engine.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(engine as *mut Engine));
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_engine_free".to_string());
    }
}

// ---------------------------------------------------------------------------
// Model Management
// ---------------------------------------------------------------------------

/// Load a model from a TOML manifest file path. Returns null on error.
///
/// # Safety
/// - `engine` must be a valid engine pointer.
/// - `manifest_path` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_load_model(
    engine: *mut SparrowEngine,
    manifest_path: *const c_char,
) -> *mut SparrowEngineModel {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngineModel, String> {
            if engine.is_null() {
                return Err("engine pointer is null".to_string());
            }
            let engine_ref = &*(engine as *const Engine);
            let path_str = cstr_to_str(manifest_path)?;
            let handle = engine_ref.load_model(path_str).map_err(|e| e.to_string())?;
            Ok(Box::into_raw(Box::new(handle)) as *mut SparrowEngineModel)
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_load_model".to_string());
            ptr::null_mut()
        }
    }
}

/// Load a model by its ID from the model directory. Returns null on error.
///
/// Idempotent / lazy: if the model is already loaded, returns a fresh handle
/// to the existing ORT session (no re-creation). Mirrors the lazy-load
/// contract exposed by the HTTP `/v1/models/load` and `/v1/detect`+`/v1/classify`+
/// `/v1/audio` endpoints and the `sparrow-engine-python` package — calling twice with
/// the same id does not invalidate previously-issued handles.
///
/// # Safety
/// - `engine` must be a valid engine pointer.
/// - `model_id` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_load_model_by_id(
    engine: *mut SparrowEngine,
    model_id: *const c_char,
) -> *mut SparrowEngineModel {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngineModel, String> {
            if engine.is_null() {
                return Err("engine pointer is null".to_string());
            }
            let engine_ref = &*(engine as *const Engine);
            let id = cstr_to_str(model_id)?;
            let handle = engine_ref
                .get_or_load_model(id)
                .map_err(|e| e.to_string())?;
            Ok(Box::into_raw(Box::new(handle)) as *mut SparrowEngineModel)
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_load_model_by_id".to_string());
            ptr::null_mut()
        }
    }
}

/// Unload a model and free its resources.
///
/// # Safety
/// `model` must be a pointer returned by `sparrow_engine_load_model` / `sparrow_engine_load_model_by_id`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_unload_model(model: *mut SparrowEngineModel) {
    clear_last_error();
    if model.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let handle = Box::from_raw(model as *mut ModelHandle);
        // Deactivate the model so other holders of cloned handles see it as unloaded.
        // The map entry leaks until engine drop — acceptable since unload is rare.
        handle
            .active
            .store(false, std::sync::atomic::Ordering::Release);
        drop(handle);
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_unload_model".to_string());
    }
}

// ---------------------------------------------------------------------------
// Pipeline Management
// ---------------------------------------------------------------------------

/// Load a pipeline from a TOML manifest. Returns 0 on success, -1 on error.
///
/// # Safety
/// - `engine` must be a valid engine pointer.
/// - `manifest_path` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_load_pipeline(
    engine: *mut SparrowEngine,
    manifest_path: *const c_char,
) -> i32 {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if engine.is_null() {
            return Err("engine pointer is null".to_string());
        }
        let engine_ref = &*(engine as *const Engine);
        let path_str = cstr_to_str(manifest_path)?;
        engine_ref
            .load_pipeline(path_str)
            .map_err(|e| e.to_string())?;
        Ok(0)
    }));
    match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_load_pipeline".to_string());
            -1
        }
    }
}

/// Load a pipeline by its ID from the model directory. Returns 0 on success, -1 on error.
///
/// # Safety
/// - `engine` must be a valid engine pointer.
/// - `pipeline_id` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_load_pipeline_by_id(
    engine: *mut SparrowEngine,
    pipeline_id: *const c_char,
) -> i32 {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if engine.is_null() {
            return Err("engine pointer is null".to_string());
        }
        let engine_ref = &*(engine as *const Engine);
        let id = cstr_to_str(pipeline_id)?;
        engine_ref
            .load_pipeline_by_id(id)
            .map_err(|e| e.to_string())?;
        Ok(0)
    }));
    match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(_panic) => {
            set_last_error(
                "internal error: panic in sparrow_engine_load_pipeline_by_id".to_string(),
            );
            -1
        }
    }
}

/// Unload a pipeline by ID. Returns 0 on success, -1 on error.
///
/// # Safety
/// - `engine` must be a valid engine pointer.
/// - `pipeline_id` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_unload_pipeline(
    engine: *mut SparrowEngine,
    pipeline_id: *const c_char,
) -> i32 {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<i32, String> {
        if engine.is_null() {
            return Err("engine pointer is null".to_string());
        }
        let engine_ref = &*(engine as *const Engine);
        let id = cstr_to_str(pipeline_id)?;
        engine_ref.unload_pipeline(id).map_err(|e| e.to_string())?;
        Ok(0)
    }));
    match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_unload_pipeline".to_string());
            -1
        }
    }
}

// ---------------------------------------------------------------------------
// Inference (Hot Path)
// ---------------------------------------------------------------------------

/// Run detection on an encoded image buffer (JPEG/PNG). Returns null on error.
///
/// # Safety
/// - `model` must be a valid model pointer.
/// - `image` must point to `len` bytes of encoded image data.
/// - `opts` may be null (use defaults).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detect(
    model: *const SparrowEngineModel,
    image: *const u8,
    len: usize,
    opts: *const SparrowEngineDetectOpts,
) -> *mut SparrowEngineDetections {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngineDetections, String> {
            if model.is_null() {
                return Err("model pointer is null".to_string());
            }
            if image.is_null() || len == 0 {
                return Err("image data is null or empty".to_string());
            }
            let handle = &*(model as *const ModelHandle);
            let image_data = std::slice::from_raw_parts(image, len);
            let input = ImageInput::Encoded(image_data.to_vec());
            let d_opts = detect_opts_from_c(opts);
            let result =
                crate::detect::detect(handle, &input, &d_opts).map_err(|e| e.to_string())?;
            Ok(detect_result_to_c(result))
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_detect".to_string());
            ptr::null_mut()
        }
    }
}

/// Run detection on a raw pixel buffer. Returns null on error.
///
/// # Safety
/// - `model` must be a valid model pointer.
/// - `pixels` must point to `h * stride` bytes of pixel data.
/// - `opts` may be null (use defaults).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detect_raw(
    model: *const SparrowEngineModel,
    pixels: *const u8,
    w: u32,
    h: u32,
    stride: u32,
    format: SparrowEnginePixelFormat,
    opts: *const SparrowEngineDetectOpts,
) -> *mut SparrowEngineDetections {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngineDetections, String> {
            if model.is_null() {
                return Err("model pointer is null".to_string());
            }
            if pixels.is_null() {
                return Err("pixels pointer is null".to_string());
            }
            let handle = &*(model as *const ModelHandle);
            let buf_len = (h as usize)
                .checked_mul(stride as usize)
                .ok_or_else(|| "h * stride overflows usize".to_string())?;
            let pixel_data = std::slice::from_raw_parts(pixels, buf_len);
            let input = ImageInput::Raw {
                data: pixel_data.to_vec(),
                width: w,
                height: h,
                stride,
                format: pixel_format_from_c(format)?,
            };
            let d_opts = detect_opts_from_c(opts);
            let result =
                crate::detect::detect(handle, &input, &d_opts).map_err(|e| e.to_string())?;
            Ok(detect_result_to_c(result))
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_detect_raw".to_string());
            ptr::null_mut()
        }
    }
}

/// A single image buffer for batch detection.
#[repr(C)]
pub struct SparrowEngineImageBuffer {
    pub data: *const u8,
    pub len: usize,
}

/// Per-image callback contract for `sparrow_engine_detect_batch`.
///
/// Called once per image after its detections are ready.
///
/// # Contract (DO NOT BREAK — extension rules below)
///
/// Signature: `extern "C" fn(image_index, detections, user_data)`.
///
/// Arguments:
/// - `image_index`: 0-based index into the `images` slice originally
///   passed to `sparrow_engine_detect_batch`. Monotonically non-decreasing across
///   calls within a single `sparrow_engine_detect_batch` invocation (images are
///   processed in input order within each batch chunk).
/// - `detections`: pointer to a `SparrowEngineDetections` struct owned by
///   sparrow-engine-cpu. Valid **only** for the duration of this callback. The
///   callee MUST NOT retain the pointer past return; retained copies
///   become dangling. Copy fields out before returning if persistence
///   is needed.
/// - `user_data`: the `user_data` argument passed to `sparrow_engine_detect_batch`.
///   Opaque to sparrow-engine-cpu.
///
/// # Extension rules (additive-only)
///
/// Any Phase 3.5+ extension to this callback MUST be additive:
/// - New fields appended to `SparrowEngineDetections` (never reordered, renamed,
///   or removed).
/// - New callback variants (e.g. `SparrowEngineDetectBatchCallbackV2`) added
///   alongside the existing one; the original type stays wire-compatible.
/// - New out-of-band signals routed via a sibling callback pointer in a
///   new opts struct, not by mutating this signature.
///
/// This rule coordinates S5 (`sparrow-engine-cli` progress bar, which merged
/// first and authored this contract) with S6 (`sparrow-engine-python` progress
/// callback + `tracing` bridge, which extends additively). See
/// `docs/design/phase3.5/final_design.md` §4 S5 / §4 S6.
pub type SparrowEngineDetectBatchCallback = unsafe extern "C" fn(
    image_index: usize,
    detections: *const SparrowEngineDetections,
    user_data: *mut c_void,
);

/// Run batch detection on multiple encoded images. Returns 0 on success, -1 on error.
/// The callback is invoked per-image with detection results.
/// Images are processed in batches (default 4) for higher GPU throughput.
///
/// See `SparrowEngineDetectBatchCallback` for the per-image callback contract and
/// additive-extension rules (both enforced by the Phase 3.5 S5/S6 handoff).
///
/// # Safety
/// - `model` must be a valid model pointer.
/// - `images` must point to `image_count` `SparrowEngineImageBuffer` structs.
/// - `opts` may be null (use defaults).
/// - `callback` must be valid for the duration of this call.
/// - `batch_size` of 0 uses the default (4).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detect_batch(
    model: *const SparrowEngineModel,
    images: *const SparrowEngineImageBuffer,
    image_count: usize,
    opts: *const SparrowEngineDetectOpts,
    batch_size: usize,
    callback: SparrowEngineDetectBatchCallback,
    user_data: *mut c_void,
) -> i32 {
    clear_last_error();
    let result =
        std::panic::catch_unwind(AssertUnwindSafe(|| -> std::result::Result<(), String> {
            if model.is_null() || images.is_null() {
                return Err("model or images pointer is null".to_string());
            }
            let handle = &*(model as *const ModelHandle);
            let d_opts = detect_opts_from_c(opts);

            // Build ImageInput slice from C buffers.
            let bufs = std::slice::from_raw_parts(images, image_count);
            let mut inputs: Vec<ImageInput> = Vec::with_capacity(bufs.len());
            for (i, b) in bufs.iter().enumerate() {
                if b.data.is_null() || b.len == 0 {
                    return Err(format!(
                        "image buffer {i}: null data pointer or zero length"
                    ));
                }
                let bytes = std::slice::from_raw_parts(b.data, b.len).to_vec();
                inputs.push(ImageInput::Encoded(bytes));
            }

            crate::detect::detect_batch(
                handle,
                &inputs,
                &d_opts,
                batch_size,
                Some(&mut |idx: usize, result: &crate::types::DetectResult| {
                    let c_result = detect_result_to_c(result.clone());
                    callback(idx, c_result, user_data);
                    // Reclaim the temporary C struct immediately — callback must not hold the pointer.
                    drop(Box::from_raw(c_result as *mut DetectionsWithOwner));
                }),
            )
            .map_err(|e| e.to_string())?;
            Ok(())
        }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_detect_batch".to_string());
            -1
        }
    }
}

/// Run classification on an encoded image buffer (JPEG/PNG). Returns null on error.
///
/// # Safety
/// - `model` must be a valid model pointer.
/// - `image` must point to `len` bytes of encoded image data.
/// - `opts` may be null (use defaults).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_classify(
    model: *const SparrowEngineModel,
    image: *const u8,
    len: usize,
    opts: *const SparrowEngineClassifyOpts,
) -> *mut SparrowEngineClassifyResult {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngineClassifyResult, String> {
            if model.is_null() {
                return Err("model pointer is null".to_string());
            }
            if image.is_null() || len == 0 {
                return Err("image data is null or empty".to_string());
            }
            let handle = &*(model as *const ModelHandle);
            let image_data = std::slice::from_raw_parts(image, len);
            let input = ImageInput::Encoded(image_data.to_vec());
            let c_opts = classify_opts_from_c(opts);
            let result =
                crate::classify::classify(handle, &input, &c_opts).map_err(|e| e.to_string())?;
            Ok(classify_result_to_c(result))
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_classify".to_string());
            ptr::null_mut()
        }
    }
}

/// Run a pipeline (detect → classify) on an encoded image. Returns null on error.
///
/// # Safety
/// - `engine` must be a valid engine pointer.
/// - `pipeline_id` must be a valid, non-null, null-terminated UTF-8 string.
/// - `image` must point to `len` bytes of encoded image data.
/// - `detect_opts` and `classify_opts` may be null (use defaults).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_run_pipeline(
    engine: *const SparrowEngine,
    pipeline_id: *const c_char,
    image: *const u8,
    len: usize,
    detect_opts: *const SparrowEngineDetectOpts,
    classify_opts: *const SparrowEngineClassifyOpts,
) -> *mut SparrowEnginePipelineResult {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEnginePipelineResult, String> {
            if engine.is_null() {
                return Err("engine pointer is null".to_string());
            }
            if image.is_null() || len == 0 {
                return Err("image data is null or empty".to_string());
            }
            let engine_ref = &*(engine as *const Engine);
            let pid = cstr_to_str(pipeline_id)?;
            let image_data = std::slice::from_raw_parts(image, len);
            let input = ImageInput::Encoded(image_data.to_vec());
            let d_opts = detect_opts_from_c(detect_opts);
            let c_opts = classify_opts_from_c(classify_opts);
            let result = crate::pipeline::run_pipeline(engine_ref, pid, &input, &d_opts, &c_opts)
                .map_err(|e| e.to_string())?;
            Ok(pipeline_result_to_c(result))
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_run_pipeline".to_string());
            ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// Audio Inference
// ---------------------------------------------------------------------------

/// Run audio detection on a WAV file. Returns null on error.
///
/// # Safety
/// - `model` must be a valid model pointer (audio model with mel spectrogram preprocessing).
/// - `audio_path` must be a valid, non-null, null-terminated UTF-8 path to a WAV file.
/// - `opts` may be null (use defaults).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detect_audio(
    model: *const SparrowEngineModel,
    audio_path: *const c_char,
    opts: *const SparrowEngineAudioDetectOpts,
) -> *mut SparrowEngineAudioResult {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngineAudioResult, String> {
            if model.is_null() {
                return Err("model pointer is null".to_string());
            }
            let handle = &*(model as *const ModelHandle);
            let path_str = cstr_to_str(audio_path)?;
            let input = AudioInput::FilePath(PathBuf::from(path_str));
            let a_opts = audio_detect_opts_from_c(opts);
            let result = crate::detect_audio::detect_audio(handle, &input, &a_opts)
                .map_err(|e| e.to_string())?;
            Ok(audio_result_to_c(result))
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_detect_audio".to_string());
            ptr::null_mut()
        }
    }
}

/// Run audio detection on a WAV file with V2 top-K classes. Returns null on error.
///
/// # Safety
/// - `model` must be a valid model pointer (audio model with mel spectrogram preprocessing).
/// - `audio_path` must be a valid, non-null, null-terminated UTF-8 path to a WAV file.
/// - `opts` may be null (use defaults).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detect_audio_v2(
    model: *const SparrowEngineModel,
    audio_path: *const c_char,
    opts: *const SparrowEngineAudioDetectOpts,
) -> *mut SparrowEngineAudioResult_v2 {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngineAudioResult_v2, String> {
            if model.is_null() {
                return Err("model pointer is null".to_string());
            }
            let handle = &*(model as *const ModelHandle);
            let path_str = cstr_to_str(audio_path)?;
            let input = AudioInput::FilePath(PathBuf::from(path_str));
            let a_opts = audio_detect_opts_from_c(opts);
            let result = crate::detect_audio::detect_audio(handle, &input, &a_opts)
                .map_err(|e| e.to_string())?;
            Ok(audio_result_v2_to_c(result))
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_detect_audio_v2".to_string());
            ptr::null_mut()
        }
    }
}

/// Callback type for streaming audio detection.
/// Called once per segment that exceeds the confidence threshold.
/// `user_data` is passed through from the caller (opaque context pointer).
pub type SparrowEngineAudioSegmentCallback =
    unsafe extern "C" fn(segment: *const SparrowEngineAudioSegment, user_data: *mut c_void);

/// Run audio detection with per-segment streaming callback.
///
/// CPU callback cadence is per-segment: the callback is invoked immediately
/// after each detected segment is produced, allowing the caller to update
/// progress incrementally.
///
/// Note: the GPU flavor of this symbol fires callbacks post-detect (after the
/// full chunk loop completes) rather than per-segment. Callers writing
/// flavor-agnostic UI code should not assume per-segment cadence; see the
/// matching doc-comment in `sparrow-engine-gpu/src/ffi.rs`.
///
/// Returns the complete result (same as `sparrow_engine_detect_audio`).
///
/// # Safety
/// - `model` must be a valid model pointer.
/// - `audio_path` must be a valid, non-null, null-terminated UTF-8 path.
/// - `opts` may be null (use defaults).
/// - `callback` must be a valid function pointer for the duration of this call.
/// - `user_data` is passed through to the callback unchanged.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detect_audio_streaming(
    model: *const SparrowEngineModel,
    audio_path: *const c_char,
    opts: *const SparrowEngineAudioDetectOpts,
    callback: SparrowEngineAudioSegmentCallback,
    user_data: *mut c_void,
) -> *mut SparrowEngineAudioResult {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> std::result::Result<*mut SparrowEngineAudioResult, String> {
            if model.is_null() {
                return Err("model pointer is null".to_string());
            }
            let handle = &*(model as *const ModelHandle);
            let path_str = cstr_to_str(audio_path)?;
            let input = AudioInput::FilePath(PathBuf::from(path_str));
            let a_opts = audio_detect_opts_from_c(opts);

            let result =
                crate::detect_audio::detect_audio_streaming(handle, &input, &a_opts, |seg| {
                    let c_seg = SparrowEngineAudioSegment {
                        start_time_s: seg.start_time_s,
                        end_time_s: seg.end_time_s,
                        confidence: seg.confidence,
                    };
                    callback(&c_seg as *const SparrowEngineAudioSegment, user_data);
                })
                .map_err(|e| e.to_string())?;
            Ok(audio_result_to_c(result))
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error(
                "internal error: panic in sparrow_engine_detect_audio_streaming".to_string(),
            );
            ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

/// Free a `SparrowEngineAudioResult` returned by `sparrow_engine_detect_audio`.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_detect_audio`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_audio_result_free(ptr: *mut SparrowEngineAudioResult) {
    clear_last_error();
    if ptr.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(ptr as *mut AudioResultWithOwner));
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_audio_result_free".to_string());
    }
}

/// Free a `SparrowEngineAudioResult_v2` returned by `sparrow_engine_detect_audio_v2`.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_detect_audio_v2`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_audio_result_v2_free(
    ptr: *mut SparrowEngineAudioResult_v2,
) {
    clear_last_error();
    if ptr.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(ptr as *mut AudioResultV2WithOwner));
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_audio_result_v2_free".to_string());
    }
}

/// Free a `SparrowEngineDetections` returned by `sparrow_engine_detect` or `sparrow_engine_detect_raw`.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_detect`/`sparrow_engine_detect_raw`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detections_free(ptr: *mut SparrowEngineDetections) {
    clear_last_error();
    if ptr.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(ptr as *mut DetectionsWithOwner));
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_detections_free".to_string());
    }
}

/// Free a `SparrowEngineClassifyResult` returned by `sparrow_engine_classify`.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_classify`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_classify_result_free(
    ptr: *mut SparrowEngineClassifyResult,
) {
    clear_last_error();
    if ptr.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(ptr as *mut ClassifyResultWithOwner));
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_classify_result_free".to_string());
    }
}

/// Free a `SparrowEnginePipelineResult` returned by `sparrow_engine_run_pipeline`.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_run_pipeline`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_pipeline_result_free(
    ptr: *mut SparrowEnginePipelineResult,
) {
    clear_last_error();
    if ptr.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(ptr as *mut PipelineResultWithOwner));
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_pipeline_result_free".to_string());
    }
}

/// Free a string returned by `sparrow_engine_list_models` or `sparrow_engine_health`.
///
/// # Safety
/// `ptr` must be a pointer returned by a sparrow-engine function that allocates strings, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_free_string(ptr: *mut c_char) {
    clear_last_error();
    if ptr.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(CString::from_raw(ptr));
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_free_string".to_string());
    }
}

// ---------------------------------------------------------------------------
// Management (JSON responses)
// ---------------------------------------------------------------------------

/// List loaded models as a JSON string. Returns null on error.
/// Caller must free with `sparrow_engine_free_string`.
///
/// # Safety
/// `engine` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_list_models(engine: *const SparrowEngine) -> *mut c_char {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_char, String> {
        if engine.is_null() {
            return Err("engine pointer is null".to_string());
        }
        let engine_ref = &*(engine as *const Engine);
        let models = engine_ref.loaded_models();
        // Use serde_json for safe escaping of model IDs and paths (may contain quotes/backslashes).
        let json_values: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "path": m.path.display().to_string(),
                    "model_type": m.model_type.as_str(),
                })
            })
            .collect();
        let json = serde_json::Value::Array(json_values).to_string();
        Ok(string_to_c(json))
    }));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_list_models".to_string());
            ptr::null_mut()
        }
    }
}

/// Return engine health as a JSON string. Returns null on error.
/// Caller must free with `sparrow_engine_free_string`.
///
/// # Safety
/// `engine` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_health(engine: *const SparrowEngine) -> *mut c_char {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_char, String> {
        if engine.is_null() {
            return Err("engine pointer is null".to_string());
        }
        let engine_ref = &*(engine as *const Engine);
        let model_count = engine_ref.loaded_models().len();
        let json = format!(r#"{{"status":"ok","models_loaded":{}}}"#, model_count);
        Ok(string_to_c(json))
    }));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_health".to_string());
            ptr::null_mut()
        }
    }
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

/// Return the last error message for this thread, or null if no error.
/// The returned pointer is valid until the next FFI call on the same thread.
///
/// # Safety
/// Thread-safe. Returned pointer must not be freed by the caller.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_last_error() -> *const c_char {
    LAST_ERROR.with(|cell| {
        let borrow = cell.borrow();
        match borrow.as_ref() {
            Some(cstr) => cstr.as_ptr(),
            None => ptr::null(),
        }
    })
}

// ===========================================================================
// Phase 3: Utility exports
// ===========================================================================

// ---------------------------------------------------------------------------
// C-compatible structs (Phase 3)
// ---------------------------------------------------------------------------

/// Day/night result. Returned by value (small struct).
#[repr(C)]
pub struct SparrowEngineDayNightResultC {
    /// 0 = success, -1 = error (check `sparrow_engine_last_error`).
    pub status: i32,
    /// 1 = day, 0 = night. Undefined if status != 0.
    pub is_day: i32,
    /// Mean brightness [0,255]. -1.0 if status != 0.
    pub brightness: f32,
}

/// Model verification result. Freed by `sparrow_engine_verify_result_free`.
#[repr(C)]
pub struct SparrowEngineVerifyResultC {
    /// 0=Ok, 1=NoChecksum, 2=SizeMismatch, 3=ChecksumMismatch.
    /// On error, `sparrow_engine_verify_model` returns null (call `sparrow_engine_last_error` for detail).
    pub status: i32,
    /// Detail message. Null if status=0 or status=1.
    pub detail: *mut c_char,
}

// ---------------------------------------------------------------------------
// Standalone (no Engine)
// ---------------------------------------------------------------------------

/// Compute SHA-256 hash of a file. Returns hex string or null on error.
/// Caller must free with `sparrow_engine_hash_result_free`.
///
/// # Safety
/// `path` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_hash_file(path: *const c_char) -> *mut c_char {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_char, String> {
        let path_str = cstr_to_str(path)?;
        let hash =
            crate::hash::hash_file(std::path::Path::new(path_str)).map_err(|e| e.to_string())?;
        Ok(string_to_c(hash))
    }));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("internal error: panic in sparrow_engine_hash_file".to_string());
            ptr::null_mut()
        }
    }
}

/// Free a hash string returned by `sparrow_engine_hash_file`.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_hash_file`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_hash_result_free(ptr: *mut c_char) {
    sparrow_engine_free_string(ptr);
}

/// Classify image as day or night. Returns result by value.
/// On error: status=-1, check `sparrow_engine_last_error`.
///
/// # Safety
/// `image` must point to `len` bytes of encoded image data (JPEG/PNG).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_day_night(
    image: *const u8,
    len: usize,
) -> SparrowEngineDayNightResultC {
    clear_last_error();
    let err_result = SparrowEngineDayNightResultC {
        status: -1,
        is_day: 0,
        brightness: -1.0,
    };
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<SparrowEngineDayNightResultC, String> {
            if image.is_null() || len == 0 {
                return Err("image data is null or empty".to_string());
            }
            let data = std::slice::from_raw_parts(image, len);
            let dn = crate::daynight::day_night(data).map_err(|e| e.to_string())?;
            Ok(SparrowEngineDayNightResultC {
                status: 0,
                is_day: if dn.classification == crate::daynight::DayNight::Day {
                    1
                } else {
                    0
                },
                brightness: dn.mean_brightness,
            })
        },
    ));
    match result {
        Ok(Ok(r)) => r,
        Ok(Err(e)) => {
            set_last_error(e);
            err_result
        }
        Err(_) => {
            set_last_error("internal error: panic in sparrow_engine_day_night".to_string());
            err_result
        }
    }
}

/// Compute mean image brightness [0,255]. Returns -1.0 on error.
///
/// # Safety
/// `image` must point to `len` bytes of encoded image data (JPEG/PNG).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_image_brightness(image: *const u8, len: usize) -> f32 {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<f32, String> {
        if image.is_null() || len == 0 {
            return Err("image data is null or empty".to_string());
        }
        let data = std::slice::from_raw_parts(image, len);
        crate::daynight::image_brightness(data).map_err(|e| e.to_string())
    }));
    match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_last_error(e);
            -1.0
        }
        Err(_) => {
            set_last_error("internal error: panic in sparrow_engine_image_brightness".to_string());
            -1.0
        }
    }
}

/// Verify a model's ONNX file against manifest checksums.
/// Returns null on error. Caller must free with `sparrow_engine_verify_result_free`.
///
/// # Safety
/// `model_dir` and `model_id` must be valid, non-null, null-terminated UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_verify_model(
    model_dir: *const c_char,
    model_id: *const c_char,
) -> *mut SparrowEngineVerifyResultC {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngineVerifyResultC, String> {
            let dir_str = cstr_to_str(model_dir)?;
            let id_str = cstr_to_str(model_id)?;
            let vr = crate::catalog::verify_model(std::path::Path::new(dir_str), id_str)
                .map_err(|e| e.to_string())?;
            Ok(verify_result_to_c(vr))
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("internal error: panic in sparrow_engine_verify_model".to_string());
            ptr::null_mut()
        }
    }
}

/// Free a `SparrowEngineVerifyResultC` returned by `sparrow_engine_verify_model` or `sparrow_engine_engine_verify_model`.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_verify_model`/`sparrow_engine_engine_verify_model`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_verify_result_free(ptr: *mut SparrowEngineVerifyResultC) {
    clear_last_error();
    if ptr.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let vr = Box::from_raw(ptr);
        if !vr.detail.is_null() {
            drop(CString::from_raw(vr.detail));
        }
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_verify_result_free".to_string());
    }
}

fn verify_result_to_c(vr: crate::catalog::VerifyResult) -> *mut SparrowEngineVerifyResultC {
    let (status, detail) = match vr {
        crate::catalog::VerifyResult::Ok => (0, ptr::null_mut()),
        crate::catalog::VerifyResult::NoChecksum => (1, ptr::null_mut()),
        crate::catalog::VerifyResult::SizeMismatch { expected, actual } => (
            2,
            string_to_c(format!("expected {expected} bytes, got {actual}")),
        ),
        crate::catalog::VerifyResult::ChecksumMismatch { expected, actual } => {
            (3, string_to_c(format!("expected {expected}, got {actual}")))
        }
    };
    Box::into_raw(Box::new(SparrowEngineVerifyResultC { status, detail }))
}

// ---------------------------------------------------------------------------
// Engine wrappers (Phase 3)
// ---------------------------------------------------------------------------

/// Verify a model using the engine's model directory. Returns null on error.
/// Caller must free with `sparrow_engine_verify_result_free`.
///
/// # Safety
/// - `engine` must be a valid engine pointer.
/// - `model_id` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_engine_verify_model(
    engine: *const SparrowEngine,
    model_id: *const c_char,
) -> *mut SparrowEngineVerifyResultC {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowEngineVerifyResultC, String> {
            if engine.is_null() {
                return Err("engine pointer is null".to_string());
            }
            let engine_ref = &*(engine as *const Engine);
            let id_str = cstr_to_str(model_id)?;
            let model_dir = &engine_ref.config().model_dir;
            let vr = crate::catalog::verify_model(model_dir, id_str).map_err(|e| e.to_string())?;
            Ok(verify_result_to_c(vr))
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(
                "internal error: panic in sparrow_engine_engine_verify_model".to_string(),
            );
            ptr::null_mut()
        }
    }
}

/// Get model info as JSON string. Searches loaded models first, then disk.
/// Returns null on error. Caller must free with `sparrow_engine_free_string`.
///
/// # Safety
/// - `engine` must be a valid engine pointer.
/// - `model_id` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_engine_model_info(
    engine: *const SparrowEngine,
    model_id: *const c_char,
) -> *mut c_char {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_char, String> {
        if engine.is_null() {
            return Err("engine pointer is null".to_string());
        }
        let engine_ref = &*(engine as *const Engine);
        let id_str = cstr_to_str(model_id)?;

        // Search loaded models first, then available models on disk.
        let loaded = engine_ref.loaded_models();
        let info = match loaded.iter().find(|m| m.id == id_str) {
            Some(m) => m.clone(),
            None => {
                let available = engine_ref.list_available_models();
                available
                    .into_iter()
                    .find(|m| m.id == id_str)
                    .ok_or_else(|| format!("model '{id_str}' not found"))?
            }
        };

        let json = serde_json::json!({
            "id": info.id,
            "path": info.path.display().to_string(),
            "model_type": info.model_type.as_str(),
            "default": info.default,
            "version": info.version,
            "description": info.description,
            "onnx_sha256": info.onnx_sha256,
            "onnx_size_bytes": info.onnx_size_bytes,
        });
        let json_str = serde_json::to_string(&json).map_err(|e| e.to_string())?;
        Ok(string_to_c(json_str))
    }));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("internal error: panic in sparrow_engine_engine_model_info".to_string());
            ptr::null_mut()
        }
    }
}

/// List all available models (on disk) as JSON array with extended info.
/// Includes version, description, and checksums. Returns null on error.
/// Caller must free with `sparrow_engine_free_string`.
///
/// # Safety
/// `engine` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_engine_list_models_extended(
    engine: *const SparrowEngine,
) -> *mut c_char {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<*mut c_char, String> {
        if engine.is_null() {
            return Err("engine pointer is null".to_string());
        }
        let engine_ref = &*(engine as *const Engine);
        let models = engine_ref.list_available_models();
        let json_values: Vec<serde_json::Value> = models
            .iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "path": m.path.display().to_string(),
                    "model_type": m.model_type.as_str(),
                    "default": m.default,
                    "version": m.version,
                    "description": m.description,
                    "onnx_sha256": m.onnx_sha256,
                    "onnx_size_bytes": m.onnx_size_bytes,
                })
            })
            .collect();
        let json_str = serde_json::Value::Array(json_values).to_string();
        Ok(string_to_c(json_str))
    }));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error(
                "internal error: panic in sparrow_engine_engine_list_models_extended".to_string(),
            );
            ptr::null_mut()
        }
    }
}

// ===========================================================================
// Version / metadata exports
// ===========================================================================

/// Returns a pointer to a static, null-terminated UTF-8 string with the
/// sparrow-engine-cpu crate version (matches `[package].version` in
/// `sparrow-engine-cpu/Cargo.toml`). Caller MUST NOT free.
///
/// Phase D B-12: useful for installer / Studio Local / brew `test do` smoke
/// tests — a zero-arg, zero-allocation entry point that proves DLL load +
/// symbol resolution without spinning up an engine. Byte-for-byte mirror of
/// the GPU FFI surface (G5 acceptance gate enforces CPU/GPU symbol parity).
///
/// # Safety
/// Thread-safe. Returned pointer is valid for the lifetime of the process.
#[no_mangle]
pub extern "C" fn sparrow_engine_version() -> *const c_char {
    // concat! evaluates at compile time; appending "\0" lets us reuse the
    // static byte slice as a C string without a runtime CString allocation.
    static VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), "\0");
    VERSION.as_ptr() as *const c_char
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{AudioClass, AudioSegment};
    use std::ffi::CStr;

    #[test]
    fn detect_result_to_c_uses_null_data_for_empty_detections() {
        let result = DetectResult {
            detections: Vec::new(),
            image_width: 640,
            image_height: 480,
            processing_time_ms: 0.0,
        };

        let ptr = detect_result_to_c(result);
        assert!(!ptr.is_null());

        unsafe {
            let header = &*ptr;
            assert_eq!(header.len, 0);
            assert!(header.data.is_null());
            sparrow_engine_detections_free(ptr);
        }
    }

    #[test]
    fn classify_result_to_c_uses_null_top_results_for_empty_classifications() {
        let result = ClassifyResult {
            classifications: Vec::new(),
            image_width: 640,
            image_height: 480,
            processing_time_ms: 0.0,
        };

        let ptr = classify_result_to_c(result);
        assert!(!ptr.is_null());

        unsafe {
            let header = &*ptr;
            assert_eq!(header.top_results_len, 0);
            assert!(header.top_results.is_null());
            sparrow_engine_classify_result_free(ptr);
        }
    }

    #[test]
    fn pipeline_result_to_c_uses_null_data_for_empty_detections() {
        let result = PipelineResult {
            pipeline_id: "pipe".to_string(),
            detections: Vec::new(),
            image_width: 640,
            image_height: 480,
            processing_time_ms: 0.0,
        };

        let ptr = pipeline_result_to_c(result);
        assert!(!ptr.is_null());

        unsafe {
            let header = &*ptr;
            assert_eq!(header.len, 0);
            assert!(header.data.is_null());
            sparrow_engine_pipeline_result_free(ptr);
        }
    }

    #[test]
    fn audio_result_v2_to_c_preserves_top_k_classes_and_labels() {
        let result = AudioDetectResult {
            segments: vec![
                AudioSegment {
                    start_time_s: 0.0,
                    end_time_s: 2.5,
                    confidence: 0.91,
                    classes: vec![
                        AudioClass {
                            class_idx: 7,
                            label: Some("sparrow".to_string()),
                            probability: 0.91,
                        },
                        AudioClass {
                            class_idx: 3,
                            label: Some("finch".to_string()),
                            probability: 0.07,
                        },
                        AudioClass {
                            class_idx: 1,
                            label: None,
                            probability: 0.02,
                        },
                    ],
                },
                AudioSegment {
                    start_time_s: 2.5,
                    end_time_s: 5.0,
                    confidence: 0.84,
                    classes: vec![
                        AudioClass {
                            class_idx: 11,
                            label: Some("owl".to_string()),
                            probability: 0.84,
                        },
                        AudioClass {
                            class_idx: 13,
                            label: Some("raven".to_string()),
                            probability: 0.12,
                        },
                        AudioClass {
                            class_idx: 17,
                            label: None,
                            probability: 0.04,
                        },
                    ],
                },
            ],
            duration_s: 5.0,
            sample_rate: 48_000,
            processing_time_ms: 12.5,
        };

        let ptr = audio_result_v2_to_c(result);
        assert!(!ptr.is_null());

        unsafe {
            let header = &*ptr;
            assert_eq!(header.len, 2);
            assert_eq!(header.duration_s, 5.0);
            assert_eq!(header.sample_rate, 48_000);
            assert_eq!(header.processing_time_ms, 12.5);
            assert!(!header.data.is_null());

            let segments = std::slice::from_raw_parts(header.data, header.len);
            let expected = [
                (
                    0.0,
                    2.5,
                    0.91,
                    [
                        (7, Some("sparrow"), 0.91),
                        (3, Some("finch"), 0.07),
                        (1, None, 0.02),
                    ],
                ),
                (
                    2.5,
                    5.0,
                    0.84,
                    [
                        (11, Some("owl"), 0.84),
                        (13, Some("raven"), 0.12),
                        (17, None, 0.04),
                    ],
                ),
            ];

            for (segment, (start, end, confidence, expected_classes)) in
                segments.iter().zip(expected.iter())
            {
                assert_eq!(segment.start_time_s, *start);
                assert_eq!(segment.end_time_s, *end);
                assert_eq!(segment.confidence, *confidence);
                assert_eq!(segment.classes_len, expected_classes.len());
                assert!(!segment.classes.is_null());

                let classes = std::slice::from_raw_parts(segment.classes, segment.classes_len);
                for (class, (class_idx, label, probability)) in
                    classes.iter().zip(expected_classes.iter())
                {
                    assert_eq!(class.class_idx, *class_idx);
                    assert_eq!(class.probability, *probability);
                    match label {
                        Some(expected_label) => {
                            assert!(!class.label.is_null());
                            let actual_label = CStr::from_ptr(class.label)
                                .to_str()
                                .expect("label should be valid UTF-8");
                            assert_eq!(actual_label, *expected_label);
                        }
                        None => assert!(class.label.is_null()),
                    }
                }
            }

            sparrow_engine_audio_result_v2_free(ptr);
        }
    }

    #[test]
    fn audio_result_to_c_uses_null_data_for_empty_segments() {
        let result = AudioDetectResult {
            segments: Vec::new(),
            duration_s: 0.0,
            sample_rate: 48_000,
            processing_time_ms: 0.0,
        };

        let ptr = audio_result_to_c(result);
        assert!(!ptr.is_null());

        unsafe {
            let header = &*ptr;
            assert_eq!(header.len, 0);
            assert!(header.data.is_null());
            sparrow_engine_audio_result_free(ptr);
        }
    }

    #[test]
    fn audio_result_v2_to_c_handles_empty_and_zero_class_segments() {
        let empty = AudioDetectResult {
            segments: Vec::new(),
            duration_s: 0.0,
            sample_rate: 48_000,
            processing_time_ms: 0.0,
        };
        let empty_ptr = audio_result_v2_to_c(empty);
        assert!(!empty_ptr.is_null());
        unsafe {
            let header = &*empty_ptr;
            assert_eq!(header.len, 0);
            assert!(header.data.is_null());
            sparrow_engine_audio_result_v2_free(empty_ptr);
        }

        let result = AudioDetectResult {
            segments: vec![
                AudioSegment {
                    start_time_s: 1.0,
                    end_time_s: 2.5,
                    confidence: 0.9,
                    classes: vec![
                        AudioClass {
                            class_idx: 7,
                            label: Some("owl\0night".to_string()),
                            probability: 0.9,
                        },
                        AudioClass {
                            class_idx: 3,
                            label: None,
                            probability: 0.2,
                        },
                    ],
                },
                AudioSegment {
                    start_time_s: 3.0,
                    end_time_s: 4.0,
                    confidence: 0.0,
                    classes: Vec::new(),
                },
            ],
            duration_s: 10.0,
            sample_rate: 48_000,
            processing_time_ms: 12.5,
        };

        let ptr = audio_result_v2_to_c(result);
        assert!(!ptr.is_null());

        unsafe {
            let header = &*ptr;
            assert_eq!(header.len, 2);
            assert!(!header.data.is_null());

            let segments = std::slice::from_raw_parts(header.data, header.len);
            assert_eq!(segments[0].classes_len, 2);
            assert!(!segments[0].classes.is_null());
            let classes = std::slice::from_raw_parts(segments[0].classes, segments[0].classes_len);
            assert_eq!(classes[0].class_idx, 7);
            assert_eq!(
                CStr::from_ptr(classes[0].label).to_str().unwrap(),
                "owlnight"
            );
            assert_eq!(classes[0].probability, 0.9);
            assert_eq!(classes[1].class_idx, 3);
            assert!(classes[1].label.is_null());
            assert_eq!(classes[1].probability, 0.2);

            assert_eq!(segments[1].classes_len, 0);
            assert!(segments[1].classes.is_null());

            sparrow_engine_audio_result_v2_free(ptr);
        }
    }
}
