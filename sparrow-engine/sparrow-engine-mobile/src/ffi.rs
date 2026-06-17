//! Generic C FFI boundary for the mobile (LiteRT) flavor — RP-25-FU-1.
//!
//! Replaces the focused 5-export orca cascade API with a documented ~18-symbol
//! subset of the cpu/gpu 35-symbol surface: engine lifecycle, model management,
//! image inference (`detect` implemented in RP-42; `classify` still deferred),
//! single-model audio detection, and the audio-cascade pipeline family (the orca
//! cascade is now a `pipeline.toml`, not C code).
//!
//! Conventions (shared with the cpu/gpu/mobile flavors): a thread-local errno
//! style last-error, `catch_unwind` on every export, opaque `Box::into_raw` /
//! `Box::from_raw` handles, and "the engine allocates / the engine frees, free
//! exactly once" for every returned buffer.
//!
//! Threading: the engine is single-threaded / thread-affine (see
//! [`crate::engine`]). Calls from a foreign thread return a clear error.

use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::AssertUnwindSafe;
use std::path::PathBuf;
use std::ptr;

use sparrow_engine_types::types::{AudioDetectOpts, AudioInput, DetectOpts, ImageInput};
use sparrow_engine_types::{Device, EngineConfig};

use crate::engine::Engine;
use crate::pipeline::CascadeOpts;

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

/// Opaque model handle returned by `sparrow_engine_load_model_by_id`.
pub type SparrowEngineModel = c_void;

// ===========================================================================
// C-compatible structs
// ===========================================================================

/// Detection bounding box, normalized `[0, 1]`.
#[repr(C)]
pub struct SparrowEngineBBox {
    pub x_min: f32,
    pub y_min: f32,
    pub x_max: f32,
    pub y_max: f32,
}

/// One image detection.
#[repr(C)]
pub struct SparrowEngineDetection {
    pub bbox: SparrowEngineBBox,
    pub label: *const c_char,
    pub label_id: u32,
    pub confidence: f32,
}

/// Image detection output (image inference is deferred to RP-42; see
/// `sparrow_engine_detect`).
#[repr(C)]
pub struct SparrowEngineDetections {
    pub data: *const SparrowEngineDetection,
    pub len: usize,
    pub image_width: u32,
    pub image_height: u32,
}

/// One image classification.
#[repr(C)]
pub struct SparrowEngineClassification {
    pub label: *const c_char,
    pub label_id: u32,
    pub confidence: f32,
}

/// Image classification output (deferred to RP-42; see `sparrow_engine_classify`).
#[repr(C)]
pub struct SparrowEngineClassifyResult {
    pub data: *const SparrowEngineClassification,
    pub len: usize,
    pub image_width: u32,
    pub image_height: u32,
    pub processing_time_ms: f32,
}

/// Image detection options.
#[repr(C)]
pub struct SparrowEngineDetectOpts {
    pub confidence_threshold: f32,
    pub max_detections: u32,
}

/// Image classification options.
#[repr(C)]
pub struct SparrowEngineClassifyOpts {
    pub top_k: u32,
}

/// One detected audio segment (single-model `sparrow_engine_detect_audio`).
#[repr(C)]
pub struct SparrowEngineAudioSegment {
    pub start_time_s: f32,
    pub end_time_s: f32,
    pub confidence: f32,
}

/// Single-model audio detection output.
#[repr(C)]
pub struct SparrowEngineAudioResult {
    pub data: *const SparrowEngineAudioSegment,
    pub len: usize,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
}

/// Single-model audio detection options. A `NaN` field means "use the manifest
/// default" (C has no `Option`).
#[repr(C)]
pub struct SparrowEngineAudioDetectOpts {
    pub confidence_threshold: f32,
    pub segment_duration_s: f32,
    pub segment_stride_s: f32,
}

/// One audio-cascade window result.
#[repr(C)]
pub struct SparrowEngineCascadeSegment {
    pub start_s: f32,
    pub end_s: f32,
    pub detector_logit: f32,
    pub detector_probability: f32,
    /// 1 if stage 1 fired (`detector_probability >= threshold`), else 0.
    pub is_detected: u8,
    /// 1 if stage 2 (classifier) ran, else 0.
    pub stage2_ran: u8,
    /// Stage-2 argmax class index, or -1 when stage 2 did not run.
    pub stage2_argmax: i32,
    /// Stage-2 top probability, or 0 when stage 2 did not run.
    pub stage2_confidence: f32,
}

/// Audio-cascade output. `stage2_probabilities` is a flat row-major buffer of
/// `len * num_stage2_classes` values (segment `i`, class `c` lives at
/// `stage2_probabilities[i * num_stage2_classes + c]`); rows where stage 2 did
/// not run are all-zero.
#[repr(C)]
pub struct SparrowEngineCascadeResult {
    pub pipeline_id: *const c_char,
    pub data: *const SparrowEngineCascadeSegment,
    pub len: usize,
    pub num_stage2_classes: usize,
    pub stage2_probabilities: *const f32,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
}

/// Audio-cascade options. A `NaN` field means "use the pipeline default".
#[repr(C)]
pub struct SparrowEngineCascadeOpts {
    pub window_sec: f32,
    pub overlap_sec: f32,
    pub detector_threshold: f32,
}

// ===========================================================================
// Helpers
// ===========================================================================

unsafe fn cstr_to_str<'a>(ptr: *const c_char) -> Result<&'a str, String> {
    if ptr.is_null() {
        return Err("null string pointer".to_string());
    }
    CStr::from_ptr(ptr)
        .to_str()
        .map_err(|e| format!("invalid UTF-8: {e}"))
}

/// `NaN` sentinel → `None` (C has no `Option<f32>`).
fn opt_f32(v: f32) -> Option<f32> {
    if v.is_nan() {
        None
    } else {
        Some(v)
    }
}

/// Run a fallible body that returns a pointer; on `Err`/panic set the last error
/// and return null.
fn guard_ptr<T>(f: impl FnOnce() -> Result<*mut T, String>) -> *mut T {
    clear_last_error();
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("internal error: panic at FFI boundary".to_string());
            ptr::null_mut()
        }
    }
}

/// Run a fallible body that returns 0/-1; on `Err`/panic set the last error and
/// return -1.
fn guard_int(f: impl FnOnce() -> Result<(), String>) -> i32 {
    clear_last_error();
    match std::panic::catch_unwind(AssertUnwindSafe(f)) {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(_) => {
            set_last_error("internal error: panic at FFI boundary".to_string());
            -1
        }
    }
}

#[derive(serde::Deserialize)]
struct RawConfig {
    #[serde(default)]
    intra_threads: Option<u32>,
    model_dir: String,
}

// ===========================================================================
// Engine lifecycle
// ===========================================================================

/// Create an engine from a JSON config string: `{"model_dir": "...",
/// "intra_threads": 4}`. `intra_threads` is the LiteRT CPU thread count
/// (defaults to 4 — the Pi Zero 2W validated setting); `0` = LiteRT default.
/// Returns null on error; call `sparrow_engine_last_error` for details.
///
/// # Safety
/// `config_json` must be a valid, non-null, null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_engine_new(
    config_json: *const c_char,
) -> *mut SparrowEngine {
    guard_ptr(|| {
        let json = cstr_to_str(config_json)?;
        let raw: RawConfig =
            serde_json::from_str(json).map_err(|e| format!("invalid config JSON: {e}"))?;
        let config = EngineConfig {
            device: Device::Cpu,
            inter_threads: 0,
            intra_threads: raw.intra_threads.unwrap_or(4),
            model_dir: PathBuf::from(raw.model_dir),
        };
        let engine = Engine::new(config).map_err(|e| format!("{e:#}"))?;
        Ok(Box::into_raw(Box::new(engine)) as *mut SparrowEngine)
    })
}

/// Free an engine. Null-safe. Each non-null engine must be freed exactly once.
///
/// # Safety
/// `engine` must be a pointer returned by `sparrow_engine_engine_new`, or null.
/// Like every engine call, this must run on the thread that created the engine
/// (single-threaded contract — see the crate-level threading note); freeing from
/// another thread while the owner thread is mid-call is undefined behaviour.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_engine_free(engine: *mut SparrowEngine) {
    clear_last_error();
    if engine.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(engine as *mut Engine));
    }));
}

/// Return the last error message for this thread, or null if none. The pointer
/// is valid until the next FFI call on the same thread; do not free it.
///
/// # Safety
/// Thread-safe with respect to other threads' last-error state.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_last_error() -> *const c_char {
    std::panic::catch_unwind(|| {
        LAST_ERROR.with(|cell| {
            cell.borrow()
                .as_ref()
                .map(|c| c.as_ptr())
                .unwrap_or(ptr::null())
        })
    })
    .unwrap_or(ptr::null())
}

/// Free a string returned by the engine (e.g. `sparrow_engine_list_models`).
/// Null-safe.
///
/// # Safety
/// `ptr` must be a string returned by an engine FFI function, or null, and freed
/// exactly once.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_free_string(ptr: *mut c_char) {
    if ptr.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(CString::from_raw(ptr));
    }));
}

/// Return the engine version (static; do not free).
#[no_mangle]
pub extern "C" fn sparrow_engine_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr() as *const c_char
}

// ===========================================================================
// Model management
// ===========================================================================

/// Load a model by catalog id. Returns null on error.
///
/// # Safety
/// `engine` must be a valid engine pointer; `model_id` a valid C string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_load_model_by_id(
    engine: *mut SparrowEngine,
    model_id: *const c_char,
) -> *mut SparrowEngineModel {
    guard_ptr(|| {
        if engine.is_null() {
            return Err("null engine handle".to_string());
        }
        let engine = &*(engine as *const Engine);
        let id = cstr_to_str(model_id)?;
        let handle = engine.load_model_by_id(id).map_err(|e| format!("{e:#}"))?;
        Ok(Box::into_raw(Box::new(handle)) as *mut SparrowEngineModel)
    })
}

/// Unload the model this handle refers to and free the handle. Null-safe.
///
/// # Safety
/// `model` must be a pointer returned by `sparrow_engine_load_model_by_id`, or
/// null, and freed exactly once. Must run on the engine's owner thread
/// (single-threaded contract — see the crate-level threading note).
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_unload_model(model: *mut SparrowEngineModel) {
    clear_last_error();
    if model.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let handle = Box::from_raw(model as *mut crate::engine::MobileModel);
        // Best-effort unload from the engine cache; ignore errors (e.g. engine
        // already freed) — the handle is freed regardless when `handle` drops.
        let _ = handle.unload();
    }));
}

/// Return a JSON array of available models in the model directory. Caller frees
/// with `sparrow_engine_free_string`. Returns null on error.
///
/// # Safety
/// `engine` must be a valid engine pointer.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_list_models(engine: *const SparrowEngine) -> *mut c_char {
    guard_ptr(|| {
        if engine.is_null() {
            return Err("null engine handle".to_string());
        }
        let engine = &*(engine as *const Engine);
        let models = engine.list_models().map_err(|e| format!("{e:#}"))?;
        let json: Vec<_> = models
            .into_iter()
            .map(|m| {
                serde_json::json!({
                    "id": m.id,
                    "model_type": m.model_type.as_str(),
                    "default": m.default,
                    "version": m.version,
                    "description": m.description,
                })
            })
            .collect();
        let s = serde_json::to_string(&json).map_err(|e| format!("{e}"))?;
        Ok(CString::new(s)
            .map_err(|e| format!("{e}"))?
            .into_raw())
    })
}

// ===========================================================================
// Image inference
// ===========================================================================

/// Run single-shot image detection over an encoded image buffer (JPEG/PNG).
/// Returns null on error; call `sparrow_engine_last_error` for details. Free the
/// result with `sparrow_engine_detections_free`.
///
/// # Safety
/// `model` must be a valid model pointer; `image` must point to `len` readable
/// bytes; `opts` a valid pointer or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detect(
    model: *const SparrowEngineModel,
    image: *const u8,
    len: usize,
    opts: *const SparrowEngineDetectOpts,
) -> *mut SparrowEngineDetections {
    guard_ptr(|| {
        if model.is_null() {
            return Err("null model handle".to_string());
        }
        if image.is_null() || len == 0 {
            return Err("null or empty image buffer".to_string());
        }
        let model = &*(model as *const crate::engine::MobileModel);
        let bytes = std::slice::from_raw_parts(image, len).to_vec();

        let detect_opts = if opts.is_null() {
            DetectOpts::default()
        } else {
            let o = &*opts;
            DetectOpts {
                confidence_threshold: opt_f32(o.confidence_threshold),
                max_detections: if o.max_detections == 0 {
                    None
                } else {
                    Some(o.max_detections)
                },
            }
        };

        let result = model
            .detect(&ImageInput::Encoded(bytes), &detect_opts)
            .map_err(|e| format!("{e:#}"))?;

        let detections: Vec<SparrowEngineDetection> = result
            .detections
            .iter()
            .map(|d| SparrowEngineDetection {
                bbox: SparrowEngineBBox {
                    x_min: d.bbox.x_min,
                    y_min: d.bbox.y_min,
                    x_max: d.bbox.x_max,
                    y_max: d.bbox.y_max,
                },
                label: CString::new(d.label.as_str())
                    .unwrap_or_else(|_| CString::new("?").unwrap())
                    .into_raw(),
                label_id: d.label_id,
                confidence: d.confidence,
            })
            .collect();
        let out_len = detections.len();
        let boxed = detections.into_boxed_slice();
        let data = boxed.as_ptr();
        std::mem::forget(boxed);

        Ok(Box::into_raw(Box::new(SparrowEngineDetections {
            data,
            len: out_len,
            image_width: result.image_width,
            image_height: result.image_height,
        })))
    })
}

/// Image classification — not yet available on the mobile flavor (no `.tflite`
/// classifier onboarded). Image *detection* (`sparrow_engine_detect`) is
/// available as of RP-42. Always returns null with a clear last-error.
///
/// # Safety
/// `model` must be a valid model pointer.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_classify(
    _model: *const SparrowEngineModel,
    _image: *const u8,
    _len: usize,
    _opts: *const SparrowEngineClassifyOpts,
) -> *mut SparrowEngineClassifyResult {
    guard_ptr(|| Err(classify_unsupported_msg()))
}

fn classify_unsupported_msg() -> String {
    crate::engine::CLASSIFY_UNSUPPORTED_MSG.to_string()
}

/// Free a detections result. Null-safe.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_detect`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detections_free(ptr: *mut SparrowEngineDetections) {
    if ptr.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let result = Box::from_raw(ptr);
        if !result.data.is_null() && result.len > 0 {
            let detections = Vec::from_raw_parts(
                result.data as *mut SparrowEngineDetection,
                result.len,
                result.len,
            );
            for d in &detections {
                if !d.label.is_null() {
                    drop(CString::from_raw(d.label as *mut c_char));
                }
            }
            drop(detections);
        }
    }));
}

/// Free a classify result. Null-safe.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_classify`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_classify_result_free(
    ptr: *mut SparrowEngineClassifyResult,
) {
    if ptr.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let result = Box::from_raw(ptr);
        if !result.data.is_null() && result.len > 0 {
            let classes = Vec::from_raw_parts(
                result.data as *mut SparrowEngineClassification,
                result.len,
                result.len,
            );
            for c in &classes {
                if !c.label.is_null() {
                    drop(CString::from_raw(c.label as *mut c_char));
                }
            }
            drop(classes);
        }
    }));
}

// ===========================================================================
// Audio detection (single model)
// ===========================================================================

/// Run single-model audio detection over a WAV file. Returns null on error.
///
/// # Safety
/// `model` must be a valid model pointer; `audio_path` a valid C string; `opts`
/// a valid pointer or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_detect_audio(
    model: *const SparrowEngineModel,
    audio_path: *const c_char,
    opts: *const SparrowEngineAudioDetectOpts,
) -> *mut SparrowEngineAudioResult {
    guard_ptr(|| {
        if model.is_null() {
            return Err("null model handle".to_string());
        }
        let model = &*(model as *const crate::engine::MobileModel);
        let path = cstr_to_str(audio_path)?;

        let audio_opts = if opts.is_null() {
            AudioDetectOpts::default()
        } else {
            let o = &*opts;
            AudioDetectOpts {
                confidence_threshold: opt_f32(o.confidence_threshold),
                segment_duration_s: opt_f32(o.segment_duration_s),
                stride_s: opt_f32(o.segment_stride_s),
            }
        };

        let result = model
            .detect_audio(&AudioInput::FilePath(PathBuf::from(path)), &audio_opts)
            .map_err(|e| format!("{e:#}"))?;

        let segments: Vec<SparrowEngineAudioSegment> = result
            .segments
            .iter()
            .map(|s| SparrowEngineAudioSegment {
                start_time_s: s.start_time_s,
                end_time_s: s.end_time_s,
                confidence: s.confidence,
            })
            .collect();
        let len = segments.len();
        let boxed = segments.into_boxed_slice();
        let data = boxed.as_ptr();
        std::mem::forget(boxed);

        Ok(Box::into_raw(Box::new(SparrowEngineAudioResult {
            data,
            len,
            duration_s: result.duration_s,
            sample_rate: result.sample_rate,
            processing_time_ms: result.processing_time_ms,
        })))
    })
}

/// Free an audio result. Null-safe.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_detect_audio`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_audio_result_free(ptr: *mut SparrowEngineAudioResult) {
    if ptr.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let result = Box::from_raw(ptr);
        if !result.data.is_null() && result.len > 0 {
            drop(Vec::from_raw_parts(
                result.data as *mut SparrowEngineAudioSegment,
                result.len,
                result.len,
            ));
        }
    }));
}

// ===========================================================================
// Pipeline (audio cascade)
// ===========================================================================

/// Load an audio-cascade pipeline by catalog id. Returns 0 on success, -1 on
/// error (call `sparrow_engine_last_error`).
///
/// # Safety
/// `engine` must be a valid engine pointer; `pipeline_id` a valid C string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_load_pipeline_by_id(
    engine: *mut SparrowEngine,
    pipeline_id: *const c_char,
) -> i32 {
    guard_int(|| {
        if engine.is_null() {
            return Err("null engine handle".to_string());
        }
        let engine = &*(engine as *const Engine);
        let id = cstr_to_str(pipeline_id)?;
        engine.load_pipeline_by_id(id).map_err(|e| format!("{e:#}"))
    })
}

/// Run a loaded audio-cascade pipeline over raw mono `f32` samples. Returns null
/// on error.
///
/// # Safety
/// `engine` must be valid; `pipeline_id` a valid C string; `samples` must point
/// to `n_samples` finite `f32`; `opts` a valid pointer or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_run_pipeline(
    engine: *const SparrowEngine,
    pipeline_id: *const c_char,
    samples: *const f32,
    n_samples: usize,
    sample_rate: u32,
    opts: *const SparrowEngineCascadeOpts,
) -> *mut SparrowEngineCascadeResult {
    guard_ptr(|| {
        if engine.is_null() {
            return Err("null engine handle".to_string());
        }
        if samples.is_null() {
            return Err("null samples pointer".to_string());
        }
        if n_samples == 0 {
            return Err("n_samples must be greater than 0".to_string());
        }
        if sample_rate == 0 {
            return Err("sample_rate must be greater than 0".to_string());
        }
        let engine = &*(engine as *const Engine);
        let id = cstr_to_str(pipeline_id)?;
        let samples = std::slice::from_raw_parts(samples, n_samples);
        if !samples.iter().all(|s| s.is_finite()) {
            return Err("audio samples must be finite".to_string());
        }

        let cascade_opts = if opts.is_null() {
            CascadeOpts::default()
        } else {
            let o = &*opts;
            CascadeOpts {
                window_sec: opt_f32(o.window_sec),
                overlap_sec: opt_f32(o.overlap_sec),
                detector_threshold: opt_f32(o.detector_threshold),
            }
        };

        let result = engine
            .run_pipeline(
                id,
                &AudioInput::Samples {
                    data: samples.to_vec(),
                    sample_rate,
                },
                &cascade_opts,
            )
            .map_err(|e| format!("{e:#}"))?;

        Ok(cascade_result_to_c(result))
    })
}

fn cascade_result_to_c(
    result: crate::pipeline::CascadeResult,
) -> *mut SparrowEngineCascadeResult {
    let num_classes = result.num_stage2_classes;
    let len = result.segments.len();

    let mut segs: Vec<SparrowEngineCascadeSegment> = Vec::with_capacity(len);
    let mut probs: Vec<f32> = Vec::with_capacity(len * num_classes);
    for s in &result.segments {
        segs.push(SparrowEngineCascadeSegment {
            start_s: s.start_s,
            end_s: s.end_s,
            detector_logit: s.detector_logit,
            detector_probability: s.detector_probability,
            is_detected: u8::from(s.is_detected),
            stage2_ran: u8::from(s.stage2_ran),
            stage2_argmax: s.stage2_argmax.map(|i| i as i32).unwrap_or(-1),
            stage2_confidence: s.stage2_confidence,
        });
        if s.stage2_probabilities.len() == num_classes {
            probs.extend_from_slice(&s.stage2_probabilities);
        } else {
            probs.extend(std::iter::repeat_n(0.0f32, num_classes));
        }
    }

    let segs_boxed = segs.into_boxed_slice();
    let data = segs_boxed.as_ptr();
    std::mem::forget(segs_boxed);

    // Alloc/free length invariant: the probs buffer is exactly len * num_classes
    // (each segment contributes num_classes values, zero-filled when stage 2 was
    // skipped). pipeline_result_free reconstructs it with the same product, so a
    // future change to the row-fill loop that breaks this would be UB.
    debug_assert_eq!(probs.len(), len * num_classes);

    let (probs_ptr, _probs_len) = if probs.is_empty() {
        (ptr::null(), 0)
    } else {
        let probs_boxed = probs.into_boxed_slice();
        let p = probs_boxed.as_ptr();
        let n = probs_boxed.len();
        std::mem::forget(probs_boxed);
        (p, n)
    };

    let pipeline_id = CString::new(result.pipeline_id)
        .unwrap_or_default()
        .into_raw();

    Box::into_raw(Box::new(SparrowEngineCascadeResult {
        pipeline_id,
        data,
        len,
        num_stage2_classes: num_classes,
        stage2_probabilities: probs_ptr,
        duration_s: result.duration_s,
        sample_rate: result.sample_rate,
        processing_time_ms: result.processing_time_ms,
    }))
}

/// Unload a pipeline by id (its stage models stay loaded). Returns 0 / -1.
///
/// # Safety
/// `engine` must be valid; `pipeline_id` a valid C string.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_unload_pipeline(
    engine: *mut SparrowEngine,
    pipeline_id: *const c_char,
) -> i32 {
    guard_int(|| {
        if engine.is_null() {
            return Err("null engine handle".to_string());
        }
        let engine = &*(engine as *const Engine);
        let id = cstr_to_str(pipeline_id)?;
        engine.unload_pipeline(id).map_err(|e| format!("{e:#}"))
    })
}

/// Free a cascade result. Null-safe.
///
/// # Safety
/// `ptr` must be a pointer returned by `sparrow_engine_run_pipeline`, or null.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_pipeline_result_free(
    ptr: *mut SparrowEngineCascadeResult,
) {
    if ptr.is_null() {
        return;
    }
    let _ = std::panic::catch_unwind(AssertUnwindSafe(|| {
        let result = Box::from_raw(ptr);
        if !result.pipeline_id.is_null() {
            drop(CString::from_raw(result.pipeline_id as *mut c_char));
        }
        if !result.data.is_null() && result.len > 0 {
            drop(Vec::from_raw_parts(
                result.data as *mut SparrowEngineCascadeSegment,
                result.len,
                result.len,
            ));
        }
        if !result.stage2_probabilities.is_null() {
            let n = result.len * result.num_stage2_classes;
            if n > 0 {
                drop(Vec::from_raw_parts(
                    result.stage2_probabilities as *mut f32,
                    n,
                    n,
                ));
            }
        }
    }));
}
