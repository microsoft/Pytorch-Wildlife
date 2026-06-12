//! Focused C FFI boundary for the mobile orca cascade.
//!
//! This module intentionally exposes only the RP-25 P2.3 orca cascade API, not
//! the full CPU/GPU 34-export flavor parity surface.

use crate::cascade::{OrcaCascade, OrcaCascadeResult};
use std::cell::RefCell;
use std::ffi::{c_char, c_void, CStr, CString};
use std::panic::AssertUnwindSafe;
use std::path::Path;
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
// Opaque handle type
// ===========================================================================

/// Opaque orca cascade handle. Consumers must not inspect or dereference.
pub type SparrowOrcaCascade = c_void;

struct SparrowOrcaCascadeOwned {
    cascade: OrcaCascade,
}

// ===========================================================================
// C-compatible result struct
// ===========================================================================

#[repr(C)]
pub struct SparrowOrcaResult {
    pub detector_logit: f32,
    pub detector_probability: f32,
    pub is_orca: u8,
    pub ecotype_ran: u8,
    /// Ecotype class index, or -1 when the detector gate skipped ecotype.
    pub ecotype_argmax: i32,
    /// Five ecotype probabilities. All zeros when ecotype did not run.
    pub ecotype_probabilities: [f32; 5],
}

impl Default for SparrowOrcaResult {
    fn default() -> Self {
        Self {
            detector_logit: 0.0,
            detector_probability: 0.0,
            is_orca: 0,
            ecotype_ran: 0,
            ecotype_argmax: -1,
            ecotype_probabilities: [0.0; 5],
        }
    }
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

fn orca_result_to_c(result: OrcaCascadeResult) -> SparrowOrcaResult {
    let mut out = SparrowOrcaResult {
        detector_logit: result.detector_logit,
        detector_probability: result.detector_probability,
        is_orca: u8::from(result.is_orca),
        ..SparrowOrcaResult::default()
    };

    if let Some(argmax) = result.ecotype_argmax {
        out.ecotype_ran = 1;
        out.ecotype_argmax = argmax as i32;
    }
    if let Some(probabilities) = result.ecotype_probabilities {
        for (dst, src) in out.ecotype_probabilities.iter_mut().zip(probabilities) {
            *dst = src;
        }
    }
    out
}

// ===========================================================================
// Exports
// ===========================================================================

/// Create a focused orca cascade handle from detector and ecotype TFLite paths.
///
/// Returns null on error. Call `sparrow_engine_orca_last_error` for details.
///
/// # Safety
/// `detector_path` and `ecotype_path` must be valid, non-null, null-terminated
/// UTF-8 strings.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_orca_cascade_new(
    detector_path: *const c_char,
    ecotype_path: *const c_char,
    num_threads: usize,
) -> *mut SparrowOrcaCascade {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(
        || -> Result<*mut SparrowOrcaCascade, String> {
            let detector_path = cstr_to_str(detector_path)?;
            let ecotype_path = cstr_to_str(ecotype_path)?;
            let cascade = OrcaCascade::load(
                Path::new(detector_path),
                Path::new(ecotype_path),
                num_threads,
            )
            .map_err(|e| e.to_string())?;
            Ok(Box::into_raw(Box::new(SparrowOrcaCascadeOwned { cascade }))
                as *mut SparrowOrcaCascade)
        },
    ));
    match result {
        Ok(Ok(ptr)) => ptr,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_orca_cascade_new".to_string());
            ptr::null_mut()
        }
    }
}

/// Run one raw-audio segment through the orca cascade.
///
/// Returns 0 on success and nonzero on error. On error, call
/// `sparrow_engine_orca_last_error` for details.
/// A handle is single-owner and not thread-safe: do not call `run` concurrently
/// on the same handle, and free each handle exactly once.
/// This per-segment API operates on a single 3 s window. Input is resampled to
/// 24 kHz, then truncated to or zero-padded to 72,000 samples. The caller is
/// responsible for sliding-window segmentation before calling this function.
///
/// # Safety
/// - `handle` must be a valid pointer returned by `sparrow_engine_orca_cascade_new`.
/// - `samples` must point to `n_samples` finite `f32` samples.
/// - `out` must be a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_orca_cascade_run(
    handle: *mut SparrowOrcaCascade,
    samples: *const f32,
    n_samples: usize,
    sample_rate: u32,
    out: *mut SparrowOrcaResult,
) -> i32 {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        if handle.is_null() {
            return Err("null orca cascade handle".to_string());
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
        if out.is_null() {
            return Err("null output pointer".to_string());
        }

        let owned = &mut *(handle as *mut SparrowOrcaCascadeOwned);
        let samples = std::slice::from_raw_parts(samples, n_samples);
        if !samples.iter().all(|sample| sample.is_finite()) {
            return Err("raw audio samples must be finite".to_string());
        }
        let result = owned
            .cascade
            .run_segment(samples, sample_rate)
            .map_err(|e| e.to_string())?;
        *out = orca_result_to_c(result);
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_orca_cascade_run".to_string());
            -1
        }
    }
}

/// Initialize an orca result struct to its empty/default value.
///
/// Returns 0 on success and nonzero on error. This helper lets ctypes callers
/// reset a stack-allocated result before reuse without knowing Rust defaults.
///
/// # Safety
/// `out` must be a valid writable pointer.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_orca_result_init(out: *mut SparrowOrcaResult) -> i32 {
    clear_last_error();
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        if out.is_null() {
            return Err("null output pointer".to_string());
        }
        *out = SparrowOrcaResult::default();
        Ok(())
    }));

    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(_panic) => {
            set_last_error("internal error: panic in sparrow_engine_orca_result_init".to_string());
            -1
        }
    }
}

/// Free an orca cascade handle. Null-safe.
///
/// # Safety
/// `handle` must be a pointer returned by `sparrow_engine_orca_cascade_new`, or null.
/// Each non-null handle must be freed exactly once.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_orca_cascade_free(handle: *mut SparrowOrcaCascade) {
    clear_last_error();
    if handle.is_null() {
        return;
    }
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(handle as *mut SparrowOrcaCascadeOwned));
    }));
    if result.is_err() {
        set_last_error("internal error: panic in sparrow_engine_orca_cascade_free".to_string());
    }
}

/// Return the last orca FFI error message for this thread, or null if no error.
/// The returned pointer is valid until the next orca FFI call on the same thread.
///
/// # Safety
/// Thread-safe. Returned pointer must not be freed by the caller.
#[no_mangle]
pub unsafe extern "C" fn sparrow_engine_orca_last_error() -> *const c_char {
    let result = std::panic::catch_unwind(AssertUnwindSafe(|| {
        LAST_ERROR.with(|cell| {
            let borrow = cell.borrow();
            match borrow.as_ref() {
                Some(cstr) => cstr.as_ptr(),
                None => ptr::null(),
            }
        })
    }));
    result.unwrap_or(ptr::null())
}
