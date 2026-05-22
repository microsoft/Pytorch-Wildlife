//! Lightweight per-stage timing capture for performance investigation.
//!
//! Activated via env var `SPARROW_ENGINE_GPU_PROFILE_DUMP=<json_path>`. When set,
//! every `detect()` call records per-stage durations into a thread-local
//! buffer.
//!
//! Persistence is explicit: callers must invoke
//! [`crate::profile::dump_to_path`] at the end of their bench / profile run.
//! There is no Drop-time auto-flush — Rust's `Drop` is not guaranteed to
//! run on process termination via signal or `std::process::exit`, so
//! relying on Drop for IO is unsafe. The bench harness at
//! `examples/bench_step1_full.rs` calls `dump_to_path()` from `run()`'s
//! exit path; consumers writing their own bench loops must do the same.
//!
//! In multi-threaded callers, each worker thread holds its own
//! `THREAD_BUF`. Records are merged into the global buffer on
//! [`flush_thread`] (called automatically inside `dump_to_path`); a thread
//! that exits without flushing loses its records. Wave 5 may add an
//! automatic per-thread flush hook; Wave 1/2/3/4 callers are
//! single-threaded inference loops where this is a non-issue.
//!
//! Stage labels match the Phase 3.7 prototype's per-stage profile so
//! results are directly comparable:
//!
//! | label             | what it covers                                     |
//! |-------------------|----------------------------------------------------|
//! | `bytes`           | ImageInput → byte buffer (file read or clone)      |
//! | `decode`          | nvjpeg (or CPU fallback) + sync                    |
//! | `letterbox`       | CUDA letterbox kernel + sync                       |
//! | `dtoh_in`         | DtoH copy of preprocess tensor (host roundtrip)    |
//! | `htod_in`         | HtoD re-upload by ORT.run                          |
//! | `infer`           | ORT Session::run                                    |
//! | `extract`         | output extract + Array2 owned copy                 |
//! | `postprocess`     | sparrow_engine_core::postprocess::yolo_e2e                  |
//! | `total`           | wall time inside detect()                          |
//!
//! NOT in production builds: gated by env var, no overhead when disabled.
//! Default off; set `SPARROW_ENGINE_GPU_PROFILE_DUMP=/tmp/foo.json` to enable.

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

thread_local! {
    pub static THREAD_BUF: RefCell<Vec<HashMap<&'static str, f64>>> = const { RefCell::new(Vec::new()) };
}

static GLOBAL_BUF: Mutex<Vec<HashMap<&'static str, f64>>> = Mutex::new(Vec::new());

/// Cached boolean for `SPARROW_ENGINE_GPU_PROFILE_DUMP` presence. Resolved once at
/// first `enabled()` call; subsequent calls are a single atomic load.
/// Process-lifetime: changing the env var after the first call has no
/// effect (single-process bench is the supported usage; matches the
/// `YoloModel.use_host_roundtrip` pattern at load() time).
static ENABLED: OnceLock<bool> = OnceLock::new();

/// Returns true if profiling is enabled for this process.
///
/// Cached on first call to avoid one `std::env::var` syscall per `detect()`
/// invocation in the bench-run hot path. The env var is read exactly once;
/// later mutations are not observed.
pub fn enabled() -> bool {
    *ENABLED.get_or_init(|| std::env::var("SPARROW_ENGINE_GPU_PROFILE_DUMP").is_ok())
}

/// Push a single per-call profile record.
pub fn push(rec: HashMap<&'static str, f64>) {
    THREAD_BUF.with(|tb| tb.borrow_mut().push(rec));
}

/// Flush thread-local records into the global buffer.
/// Called at end of bench loop to ensure JSON dump captures everything.
pub fn flush_thread() {
    THREAD_BUF.with(|tb| {
        let mut tb = tb.borrow_mut();
        if tb.is_empty() {
            return;
        }
        let mut g = GLOBAL_BUF.lock().expect("profile global mutex poisoned");
        g.append(&mut tb);
    });
}

/// Dump all collected records as JSON to the path in `SPARROW_ENGINE_GPU_PROFILE_DUMP`.
/// Call at the end of a bench run.
pub fn dump_to_path() {
    flush_thread();
    let path = match std::env::var("SPARROW_ENGINE_GPU_PROFILE_DUMP") {
        Ok(p) => p,
        Err(_) => return,
    };
    let g = GLOBAL_BUF.lock().expect("profile global mutex poisoned");
    let mut s = String::from("[");
    for (i, rec) in g.iter().enumerate() {
        if i > 0 {
            s.push(',');
        }
        s.push('{');
        let mut first = true;
        // Stable iteration: sort keys alphabetically.
        let mut keys: Vec<&&'static str> = rec.keys().collect();
        keys.sort();
        for k in keys {
            if !first {
                s.push(',');
            }
            first = false;
            s.push('"');
            s.push_str(k);
            s.push_str("\":");
            // Format with 4 decimals.
            s.push_str(&format!("{:.4}", rec[k]));
        }
        s.push('}');
    }
    s.push(']');
    let _ = std::fs::write(&path, s);
    eprintln!("[sparrow_engine_gpu::profile] wrote {} records to {}", g.len(), path);
}
