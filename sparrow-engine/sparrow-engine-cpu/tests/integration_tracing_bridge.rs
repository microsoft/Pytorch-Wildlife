//
// Phase 3.8 Phase A M2 + v2 N12 closure: sparrow-engine-cpu/Cargo.toml declares
// `tracing = { version = "0.1", features = ["log"] }` UNCONDITIONALLY. The
// `log` feature wires tracing events through the `log` facade, which is what
// `pyo3-log` consumes in sparrow-engine-python without forcing pyo3 into sparrow-engine-cpu.
//
// This test verifies the bridge is actually wired by:
//   1. Installing a custom `log::Log` recorder (no `tracing-subscriber`).
//   2. Emitting `tracing::info!` from the test thread.
//   3. Asserting the recorder captured the event.
//
// If `tracing[log]` were misconfigured (e.g., a future refactor drops the
// `log` feature), the recorder would see zero records and this test fails.
//
// Dev-dep requirement: sparrow-engine-cpu/Cargo.toml [dev-dependencies] must include
//   log = "0.4"
// (or the workspace `log` if one is added). This is the ONLY new dev-dep
// required by Round 01 audit-fix.

use std::sync::Mutex;

// -----------------------------------------------------------------------------
// Test recorder — a `log::Log` impl that captures records into a Mutex<Vec>.
// -----------------------------------------------------------------------------

struct VecRecorder {
    records: Mutex<Vec<String>>,
}

impl log::Log for VecRecorder {
    fn enabled(&self, _: &log::Metadata) -> bool {
        true
    }

    fn log(&self, record: &log::Record) {
        let formatted = format!(
            "[{}] {}: {}",
            record.level(),
            record.target(),
            record.args()
        );
        if let Ok(mut v) = self.records.lock() {
            v.push(formatted);
        }
    }

    fn flush(&self) {}
}

// `set_logger` accepts only `&'static dyn log::Log`. We leak a Box (one-shot,
// test-process-scoped) — this is the standard pattern for installing a logger.
fn install_recorder() -> &'static VecRecorder {
    let recorder = Box::leak(Box::new(VecRecorder {
        records: Mutex::new(Vec::new()),
    }));
    // `set_logger` returns Err on second call. Tolerate that — the global
    // logger may already be set by a sibling integration test if multiple are
    // running in the same process. In that case we still return our recorder,
    // but it won't see records. The single-test invocation (the default) sets
    // the logger first, so the assertion fires.
    let _ = log::set_logger(recorder);
    log::set_max_level(log::LevelFilter::Trace);
    recorder
}

// -----------------------------------------------------------------------------
// Test: tracing::info! from a sparrow_engine:: function emerges through the log facade.
// -----------------------------------------------------------------------------

#[test]
fn tracing_event_reaches_log_recorder_via_bridge() {
    let recorder = install_recorder();

    // Emit through tracing — with no tracing-subscriber installed, the `log`
    // feature in tracing 0.1 makes events fall through to the `log` facade.
    // We use a unique sentinel string so we can find our event among any
    // others the test runtime might emit.
    let sentinel = "phase_a_r1_tracing_bridge_sentinel_4f8c2d";
    tracing::info!(target: "bongo_cpu_test", "{}", sentinel);

    // Allow a moment for any cross-thread bridging (tracing 0.1 + log is
    // synchronous, but a tiny yield is harmless).
    std::thread::yield_now();

    let records = recorder.records.lock().expect("recorder mutex");
    let found = records.iter().any(|r| r.contains(sentinel));

    // Diagnostic: print captured records on failure so the user can see what
    // the recorder DID see.
    assert!(
        found,
        "Sentinel '{}' not found in recorder log. Captured {} records: {:?}\n\
         This typically means tracing's `log` feature is not enabled in \
         sparrow-engine-cpu/Cargo.toml — verify [dependencies] tracing has \
         `features = [\"log\"]`.",
        sentinel,
        records.len(),
        *records
    );
}
