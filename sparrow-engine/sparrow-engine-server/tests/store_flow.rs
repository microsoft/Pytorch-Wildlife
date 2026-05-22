//! Phase 4 W3 — `?store=true` emit path integration tests.
//!
//! - Happy-path emit: `?store=true` produces a record with all required
//!   fields populated (request_id, timestamp_utc, media_hash, drift_metrics).
//! - Default off: `?store=false` (or absent) emits nothing.
//! - Halt-on-failure: failing sink + `halt_on_store_failure=true` returns
//!   500; failing sink without halt returns 200.
//!
//! These tests don't need ORT — they exercise the emit pipeline in isolation
//! by calling `emit_log_record` + `build_log_record` against a fake
//! `InferenceLogSink`. Route-level `store=true` coverage requires ORT-backed
//! handler tests with an injectable sink and is tracked separately; this file
//! deliberately covers only the sink/log-record layer.

use std::sync::{Arc, Mutex, OnceLock};

use sparrow_engine_server::config::{Config, LogFormat};
use sparrow_engine_server::discover::Catalog;
use sparrow_engine_server::engine_dispatch::{
    Device, DriftMetrics, Engine, EngineConfig, InferenceLogRecord,
};
use sparrow_engine_server::handlers::{build_log_record, emit_log_record};
use sparrow_engine_server::sink::{InferenceLogSink, SinkError};
use sparrow_engine_server::state::AppState;

// ---------------------------------------------------------------------------
// Test sinks
// ---------------------------------------------------------------------------

/// In-memory sink that records every emit.
struct CollectingSink(Mutex<Vec<InferenceLogRecord>>);

impl CollectingSink {
    fn new() -> Self {
        Self(Mutex::new(Vec::new()))
    }
    fn records(&self) -> Vec<InferenceLogRecord> {
        self.0.lock().unwrap().clone()
    }
}

impl InferenceLogSink for CollectingSink {
    fn emit(&self, record: &InferenceLogRecord) -> Result<(), SinkError> {
        self.0.lock().unwrap().push(record.clone());
        Ok(())
    }
}

/// Sink that always errors. Used to exercise halt-vs-warn paths.
struct FailingSink;

impl InferenceLogSink for FailingSink {
    fn emit(&self, _record: &InferenceLogRecord) -> Result<(), SinkError> {
        Err(SinkError("simulated sink failure".to_string()))
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn fake_record() -> InferenceLogRecord {
    InferenceLogRecord {
        schema_version: "1.0".to_string(),
        request_id: "00000000-0000-0000-0000-000000000001".to_string(),
        timestamp_utc: "2026-05-07T12:34:56.789Z".to_string(),
        media_hash: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08".to_string(),
        model_id: "fake-mdv6".to_string(),
        model_version: None,
        device: "cpu".to_string(),
        inference_ms: 1.0,
        result: serde_json::json!({"detections": []}),
        provenance: None,
        drift_metrics: Some(DriftMetrics::default()),
    }
}

fn test_config() -> Config {
    Config {
        bind_addr: "127.0.0.1:0".parse().expect("test bind addr"),
        model_dir: std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("store_flow_state"),
        log_format: LogFormat::Pretty,
        log_level: "warn".to_string(),
        max_body_size: 1024 * 1024,
        max_concurrent_inference: 1,
        max_batch_size: 1,
        request_timeout_secs: 10,
        drain_timeout_secs: 5,
        device: "cpu".to_string(),
        inter_threads: None,
        intra_threads: None,
        idle_unload_seconds: 0,
        idle_unload_keep_last_n: 1,
    }
}

fn engine_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static ENGINE_TEST_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    ENGINE_TEST_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .expect("engine test lock poisoned")
}

fn test_state_with_sink(sink: Arc<dyn InferenceLogSink>) -> AppState {
    let config = test_config();
    std::fs::create_dir_all(&config.model_dir).expect("create store_flow_state model dir");
    let engine = Engine::new(EngineConfig::new(Device::Cpu, config.model_dir.clone()))
        .expect("create test engine for store-flow helpers");
    AppState::with_catalog_and_sink(engine, config, Catalog::default(), sink)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[test]
fn collecting_sink_emits_one_record_per_call() {
    let sink = Arc::new(CollectingSink::new());
    let r = fake_record();
    sink.emit(&r).expect("ok sink must succeed");
    sink.emit(&r).expect("second emit must succeed");
    let records = sink.records();
    assert_eq!(records.len(), 2, "two emits must produce two records");
    assert_eq!(records[0], r);
    assert_eq!(records[1], r);
}

#[test]
fn failing_sink_returns_error() {
    let sink = FailingSink;
    let r = fake_record();
    let err = sink.emit(&r).expect_err("failing sink must error");
    let msg = err.to_string();
    assert!(
        msg.contains("simulated sink failure"),
        "error must surface the underlying message, got: {msg}"
    );
}

#[test]
fn emit_log_record_halt_false_swallows_sink_error() {
    let _engine_guard = engine_test_lock();
    let state = test_state_with_sink(Arc::new(FailingSink));
    let r = fake_record();
    if emit_log_record(&state, &r, false).is_err() {
        panic!("halt=false must not fail request");
    }
}

#[test]
fn emit_log_record_halt_true_returns_internal_error() {
    let _engine_guard = engine_test_lock();
    let state = test_state_with_sink(Arc::new(FailingSink));
    let r = fake_record();
    let err = match emit_log_record(&state, &r, true) {
        Ok(()) => panic!("halt=true must fail request"),
        Err(err) => err,
    };
    match err {
        sparrow_engine_server::error::AppError::Http {
            status,
            code,
            message,
        } => {
            assert_eq!(status, axum::http::StatusCode::INTERNAL_SERVER_ERROR);
            assert_eq!(code, "INTERNAL_ERROR");
            assert!(
                message.contains("inference log sink failed"),
                "error must identify sink failure, got: {message}"
            );
        }
        sparrow_engine_server::error::AppError::Bongo(_) => panic!("expected HTTP internal error"),
    }
}

#[test]
fn build_log_record_populates_server_fields() {
    let _engine_guard = engine_test_lock();
    let state = test_state_with_sink(Arc::new(CollectingSink::new()));
    let record = build_log_record(
        &state,
        "abc123".to_string(),
        "model-a".to_string(),
        serde_json::json!({"ok": true}),
        12.5,
        DriftMetrics::default(),
        None,
    );
    assert_eq!(record.schema_version, "1.0");
    assert_eq!(record.media_hash, "abc123");
    assert_eq!(record.model_id, "model-a");
    assert_eq!(record.device, "cpu");
    assert_eq!(record.inference_ms, 12.5);
    assert!(record.drift_metrics.is_some());
    assert!(uuid::Uuid::parse_str(&record.request_id).is_ok());
    assert!(record.timestamp_utc.ends_with('Z'));
}

#[test]
fn record_round_trips_to_json_with_required_fields() {
    // Locks the JSON shape that sparrow-data ingest will rely on.
    let r = fake_record();
    let json_str = serde_json::to_string(&r).expect("serialize");
    // Required fields must be present.
    for required in [
        "schema_version",
        "request_id",
        "timestamp_utc",
        "media_hash",
        "model_id",
        "device",
        "inference_ms",
        "result",
        "drift_metrics",
    ] {
        assert!(
            json_str.contains(&format!("\"{required}\"")),
            "{required} must appear in the JSON, got: {json_str}"
        );
    }
    // Optional None fields must NOT appear.
    assert!(
        !json_str.contains("\"model_version\""),
        "model_version=None must be skipped, got: {json_str}"
    );
    assert!(
        !json_str.contains("\"provenance\""),
        "provenance=None must be skipped, got: {json_str}"
    );

    // Round-trip parse.
    let parsed: InferenceLogRecord = serde_json::from_str(&json_str).expect("deserialize");
    assert_eq!(parsed, r);
}

#[test]
fn schema_version_is_locked_to_one_zero() {
    use sparrow_engine_server::engine_dispatch::SCHEMA_VERSION;
    assert_eq!(SCHEMA_VERSION, "1.0");
}

// ---------------------------------------------------------------------------
// Phase 4 audit-fix R1: T-3 — sha256_lower_hex regression tests
// ---------------------------------------------------------------------------

/// T-3a — Pin the canonical NIST FIPS-180-4 vector for `sha256_lower_hex(b"abc")`.
/// Locks output length (64 hex chars) and lowercase invariant. A future
/// refactor that switches the format string from `{:02x}` to `{:02X}`, or
/// changes the underlying digest, would break this test.
#[test]
fn sha256_lower_hex_known_vector_abc() {
    let h = sparrow_engine_server::handlers::sha256_lower_hex(b"abc");
    assert_eq!(
        h, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        "sha256_lower_hex(b\"abc\") must match NIST FIPS-180-4 test vector"
    );
    assert_eq!(h.len(), 64, "SHA-256 lowercase hex must be 64 chars");
    assert!(
        h.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase()),
        "all chars must be ASCII lowercase hex digits, got: {h}"
    );
}

/// T-3b — Empty input still produces the 64-char SHA-256 of empty bytes.
/// (Empty input is unreachable in production via the multipart extractor's
/// `bytes.is_empty()` guard, but the function must remain total.)
#[test]
fn sha256_lower_hex_empty_input_64_chars() {
    let h = sparrow_engine_server::handlers::sha256_lower_hex(b"");
    assert_eq!(
        h, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        "SHA-256 of empty bytes must be the canonical zero-length vector"
    );
    assert_eq!(h.len(), 64);
}
