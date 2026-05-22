//! Pluggable inference-log emission (Phase 4 W3).
//!
//! Bongo defines the wire format (`sparrow_engine_types::InferenceLogRecord`); this
//! module defines how records leave the process. The default sink writes
//! one JSON line per record to stderr so deployments without a `sparrow-data`
//! sibling still observe the data flow. Custom sinks (filesystem, HTTP
//! POST to sparrow-data once it exists) plug in via `Arc<dyn InferenceLogSink>`
//! on `AppState`.
//!
//! Idempotency note: implementations should treat `(media_hash, model_id)`
//! as a UNIQUE constraint and silently drop duplicates. The default stderr
//! sink does NOT enforce this — uniqueness is a backend property, not a
//! wire-format property. The v3 `UNIQUE(media_hash, model_id)` decision
//! lives at the storage layer (sparrow-data sibling), not in sparrow-engine.

use std::io::Write;

use crate::engine_dispatch::InferenceLogRecord;

/// A single error string from a sink. Kept dependency-free
/// (no `thiserror`, no `anyhow`) so sinks compile in minimal
/// downstream consumers.
#[derive(Debug)]
pub struct SinkError(pub String);

impl std::fmt::Display for SinkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "inference-log sink error: {}", self.0)
    }
}

impl std::error::Error for SinkError {}

/// Strategy for emitting inference-log records out of sparrow-engine-server.
///
/// Implementors must be `Send + Sync` so an `Arc<dyn InferenceLogSink>` can
/// be cloned into `tokio::task::spawn_blocking` closures from the request
/// handlers.
///
/// `emit` is sync. The default `StderrJsonLinesSink` runs in microseconds and
/// is called inline on the tokio reactor thread without blocking risk. Future
/// HTTP / network sinks (e.g., sparrow-data ingest) MUST internally wrap their
/// network call in `tokio::task::spawn_blocking` (or upgrade this trait to
/// `async fn emit`) — calling `reqwest::blocking` directly from `emit` would
/// stall the reactor.
pub trait InferenceLogSink: Send + Sync {
    fn emit(&self, record: &InferenceLogRecord) -> Result<(), SinkError>;
}

/// Default sink — writes one JSON line per record to stderr. Lock-stderr
/// keeps lines from interleaving with each other or with `tracing` output.
pub struct StderrJsonLinesSink;

impl InferenceLogSink for StderrJsonLinesSink {
    fn emit(&self, record: &InferenceLogRecord) -> Result<(), SinkError> {
        let line = serde_json::to_string(record).map_err(|e| SinkError(e.to_string()))?;
        let mut stderr = std::io::stderr().lock();
        writeln!(stderr, "{line}").map_err(|e| SinkError(e.to_string()))?;
        Ok(())
    }
}
