pub mod audio;
pub mod catalog;
pub mod classify;
pub mod detect;
pub mod health;
pub mod models;
pub mod pipeline;
pub mod pipelines;
pub mod pipelines_mgmt;

use std::sync::Arc;

use axum::extract::Multipart;
use tokio::sync::{OwnedSemaphorePermit, Semaphore};

use crate::engine_dispatch::{SparrowEngineError, DriftMetrics, InferenceLogRecord, ProvenanceRecord, SCHEMA_VERSION};
use crate::error::AppError;
use crate::state::AppState;

/// Extract a single named field from multipart. Returns 400 if missing.
pub async fn extract_field(multipart: &mut Multipart, name: &str) -> Result<Vec<u8>, AppError> {
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(e.to_string()))?
    {
        if field.name() == Some(name) {
            let bytes = field
                .bytes()
                .await
                .map_err(|e| AppError::bad_request(e.to_string()))?;
            if bytes.is_empty() {
                return Err(AppError::bad_request(format!(
                    "Field '{name}' must not be empty"
                )));
            }
            return Ok(bytes.to_vec());
        }
    }
    Err(AppError::bad_request(format!(
        "Missing required field '{name}'"
    )))
}

/// Validate that an ID is a simple name (no path traversal or absolute paths).
pub fn validate_id(id: &str, field: &str) -> Result<(), AppError> {
    if id.is_empty() || id.contains("..") || id.contains('/') || id.contains('\\') {
        return Err(AppError::bad_request(format!(
            "{field} must not be empty or contain path separators"
        )));
    }
    Ok(())
}

/// Try to acquire an inference permit, returning 503 if the queue is full.
pub fn acquire_inference_permit(
    semaphore: &Arc<Semaphore>,
) -> Result<OwnedSemaphorePermit, AppError> {
    Arc::clone(semaphore)
        .try_acquire_owned()
        .map_err(|_| AppError::service_unavailable("inference queue full"))
}

/// Validate that an optional max_detections is not zero.
pub fn validate_max_detections(max_detections: Option<u32>) -> Result<(), AppError> {
    if max_detections == Some(0) {
        return Err(AppError::bad_request("max_detections must be >= 1"));
    }
    Ok(())
}

/// Validate that an optional confidence threshold is in [0.0, 1.0] and not NaN.
pub fn validate_threshold(threshold: Option<f32>) -> Result<(), AppError> {
    if let Some(t) = threshold {
        if t.is_nan() || !(0.0..=1.0).contains(&t) {
            return Err(AppError::bad_request(
                "threshold must be between 0.0 and 1.0",
            ));
        }
    }
    Ok(())
}

/// Run a blocking closure on the tokio blocking pool, mapping errors to `AppError`.
pub async fn run_blocking<T, F>(f: F) -> Result<T, AppError>
where
    T: Send + 'static,
    F: FnOnce() -> Result<T, SparrowEngineError> + Send + 'static,
{
    tokio::task::spawn_blocking(f)
        .await
        .map_err(|e| AppError::internal(e.to_string()))?
        .map_err(Into::into)
}

// -- Phase 4 W3 helpers ------------------------------------------------------

/// SHA-256 lowercase hex over `bytes`. Used as `media_hash` in the
/// inference-log record.
pub fn sha256_lower_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(bytes);
    let mut s = String::with_capacity(digest.len() * 2);
    for b in digest {
        use std::fmt::Write;
        let _ = write!(s, "{b:02x}");
    }
    s
}

/// Emit an inference-log record. When the sink errors:
/// - `halt_on_failure = true`  → returns `AppError::internal` (HTTP 500).
/// - `halt_on_failure = false` → logs `tracing::warn!` and returns `Ok`.
pub fn emit_log_record(
    state: &AppState,
    record: &InferenceLogRecord,
    halt_on_failure: bool,
) -> Result<(), AppError> {
    match state.log_sink.emit(record) {
        Ok(()) => Ok(()),
        Err(e) if halt_on_failure => {
            Err(AppError::internal(format!("inference log sink failed: {e}")))
        }
        Err(e) => {
            tracing::warn!(
                error = %e,
                "inference log sink failed; continuing because halt_on_store_failure=false"
            );
            Ok(())
        }
    }
}

/// Build an `InferenceLogRecord` with the sparrow-engine-server-supplied fields
/// (request_id, timestamp_utc, device) populated. Per-handler caller fills
/// in media_hash / model_id / response payload / drift snapshot / provenance.
///
/// `provenance` is read from the active manifest's `[provenance]` section
/// (`ModelManifest.provenance`); pass `None` when the manifest omits it.
pub fn build_log_record(
    state: &AppState,
    media_hash: String,
    model_id: String,
    response_value: serde_json::Value,
    inference_ms: f64,
    drift: DriftMetrics,
    provenance: Option<ProvenanceRecord>,
) -> InferenceLogRecord {
    InferenceLogRecord {
        schema_version: SCHEMA_VERSION.to_string(),
        request_id: uuid::Uuid::new_v4().to_string(),
        timestamp_utc: chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
        media_hash,
        model_id,
        model_version: None,
        device: state.engine.active_device().to_string(),
        inference_ms,
        result: response_value,
        provenance,
        drift_metrics: Some(drift),
    }
}
