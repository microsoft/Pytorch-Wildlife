//! Detection HTTP handlers: single image and batch detection.
//!
//! - `POST /v1/detect` — single image, dispatches to tiled/single via manifest
//! - `POST /v1/detect/batch` — multiple images in one request

use axum::extract::multipart::MultipartRejection;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Multipart, Query, State};
use axum::Json;
use serde::Deserialize;

use crate::engine_dispatch::{detect, DetectOpts, ImageInput};
use crate::error::AppError;
use crate::response::{
    BatchDetectResponse, BatchDetectResultItem, DetectResponse, DetectionResponse,
};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Query parameters
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
pub struct DetectParams {
    pub model: String,
    pub threshold: Option<f32>,
    pub max_detections: Option<u32>,
    /// Phase 4 W3 — when true, emit an `InferenceLogRecord` after success.
    #[serde(default)]
    pub store: bool,
    /// Phase 4 W3 — when true, sink errors return HTTP 500. Default warn-only.
    #[serde(default)]
    pub halt_on_store_failure: bool,
}

#[derive(Deserialize)]
pub struct BatchDetectParams {
    pub model: String,
    pub threshold: Option<f32>,
    pub max_detections: Option<u32>,
    pub batch_size: Option<u32>,
    /// Phase 4 W3 — when true, emit one `InferenceLogRecord` for the whole
    /// batch (NOT one per image). Operator-visible consequence: the record's
    /// `media_hash` is the SHA-256 of the FIRST image only (per
    /// `docs/design/phase4/schema.md` §4 batch-section, chosen to avoid a
    /// concatenation buffer). Two batches that share the same first image
    /// will collide on the storage `(media_hash, model_id)` UNIQUE key and
    /// silently dedupe — operators relying on per-image idempotency must
    /// either disable batch dedup at the storage layer or call
    /// `/v1/detect` per image. Per-image detail still lives inside
    /// `record.result.results[i]`.
    #[serde(default)]
    pub store: bool,
    #[serde(default)]
    pub halt_on_store_failure: bool,
}

// ---------------------------------------------------------------------------
// POST /v1/detect
// ---------------------------------------------------------------------------

pub async fn detect(
    State(state): State<AppState>,
    query: Result<Query<DetectParams>, QueryRejection>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<DetectResponse>, AppError> {
    let Query(params) = query.map_err(|e| AppError::bad_request(format!("invalid query: {e}")))?;
    let mut multipart = multipart.map_err(|e| AppError::bad_request(e.body_text()))?;
    let image_bytes = super::extract_field(&mut multipart, "image").await?;
    super::validate_threshold(params.threshold)?;
    super::validate_max_detections(params.max_detections)?;

    let permit = super::acquire_inference_permit(&state.inference_semaphore)?;

    let model_id = params.model.clone();
    let opts = DetectOpts {
        confidence_threshold: params.threshold,
        max_detections: params.max_detections,
    };

    // Phase 4 W3 — hash the image bytes before they move into run_blocking.
    let media_hash = params.store.then(|| super::sha256_lower_hex(&image_bytes));

    let engine = std::sync::Arc::clone(&state.engine);
    let model_id_for_load = params.model.clone();
    let want_manifest_meta = params.store;
    let (result, drift_reference, provenance) = super::run_blocking(move || {
        let _permit = permit;
        // Phase 4.2 lazy-load: resolve (or load on demand) inside the blocking
        // pool so the async runtime stays responsive. Matches pipeline.rs.
        let handle = engine.get_or_load_model(&model_id_for_load)?;
        let image = ImageInput::Encoded(image_bytes);
        let result = detect::detect(&handle, &image, &opts)?;
        let (drift_ref, prov) = if want_manifest_meta {
            (
                handle.manifest().drift_reference.clone(),
                handle.manifest().provenance.clone(),
            )
        } else {
            (None, None)
        };
        Ok((result, drift_ref, prov))
    })
    .await?;

    let response = DetectResponse {
        model_id: model_id.clone(),
        image_size: [result.image_width, result.image_height],
        processing_time_ms: result.processing_time_ms,
        detections: result
            .detections
            .into_iter()
            .map(DetectionResponse::from)
            .collect(),
    };

    if params.store {
        let confidences: Vec<f32> = response.detections.iter().map(|d| d.confidence).collect();
        let labels: Vec<String> = response
            .detections
            .iter()
            .map(|d| d.label.clone())
            .collect();
        let drift = crate::drift::compute_drift_metrics(
            &confidences,
            1,
            &labels,
            drift_reference.as_ref(),
        );
        let value = serde_json::to_value(&response).unwrap_or(serde_json::Value::Null);
        let record = super::build_log_record(
            &state,
            media_hash.ok_or_else(|| AppError::internal("media hash missing when store=true"))?,
            model_id,
            value,
            response.processing_time_ms as f64,
            drift,
            provenance,
        );
        super::emit_log_record(&state, &record, params.halt_on_store_failure)?;
    }

    Ok(Json(response))
}

// ---------------------------------------------------------------------------
// POST /v1/detect/batch
// ---------------------------------------------------------------------------

pub async fn detect_batch(
    State(state): State<AppState>,
    query: Result<Query<BatchDetectParams>, QueryRejection>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<BatchDetectResponse>, AppError> {
    let Query(params) = query.map_err(|e| AppError::bad_request(format!("invalid query: {e}")))?;
    let mut multipart = multipart.map_err(|e| AppError::bad_request(e.body_text()))?;
    let images_bytes = extract_images_fields(&mut multipart).await?;
    super::validate_threshold(params.threshold)?;
    super::validate_max_detections(params.max_detections)?;
    if images_bytes.is_empty() {
        return Err(AppError::bad_request(
            "No images provided in 'images' fields",
        ));
    }
    if images_bytes.len() > state.config.max_batch_size {
        return Err(AppError::payload_too_large(format!(
            "Batch size {} exceeds maximum {}",
            images_bytes.len(),
            state.config.max_batch_size,
        )));
    }

    let permit = super::acquire_inference_permit(&state.inference_semaphore)?;

    let model_id = params.model.clone();
    let batch_size = params
        .batch_size
        .unwrap_or(4)
        .min(state.config.max_batch_size as u32) as usize;
    let opts = DetectOpts {
        confidence_threshold: params.threshold,
        max_detections: params.max_detections,
    };
    let count = images_bytes.len();
    let start = std::time::Instant::now();

    // Phase 4 W3 — for batch we hash the FIRST image only (per
    // schema.md: cheaper than concatenating; per-image detail lives
    // inside `result.results[i]`).
    let media_hash = params.store.then(|| {
        images_bytes
            .first()
            .map(|b| super::sha256_lower_hex(b))
            .unwrap_or_default()
    });

    let engine = std::sync::Arc::clone(&state.engine);
    let model_id_for_load = params.model.clone();
    let want_manifest_meta = params.store;
    let (results, drift_reference, provenance) = super::run_blocking(move || {
        let _permit = permit;
        // Phase 4.2 lazy-load: resolve (or load on demand) inside the blocking
        // pool so the async runtime stays responsive.
        let handle = engine.get_or_load_model(&model_id_for_load)?;
        let images: Vec<ImageInput> = images_bytes.into_iter().map(ImageInput::Encoded).collect();
        let results = detect::detect_batch(&handle, &images, &opts, batch_size, None)?;
        let (drift_ref, prov) = if want_manifest_meta {
            (
                handle.manifest().drift_reference.clone(),
                handle.manifest().provenance.clone(),
            )
        } else {
            (None, None)
        };
        Ok((results, drift_ref, prov))
    })
    .await?;

    let processing_time_ms = start.elapsed().as_secs_f32() * 1000.0;

    let batch_results: Vec<BatchDetectResultItem> = results
        .into_iter()
        .enumerate()
        .map(|(i, r)| BatchDetectResultItem {
            index: i,
            image_size: [r.image_width, r.image_height],
            detections: r
                .detections
                .into_iter()
                .map(DetectionResponse::from)
                .collect(),
        })
        .collect();

    let response = BatchDetectResponse {
        model_id: model_id.clone(),
        count,
        processing_time_ms,
        results: batch_results,
    };

    if params.store {
        // Flatten confidences + labels across every image in the batch.
        let mut confidences: Vec<f32> = Vec::new();
        let mut labels: Vec<String> = Vec::new();
        for r in &response.results {
            for d in &r.detections {
                confidences.push(d.confidence);
                labels.push(d.label.clone());
            }
        }
        let drift = crate::drift::compute_drift_metrics(
            &confidences,
            count,
            &labels,
            drift_reference.as_ref(),
        );
        let value = serde_json::to_value(&response).unwrap_or(serde_json::Value::Null);
        let record = super::build_log_record(
            &state,
            media_hash.ok_or_else(|| AppError::internal("media hash missing when store=true"))?,
            model_id,
            value,
            response.processing_time_ms as f64,
            drift,
            provenance,
        );
        super::emit_log_record(&state, &record, params.halt_on_store_failure)?;
    }

    Ok(Json(response))
}

// ---------------------------------------------------------------------------
// Multipart helpers
// ---------------------------------------------------------------------------

/// Extract all `images` fields from multipart (for batch endpoint).
async fn extract_images_fields(multipart: &mut Multipart) -> Result<Vec<Vec<u8>>, AppError> {
    let mut images = Vec::new();
    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|e| AppError::bad_request(e.to_string()))?
    {
        if field.name() == Some("images") {
            let bytes = field
                .bytes()
                .await
                .map_err(|e| AppError::bad_request(e.to_string()))?;
            if bytes.is_empty() {
                return Err(AppError::bad_request(
                    "Each 'images' field must not be empty",
                ));
            }
            images.push(bytes.to_vec());
        }
    }
    Ok(images)
}
