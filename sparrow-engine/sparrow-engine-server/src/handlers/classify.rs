//! Handler for POST /v1/classify — single image classification.

use axum::extract::multipart::MultipartRejection;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Multipart, Query, State};
use axum::Json;
use serde::Deserialize;

use crate::engine_dispatch::{classify, ClassifyOpts, ImageInput};
use crate::error::AppError;
use crate::response::{ClassificationResponse, ClassifyResponse};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct ClassifyParams {
    pub model: String,
    pub top_k: Option<u32>,
    #[serde(default)]
    pub store: bool,
    #[serde(default)]
    pub halt_on_store_failure: bool,
}

pub async fn classify(
    State(state): State<AppState>,
    query: Result<Query<ClassifyParams>, QueryRejection>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<ClassifyResponse>, AppError> {
    let Query(params) = query.map_err(|e| AppError::bad_request(format!("invalid query: {e}")))?;
    let mut multipart = multipart.map_err(|e| AppError::bad_request(e.body_text()))?;
    let image_bytes = super::extract_field(&mut multipart, "image").await?;
    if params.top_k == Some(0) {
        return Err(AppError::bad_request("top_k must be >= 1"));
    }

    let permit = super::acquire_inference_permit(&state.inference_semaphore)?;

    let model_id = params.model.clone();
    let opts = ClassifyOpts {
        top_k: params.top_k.or(Some(5)),
    };

    // Phase 4 W3 — hash the image before it moves into run_blocking.
    let media_hash = params.store.then(|| super::sha256_lower_hex(&image_bytes));

    let engine = std::sync::Arc::clone(&state.engine);
    let model_id_for_load = params.model.clone();
    let want_manifest_meta = params.store;
    let (result, drift_reference, provenance) = super::run_blocking(move || {
        let _permit = permit;
        // Phase 4.2 lazy-load: resolve (or load on demand) inside the blocking
        // pool so the async runtime stays responsive.
        let handle = engine.get_or_load_model(&model_id_for_load)?;
        let image = ImageInput::Encoded(image_bytes);
        let result = classify::classify(&handle, &image, &opts)?;
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

    let response = ClassifyResponse {
        model_id: model_id.clone(),
        image_size: [result.image_width, result.image_height],
        processing_time_ms: result.processing_time_ms,
        classifications: result
            .classifications
            .into_iter()
            .map(ClassificationResponse::from)
            .collect(),
    };

    if params.store {
        let confidences: Vec<f32> = response
            .classifications
            .iter()
            .map(|c| c.confidence)
            .collect();
        let labels: Vec<String> = response
            .classifications
            .iter()
            .map(|c| c.label.clone())
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
