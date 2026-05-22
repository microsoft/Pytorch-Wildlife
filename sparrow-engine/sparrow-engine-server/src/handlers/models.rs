//! Model management handlers: list, load, unload.

use axum::extract::rejection::JsonRejection;
use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::engine_dispatch::{ModelInfo, ModelType};
use crate::error::AppError;
use crate::state::AppState;

// -- Request / response types ------------------------------------------------

#[derive(Deserialize)]
pub struct LoadModelRequest {
    pub model_id: String,
}

#[derive(Serialize)]
pub struct ModelResponse {
    pub id: String,
    pub model_type: String,
    /// Whether this model is the default for its type (manifest `default = true`).
    pub default: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onnx_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub onnx_size_bytes: Option<u64>,
}

impl From<ModelInfo> for ModelResponse {
    fn from(m: ModelInfo) -> Self {
        Self {
            id: m.id,
            model_type: model_type_str(m.model_type).to_string(),
            default: m.default,
            version: m.version,
            description: m.description,
            onnx_sha256: m.onnx_sha256,
            onnx_size_bytes: m.onnx_size_bytes,
        }
    }
}

#[derive(Serialize)]
pub struct ModelsListResponse {
    pub models: Vec<ModelResponse>,
}

fn model_type_str(mt: ModelType) -> &'static str {
    match mt {
        ModelType::Detector => "detector",
        ModelType::OverheadDetector => "overhead_detector",
        ModelType::Classifier => "classifier",
        ModelType::AudioDetector => "audio_detector",
        ModelType::AudioClassifier => "audio_classifier",
    }
}

// -- Handlers ----------------------------------------------------------------

/// GET /v1/models
pub async fn list_models(State(state): State<AppState>) -> Json<ModelsListResponse> {
    let models = state
        .engine
        .loaded_models()
        .into_iter()
        .map(ModelResponse::from)
        .collect();
    Json(ModelsListResponse { models })
}

/// POST /v1/models/load — load a model by ID (idempotent, 200 on reload).
pub async fn load_model(
    State(state): State<AppState>,
    body: Result<Json<LoadModelRequest>, JsonRejection>,
) -> Result<Json<ModelResponse>, AppError> {
    let Json(req) = body.map_err(AppError::from_json_rejection)?;
    let engine = state.engine.clone();
    let model_id = req.model_id;
    super::validate_id(&model_id, "model_id")?;

    let handle = super::run_blocking({
        let mid = model_id.clone();
        move || engine.get_or_load_model(&mid)
    })
    .await?;

    // After load, look up the full ModelInfo so the response surfaces
    // Phase 3 fields (default, version, description, onnx_sha256,
    // onnx_size_bytes). Fall back to a minimal response if lookup misses
    // (the handle itself is sufficient for the caller to proceed).
    let info = state
        .engine
        .loaded_models()
        .into_iter()
        .find(|m| m.id == handle.model_id());
    let response = match info {
        Some(info) => ModelResponse::from(info),
        None => ModelResponse {
            id: handle.model_id().to_string(),
            model_type: model_type_str(handle.model_type()).to_string(),
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        },
    };
    Ok(Json(response))
}

/// DELETE /v1/models/{model_id} — 204 on success, 404 if not loaded, 410 if already unloaded.
pub async fn unload_model(
    State(state): State<AppState>,
    Path(model_id): Path<String>,
) -> Result<StatusCode, AppError> {
    // Non-loading lookup: unloading a model that isn't loaded is a 404 by
    // design (lazy-load on the inference path doesn't apply to unload).
    let handle = state
        .engine
        .get_model_handle(&model_id)
        .ok_or_else(|| AppError::model_not_loaded(&model_id))?;
    let engine = state.engine.clone();
    super::run_blocking(move || engine.unload_model(&handle)).await?;
    Ok(StatusCode::NO_CONTENT)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    // Regression (SRV1): ModelResponse must surface the Phase 3 fields so HTTP
    // clients can read version / description / onnx_sha256 / onnx_size_bytes /
    // default. All four of the last four serialize only when Some (keeps the
    // legacy wire format intact for manifests without these fields).
    #[test]
    fn model_response_serializes_phase3_fields_when_present() {
        let info = ModelInfo {
            id: "megadetector-v6".to_string(),
            path: PathBuf::from("/models/megadetector-v6/manifest.toml"),
            model_type: ModelType::Detector,
            default: true,
            version: Some("6.1.2".to_string()),
            description: Some("MegaDetector v6".to_string()),
            onnx_sha256: Some("abc123".to_string()),
            onnx_size_bytes: Some(104857600),
        };
        let resp = ModelResponse::from(info);
        let json = serde_json::to_value(&resp).unwrap();

        assert_eq!(json["id"], "megadetector-v6");
        assert_eq!(json["model_type"], "detector");
        assert_eq!(json["default"], true);
        assert_eq!(json["version"], "6.1.2");
        assert_eq!(json["description"], "MegaDetector v6");
        assert_eq!(json["onnx_sha256"], "abc123");
        assert_eq!(json["onnx_size_bytes"], 104857600);
    }

    #[test]
    fn model_response_skips_missing_optional_fields() {
        let info = ModelInfo {
            id: "legacy".to_string(),
            path: PathBuf::from("/models/legacy/manifest.toml"),
            model_type: ModelType::Classifier,
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        };
        let resp = ModelResponse::from(info);
        let json = serde_json::to_value(&resp).unwrap();

        // Required fields always present.
        assert_eq!(json["id"], "legacy");
        assert_eq!(json["model_type"], "classifier");
        assert_eq!(json["default"], false);

        // Optional fields must be absent (not null) — skip_serializing_if.
        let obj = json.as_object().unwrap();
        assert!(!obj.contains_key("version"));
        assert!(!obj.contains_key("description"));
        assert!(!obj.contains_key("onnx_sha256"));
        assert!(!obj.contains_key("onnx_size_bytes"));
    }
}
