//! Pipeline management handlers: list, load, unload.

use axum::extract::rejection::JsonRejection;
use axum::extract::{Json, Path, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::engine_dispatch::manifest::{self, PipelineManifest, PipelineRole};
use crate::error::AppError;
use crate::state::AppState;

// -- Request / response types ------------------------------------------------

#[derive(Deserialize)]
pub struct LoadPipelineRequest {
    pub pipeline_id: String,
}

#[derive(Serialize)]
pub struct PipelineStepResponse {
    pub role: String,
    pub model_id: String,
}

#[derive(Serialize)]
pub struct PipelineInfoResponse {
    pub id: String,
    pub steps: Vec<PipelineStepResponse>,
}

#[derive(Serialize)]
pub struct PipelinesListResponse {
    pub pipelines: Vec<PipelineInfoResponse>,
}

fn validate_manifest_against_catalog(
    state: &AppState,
    manifest: &PipelineManifest,
) -> Result<(), AppError> {
    let mut missing = Vec::<String>::new();
    let mut detector_type = None;
    let mut classifier_types = Vec::new();

    for step in &manifest.steps {
        match state.catalog.models.get(&step.model) {
            Some(info) => match step.role {
                PipelineRole::Detector => detector_type = Some(info.model_type),
                PipelineRole::Classifier => classifier_types.push(info.model_type),
            },
            None if !missing.contains(&step.model) => missing.push(step.model.clone()),
            None => {}
        }
    }

    if let Some(model_id) = missing.first() {
        return Err(super::pipelines_mgmt::model_not_in_catalog(model_id));
    }

    if classifier_types.is_empty() {
        crate::engine_dispatch::pipeline_compat::validate_pipeline_compat(detector_type, None)?;
    } else {
        // Validate EVERY classifier step. The current CPU/GPU runtime only executes
        // classifier_model_ids[0], but skipped steps must still match the documented
        // pipeline contract — otherwise explicit AudioClassifier rejection is bypassable
        // by trailing it with a compatible classifier step.
        for classifier_type in classifier_types {
            crate::engine_dispatch::pipeline_compat::validate_pipeline_compat(
                detector_type,
                Some(classifier_type),
            )?;
        }
    }
    Ok(())
}

fn manifest_to_response(p: &PipelineManifest) -> PipelineInfoResponse {
    PipelineInfoResponse {
        id: p.id.clone(),
        steps: p
            .steps
            .iter()
            .map(|s| PipelineStepResponse {
                role: match s.role {
                    PipelineRole::Detector => "detector".to_string(),
                    PipelineRole::Classifier => "classifier".to_string(),
                },
                model_id: s.model.clone(),
            })
            .collect(),
    }
}

// -- Handlers ----------------------------------------------------------------

/// GET /v1/pipelines
pub async fn list_pipelines(State(state): State<AppState>) -> Json<PipelinesListResponse> {
    let pipelines = state
        .engine
        .loaded_pipelines()
        .iter()
        .map(manifest_to_response)
        .collect();
    Json(PipelinesListResponse { pipelines })
}

/// POST /v1/pipelines/load — load a pipeline by ID (idempotent, 200 on reload).
pub async fn load_pipeline(
    State(state): State<AppState>,
    body: Result<Json<LoadPipelineRequest>, JsonRejection>,
) -> Result<Json<PipelineInfoResponse>, AppError> {
    let Json(req) = body.map_err(AppError::from_json_rejection)?;
    let id = req.pipeline_id;
    super::pipelines_mgmt::validate_alias_id(&id)?;

    let lock = super::pipelines_mgmt::alias_lock(&state, &id)?;
    let lock_for_prune = lock.clone();
    let state_for_load = state.clone();
    let id_for_load = id.clone();
    let blocking_result = tokio::task::spawn_blocking(move || -> Result<PipelineManifest, AppError> {
        let _guard = lock
            .lock()
            .map_err(|_| AppError::internal("pipeline alias lock poisoned"))?;
        let pipeline_path = state_for_load
            .config
            .model_dir
            .join(&id_for_load)
            .join("pipeline.toml");
        if pipeline_path
            .parent()
            .map(|dir| dir.join("manifest.toml").is_file())
            .unwrap_or(false)
        {
            return Err(AppError::Http {
                status: StatusCode::CONFLICT,
                code: "PIPELINE_MODEL_ID_COLLISION".to_string(),
                message: "pipeline alias collides with an existing model manifest directory"
                    .to_string(),
            });
        }
        let manifest = manifest::load_pipeline_manifest(&pipeline_path)?;
        if manifest.id != id_for_load {
            return Err(AppError::bad_request(format!(
                "pipeline manifest id '{}' does not match requested pipeline_id '{}'",
                manifest.id, id_for_load
            )));
        }
        validate_manifest_against_catalog(&state_for_load, &manifest)?;
        state_for_load
            .engine
            .register_pipeline_manifest(manifest.clone())?;
        Ok(manifest)
    })
    .await
    .map_err(|e| AppError::internal(e.to_string()))?;
    if blocking_result.is_err() {
        super::pipelines_mgmt::prune_alias_lock_if_unused(&state, &id, &lock_for_prune);
    }
    let manifest = blocking_result?;

    Ok(Json(manifest_to_response(&manifest)))
}

/// DELETE /v1/pipelines/{pipeline_id} — 204 on success, 404 if not found.
pub async fn unload_pipeline(
    State(state): State<AppState>,
    Path(pipeline_id): Path<String>,
) -> Result<StatusCode, AppError> {
    state.engine.unload_pipeline(&pipeline_id)?;
    Ok(StatusCode::NO_CONTENT)
}
