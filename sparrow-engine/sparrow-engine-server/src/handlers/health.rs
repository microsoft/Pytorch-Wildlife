use axum::extract::State;
use axum::Json;
use serde_json::{json, Value};

use crate::response::HealthResponse;
use crate::state::AppState;

pub async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    let models_loaded = state.engine.loaded_models().len();
    let pipelines_loaded = state.engine.loaded_pipelines().len();
    let catalog_size = state.catalog.models.len();

    // Phase 4.2: lazy boot is the default. The server is "ready" as soon as
    // its catalog is non-empty — sessions load on demand. "no_models" means
    // discovery found zero parseable manifests (operator config error).
    let status = if catalog_size > 0 {
        "ready"
    } else {
        "no_models"
    };

    Json(HealthResponse {
        status: status.to_string(),
        models_loaded,
        pipelines_loaded,
        catalog_size,
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

pub async fn liveness() -> Json<Value> {
    Json(json!({"alive": true}))
}
