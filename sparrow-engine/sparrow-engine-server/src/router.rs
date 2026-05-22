use std::time::Duration;

use axum::extract::DefaultBodyLimit;
use axum::error_handling::HandleErrorLayer;
use axum::http::StatusCode;
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use serde_json::json;
use tower::ServiceBuilder;
use tower_http::cors::CorsLayer;
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;
use tracing::Level;

use crate::handlers;
use crate::state::AppState;

pub fn build_router(state: AppState) -> Router {
    let max_body = state.config.max_body_size;

    // Inference routes — longer timeouts
    let inference = with_timeout(
        Router::new()
            .route("/v1/detect", post(handlers::detect::detect))
            .route(
                "/v1/detect/batch",
                post(handlers::detect::detect_batch),
            )
            .route("/v1/classify", post(handlers::classify::classify))
            .route("/v1/pipeline", post(handlers::pipeline::pipeline))
            .route("/v1/audio/detect", post(handlers::audio::audio_detect)),
        state.config.request_timeout_secs,
    );

    // Management routes — shorter timeout
    let management = with_timeout(
        Router::new()
            .route("/v1/catalog", get(handlers::catalog::list_catalog))
            .route("/v1/models", get(handlers::models::list_models))
            .route("/v1/models/load", post(handlers::models::load_model))
            .route("/v1/models/{id}", delete(handlers::models::unload_model))
            .route(
                "/v1/pipelines",
                get(handlers::pipelines_mgmt::list_pipelines)
                    .post(handlers::pipelines_mgmt::create_pipeline),
            )
            .route(
                "/v1/pipelines/load",
                post(handlers::pipelines::load_pipeline),
            )
            .route(
                "/v1/pipelines/{id}",
                delete(handlers::pipelines_mgmt::delete_pipeline),
            ),
        30,
    );

    // Health routes — fast timeout
    let health = with_timeout(
        Router::new()
            .route("/v1/health", get(handlers::health::health))
            .route("/healthz", get(handlers::health::liveness)),
        5,
    );

    Router::new()
        .merge(inference)
        .merge(management)
        .merge(health)
        // Request ID + tracing: SetRequestId (outer) → Trace → PropagateRequestId (inner).
        // axum applies layers bottom-to-top, so list inner layers first.
        .layer(PropagateRequestIdLayer::x_request_id())
        .layer(
            TraceLayer::new_for_http().make_span_with(|req: &axum::http::Request<_>| {
                let request_id = req
                    .headers()
                    .get("x-request-id")
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("-");
                tracing::span!(
                    Level::INFO,
                    "request",
                    method = %req.method(),
                    uri = %req.uri(),
                    request_id = %request_id,
                )
            }),
        )
        .layer(SetRequestIdLayer::x_request_id(MakeRequestUuid))
        .layer(DefaultBodyLimit::max(max_body))
        .layer(CorsLayer::permissive())
        .with_state(state)
}

/// Return a JSON 408 response matching the `{"error":{...}}` contract.
async fn handle_timeout(_err: tower::BoxError) -> (StatusCode, Json<serde_json::Value>) {
    (
        StatusCode::REQUEST_TIMEOUT,
        Json(json!({
            "error": {
                "code": "REQUEST_TIMEOUT",
                "message": "Request timed out",
                "status": 408
            }
        })),
    )
}

/// Attach the shared timeout-layer stack (HandleErrorLayer + timeout) to a
/// Router. Dedupes the 3 identical ServiceBuilder invocations that previously
/// lived inline in inference / management / health groups. Tower applies
/// layers bottom-to-top, so this call stacks the timeout INSIDE the
/// HandleErrorLayer envelope — identical to the pre-extraction order.
fn with_timeout(router: Router<AppState>, secs: u64) -> Router<AppState> {
    router.layer(
        ServiceBuilder::new()
            .layer(HandleErrorLayer::new(handle_timeout))
            .timeout(Duration::from_secs(secs)),
    )
}
