//! Pipeline alias management handlers.

use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use axum::extract::rejection::JsonRejection;
use axum::extract::{Json, Path as AxumPath, State};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};

use crate::engine_dispatch::manifest::{PipelineManifest, PipelineRole, PipelineStep};
use crate::error::AppError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct CreatePipelineRequest {
    pub id: String,
    pub detector: String,
    pub classifier: String,
    #[serde(default)]
    pub replace: bool,
    #[serde(default)]
    pub persist: bool,
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

pub async fn list_pipelines(State(state): State<AppState>) -> Json<PipelinesListResponse> {
    let pipelines = state
        .engine
        .loaded_pipelines()
        .iter()
        .map(manifest_to_response)
        .collect();
    Json(PipelinesListResponse { pipelines })
}

pub async fn create_pipeline(
    State(state): State<AppState>,
    body: Result<Json<CreatePipelineRequest>, JsonRejection>,
) -> Result<(StatusCode, Json<PipelineInfoResponse>), AppError> {
    let Json(req) = body.map_err(AppError::from_json_rejection)?;
    validate_alias_id(&req.id)?;
    super::validate_id(&req.detector, "detector")?;
    super::validate_id(&req.classifier, "classifier")?;

    let detector_type = state
        .catalog
        .models
        .get(&req.detector)
        .map(|m| m.model_type)
        .ok_or_else(|| model_not_in_catalog(&req.detector))?;
    let classifier_type = state
        .catalog
        .models
        .get(&req.classifier)
        .map(|m| m.model_type)
        .ok_or_else(|| model_not_in_catalog(&req.classifier))?;
    crate::engine_dispatch::pipeline_compat::validate_pipeline_compat(
        Some(detector_type),
        Some(classifier_type),
    )?;

    let lock = alias_lock(&state, &req.id)?;
    let state_for_prune = state.clone();
    let id_for_prune = req.id.clone();
    let lock_for_prune = lock.clone();
    let state_for_blocking = state.clone();
    let blocking_result = tokio::task::spawn_blocking(move || {
        let new_manifest = PipelineManifest {
            id: req.id.clone(),
            steps: vec![
                PipelineStep {
                    role: PipelineRole::Detector,
                    model: req.detector.clone(),
                },
                PipelineStep {
                    role: PipelineRole::Classifier,
                    model: req.classifier.clone(),
                },
            ],
        };

        let _guard = lock
            .lock()
            .map_err(|_| AppError::internal("pipeline alias lock poisoned"))?;
        let existed = match state_for_blocking.engine.get_pipeline(&req.id) {
            Ok(existing) if same_definition(&existing, &new_manifest) => {
                if req.persist {
                    persist_pipeline(&state_for_blocking, &new_manifest)?;
                }
                return Ok((StatusCode::OK, manifest_to_response(&existing)));
            }
            Ok(_) if !req.replace => {
                return Err(AppError::Http {
                    status: StatusCode::CONFLICT,
                    code: "PIPELINE_ALIAS_CONFLICT".to_string(),
                    message: "pipeline alias exists with a different definition; set replace=true to overwrite".to_string(),
                });
            }
            Ok(_) => true,
            Err(crate::engine_dispatch::SparrowEngineError::PipelineNotFound { .. }) => false,
            Err(e) => return Err(e.into()),
        };
        if req.persist {
            persist_pipeline(&state_for_blocking, &new_manifest)?;
        }
        state_for_blocking
            .engine
            .register_pipeline_manifest(new_manifest.clone())?;
        Ok((status_for_create(existed), manifest_to_response(&new_manifest)))
    })
    .await
    .map_err(|e| AppError::internal(e.to_string()))?;
    if blocking_result.is_err() {
        prune_alias_lock_if_unused(&state_for_prune, &id_for_prune, &lock_for_prune);
    }
    let (status, response) = blocking_result?;

    Ok((status, Json(response)))
}

pub async fn delete_pipeline(
    State(state): State<AppState>,
    AxumPath(pipeline_id): AxumPath<String>,
) -> Result<StatusCode, AppError> {
    validate_alias_id(&pipeline_id)?;
    let lock = alias_lock(&state, &pipeline_id)?;
    let state_for_prune = state.clone();
    let id_for_prune = pipeline_id.clone();
    let lock_for_prune = lock.clone();
    let state_for_blocking = state.clone();
    let blocking_result = tokio::task::spawn_blocking(move || {
        let _guard = lock
            .lock()
            .map_err(|_| AppError::internal("pipeline alias lock poisoned"))?;
        state_for_blocking.engine.get_pipeline(&pipeline_id)?;
        let dir = state_for_blocking.config.model_dir.join(&pipeline_id);
        let pipeline_path = dir.join("pipeline.toml");
        match fs::remove_file(&pipeline_path) {
            Ok(()) => remove_dir_if_empty(&dir)?,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => return Err(AppError::internal(e.to_string())),
        }
        state_for_blocking.engine.unload_pipeline(&pipeline_id)?;
        Ok(StatusCode::NO_CONTENT)
    })
    .await
    .map_err(|e| AppError::internal(e.to_string()))?;
    if blocking_result.is_err() {
        prune_alias_lock_if_unused(&state_for_prune, &id_for_prune, &lock_for_prune);
    }
    blocking_result
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

pub(crate) fn validate_alias_id(id: &str) -> Result<(), AppError> {
    if id.is_empty()
        || id == "."
        || id == ".."
        || id.contains("..")
        || !id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(AppError::bad_request("invalid alias id"));
    }
    Ok(())
}

fn same_definition(a: &PipelineManifest, b: &PipelineManifest) -> bool {
    a.steps.len() == b.steps.len()
        && a.steps
            .iter()
            .zip(&b.steps)
            .all(|(a, b)| a.role == b.role && a.model == b.model)
}

fn status_for_create(existed: bool) -> StatusCode {
    if existed {
        StatusCode::OK
    } else {
        StatusCode::CREATED
    }
}

pub(crate) fn alias_lock(state: &AppState, id: &str) -> Result<Arc<Mutex<()>>, AppError> {
    let mut locks = state
        .pipeline_write_locks
        .lock()
        .map_err(|_| AppError::internal("pipeline write locks poisoned"))?;
    Ok(locks
        .entry(id.to_string())
        .or_insert_with(|| Arc::new(Mutex::new(())))
        .clone())
}

pub(crate) fn prune_alias_lock_if_unused(state: &AppState, id: &str, lock: &Arc<Mutex<()>>) {
    let Ok(mut locks) = state.pipeline_write_locks.lock() else {
        return;
    };
    let Some(existing) = locks.get(id) else {
        return;
    };
    if Arc::ptr_eq(existing, lock) && Arc::strong_count(lock) <= 2 {
        locks.remove(id);
    }
}

fn persist_pipeline(state: &AppState, manifest: &PipelineManifest) -> Result<(), AppError> {
    let dir = state.config.model_dir.join(&manifest.id);
    if dir.join("manifest.toml").exists() {
        return Err(AppError::Http {
            status: StatusCode::CONFLICT,
            code: "PIPELINE_MODEL_ID_COLLISION".to_string(),
            message: "pipeline alias collides with an existing model manifest directory"
                .to_string(),
        });
    }
    fs::create_dir_all(&dir).map_err(|e| AppError::internal(e.to_string()))?;
    let final_path = dir.join("pipeline.toml");
    let temp_path = dir.join(format!("pipeline.toml.next.{}", std::process::id()));
    fs::write(&temp_path, pipeline_toml(manifest))
        .map_err(|e| AppError::internal(e.to_string()))?;
    fs::rename(&temp_path, &final_path).map_err(|e| AppError::internal(e.to_string()))?;
    Ok(())
}

fn pipeline_toml(manifest: &PipelineManifest) -> String {
    let mut out = format!("[pipeline]\nid = {}\n", toml_basic_string(&manifest.id));
    for step in &manifest.steps {
        let role = match step.role {
            PipelineRole::Detector => "detector",
            PipelineRole::Classifier => "classifier",
        };
        out.push_str(&format!(
            "\n[[pipeline.steps]]\nrole = {}\nmodel = {}\n",
            toml_basic_string(role),
            toml_basic_string(&step.model)
        ));
    }
    out
}

fn toml_basic_string(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 2);
    escaped.push('"');
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\u{08}' => escaped.push_str("\\b"),
            '\t' => escaped.push_str("\\t"),
            '\n' => escaped.push_str("\\n"),
            '\u{0c}' => escaped.push_str("\\f"),
            '\r' => escaped.push_str("\\r"),
            c if c <= '\u{1f}' || c == '\u{7f}' => {
                use std::fmt::Write;
                let _ = write!(escaped, "\\u{:04X}", c as u32);
            }
            c => escaped.push(c),
        }
    }
    escaped.push('"');
    escaped
}

fn remove_dir_if_empty(dir: &Path) -> Result<(), AppError> {
    match fs::read_dir(dir) {
        Ok(mut entries) => {
            if entries.next().is_none() {
                fs::remove_dir(dir).map_err(|e| AppError::internal(e.to_string()))?;
            }
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(AppError::internal(e.to_string())),
    }
    Ok(())
}

pub(crate) fn model_not_in_catalog(model_id: &str) -> AppError {
    AppError::Http {
        status: StatusCode::NOT_FOUND,
        code: "MODEL_NOT_IN_CATALOG".to_string(),
        message: format!("Model '{model_id}' is not in the discovered catalog"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::{to_bytes, Body};
    use axum::http::{Method, Request};
    use serde_json::{json, Value};
    use std::collections::BTreeMap;
    use std::net::SocketAddr;
    use std::path::PathBuf;
    use tower::Service;

    use crate::config::{Config, LogFormat};
    use crate::discover::Catalog;
    use crate::engine_dispatch::{Device, Engine, EngineConfig, ModelInfo, ModelType};

    #[test]
    fn validate_alias_id_rejects_path_traversal_and_non_slug() {
        for id in ["", "..", "a/../b", "a/b", "a.b", "a b"] {
            assert!(validate_alias_id(id).is_err(), "expected invalid id: {id}");
        }
        for id in ["my-pipeline", "my_pipeline", "abc123"] {
            assert!(validate_alias_id(id).is_ok(), "expected valid id: {id}");
        }
    }

    #[test]
    fn same_definition_compares_steps() {
        let a = PipelineManifest {
            id: "p".to_string(),
            steps: vec![PipelineStep {
                role: PipelineRole::Detector,
                model: "d".to_string(),
            }],
        };
        let b = PipelineManifest {
            id: "p".to_string(),
            steps: vec![PipelineStep {
                role: PipelineRole::Detector,
                model: "d".to_string(),
            }],
        };
        assert!(same_definition(&a, &b));
    }

    #[test]
    fn status_for_create_ignores_replace_flag_and_depends_on_existing_alias() {
        assert_eq!(status_for_create(false), StatusCode::CREATED);
        assert_eq!(status_for_create(true), StatusCode::OK);
    }

    #[tokio::test]
    async fn pipeline_management_endpoints_preserve_validation_and_error_contracts() {
        let (mut app, model_dir, state) = test_router_with_model_dir("pipelines_mgmt_integration");

        let create_body = json!({
            "id": "wildlife-pipeline",
            "detector": "detector-a",
            "classifier": "classifier-a"
        });
        let expected_wildlife_manifest = PipelineManifest {
            id: "wildlife-pipeline".to_string(),
            steps: vec![
                PipelineStep {
                    role: PipelineRole::Detector,
                    model: "detector-a".to_string(),
                },
                PipelineStep {
                    role: PipelineRole::Classifier,
                    model: "classifier-a".to_string(),
                },
            ],
        };

        let response = request(
            &mut app,
            Method::POST,
            "/v1/pipelines",
            Some(create_body.clone()),
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::CREATED,
            "scenario 1: alias create"
        );
        let body = json_body(response).await;
        assert_eq!(body["id"], "wildlife-pipeline", "scenario 1: created id");
        assert_eq!(
            body["steps"][0]["model_id"], "detector-a",
            "scenario 1: detector step"
        );
        assert_eq!(
            body["steps"][1]["model_id"], "classifier-a",
            "scenario 1: classifier step"
        );

        let response = request(&mut app, Method::GET, "/v1/pipelines", None).await;
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "scenario 1: list after create"
        );
        let body = json_body(response).await;
        assert!(
            body["pipelines"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["id"] == "wildlife-pipeline"),
            "scenario 1: created alias appears in list"
        );

        let response = request(&mut app, Method::POST, "/v1/pipelines", Some(create_body)).await;
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "scenario 1: idempotent recreate"
        );
        let body = json_body(response).await;
        assert_eq!(body["id"], "wildlife-pipeline", "scenario 1: idempotent id");

        let persist_body = json!({
            "id": "wildlife-pipeline",
            "detector": "detector-a",
            "classifier": "classifier-a",
            "persist": true
        });
        let response = request(&mut app, Method::POST, "/v1/pipelines", Some(persist_body)).await;
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "scenario 1: idempotent persist keeps existing-status contract"
        );
        let persisted_path = model_dir.join("wildlife-pipeline").join("pipeline.toml");
        assert!(
            persisted_path.exists(),
            "scenario 1: idempotent persist writes pipeline.toml"
        );
        let persisted_manifest =
            crate::engine_dispatch::manifest::load_pipeline_manifest(&persisted_path)
                .expect("scenario 1: persisted pipeline manifest parses");
        assert_eq!(
            &persisted_manifest.id, &expected_wildlife_manifest.id,
            "scenario 1: persisted manifest id"
        );
        assert!(
            same_definition(&persisted_manifest, &expected_wildlife_manifest),
            "scenario 1: persisted manifest keeps the requested definition"
        );

        let conflict_body = json!({
            "id": "wildlife-pipeline",
            "detector": "detector-a",
            "classifier": "classifier-b"
        });
        let response = request(&mut app, Method::POST, "/v1/pipelines", Some(conflict_body)).await;
        assert_eq!(
            response.status(),
            StatusCode::CONFLICT,
            "scenario 1: conflicting alias"
        );
        let body = json_body(response).await;
        assert_eq!(
            body["error"]["code"], "PIPELINE_ALIAS_CONFLICT",
            "scenario 1: conflict error code"
        );

        let response = request(
            &mut app,
            Method::DELETE,
            "/v1/pipelines/wildlife-pipeline",
            None,
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::NO_CONTENT,
            "scenario 1: delete alias"
        );

        let response = request(&mut app, Method::GET, "/v1/pipelines", None).await;
        assert_eq!(
            response.status(),
            StatusCode::OK,
            "scenario 1: list after delete"
        );
        let body = json_body(response).await;
        assert!(
            !body["pipelines"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["id"] == "wildlife-pipeline"),
            "scenario 1: deleted alias absent from list"
        );

        let response = request(
            &mut app,
            Method::DELETE,
            "/v1/pipelines/missing-pipeline",
            None,
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "scenario 1: missing delete"
        );
        let body = json_body(response).await;
        assert_eq!(
            body["error"]["code"], "PIPELINE_NOT_FOUND",
            "scenario 1: missing delete error code"
        );

        write_pipeline_manifest(
            &model_dir,
            PipelineManifest {
                id: "legacy-mixed".to_string(),
                steps: vec![
                    PipelineStep {
                        role: PipelineRole::Detector,
                        model: "audio-detector".to_string(),
                    },
                    PipelineStep {
                        role: PipelineRole::Classifier,
                        model: "classifier-a".to_string(),
                    },
                ],
            },
        );

        let response = request(
            &mut app,
            Method::POST,
            "/v1/pipelines/load",
            Some(json!({ "pipeline_id": "legacy-mixed" })),
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "scenario 2: incompatible legacy manifest"
        );
        let body = json_body(response).await;
        assert_eq!(
            body["error"]["code"], "INCOMPATIBLE_PIPELINE",
            "scenario 2: incompatible error code"
        );
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("modality mismatch"),
            "scenario 2: incompatible error reason"
        );

        let response = request(&mut app, Method::GET, "/v1/pipelines", None).await;
        let body = json_body(response).await;
        assert!(
            !body["pipelines"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["id"] == "legacy-mixed"),
            "scenario 2: rejected pipeline not registered"
        );

        write_pipeline_manifest(
            &model_dir,
            PipelineManifest {
                id: "legacy-multi-classifier-mixed".to_string(),
                steps: vec![
                    PipelineStep {
                        role: PipelineRole::Detector,
                        model: "detector-a".to_string(),
                    },
                    PipelineStep {
                        role: PipelineRole::Classifier,
                        model: "audio-classifier".to_string(),
                    },
                    PipelineStep {
                        role: PipelineRole::Classifier,
                        model: "classifier-a".to_string(),
                    },
                ],
            },
        );

        let response = request(
            &mut app,
            Method::POST,
            "/v1/pipelines/load",
            Some(json!({ "pipeline_id": "legacy-multi-classifier-mixed" })),
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "scenario 2b: incompatible first classifier in legacy manifest"
        );
        let body = json_body(response).await;
        assert_eq!(
            body["error"]["code"], "INCOMPATIBLE_PIPELINE",
            "scenario 2b: incompatible first classifier error code"
        );
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("audio classifiers are not supported"),
            "scenario 2b: incompatible first classifier error reason"
        );

        let response = request(&mut app, Method::GET, "/v1/pipelines", None).await;
        let body = json_body(response).await;
        assert!(
            !body["pipelines"]
                .as_array()
                .unwrap()
                .iter()
                .any(|p| p["id"] == "legacy-multi-classifier-mixed"),
            "scenario 2b: rejected multi-classifier pipeline not registered"
        );

        write_pipeline_manifest(
            &model_dir,
            PipelineManifest {
                id: "legacy-missing".to_string(),
                steps: vec![
                    PipelineStep {
                        role: PipelineRole::Detector,
                        model: "detector-a".to_string(),
                    },
                    PipelineStep {
                        role: PipelineRole::Classifier,
                        model: "ghost-classifier".to_string(),
                    },
                ],
            },
        );

        let response = request(
            &mut app,
            Method::POST,
            "/v1/pipelines/load",
            Some(json!({ "pipeline_id": "legacy-missing" })),
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::NOT_FOUND,
            "scenario 3: missing catalog model"
        );
        let body = json_body(response).await;
        assert_eq!(
            body["error"]["code"], "MODEL_NOT_IN_CATALOG",
            "scenario 3: missing catalog error code"
        );
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("ghost-classifier"),
            "scenario 3: missing catalog model named"
        );

        let lock_count_before_missing_alias = state.pipeline_write_locks.lock().unwrap().len();
        let response = request(
            &mut app,
            Method::POST,
            "/v1/pipelines/load",
            Some(json!({ "pipeline_id": "missing-alias" })),
        )
        .await;
        assert!(
            !response.status().is_success(),
            "scenario 3b: missing pipeline alias must fail"
        );
        {
            let locks = state.pipeline_write_locks.lock().unwrap();
            assert_eq!(
                locks.len(),
                lock_count_before_missing_alias,
                "scenario 3b: failed load must not grow alias locks"
            );
            assert!(
                !locks.contains_key("missing-alias"),
                "scenario 3b: failed missing alias load must prune its transient lock"
            );
        }

        for uri in ["/v1/models/load", "/v1/pipelines/load", "/v1/pipelines"] {
            let response = raw_request(&mut app, Method::POST, uri, "{").await;
            assert_eq!(
                response.status(),
                StatusCode::BAD_REQUEST,
                "scenario 4: bad JSON status uri={uri}"
            );
            let body = json_body(response).await;
            assert_eq!(
                body["error"]["code"], "BAD_REQUEST",
                "scenario 4: bad JSON code uri={uri}"
            );
            assert_eq!(
                body["error"]["status"], 400,
                "scenario 4: bad JSON envelope status uri={uri}"
            );
        }

        for (uri, body_text) in [
            ("/v1/models/load", r#"{"model_id":"detector-a"}"#),
            (
                "/v1/pipelines/load",
                r#"{"pipeline_id":"wildlife-pipeline"}"#,
            ),
            (
                "/v1/pipelines",
                r#"{"id":"p","detector":"detector-a","classifier":"classifier-a"}"#,
            ),
        ] {
            let response = raw_request_with_content_type(
                &mut app,
                Method::POST,
                uri,
                body_text,
                Some("text/plain"),
            )
            .await;
            assert_eq!(
                response.status(),
                StatusCode::UNSUPPORTED_MEDIA_TYPE,
                "scenario 4b: text/plain JSON rejection status uri={uri}"
            );
            let body = json_body(response).await;
            assert_eq!(
                body["error"]["code"], "UNSUPPORTED_MEDIA_TYPE",
                "scenario 4b: text/plain JSON rejection code uri={uri}"
            );
            assert_eq!(
                body["error"]["status"], 415,
                "scenario 4b: text/plain JSON rejection envelope status uri={uri}"
            );
        }

        let response = request(
            &mut app,
            Method::POST,
            "/v1/pipelines/load",
            Some(json!({ "pipeline_id": "bad.alias" })),
        )
        .await;
        assert_eq!(
            response.status(),
            StatusCode::BAD_REQUEST,
            "scenario 5: non-slug alias id"
        );
        let body = json_body(response).await;
        assert_eq!(
            body["error"]["code"], "BAD_REQUEST",
            "scenario 5: non-slug alias error code"
        );
        assert!(
            body["error"]["message"]
                .as_str()
                .unwrap()
                .contains("invalid alias id"),
            "scenario 5: non-slug alias message"
        );
    }

    #[test]
    fn pipeline_toml_escapes_basic_string_values() {
        let manifest = PipelineManifest {
            id: "alias".to_string(),
            steps: vec![PipelineStep {
                role: PipelineRole::Detector,
                model: "detector\"with\nquote".to_string(),
            }],
        };
        let toml = pipeline_toml(&manifest);
        assert!(
            toml.contains("model = \"detector\\\"with\\nquote\""),
            "{toml}"
        );
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("pipeline.toml");
        fs::write(&path, toml).expect("write generated TOML");
        let parsed = crate::engine_dispatch::manifest::load_pipeline_manifest(&path)
            .expect("generated TOML should parse");
        assert_eq!(parsed.steps[0].model, "detector\"with\nquote");
    }

    fn test_router_with_model_dir(name: &str) -> (axum::Router, PathBuf, AppState) {
        let model_dir = unique_model_dir(name);
        let engine =
            Engine::new(EngineConfig::new(Device::Cpu, model_dir.clone())).expect("test engine");
        let state = AppState::with_catalog(engine, test_config(model_dir.clone()), test_catalog());
        (crate::router::build_router(state.clone()), model_dir, state)
    }

    fn unique_model_dir(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("phase4_2_pipeline_mgmt_tests")
            .join(format!("{name}_{}", std::process::id()))
    }

    fn test_config(model_dir: PathBuf) -> Config {
        Config {
            bind_addr: "127.0.0.1:0".parse::<SocketAddr>().unwrap(),
            model_dir,
            log_format: LogFormat::Pretty,
            log_level: "debug".to_string(),
            max_body_size: 1024 * 1024,
            max_concurrent_inference: 1,
            max_batch_size: 4,
            request_timeout_secs: 30,
            drain_timeout_secs: 1,
            device: "cpu".to_string(),
            inter_threads: Some(1),
            intra_threads: Some(1),
            idle_unload_seconds: 0,
            idle_unload_keep_last_n: 1,
        }
    }

    fn test_catalog() -> Catalog {
        let mut models = BTreeMap::new();
        models.insert(
            "detector-a".to_string(),
            model_info("detector-a", ModelType::Detector),
        );
        models.insert(
            "classifier-a".to_string(),
            model_info("classifier-a", ModelType::Classifier),
        );
        models.insert(
            "classifier-b".to_string(),
            model_info("classifier-b", ModelType::Classifier),
        );
        models.insert(
            "audio-detector".to_string(),
            model_info("audio-detector", ModelType::AudioDetector),
        );
        models.insert(
            "audio-classifier".to_string(),
            model_info("audio-classifier", ModelType::AudioClassifier),
        );
        models.insert(
            "overhead-detector".to_string(),
            model_info("overhead-detector", ModelType::OverheadDetector),
        );
        Catalog {
            models,
            pipelines: BTreeMap::new(),
        }
    }

    fn model_info(id: &str, model_type: ModelType) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            path: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("target")
                .join("phase4_2_pipeline_mgmt_tests")
                .join(id)
                .join("manifest.toml"),
            model_type,
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        }
    }

    async fn request(
        app: &mut axum::Router,
        method: Method,
        uri: &str,
        body: Option<Value>,
    ) -> axum::response::Response {
        let mut builder = Request::builder().method(method).uri(uri);
        let body = match body {
            Some(value) => {
                builder = builder.header("content-type", "application/json");
                Body::from(value.to_string())
            }
            None => Body::empty(),
        };
        app.call(builder.body(body).unwrap()).await.unwrap()
    }

    async fn raw_request(
        app: &mut axum::Router,
        method: Method,
        uri: &str,
        body: &str,
    ) -> axum::response::Response {
        raw_request_with_content_type(app, method, uri, body, Some("application/json")).await
    }

    async fn raw_request_with_content_type(
        app: &mut axum::Router,
        method: Method,
        uri: &str,
        body: &str,
        content_type: Option<&str>,
    ) -> axum::response::Response {
        let mut builder = Request::builder().method(method).uri(uri);
        if let Some(content_type) = content_type {
            builder = builder.header("content-type", content_type);
        }
        app.call(builder.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap()
    }

    fn write_pipeline_manifest(model_dir: &Path, manifest: PipelineManifest) {
        let dir = model_dir.join(&manifest.id);
        fs::create_dir_all(&dir).expect("create pipeline dir");
        fs::write(dir.join("pipeline.toml"), pipeline_toml(&manifest))
            .expect("write pipeline manifest");
    }

    async fn json_body(response: axum::response::Response) -> Value {
        let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }
}
