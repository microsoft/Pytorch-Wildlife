//! Handler for POST /v1/pipeline — named or ad-hoc detect + classify pipeline.

use std::sync::Arc;

use axum::extract::multipart::MultipartRejection;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Multipart, Query, State};
use axum::http::StatusCode;
use axum::Json;
use serde::Deserialize;

use crate::engine_dispatch::manifest::{PipelineManifest, PipelineRole};
use crate::engine_dispatch::{
    pipeline as engine_pipeline, ClassifyOpts, DetectOpts, ImageInput, ModelType,
};
use crate::error::AppError;
use crate::response::{PipelineDetectionResponse, PipelineResponse};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct PipelineParams {
    pub pipeline: Option<String>,
    pub detector: Option<String>,
    pub classifier: Option<String>,
    pub threshold: Option<f32>,
    pub max_detections: Option<u32>,
    pub top_k: Option<u32>,
    pub store: Option<bool>,
    pub halt_on_store_failure: Option<bool>,
}

enum PipelineSelection {
    Named {
        pipeline_id: String,
        manifest: PipelineManifest,
    },
    Adhoc {
        detector_id: String,
        classifier_id: String,
    },
}

#[derive(Debug)]
enum PipelineQueryShape {
    Named {
        pipeline_id: String,
    },
    Adhoc {
        detector_id: String,
        classifier_id: String,
    },
}

pub async fn pipeline(
    State(state): State<AppState>,
    query: Result<Query<PipelineParams>, QueryRejection>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<PipelineResponse>, AppError> {
    let Query(params) = query.map_err(|e| AppError::bad_request(format!("invalid query: {e}")))?;
    let selection = classify_pipeline_request(&state, &params)?;
    super::validate_threshold(params.threshold)?;
    super::validate_max_detections(params.max_detections)?;
    if params.top_k == Some(0) {
        return Err(AppError::bad_request("top_k must be >= 1"));
    }
    let mut multipart = multipart.map_err(|e| AppError::bad_request(e.body_text()))?;
    let image_bytes = super::extract_field(&mut multipart, "image").await?;
    let (pipeline_id_for_log, detector_for_log, classifier_for_log) = log_model_ids(&selection);
    let log_step_model_id = classifier_for_log.or(detector_for_log);

    let permit = super::acquire_inference_permit(&state.inference_semaphore)?;

    let detect_opts = DetectOpts {
        confidence_threshold: params.threshold,
        max_detections: params.max_detections,
    };
    let classify_opts = ClassifyOpts {
        top_k: params.top_k.or(Some(5)),
    };

    let store = params.store.unwrap_or(false);
    let halt_on_store_failure = params.halt_on_store_failure.unwrap_or(false);
    let media_hash = store.then(|| super::sha256_lower_hex(&image_bytes));

    let engine = Arc::clone(&state.engine);
    let store_for_log = store;
    let (result, log_drift_ref, log_provenance) = super::run_blocking(move || {
        let _permit = permit;
        let log_metadata = if store_for_log {
            if let Some(model_id) = log_step_model_id.as_deref() {
                let handle = engine.get_or_load_model(model_id)?;
                (
                    handle.manifest().drift_reference.clone(),
                    handle.manifest().provenance.clone(),
                )
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let image = ImageInput::Encoded(image_bytes);
        let result = match selection {
            PipelineSelection::Named {
                pipeline_id,
                manifest,
            } => {
                for step in &manifest.steps {
                    engine.get_or_load_model(&step.model)?;
                }
                engine_pipeline::run_pipeline(
                    &engine,
                    &pipeline_id,
                    &image,
                    &detect_opts,
                    &classify_opts,
                )
            }
            PipelineSelection::Adhoc {
                detector_id,
                classifier_id,
            } => engine_pipeline::run_pipeline_adhoc(
                &engine,
                &image,
                &detector_id,
                &classifier_id,
                &detect_opts,
                &classify_opts,
            ),
        }?;
        Ok((result, log_metadata.0, log_metadata.1))
    })
    .await?;

    let response = PipelineResponse {
        pipeline_id: result.pipeline_id,
        image_size: [result.image_width, result.image_height],
        processing_time_ms: result.processing_time_ms,
        detections: result
            .detections
            .into_iter()
            .map(PipelineDetectionResponse::from)
            .collect(),
    };

    if store {
        let confidences: Vec<f32> = response.detections.iter().map(|d| d.confidence).collect();
        let labels: Vec<String> = response
            .detections
            .iter()
            .map(|d| {
                d.classification
                    .as_ref()
                    .map(|c| c.label.clone())
                    .unwrap_or_else(|| d.label.clone())
            })
            .collect();

        let drift =
            crate::drift::compute_drift_metrics(&confidences, 1, &labels, log_drift_ref.as_ref());
        let value = serde_json::to_value(&response).unwrap_or(serde_json::Value::Null);
        let record = super::build_log_record(
            &state,
            media_hash.ok_or_else(|| AppError::internal("media hash missing when store=true"))?,
            pipeline_id_for_log,
            value,
            response.processing_time_ms as f64,
            drift,
            log_provenance,
        );
        super::emit_log_record(&state, &record, halt_on_store_failure)?;
    }

    Ok(Json(response))
}

fn classify_pipeline_request(
    state: &AppState,
    params: &PipelineParams,
) -> Result<PipelineSelection, AppError> {
    match classify_pipeline_query_shape(params)? {
        PipelineQueryShape::Named { pipeline_id } => {
            let manifest = state.engine.get_pipeline(&pipeline_id)?;
            Ok(PipelineSelection::Named {
                pipeline_id,
                manifest,
            })
        }
        PipelineQueryShape::Adhoc {
            detector_id,
            classifier_id,
        } => {
            let detector_type = catalog_model_type(state, &detector_id)?;
            let classifier_type = catalog_model_type(state, &classifier_id)?;
            crate::engine_dispatch::pipeline_compat::validate_pipeline_compat(
                Some(detector_type),
                Some(classifier_type),
            )?;
            Ok(PipelineSelection::Adhoc {
                detector_id,
                classifier_id,
            })
        }
    }
}

fn classify_pipeline_query_shape(params: &PipelineParams) -> Result<PipelineQueryShape, AppError> {
    let pipeline = clean_query_string("pipeline", params.pipeline.as_deref())?;
    let detector = clean_query_string("detector", params.detector.as_deref())?;
    let classifier = clean_query_string("classifier", params.classifier.as_deref())?;

    if pipeline.is_some() && (detector.is_some() || classifier.is_some()) {
        return Err(AppError::bad_request(
            "specify either `pipeline=` OR `detector=`+`classifier=`, not both",
        ));
    }

    match (pipeline, detector, classifier) {
        (Some(pipeline_id), None, None) => {
            super::pipelines_mgmt::validate_alias_id(&pipeline_id)?;
            Ok(PipelineQueryShape::Named { pipeline_id })
        }
        (None, Some(detector_id), Some(classifier_id)) => {
            super::validate_id(&detector_id, "detector")?;
            super::validate_id(&classifier_id, "classifier")?;
            Ok(PipelineQueryShape::Adhoc {
                detector_id,
                classifier_id,
            })
        }
        (None, Some(_), None) => Err(AppError::bad_request(
            "Shape X requires both `detector=` and `classifier=`",
        )),
        (None, None, Some(_)) => Err(AppError::bad_request(
            "classifier-only pipelines must use POST /v1/classify?model=<id>; named-pipeline aliases require both detector and classifier",
        )),
        (None, None, None) => Err(AppError::bad_request(
            "one of `pipeline=` or `detector=`+`classifier=` required",
        )),
        _ => Err(AppError::bad_request("invalid pipeline query shape")),
    }
}

fn clean_query_string(name: &str, value: Option<&str>) -> Result<Option<String>, AppError> {
    match value {
        Some(value) => {
            let trimmed = value.trim();
            if trimmed.is_empty() {
                Err(AppError::bad_request(format!("empty value for `{name}`")))
            } else {
                Ok(Some(trimmed.to_string()))
            }
        }
        None => Ok(None),
    }
}

fn catalog_model_type(state: &AppState, model_id: &str) -> Result<ModelType, AppError> {
    state
        .catalog
        .models
        .get(model_id)
        .map(|m| m.model_type)
        .ok_or_else(|| AppError::Http {
            status: StatusCode::NOT_FOUND,
            code: "MODEL_NOT_IN_CATALOG".to_string(),
            message: format!("Model '{model_id}' is not in the discovered catalog"),
        })
}

fn log_model_ids(selection: &PipelineSelection) -> (String, Option<String>, Option<String>) {
    match selection {
        PipelineSelection::Named {
            pipeline_id,
            manifest,
        } => {
            let detector = manifest
                .steps
                .iter()
                .find(|s| s.role == PipelineRole::Detector)
                .map(|s| s.model.clone());
            let classifier = manifest
                .steps
                .iter()
                .find(|s| s.role == PipelineRole::Classifier)
                .map(|s| s.model.clone());
            (pipeline_id.clone(), detector, classifier)
        }
        PipelineSelection::Adhoc {
            detector_id,
            classifier_id,
        } => (
            format!("adhoc:{detector_id}+{classifier_id}"),
            Some(detector_id.clone()),
            Some(classifier_id.clone()),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::StatusCode;

    #[test]
    fn clean_query_string_rejects_empty_values() {
        assert!(clean_query_string("pipeline", Some("  ")).is_err());
        match clean_query_string("pipeline", Some(" alias ")) {
            Ok(value) => assert_eq!(value, Some("alias".to_string())),
            Err(_) => panic!("expected trimmed query value"),
        }
    }

    #[test]
    fn named_pipeline_rejects_non_slug_alias_without_catalog_or_multipart() {
        let params = PipelineParams {
            pipeline: Some("alias.with.dot".to_string()),
            detector: None,
            classifier: None,
            threshold: None,
            max_detections: None,
            top_k: None,
            store: None,
            halt_on_store_failure: None,
        };

        match classify_pipeline_query_shape(&params).unwrap_err() {
            AppError::Http {
                status,
                code,
                message,
            } => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert_eq!(code, "BAD_REQUEST");
                assert!(message.contains("invalid alias id"));
            }
            AppError::Bongo(e) => panic!("expected HTTP alias error, got {e}"),
        }
    }

    #[test]
    fn query_shape_conflict_rejects_without_catalog_or_multipart() {
        let params = PipelineParams {
            pipeline: Some("alias".to_string()),
            detector: Some("detector-a".to_string()),
            classifier: None,
            threshold: None,
            max_detections: None,
            top_k: None,
            store: None,
            halt_on_store_failure: None,
        };

        match classify_pipeline_query_shape(&params).unwrap_err() {
            AppError::Http {
                status,
                code,
                message,
            } => {
                assert_eq!(status, StatusCode::BAD_REQUEST);
                assert_eq!(code, "BAD_REQUEST");
                assert!(message
                    .contains("specify either `pipeline=` OR `detector=`+`classifier=`, not both"));
            }
            AppError::Bongo(e) => panic!("expected HTTP shape error, got {e}"),
        }
    }
}
