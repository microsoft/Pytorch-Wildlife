//! Handler for POST /v1/audio/detect — audio detection with sliding window.

use std::io::Write;

use axum::extract::multipart::MultipartRejection;
use axum::extract::rejection::QueryRejection;
use axum::extract::{Multipart, Query, State};
use axum::Json;
use serde::Deserialize;

use crate::engine_dispatch::{
    detect_audio, AudioDetectOpts, AudioInput, AudioSegment, SparrowEngineError,
};
use crate::error::AppError;
use crate::response::{AudioDetectResponse, AudioSegmentResponse};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct AudioDetectParams {
    pub model: String,
    pub threshold: Option<f32>,
    pub segment_duration: Option<f32>,
    pub stride: Option<f32>,
    #[serde(default)]
    pub store: bool,
    #[serde(default)]
    pub halt_on_store_failure: bool,
}

fn drift_label_for_audio_segment(segment: &AudioSegment, model_id: &str) -> String {
    segment
        .classes
        .first()
        .and_then(|class| class.label.as_ref())
        .cloned()
        .unwrap_or_else(|| model_id.to_string())
}

pub async fn audio_detect(
    State(state): State<AppState>,
    query: Result<Query<AudioDetectParams>, QueryRejection>,
    multipart: Result<Multipart, MultipartRejection>,
) -> Result<Json<AudioDetectResponse>, AppError> {
    let Query(params) = query.map_err(|e| AppError::bad_request(format!("invalid query: {e}")))?;
    let mut multipart = multipart.map_err(|e| AppError::bad_request(e.body_text()))?;
    let audio_bytes = super::extract_field(&mut multipart, "audio").await?;
    super::validate_threshold(params.threshold)?;
    if let Some(d) = params.segment_duration {
        if !d.is_finite() || d <= 0.0 {
            return Err(AppError::bad_request(
                "segment_duration must be a finite positive number",
            ));
        }
    }
    if let Some(s) = params.stride {
        if !s.is_finite() || s <= 0.0 {
            return Err(AppError::bad_request(
                "stride must be a finite positive number",
            ));
        }
    }

    let permit = super::acquire_inference_permit(&state.inference_semaphore)?;

    // Phase 4 W3 — hash the audio bytes before they go into the temp file.
    let media_hash = params.store.then(|| super::sha256_lower_hex(&audio_bytes));

    let model_id = params.model.clone();
    let opts = AudioDetectOpts {
        confidence_threshold: params.threshold,
        segment_duration_s: params.segment_duration,
        stride_s: params.stride,
    };

    let engine = std::sync::Arc::clone(&state.engine);
    let model_id_for_load = params.model.clone();
    let want_manifest_meta = params.store;
    let (result, drift_reference, provenance) = super::run_blocking(move || {
        let _permit = permit;
        // Phase 4.2 lazy-load: resolve (or load on demand) inside the blocking
        // pool so the async runtime stays responsive.
        let handle = engine.get_or_load_model(&model_id_for_load)?;
        // Write audio to a temp file on the blocking pool for sparrow-engine-cpu
        // (AudioInput::FilePath), keeping the file alive through inference.
        let mut tmp = tempfile::NamedTempFile::new().map_err(SparrowEngineError::Io)?;
        tmp.write_all(&audio_bytes).map_err(SparrowEngineError::Io)?;
        let audio_input = AudioInput::FilePath(tmp.path().to_path_buf());
        let _keep = tmp;
        let result = detect_audio::detect_audio(&handle, &audio_input, &opts)?;
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

    let store_metrics = params.store.then(|| {
        let confidences: Vec<f32> = result.segments.iter().map(|s| s.confidence).collect();
        let labels: Vec<String> = result
            .segments
            .iter()
            .map(|s| drift_label_for_audio_segment(s, &model_id))
            .collect();
        (confidences, labels)
    });

    let response = AudioDetectResponse {
        model_id: model_id.clone(),
        duration_s: result.duration_s,
        sample_rate: result.sample_rate,
        processing_time_ms: result.processing_time_ms,
        segments: result
            .segments
            .into_iter()
            .map(AudioSegmentResponse::from)
            .collect(),
    };

    if params.store {
        let (confidences, labels) = store_metrics
            .ok_or_else(|| AppError::internal("store metrics missing when store=true"))?;
        let drift =
            crate::drift::compute_drift_metrics(&confidences, 1, &labels, drift_reference.as_ref());
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine_dispatch::AudioClass;

    fn segment(classes: Vec<AudioClass>) -> AudioSegment {
        AudioSegment {
            start_time_s: 0.0,
            end_time_s: 1.0,
            confidence: 0.5,
            classes,
        }
    }

    fn class(label: Option<&str>, probability: f32) -> AudioClass {
        AudioClass {
            class_idx: 0,
            label: label.map(str::to_string),
            probability,
        }
    }

    #[test]
    fn drift_label_uses_index_zero_class_label_without_resorting() {
        let segment = segment(vec![class(Some("first"), 0.1), class(Some("second"), 0.9)]);

        assert_eq!(drift_label_for_audio_segment(&segment, "model"), "first");
    }

    #[test]
    fn drift_label_uses_single_labeled_class() {
        let segment = segment(vec![class(Some("bird"), 0.8)]);

        assert_eq!(drift_label_for_audio_segment(&segment, "model"), "bird");
    }

    #[test]
    fn drift_label_falls_back_when_classes_are_empty() {
        assert_eq!(
            drift_label_for_audio_segment(&segment(Vec::new()), "model"),
            "model"
        );
    }

    #[test]
    fn drift_label_does_not_substitute_lower_ranked_label_when_top1_unlabeled() {
        let segment = segment(vec![class(None, 0.8), class(Some("second"), 0.2)]);

        assert_eq!(drift_label_for_audio_segment(&segment, "model"), "model");
    }

    #[test]
    fn drift_label_preserves_empty_string_label() {
        let segment = segment(vec![class(Some(""), 0.8)]);

        assert_eq!(drift_label_for_audio_segment(&segment, "model"), "");
    }
}
