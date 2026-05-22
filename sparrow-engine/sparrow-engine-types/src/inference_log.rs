//! Inference-log wire format (Phase 4 W2).
//!
//! Bongo emits one `InferenceLogRecord` per `?store=true` inference request.
//! The eventual `sparrow-data` sibling ingests these records; `sparrow-ops` reads
//! the `drift_metrics` field to drive Tier-3 cross-request analysis. Bongo
//! never reads its own emit — this module only defines the wire shape.
//!
//! See `docs/design/phase4/schema.md` for the canonical schema spec
//! (worked example, field semantics, schema-version policy).

use serde::{Deserialize, Serialize};

use crate::drift_metrics::DriftMetrics;
use crate::manifest::ProvenanceRecord;

/// Schema version for the inference-log wire format.
///
/// Additive changes (new optional field with `#[serde(default)]`) keep "1.0".
/// Any breaking change (rename, type change, semantic shift) bumps to "2.0"
/// with a coordinated `sparrow-data` ingester change.
pub const SCHEMA_VERSION: &str = "1.0";

/// One inference log record. Emitted as a JSON line by the default sink;
/// alternative sinks (`sparrow-data` HTTP ingest, future filesystem store) are
/// pluggable via the `InferenceLogSink` trait in `sparrow-engine-server`.
///
/// Idempotency: implementations should treat `(media_hash, model_id)` as a
/// UNIQUE constraint and silently drop duplicates. The default stderr sink
/// does NOT enforce this — uniqueness is a backend property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InferenceLogRecord {
    /// Wire-format version. Currently `SCHEMA_VERSION = "1.0"`.
    pub schema_version: String,

    /// Per-request identifier. UUID v4 hex.
    pub request_id: String,

    /// RFC3339 millisecond-precision UTC timestamp,
    /// e.g. `"2026-05-07T12:34:56.789Z"`.
    pub timestamp_utc: String,

    /// SHA-256 lowercase hex of the request media bytes. Image: the image
    /// bytes; audio: the audio bytes; batch detect: the first image's bytes
    /// (per-image detail lives inside `result`).
    pub media_hash: String,

    /// Model ID from the request (`model` query param). For
    /// `/v1/pipeline`, this is the `pipeline` query param —
    /// pipeline_id is the dedup key, not a constituent step's model id.
    pub model_id: String,

    /// Optional model version string. Currently always `None`; reserved for
    /// `sparrow-data` to populate from a manifest provenance lookup.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,

    /// Active device label (`"cpu"`, `"cuda:0"`, etc). Never `"auto"` —
    /// `Engine::active_device` resolves Auto.
    pub device: String,

    /// Inference processing time in milliseconds (widened from the engine's
    /// f32 to make downstream f64 stats analysis straightforward).
    pub inference_ms: f64,

    /// The full HTTP response payload that was returned to the client.
    /// Round-trips the inference shape unchanged.
    pub result: serde_json::Value,

    /// Optional manifest `[provenance]` snapshot. Populated from the active
    /// `ModelManifest.provenance` when the manifest exposes a `[provenance]`
    /// section; `None` when the manifest omits it. For `/v1/pipeline`, the
    /// classifier-step manifest is preferred (falls back to the detector step
    /// when no classifier exists).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<ProvenanceRecord>,

    /// Per-request Tier-1/2 drift metrics. Populated when `?store=true`,
    /// `None` otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub drift_metrics: Option<DriftMetrics>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn sample_record() -> InferenceLogRecord {
        InferenceLogRecord {
            schema_version: SCHEMA_VERSION.to_string(),
            request_id: "5b2f6d8c-9a4e-4c1f-8b3d-6e7a1f2c3d4e".to_string(),
            timestamp_utc: "2026-05-07T12:34:56.789Z".to_string(),
            media_hash: "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
                .to_string(),
            model_id: "megadetector-v6-yolov10e".to_string(),
            model_version: None,
            device: "cuda:0".to_string(),
            inference_ms: 13.46,
            result: json!({
                "model_id": "megadetector-v6-yolov10e",
                "image_size": [1920, 1080],
                "processing_time_ms": 13.46,
                "detections": []
            }),
            provenance: None,
            drift_metrics: Some(DriftMetrics {
                confidence_p50: 0.94,
                confidence_p95: 0.94,
                detections_per_image: 1.0,
                class_distribution_psi: Some(0.0123),
            }),
        }
    }

    #[test]
    fn round_trip_serialize_deserialize_equals() {
        let original = sample_record();
        let json_str =
            serde_json::to_string(&original).expect("InferenceLogRecord must serialize");
        let parsed: InferenceLogRecord =
            serde_json::from_str(&json_str).expect("InferenceLogRecord must deserialize");
        assert_eq!(parsed, original, "round-trip must preserve every field");
    }

    #[test]
    fn none_optional_fields_omitted_from_json() {
        // model_version=None and provenance=None must be skipped — sparrow-data
        // ingest schemas should treat them as absent, not as JSON null.
        let record = sample_record();
        let json_str = serde_json::to_string(&record).unwrap();
        assert!(
            !json_str.contains("\"model_version\""),
            "model_version=None must be omitted from JSON, got: {json_str}"
        );
        assert!(
            !json_str.contains("\"provenance\""),
            "provenance=None must be omitted from JSON, got: {json_str}"
        );
        // drift_metrics IS Some, so it must still be present.
        assert!(
            json_str.contains("\"drift_metrics\""),
            "drift_metrics=Some must remain in JSON, got: {json_str}"
        );
    }

    #[test]
    fn schema_version_constant_is_stable() {
        // Locked-in surface — bumping this value is a breaking change that
        // requires a coordinated sparrow-data ingester update.
        assert_eq!(SCHEMA_VERSION, "1.0");
    }

    #[test]
    fn provenance_some_serializes_with_pointer_fields() {
        // Phase 4 W3 wires provenance from the active manifest into every
        // ?store=true emit; the wire shape must round-trip the three pointer
        // fields when they are populated.
        let mut record = sample_record();
        record.provenance = Some(ProvenanceRecord {
            training_dataset_id: Some("ds-2026-04-camera-trap-r1".to_string()),
            training_experiment_id: Some("exp-mdv6-fp16-r3".to_string()),
            training_repo_commit: Some("9c4b6a3".to_string()),
        });
        let json_str = serde_json::to_string(&record).expect("serialize");
        assert!(
            json_str.contains("\"provenance\""),
            "provenance=Some must appear in JSON, got: {json_str}"
        );
        assert!(json_str.contains("\"training_dataset_id\":\"ds-2026-04-camera-trap-r1\""));
        assert!(json_str.contains("\"training_experiment_id\":\"exp-mdv6-fp16-r3\""));
        assert!(json_str.contains("\"training_repo_commit\":\"9c4b6a3\""));
        let parsed: InferenceLogRecord = serde_json::from_str(&json_str).expect("deserialize");
        assert_eq!(parsed, record);
    }
}
