//! Per-request Tier-1/2 drift metrics + manifest reference distribution
//! (Phase 4 W4).
//!
//! These types are wire-format only. The actual compute lives in
//! `sparrow-engine-server/src/drift.rs` (request-level percentile + PSI). Tier-3
//! (cross-request reference + CUSUM + alarm path) is the eventual `sparrow-ops`
//! sibling's responsibility — sparrow-engine only emits the per-request snapshot.
//!
//! Why split metrics from compute:
//! - `DriftMetrics` ships in `InferenceLogRecord` (W2) on the wire; the type
//!   has to live in `sparrow-engine-types` (the leaf crate, zero ORT/CUDA deps) so
//!   sibling repos can consume it without dragging the engine in.
//! - `compute_drift_metrics` lives in `sparrow-engine-server` because it joins request
//!   results to the active manifest's optional `[drift_reference]`.

use serde::{Deserialize, Serialize};

/// Per-request stateless drift metrics (Tier-1/2). Embedded in
/// `InferenceLogRecord.drift_metrics`.
///
/// Fields:
/// - `confidence_p50` / `confidence_p95`: nearest-rank percentile over the
///   request's detection / classification / segment confidences (NaN dropped).
///   `0.0` when the request has zero outputs.
/// - `detections_per_image`: outputs per input image. Single-image endpoints
///   use `1`; batch detect uses `images.len()`; audio treats the call as
///   one "image" with the segment count as the numerator.
/// - `class_distribution_psi`: Population Stability Index against the
///   manifest `[drift_reference] class_distribution` map. `None` when no
///   reference is configured.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DriftMetrics {
    pub confidence_p50: f32,
    pub confidence_p95: f32,
    pub detections_per_image: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub class_distribution_psi: Option<f32>,
}

/// Manifest `[drift_reference]` section — the per-model reference
/// distribution against which `DriftMetrics::class_distribution_psi` is
/// computed.
///
/// TOML form:
/// ```toml
/// [drift_reference.class_distribution]
/// animal  = 0.7
/// person  = 0.2
/// vehicle = 0.1
/// ```
///
/// Missing section ⇒ `ModelManifest.drift_reference = None` ⇒ PSI is `None`
/// in every request's `DriftMetrics`. Frequencies should sum to ~1.0 but the
/// implementation does not require it (the smoothing handles whatever the
/// operator provides).
///
/// `BTreeMap` is used (not `HashMap`) so serialize order is deterministic;
/// JSON-line emit must be byte-stable across runs to keep diff-based testing
/// + downstream content-addressed storage simple.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct DriftReference {
    pub class_distribution: std::collections::BTreeMap<String, f32>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_metrics_default_is_zero() {
        let m: DriftMetrics = Default::default();
        assert_eq!(m.confidence_p50, 0.0);
        assert_eq!(m.confidence_p95, 0.0);
        assert_eq!(m.detections_per_image, 0.0);
        assert_eq!(m.class_distribution_psi, None);
    }

    #[test]
    fn drift_metrics_omits_psi_when_none() {
        let m = DriftMetrics::default();
        let json_str = serde_json::to_string(&m).unwrap();
        assert!(
            !json_str.contains("class_distribution_psi"),
            "class_distribution_psi=None must be omitted, got: {json_str}"
        );
    }

    #[test]
    fn drift_metrics_includes_psi_when_some() {
        let m = DriftMetrics {
            class_distribution_psi: Some(0.5),
            ..Default::default()
        };
        let json_str = serde_json::to_string(&m).unwrap();
        assert!(
            json_str.contains("\"class_distribution_psi\":0.5"),
            "class_distribution_psi=Some must serialize, got: {json_str}"
        );
    }

    #[test]
    fn drift_reference_serializes_with_deterministic_key_order() {
        // BTreeMap iteration is sorted by key, so re-serialization must be
        // byte-identical regardless of insertion order. Important for
        // content-addressed storage on the sparrow-data side.
        let mut a = std::collections::BTreeMap::new();
        a.insert("zebra".to_string(), 0.1);
        a.insert("antelope".to_string(), 0.5);
        a.insert("lion".to_string(), 0.4);
        let r1 = DriftReference {
            class_distribution: a,
        };

        let mut b = std::collections::BTreeMap::new();
        b.insert("lion".to_string(), 0.4);
        b.insert("antelope".to_string(), 0.5);
        b.insert("zebra".to_string(), 0.1);
        let r2 = DriftReference {
            class_distribution: b,
        };

        assert_eq!(
            serde_json::to_string(&r1).unwrap(),
            serde_json::to_string(&r2).unwrap(),
            "BTreeMap serialization must be insertion-order-independent"
        );
    }
}
