//! Per-request Tier-1/2 drift compute (Phase 4 W3).
//!
//! Stateless — every request computes its own snapshot. Tier-3 (cross-request
//! reference + CUSUM + alarm path) lives in the eventual `sparrow-ops` sibling.
//!
//! Media-agnostic: image, audio, and pipeline handlers all call
//! `compute_drift_metrics`. The W4 wire types live in `sparrow-engine-types`; the
//! compute lives here so it can join request results to the active manifest.
//!
//! See `docs/design/phase4/schema.md` for field semantics + smoothing
//! constant + worked example.

use crate::engine_dispatch::{DriftMetrics, DriftReference};

/// Compute Tier-1/2 drift for a single inference request.
///
/// - `confidences`: every detection / classification / segment confidence
///   in this request.
/// - `image_count`: 1 for single-image and audio; `images.len()` for batch.
///   `0` is treated as `1` to avoid division by zero.
/// - `class_labels`: the observed class label per output (or per segment
///   for audio). Same length as `confidences`. Used for PSI input.
/// - `reference`: optional reference distribution from the active manifest.
pub fn compute_drift_metrics(
    confidences: &[f32],
    image_count: usize,
    class_labels: &[String],
    reference: Option<&DriftReference>,
) -> DriftMetrics {
    let confidence_p50 = percentile(confidences, 0.50);
    let confidence_p95 = percentile(confidences, 0.95);
    let denom = image_count.max(1) as f32;
    let detections_per_image = confidences.len() as f32 / denom;
    let class_distribution_psi = reference.and_then(|r| psi(class_labels, &r.class_distribution));
    DriftMetrics {
        confidence_p50,
        confidence_p95,
        detections_per_image,
        class_distribution_psi,
    }
}

/// Population Stability Index against a reference distribution.
///
/// `Σ (p_i - q_i) * ln(p_i / q_i)` over the union of class buckets, where
/// `p` is observed and `q` is reference. Both are smoothed with `eps = 1e-4`
/// so a missing bucket on either side never produces `inf` or `NaN`.
///
/// Returns `None` when both observed and reference are empty (PSI is
/// undefined; no signal to report).
fn psi(
    observed_labels: &[String],
    reference: &std::collections::BTreeMap<String, f32>,
) -> Option<f32> {
    if observed_labels.is_empty() && reference.is_empty() {
        return None;
    }
    let total = observed_labels.len() as f32;
    let mut observed: std::collections::BTreeMap<&str, f32> = std::collections::BTreeMap::new();
    // When `total == 0.0` (observed empty, reference non-empty), the loop is
    // a natural no-op — `observed` stays empty, the union below runs over
    // reference keys only, and each `p` falls back to `eps` via `.max(eps)`.
    for label in observed_labels {
        *observed.entry(label.as_str()).or_insert(0.0) += 1.0 / total;
    }
    let eps = 1e-4_f32;
    let mut sum = 0.0_f32;
    let union: std::collections::BTreeSet<&str> = observed
        .keys()
        .copied()
        .chain(reference.keys().map(String::as_str))
        .collect();
    for k in union {
        let p = observed.get(k).copied().unwrap_or(0.0).max(eps);
        let q = reference.get(k).copied().unwrap_or(0.0).max(eps);
        sum += (p - q) * (p / q).ln();
    }
    Some(sum)
}

/// Nearest-rank percentile. NaN values are dropped before sort.
/// `p` is the percentile rank in [0,1]. Empty input — or input that is
/// entirely NaN after the filter — returns `0.0` as a sentinel.
fn percentile(values: &[f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f32> = values.iter().copied().filter(|x| !x.is_nan()).collect();
    if sorted.is_empty() {
        return 0.0;
    }
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let p = p.clamp(0.0, 1.0);
    let idx = if p <= 0.0 {
        0
    } else {
        ((p * sorted.len() as f32).ceil() as usize).saturating_sub(1)
    };
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_reference_returns_psi_none() {
        let m = compute_drift_metrics(&[0.9, 0.8], 1, &["cat".into()], None);
        assert_eq!(m.class_distribution_psi, None);
        assert!((m.detections_per_image - 2.0).abs() < 1e-6);
    }

    #[test]
    fn psi_matches_reference_distribution_is_near_zero() {
        let mut ref_dist = std::collections::BTreeMap::new();
        ref_dist.insert("cat".to_string(), 0.5);
        ref_dist.insert("dog".to_string(), 0.5);
        let r = DriftReference {
            class_distribution: ref_dist,
        };
        let labels = vec!["cat".to_string(), "dog".to_string()];
        let m = compute_drift_metrics(&[0.9, 0.8], 1, &labels, Some(&r));
        let psi = m
            .class_distribution_psi
            .expect("reference configured ⇒ PSI must be Some");
        assert!(
            psi.abs() < 0.01,
            "matching distribution PSI must be ≈0, got {psi}"
        );
    }

    #[test]
    fn psi_diverged_distribution_is_substantial() {
        let mut ref_dist = std::collections::BTreeMap::new();
        ref_dist.insert("cat".to_string(), 0.5);
        ref_dist.insert("dog".to_string(), 0.5);
        let r = DriftReference {
            class_distribution: ref_dist,
        };
        let labels = vec!["cat".to_string(); 4];
        let m = compute_drift_metrics(&[0.9, 0.8, 0.7, 0.6], 1, &labels, Some(&r));
        let psi = m
            .class_distribution_psi
            .expect("reference configured ⇒ PSI must be Some");
        assert!(
            psi > 0.1,
            "100%-cat observed vs 50/50 reference must produce substantial PSI, got {psi}"
        );
    }

    #[test]
    fn reference_only_observed_empty_returns_some_psi() {
        // Regression guard for the dropped `if total > 0.0` branch in `psi`:
        // observed empty + reference non-empty must still produce a finite
        // PSI value (smoothing folds observed back to `eps`).
        let mut ref_dist = std::collections::BTreeMap::new();
        ref_dist.insert("cat".to_string(), 0.7);
        ref_dist.insert("dog".to_string(), 0.3);
        let r = DriftReference {
            class_distribution: ref_dist,
        };
        let m = compute_drift_metrics(&[], 1, &[], Some(&r));
        let psi = m
            .class_distribution_psi
            .expect("reference configured + observed empty ⇒ PSI must be Some");
        assert!(psi.is_finite(), "PSI must be finite, got {psi}");
        assert!(
            psi > 0.0,
            "diverged distribution must produce positive PSI, got {psi}"
        );
    }

    #[test]
    fn percentile_basic_ascending_input() {
        let v: Vec<f32> = (1..=10).map(|i| i as f32 / 10.0).collect();
        let p50 = percentile(&v, 0.5);
        let p95 = percentile(&v, 0.95);
        // Nearest-rank: ceil(p*N)-1 after sorting.
        assert!(
            (p50 - 0.5).abs() < f32::EPSILON,
            "p50 should be 0.5, got {p50}"
        );
        assert!(
            (p95 - 1.0).abs() < f32::EPSILON,
            "p95 should be 1.0, got {p95}"
        );
    }

    #[test]
    fn empty_confidences_yield_zero_metrics() {
        let m = compute_drift_metrics(&[], 0, &[], None);
        assert_eq!(m.confidence_p50, 0.0);
        assert_eq!(m.confidence_p95, 0.0);
        assert_eq!(m.detections_per_image, 0.0);
        assert_eq!(m.class_distribution_psi, None);
    }

    #[test]
    fn nan_confidences_dropped_before_percentile() {
        let v = vec![0.5_f32, f32::NAN, 0.9];
        let p50 = percentile(&v, 0.5);
        // After dropping NaN: [0.5, 0.9] sorted → ceil(0.5*2)-1=0 → 0.5.
        assert!(
            (p50 - 0.5).abs() < f32::EPSILON,
            "NaN must not poison percentile, got {p50}"
        );
    }

    // -- Phase 4 audit-fix R1 regression tests (T-1, T-2, T-4) --------------

    /// T-1 — Hand-traced behavior pinned: when `observed_labels` is empty but
    /// `reference` is non-empty, `psi` returns `Some(positive)` (not `None`,
    /// not panic, not NaN/Inf). Prevents a future refactor of the
    /// `if total > 0.0` guard from regressing into a 0/0 division.
    #[test]
    fn psi_empty_observed_non_empty_reference_returns_some_positive() {
        let mut ref_dist = std::collections::BTreeMap::new();
        ref_dist.insert("animal".to_string(), 0.7);
        ref_dist.insert("person".to_string(), 0.3);
        let r = DriftReference {
            class_distribution: ref_dist,
        };
        let m = compute_drift_metrics(&[], 1, &[], Some(&r));
        let psi = m
            .class_distribution_psi
            .expect("non-empty reference must produce Some, never None");
        assert!(
            psi.is_finite(),
            "empty observed must NOT produce NaN/Inf, got {psi}"
        );
        assert!(
            psi > 0.0,
            "empty observed against non-empty reference must produce positive PSI, got {psi}"
        );
    }

    /// T-2 — All-NaN input returns the documented 0.0 sentinel (rather than
    /// panicking on the unwrap inside `partial_cmp(b).unwrap()` or sliding
    /// past the `sorted.is_empty()` guard).
    #[test]
    fn percentile_all_nan_input_returns_zero_sentinel() {
        let v = vec![f32::NAN, f32::NAN, f32::NAN];
        assert_eq!(
            percentile(&v, 0.5),
            0.0,
            "all-NaN input must return 0.0 sentinel"
        );
        assert_eq!(
            percentile(&v, 0.95),
            0.0,
            "all-NaN input must return 0.0 sentinel even at p95"
        );
    }

    /// T-4a — Single value: every percentile maps to that one value.
    #[test]
    fn percentile_single_value_returns_that_value() {
        let v = vec![0.42_f32];
        assert_eq!(percentile(&v, 0.0), 0.42);
        assert_eq!(percentile(&v, 0.5), 0.42);
        assert_eq!(percentile(&v, 0.95), 0.42);
        assert_eq!(percentile(&v, 1.0), 0.42);
    }

    /// T-4b — `p = 0.0` returns the minimum (nearest-rank index 0 after sort).
    #[test]
    fn percentile_p_zero_returns_min() {
        let v = vec![0.7_f32, 0.2, 0.9, 0.1, 0.5];
        assert!(
            (percentile(&v, 0.0) - 0.1).abs() < f32::EPSILON,
            "p=0 should return minimum 0.1, got {}",
            percentile(&v, 0.0)
        );
    }

    /// T-4c — `p = 1.0` returns the maximum (nearest-rank index N-1 after sort).
    #[test]
    fn percentile_p_one_returns_max() {
        let v = vec![0.7_f32, 0.2, 0.9, 0.1, 0.5];
        assert!(
            (percentile(&v, 1.0) - 0.9).abs() < f32::EPSILON,
            "p=1 should return maximum 0.9, got {}",
            percentile(&v, 1.0)
        );
    }

    // -- Phase 4 audit-fix R2 regression tests (T-7, T-8) -------------------
    //
    // Convergence-confirming regression tests pinning the R2 reviewer's
    // PSI hand-traces for two boundary cases not previously covered:
    //   T-7 = zero-overlap (observed labels disjoint from reference keys)
    //   T-8 = heavy concentration (1000-deep single-label observed matches a
    //         single-bucket reference)
    // Together they fence the symmetric-smoothed-disjoint upper bound
    // and the matched-singular near-zero floor.

    /// T-7 — Zero overlap. observed=["a","b"], reference={"c":0.5, "d":0.5}.
    /// Hand-trace: union={"a","b","c","d"}; each bucket contributes
    /// `(0.5 - eps) * ln(0.5/eps)` ≈ 4.258. Sum ≈ 17.03. The eps-floor
    /// caps growth — locking this near-17 result guards against any change
    /// to `eps` that would explode the disjoint-distribution PSI.
    #[test]
    fn psi_zero_overlap_distribution_returns_bounded_large_positive() {
        let mut ref_dist = std::collections::BTreeMap::new();
        ref_dist.insert("cat".to_string(), 0.5);
        ref_dist.insert("dog".to_string(), 0.5);
        let r = DriftReference {
            class_distribution: ref_dist,
        };
        let labels = vec!["wolf".to_string(), "fox".to_string()];
        let m = compute_drift_metrics(&[0.9, 0.8], 1, &labels, Some(&r));
        let psi = m
            .class_distribution_psi
            .expect("disjoint observed + reference ⇒ PSI must be Some");
        assert!(
            psi.is_finite(),
            "PSI must be finite under disjoint inputs, got {psi}"
        );
        assert!(
            psi > 10.0,
            "disjoint distribution PSI must be substantially positive, got {psi}"
        );
        assert!(
            psi < 25.0,
            "disjoint distribution PSI must be eps-bounded (~17 expected, got {psi})"
        );
    }

    /// T-8 — Heavy concentration. observed=["a"]×1000, reference={"a":1.0}.
    /// Hand-trace: floating-point accumulation of `1.0/1000.0 × 1000` produces
    /// `p ≈ 1.0` with bounded drift, never exits the eps-protected range;
    /// `q = 1.0` exactly; sum ≈ 0 (1e-12 magnitude). Locks: matched-bucket
    /// PSI is near-zero even at 1000-deep observation.
    #[test]
    fn psi_heavy_concentration_matched_returns_near_zero() {
        let mut ref_dist = std::collections::BTreeMap::new();
        ref_dist.insert("animal".to_string(), 1.0);
        let r = DriftReference {
            class_distribution: ref_dist,
        };
        let labels = vec!["animal".to_string(); 1000];
        let confidences = vec![0.9_f32; 1000];
        let m = compute_drift_metrics(&confidences, 1, &labels, Some(&r));
        let psi = m
            .class_distribution_psi
            .expect("matched single-bucket ⇒ PSI must be Some");
        assert!(psi.is_finite(), "PSI must be finite, got {psi}");
        assert!(
            psi.abs() < 1e-3,
            "1000-deep matched single-bucket PSI must be near-zero, got {psi}"
        );
    }
}
