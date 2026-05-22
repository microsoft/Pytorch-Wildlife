//! Detection summary statistics.

use std::collections::HashMap;

use sparrow_engine_types::DetectResult;

/// Accumulator for per-category aggregation during summarization.
struct CatAccum {
    count: usize,
    confidence_sum: f64,
    // Sentinels until the first non-NaN confidence lands; gated by
    // `non_nan_count` on finalization to avoid leaking +/-Inf when every
    // confidence in the category is NaN (mirrors the global non_nan_count
    // reset block in `summarize_detections`).
    confidence_min: f32,
    confidence_max: f32,
    non_nan_count: usize,
}

impl Default for CatAccum {
    fn default() -> Self {
        Self {
            count: 0,
            confidence_sum: 0.0,
            confidence_min: f32::INFINITY,
            confidence_max: f32::NEG_INFINITY,
            non_nan_count: 0,
        }
    }
}

/// Per-category statistics.
#[derive(Debug, Clone)]
pub struct CategoryStats {
    pub count: usize,
    pub confidence_mean: f32,
    // `confidence_min`/`confidence_max` added for Sparrow Studio parity
    // (per-species box-plot axes). See `docs/design/phase3.5/adrs/sparrow_parity.md`.
    pub confidence_min: f32,
    pub confidence_max: f32,
}

/// Summary of detection results across multiple images.
#[derive(Debug, Clone)]
pub struct DetectionSummary {
    pub total_images: usize,
    pub images_with_detections: usize,
    pub empty_images: usize,
    pub total_detections: usize,
    pub per_category: HashMap<String, CategoryStats>,
    pub confidence_min: f32,
    pub confidence_max: f32,
    pub confidence_mean: f32,
}

/// Compute summary statistics over a slice of detection results.
pub fn summarize_detections(results: &[DetectResult]) -> DetectionSummary {
    let total_images = results.len();
    let mut images_with_detections = 0usize;
    let mut total_detections = 0usize;
    let mut non_nan_count = 0usize;
    let mut confidence_sum = 0.0f64;
    let mut confidence_min = f32::INFINITY;
    let mut confidence_max = f32::NEG_INFINITY;
    let mut per_category_acc: HashMap<String, CatAccum> = HashMap::new();

    for r in results {
        if !r.detections.is_empty() {
            images_with_detections += 1;
        }
        total_detections += r.detections.len();
        for d in &r.detections {
            // Skip NaN confidences. They would propagate through sum → mean as
            // NaN and silently corrupt aggregate output. f32::min/max ignore
            // NaN already, but the sum path does not.
            if d.confidence.is_nan() {
                continue;
            }
            non_nan_count += 1;
            confidence_sum += d.confidence as f64;
            confidence_min = confidence_min.min(d.confidence);
            confidence_max = confidence_max.max(d.confidence);
            let entry = per_category_acc.entry(d.label.clone()).or_default();
            entry.count += 1;
            entry.confidence_sum += d.confidence as f64;
            entry.non_nan_count += 1;
            entry.confidence_min = entry.confidence_min.min(d.confidence);
            entry.confidence_max = entry.confidence_max.max(d.confidence);
        }
    }

    // Divisor for the mean excludes NaN contributions so the mean stays a plain
    // average of valid confidences. total_detections still counts NaN entries so
    // callers see the true population size.
    let confidence_mean = if non_nan_count > 0 {
        (confidence_sum / non_nan_count as f64) as f32
    } else {
        0.0
    };

    // Reset min/max sentinels when no non-NaN confidence reached the
    // .min()/.max() updates. Gating on `non_nan_count == 0` covers both
    // empty input AND the all-NaN case; the older `total_detections == 0`
    // guard missed the latter and leaked f32::INFINITY / NEG_INFINITY.
    if non_nan_count == 0 {
        confidence_min = 0.0;
        confidence_max = 0.0;
    }

    let per_category = per_category_acc
        .into_iter()
        .map(|(label, acc)| {
            // Gating mirrors the global non_nan_count reset block above:
            // when no non-NaN confidence reached the min/max updates, clear
            // the +/-Inf sentinels so downstream JSON/CSV serialization
            // stays strict-finite.
            let (confidence_min, confidence_max) = if acc.non_nan_count == 0 {
                (0.0, 0.0)
            } else {
                (acc.confidence_min, acc.confidence_max)
            };
            let confidence_mean = if acc.non_nan_count == 0 {
                0.0
            } else {
                (acc.confidence_sum / acc.non_nan_count as f64) as f32
            };
            (
                label,
                CategoryStats {
                    count: acc.count,
                    confidence_mean,
                    confidence_min,
                    confidence_max,
                },
            )
        })
        .collect();

    DetectionSummary {
        total_images,
        images_with_detections,
        empty_images: total_images - images_with_detections,
        total_detections,
        per_category,
        confidence_min,
        confidence_max,
        confidence_mean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::{BBox, Detection};

    fn make_result(detections: Vec<Detection>) -> DetectResult {
        DetectResult {
            detections,
            image_width: 640,
            image_height: 480,
            processing_time_ms: 10.0,
        }
    }

    fn make_det(label: &str, confidence: f32) -> Detection {
        Detection {
            bbox: BBox {
                x_min: 0.0,
                y_min: 0.0,
                x_max: 0.5,
                y_max: 0.5,
            },
            label: label.to_string(),
            label_id: 0,
            confidence,
        }
    }

    #[test]
    fn empty_results() {
        let summary = summarize_detections(&[]);
        assert_eq!(summary.total_images, 0);
        assert_eq!(summary.total_detections, 0);
        assert_eq!(summary.confidence_min, 0.0);
        assert_eq!(summary.confidence_max, 0.0);
        assert_eq!(summary.confidence_mean, 0.0);
    }

    #[test]
    fn single_image_no_detections() {
        let results = vec![make_result(vec![])];
        let summary = summarize_detections(&results);
        assert_eq!(summary.total_images, 1);
        assert_eq!(summary.images_with_detections, 0);
        assert_eq!(summary.empty_images, 1);
        assert_eq!(summary.total_detections, 0);
    }

    #[test]
    fn mixed_results() {
        let results = vec![
            make_result(vec![make_det("animal", 0.9), make_det("person", 0.8)]),
            make_result(vec![]),
            make_result(vec![make_det("animal", 0.7)]),
        ];
        let summary = summarize_detections(&results);
        assert_eq!(summary.total_images, 3);
        assert_eq!(summary.images_with_detections, 2);
        assert_eq!(summary.empty_images, 1);
        assert_eq!(summary.total_detections, 3);
        assert!((summary.confidence_min - 0.7).abs() < 1e-6);
        assert!((summary.confidence_max - 0.9).abs() < 1e-6);

        let animal = summary.per_category.get("animal").unwrap();
        assert_eq!(animal.count, 2);
        assert!((animal.confidence_mean - 0.8).abs() < 1e-5);
        assert!((animal.confidence_min - 0.7).abs() < 1e-6);
        assert!((animal.confidence_max - 0.9).abs() < 1e-6);

        let person = summary.per_category.get("person").unwrap();
        assert_eq!(person.count, 1);
        assert!((person.confidence_min - 0.8).abs() < 1e-6);
        assert!((person.confidence_max - 0.8).abs() < 1e-6);
    }

    // Sparrow Studio parity (S7): per-category confidence_min/max covers
    // the min/max whiskers of Sparrow's per-species confidence box plot
    // (`ResultsWindow.xaml.cs` speciesProbabilities[species] → OxyPlot BoxPlot).
    // See docs/design/phase3.5/adrs/sparrow_parity.md.
    #[test]
    fn per_category_confidence_min_max_tracks_distribution() {
        let results = vec![make_result(vec![
            make_det("deer", 0.55),
            make_det("deer", 0.92),
            make_det("deer", 0.70),
            make_det("fox", 0.40),
        ])];
        let summary = summarize_detections(&results);

        let deer = summary.per_category.get("deer").unwrap();
        assert_eq!(deer.count, 3);
        assert!((deer.confidence_min - 0.55).abs() < 1e-6);
        assert!((deer.confidence_max - 0.92).abs() < 1e-6);
        assert!((deer.confidence_mean - (0.55 + 0.92 + 0.70) / 3.0).abs() < 1e-5);

        let fox = summary.per_category.get("fox").unwrap();
        assert_eq!(fox.count, 1);
        assert!((fox.confidence_min - 0.40).abs() < 1e-6);
        assert!((fox.confidence_max - 0.40).abs() < 1e-6);
    }

    // Sparrow Studio parity (S7): per-category min/max must NOT leak
    // f32::INFINITY / NEG_INFINITY sentinels when a category has detections
    // but all confidences are NaN. Mirrors the global-level guard in
    // `all_nan_confidence_does_not_leak_sentinels` — the NaN-skip `continue`
    // at the match keeps this path unreachable today, but the finalization
    // gate is the invariant that keeps it unreachable tomorrow.
    #[test]
    fn per_category_all_nan_does_not_leak_sentinels() {
        // Mix: one category stays all-non-NaN, another is skipped entirely.
        // Today the `continue` at the top of the inner loop keeps all-NaN
        // categories out of the map (same as the existing
        // `all_nan_confidence_does_not_leak_sentinels` test). This test
        // pins that invariant AND the finalization gate behind it.
        let results = vec![make_result(vec![
            make_det("bird", 0.5),
            make_det("bird", f32::NAN),
            make_det("mystery", f32::NAN),
        ])];
        let summary = summarize_detections(&results);

        let bird = summary.per_category.get("bird").unwrap();
        assert!(
            bird.confidence_min.is_finite() && bird.confidence_max.is_finite(),
            "per-category min/max must stay finite when some confidences are NaN"
        );
        assert!((bird.confidence_min - 0.5).abs() < 1e-6);
        assert!((bird.confidence_max - 0.5).abs() < 1e-6);

        assert!(
            !summary.per_category.contains_key("mystery"),
            "all-NaN category must not leak as an entry in per_category"
        );
    }

    // Regression (V3): NaN confidences must not corrupt aggregate stats.
    // - confidence_sum must skip NaN (NaN propagates through f64 arithmetic).
    // - mean divisor must skip NaN (otherwise mean is biased low).
    // - total_detections still counts NaN (it is the true detection population).
    // - per_category count excludes NaN (category stats stay meaningful).
    // - f32::min / f32::max already ignore NaN per std, so min/max are safe.
    #[test]
    fn nan_confidence_does_not_corrupt_stats() {
        let results = vec![make_result(vec![
            make_det("animal", 0.9),
            make_det("animal", f32::NAN),
            make_det("animal", 0.7),
        ])];
        let summary = summarize_detections(&results);

        assert_eq!(summary.total_detections, 3, "NaN still counted in total");
        assert!(!summary.confidence_mean.is_nan(), "mean must not be NaN");
        assert!(
            (summary.confidence_mean - 0.8).abs() < 1e-5,
            "mean = (0.9 + 0.7) / 2 = 0.8, got {}",
            summary.confidence_mean
        );
        assert!((summary.confidence_min - 0.7).abs() < 1e-6);
        assert!((summary.confidence_max - 0.9).abs() < 1e-6);

        let animal = summary.per_category.get("animal").unwrap();
        assert_eq!(animal.count, 2, "per-category count excludes NaN");
        assert!((animal.confidence_mean - 0.8).abs() < 1e-5);
    }

    // Regression (MI-1, R4): when every confidence is NaN but detections exist,
    // the reset guard must still clear the f32::INFINITY / NEG_INFINITY sentinels.
    // Gating on `non_nan_count == 0` (not `total_detections == 0`) covers this case.
    // Without the fix, `confidence_min` leaks as +Inf and `confidence_max` as -Inf,
    // which then fails strict JSON encoding in downstream consumers.
    #[test]
    fn all_nan_confidence_does_not_leak_sentinels() {
        let results = vec![make_result(vec![
            make_det("animal", f32::NAN),
            make_det("animal", f32::NAN),
        ])];
        let summary = summarize_detections(&results);

        assert_eq!(summary.total_detections, 2, "NaN still counted in total");
        assert_eq!(summary.confidence_min, 0.0, "min must reset when all NaN");
        assert_eq!(summary.confidence_max, 0.0, "max must reset when all NaN");
        assert!(
            !summary.confidence_min.is_infinite(),
            "min must not leak f32::INFINITY sentinel"
        );
        assert!(
            !summary.confidence_max.is_infinite(),
            "max must not leak f32::NEG_INFINITY sentinel"
        );
        assert_eq!(summary.confidence_mean, 0.0, "mean must be 0 when all NaN");
        assert!(
            summary.per_category.is_empty(),
            "per_category must be empty — continue at :54 also skips per-category accumulation"
        );
    }
}

#[cfg(test)]
mod phase_a_r1_stats {
    use super::*;
    use sparrow_engine_types::{BBox, Detection};

    fn det(label: &str, conf: f32) -> Detection {
        Detection {
            bbox: BBox {
                x_min: 0.0,
                y_min: 0.0,
                x_max: 0.5,
                y_max: 0.5,
            },
            label: label.to_string(),
            label_id: 0,
            confidence: conf,
        }
    }

    fn result(dets: Vec<Detection>) -> sparrow_engine_types::DetectResult {
        sparrow_engine_types::DetectResult {
            detections: dets,
            image_width: 640,
            image_height: 480,
            processing_time_ms: 10.0,
        }
    }

    /// Empty input => zero counters and an empty `per_category` map. Pins the
    /// expected `DetectionSummary` shape on a no-input call (existing
    /// `empty_results` test only checks 4 fields; this widens to 7).
    #[test]
    fn empty_results_yields_zero_summary() {
        let s = summarize_detections(&[]);
        assert_eq!(s.total_images, 0);
        assert_eq!(s.images_with_detections, 0);
        assert_eq!(s.empty_images, 0);
        assert_eq!(s.total_detections, 0);
        assert_eq!(s.confidence_min, 0.0);
        assert_eq!(s.confidence_max, 0.0);
        assert_eq!(s.confidence_mean, 0.0);
        assert!(
            s.per_category.is_empty(),
            "empty input must produce empty per_category map"
        );
    }

    /// Single image with zero detections: `images_with_detections == 0`,
    /// `empty_images == 1`, `per_category` empty. Distinguishes "no images"
    /// from "one empty image" — the existing `single_image_no_detections`
    /// asserts the counts but not that `per_category` stays empty.
    #[test]
    fn single_empty_image_keeps_per_category_empty() {
        let s = summarize_detections(&[result(vec![])]);
        assert_eq!(s.total_images, 1);
        assert_eq!(s.empty_images, 1);
        assert_eq!(s.images_with_detections, 0);
        assert!(
            s.per_category.is_empty(),
            "no detections => no category entries"
        );
    }

    /// Multi-image batch where the same label appears across different
    /// `DetectResult`s. Verifies that per-category aggregation accumulates
    /// across images, not just within a single image. Sums must match
    /// hand-computed totals.
    #[test]
    fn overlapping_labels_aggregate_across_images() {
        let results = vec![
            result(vec![det("deer", 0.9), det("fox", 0.6)]),
            result(vec![det("deer", 0.5)]),
            result(vec![det("deer", 0.7), det("fox", 0.8)]),
        ];
        let s = summarize_detections(&results);
        assert_eq!(s.total_images, 3);
        assert_eq!(s.total_detections, 5);

        let deer = s.per_category.get("deer").expect("deer must be present");
        assert_eq!(deer.count, 3, "deer detected in all three images");
        assert!((deer.confidence_min - 0.5).abs() < 1e-6);
        assert!((deer.confidence_max - 0.9).abs() < 1e-6);
        assert!(
            (deer.confidence_mean - (0.9 + 0.5 + 0.7) / 3.0).abs() < 1e-5,
            "deer mean must average across images, got {}",
            deer.confidence_mean
        );

        let fox = s.per_category.get("fox").expect("fox must be present");
        assert_eq!(fox.count, 2);
        assert!((fox.confidence_mean - 0.7).abs() < 1e-5);
    }
}
