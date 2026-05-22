//! Model-type derivation from preprocess + postprocess + subtype hints.
//!
//! Moved from the legacy monolithic engine crate for Phase 3.8 Phase A crate split.
//! Pure function over enum types from `crate::manifest` and `crate::types`.
//! C2 closure: visibility flipped from `pub(crate) fn` (legacy-monolith-only) to
//! `pub fn` so the workspace re-export `pub use sparrow_engine_types::*;` keeps it
//! reachable from external consumers (CLI, Python, server).

use crate::manifest::{PostprocessMethod, PreprocessMethod};
use crate::types::{ModelSubtype, ModelType};

/// Derive model type from preprocessing + postprocessing method + subtype.
///
/// `subtype` is the manifest's `[model].subtype` hint (Phase 3.5 S3, MT-9):
/// when `Overhead` and the pre/post combination resolves to `Detector`, the
/// result is promoted to `OverheadDetector` so viz dispatches to the centroid
/// dot path instead of the bbox path. `Overhead` is ignored for classifiers
/// and audio models — it only affects vision detection rendering.
pub fn derive_model_type(
    preprocess: &PreprocessMethod,
    postprocess: &PostprocessMethod,
    subtype: ModelSubtype,
) -> ModelType {
    let base = match (preprocess, postprocess) {
        (PreprocessMethod::MelSpectrogram { .. }, PostprocessMethod::Sigmoid { .. }) => {
            ModelType::AudioDetector
        }
        (PreprocessMethod::MelSpectrogram { .. }, PostprocessMethod::Softmax) => {
            ModelType::AudioClassifier
        }
        (_, PostprocessMethod::Softmax) => ModelType::Classifier,
        _ => ModelType::Detector,
    };
    // Subtype promotion: only a vision Detector is eligible for overhead-dot
    // rendering. Audio + Classifier ignore the hint.
    match (base, subtype) {
        (ModelType::Detector, ModelSubtype::Overhead) => ModelType::OverheadDetector,
        _ => base,
    }
}

#[cfg(test)]
mod phase_a_r1_model_type_tests {
    use super::*;
    use crate::manifest::{PostprocessMethod, PreprocessMethod};
    use crate::types::{ModelSubtype, ModelType};

    /// Canonical placeholder MelSpectrogram constructor. All fields are required
    /// (no Default for enum variants), so we centralize it for reuse.
    fn mel() -> PreprocessMethod {
        PreprocessMethod::MelSpectrogram {
            sample_rate: 22050,
            n_fft: 1024,
            hop_length: 256,
            n_mels: 64,
            fmin: 50.0,
            fmax: 10000.0,
            top_db: 80.0,
            window: "hann".to_string(),
            mel_scale: "slaney".to_string(),
            filter_norm: "slaney".to_string(),
        }
    }

    fn heatmap() -> PostprocessMethod {
        PostprocessMethod::HeatmapPeaks {
            peak_threshold: 0.5,
            adaptive: false,
            point_to_box_half_size: 16,
        }
    }

    fn sigmoid() -> PostprocessMethod {
        PostprocessMethod::Sigmoid {
            confidence_threshold: 0.5,
        }
    }

    #[test]
    fn audio_detector_when_mel_plus_sigmoid_standard() {
        assert_eq!(
            derive_model_type(&mel(), &sigmoid(), ModelSubtype::Standard),
            ModelType::AudioDetector
        );
    }

    #[test]
    fn audio_detector_when_mel_plus_sigmoid_overhead_does_not_promote() {
        // Overhead hint must NOT promote audio models — only vision Detectors.
        assert_eq!(
            derive_model_type(&mel(), &sigmoid(), ModelSubtype::Overhead),
            ModelType::AudioDetector
        );
    }

    #[test]
    fn audio_classifier_when_mel_plus_softmax_either_subtype() {
        assert_eq!(
            derive_model_type(&mel(), &PostprocessMethod::Softmax, ModelSubtype::Standard),
            ModelType::AudioClassifier
        );
        // Overhead must NOT promote audio classifier.
        assert_eq!(
            derive_model_type(&mel(), &PostprocessMethod::Softmax, ModelSubtype::Overhead),
            ModelType::AudioClassifier
        );
    }

    #[test]
    fn mel_plus_yolo_falls_through_to_detector() {
        // Per code: only (Mel,Sigmoid)+(Mel,Softmax) match audio arms; everything
        // else with Mel falls through to the wildcard Detector arm.
        assert_eq!(
            derive_model_type(&mel(), &PostprocessMethod::YoloE2e, ModelSubtype::Standard),
            ModelType::Detector
        );
        // Even more: with Overhead subtype, that base Detector promotes to OverheadDetector.
        assert_eq!(
            derive_model_type(&mel(), &PostprocessMethod::YoloE2e, ModelSubtype::Overhead),
            ModelType::OverheadDetector
        );
    }

    #[test]
    fn classifier_when_softmax_with_image_preprocess() {
        // Non-Mel + Softmax → Classifier. Both image preprocess methods.
        assert_eq!(
            derive_model_type(
                &PreprocessMethod::Letterbox,
                &PostprocessMethod::Softmax,
                ModelSubtype::Standard
            ),
            ModelType::Classifier
        );
        assert_eq!(
            derive_model_type(
                &PreprocessMethod::Resize,
                &PostprocessMethod::Softmax,
                ModelSubtype::Standard
            ),
            ModelType::Classifier
        );
    }

    #[test]
    fn classifier_overhead_does_not_promote() {
        // Subtype promotion only fires for vision Detector base. Classifier must
        // remain Classifier even with Overhead hint.
        assert_eq!(
            derive_model_type(
                &PreprocessMethod::Letterbox,
                &PostprocessMethod::Softmax,
                ModelSubtype::Overhead
            ),
            ModelType::Classifier
        );
    }

    #[test]
    fn detector_for_image_preprocess_plus_non_softmax_postprocess() {
        // Letterbox + {YoloE2e, MegadetV5a, HeatmapPeaks, Sigmoid} → Detector when Standard.
        let postprocess_methods: Vec<PostprocessMethod> = vec![
            PostprocessMethod::YoloE2e,
            PostprocessMethod::MegadetV5a { iou_threshold: 0.45 },
            heatmap(),
            sigmoid(),
        ];
        for pp in &postprocess_methods {
            assert_eq!(
                derive_model_type(&PreprocessMethod::Letterbox, pp, ModelSubtype::Standard),
                ModelType::Detector,
                "Letterbox + {} should be Detector",
                pp.as_str()
            );
            assert_eq!(
                derive_model_type(&PreprocessMethod::Resize, pp, ModelSubtype::Standard),
                ModelType::Detector,
                "Resize + {} should be Detector",
                pp.as_str()
            );
        }
    }

    #[test]
    fn overhead_detector_when_image_detector_combo_with_overhead_subtype() {
        // Same matrix as above but Overhead subtype must promote to OverheadDetector.
        let postprocess_methods: Vec<PostprocessMethod> = vec![
            PostprocessMethod::YoloE2e,
            PostprocessMethod::MegadetV5a { iou_threshold: 0.45 },
            heatmap(),
            sigmoid(),
        ];
        for pp in &postprocess_methods {
            assert_eq!(
                derive_model_type(&PreprocessMethod::Letterbox, pp, ModelSubtype::Overhead),
                ModelType::OverheadDetector,
                "Letterbox + {} + Overhead should be OverheadDetector",
                pp.as_str()
            );
            assert_eq!(
                derive_model_type(&PreprocessMethod::Resize, pp, ModelSubtype::Overhead),
                ModelType::OverheadDetector,
                "Resize + {} + Overhead should be OverheadDetector",
                pp.as_str()
            );
        }
    }

    #[test]
    fn cartesian_full_matrix_no_panic_and_no_unknown_variants() {
        // Exhaustive cartesian: 3 preprocess × 5 postprocess × 2 subtype = 30 combos.
        // The point of this test is twofold:
        //   1) every combo derives without panicking,
        //   2) every result is one of the 5 known ModelType variants (sanity for refactor regressions).
        let preprocesses: Vec<PreprocessMethod> = vec![
            PreprocessMethod::Letterbox,
            PreprocessMethod::Resize,
            mel(),
        ];
        let postprocesses: Vec<PostprocessMethod> = vec![
            PostprocessMethod::YoloE2e,
            PostprocessMethod::MegadetV5a { iou_threshold: 0.45 },
            heatmap(),
            PostprocessMethod::Softmax,
            sigmoid(),
        ];
        let subtypes: [ModelSubtype; 2] = [ModelSubtype::Standard, ModelSubtype::Overhead];

        let mut combo_count = 0;
        for pre in &preprocesses {
            for post in &postprocesses {
                for sub in &subtypes {
                    let mt = derive_model_type(pre, post, *sub);
                    let known = matches!(
                        mt,
                        ModelType::Detector
                            | ModelType::OverheadDetector
                            | ModelType::Classifier
                            | ModelType::AudioDetector
                            | ModelType::AudioClassifier
                    );
                    assert!(known, "unknown ModelType returned: {mt:?}");
                    combo_count += 1;
                }
            }
        }
        assert_eq!(combo_count, 3 * 5 * 2);
    }
}
