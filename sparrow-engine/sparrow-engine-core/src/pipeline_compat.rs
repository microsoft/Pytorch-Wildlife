//! Pipeline compatibility validator.
//!
//! This module is the code-defined Phase 4.2 compatibility contract for
//! ad-hoc detector + classifier composition. It stays in sparrow-engine-core so both
//! CPU and GPU flavors re-export the same logic.

use sparrow_engine_types::{SparrowEngineError, ModelType, Result};

/// Match pattern used by [`PIPELINE_COMPAT_MATRIX`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelTypePattern {
    None,
    Some(ModelType),
    Any,
}

impl ModelTypePattern {
    const fn matches(self, value: Option<ModelType>) -> bool {
        match (self, value) {
            (Self::Any, _) => true,
            (Self::None, None) => true,
            (Self::Some(expected), Some(actual)) => expected as u8 == actual as u8,
            _ => false,
        }
    }
}

/// One row in the Phase 4.2 compatibility matrix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PipelineCompatRule {
    pub detector: ModelTypePattern,
    pub classifier: ModelTypePattern,
    pub compatible: bool,
    pub reason: &'static str,
}

impl PipelineCompatRule {
    const fn matches(self, detector: Option<ModelType>, classifier: Option<ModelType>) -> bool {
        self.detector.matches(detector) && self.classifier.matches(classifier)
    }
}

/// Phase 4.2 pipeline compatibility matrix.
///
/// AudioClassifier combinations are intentionally not included; the current
/// model zoo has no true audio-classifier chaining path.
pub const PIPELINE_COMPAT_MATRIX: &[PipelineCompatRule] = &[
    PipelineCompatRule {
        detector: ModelTypePattern::Some(ModelType::Detector),
        classifier: ModelTypePattern::Some(ModelType::Classifier),
        compatible: true,
        reason: "bbox crop → image classify",
    },
    PipelineCompatRule {
        detector: ModelTypePattern::Some(ModelType::Detector),
        classifier: ModelTypePattern::None,
        compatible: true,
        reason: "detect-only",
    },
    PipelineCompatRule {
        detector: ModelTypePattern::None,
        classifier: ModelTypePattern::Some(ModelType::Classifier),
        compatible: true,
        reason: "classify full image — degenerate 1-step pipeline",
    },
    PipelineCompatRule {
        detector: ModelTypePattern::Some(ModelType::OverheadDetector),
        classifier: ModelTypePattern::Some(ModelType::Classifier),
        compatible: false,
        reason: "point detection produces dot, no crop",
    },
    PipelineCompatRule {
        detector: ModelTypePattern::Some(ModelType::OverheadDetector),
        classifier: ModelTypePattern::None,
        compatible: true,
        reason: "overhead-only (HerdNet)",
    },
    PipelineCompatRule {
        detector: ModelTypePattern::Some(ModelType::AudioDetector),
        classifier: ModelTypePattern::Some(ModelType::Classifier),
        compatible: false,
        reason: "modality mismatch",
    },
    PipelineCompatRule {
        detector: ModelTypePattern::Some(ModelType::AudioDetector),
        classifier: ModelTypePattern::None,
        compatible: true,
        reason: "audio-only (MD_AudioBirds_V1 used standalone)",
    },
    PipelineCompatRule {
        detector: ModelTypePattern::Some(ModelType::Classifier),
        classifier: ModelTypePattern::Any,
        compatible: false,
        reason: "classifier cannot produce regions",
    },
    PipelineCompatRule {
        detector: ModelTypePattern::None,
        classifier: ModelTypePattern::None,
        compatible: false,
        reason: "empty pipeline",
    },
];

/// Validate that a detector/classifier pair can be composed into a pipeline.
pub fn validate_pipeline_compat(
    detector: Option<ModelType>,
    classifier: Option<ModelType>,
) -> Result<()> {
    if detector == Some(ModelType::AudioClassifier)
        || classifier == Some(ModelType::AudioClassifier)
    {
        return Err(SparrowEngineError::IncompatiblePipeline {
            detector,
            classifier,
            reason: "audio classifiers are not supported by the Phase 4.2 pipeline matrix",
        });
    }

    if let Some(rule) = PIPELINE_COMPAT_MATRIX
        .iter()
        .copied()
        .find(|rule| rule.matches(detector, classifier))
    {
        if rule.compatible {
            return Ok(());
        }
        if detector.is_none() && classifier.is_none() {
            return Err(SparrowEngineError::EmptyPipeline);
        }
        return Err(SparrowEngineError::IncompatiblePipeline {
            detector,
            classifier,
            reason: rule.reason,
        });
    }

    Err(SparrowEngineError::IncompatiblePipeline {
        detector,
        classifier,
        reason: "classifier slot must be an image classifier and detector slot must be a detector",
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_ok(detector: Option<ModelType>, classifier: Option<ModelType>) {
        validate_pipeline_compat(detector, classifier).expect("expected compatible pipeline");
    }

    fn assert_incompatible(
        detector: Option<ModelType>,
        classifier: Option<ModelType>,
        expected_reason: &str,
    ) {
        let err = validate_pipeline_compat(detector, classifier).unwrap_err();
        match err {
            SparrowEngineError::IncompatiblePipeline { reason, .. } => {
                assert!(
                    reason.contains(expected_reason),
                    "expected reason containing {expected_reason:?}, got {reason:?}"
                );
            }
            other => panic!("expected IncompatiblePipeline, got {other:?}"),
        }
    }

    #[test]
    fn validate_pipeline_compat_covers_phase_4_2_matrix() {
        let expected = [
            PipelineCompatRule {
                detector: ModelTypePattern::Some(ModelType::Detector),
                classifier: ModelTypePattern::Some(ModelType::Classifier),
                compatible: true,
                reason: "bbox crop → image classify",
            },
            PipelineCompatRule {
                detector: ModelTypePattern::Some(ModelType::Detector),
                classifier: ModelTypePattern::None,
                compatible: true,
                reason: "detect-only",
            },
            PipelineCompatRule {
                detector: ModelTypePattern::None,
                classifier: ModelTypePattern::Some(ModelType::Classifier),
                compatible: true,
                reason: "classify full image — degenerate 1-step pipeline",
            },
            PipelineCompatRule {
                detector: ModelTypePattern::Some(ModelType::OverheadDetector),
                classifier: ModelTypePattern::Some(ModelType::Classifier),
                compatible: false,
                reason: "point detection produces dot, no crop",
            },
            PipelineCompatRule {
                detector: ModelTypePattern::Some(ModelType::OverheadDetector),
                classifier: ModelTypePattern::None,
                compatible: true,
                reason: "overhead-only (HerdNet)",
            },
            PipelineCompatRule {
                detector: ModelTypePattern::Some(ModelType::AudioDetector),
                classifier: ModelTypePattern::Some(ModelType::Classifier),
                compatible: false,
                reason: "modality mismatch",
            },
            PipelineCompatRule {
                detector: ModelTypePattern::Some(ModelType::AudioDetector),
                classifier: ModelTypePattern::None,
                compatible: true,
                reason: "audio-only (MD_AudioBirds_V1 used standalone)",
            },
            PipelineCompatRule {
                detector: ModelTypePattern::Some(ModelType::Classifier),
                classifier: ModelTypePattern::Any,
                compatible: false,
                reason: "classifier cannot produce regions",
            },
            PipelineCompatRule {
                detector: ModelTypePattern::None,
                classifier: ModelTypePattern::None,
                compatible: false,
                reason: "empty pipeline",
            },
        ];
        assert_eq!(PIPELINE_COMPAT_MATRIX, expected.as_slice());

        assert_ok(Some(ModelType::Detector), Some(ModelType::Classifier));
        assert_ok(Some(ModelType::Detector), None);
        assert_ok(None, Some(ModelType::Classifier));
        assert_incompatible(
            Some(ModelType::OverheadDetector),
            Some(ModelType::Classifier),
            "point detection",
        );
        assert_ok(Some(ModelType::OverheadDetector), None);
        assert_incompatible(
            Some(ModelType::AudioDetector),
            Some(ModelType::Classifier),
            "modality mismatch",
        );
        assert_ok(Some(ModelType::AudioDetector), None);
        assert_incompatible(
            Some(ModelType::Classifier),
            Some(ModelType::Classifier),
            "classifier cannot produce regions",
        );
        assert_incompatible(
            Some(ModelType::Classifier),
            None,
            "classifier cannot produce regions",
        );

        let err = validate_pipeline_compat(None, None).unwrap_err();
        assert!(matches!(err, SparrowEngineError::EmptyPipeline));
    }

    #[test]
    fn validate_pipeline_compat_rejects_bad_slot_types() {
        assert_incompatible(
            Some(ModelType::Detector),
            Some(ModelType::Detector),
            "classifier slot",
        );
        assert_incompatible(None, Some(ModelType::AudioDetector), "classifier slot");
    }

    #[test]
    fn validate_pipeline_compat_rejects_audio_classifier_combinations() {
        assert_incompatible(
            Some(ModelType::AudioDetector),
            Some(ModelType::AudioClassifier),
            "audio classifiers are not supported",
        );
        assert_incompatible(
            Some(ModelType::AudioClassifier),
            None,
            "audio classifiers are not supported",
        );
    }
}
