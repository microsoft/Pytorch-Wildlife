//! GPU audio detection inference.
//!
//! Mirrors `sparrow_engine_cpu::detect_audio`'s surface. The top-level
//! [`detect_audio`] / [`detect_audio_streaming`] free fns take a
//! [`ModelHandle`] and route to
//! [`crate::models::audio::AudioModel::detect`] /
//! [`crate::models::audio::AudioModel::detect_streaming`]. The
//! `sparrow_engine_types::AudioDetectOpts` (public) is wrapped into the GPU-side
//! [`crate::models::audio::GpuAudioDetectOpts`] with the
//! `default_strategy()` (`SingleCall`) for non-streaming and
//! `default_strategy_streaming()` (`HybridA{16}`) for streaming.
//!
//! Streaming-callback divergence (see
//! [`crate::models::audio::AudioModel::detect_streaming`] docstring):
//! callbacks fire post-detect on the GPU side because the audio
//! workspace mutex is held for the duration of the chunk loop. Phase C
//! consumer wiring decides whether to expose the divergence to the HTTP
//! / Python streaming surface.
//!
//! Re-exports `sparrow_engine_types::AudioRange` so consumers can use
//! `sparrow_engine_gpu::detect_audio::AudioRange` symmetric to
//! `engine_dispatch::detect_audio::AudioRange`.

use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{ModelManifest, PreprocessMethod};
use sparrow_engine_types::types::{AudioDetectOpts, AudioDetectResult, AudioInput, AudioSegment};

use crate::engine::{LoadedModelInner, ModelHandle};
use crate::models::audio::GpuAudioDetectOpts;

/// Public re-export so consumers can refer to `sparrow_engine_gpu::detect_audio::AudioRange`.
pub use sparrow_engine_types::AudioRange;

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that a manifest represents an audio model that the GPU flavor
/// supports. Phase D round 2 B-08: both `MelSpectrogram` and `RawAudio`
/// are accepted (raw audio routes through the parallel
/// [`crate::models::audio_raw::RawAudioModel`]).
pub(crate) fn validate_audio_model(manifest: &ModelManifest) -> Result<()> {
    match &manifest.preprocess_method {
        PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. } => Ok(()),
        other => Err(SparrowEngineError::NotAnAudioModel {
            id: manifest.id.clone(),
            method: other.as_str().to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run audio detection inference with sliding window.
///
/// # Errors
/// - [`SparrowEngineError::NotAnAudioModel`] if the model isn't audio.
/// - [`SparrowEngineError::ModelUnloaded`] / [`SparrowEngineError::EngineFreed`] if the
///   handle is invalid.
/// - [`SparrowEngineError::Ort`] on GPU pipeline / ORT runtime errors.
pub fn detect_audio(
    handle: &ModelHandle,
    audio: &AudioInput,
    opts: &AudioDetectOpts,
) -> Result<AudioDetectResult> {
    let inner = handle.pin_inner()?;
    validate_audio_model(&inner.manifest)?;
    match &inner.inner {
        LoadedModelInner::Audio(model) => {
            let gpu_opts = GpuAudioDetectOpts {
                base: opts.clone(),
                strategy: GpuAudioDetectOpts::default_strategy(),
            };
            model.detect(audio, &gpu_opts)
        }
        LoadedModelInner::AudioRaw(model) => model.detect(audio, opts, &inner.labels),
        _ => Err(SparrowEngineError::NotAnAudioModel {
            id: inner.manifest.id.clone(),
            method: inner.manifest.preprocess_method.as_str().to_string(),
        }),
    }
}

/// Run audio detection with a per-segment callback.
///
/// **Callback cadence**: this routes through
/// [`crate::models::audio::AudioModel::detect_streaming`], which fires
/// callbacks post-detect (after the full chunk loop completes) rather
/// than per-batch. Consumers that need per-batch cadence should use
/// `sparrow_engine_cpu::detect_audio::detect_audio_streaming` (CPU flavor) or
/// issue smaller-segment detect calls.
pub fn detect_audio_streaming(
    handle: &ModelHandle,
    audio: &AudioInput,
    opts: &AudioDetectOpts,
    on_segment: impl FnMut(&AudioSegment),
) -> Result<AudioDetectResult> {
    let inner = handle.pin_inner()?;
    validate_audio_model(&inner.manifest)?;
    match &inner.inner {
        LoadedModelInner::Audio(model) => {
            let gpu_opts = GpuAudioDetectOpts {
                base: opts.clone(),
                strategy: GpuAudioDetectOpts::default_strategy_streaming(),
            };
            model.detect_streaming(audio, &gpu_opts, on_segment)
        }
        LoadedModelInner::AudioRaw(model) => {
            model.detect_streaming(audio, opts, &inner.labels, on_segment)
        }
        _ => Err(SparrowEngineError::NotAnAudioModel {
            id: inner.manifest.id.clone(),
            method: inner.manifest.preprocess_method.as_str().to_string(),
        }),
    }
}

// ---------------------------------------------------------------------------
// Segment merging — pure CPU postprocess, identical to sparrow-engine-cpu.
// ---------------------------------------------------------------------------

/// Merge contiguous + same-class audio segments into wider
/// [`AudioRange`] entries. Verbatim from sparrow-engine-cpu (no GPU dependency).
pub fn merge_segments(segments: &[AudioSegment], gap_s: f32) -> Vec<AudioRange> {
    merge_segments_with_class(segments, gap_s, |_| None)
}

/// Like [`merge_segments`] but with a caller-supplied class mapper.
pub fn merge_segments_with_class<F>(
    segments: &[AudioSegment],
    gap_s: f32,
    class_of: F,
) -> Vec<AudioRange>
where
    F: Fn(&AudioSegment) -> Option<String>,
{
    let mut ranges: Vec<AudioRange> = Vec::new();
    for seg in segments {
        let class = class_of(seg);
        if let Some(last) = ranges.last_mut() {
            let same_class = last.class == class;
            let gap = seg.start_time_s - last.end_time_s;
            if same_class && gap < gap_s {
                if seg.end_time_s > last.end_time_s {
                    last.end_time_s = seg.end_time_s;
                }
                if seg.confidence > last.max_confidence {
                    last.max_confidence = seg.confidence;
                }
                continue;
            }
        }
        ranges.push(AudioRange {
            start_time_s: seg.start_time_s,
            end_time_s: seg.end_time_s,
            max_confidence: seg.confidence,
            class,
        });
    }
    ranges
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::manifest::{
        InferenceStrategy, Layout, ModelManifest, Normalization, PostprocessMethod, Precision,
    };
    use sparrow_engine_types::types::ModelSubtype;

    fn yolo_like_manifest() -> ModelManifest {
        ModelManifest {
            id: "fake_yolo".into(),
            format: "onnx".into(),
            model_file: "model.onnx".into(),
            model_file_fp16: None,
            preprocess_method: PreprocessMethod::Letterbox,
            input_size: Some([640, 640]),
            layout: Some(Layout::Nchw),
            normalization: Some(Normalization::Unit),
            pad_value: Some(114.0),
            channel_order: None,
            precision: Precision::Fp32,
            inference_strategy: InferenceStrategy::Single,
            trt: None,
            postprocess_method: PostprocessMethod::YoloE2e,
            confidence_threshold: None,
            label_file: None,
            label_format: None,
            default: false,
            subtype: ModelSubtype::Standard,
            onnx_sha256: None,
            onnx_size_bytes: None,
            version: None,
            description: None,
            provenance: None,
            drift_reference: None,
        }
    }

    #[test]
    fn validate_audio_model_rejects_image_manifest() {
        let m = yolo_like_manifest();
        let err = validate_audio_model(&m).unwrap_err();
        assert!(matches!(err, SparrowEngineError::NotAnAudioModel { .. }));
    }

    #[test]
    fn merge_segments_groups_contiguous() {
        let segs = vec![
            AudioSegment {
                start_time_s: 0.0,
                end_time_s: 1.0,
                confidence: 0.5,
                classes: Vec::new(),
            },
            AudioSegment {
                start_time_s: 1.0,
                end_time_s: 2.0,
                confidence: 0.7,
                classes: Vec::new(),
            },
            AudioSegment {
                start_time_s: 5.0,
                end_time_s: 6.0,
                confidence: 0.6,
                classes: Vec::new(),
            },
        ];
        let ranges = merge_segments(&segs, 0.5);
        assert_eq!(ranges.len(), 2);
        assert_eq!(ranges[0].start_time_s, 0.0);
        assert_eq!(ranges[0].end_time_s, 2.0);
        assert!((ranges[0].max_confidence - 0.7).abs() < 1e-6);
        assert_eq!(ranges[1].start_time_s, 5.0);
    }
}
