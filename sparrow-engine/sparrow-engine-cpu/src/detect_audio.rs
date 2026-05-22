//! Audio detection inference: sliding window over audio input.
//!
//! Orchestrates: load audio -> resample -> sliding window -> per-segment mel
//! spectrogram + ORT inference -> collect segments above threshold.

use std::time::Instant;

use ndarray::ArrayViewD;
use ort::value::TensorRef;
// Phase 3.8 Step 2 Wave 0b: per-stage `tracing::info!` timings (the workspace
// `tracing` dep is declared unconditional in sparrow-engine-cpu/Cargo.toml since Phase
// A). The bench harness in `scripts/bench_audio_breakdown.py` consumes these
// as `stage = "audio.<stage>" duration_ns = N` events. No new dep.

use crate::engine::ModelHandle;
use crate::error::{SparrowEngineError, Result};
use crate::manifest::{InferenceStrategy, PostprocessMethod, PreprocessMethod};
use crate::preprocess_audio;
use crate::types::{AudioDetectOpts, AudioDetectResult, AudioInput, AudioSegment};

// ---------------------------------------------------------------------------
// Merged-range output type (Phase 3.5 S5 / item #6)
// ---------------------------------------------------------------------------

/// A merged range of consecutive audio detections (see [`merge_segments`]).
///
/// Introduced in Phase 3.5 S5 (item #6). Raw `detect_audio` output tends
/// to produce one [`AudioSegment`] per sliding window (~198 for a 60 s
/// recording at 1.0 s window, 0.3 s stride), most at `confidence ≈ 1.0`.
/// `AudioRange` collapses consecutive above-threshold windows into a
/// single time range with the maximum observed confidence, giving a
/// ~10x–100x reduction for typical recordings.
///
/// `class` is reserved for future multiclass audio models; for binary
/// detectors (MD_AudioBirds_V1, the Phase 1 default) it is always `None`.
///
/// Phase 3.8 Phase A: hoisted to `sparrow-engine-types` (Commit 2 widening) because
/// `sparrow-engine-core::viz::render_range_overlay` consumes it in its public API
/// and sparrow-engine-core cannot reach into sparrow-engine-cpu. Re-exported here for
/// consumer back-compat — `sparrow_engine::detect_audio::AudioRange` keeps
/// resolving for sparrow-engine-cli + integration tests (lib name is now
/// "sparrow_engine" after the R2 rename).
pub use sparrow_engine_types::AudioRange;

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Run audio detection inference with sliding window.
///
/// Loads audio, resamples to the model's target sample rate, splits into
/// overlapping segments, computes mel spectrogram per segment, runs ORT
/// inference, and collects segments with confidence above threshold.
///
/// # Errors
/// - `NotAnAudioModel` if the model doesn't use mel spectrogram preprocessing
/// - `ModelUnloaded` if the handle has been invalidated — also surfaces if the
///   engine itself has been dropped (post-S1 MT-17 mitigation: `Drop for Engine`
///   in `engine.rs` leaks `Arc<EngineInner>` so `Weak::upgrade()` keeps
///   succeeding; the signal the handle actually sees is the per-model `active`
///   flag that `Drop` clears before releasing sessions — see `docs/bugs.md`
///   MT-17 for the full rationale).
/// - `EngineFreed` reserved for pre-Drop paths (e.g. `Engine::unload_model`).
/// - `Ort` on ORT runtime errors
pub fn detect_audio(
    handle: &ModelHandle,
    audio: &AudioInput,
    opts: &AudioDetectOpts,
) -> Result<AudioDetectResult> {
    let start = Instant::now();
    let prep = prepare_audio_detection(handle, audio, opts)?;
    detect_audio_loop(handle, &prep, start, None)
}

/// Run audio detection with a per-segment callback.
///
/// Same as `detect_audio`, but invokes `on_segment` after each segment that
/// exceeds the confidence threshold. This allows callers to display incremental
/// progress (e.g., updating a UI) without waiting for the entire file to finish.
///
/// The callback receives each `AudioSegment` as it is produced. The segment is
/// also collected into the returned `AudioDetectResult`, so the final result is
/// identical to `detect_audio`.
pub fn detect_audio_streaming(
    handle: &ModelHandle,
    audio: &AudioInput,
    opts: &AudioDetectOpts,
    mut on_segment: impl FnMut(&AudioSegment),
) -> Result<AudioDetectResult> {
    let start = Instant::now();
    let prep = prepare_audio_detection(handle, audio, opts)?;
    detect_audio_loop(handle, &prep, start, Some(&mut on_segment))
}

// ---------------------------------------------------------------------------
// Shared setup
// ---------------------------------------------------------------------------

/// Pre-computed state for audio detection, shared between `detect_audio` and
/// `detect_audio_streaming` to avoid duplicating the validation + loading code.
struct PreparedAudioDetection {
    audio_samples: preprocess_audio::AudioSamples,
    audio_config: preprocess_audio::AudioPreprocessConfig,
    filterbank: preprocess_audio::MelFilterbank,
    segment_samples: usize,
    stride_samples: usize,
    threshold: f32,
    sample_rate: u32,
}

/// Validate model type, load audio, resolve parameters, and pre-compute filterbank.
fn prepare_audio_detection(
    handle: &ModelHandle,
    audio: &AudioInput,
    opts: &AudioDetectOpts,
) -> Result<PreparedAudioDetection> {
    // 1. Validate model type — must have MelSpectrogram preprocessing.
    let manifest = &handle.manifest;
    let sample_rate = match &manifest.preprocess_method {
        PreprocessMethod::MelSpectrogram { sample_rate, .. } => *sample_rate,
        other => {
            let method_str = match other {
                PreprocessMethod::Letterbox => "letterbox",
                PreprocessMethod::Resize => "resize",
                PreprocessMethod::MelSpectrogram { .. } => unreachable!(),
            };
            return Err(SparrowEngineError::NotAnAudioModel {
                id: manifest.id.clone(),
                method: method_str.to_string(),
            });
        }
    };

    // 2. Fail fast: check handle validity before expensive audio loading.
    handle.check_valid()?;

    // 3. Load and resample audio to model's target sample rate.
    let audio_config =
        preprocess_audio::AudioPreprocessConfig::from_manifest(&manifest.preprocess_method)
            .ok_or_else(|| SparrowEngineError::NotAnAudioModel {
                id: manifest.id.clone(),
                method: "non-mel".to_string(),
            })?;
    // 4. Resolve sliding window parameters (manifest defaults, overridable by opts).
    let (segment_duration_s, segment_stride_s) = resolve_window_params(manifest, opts);

    // 5. Resolve confidence threshold: opts > postprocess default > 0.5 fallback.
    let default_threshold = match &manifest.postprocess_method {
        PostprocessMethod::Sigmoid {
            confidence_threshold,
        } => *confidence_threshold,
        _ => manifest.confidence_threshold.unwrap_or(0.5),
    };
    let threshold = opts.confidence_threshold.unwrap_or(default_threshold);

    let (segment_samples, stride_samples) = preprocess_audio::validate_audio_window_params(
        segment_duration_s,
        segment_stride_s,
        threshold,
        sample_rate,
        audio_config.n_fft,
    )?;

    let audio_samples = preprocess_audio::load_audio(audio, &audio_config)?;

    // 6. Pre-compute mel filterbank (constant for a given config, reuse across segments).
    let filterbank = preprocess_audio::MelFilterbank::new(&audio_config)?;

    Ok(PreparedAudioDetection {
        audio_samples,
        audio_config,
        filterbank,
        segment_samples,
        stride_samples,
        threshold,
        sample_rate,
    })
}

// ---------------------------------------------------------------------------
// Inner loop
// ---------------------------------------------------------------------------

/// Default batch size for batched audio inference.
/// Trades memory for throughput: each batch element is one mel spectrogram.
const DEFAULT_BATCH_SIZE: usize = 16;

/// Shared inner loop for both batch and streaming audio detection.
/// Processes segments in batches of DEFAULT_BATCH_SIZE for higher ORT throughput.
fn detect_audio_loop(
    handle: &ModelHandle,
    prep: &PreparedAudioDetection,
    start: Instant,
    mut on_segment: Option<&mut dyn FnMut(&AudioSegment)>,
) -> Result<AudioDetectResult> {
    let session = handle.pin_session()?;
    let total_samples = prep.audio_samples.data.len();
    let duration_s = prep.audio_samples.duration_s;
    let segment_samples = prep.segment_samples;
    let stride_samples = prep.stride_samples;
    let threshold = prep.threshold;
    let sample_rate = prep.sample_rate;

    // Pre-compute all segment offsets (matching Python golden termination logic).
    let mut offsets = Vec::new();
    let mut offset = 0usize;
    while offset < total_samples {
        offsets.push(offset);
        let remaining = total_samples - offset;
        if remaining <= segment_samples {
            break;
        }
        offset += stride_samples;
    }

    let mut segments = Vec::new();

    // Process segments in batches for higher ORT throughput.
    //
    // Per-batch stage timings (Phase 3.8 Step 2 Wave 0b): emitted as
    // `tracing::info!` events with `stage = "audio.preprocess|ort|postprocess"`
    // and `duration_ns`. The bench script in `scripts/bench_audio_breakdown.py`
    // sums these across batches per fixture run.
    for batch_offsets in offsets.chunks(DEFAULT_BATCH_SIZE) {
        let batch_len = batch_offsets.len();

        // ----- audio.preprocess (per batch): mel spectrogram + concat -----
        let t_preprocess = Instant::now();
        // Compute mel spectrograms for all segments in this batch.
        // mel_spectrogram returns Array4<f32> [1, 1, n_mels, time_steps];
        // convert to dynamic for concatenation along batch axis.
        let mut mel_views = Vec::with_capacity(batch_len);
        let mut mel_tensors = Vec::with_capacity(batch_len);
        for &seg_offset in batch_offsets {
            let remaining = total_samples - seg_offset;
            let tensor = if remaining >= segment_samples {
                preprocess_audio::mel_spectrogram(
                    &prep.audio_samples.data[seg_offset..seg_offset + segment_samples],
                    &prep.audio_config,
                    &prep.filterbank,
                )?
            } else {
                let mut padded = prep.audio_samples.data[seg_offset..].to_vec();
                padded.resize(segment_samples, 0.0);
                preprocess_audio::mel_spectrogram(&padded, &prep.audio_config, &prep.filterbank)?
            };
            mel_tensors.push(tensor.into_dyn());
        }
        for t in &mel_tensors {
            mel_views.push(t.view());
        }

        // Stack into batch tensor [N, 1, n_mels, time_steps].
        let batch_tensor = ndarray::concatenate(ndarray::Axis(0), &mel_views)
            .map_err(|e| SparrowEngineError::Ort(format!("batch concatenation failed: {e}")))?;
        tracing::info!(
            stage = "audio.preprocess",
            duration_ns = t_preprocess.elapsed().as_nanos() as u64,
            batch_len = batch_len,
        );

        // ----- audio.ort (per batch): session.run -----
        let t_ort = Instant::now();
        // Run ORT inference on the entire batch.
        let input_value =
            TensorRef::from_array_view(&batch_tensor).map_err(crate::engine::ort_err)?;

        let mut guard = session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("audio session lock poisoned".into()))?;
        let outputs = guard
            .run(ort::inputs![input_value])
            .map_err(crate::engine::ort_err)?;

        if outputs.len() == 0 {
            return Err(SparrowEngineError::Ort(
                "audio session returned no outputs".to_string(),
            ));
        }

        // Extract logits: output shape is [N, 1] for batched binary detection.
        let output_view: ArrayViewD<'_, f32> = outputs[0]
            .try_extract_array::<f32>()
            .map_err(crate::engine::ort_err)?;
        let logits: Vec<f32> = output_view.iter().copied().collect();

        drop(outputs);
        drop(guard);
        tracing::info!(
            stage = "audio.ort",
            duration_ns = t_ort.elapsed().as_nanos() as u64,
            batch_len = batch_len,
        );

        // Validate logit count matches batch size — a mismatch means the model
        // doesn't support batching or returned a malformed output. Without this
        // check, missing logits silently become sigmoid(0.0) = 0.5.
        if logits.len() != batch_len {
            return Err(SparrowEngineError::Ort(format!(
                "Audio model returned {} logits for batch of {} segments; expected exactly {}",
                logits.len(),
                batch_len,
                batch_len,
            )));
        }
        if !logits.iter().all(|logit| logit.is_finite()) {
            return Err(SparrowEngineError::Ort(
                "Audio model returned non-finite logits".to_string(),
            ));
        }

        // ----- audio.postprocess (per batch): sigmoid + threshold + collect -----
        let t_post = Instant::now();
        // Process each result in the batch.
        for (i, &seg_offset) in batch_offsets.iter().enumerate() {
            let logit = logits[i];
            let confidence = sigmoid(logit);

            if confidence >= threshold {
                let start_s = seg_offset as f32 / sample_rate as f32;
                let actual_end = (seg_offset + segment_samples).min(total_samples);
                let end_s = actual_end as f32 / sample_rate as f32;
                let seg = AudioSegment {
                    start_time_s: start_s,
                    end_time_s: end_s,
                    confidence,
                };

                if let Some(ref mut cb) = on_segment {
                    cb(&seg);
                }

                segments.push(seg);
            }
        }
        tracing::info!(
            stage = "audio.postprocess",
            duration_ns = t_post.elapsed().as_nanos() as u64,
            batch_len = batch_len,
        );
    }

    let elapsed = start.elapsed();

    Ok(AudioDetectResult {
        segments,
        duration_s,
        sample_rate,
        processing_time_ms: elapsed.as_secs_f32() * 1000.0,
    })
}

// ---------------------------------------------------------------------------
// Segment merging (Phase 3.5 S5 / item #6)
// ---------------------------------------------------------------------------

/// Merge consecutive [`AudioSegment`]s into [`AudioRange`]s.
///
/// Two segments merge when they share a class (always true for binary
/// detectors like MD_AudioBirds_V1, whose segments carry no class) and
/// the gap between the first segment's end and the second's start is
/// **strictly less than `gap_s`** seconds. A negative gap (overlap) also
/// merges. The merged range's `max_confidence` is the maximum of all
/// merged segments.
///
/// Input segments are assumed to be sorted by `start_time_s` — which is
/// what [`detect_audio`] and [`detect_audio_streaming`] produce. Unsorted
/// input still runs but may produce non-minimal ranges.
///
/// Empty input returns an empty vector.
///
/// # Threshold
///
/// `gap_s` is typically the sliding-window stride (so adjacent windows
/// merge but a true silence gap splits the range). For the Phase 1 audio
/// model MD_AudioBirds_V1 (1.0 s window, 0.3 s stride), the recommended
/// value is `0.3 + ε` (e.g. `0.31`) so strictly-adjacent windows merge.
/// The CLI uses `stride_s + 1e-3`.
///
/// # Phase 3.5 S5 (item #6)
///
/// Introduced to shrink the default `spe detect-audio` output from
/// ~198 per-window rows to a handful of merged ranges. The raw
/// per-window output remains available via the CLI `--raw-segments`
/// flag and `AudioDetectResult::segments` itself (this helper only
/// transforms; the raw vector is untouched).
pub fn merge_segments(segments: &[AudioSegment], gap_s: f32) -> Vec<AudioRange> {
    merge_segments_with_class(segments, gap_s, |_| None)
}

/// Like [`merge_segments`] but with a caller-supplied class mapper.
///
/// `class_of(segment)` returns an optional class label for a segment;
/// segments with different classes never merge. Binary detectors pass
/// `|_| None` (and use the simpler [`merge_segments`]). Future
/// multiclass audio models will plug a per-segment classifier in here.
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
// Helpers
// ---------------------------------------------------------------------------

/// Sigmoid activation: 1 / (1 + exp(-x)).
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Resolve sliding window parameters from manifest and runtime opts.
///
/// Runtime opts override manifest values. Returns `(segment_duration_s, stride_s)`.
fn resolve_window_params(
    manifest: &crate::manifest::ModelManifest,
    opts: &AudioDetectOpts,
) -> (f32, f32) {
    let (default_duration, default_stride) = match manifest.inference_strategy {
        InferenceStrategy::SlidingWindow {
            segment_duration_s,
            segment_stride_s,
        } => (segment_duration_s, segment_stride_s),
        // Fallback defaults matching MD_AudioBirds_V1.
        _ => (1.0, 0.3),
    };

    let duration = opts.segment_duration_s.unwrap_or(default_duration);
    let stride = opts.stride_s.unwrap_or(default_stride);
    (duration, stride)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn seg(start: f32, end: f32, conf: f32) -> AudioSegment {
        AudioSegment {
            start_time_s: start,
            end_time_s: end,
            confidence: conf,
        }
    }

    #[test]
    fn merge_empty_input() {
        let ranges = merge_segments(&[], 0.31);
        assert!(ranges.is_empty());
    }

    #[test]
    fn merge_single_segment() {
        let ranges = merge_segments(&[seg(0.0, 1.0, 0.9)], 0.31);
        assert_eq!(ranges.len(), 1);
        assert_eq!(ranges[0].start_time_s, 0.0);
        assert_eq!(ranges[0].end_time_s, 1.0);
        assert_eq!(ranges[0].max_confidence, 0.9);
        assert_eq!(ranges[0].class, None);
    }

    #[test]
    fn merge_adjacent_stride_windows_into_one_range() {
        // MD_AudioBirds_V1-style: 1.0 s window, 0.3 s stride, all above threshold.
        // Starts at 0.0, 0.3, 0.6, 0.9 ... ends at 1.0, 1.3, 1.6, 1.9 ...
        // gap_s = 0.31 (stride + eps). Each gap = start_next - end_prev = -0.7
        // (overlap), so all merge into one range.
        let mut segments = Vec::new();
        let mut t = 0.0f32;
        while t < 5.0 {
            segments.push(seg(t, t + 1.0, 0.95));
            t += 0.3;
        }
        let ranges = merge_segments(&segments, 0.31);
        assert_eq!(ranges.len(), 1, "all adjacent windows should merge");
        assert!((ranges[0].start_time_s - 0.0).abs() < 1e-6);
        assert!(ranges[0].end_time_s >= 5.0);
        assert_eq!(ranges[0].max_confidence, 0.95);
    }

    #[test]
    fn merge_splits_on_silence_gap() {
        // Two detection bursts separated by a 2.0 s silence. Should produce
        // two ranges even with tight gap_s.
        let segments = vec![
            seg(0.0, 1.0, 0.9),
            seg(0.3, 1.3, 0.95),
            // silence ~ 3.0–5.0 s
            seg(5.0, 6.0, 0.88),
            seg(5.3, 6.3, 0.92),
        ];
        let ranges = merge_segments(&segments, 0.31);
        assert_eq!(ranges.len(), 2, "silence gap should split into two ranges");
        assert!((ranges[0].start_time_s - 0.0).abs() < 1e-6);
        assert!((ranges[0].end_time_s - 1.3).abs() < 1e-6);
        assert_eq!(ranges[0].max_confidence, 0.95);
        assert!((ranges[1].start_time_s - 5.0).abs() < 1e-6);
        assert!((ranges[1].end_time_s - 6.3).abs() < 1e-6);
        assert_eq!(ranges[1].max_confidence, 0.92);
    }

    #[test]
    fn merge_takes_max_confidence() {
        let segments = vec![
            seg(0.0, 1.0, 0.55),
            seg(0.3, 1.3, 0.99),
            seg(0.6, 1.6, 0.77),
        ];
        let ranges = merge_segments(&segments, 0.31);
        assert_eq!(ranges.len(), 1);
        assert!((ranges[0].max_confidence - 0.99).abs() < 1e-6);
    }

    #[test]
    fn merge_gap_above_threshold_does_not_merge() {
        // Gap = 0.5 s, threshold = 0.31 s → two separate ranges.
        // Uses a gap clearly above threshold to avoid f32 boundary
        // noise (e.g. 1.31 - 1.0 is not exactly 0.31 in f32).
        let segments = vec![seg(0.0, 1.0, 0.9), seg(1.5, 2.5, 0.9)];
        let ranges = merge_segments(&segments, 0.31);
        assert_eq!(ranges.len(), 2);
    }

    #[test]
    fn merge_gap_just_below_threshold_merges() {
        // gap = 0.30 < gap_s = 0.31, so merge.
        let segments = vec![seg(0.0, 1.0, 0.9), seg(1.30, 2.30, 0.88)];
        let ranges = merge_segments(&segments, 0.31);
        assert_eq!(ranges.len(), 1);
        assert!((ranges[0].end_time_s - 2.30).abs() < 1e-6);
    }

    #[test]
    fn merge_with_class_splits_on_class_change() {
        let segments = vec![seg(0.0, 1.0, 0.9), seg(0.3, 1.3, 0.92), seg(0.6, 1.6, 0.88)];
        // Flip class on middle segment — should split into three ranges.
        let class_of = |s: &AudioSegment| -> Option<String> {
            if s.start_time_s < 0.2 {
                Some("a".to_string())
            } else if s.start_time_s < 0.5 {
                Some("b".to_string())
            } else {
                Some("a".to_string())
            }
        };
        let ranges = merge_segments_with_class(&segments, 0.31, class_of);
        assert_eq!(ranges.len(), 3);
        assert_eq!(ranges[0].class.as_deref(), Some("a"));
        assert_eq!(ranges[1].class.as_deref(), Some("b"));
        assert_eq!(ranges[2].class.as_deref(), Some("a"));
    }

    #[test]
    fn merge_preserves_end_time_when_later_segment_ends_earlier() {
        // Pathological input: segment 2 ends before segment 1's end. Merged
        // end_time must not regress.
        let segments = vec![seg(0.0, 5.0, 0.9), seg(0.3, 1.3, 0.95)];
        let ranges = merge_segments(&segments, 0.31);
        assert_eq!(ranges.len(), 1);
        assert!((ranges[0].end_time_s - 5.0).abs() < 1e-6);
    }
}
