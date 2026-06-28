//! Audio detection inference: sliding window over audio input.
//!
//! Orchestrates: load audio -> resample -> sliding window -> per-segment mel
//! spectrogram + ORT inference -> collect segments above threshold.

use std::sync::Arc;
use std::time::Instant;

use ndarray::{ArrayD, ArrayViewD};
use ort::value::TensorRef;
// Phase 3.8 Step 2 Wave 0b: per-stage `tracing::info!` timings (the workspace
// `tracing` dep is declared unconditional in sparrow-engine-cpu/Cargo.toml since Phase
// A). The bench harness in `scripts/bench_audio_breakdown.py` consumes these
// as `stage = "audio.<stage>" duration_ns = N` events. No new dep.

use crate::engine::ModelHandle;
use crate::error::{Result, SparrowEngineError};
use crate::manifest::{InferenceStrategy, PostprocessMethod, PreprocessMethod};
use crate::preprocess_audio;
use crate::types::{AudioClass, AudioDetectOpts, AudioDetectResult, AudioInput, AudioSegment};

/// Default top-K for multi-class audio classifiers (Perch 2-style).
/// Each emitted [`AudioSegment`] carries this many [`AudioClass`] entries,
/// sorted by probability desc. Binary detectors emit K=1.
const DEFAULT_AUDIO_CLASSIFIER_TOP_K: usize = 5;

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
    kind: PreparedAudioKind,
    segment_samples: usize,
    stride_samples: usize,
    threshold: f32,
    sample_rate: u32,
    /// Top-K to emit per segment on the classifier path. Ignored on the mel
    /// detector path (which always emits K=1).
    top_k: usize,
    /// Labels resolved from `manifest.labels` for class_idx → species lookup.
    /// Empty when the model has no labels file (legacy binary detectors).
    labels: Arc<Vec<String>>,
}

/// Per-path state inside `PreparedAudioDetection`, one variant per detection path:
/// - `Mel`: mel-spectrogram input, binary detector (Sigmoid postprocess).
/// - `MelClassifier`: mel-spectrogram input, multi-class classifier (Softmax postprocess).
/// - `Raw`: raw-audio input, multi-class classifier (Softmax postprocess; e.g. Perch 2).
enum PreparedAudioKind {
    Mel {
        audio_config: preprocess_audio::AudioPreprocessConfig,
        filterbank: preprocess_audio::MelFilterbank,
    },
    MelClassifier {
        audio_config: preprocess_audio::AudioPreprocessConfig,
        filterbank: preprocess_audio::MelFilterbank,
        /// Index of the ORT output tensor that carries per-class logits.
        logits_output_idx: usize,
        /// Number of softmax classes the model emits.
        num_classes: usize,
    },
    Raw {
        /// Index of the ORT output tensor that carries the per-class logits.
        /// For models like Perch 2 with multiple output heads we look up the
        /// tensor named "label" at session-load time; for single-output
        /// models this is just `0`.
        logits_output_idx: usize,
        /// Number of softmax classes the model emits.
        num_classes: usize,
        /// Opt-in (RP-27 Part 2, 2026-06-05): when true, the engine passes a
        /// second ONNX input `orig_sample_rate [1] int64` alongside the
        /// audio tensor. Used by in-graph fill_highfreq.
        pass_orig_sample_rate: bool,
    },
}

/// Validate model type, load audio, resolve parameters, and pre-compute filterbank.
fn prepare_audio_detection(
    handle: &ModelHandle,
    audio: &AudioInput,
    opts: &AudioDetectOpts,
) -> Result<PreparedAudioDetection> {
    let manifest = &handle.manifest;

    // 1. Validate model type — must use one of the audio preprocess methods.
    let sample_rate = match &manifest.preprocess_method {
        PreprocessMethod::MelSpectrogram { sample_rate, .. } => *sample_rate,
        PreprocessMethod::RawAudio { sample_rate, .. } => *sample_rate,
        other => {
            return Err(SparrowEngineError::NotAnAudioModel {
                id: manifest.id.clone(),
                method: other.as_str().to_string(),
            });
        }
    };

    // 2. Fail fast: check handle validity before expensive audio loading.
    handle.check_valid()?;

    // 3. Resolve sliding window parameters (manifest defaults, overridable by opts).
    let (segment_duration_s, segment_stride_s) = resolve_window_params(manifest, opts);

    // 4. Resolve confidence threshold for detector path.
    //    Classifiers ignore threshold entirely (emit every window), encoded
    //    here as `0.0` so the `confidence >= threshold` gate always passes.
    let default_threshold = match &manifest.postprocess_method {
        PostprocessMethod::Sigmoid {
            confidence_threshold,
        } => *confidence_threshold,
        PostprocessMethod::Softmax => 0.0,
        _ => manifest.confidence_threshold.unwrap_or(0.5),
    };
    let threshold = opts.confidence_threshold.unwrap_or(default_threshold);

    let labels = Arc::clone(&handle.labels);
    let top_k = DEFAULT_AUDIO_CLASSIFIER_TOP_K;

    // 5. Branch on preprocess method to assemble path-specific state.
    match &manifest.preprocess_method {
        PreprocessMethod::MelSpectrogram { .. } => {
            let audio_config =
                preprocess_audio::AudioPreprocessConfig::from_manifest(&manifest.preprocess_method)
                    .ok_or_else(|| SparrowEngineError::NotAnAudioModel {
                        id: manifest.id.clone(),
                        method: "non-mel".to_string(),
                    })?;
            let (segment_samples, stride_samples) = preprocess_audio::validate_audio_window_params(
                segment_duration_s,
                segment_stride_s,
                threshold,
                sample_rate,
                audio_config.n_fft,
            )?;
            let audio_samples = preprocess_audio::load_audio(audio, &audio_config)?;
            let filterbank = preprocess_audio::MelFilterbank::new(&audio_config)?;

            let kind = if matches!(manifest.postprocess_method, PostprocessMethod::Softmax) {
                let (logits_output_idx, num_classes) = resolve_mel_classifier_output(
                    handle,
                    &audio_config,
                    &filterbank,
                    segment_samples,
                )?;
                PreparedAudioKind::MelClassifier {
                    audio_config,
                    filterbank,
                    logits_output_idx,
                    num_classes,
                }
            } else {
                PreparedAudioKind::Mel {
                    audio_config,
                    filterbank,
                }
            };

            Ok(PreparedAudioDetection {
                audio_samples,
                kind,
                segment_samples,
                stride_samples,
                threshold,
                sample_rate,
                top_k,
                labels,
            })
        }
        PreprocessMethod::RawAudio {
            window_samples,
            pass_orig_sample_rate,
            ..
        } => {
            let segment_samples = *window_samples as usize;
            let stride_samples = (segment_stride_s * sample_rate as f32).round() as usize;
            if stride_samples == 0 {
                return Err(SparrowEngineError::InvalidManifest(
                    "stride_samples = 0; segment_stride_s × sample_rate rounded down to zero"
                        .to_string(),
                ));
            }

            // Surface the silent-ignore: raw_audio models bake the window length
            // into the ONNX (e.g. Perch 2: [batch, 160000] is the Conformer's
            // learned positional-embedding length, not a re-export quirk). Stride
            // overrides DO work; segment_duration overrides cannot be honored.
            // Only warn when the user's override differs from the architectural
            // window — passing the matching value is consistent intent.
            if let Some(requested_s) = opts.segment_duration_s {
                let window_seconds = segment_samples as f32 / sample_rate as f32;
                if (requested_s - window_seconds).abs() > 1e-3 {
                    tracing::warn!(
                        model_id = %manifest.id,
                        window_samples = segment_samples,
                        window_seconds,
                        requested_s,
                        "segment_duration_s override ignored: raw_audio models have an architecturally fixed window. Stride override still applies."
                    );
                }
            }

            let audio_samples = preprocess_audio::load_audio_at_sample_rate(audio, sample_rate)?;

            // Resolve the logits output: prefer the tensor named "label" (Perch 2),
            // fall back to output 0 for single-head softmax classifiers.
            // When pass_orig_sample_rate=true, probe with a dummy orig_sr=sample_rate
            // (the no-op case for fill_highfreq) so the 2-input ONNX accepts the call.
            let (logits_output_idx, num_classes) = resolve_classifier_output(
                handle,
                segment_samples,
                *pass_orig_sample_rate,
                sample_rate,
            )?;

            Ok(PreparedAudioDetection {
                audio_samples,
                kind: PreparedAudioKind::Raw {
                    logits_output_idx,
                    num_classes,
                    pass_orig_sample_rate: *pass_orig_sample_rate,
                },
                segment_samples,
                stride_samples,
                threshold,
                sample_rate,
                top_k,
                labels,
            })
        }
        _ => unreachable!("audio preprocess type guarded above"),
    }
}

/// Locate the ORT output tensor that carries per-class logits for a multi-class
/// audio classifier, and learn its class count.
///
/// For Perch 2 (4 outputs: `embedding`, `spatial_embedding`, `spectrogram`,
/// `label`) we pick the `label` head by name. For single-output classifiers we
/// fall back to output 0.
///
/// The class count is established by running ORT once on a zero-filled window
/// of the expected sample count. The number of classes returned is treated as
/// the model's authoritative class dimension and validated against
/// `manifest.labels.len()` when labels are present.
fn resolve_classifier_output(
    handle: &ModelHandle,
    window_samples: usize,
    pass_orig_sample_rate: bool,
    target_sample_rate: u32,
) -> Result<(usize, usize)> {
    let session = handle.pin_session()?;
    let mut guard = session
        .lock()
        .map_err(|_| SparrowEngineError::Ort("audio session lock poisoned".into()))?;

    let logits_idx = classifier_logits_output_idx(&guard);

    // Probe with one zero-filled window to learn the class count.
    let probe = ndarray::Array2::<f32>::zeros((1, window_samples));
    let input_value = TensorRef::from_array_view(&probe).map_err(crate::engine::ort_err)?;
    // RP-27 Part 2: 2-input ONNX needs orig_sample_rate populated even at probe
    // time. Use target_sample_rate (the no-op case for fill_highfreq).
    let probe_sr_arr;
    let outputs = if pass_orig_sample_rate {
        probe_sr_arr = ndarray::Array1::from_vec(vec![target_sample_rate as i64]);
        let orig_sr_value =
            TensorRef::from_array_view(&probe_sr_arr).map_err(crate::engine::ort_err)?;
        let inputs: Vec<(
            std::borrow::Cow<'_, str>,
            ort::session::SessionInputValue<'_>,
        )> = vec![
            (std::borrow::Cow::Borrowed("audio"), input_value.into()),
            (
                std::borrow::Cow::Borrowed("orig_sample_rate"),
                orig_sr_value.into(),
            ),
        ];
        guard.run(inputs).map_err(crate::engine::ort_err)?
    } else {
        guard
            .run(ort::inputs![input_value])
            .map_err(crate::engine::ort_err)?
    };
    if outputs.len() <= logits_idx {
        return Err(SparrowEngineError::Ort(format!(
            "classifier session probe returned {} outputs; expected at least {}",
            outputs.len(),
            logits_idx + 1
        )));
    }
    let view: ArrayViewD<'_, f32> = outputs[logits_idx]
        .try_extract_array::<f32>()
        .map_err(crate::engine::ort_err)?;
    let num_classes = validate_classifier_probe_output(&view)?;
    drop(outputs);
    drop(guard);

    validate_classifier_label_count(handle, num_classes)?;

    Ok((logits_idx, num_classes))
}

fn resolve_mel_classifier_output(
    handle: &ModelHandle,
    audio_config: &preprocess_audio::AudioPreprocessConfig,
    filterbank: &preprocess_audio::MelFilterbank,
    segment_samples: usize,
) -> Result<(usize, usize)> {
    let session = handle.pin_session()?;
    let mut guard = session
        .lock()
        .map_err(|_| SparrowEngineError::Ort("audio session lock poisoned".into()))?;

    let logits_idx = classifier_logits_output_idx(&guard);
    let probe_audio = vec![0.0f32; segment_samples];
    let probe_mel = preprocess_audio::mel_spectrogram(
        &probe_audio,
        audio_config.sample_rate,
        audio_config,
        filterbank,
    )?
    .into_dyn();
    let input_value = TensorRef::from_array_view(&probe_mel).map_err(crate::engine::ort_err)?;
    let outputs = guard
        .run(ort::inputs![input_value])
        .map_err(crate::engine::ort_err)?;
    if outputs.len() <= logits_idx {
        return Err(SparrowEngineError::Ort(format!(
            "classifier session probe returned {} outputs; expected at least {}",
            outputs.len(),
            logits_idx + 1
        )));
    }
    let view: ArrayViewD<'_, f32> = outputs[logits_idx]
        .try_extract_array::<f32>()
        .map_err(crate::engine::ort_err)?;
    let num_classes = validate_classifier_probe_output(&view)?;
    drop(outputs);
    drop(guard);

    validate_classifier_label_count(handle, num_classes)?;

    Ok((logits_idx, num_classes))
}

fn classifier_logits_output_idx(session: &ort::session::Session) -> usize {
    session
        .outputs()
        .iter()
        .position(|o| o.name() == "label")
        .unwrap_or(0)
}

fn validate_classifier_probe_output(view: &ArrayViewD<'_, f32>) -> Result<usize> {
    let shape = view.shape();
    if shape.len() != 2 || shape[0] != 1 {
        return Err(SparrowEngineError::Ort(format!(
            "classifier logits output has shape {:?}; expected [batch, num_classes]",
            shape
        )));
    }
    Ok(shape[1])
}

fn validate_classifier_label_count(handle: &ModelHandle, num_classes: usize) -> Result<()> {
    if handle.labels.is_empty() {
        return Err(SparrowEngineError::InvalidManifest(
            "softmax audio classifiers require a non-empty labels file so class probabilities can be mapped to labels"
                .to_string(),
        ));
    }
    if handle.labels.len() != num_classes {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "labels count ({}) does not match classifier output dim ({}) — manifest labels file is out of sync with the ONNX model",
            handle.labels.len(),
            num_classes
        )));
    }
    Ok(())
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
    on_segment: Option<&mut dyn FnMut(&AudioSegment)>,
) -> Result<AudioDetectResult> {
    match &prep.kind {
        PreparedAudioKind::Mel { .. } => detect_audio_loop_mel(handle, prep, start, on_segment),
        PreparedAudioKind::MelClassifier { .. } => {
            detect_audio_loop_mel_softmax(handle, prep, start, on_segment)
        }
        PreparedAudioKind::Raw { .. } => detect_audio_loop_raw(handle, prep, start, on_segment),
    }
}

fn build_mel_batch(
    prep: &PreparedAudioDetection,
    audio_config: &preprocess_audio::AudioPreprocessConfig,
    filterbank: &preprocess_audio::MelFilterbank,
    batch_offsets: &[usize],
) -> Result<ArrayD<f32>> {
    let total_samples = prep.audio_samples.data.len();
    let segment_samples = prep.segment_samples;
    let mut mel_tensors = Vec::with_capacity(batch_offsets.len());

    for &seg_offset in batch_offsets {
        let remaining = total_samples - seg_offset;
        let tensor = if remaining >= segment_samples {
            preprocess_audio::mel_spectrogram(
                &prep.audio_samples.data[seg_offset..seg_offset + segment_samples],
                prep.audio_samples.orig_sample_rate,
                audio_config,
                filterbank,
            )?
        } else {
            let mut padded = prep.audio_samples.data[seg_offset..].to_vec();
            padded.resize(segment_samples, 0.0);
            preprocess_audio::mel_spectrogram(
                &padded,
                prep.audio_samples.orig_sample_rate,
                audio_config,
                filterbank,
            )?
        };
        mel_tensors.push(tensor.into_dyn());
    }

    let mel_views: Vec<_> = mel_tensors.iter().map(|t| t.view()).collect();
    ndarray::concatenate(ndarray::Axis(0), &mel_views)
        .map_err(|e| SparrowEngineError::Ort(format!("batch concatenation failed: {e}")))
}

/// Mel-spectrogram path: binary detectors (e.g. MD_AudioBirds_V1) with
/// sigmoid postprocess. Emits one [`AudioSegment`] per above-threshold window,
/// each carrying a single-element `classes` vec.
fn detect_audio_loop_mel(
    handle: &ModelHandle,
    prep: &PreparedAudioDetection,
    start: Instant,
    mut on_segment: Option<&mut dyn FnMut(&AudioSegment)>,
) -> Result<AudioDetectResult> {
    let (audio_config, filterbank) = match &prep.kind {
        PreparedAudioKind::Mel {
            audio_config,
            filterbank,
        } => (audio_config, filterbank),
        _ => unreachable!("guarded by detect_audio_loop dispatch"),
    };

    let session = handle.pin_session()?;
    let total_samples = prep.audio_samples.data.len();
    let duration_s = prep.audio_samples.duration_s;
    let segment_samples = prep.segment_samples;
    let stride_samples = prep.stride_samples;
    let threshold = prep.threshold;
    let sample_rate = prep.sample_rate;
    // Binary detector: at most one label, used to populate AudioClass.label.
    let detector_label = prep.labels.first().cloned();

    // Pre-compute all segment offsets (matching Python golden termination logic).
    let offsets =
        preprocess_audio::compute_segment_offsets(total_samples, segment_samples, stride_samples);

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
        let batch_tensor = build_mel_batch(prep, audio_config, filterbank, batch_offsets)?;
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
                let (start_s, end_s) = preprocess_audio::segment_time_range(
                    seg_offset,
                    segment_samples,
                    total_samples,
                    sample_rate,
                );
                let seg = AudioSegment {
                    start_time_s: start_s,
                    end_time_s: end_s,
                    confidence,
                    classes: vec![AudioClass {
                        class_idx: 0,
                        label: detector_label.clone(),
                        probability: confidence,
                    }],
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

/// Mel-spectrogram path: multi-class softmax classifiers.
/// Emits one [`AudioSegment`] per window unconditionally, each carrying the top-K
/// classes. `confidence` is denormalised to the top-1 probability.
fn detect_audio_loop_mel_softmax(
    handle: &ModelHandle,
    prep: &PreparedAudioDetection,
    start: Instant,
    mut on_segment: Option<&mut dyn FnMut(&AudioSegment)>,
) -> Result<AudioDetectResult> {
    let (audio_config, filterbank, logits_output_idx, num_classes) = match &prep.kind {
        PreparedAudioKind::MelClassifier {
            audio_config,
            filterbank,
            logits_output_idx,
            num_classes,
        } => (audio_config, filterbank, *logits_output_idx, *num_classes),
        _ => unreachable!("guarded by detect_audio_loop dispatch"),
    };

    let session = handle.pin_session()?;
    let total_samples = prep.audio_samples.data.len();
    let duration_s = prep.audio_samples.duration_s;
    let segment_samples = prep.segment_samples;
    let stride_samples = prep.stride_samples;
    let sample_rate = prep.sample_rate;
    let top_k = prep.top_k.min(num_classes).max(1);

    let offsets =
        preprocess_audio::compute_segment_offsets(total_samples, segment_samples, stride_samples);
    let mut segments = Vec::with_capacity(offsets.len());

    for batch_offsets in offsets.chunks(DEFAULT_BATCH_SIZE) {
        let batch_len = batch_offsets.len();

        // ----- audio.preprocess (per batch): mel spectrogram + concat -----
        let t_preprocess = Instant::now();
        let batch_tensor = build_mel_batch(prep, audio_config, filterbank, batch_offsets)?;
        tracing::info!(
            stage = "audio.preprocess",
            duration_ns = t_preprocess.elapsed().as_nanos() as u64,
            batch_len = batch_len,
        );

        // ----- audio.ort (per batch): session.run ---------------------------
        let t_ort = Instant::now();
        let input_value =
            TensorRef::from_array_view(&batch_tensor).map_err(crate::engine::ort_err)?;
        let mut guard = session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("audio session lock poisoned".into()))?;
        let outputs = guard
            .run(ort::inputs![input_value])
            .map_err(crate::engine::ort_err)?;
        if outputs.len() <= logits_output_idx {
            return Err(SparrowEngineError::Ort(format!(
                "audio classifier returned {} outputs; expected at least {}",
                outputs.len(),
                logits_output_idx + 1
            )));
        }
        let output_view: ArrayViewD<'_, f32> = outputs[logits_output_idx]
            .try_extract_array::<f32>()
            .map_err(crate::engine::ort_err)?;
        let expected = batch_len * num_classes;
        let view_shape = output_view.shape().to_vec();
        if output_view.len() != expected {
            return Err(SparrowEngineError::Ort(format!(
                "audio classifier returned {} elements (shape {:?}) for batch of {} segments x {} classes; expected exactly {}",
                output_view.len(),
                view_shape,
                batch_len,
                num_classes,
                expected
            )));
        }
        let logits: Vec<f32> = output_view.iter().copied().collect();
        drop(outputs);
        drop(guard);
        if !logits.iter().all(|x| x.is_finite()) {
            return Err(SparrowEngineError::Ort(
                "audio classifier returned non-finite logits".to_string(),
            ));
        }
        tracing::info!(
            stage = "audio.ort",
            duration_ns = t_ort.elapsed().as_nanos() as u64,
            batch_len = batch_len,
        );

        // ----- audio.postprocess (per batch): softmax + top-K ---------------
        let t_post = Instant::now();
        for (i, &seg_offset) in batch_offsets.iter().enumerate() {
            let window_logits = &logits[i * num_classes..(i + 1) * num_classes];
            let probs = softmax(window_logits);
            let top = top_k_indices(&probs, top_k);
            let classes: Vec<AudioClass> = top
                .into_iter()
                .map(|(idx, p)| AudioClass {
                    class_idx: idx as u32,
                    label: prep.labels.get(idx).cloned(),
                    probability: p,
                })
                .collect();
            let top1_prob = classes.first().map(|c| c.probability).unwrap_or(0.0);

            let (start_s, end_s) = preprocess_audio::segment_time_range(
                seg_offset,
                segment_samples,
                total_samples,
                sample_rate,
            );
            let seg = AudioSegment {
                start_time_s: start_s,
                end_time_s: end_s,
                confidence: top1_prob,
                classes,
            };
            if let Some(ref mut cb) = on_segment {
                cb(&seg);
            }
            segments.push(seg);
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

/// Raw-audio path: multi-class softmax classifiers (e.g. Perch 2 / 14795 species).
/// Emits one [`AudioSegment`] per window unconditionally, each carrying the top-K
/// classes. `confidence` is denormalised to the top-1 probability.
fn detect_audio_loop_raw(
    handle: &ModelHandle,
    prep: &PreparedAudioDetection,
    start: Instant,
    mut on_segment: Option<&mut dyn FnMut(&AudioSegment)>,
) -> Result<AudioDetectResult> {
    let (logits_output_idx, num_classes, pass_orig_sample_rate) = match &prep.kind {
        PreparedAudioKind::Raw {
            logits_output_idx,
            num_classes,
            pass_orig_sample_rate,
        } => (*logits_output_idx, *num_classes, *pass_orig_sample_rate),
        _ => unreachable!("guarded by detect_audio_loop dispatch"),
    };

    let session = handle.pin_session()?;
    let total_samples = prep.audio_samples.data.len();
    let duration_s = prep.audio_samples.duration_s;
    let segment_samples = prep.segment_samples;
    let stride_samples = prep.stride_samples;
    let sample_rate = prep.sample_rate;
    let top_k = prep.top_k.min(num_classes).max(1);

    // Pre-compute window offsets (same termination as the mel path; see
    // `preprocess_audio::compute_segment_offsets`).
    let offsets =
        preprocess_audio::compute_segment_offsets(total_samples, segment_samples, stride_samples);

    let mut segments = Vec::with_capacity(offsets.len());

    for batch_offsets in offsets.chunks(DEFAULT_BATCH_SIZE) {
        let batch_len = batch_offsets.len();

        // ----- audio.preprocess (per batch): pack raw windows ---------------
        let t_preprocess = Instant::now();
        let mut batch_data = Vec::with_capacity(batch_len * segment_samples);
        for &seg_offset in batch_offsets {
            let remaining = total_samples - seg_offset;
            if remaining >= segment_samples {
                batch_data.extend_from_slice(
                    &prep.audio_samples.data[seg_offset..seg_offset + segment_samples],
                );
            } else {
                batch_data.extend_from_slice(&prep.audio_samples.data[seg_offset..]);
                batch_data.resize(batch_data.len() + (segment_samples - remaining), 0.0);
            }
        }
        let batch_tensor =
            ndarray::Array2::from_shape_vec((batch_len, segment_samples), batch_data).map_err(
                |e| SparrowEngineError::Ort(format!("raw audio batch reshape failed: {e}")),
            )?;
        tracing::info!(
            stage = "audio.preprocess",
            duration_ns = t_preprocess.elapsed().as_nanos() as u64,
            batch_len = batch_len,
        );

        // ----- audio.ort (per batch): session.run ---------------------------
        let t_ort = Instant::now();
        let input_value =
            TensorRef::from_array_view(&batch_tensor).map_err(crate::engine::ort_err)?;
        let mut guard = session
            .lock()
            .map_err(|_| SparrowEngineError::Ort("audio session lock poisoned".into()))?;
        // RP-27 Part 2: when manifest opts in, pass orig_sample_rate as a
        // second [1] int64 input alongside the audio tensor. The exported
        // ONNX must declare two inputs in this order: ("audio", "orig_sample_rate").
        let orig_sr_arr;
        let outputs = if pass_orig_sample_rate {
            orig_sr_arr =
                ndarray::Array1::from_vec(vec![prep.audio_samples.orig_sample_rate as i64]);
            let orig_sr_value =
                TensorRef::from_array_view(&orig_sr_arr).map_err(crate::engine::ort_err)?;
            let inputs: Vec<(
                std::borrow::Cow<'_, str>,
                ort::session::SessionInputValue<'_>,
            )> = vec![
                (std::borrow::Cow::Borrowed("audio"), input_value.into()),
                (
                    std::borrow::Cow::Borrowed("orig_sample_rate"),
                    orig_sr_value.into(),
                ),
            ];
            guard.run(inputs).map_err(crate::engine::ort_err)?
        } else {
            guard
                .run(ort::inputs![input_value])
                .map_err(crate::engine::ort_err)?
        };
        if outputs.len() <= logits_output_idx {
            return Err(SparrowEngineError::Ort(format!(
                "audio classifier returned {} outputs; expected at least {}",
                outputs.len(),
                logits_output_idx + 1
            )));
        }
        let output_view: ArrayViewD<'_, f32> = outputs[logits_output_idx]
            .try_extract_array::<f32>()
            .map_err(crate::engine::ort_err)?;
        let expected = batch_len * num_classes;
        let view_shape = output_view.shape().to_vec();
        if output_view.len() != expected {
            return Err(SparrowEngineError::Ort(format!(
                "audio classifier returned {} elements (shape {:?}) for batch of {} \
                 segments × {} classes; expected exactly {}",
                output_view.len(),
                view_shape,
                batch_len,
                num_classes,
                expected
            )));
        }
        let logits: Vec<f32> = output_view.iter().copied().collect();
        drop(outputs);
        drop(guard);
        if !logits.iter().all(|x| x.is_finite()) {
            return Err(SparrowEngineError::Ort(
                "audio classifier returned non-finite logits".to_string(),
            ));
        }
        tracing::info!(
            stage = "audio.ort",
            duration_ns = t_ort.elapsed().as_nanos() as u64,
            batch_len = batch_len,
        );

        // ----- audio.postprocess (per batch): softmax + top-K ---------------
        let t_post = Instant::now();
        for (i, &seg_offset) in batch_offsets.iter().enumerate() {
            let window_logits = &logits[i * num_classes..(i + 1) * num_classes];
            let probs = softmax(window_logits);
            let top = top_k_indices(&probs, top_k);
            let classes: Vec<AudioClass> = top
                .into_iter()
                .map(|(idx, p)| AudioClass {
                    class_idx: idx as u32,
                    label: prep.labels.get(idx).cloned(),
                    probability: p,
                })
                .collect();
            let top1_prob = classes.first().map(|c| c.probability).unwrap_or(0.0);

            let (start_s, end_s) = preprocess_audio::segment_time_range(
                seg_offset,
                segment_samples,
                total_samples,
                sample_rate,
            );
            let seg = AudioSegment {
                start_time_s: start_s,
                end_time_s: end_s,
                confidence: top1_prob,
                classes,
            };
            if let Some(ref mut cb) = on_segment {
                cb(&seg);
            }
            segments.push(seg);
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

// Local softmax + top-K helpers for the raw-audio classifier path.
//
// These intentionally duplicate the math in
// `sparrow_engine_core::postprocess::try_softmax` because the audio path
// emits `AudioClass { class_idx, label, probability }`, while `try_softmax`
// emits `Classification { label_id, label, confidence }`. The two structs
// are not interchangeable, so a primitive-level dedup would require
// splitting `try_softmax` into a `softmax_probs(&row) -> Vec<f32>` that
// both consumers wrap — tracked under the round 1 auditor plan's
// cross-scope finding #2 (`postprocess.rs` is outside the round 1
// audit-fix owned-file set).

/// Numerically-stable softmax over a slice of logits.
fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum > 0.0 {
        for e in &mut exps {
            *e /= sum;
        }
    }
    exps
}

/// Return the K largest `(index, probability)` pairs in `probs`, sorted by
/// probability desc. Stable on ties (lower index wins).
fn top_k_indices(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let k = k.min(probs.len());
    let mut pairs: Vec<(usize, f32)> = probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    pairs.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    pairs.truncate(k);
    pairs
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
            classes: Vec::new(),
        }
    }

    #[test]
    fn classifier_probe_output_accepts_single_batch_logits() {
        let logits = ndarray::Array2::<f32>::zeros((1, 3)).into_dyn();
        let num_classes = validate_classifier_probe_output(&logits.view()).unwrap();
        assert_eq!(num_classes, 3);
    }

    #[test]
    fn classifier_probe_output_rejects_wrong_rank() {
        let logits = ndarray::Array1::<f32>::zeros(3).into_dyn();
        let err = validate_classifier_probe_output(&logits.view()).unwrap_err();
        assert!(err.to_string().contains("expected [batch, num_classes]"));
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
