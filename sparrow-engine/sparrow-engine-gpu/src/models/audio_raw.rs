//! Raw-audio classifier path (GPU flavor) — Phase D round 2 B-08.
//!
//! Mirrors `sparrow_engine_cpu::detect_audio::detect_audio_loop_raw` for
//! multi-class softmax audio classifiers whose ONNX consumes raw f32
//! samples directly (no mel spectrogram preprocessing). The reference
//! model is Perch 2 / `perch-v2` (4 outputs: `embedding`,
//! `spatial_embedding`, `spectrogram`, `label`; we pick `label` by name,
//! fall back to output 0).
//!
//! # Why this is a separate type from `AudioModel`
//!
//! `AudioModel` is mel-only — load time constructs the full mel pipeline
//! (cuFFT plans, mel filterbank upload, NVRTC kernels, cuBLAS handle,
//! workspace mutex). Raw audio needs none of that: the ONNX consumes the
//! raw [batch, samples] tensor directly, and ORT's CUDA EP handles the
//! H2D copy + inference internally. A parallel struct avoids dragging
//! the mel-pipeline init cost (~150 ms cuFFT + NVRTC compile) into raw
//! audio loads, and keeps the dispatch trivial at the engine layer.
//!
//! # Performance note
//!
//! This first cut uses the standard ORT `Session::run` path with
//! host-side `TensorRef::from_array_view` inputs. The CUDA EP copies
//! the batch tensor to device internally per call. A future
//! optimization (analogous to `AudioOrtSession` + IoBinding for the mel
//! path) could pre-allocate a CUDA-resident input buffer + bind it via
//! `IoBinding`, but that requires per-batch-size dynamic buffer
//! management. For Perch 2 (single-batch, 5 s windows at 32 kHz =
//! 160 000 f32 = 640 KB per call) the host-to-device copy is sub-ms
//! and not the bottleneck.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use cudarc::driver::CudaContext;
use ndarray::ArrayViewD;
use ort::session::builder::GraphOptimizationLevel;
use ort::session::Session;
use ort::value::TensorRef;
use sparrow_engine_core::preprocess_audio;
use sparrow_engine_types::error::{Result, SparrowEngineError};
use sparrow_engine_types::manifest::{
    InferenceStrategy, ModelManifest, PostprocessMethod, Precision, PreprocessMethod,
};
use sparrow_engine_types::types::{
    AudioClass, AudioDetectOpts, AudioDetectResult, AudioInput, AudioSegment,
};

use crate::trt::ep::{manifest_cache_material, CudaEpConfig, GpuIdentity, TrtEpBuilder};

/// Default top-K when caller doesn't override. Mirrors
/// `sparrow_engine_cpu::detect_audio::DEFAULT_AUDIO_CLASSIFIER_TOP_K`.
const DEFAULT_AUDIO_CLASSIFIER_TOP_K: usize = 5;

/// Default batch size for raw-audio ORT inference. Trades memory for
/// throughput; same value as the CPU path.
const DEFAULT_BATCH_SIZE: usize = 16;

/// Per-loaded raw-audio model (CUDA-EP ORT session + manifest-derived params).
pub struct RawAudioModel {
    /// ORT session bound to the CUDA EP (CPU fallback per-op). Wrapped in
    /// `Mutex` because `Session::run` is `&mut self` — same pattern as
    /// `YoloModel` / `ClassifierModel` / `AudioModel`.
    session: Mutex<Session>,

    /// Index of the logits output tensor inside `session.outputs()`. Cached
    /// at load time. `label`-named output preferred (Perch 2 4-output
    /// model); falls back to output 0 for single-head softmax classifiers.
    logits_output_idx: usize,

    /// Number of softmax classes — established at load time by running ORT
    /// once on a zero-filled window of the expected sample count. Used to
    /// validate every subsequent batch.
    num_classes: usize,

    // Manifest-derived params (resolved at load; per-call overrides via opts).
    sample_rate: u32,
    /// Architecturally-fixed window length baked into the ONNX (Perch 2:
    /// 160 000 samples = 5 s @ 32 kHz). Cannot be overridden via opts.
    segment_samples: usize,
    /// Default stride in samples. Overrideable via `opts.stride_s`.
    stride_samples: usize,
    /// Default confidence threshold. NOTE: raw-audio path is unconditional
    /// (emits one segment per window regardless), but the value is held
    /// here for parity with the mel path + future API consistency.
    threshold: f32,
    /// RP-27 Part 2 opt-in: when true, pass orig_sample_rate as a second
    /// ONNX input alongside the audio tensor. Forwarded from the manifest.
    pass_orig_sample_rate: bool,
}

// SAFETY: `Session` is wrapped in `Mutex` (Send + Sync via Mutex's bounds).
// Manifest-derived primitives are trivially Send + Sync. Mirrors the
// patterns in `models/yolo.rs` / `models/classifier.rs` / `models/audio.rs`.
unsafe impl Send for RawAudioModel {}
unsafe impl Sync for RawAudioModel {}

impl RawAudioModel {
    /// Build a `RawAudioModel` from a parsed manifest. Called by
    /// `engine::Engine::load_from_manifest` when
    /// `manifest.preprocess_method == PreprocessMethod::RawAudio`.
    pub fn load_from_manifest(
        ctx: &Arc<CudaContext>,
        manifest: &ModelManifest,
        manifest_dir: &Path,
    ) -> Result<Self> {
        // 1. Extract sample_rate + window_samples from the RawAudio variant.
        let (sample_rate, segment_samples, pass_orig_sample_rate) =
            match &manifest.preprocess_method {
                PreprocessMethod::RawAudio {
                    sample_rate,
                    window_samples,
                    pass_orig_sample_rate,
                } => (
                    *sample_rate,
                    *window_samples as usize,
                    *pass_orig_sample_rate,
                ),
                other => {
                    return Err(SparrowEngineError::NotAnAudioModel {
                        id: manifest.id.clone(),
                        method: other.as_str().to_string(),
                    });
                }
            };

        // 2. Default stride: SlidingWindow inference strategy carries the
        // manifest-declared stride_s; fall back to the CPU default (0.3 s)
        // if the manifest uses a different strategy variant.
        let (default_segment_duration_s, default_stride_s) = match manifest.inference_strategy {
            InferenceStrategy::SlidingWindow {
                segment_duration_s,
                segment_stride_s,
            } => (segment_duration_s, segment_stride_s),
            _ => (segment_samples as f32 / sample_rate as f32, 0.3),
        };
        let stride_samples = (default_stride_s * sample_rate as f32).round() as usize;
        if stride_samples == 0 {
            return Err(SparrowEngineError::InvalidManifest(format!(
                "manifest '{}': default stride_s × sample_rate rounded down to zero",
                manifest.id
            )));
        }
        // Suppress unused warning; preserved for future per-call validation.
        let _ = default_segment_duration_s;

        // 3. Default confidence threshold from the postprocess method.
        // `Softmax` (raw-audio default per manifest validation at
        // sparrow-engine-types/src/manifest.rs:902) is unit-thresholded
        // (multi-class — every window emits the top-K regardless), so
        // the per-window threshold is not meaningful; we record 0.0.
        let threshold = match &manifest.postprocess_method {
            PostprocessMethod::Sigmoid {
                confidence_threshold,
            } => *confidence_threshold,
            _ => 0.0,
        };

        // 4. Resolve ONNX path with FP32/FP16 split (same convention as
        // AudioModel + YoloModel + ClassifierModel).
        let onnx_path = match manifest.precision {
            Precision::Fp32 => manifest_dir.join(&manifest.model_file),
            Precision::Int8 => manifest_dir.join(&manifest.model_file),
            Precision::Fp16 => {
                manifest_dir.join(manifest.model_file_fp16.as_ref().ok_or_else(|| {
                    SparrowEngineError::InvalidManifest(format!(
                        "manifest '{}': precision = fp16 requires [model] file_fp16",
                        manifest.id
                    ))
                })?)
            }
        };
        if !onnx_path.exists() {
            return Err(SparrowEngineError::Ort(format!(
                "raw-audio ONNX file does not exist: {onnx_path:?}"
            )));
        }

        // 5. Build session via the TRT→CUDA→CPU EP policy (crate::trt::ep): TRT
        // only when the manifest opts in, else CUDA-first, CPU per-op fallback.
        // NOTE: unlike AudioModel we do NOT
        // bind a dedicated compute stream — there are no cudarc kernels to
        // co-schedule (raw audio path is pure ORT), so the default stream
        // is fine and saves the `Arc<CudaStream>` plumbing.
        let device_id: i32 = ctx
            .ordinal()
            .try_into()
            .map_err(|e| SparrowEngineError::Ort(format!("ctx.ordinal as i32: {e}")))?;

        let gpu = GpuIdentity::from_context(ctx)?;
        let manifest_cache_material = manifest_cache_material(manifest);
        let providers = TrtEpBuilder::new(
            &manifest.id,
            manifest.trt.as_ref(),
            &gpu,
            CudaEpConfig::new(device_id),
            &onnx_path,
            &manifest_cache_material,
        )
        .execution_providers()?;
        let session = Session::builder()
            .map_err(|e| SparrowEngineError::Ort(format!("ort Session::builder: {e}")))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| SparrowEngineError::Ort(format!("with_optimization_level: {e}")))?
            .with_execution_providers(providers)
            .map_err(|e| {
                SparrowEngineError::Ort(format!("with_execution_providers(TRT, CUDA, CPU): {e}"))
            })?
            .commit_from_file(&onnx_path)
            .map_err(|e| {
                SparrowEngineError::Ort(format!("commit_from_file({onnx_path:?}): {e}"))
            })?;

        // 6. Resolve logits output: prefer "label" (Perch 2), fall back to 0.
        let logits_output_idx = session
            .outputs()
            .iter()
            .position(|o| o.name() == "label")
            .unwrap_or(0);

        // 7. Probe num_classes by running a single zero-filled window. This
        // is the same technique the CPU path uses
        // (`resolve_classifier_output` in
        // `sparrow-engine-cpu/src/detect_audio.rs:286`).
        let num_classes = {
            let mut probe_session = session;
            let probe_input = ndarray::Array2::<f32>::zeros((1, segment_samples));
            let probe_value = TensorRef::from_array_view(&probe_input)
                .map_err(|e| SparrowEngineError::Ort(format!("probe TensorRef: {e}")))?;
            // RP-27 Part 2: 2-input ONNX needs orig_sample_rate populated even
            // at probe time. Use sample_rate (the no-op case for fill_highfreq).
            let probe_sr_arr;
            let probe_outputs = if pass_orig_sample_rate {
                probe_sr_arr = ndarray::Array1::from_vec(vec![sample_rate as i64]);
                let probe_sr_value = TensorRef::from_array_view(&probe_sr_arr).map_err(|e| {
                    SparrowEngineError::Ort(format!("probe orig_sr TensorRef: {e}"))
                })?;
                let inputs: Vec<(
                    std::borrow::Cow<'_, str>,
                    ort::session::SessionInputValue<'_>,
                )> = vec![
                    (std::borrow::Cow::Borrowed("audio"), probe_value.into()),
                    (
                        std::borrow::Cow::Borrowed("orig_sample_rate"),
                        probe_sr_value.into(),
                    ),
                ];
                probe_session.run(inputs).map_err(|e| {
                    SparrowEngineError::Ort(format!("probe Session::run (2-input): {e}"))
                })?
            } else {
                probe_session
                    .run(ort::inputs![probe_value])
                    .map_err(|e| SparrowEngineError::Ort(format!("probe Session::run: {e}")))?
            };
            if probe_outputs.len() <= logits_output_idx {
                return Err(SparrowEngineError::Ort(format!(
                    "raw-audio probe: model has {} outputs, expected logits at index {}",
                    probe_outputs.len(),
                    logits_output_idx
                )));
            }
            let probe_view: ArrayViewD<'_, f32> = probe_outputs[logits_output_idx]
                .try_extract_array::<f32>()
                .map_err(|e| SparrowEngineError::Ort(format!("probe extract: {e}")))?;
            // Last axis = class dimension. Shape is [batch, classes] or
            // [batch, 1, classes]; we want the product / batch.
            let total = probe_view.len();
            if total == 0 {
                return Err(SparrowEngineError::Ort(
                    "raw-audio probe: logits output has zero elements".to_string(),
                ));
            }
            // batch was 1, so total == num_classes (no need to divide).
            let n = total;
            drop(probe_outputs);
            // Move session back out — re-wrap in Mutex below.
            // (We held it by value during the probe to avoid an extra
            // Mutex hop inside `load`; this is the only call site.)
            let _ = probe_session;
            n
        };

        // Re-open the session for the persistent Mutex. The probe consumed
        // `session` by-value, so rebuild from disk. This pays the
        // commit-from-file cost twice on load, but load only happens once
        // per process per model; runtime detect calls are unaffected.
        let providers = TrtEpBuilder::new(
            &manifest.id,
            manifest.trt.as_ref(),
            &gpu,
            CudaEpConfig::new(device_id),
            &onnx_path,
            &manifest_cache_material,
        )
        .execution_providers()?;
        let session = Session::builder()
            .map_err(|e| SparrowEngineError::Ort(format!("ort Session::builder (retain): {e}")))?
            .with_optimization_level(GraphOptimizationLevel::All)
            .map_err(|e| SparrowEngineError::Ort(format!("with_optimization_level (retain): {e}")))?
            .with_execution_providers(providers)
            .map_err(|e| {
                SparrowEngineError::Ort(format!(
                    "with_execution_providers(TRT, CUDA, CPU) (retain): {e}"
                ))
            })?
            .commit_from_file(&onnx_path)
            .map_err(|e| {
                SparrowEngineError::Ort(format!("commit_from_file({onnx_path:?}) (retain): {e}"))
            })?;

        // NOTE: label-count vs num_classes consistency check is handled at
        // the engine layer (where the loaded label vec is constructed).
        // Here we only know the model's class count; the caller passes
        // labels in at detect time.

        Ok(RawAudioModel {
            session: Mutex::new(session),
            logits_output_idx,
            num_classes,
            sample_rate,
            segment_samples,
            stride_samples,
            threshold,
            pass_orig_sample_rate,
        })
    }

    /// Run raw-audio inference with sliding window, returning all segments
    /// (one per window) ordered by start time.
    pub fn detect(
        &self,
        audio: &AudioInput,
        opts: &AudioDetectOpts,
        labels: &[String],
    ) -> Result<AudioDetectResult> {
        self.detect_inner(audio, opts, labels, None)
    }

    /// Streaming variant — fires `on_segment` per emitted segment in chunk order.
    pub fn detect_streaming<F>(
        &self,
        audio: &AudioInput,
        opts: &AudioDetectOpts,
        labels: &[String],
        mut on_segment: F,
    ) -> Result<AudioDetectResult>
    where
        F: FnMut(&AudioSegment),
    {
        self.detect_inner(
            audio,
            opts,
            labels,
            Some(&mut on_segment as &mut dyn FnMut(&AudioSegment)),
        )
    }

    fn detect_inner(
        &self,
        audio: &AudioInput,
        opts: &AudioDetectOpts,
        labels: &[String],
        mut on_segment: Option<&mut dyn FnMut(&AudioSegment)>,
    ) -> Result<AudioDetectResult> {
        let start = Instant::now();

        // 1. Decode + resample to model SR (CPU path; cheap and ORT CUDA EP
        // handles the H2D copy of the resulting f32 batch internally).
        let audio_samples = preprocess_audio::load_audio_at_sample_rate(audio, self.sample_rate)?;
        let total_samples = audio_samples.data.len();
        let duration_s = audio_samples.duration_s;
        let segment_samples = self.segment_samples;

        // 2. Resolve per-call params from opts (stride only — segment is
        // architecturally fixed). Warn-once on segment override mismatch
        // matches CPU semantics in `prepare_audio_detection` at
        // `sparrow-engine-cpu/src/detect_audio.rs:236-247`.
        let stride_samples = if let Some(stride_s) = opts.stride_s {
            let s = (stride_s * self.sample_rate as f32).round() as usize;
            if s == 0 {
                return Err(SparrowEngineError::InvalidManifest(
                    "stride_s × sample_rate rounded down to zero".to_string(),
                ));
            }
            s
        } else {
            self.stride_samples
        };
        if let Some(requested_s) = opts.segment_duration_s {
            let window_seconds = segment_samples as f32 / self.sample_rate as f32;
            if (requested_s - window_seconds).abs() > 1e-3 {
                tracing::warn!(
                    window_samples = segment_samples,
                    window_seconds,
                    requested_s,
                    "segment_duration_s override ignored: raw_audio models have an architecturally fixed window. Stride override still applies."
                );
            }
        }
        let _ = opts.confidence_threshold.unwrap_or(self.threshold); // threshold unused for raw path.

        let top_k = DEFAULT_AUDIO_CLASSIFIER_TOP_K.min(self.num_classes).max(1);

        let offsets = preprocess_audio::compute_segment_offsets(
            total_samples,
            segment_samples,
            stride_samples,
        );

        let mut segments = Vec::with_capacity(offsets.len());

        for batch_offsets in offsets.chunks(DEFAULT_BATCH_SIZE) {
            let batch_len = batch_offsets.len();

            // ----- preprocess: pack raw windows (zero-pad short tail) -----
            let mut batch_data = Vec::with_capacity(batch_len * segment_samples);
            for &seg_offset in batch_offsets {
                let remaining = total_samples - seg_offset;
                if remaining >= segment_samples {
                    batch_data.extend_from_slice(
                        &audio_samples.data[seg_offset..seg_offset + segment_samples],
                    );
                } else {
                    batch_data.extend_from_slice(&audio_samples.data[seg_offset..]);
                    batch_data.resize(batch_data.len() + (segment_samples - remaining), 0.0);
                }
            }
            let batch_tensor = ndarray::Array2::from_shape_vec(
                (batch_len, segment_samples),
                batch_data,
            )
            .map_err(|e| {
                SparrowEngineError::Ort(format!("raw audio batch reshape failed (GPU): {e}"))
            })?;

            // ----- ORT (CUDA EP handles H2D internally) ------------------
            let input_value = TensorRef::from_array_view(&batch_tensor)
                .map_err(|e| SparrowEngineError::Ort(format!("TensorRef::from_array_view: {e}")))?;
            let mut guard = self
                .session
                .lock()
                .map_err(|_| SparrowEngineError::Ort("raw audio session lock poisoned".into()))?;
            // RP-27 Part 2: when manifest opts in, pass orig_sample_rate as a
            // second ONNX input (same as CPU path).
            let orig_sr_arr;
            let outputs = if self.pass_orig_sample_rate {
                orig_sr_arr =
                    ndarray::Array1::from_vec(vec![audio_samples.orig_sample_rate as i64]);
                let orig_sr_value = TensorRef::from_array_view(&orig_sr_arr)
                    .map_err(|e| SparrowEngineError::Ort(format!("orig_sr TensorRef: {e}")))?;
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
                guard
                    .run(inputs)
                    .map_err(|e| SparrowEngineError::Ort(format!("Session::run (2-input): {e}")))?
            } else {
                guard
                    .run(ort::inputs![input_value])
                    .map_err(|e| SparrowEngineError::Ort(format!("Session::run: {e}")))?
            };
            if outputs.len() <= self.logits_output_idx {
                return Err(SparrowEngineError::Ort(format!(
                    "raw-audio classifier returned {} outputs; expected at least {}",
                    outputs.len(),
                    self.logits_output_idx + 1
                )));
            }
            let output_view: ArrayViewD<'_, f32> = outputs[self.logits_output_idx]
                .try_extract_array::<f32>()
                .map_err(|e| SparrowEngineError::Ort(format!("output extract: {e}")))?;
            let expected = batch_len * self.num_classes;
            if output_view.len() != expected {
                let view_shape = output_view.shape().to_vec();
                return Err(SparrowEngineError::Ort(format!(
                    "raw-audio classifier returned {} elements (shape {:?}) for batch of {} \
                     segments × {} classes; expected exactly {}",
                    output_view.len(),
                    view_shape,
                    batch_len,
                    self.num_classes,
                    expected
                )));
            }
            let logits: Vec<f32> = output_view.iter().copied().collect();
            drop(outputs);
            drop(guard);
            if !logits.iter().all(|x| x.is_finite()) {
                return Err(SparrowEngineError::Ort(
                    "raw-audio classifier returned non-finite logits".to_string(),
                ));
            }

            // ----- postprocess: softmax + top-K --------------------------
            for (i, &seg_offset) in batch_offsets.iter().enumerate() {
                let window_logits = &logits[i * self.num_classes..(i + 1) * self.num_classes];
                let probs = softmax(window_logits);
                let top = top_k_indices(&probs, top_k);
                let classes: Vec<AudioClass> = top
                    .into_iter()
                    .map(|(idx, p)| AudioClass {
                        class_idx: idx as u32,
                        label: labels.get(idx).cloned(),
                        probability: p,
                    })
                    .collect();
                let top1_prob = classes.first().map(|c| c.probability).unwrap_or(0.0);

                let (start_s, end_s) = preprocess_audio::segment_time_range(
                    seg_offset,
                    segment_samples,
                    total_samples,
                    self.sample_rate,
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
        }

        let elapsed = start.elapsed();
        Ok(AudioDetectResult {
            segments,
            duration_s,
            sample_rate: self.sample_rate,
            processing_time_ms: elapsed.as_secs_f32() * 1000.0,
        })
    }
}

// ---------------------------------------------------------------------------
// Local helpers — softmax + top-K (mirror sparrow-engine-cpu/src/detect_audio.rs).
// ---------------------------------------------------------------------------

fn softmax(logits: &[f32]) -> Vec<f32> {
    if logits.is_empty() {
        return Vec::new();
    }
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

fn top_k_indices(probs: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    indexed.truncate(k);
    indexed
}
