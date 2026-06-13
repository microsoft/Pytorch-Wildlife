//! Audio-cascade pipeline (RP-25-FU-1).
//!
//! The orca cascade — "is there a whale call? (stage 1 detector) → if so, which
//! ecotype? (stage 2 classifier)" — is described by a `pipeline.toml` and run by
//! this module, instead of being hardcoded C. It is the audio counterpart of the
//! cpu/gpu image pipeline (detect → crop → classify): the cpu pipeline is
//! image-only and its `validate_pipeline_compat` matrix rejects audio cascades,
//! so the mobile flavor validates and runs the cascade locally.
//!
//! Both stages share one mel front-end (computed once per window) and stage 2
//! runs only when stage 1 fires — the share-one-front-end + skip-stage-2
//! efficiency that keeps the cascade within the Pi Zero 2W budget.

use std::rc::Rc;
use std::time::Instant;

use anyhow::{anyhow, bail, Context, Result};

use sparrow_engine_core::preprocess_audio::{
    compute_segment_offsets, load_audio_at_sample_rate, segment_time_range, AudioPreprocessConfig,
    MelFilterbank,
};
use sparrow_engine_types::manifest::{
    self, InferenceStrategy, PipelineRole, PostprocessMethod,
};
use sparrow_engine_types::types::{AudioInput, ModelType};

use crate::cascade::{argmax, sigmoid, softmax};
use crate::engine::{mel_bytes_for_segment, EngineInner, LoadedModel};
use crate::sys::LiteRtElementType;

/// Default stage-1 gate threshold when the detector manifest omits one.
const DEFAULT_DETECTOR_THRESHOLD: f32 = 0.5;

/// A validated two-stage audio cascade ready to run.
pub struct MobilePipeline {
    pub id: String,
    detector: Rc<LoadedModel>,
    classifier: Rc<LoadedModel>,
    config: AudioPreprocessConfig,
    filterbank: MelFilterbank,
    detector_threshold: f32,
    segment_duration_s: f32,
    segment_stride_s: f32,
}

/// Options for [`run_pipeline`]. `None` fields fall back to the detector
/// manifest's sliding-window parameters / confidence threshold.
#[derive(Debug, Clone, Default)]
pub struct CascadeOpts {
    /// Sliding-window length in seconds.
    pub window_sec: Option<f32>,
    /// Sliding-window overlap in seconds (must be < window).
    pub overlap_sec: Option<f32>,
    /// Stage-1 gate threshold override.
    pub detector_threshold: Option<f32>,
}

/// One cascade segment (one sliding window).
#[derive(Debug, Clone)]
pub struct CascadeSegment {
    pub start_s: f32,
    pub end_s: f32,
    /// Raw stage-1 detector logit.
    pub detector_logit: f32,
    /// Sigmoid of the detector logit.
    pub detector_probability: f32,
    /// Whether stage 1 fired (probability >= threshold).
    pub is_detected: bool,
    /// Whether stage 2 ran (only when `is_detected`).
    pub stage2_ran: bool,
    /// Stage-2 argmax class index, or `None` when stage 2 did not run.
    pub stage2_argmax: Option<usize>,
    /// Stage-2 top probability, or `0.0` when stage 2 did not run.
    pub stage2_confidence: f32,
    /// Stage-2 per-class probabilities (length = `num_stage2_classes`), or empty
    /// when stage 2 did not run.
    pub stage2_probabilities: Vec<f32>,
}

/// Full audio-cascade output.
#[derive(Debug, Clone)]
pub struct CascadeResult {
    pub pipeline_id: String,
    pub segments: Vec<CascadeSegment>,
    /// Number of stage-2 classes (constant across segments).
    pub num_stage2_classes: usize,
    pub duration_s: f32,
    pub sample_rate: u32,
    pub processing_time_ms: f32,
}

/// Load a cascade pipeline by id from `{model_dir}/{id}/pipeline.toml`.
pub(crate) fn load_pipeline_by_id(inner: &EngineInner, id: &str) -> Result<()> {
    inner.check_thread()?;

    let pipeline_path = inner.model_dir().join(id).join("pipeline.toml");
    let manifest = manifest::load_pipeline_manifest(&pipeline_path)
        .map_err(|e| anyhow!("load pipeline {}: {e}", pipeline_path.display()))?;

    let detector_id = manifest
        .steps
        .iter()
        .find(|s| s.role == PipelineRole::Detector)
        .map(|s| s.model.as_str())
        .context("pipeline has no detector step")?;
    let classifier_id = manifest
        .steps
        .iter()
        .find(|s| s.role == PipelineRole::Classifier)
        .map(|s| s.model.as_str())
        .ok_or_else(|| {
            anyhow!(
                "pipeline '{id}' has no classifier step; the mobile audio cascade requires a \
                 stage-1 detector and a stage-2 classifier"
            )
        })?;

    let detector = inner.load_model(detector_id)?;
    let classifier = inner.load_model(classifier_id)?;

    // Mobile-local validation: the cpu/gpu `validate_pipeline_compat` matrix is
    // image-only and rejects an AudioDetector→AudioClassifier pair as a "modality
    // mismatch". The mobile audio cascade is exactly that pair.
    if detector.model_type != ModelType::AudioDetector {
        bail!(
            "pipeline '{id}' stage 1 model '{}' is {:?}, expected an AudioDetector \
             (mel_spectrogram + sigmoid)",
            detector.id,
            detector.model_type
        );
    }
    if classifier.model_type != ModelType::AudioClassifier {
        bail!(
            "pipeline '{id}' stage 2 model '{}' is {:?}, expected an AudioClassifier \
             (mel_spectrogram + softmax)",
            classifier.id,
            classifier.model_type
        );
    }

    let config = AudioPreprocessConfig::from_manifest(&detector.manifest.preprocess_method)
        .ok_or_else(|| anyhow!("detector '{}' is not a mel audio model", detector.id))?;
    config.validate().map_err(|e| anyhow!("{e}"))?;

    // Both stages must share one mel front-end (that is the whole point of the
    // mel-input ecotype re-export); reject a mismatch loudly.
    let classifier_config =
        AudioPreprocessConfig::from_manifest(&classifier.manifest.preprocess_method)
            .ok_or_else(|| anyhow!("classifier '{}' is not a mel audio model", classifier.id))?;
    if !same_mel_config(&config, &classifier_config) {
        bail!(
            "pipeline '{id}' stages do not share an identical mel front-end; the cascade requires \
             both stages to consume the same dB-mel"
        );
    }

    let detector_threshold = match &detector.manifest.postprocess_method {
        PostprocessMethod::Sigmoid {
            confidence_threshold,
        } => *confidence_threshold,
        _ => detector
            .manifest
            .confidence_threshold
            .unwrap_or(DEFAULT_DETECTOR_THRESHOLD),
    };

    let (segment_duration_s, segment_stride_s) = match detector.manifest.inference_strategy {
        InferenceStrategy::SlidingWindow {
            segment_duration_s,
            segment_stride_s,
        } => (segment_duration_s, segment_stride_s),
        _ => bail!(
            "pipeline '{id}' detector '{}' has no sliding-window inference strategy",
            detector.id
        ),
    };

    let filterbank = MelFilterbank::new(&config).map_err(|e| anyhow!("{e}"))?;

    let pipeline = MobilePipeline {
        id: id.to_string(),
        detector,
        classifier,
        config,
        filterbank,
        detector_threshold,
        segment_duration_s,
        segment_stride_s,
    };
    inner
        .pipelines()
        .borrow_mut()
        .insert(id.to_string(), Rc::new(pipeline));
    Ok(())
}

/// Run a loaded cascade over an audio input (WAV file or raw mono samples).
pub(crate) fn run_pipeline(
    inner: &EngineInner,
    pipeline_id: &str,
    input: &AudioInput,
    opts: &CascadeOpts,
) -> Result<CascadeResult> {
    inner.check_thread()?;
    let start = Instant::now();

    let pipeline = inner
        .pipelines()
        .borrow()
        .get(pipeline_id)
        .cloned()
        .ok_or_else(|| anyhow!("pipeline '{pipeline_id}' is not loaded"))?;

    let target_sr = pipeline.config.sample_rate;
    // Resample the whole buffer to the model rate ONCE, then window — matches the
    // proven OrcaCascade + CLI contract (resample-before-windowing).
    let audio = load_audio_at_sample_rate(input, target_sr).map_err(|e| anyhow!("{e}"))?;
    let total = audio.data.len();
    let duration_s = total as f32 / target_sr as f32;

    let window_sec = opts.window_sec.unwrap_or(pipeline.segment_duration_s);
    let overlap_sec = opts
        .overlap_sec
        .unwrap_or(pipeline.segment_duration_s - pipeline.segment_stride_s);
    if window_sec <= 0.0 {
        bail!("window_sec must be > 0");
    }
    if overlap_sec >= window_sec {
        bail!("overlap_sec ({overlap_sec}) must be < window_sec ({window_sec})");
    }
    let detector_threshold = opts.detector_threshold.unwrap_or(pipeline.detector_threshold);

    let segment_samples = (window_sec * target_sr as f32).round() as usize;
    let stride_samples = (((window_sec - overlap_sec) * target_sr as f32).round() as usize).max(1);
    if segment_samples == 0 {
        bail!("window_sec resolves to zero samples");
    }

    let mut detector_backend = pipeline.detector.backend.borrow_mut();
    let mut classifier_backend = pipeline.classifier.backend.borrow_mut();

    // The mel's `orig_sample_rate` is the input's ORIGINAL rate (before the
    // whole-buffer resample to `target_sr`), matching the proven OrcaCascade —
    // it drives `fill_highfreq`. For already-target-rate input (the deployed
    // water-sparrow path resamples to 24 kHz first) it equals `target_sr`.
    let orig_sr = audio.orig_sample_rate;
    let mut num_stage2_classes = 0usize;
    let mut segments = Vec::new();
    for offset in compute_segment_offsets(total, segment_samples, stride_samples) {
        // Compute the dB-mel ONCE for this window and feed both stages.
        let mel_bytes = mel_bytes_for_segment(
            &audio.data,
            offset,
            segment_samples,
            orig_sr,
            &pipeline.config,
            &pipeline.filterbank,
        )?;
        let (start_s, end_s) = segment_time_range(offset, segment_samples, total, target_sr);

        let detector_out = detector_backend
            .invoke_single(mel_bytes.clone(), LiteRtElementType::kLiteRtElementTypeFloat32)?;
        let detector_logit = *detector_out
            .first()
            .and_then(|v| v.first())
            .context("detector returned no logit")?;
        let detector_probability = sigmoid(detector_logit);
        let is_detected = detector_probability >= detector_threshold;

        let mut seg = CascadeSegment {
            start_s,
            end_s,
            detector_logit,
            detector_probability,
            is_detected,
            stage2_ran: false,
            stage2_argmax: None,
            stage2_confidence: 0.0,
            stage2_probabilities: Vec::new(),
        };

        if is_detected {
            let classifier_out = classifier_backend
                .invoke_single(mel_bytes, LiteRtElementType::kLiteRtElementTypeFloat32)?;
            let logits = classifier_out
                .into_iter()
                .next()
                .context("classifier returned no logits")?;
            if num_stage2_classes == 0 {
                num_stage2_classes = logits.len();
            }
            let probs = softmax(&logits);
            seg.stage2_ran = true;
            seg.stage2_argmax = argmax(&probs);
            seg.stage2_confidence = seg
                .stage2_argmax
                .and_then(|i| probs.get(i).copied())
                .unwrap_or(0.0);
            seg.stage2_probabilities = probs;
        }

        segments.push(seg);
    }

    // If no window fired stage 2, fall back to the classifier's declared class
    // count so consumers can still size their probability buffers.
    if num_stage2_classes == 0 {
        num_stage2_classes = pipeline.classifier.labels.len();
    }

    Ok(CascadeResult {
        pipeline_id: pipeline_id.to_string(),
        segments,
        num_stage2_classes,
        duration_s,
        sample_rate: target_sr,
        processing_time_ms: start.elapsed().as_secs_f32() * 1000.0,
    })
}

/// Two mel configs are interchangeable for the cascade when every field matches.
fn same_mel_config(a: &AudioPreprocessConfig, b: &AudioPreprocessConfig) -> bool {
    a.sample_rate == b.sample_rate
        && a.n_fft == b.n_fft
        && a.hop_length == b.hop_length
        && a.n_mels == b.n_mels
        && a.fmin == b.fmin
        && a.fmax == b.fmax
        && a.top_db == b.top_db
        && a.fill_highfreq == b.fill_highfreq
}
