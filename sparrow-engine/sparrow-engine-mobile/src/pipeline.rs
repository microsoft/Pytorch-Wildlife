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
use crate::timing;

/// Default stage-1 gate threshold when the detector manifest omits one.
const DEFAULT_DETECTOR_THRESHOLD: f32 = 0.5;

/// A validated two-stage audio cascade ready to run.
pub struct MobilePipeline {
    pub id: String,
    detector: Rc<LoadedModel>,
    /// Stage-2 classifier. `None` for a detector-only pipeline (stage 2 disabled).
    classifier: Option<Rc<LoadedModel>>,
    config: AudioPreprocessConfig,
    filterbank: MelFilterbank,
    detector_threshold: f32,
    segment_duration_s: f32,
    segment_stride_s: f32,
}

/// Options for [`crate::engine::Engine::run_pipeline`]. `None` fields fall back
/// to the detector manifest's sliding-window parameters / confidence threshold.
#[derive(Debug, Clone, Default)]
pub struct CascadeOpts {
    /// Sliding-window length in seconds.
    pub window_sec: Option<f32>,
    /// Sliding-window overlap in seconds (must be < window). May be negative for a
    /// gapped/duty-cycled window — the cadence is `window - overlap`.
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
    // Stage 2 (classifier) is OPTIONAL: a pipeline with only a detector step runs
    // detector-only (stage 2 disabled) — the single-stage mode that emits
    // per-window detection results with no ecotype. A 2-stage cascade still runs
    // exactly one classifier; reject 2+ classifier steps (the mobile cascade runs
    // exactly one, and a multi-classifier manifest would silently run only the
    // first). PipelineRole is {Detector, Classifier}, so 0-or-1 classifier plus
    // the single detector pins the step count at one or two.
    let classifier_id = manifest
        .steps
        .iter()
        .find(|s| s.role == PipelineRole::Classifier)
        .map(|s| s.model.as_str());

    let classifier_steps = manifest
        .steps
        .iter()
        .filter(|s| s.role == PipelineRole::Classifier)
        .count();
    if classifier_steps > 1 {
        bail!(
            "pipeline '{id}' has {classifier_steps} classifier steps; the mobile audio cascade \
             runs at most one stage-2 classifier (or none, for a detector-only pipeline)"
        );
    }

    let detector = inner.load_model(detector_id)?;
    let classifier = match classifier_id {
        Some(cid) => Some(inner.load_model(cid)?),
        None => None,
    };

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
    if let Some(ref classifier) = classifier {
        if classifier.model_type != ModelType::AudioClassifier {
            bail!(
                "pipeline '{id}' stage 2 model '{}' is {:?}, expected an AudioClassifier \
                 (mel_spectrogram + softmax)",
                classifier.id,
                classifier.model_type
            );
        }
    }

    let config = AudioPreprocessConfig::from_manifest(&detector.manifest.preprocess_method)
        .ok_or_else(|| anyhow!("detector '{}' is not a mel audio model", detector.id))?;
    config.validate().map_err(|e| anyhow!("{e}"))?;

    // When stage 2 is present, both stages must share one mel front-end (that is
    // the whole point of the mel-input ecotype re-export); reject a mismatch
    // loudly. A detector-only pipeline has no second stage to match.
    if let Some(ref classifier) = classifier {
        let classifier_config =
            AudioPreprocessConfig::from_manifest(&classifier.manifest.preprocess_method)
                .ok_or_else(|| anyhow!("classifier '{}' is not a mel audio model", classifier.id))?;
        if !same_mel_config(&config, &classifier_config) {
            bail!(
                "pipeline '{id}' stages do not share an identical mel front-end; the cascade \
                 requires both stages to consume the same dB-mel"
            );
        }
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

/// Per-window timing accumulator for the `SPE_TIMING` E2 batching micro-benchmark.
/// Sums (not means) are kept; `emit` divides by the window count. Detector spans
/// are tracked separately so the GO/NO-GO "fixed-overhead fraction" can be read at
/// the aggregate line without re-parsing the per-window records.
#[derive(Default)]
struct TimingWindows {
    n: u64,
    gated: u64,
    mel_ns: u128,
    det_ns: u128,
    det_setup_ns: u128,
    det_run_ns: u128,
    det_read_ns: u128,
    eco_ns: u128,
    resid_ns: u128,
    win_ns: u128,
}

impl TimingWindows {
    #[allow(clippy::too_many_arguments)]
    fn add(
        &mut self,
        mel_ns: u128,
        det_ns: u128,
        det_spans: (u128, u128, u128),
        stage2_ran: bool,
        eco_ns: u128,
        resid_ns: u128,
        win_ns: u128,
    ) {
        self.n += 1;
        if stage2_ran {
            self.gated += 1;
        }
        self.mel_ns += mel_ns;
        self.det_ns += det_ns;
        self.det_setup_ns += det_spans.0;
        self.det_run_ns += det_spans.1;
        self.det_read_ns += det_spans.2;
        self.eco_ns += eco_ns;
        self.resid_ns += resid_ns;
        self.win_ns += win_ns;
    }

    /// Emit the aggregate `SPE_TIMING_AGG` line: per-window means (ms) plus the
    /// detector invoke's fixed-overhead fraction = (setup + readout) / invoke,
    /// the quantity the E2 decision tree thresholds for the batching GO/NO-GO.
    fn emit(&self) {
        if self.n == 0 {
            return;
        }
        let n = self.n as f64;
        let mean = |x: u128| timing::ns_ms(x) / n;
        let det_invoke = (self.det_setup_ns + self.det_run_ns + self.det_read_ns).max(1) as f64;
        let det_fixed = (self.det_setup_ns + self.det_read_ns) as f64;
        eprintln!(
            "SPE_TIMING_AGG windows={} gated={} mel_ms_mean={:.3} det_ms_mean={:.3} \
             det_setup_ms_mean={:.3} det_run_ms_mean={:.3} det_read_ms_mean={:.3} \
             eco_ms_mean={:.3} resid_ms_mean={:.3} win_ms_mean={:.3} \
             det_fixed_overhead_frac={:.3}",
            self.n,
            self.gated,
            mean(self.mel_ns),
            mean(self.det_ns),
            mean(self.det_setup_ns),
            mean(self.det_run_ns),
            mean(self.det_read_ns),
            mean(self.eco_ns),
            mean(self.resid_ns),
            mean(self.win_ns),
            det_fixed / det_invoke,
        );
    }
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
    if !window_sec.is_finite() || window_sec <= 0.0 {
        bail!("window_sec must be finite and > 0 (got {window_sec})");
    }
    if !overlap_sec.is_finite() || overlap_sec >= window_sec {
        bail!("overlap_sec ({overlap_sec}) must be finite and < window_sec ({window_sec})");
    }
    if let Some(t) = opts.detector_threshold {
        if !t.is_finite() || !(0.0..=1.0).contains(&t) {
            bail!("detector_threshold ({t}) must be finite and in [0, 1]");
        }
    }
    let detector_threshold = opts.detector_threshold.unwrap_or(pipeline.detector_threshold);

    // Bound the sample count before the float→usize cast: an out-of-range
    // `window_sec` would otherwise saturate to usize::MAX and panic the later
    // `Vec::resize`. (window_sec is already finite + positive here.)
    let segment_samples_f = (window_sec * target_sr as f32).round();
    if !(0.0..=usize::MAX as f32).contains(&segment_samples_f) {
        bail!("window_sec ({window_sec}) resolves to too many samples");
    }
    let segment_samples = segment_samples_f as usize;
    let stride_samples = (((window_sec - overlap_sec) * target_sr as f32).round() as usize).max(1);
    if segment_samples == 0 {
        bail!("window_sec resolves to zero samples");
    }

    let mut detector_backend = pipeline.detector.backend.borrow_mut();
    let mut classifier_backend = pipeline
        .classifier
        .as_ref()
        .map(|c| c.backend.borrow_mut());

    // The mel's `orig_sample_rate` is the input's ORIGINAL rate (before the
    // whole-buffer resample to `target_sr`), matching the proven OrcaCascade —
    // it drives `fill_highfreq`. For already-target-rate input (the deployed
    // water-sparrow path resamples to 24 kHz first) it equals `target_sr`.
    let orig_sr = audio.orig_sample_rate;
    let mut num_stage2_classes = 0usize;
    let mut segments = Vec::new();

    let timed = timing::enabled();
    let mut tw = TimingWindows::default();

    for offset in compute_segment_offsets(total, segment_samples, stride_samples) {
        let w_start = timed.then(Instant::now);
        // Compute the dB-mel ONCE for this window and feed both stages.
        let t_mel = timed.then(Instant::now);
        let mel_bytes = mel_bytes_for_segment(
            &audio.data,
            offset,
            segment_samples,
            orig_sr,
            &pipeline.config,
            &pipeline.filterbank,
        )?;
        let mel_ns = t_mel.map(|t| t.elapsed().as_nanos()).unwrap_or(0);
        let (start_s, end_s) = segment_time_range(offset, segment_samples, total, target_sr);

        if timed {
            timing::reset_invoke();
        }
        let t_det = timed.then(Instant::now);
        let detector_out = detector_backend
            .invoke_single(mel_bytes.clone(), LiteRtElementType::kLiteRtElementTypeFloat32)?;
        let det_ns = t_det.map(|t| t.elapsed().as_nanos()).unwrap_or(0);
        let det_spans = if timed { timing::take_invoke() } else { (0, 0, 0) };
        // Per-window ecotype timing — filled when stage 2 runs below.
        let mut eco_ns = 0u128;
        let mut eco_spans = (0u128, 0u128, 0u128);
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
            // Stage 2 runs only when a classifier is present (a detector-only
            // pipeline leaves `stage2_ran = false` and empty probabilities).
            if let (Some(classifier), Some(classifier_backend)) =
                (pipeline.classifier.as_ref(), classifier_backend.as_mut())
            {
                if timed {
                    timing::reset_invoke();
                }
                let t_eco = timed.then(Instant::now);
                let classifier_out = classifier_backend
                    .invoke_single(mel_bytes, LiteRtElementType::kLiteRtElementTypeFloat32)?;
                eco_ns = t_eco.map(|t| t.elapsed().as_nanos()).unwrap_or(0);
                eco_spans = if timed { timing::take_invoke() } else { (0, 0, 0) };
                let logits = classifier_out
                    .into_iter()
                    .next()
                    .context("classifier returned no logits")?;
                if logits.is_empty() {
                    bail!("classifier '{}' returned empty logits", classifier.id);
                }
                if num_stage2_classes == 0 {
                    num_stage2_classes = logits.len();
                } else if logits.len() != num_stage2_classes {
                    // Fail fast (matching the proven OrcaCascade) rather than silently
                    // zero-filling an inconsistent per-window class count.
                    bail!(
                        "classifier '{}' returned {} logits, expected {} (inconsistent per-window \
                         class count)",
                        classifier.id,
                        logits.len(),
                        num_stage2_classes
                    );
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
        }

        if timed {
            let win_ns = w_start.map(|t| t.elapsed().as_nanos()).unwrap_or(0);
            let resid_ns = win_ns.saturating_sub(mel_ns + det_ns + eco_ns);
            eprintln!(
                "SPE_TIMING win={} det_prob={:.4} gated={} mel_ms={:.3} \
                 det_ms={:.3} det_setup_ms={:.3} det_run_ms={:.3} det_read_ms={:.3} \
                 eco_ms={:.3} eco_setup_ms={:.3} eco_run_ms={:.3} eco_read_ms={:.3} \
                 resid_ms={:.3} win_ms={:.3}",
                tw.n,
                detector_probability,
                seg.stage2_ran as u8,
                timing::ns_ms(mel_ns),
                timing::ns_ms(det_ns),
                timing::ns_ms(det_spans.0),
                timing::ns_ms(det_spans.1),
                timing::ns_ms(det_spans.2),
                timing::ns_ms(eco_ns),
                timing::ns_ms(eco_spans.0),
                timing::ns_ms(eco_spans.1),
                timing::ns_ms(eco_spans.2),
                timing::ns_ms(resid_ns),
                timing::ns_ms(win_ns),
            );
            tw.add(
                mel_ns,
                det_ns,
                det_spans,
                seg.stage2_ran,
                eco_ns,
                resid_ns,
                win_ns,
            );
        }
        segments.push(seg);
    }

    if timed {
        tw.emit();
    }

    // If no window fired stage 2, fall back to the classifier's declared class
    // count so consumers can still size their probability buffers. A detector-only
    // pipeline has no classifier, so the count stays 0.
    if num_stage2_classes == 0 {
        num_stage2_classes = pipeline
            .classifier
            .as_ref()
            .map(|c| c.labels.len())
            .unwrap_or(0);
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
