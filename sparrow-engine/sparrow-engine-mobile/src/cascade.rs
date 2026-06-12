//! Orca two-stage mobile cascade.

use crate::sys;
use crate::tflite::{LiteRtBackend, LiteRtRuntime};
use anyhow::{bail, Context, Result};
use ndarray::Array4;
use sparrow_engine_core::preprocess_audio::{
    load_audio_at_sample_rate, mel_spectrogram, AudioPreprocessConfig, MelFilterbank,
};
use sparrow_engine_types::AudioInput;
use std::path::Path;

/// Orca detector manifest preprocessing constants.
///
/// Source: `.zenodo-staging/orca-dclde2026-onboarding-workdir/
/// orca-detector-dclde2026-v1/manifest.toml`, `[preprocessing]`.
pub const ORCA_SAMPLE_RATE: u32 = 24_000;
pub const ORCA_SEGMENT_SAMPLES: usize = 72_000;
pub const ORCA_N_FFT: u32 = 1_024;
pub const ORCA_HOP_LENGTH: u32 = 128;
pub const ORCA_N_MELS: u32 = 256;
pub const ORCA_FMIN: f32 = 200.0;
pub const ORCA_FMAX: f32 = 12_000.0;
pub const ORCA_TOP_DB: f32 = 80.0;
pub const ORCA_THRESHOLD: f32 = 0.5;

/// Two-stage orca cascade output for one 3 s segment.
#[derive(Debug, Clone)]
pub struct OrcaCascadeResult {
    pub detector_logit: f32,
    pub detector_probability: f32,
    pub is_orca: bool,
    pub ecotype_logits: Option<Vec<f32>>,
    pub ecotype_probabilities: Option<Vec<f32>>,
    pub ecotype_argmax: Option<usize>,
}

/// Mobile two-stage orca cascade.
pub struct OrcaCascade {
    _runtime: LiteRtRuntime,
    detector: LiteRtBackend,
    ecotype: LiteRtBackend,
    audio_config: AudioPreprocessConfig,
    filterbank: MelFilterbank,
}

impl OrcaCascade {
    /// Load detector and ecotype TFLite models into one shared LiteRT runtime.
    pub fn load(detector_path: &Path, ecotype_path: &Path, num_threads: usize) -> Result<Self> {
        let runtime = LiteRtRuntime::new()?;
        let detector = runtime
            .load(detector_path, num_threads)
            .with_context(|| format!("load detector {}", detector_path.display()))?;
        let ecotype = runtime
            .load(ecotype_path, num_threads)
            .with_context(|| format!("load ecotype {}", ecotype_path.display()))?;
        let audio_config = orca_audio_config();
        let filterbank = MelFilterbank::new(&audio_config)?;
        Ok(Self {
            _runtime: runtime,
            detector,
            ecotype,
            audio_config,
            filterbank,
        })
    }

    /// Run one raw-audio window through core mel preprocessing and the cascade.
    ///
    /// This per-segment API operates on a single 3 s window. Input is resampled
    /// to 24 kHz, then truncated to or zero-padded to 72,000 samples. The caller
    /// is responsible for sliding-window segmentation before calling this method.
    pub fn run_segment(&mut self, samples: &[f32], sample_rate: u32) -> Result<OrcaCascadeResult> {
        let mel = orca_mel_spectrogram(samples, sample_rate, &self.audio_config, &self.filterbank)?;
        self.run_mel(&mel)
    }

    /// Run a precomputed core mel tensor through detector and, when positive, ecotype.
    pub fn run_mel(&mut self, mel: &Array4<f32>) -> Result<OrcaCascadeResult> {
        let mel_bytes = nchw_mel_to_nhwc_le_bytes(mel)?;
        let detector_outputs = self.detector.invoke_named(&[(
            "input",
            mel_bytes.clone(),
            sys::LiteRtElementType::kLiteRtElementTypeFloat32,
        )])?;
        let detector_logit = *detector_outputs
            .first()
            .and_then(|v| v.first())
            .context("detector returned no logit")?;
        let detector_probability = sigmoid(detector_logit);
        let is_orca = detector_probability >= ORCA_THRESHOLD;

        if !is_orca {
            return Ok(OrcaCascadeResult {
                detector_logit,
                detector_probability,
                is_orca,
                ecotype_logits: None,
                ecotype_probabilities: None,
                ecotype_argmax: None,
            });
        }

        let ecotype_outputs = self.ecotype.invoke_named(&[(
            "mel",
            mel_bytes,
            sys::LiteRtElementType::kLiteRtElementTypeFloat32,
        )])?;
        let ecotype_logits = ecotype_outputs
            .into_iter()
            .next()
            .context("ecotype returned no logits")?;
        let ecotype_argmax = argmax(&ecotype_logits).context("ecotype logits were empty")?;
        let ecotype_probabilities = softmax(&ecotype_logits);
        Ok(OrcaCascadeResult {
            detector_logit,
            detector_probability,
            is_orca,
            ecotype_logits: Some(ecotype_logits),
            ecotype_probabilities: Some(ecotype_probabilities),
            ecotype_argmax: Some(ecotype_argmax),
        })
    }
}

pub fn orca_audio_config() -> AudioPreprocessConfig {
    AudioPreprocessConfig {
        sample_rate: ORCA_SAMPLE_RATE,
        n_fft: ORCA_N_FFT,
        hop_length: ORCA_HOP_LENGTH,
        n_mels: ORCA_N_MELS,
        fmin: ORCA_FMIN,
        fmax: ORCA_FMAX,
        top_db: ORCA_TOP_DB,
        fill_highfreq: true,
    }
}

pub fn orca_mel_spectrogram(
    samples: &[f32],
    sample_rate: u32,
    config: &AudioPreprocessConfig,
    filterbank: &MelFilterbank,
) -> Result<Array4<f32>> {
    let audio = load_audio_at_sample_rate(
        &AudioInput::Samples {
            data: samples.to_vec(),
            sample_rate,
        },
        ORCA_SAMPLE_RATE,
    )?;
    let mut segment = audio.data;
    if segment.len() > ORCA_SEGMENT_SAMPLES {
        segment.truncate(ORCA_SEGMENT_SAMPLES);
    } else if segment.len() < ORCA_SEGMENT_SAMPLES {
        segment.resize(ORCA_SEGMENT_SAMPLES, 0.0);
    }
    Ok(mel_spectrogram(
        &segment,
        audio.orig_sample_rate,
        config,
        filterbank,
    )?)
}

/// Convert core NCHW `[1, 1, 256, 555]` mel to TFLite NHWC bytes.
///
/// With one channel, NCHW and NHWC have the same contiguous element order:
/// both flatten as `mel_bin` outer, `time_frame` inner.
pub fn nchw_mel_to_nhwc_le_bytes(mel: &Array4<f32>) -> Result<Vec<u8>> {
    let shape = mel.shape();
    if shape.len() != 4 || shape[0] != 1 || shape[1] != 1 {
        bail!("expected mel shape [1, 1, n_mels, frames], got {shape:?}");
    }
    let n_mels = shape[2];
    let frames = shape[3];
    let mut bytes = Vec::with_capacity(n_mels * frames * std::mem::size_of::<f32>());
    for m in 0..n_mels {
        for t in 0..frames {
            bytes.extend_from_slice(&mel[[0, 0, m, t]].to_le_bytes());
        }
    }
    Ok(bytes)
}

pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

pub fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits
        .iter()
        .copied()
        .fold(f32::NEG_INFINITY, |a, b| a.max(b));
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|x| x / sum).collect()
}

pub fn argmax(values: &[f32]) -> Option<usize> {
    values
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
}
