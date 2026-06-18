//! Audio preprocessing: WAV decode, resampling, mel spectrogram computation.
//!
//! Transforms an [`AudioInput`] into an `ndarray::Array4<f32>` tensor ready for
//! ONNX Runtime inference. Implements the verified parameter set from
//! empirical testing (docs/design/audio/).
//!
//! Pipeline per segment: WAV decode → resample → STFT → mel filterbank → dB → tensor.

use std::path::Path;
use std::time::Instant;

use ndarray::Array4;
use realfft::RealFftPlanner;
use rubato::{FftFixedInOut, Resampler};

use sparrow_engine_types::manifest::PreprocessMethod;
use sparrow_engine_types::AudioInput;
use sparrow_engine_types::{SparrowEngineError, Result};

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Audio preprocessing parameters.
///
/// Default values match MD_AudioBirds_V1.onnx (verified empirically in
/// docs/design/audio/reviewer_sparrow_r3.md).
#[derive(Debug, Clone)]
pub struct AudioPreprocessConfig {
    pub sample_rate: u32,
    pub n_fft: u32,
    pub hop_length: u32,
    pub n_mels: u32,
    pub fmin: f32,
    pub fmax: f32,
    pub top_db: f32,
    /// Opt-in high-frequency mel-band fill for upsampled inputs (RP-27,
    /// 2026-06-01). Default `false` preserves md-audiobirds-v1 behavior.
    /// See [`mel_spectrogram`] for the algorithm details.
    pub fill_highfreq: bool,
}

impl AudioPreprocessConfig {
    /// Validate static audio preprocessing parameters before deriving buffer sizes.
    pub fn validate(&self) -> Result<()> {
        if self.sample_rate == 0 {
            return Err(SparrowEngineError::AudioPreprocess(
                "audio sample_rate must be greater than 0".to_string(),
            ));
        }
        if self.n_fft < 2 {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "audio n_fft must be at least 2, got {}",
                self.n_fft
            )));
        }
        if self.hop_length == 0 {
            return Err(SparrowEngineError::AudioPreprocess(
                "audio hop_length must be greater than 0".to_string(),
            ));
        }
        if self.n_mels == 0 {
            return Err(SparrowEngineError::AudioPreprocess(
                "audio n_mels must be greater than 0".to_string(),
            ));
        }
        if !self.fmin.is_finite()
            || !self.fmax.is_finite()
            || self.fmin < 0.0
            || self.fmax <= self.fmin
        {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "audio frequency bounds must be finite with 0 <= fmin < fmax, got fmin={} fmax={}",
                self.fmin, self.fmax
            )));
        }
        let nyquist = self.sample_rate as f32 / 2.0;
        if self.fmax > nyquist {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "audio fmax must not exceed Nyquist frequency {nyquist}, got {}",
                self.fmax
            )));
        }
        if !self.top_db.is_finite() || self.top_db <= 0.0 {
            return Err(SparrowEngineError::AudioPreprocess(format!(
                "audio top_db must be finite and positive, got {}",
                self.top_db
            )));
        }
        Ok(())
    }

    /// Construct from a manifest's `PreprocessMethod::MelSpectrogram`.
    ///
    /// Returns `None` if the method is not `MelSpectrogram`.
    ///
    /// Note: `window`, `mel_scale`, and `filter_norm` are validated at manifest
    /// load time (only "hann_symmetric", "slaney", "slaney" are accepted). They
    /// are not carried in `AudioPreprocessConfig` because the DSP implementation
    /// hardcodes these algorithms — symmetric Hann window, Slaney mel scale,
    /// Slaney filter normalization. If additional algorithms are added,
    /// propagate the values here and dispatch accordingly.
    ///
    /// Phase 3.8 Step 2 Wave 0a (F0.8 corrective fix, 2026-05-04): switched
    /// HTK → Slaney + area → slaney to match `MD_AudioBirds_V1` training (PW
    /// Bioacoustics `mel_scale="slaney"` + `norm="slaney"`). See
    /// `docs/research/phase3.8/step2/cpu_pre_fix_log.md` for drift details.
    pub fn from_manifest(method: &PreprocessMethod) -> Option<Self> {
        match method {
            PreprocessMethod::MelSpectrogram {
                sample_rate,
                n_fft,
                hop_length,
                n_mels,
                fmin,
                fmax,
                top_db,
                fill_highfreq,
                .. // window, mel_scale, filter_norm: validated at load time, only one implementation exists
            } => Some(Self {
                sample_rate: *sample_rate,
                n_fft: *n_fft,
                hop_length: *hop_length,
                n_mels: *n_mels,
                fmin: *fmin,
                fmax: *fmax,
                top_db: *top_db,
                fill_highfreq: *fill_highfreq,
            }),
            _ => None,
        }
    }
}

impl Default for AudioPreprocessConfig {
    fn default() -> Self {
        Self {
            sample_rate: 48_000,
            n_fft: 2048,
            hop_length: 512,
            n_mels: 224,
            fmin: 0.0,
            fmax: 24_000.0,
            top_db: 80.0,
            fill_highfreq: false,
        }
    }
}

/// Decoded audio samples ready for segmentation and spectrogram computation.
pub struct AudioSamples {
    pub data: Vec<f32>,
    pub sample_rate: u32,
    pub duration_s: f32,
    /// Original sample rate of the source file, before resampling to `sample_rate`.
    /// Equal to `sample_rate` when no resample happened. Used by
    /// [`mel_spectrogram`] to drive the optional `fill_highfreq` step.
    pub orig_sample_rate: u32,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Load audio from input, decode WAV if needed, resample to target rate.
///
/// Emits tracing events with stage timings (`audio.decode`, `audio.resample`)
/// for the Step 2 Wave 0c bench harness. Set `RUST_LOG=sparrow_engine_core=info` to see
/// them.
pub fn load_audio(input: &AudioInput, config: &AudioPreprocessConfig) -> Result<AudioSamples> {
    config.validate()?;
    if config.sample_rate == 0 {
        return Err(SparrowEngineError::InvalidManifest(
            "audio config sample_rate must be greater than 0".to_string(),
        ));
    }
    load_audio_at_sample_rate(input, config.sample_rate)
}

/// Load audio and resample to the given target sample rate.
///
/// Slimmer sibling of [`load_audio`] used by raw-audio (non-mel) classifier
/// preprocessing paths (e.g. Perch 2). Skips the mel-spectrogram config
/// validation in [`AudioPreprocessConfig::validate`] — callers that need it
/// should call [`load_audio`].
pub fn load_audio_at_sample_rate(
    input: &AudioInput,
    target_sample_rate: u32,
) -> Result<AudioSamples> {
    if target_sample_rate == 0 {
        return Err(SparrowEngineError::InvalidManifest(
            "target_sample_rate must be greater than 0".to_string(),
        ));
    }
    let t_decode = Instant::now();
    let (samples, sr) = match input {
        AudioInput::FilePath(path) => decode_wav(path)?,
        AudioInput::Samples { data, sample_rate } => {
            if !data.iter().all(|sample| sample.is_finite()) {
                return Err(SparrowEngineError::AudioDecode(
                    "raw audio samples must be finite".to_string(),
                ));
            }
            (data.clone(), *sample_rate)
        }
    };
    if sr == 0 {
        return Err(SparrowEngineError::AudioDecode(
            "audio sample_rate must be greater than 0".to_string(),
        ));
    }
    tracing::info!(
        stage = "audio.decode",
        duration_ns = t_decode.elapsed().as_nanos() as u64,
        samples = samples.len(),
        sample_rate = sr,
    );

    // Phase 3.8 Step 2 perf-fix B (post Wave-4 triage): for SR-matched
    // input (the production case for DUNAS — 48 kHz native), move the
    // already-decoded `samples` directly without a `to_vec()` clone.
    let t_resample = Instant::now();
    let resampled = if sr == target_sample_rate {
        samples
    } else {
        resample(&samples, sr, target_sample_rate)?
    };
    tracing::info!(
        stage = "audio.resample",
        duration_ns = t_resample.elapsed().as_nanos() as u64,
        from_sr = sr,
        to_sr = target_sample_rate,
        out_samples = resampled.len(),
    );
    let duration_s = resampled.len() as f32 / target_sample_rate as f32;

    Ok(AudioSamples {
        data: resampled,
        sample_rate: target_sample_rate,
        duration_s,
        orig_sample_rate: sr,
    })
}

/// Validate resolved sliding-window audio options and return sample counts.
pub fn validate_audio_window_params(
    segment_duration_s: f32,
    stride_s: f32,
    confidence_threshold: f32,
    sample_rate: u32,
    n_fft: u32,
) -> Result<(usize, usize)> {
    if sample_rate == 0 {
        return Err(SparrowEngineError::InvalidManifest(
            "audio sample_rate must be greater than 0".to_string(),
        ));
    }
    if n_fft < 2 {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "audio n_fft must be at least 2, got {n_fft}"
        )));
    }
    if !segment_duration_s.is_finite() || segment_duration_s <= 0.0 {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "segment_duration_s must be finite and > 0, got {segment_duration_s}"
        )));
    }
    if !stride_s.is_finite() || stride_s <= 0.0 {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "stride_s must be finite and > 0, got {stride_s}"
        )));
    }
    if !confidence_threshold.is_finite() || !(0.0..=1.0).contains(&confidence_threshold) {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "confidence_threshold must be finite and in [0,1], got {confidence_threshold}"
        )));
    }

    let segment_samples_f = segment_duration_s * sample_rate as f32;
    let stride_samples_f = stride_s * sample_rate as f32;
    if !segment_samples_f.is_finite() || segment_samples_f > usize::MAX as f32 {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "segment_duration_s * sample_rate must produce a finite sample count, got {segment_samples_f}"
        )));
    }
    if !stride_samples_f.is_finite() || stride_samples_f > usize::MAX as f32 {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "stride_s * sample_rate must produce a finite sample count, got {stride_samples_f}"
        )));
    }
    let segment_samples = segment_samples_f.round() as usize;
    let stride_samples = stride_samples_f.round() as usize;
    if segment_samples == 0 {
        return Err(SparrowEngineError::InvalidManifest(
            "segment duration results in 0 samples".to_string(),
        ));
    }
    if stride_samples == 0 {
        return Err(SparrowEngineError::InvalidManifest(
            "segment stride results in 0 samples".to_string(),
        ));
    }
    if segment_samples < n_fft as usize {
        return Err(SparrowEngineError::InvalidManifest(format!(
            "segment_samples ({segment_samples}) must be >= n_fft ({n_fft}) — adjust segment_duration_s or n_fft in the audio manifest"
        )));
    }
    Ok((segment_samples, stride_samples))
}

/// Enumerate sliding-window start offsets over `total_samples`.
///
/// Mirrors the inclusive-tail termination contract used by the audio detect
/// paths: every offset `< total_samples` is emitted, and iteration stops
/// after the first offset whose remaining `total_samples - offset` is
/// `<= segment_samples` (i.e. the last window may be tail-padded by the
/// caller). Empty input (`total_samples == 0`) returns an empty `Vec`.
///
/// Used by `sparrow-engine-cpu::detect_audio` (mel + raw paths) and
/// `sparrow-engine-gpu::models::audio` (whole-clip + per-batch strategies).
/// Centralizing here keeps the windowing termination invariant in a single
/// place — previously each call site carried a hand-rolled copy.
pub fn compute_segment_offsets(
    total_samples: usize,
    segment_samples: usize,
    stride_samples: usize,
) -> Vec<usize> {
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
    offsets
}

/// Compute the `(start_s, end_s)` time range for a sliding-window segment.
///
/// `end_s` clamps the segment end to `total_samples` so a tail-padded window
/// reports its real (unpadded) end on the time axis — matching the contract
/// shared by the CPU mel/raw audio detect paths and the GPU `collect_segments`.
pub fn segment_time_range(
    seg_offset: usize,
    segment_samples: usize,
    total_samples: usize,
    sample_rate: u32,
) -> (f32, f32) {
    let start_s = seg_offset as f32 / sample_rate as f32;
    let actual_end = (seg_offset + segment_samples).min(total_samples);
    let end_s = actual_end as f32 / sample_rate as f32;
    (start_s, end_s)
}


/// Non-zero band `[start, start + weights.len())` of one triangular mel filter.
/// Mel filters are triangular, so each row of the dense filterbank is non-zero
/// only over a small contiguous freq span. Storing just that span lets the
/// projection skip the ~95% zero bins.
struct MelBand {
    start: usize,
    weights: Vec<f32>,
}

/// Pre-computed mel filterbank matrix for reuse across segments.
///
/// The filterbank depends only on static config parameters (n_mels, n_fft,
/// sample_rate, fmin, fmax) and is identical for every segment. Computing it
/// once avoids ~700 MB of redundant allocation + computation for long audio
/// (e.g., 480s at 0.3s stride = 1600 segments).
pub struct MelFilterbank {
    /// Flat row-major [n_mels, n_freqs] filterbank weights.
    pub data: Vec<f32>,
    pub n_mels: usize,
    pub n_freqs: usize,
    /// Banded form of `data`, pre-computed once: the non-zero span of each mel
    /// filter. The projection multiplies only the band — bit-identical to the
    /// dense matmul (adding 0.0 is exact in IEEE-754) but ~20-50× fewer FLOPs,
    /// which is the bulk of the per-window mel cost on ARM edge devices.
    bands: Vec<MelBand>,
}

impl MelFilterbank {
    /// Build mel filterbank from audio config.
    pub fn new(config: &AudioPreprocessConfig) -> Result<Self> {
        config.validate()?;
        let n_mels = config.n_mels as usize;
        let n_fft = config.n_fft as usize;
        let n_freqs = n_fft / 2 + 1;
        let data = mel_filterbank(n_mels, n_fft, config.sample_rate, config.fmin, config.fmax);
        let bands = build_mel_bands(&data, n_mels, n_freqs);
        Ok(Self {
            data,
            n_mels,
            n_freqs,
            bands,
        })
    }
}

/// Extract each mel filter's non-zero band from the dense filterbank rows.
fn build_mel_bands(data: &[f32], n_mels: usize, n_freqs: usize) -> Vec<MelBand> {
    let mut bands = Vec::with_capacity(n_mels);
    for m in 0..n_mels {
        let row = &data[m * n_freqs..(m + 1) * n_freqs];
        let start = row.iter().position(|&w| w != 0.0).unwrap_or(0);
        let end = row
            .iter()
            .rposition(|&w| w != 0.0)
            .map(|e| e + 1)
            .unwrap_or(start);
        bands.push(MelBand {
            start,
            weights: row[start..end].to_vec(),
        });
    }
    bands
}

/// Compute mel spectrogram for one segment of audio.
///
/// Accepts a pre-computed [`MelFilterbank`] to avoid recomputing it per segment.
/// Returns tensor `[1, 1, n_mels, time_steps]` (NCHW, single-channel).
/// For 48000 samples with n_fft=2048, hop=512: time_steps=90.
///
/// The `orig_sample_rate` argument is the **input file's** native sample rate
/// (before any engine-side resampling to `config.sample_rate`). When
/// `config.fill_highfreq == true` AND `orig_sample_rate < config.sample_rate`,
/// the engine applies the upstream PytorchWildlife "fill_highfreq" treatment
/// after power-to-dB: mel bins whose center frequency exceeds
/// `orig_sample_rate / 2 - 2500 Hz` are replaced with the 10th-percentile dB
/// value of the valid (below-boundary) bins, then the whole spectrogram is
/// clamped to `[-top_db, +20.0]`. This matches
/// `bioacoustics_spectrograms.compute_mel_spectrograms_gpu(fill_highfreq=True,
/// fill_mean_below_sr=False)` exactly (RP-27, 2026-06-01). When the flag is
/// off, or when no resample happened, the fill step is a no-op and behavior
/// matches the pre-RP-27 implementation.
///
/// Emits tracing events for `audio.preprocess.mel_gemm` and
/// `audio.preprocess.power_to_db`. The internal STFT call emits
/// `audio.preprocess.window_frame` and `audio.preprocess.fft` from inside
/// [`stft`].
pub fn mel_spectrogram(
    samples: &[f32],
    orig_sample_rate: u32,
    config: &AudioPreprocessConfig,
    filterbank: &MelFilterbank,
) -> Result<Array4<f32>> {
    config.validate()?;
    if filterbank.n_mels == 0
        || filterbank.n_freqs == 0
        || filterbank.data.len() != filterbank.n_mels * filterbank.n_freqs
    {
        return Err(SparrowEngineError::AudioPreprocess(format!(
            "invalid mel filterbank dimensions: n_mels={} n_freqs={} data_len={}",
            filterbank.n_mels,
            filterbank.n_freqs,
            filterbank.data.len()
        )));
    }
    let n_fft = config.n_fft as usize;
    let expected_n_freqs = n_fft / 2 + 1;
    if filterbank.n_mels != config.n_mels as usize || filterbank.n_freqs != expected_n_freqs {
        return Err(SparrowEngineError::AudioPreprocess(format!(
            "mel filterbank dimensions do not match config: filterbank n_mels={} n_freqs={}, expected n_mels={} n_freqs={}",
            filterbank.n_mels,
            filterbank.n_freqs,
            config.n_mels,
            expected_n_freqs
        )));
    }
    let hop = config.hop_length as usize;
    let n_mels = filterbank.n_mels;
    let n_freqs = filterbank.n_freqs;

    // Step 1: STFT → power spectrum per frame
    let power = stft(samples, n_fft, hop)?;
    if power.is_empty() {
        return Err(SparrowEngineError::AudioPreprocess(
            "Audio segment too short for STFT".into(),
        ));
    }
    let n_frames = power.len();

    // Step 2: Apply pre-computed filterbank → mel spectrogram [n_mels, n_frames].
    // Use the banded form (skips the ~95% zero freq bins per triangular filter);
    // fall back to the dense matmul if bands are unavailable (e.g. a filterbank
    // built by struct literal rather than `MelFilterbank::new`). Both paths are
    // bit-identical — the band carries exactly the non-zero weights.
    let t_gemm = Instant::now();
    let mut mel = vec![0.0f32; n_mels * n_frames];
    if filterbank.bands.len() == n_mels {
        for (t, frame) in power.iter().enumerate() {
            for (m, band) in filterbank.bands.iter().enumerate() {
                let hi = (band.start + band.weights.len()).min(frame.len());
                let lo = band.start.min(hi);
                let sum: f32 = band.weights[..hi - lo]
                    .iter()
                    .zip(&frame[lo..hi])
                    .map(|(w, p)| w * p)
                    .sum();
                mel[m * n_frames + t] = sum;
            }
        }
    } else {
        for (t, frame) in power.iter().enumerate() {
            for m in 0..n_mels {
                let filter_row = &filterbank.data[m * n_freqs..(m + 1) * n_freqs];
                let sum: f32 = filter_row
                    .iter()
                    .zip(frame.iter())
                    .map(|(f, p)| f * p)
                    .sum();
                mel[m * n_frames + t] = sum;
            }
        }
    }
    tracing::info!(
        stage = "audio.preprocess.mel_gemm",
        duration_ns = t_gemm.elapsed().as_nanos() as u64,
        n_mels = n_mels,
        n_frames = n_frames,
    );

    // Step 3: Power-to-dB (absolute reference, ref=1.0) with top_db clamping
    let t_db = Instant::now();
    power_to_db(&mut mel, config.top_db);
    tracing::info!(
        stage = "audio.preprocess.power_to_db",
        duration_ns = t_db.elapsed().as_nanos() as u64,
        n_values = mel.len(),
    );

    // Step 3b (RP-27): optional fill_highfreq for upsampled inputs.
    if config.fill_highfreq && orig_sample_rate < config.sample_rate {
        let t_fill = Instant::now();
        apply_fill_highfreq(&mut mel, n_mels, n_frames, orig_sample_rate, config);
        tracing::info!(
            stage = "audio.preprocess.fill_highfreq",
            duration_ns = t_fill.elapsed().as_nanos() as u64,
            orig_sr = orig_sample_rate,
            target_sr = config.sample_rate,
        );
    }

    // Step 4: Tensor [1, 1, n_mels, n_frames]
    let tensor = Array4::from_shape_vec([1, 1, n_mels, n_frames], mel)
        .map_err(|e| SparrowEngineError::AudioPreprocess(e.to_string()))?;

    Ok(tensor)
}

/// Apply the PytorchWildlife `fill_highfreq` treatment to a dB-scale mel
/// spectrogram in-place (RP-27, 2026-06-01).
///
/// For inputs whose native sample rate is below `config.sample_rate`, mel
/// bins above `orig_sample_rate/2 - 2500 Hz` carry no useful signal — at
/// training time these bins were replaced with a noise-floor estimate (the
/// 10th-percentile dB value over all valid bins) so the model never learned
/// to depend on them. At inference time, leaving them at the power-to-dB
/// clamp floor (`max − top_db`) produces a different distribution and biases
/// the model. This routine reproduces the training-time fill exactly.
///
/// `mel` is laid out as `[n_mels, n_frames]` (row-major). Caller guarantees
/// `orig_sample_rate < config.sample_rate` and `mel.len() == n_mels * n_frames`.
fn apply_fill_highfreq(
    mel: &mut [f32],
    n_mels: usize,
    n_frames: usize,
    orig_sample_rate: u32,
    config: &AudioPreprocessConfig,
) {
    debug_assert!(orig_sample_rate < config.sample_rate);
    debug_assert_eq!(mel.len(), n_mels * n_frames);

    // Mel bin center frequencies, matching `librosa.mel_frequencies(n_mels,
    // fmin=config.fmin, fmax=config.fmax)`. librosa uses endpoint-inclusive
    // linspace over n_mels positions, NOT the n_mels+2 triangular-filter
    // anchors used by [`mel_filterbank`]. This distinction is load-bearing
    // for `fill_highfreq`: torchaudio's MelSpectrogram (used for the filterbank
    // matmul) and librosa's mel_frequencies (used by PW Bioacoustics
    // `fill_highfreq` to decide which bins are "noise") have different
    // center-frequency conventions; we must match the latter exactly here.
    let mel_min = slaney_hz_to_mel(config.fmin);
    let mel_max = slaney_hz_to_mel(config.fmax);
    let mel_centers_hz: Vec<f32> = (0..n_mels)
        .map(|i| {
            let mel = mel_min + (mel_max - mel_min) * i as f32 / (n_mels - 1).max(1) as f32;
            slaney_mel_to_hz(mel)
        })
        .collect();

    let nyq_orig = (orig_sample_rate as f32 / 2.0) - 2500.0;
    let noise_mask: Vec<bool> = mel_centers_hz.iter().map(|&hz| hz > nyq_orig).collect();
    let n_noise: usize = noise_mask.iter().filter(|&&b| b).count();
    if n_noise == 0 {
        return; // no bins above boundary — nothing to fill, no clamp.
    }

    // 10th-percentile dB of valid (below-boundary) bins.
    // librosa uses k = ceil(0.10 * len(valid_vals)); torch.kthvalue is
    // 1-indexed and returns the value at position k of the sorted ascending
    // sequence. Mirror that semantics exactly.
    let n_valid = n_mels - n_noise;
    debug_assert!(n_valid > 0); // when n_noise = n_mels we'd've returned above
    let mut valid_vals: Vec<f32> = Vec::with_capacity(n_valid * n_frames);
    for (m, &is_noise) in noise_mask.iter().enumerate() {
        if !is_noise {
            valid_vals.extend_from_slice(&mel[m * n_frames..(m + 1) * n_frames]);
        }
    }
    let k = (0.10_f32 * valid_vals.len() as f32).ceil() as usize;
    let k = k.max(1).min(valid_vals.len());
    // Partial sort: nth_element semantics. select_nth_unstable is O(n) and
    // gives us the kth-smallest element in valid_vals[k-1] after the call.
    valid_vals.select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap());
    let mu = valid_vals[k - 1];

    // Replace noise bins with mu.
    for (m, &is_noise) in noise_mask.iter().enumerate() {
        if is_noise {
            for v in &mut mel[m * n_frames..(m + 1) * n_frames] {
                *v = mu;
            }
        }
    }

    // Final clamp to [-top_db, +20.0]. Matches PW upstream: after the fill,
    // the spectrogram is clamped to the broader [-top_db, +20] range
    // (regardless of the per-segment amax used in step 3's top_db clamp).
    let lo = -config.top_db;
    let hi = 20.0_f32;
    for v in mel.iter_mut() {
        if *v < lo {
            *v = lo;
        } else if *v > hi {
            *v = hi;
        }
    }
}

// ---------------------------------------------------------------------------
// WAV decoding
// ---------------------------------------------------------------------------

/// Decode a WAV file from path. Returns mono f32 samples in [-1, 1] + sample rate.
fn decode_wav(path: &Path) -> Result<(Vec<f32>, u32)> {
    let reader =
        hound::WavReader::open(path).map_err(|e| SparrowEngineError::AudioDecode(e.to_string()))?;
    decode_wav_reader(reader)
}

/// Shared WAV decoding: read samples, convert to mono f32 in [-1, 1].
///
/// Phase 3.8 Step 2 perf-fix B (post Wave-4 triage,
/// `docs/research/phase3.8/step2/perf_triage_report.md`): the
/// `audio.decode` stage was 32.1 % of total wall-clock on the 60 s
/// DUNAS clip (`hound`-iterator-based). The 16-bit PCM fast path here
/// bulk-reads the data chunk and converts via a tight indexed loop
/// (`i16::from_le_bytes` + `* (1.0 / 32768.0)`) which auto-vectorizes
/// to SIMD on x86_64 (SSE2/AVX2) under release `-Copt-level=3`. The
/// formula is bit-identical to the slow path's `(s as i32 as f32) /
/// 32768.0` — both are a power-of-2 divide which f32 represents
/// exactly, so the multiplication and division produce identical
/// results.
///
/// Other formats (24-bit int, 32-bit int, 32-bit float) fall back to
/// the original `hound::into_samples` iterator path. Field recordings
/// for the sparrow-engine audio model (DUNAS + synthetic bench fixtures) are
/// all 16-bit PCM, so the fast path covers the production workload
/// without changing the slow path's semantics.
fn decode_wav_reader<R: std::io::Read>(reader: hound::WavReader<R>) -> Result<(Vec<f32>, u32)> {
    let spec = reader.spec();
    let sr = spec.sample_rate;
    let channels = spec.channels as usize;

    // Fast path: 16-bit PCM — most common WAV format for field
    // recordings + the entire sparrow-engine audio bench corpus.
    if spec.sample_format == hound::SampleFormat::Int && spec.bits_per_sample == 16 {
        return decode_wav_int16_fast(reader, sr, channels);
    }

    let samples_f32: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            let max_val = (1i64 << (bits - 1)) as f32;
            reader
                .into_samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max_val))
                .collect::<std::result::Result<Vec<f32>, _>>()
                .map_err(|e| SparrowEngineError::AudioDecode(e.to_string()))?
        }
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<f32>, _>>()
            .map_err(|e| SparrowEngineError::AudioDecode(e.to_string()))?,
    };

    // Downmix to mono by averaging channels
    let mono = if channels == 1 {
        samples_f32
    } else {
        samples_f32
            .chunks_exact(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok((mono, sr))
}

/// 16-bit PCM fast path — bulk-read data chunk + auto-vectorizable
/// i16→f32 conversion. Output is bit-identical to the slow-path's
/// `(s as i32 as f32) / 32768.0` because `1.0 / 32768.0 = 2^-15` is
/// exactly representable in f32 and multiplication by a power-of-2
/// f32 is bit-exact.
///
/// Reads `reader.len() * 2` raw bytes from the data chunk via
/// `reader.into_inner()` (hound positions the reader at the start of
/// the data chunk after parsing the RIFF + fmt headers).
fn decode_wav_int16_fast<R: std::io::Read>(
    reader: hound::WavReader<R>,
    sr: u32,
    channels: usize,
) -> Result<(Vec<f32>, u32)> {
    // `len()` returns total i16 samples (frames * channels).
    let n_samples = reader.len() as usize;
    let n_bytes = n_samples
        .checked_mul(2)
        .ok_or_else(|| SparrowEngineError::AudioDecode("WAV size overflow".to_string()))?;

    let mut inner = reader.into_inner();
    let mut raw = vec![0u8; n_bytes];
    std::io::Read::read_exact(&mut inner, &mut raw)
        .map_err(|e| SparrowEngineError::AudioDecode(format!("WAV data read: {e}")))?;

    // 1.0 / 32768.0 = 2^-15 — exactly representable in f32, so
    // `i16 as f32 * SCALE` is bit-identical to `i16 as f32 / 32768.0`
    // (both reduce to a mantissa-exponent shift on f32).
    const SCALE: f32 = 1.0 / 32768.0;
    let mut samples_f32: Vec<f32> = vec![0.0; n_samples];
    // Tight indexed loop over (raw, samples_f32) pairs; the bounds
    // check on `samples_f32[i]` lifts out under -Copt-level=3 because
    // `samples_f32.len() == n_samples` is provable from the
    // construction above. SSE2 / AVX2 auto-vectorize this on x86_64.
    let raw_chunks = raw.chunks_exact(2);
    for (i, chunk) in raw_chunks.enumerate() {
        let v = i16::from_le_bytes([chunk[0], chunk[1]]);
        samples_f32[i] = v as f32 * SCALE;
    }

    // Downmix to mono by averaging channels (same formula as the
    // slow path: `sum(channels) / n_channels`).
    let mono = if channels == 1 {
        samples_f32
    } else {
        samples_f32
            .chunks_exact(channels)
            .map(|ch| ch.iter().sum::<f32>() / channels as f32)
            .collect()
    };

    Ok((mono, sr))
}

// ---------------------------------------------------------------------------
// Resampling
// ---------------------------------------------------------------------------

/// Resample audio to target sample rate.
///
/// Fast path: integer-ratio decimation (96/192kHz→48kHz) or interpolation (16kHz→48kHz).
/// Slow path: rubato FFT resampler for non-integer ratios (44.1kHz, 22.05kHz→48kHz).
fn resample(samples: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    if from_sr == to_sr {
        return Ok(samples.to_vec());
    }

    // Integer decimation fast path (96kHz→48kHz, 192kHz→48kHz)
    if from_sr > to_sr && from_sr.is_multiple_of(to_sr) {
        let factor = (from_sr / to_sr) as usize;
        return Ok(decimate(samples, factor));
    }

    // Integer interpolation fast path (16kHz→48kHz)
    if to_sr > from_sr && to_sr.is_multiple_of(from_sr) {
        let factor = (to_sr / from_sr) as usize;
        return Ok(interpolate(samples, factor));
    }

    // Non-integer ratio: rubato (44.1kHz→48kHz, 22.05kHz→48kHz)
    resample_rubato(samples, from_sr, to_sr)
}

/// Integer decimation: average every N samples (simple anti-alias + downsample).
fn decimate(samples: &[f32], factor: usize) -> Vec<f32> {
    samples
        .chunks(factor)
        .map(|chunk| chunk.iter().sum::<f32>() / chunk.len() as f32)
        .collect()
}

/// Integer interpolation via linear blend.
///
/// The last `factor` output samples are copies of the last input sample (clamped,
/// not extrapolated). This produces a flat tail rather than linear extrapolation.
/// For audio resampling (e.g. 16kHz -> 48kHz) the effect is 3 identical samples
/// at the end of a 48000+ sample buffer — negligible impact on spectrogram
/// computation since the last STFT frame typically covers hundreds of samples.
fn interpolate(samples: &[f32], factor: usize) -> Vec<f32> {
    if samples.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(samples.len() * factor);
    for i in 0..samples.len() - 1 {
        for j in 0..factor {
            let t = j as f32 / factor as f32;
            out.push(samples[i] * (1.0 - t) + samples[i + 1] * t);
        }
    }
    // Last sample: clamp (repeat) rather than extrapolate. See doc comment above.
    for _ in 0..factor {
        out.push(samples[samples.len() - 1]);
    }
    out
}

/// Resample via rubato FFT resampler for non-integer ratios.
fn resample_rubato(samples: &[f32], from_sr: u32, to_sr: u32) -> Result<Vec<f32>> {
    let chunk_size = 1024;
    let mut resampler = FftFixedInOut::<f32>::new(
        from_sr as usize,
        to_sr as usize,
        chunk_size,
        1, // mono
    )
    .map_err(|e| SparrowEngineError::Resample(e.to_string()))?;

    let input_frames = resampler.input_frames_next();
    let mut output = Vec::new();

    // Process complete chunks
    let mut offset = 0;
    while offset + input_frames <= samples.len() {
        let chunk = &samples[offset..offset + input_frames];
        let result = resampler
            .process(&[chunk], None)
            .map_err(|e| SparrowEngineError::Resample(e.to_string()))?;
        output.extend_from_slice(&result[0]);
        offset += input_frames;
    }

    // Handle remaining samples: zero-pad to fill last chunk
    if offset < samples.len() {
        let mut padded = samples[offset..].to_vec();
        padded.resize(input_frames, 0.0);
        let result = resampler
            .process(&[&padded], None)
            .map_err(|e| SparrowEngineError::Resample(e.to_string()))?;
        // Only keep output proportional to real input samples
        let real_ratio = (samples.len() - offset) as f64 / input_frames as f64;
        let real_out = (result[0].len() as f64 * real_ratio).round() as usize;
        output.extend_from_slice(&result[0][..real_out]);
    }

    Ok(output)
}

// ---------------------------------------------------------------------------
// STFT
// ---------------------------------------------------------------------------

/// Short-Time Fourier Transform → power spectrum per frame.
///
/// Each frame: apply symmetric Hann window, real FFT, compute |X[k]|^2.
/// Returns Vec of frames, each with n_fft/2+1 power bins.
///
/// Emits tracing events for `audio.preprocess.window_frame` (per-frame copy +
/// Hann window multiply, accumulated) and `audio.preprocess.fft` (per-frame
/// FFT plan exec + power computation, accumulated).
fn stft(samples: &[f32], n_fft: usize, hop_length: usize) -> Result<Vec<Vec<f32>>> {
    let n_freqs = n_fft / 2 + 1;
    let window = hann_window(n_fft);

    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut fft_input = fft.make_input_vec();
    let mut fft_output = fft.make_output_vec();

    let mut frames = Vec::new();
    let mut start = 0;
    let mut window_frame_ns: u64 = 0;
    let mut fft_ns: u64 = 0;

    while start + n_fft <= samples.len() {
        // Apply Hann window
        let t_wf = Instant::now();
        for (i, val) in fft_input.iter_mut().enumerate() {
            *val = samples[start + i] * window[i];
        }
        window_frame_ns += t_wf.elapsed().as_nanos() as u64;

        // Forward real FFT + power spectrum: re^2 + im^2
        let t_fft = Instant::now();
        fft.process(&mut fft_input, &mut fft_output)
            .map_err(|e| SparrowEngineError::AudioPreprocess(format!("FFT error: {e}")))?;
        let mut power = Vec::with_capacity(n_freqs);
        for c in &fft_output {
            power.push(c.re * c.re + c.im * c.im);
        }
        fft_ns += t_fft.elapsed().as_nanos() as u64;
        frames.push(power);

        start += hop_length;
    }

    let n_frames = frames.len() as u64;
    tracing::info!(
        stage = "audio.preprocess.window_frame",
        duration_ns = window_frame_ns,
        n_frames = n_frames,
    );
    tracing::info!(
        stage = "audio.preprocess.fft",
        duration_ns = fft_ns,
        n_frames = n_frames,
    );

    Ok(frames)
}

/// Symmetric Hann window: `w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))`.
fn hann_window(n: usize) -> Vec<f32> {
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / denom).cos()))
        .collect()
}

// ---------------------------------------------------------------------------
// Mel filterbank
// ---------------------------------------------------------------------------

/// Build mel filterbank matrix, stored flat as [n_mels * n_freqs] (row-major).
///
/// **Slaney mel scale** (matches `torchaudio.transforms.MelSpectrogram(mel_scale="slaney")`
/// and `librosa.filters.mel(htk=False)`): linear from 0 to 1000 Hz with slope
/// `200/3` mels per kHz; logarithmic above 1000 Hz with `mel = 15 + log(hz/1000) / log(6.4) * 27`.
/// **Slaney filter normalization** (`torchaudio` `norm="slaney"`,
/// `librosa.filters.mel(norm="slaney")` eq. (6)): each triangle is divided by
/// `2.0 / (mel_hz_centers[i+2] - mel_hz_centers[i])`, giving each filter equal
/// energy weighting per Hz of bandwidth.
///
/// Phase 3.8 Step 2 Wave 0a (F0.8 corrective fix, 2026-05-04): switched from
/// HTK + area normalization to Slaney + Slaney normalization to match
/// `MD_AudioBirds_V1` training (PW Bioacoustics
/// `mel_scale="slaney", norm="slaney"`). Pre-fix sparrow-engine used HTK mel scale
/// (`mel = 2595 * log10(1 + hz/700)`) and area normalization (`sum(filter * df) = 1`).
fn mel_filterbank(n_mels: usize, n_fft: usize, sample_rate: u32, fmin: f32, fmax: f32) -> Vec<f32> {
    let n_freqs = n_fft / 2 + 1;
    let sr = sample_rate as f32;
    let df = sr / n_fft as f32;

    let hz_to_mel = slaney_hz_to_mel;
    let mel_to_hz = slaney_mel_to_hz;

    let mel_min = hz_to_mel(fmin);
    let mel_max = hz_to_mel(fmax);

    // n_mels + 2 equally spaced points in mel space
    let n_points = n_mels + 2;
    let mel_points: Vec<f32> = (0..n_points)
        .map(|i| mel_min + (mel_max - mel_min) * i as f32 / (n_points - 1) as f32)
        .collect();
    let hz_points: Vec<f32> = mel_points.iter().map(|&m| mel_to_hz(m)).collect();

    // Hz → FFT bin index (float for interpolation)
    let bin_points: Vec<f32> = hz_points.iter().map(|&hz| hz / df).collect();

    // Build triangular filters
    let mut bank = vec![0.0f32; n_mels * n_freqs];

    for m in 0..n_mels {
        let left = bin_points[m];
        let center = bin_points[m + 1];
        let right = bin_points[m + 2];

        for k in 0..n_freqs {
            let kf = k as f32;
            let weight = if kf >= left && kf < center && center > left {
                (kf - left) / (center - left)
            } else if kf >= center && kf <= right && right > center {
                (right - kf) / (right - center)
            } else {
                0.0
            };
            bank[m * n_freqs + k] = weight;
        }

        // Slaney normalization: each filter divided by 2.0 / (Hz bandwidth) so
        // equal-loudness filters get equivalent energy weighting (librosa
        // eq. 6). The bandwidth is `hz_points[m+2] - hz_points[m]`, the Hz
        // distance between the filter's left and right edges.
        let enorm = 2.0 / (hz_points[m + 2] - hz_points[m]);
        for x in bank[m * n_freqs..(m + 1) * n_freqs].iter_mut() {
            *x *= enorm;
        }
    }

    bank
}

/// Slaney mel scale: `hz → mel` (matches `torchaudio._hz_to_mel("slaney")` and
/// `librosa.filters.mel(htk=False)`).
///
/// Linear below 1000 Hz with slope `200/3` mels per kHz (i.e. `mel = hz / (1000/15)`),
/// logarithmic above with `mel = 15 + log(hz/1000) / log(6.4) * 27`. The pivot
/// at 1000 Hz keeps the curve `C^0`-continuous (linear approximation tangent
/// to the log scale).
fn slaney_hz_to_mel(hz: f32) -> f32 {
    let f_min: f32 = 0.0;
    let f_sp: f32 = 200.0 / 3.0; // mels per Hz in the linear region
    let min_log_hz: f32 = 1000.0;
    let min_log_mel: f32 = (min_log_hz - f_min) / f_sp; // = 15.0
    let logstep: f32 = (6.4_f32).ln() / 27.0;
    if hz < min_log_hz {
        (hz - f_min) / f_sp
    } else {
        min_log_mel + ((hz / min_log_hz).ln() / logstep)
    }
}

/// Inverse of [`slaney_hz_to_mel`]: `mel → hz`.
fn slaney_mel_to_hz(mel: f32) -> f32 {
    let f_min: f32 = 0.0;
    let f_sp: f32 = 200.0 / 3.0;
    let min_log_hz: f32 = 1000.0;
    let min_log_mel: f32 = (min_log_hz - f_min) / f_sp; // = 15.0
    let logstep: f32 = (6.4_f32).ln() / 27.0;
    if mel < min_log_mel {
        f_min + f_sp * mel
    } else {
        min_log_hz * ((mel - min_log_mel) * logstep).exp()
    }
}

// ---------------------------------------------------------------------------
// Power-to-dB
// ---------------------------------------------------------------------------

/// Convert power values to dB with absolute reference (ref=1.0).
///
/// `dB = 10 * log10(max(power, 1e-10))`, then clamp: `max(dB, max_dB - top_db)`.
/// Applied per-segment (each segment's spectrogram clamped to its own max).
fn power_to_db(values: &mut [f32], top_db: f32) {
    let epsilon: f32 = 1e-10;

    for x in values.iter_mut() {
        *x = 10.0 * (*x).max(epsilon).log10();
    }

    let max_db = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let floor = max_db - top_db;
    for x in values.iter_mut() {
        *x = (*x).max(floor);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        // Symmetric Hann for N=4: w[n] = 0.5*(1 - cos(2*pi*n/3))
        // w = [0.0, 0.75, 0.75, 0.0]
        let w = hann_window(4);
        assert_eq!(w.len(), 4);
        assert!(w[0].abs() < 1e-6, "w[0] should be 0, got {}", w[0]);
        assert!(
            (w[1] - 0.75).abs() < 1e-6,
            "w[1] should be 0.75, got {}",
            w[1]
        );
        assert!(
            (w[2] - 0.75).abs() < 1e-6,
            "w[2] should be 0.75, got {}",
            w[2]
        );
        assert!(w[3].abs() < 1e-6, "w[3] should be 0, got {}", w[3]);

        // Endpoints are zero, midpoint is 1.0 for odd N
        let w5 = hann_window(5);
        assert!(w5[0].abs() < 1e-6);
        assert!((w5[2] - 1.0).abs() < 1e-6);
        assert!(w5[4].abs() < 1e-6);
    }

    #[test]
    fn test_mel_filterbank_shape() {
        let n_mels = 224;
        let n_fft = 1024;
        let n_freqs = n_fft / 2 + 1; // 513
        let bank = mel_filterbank(n_mels, n_fft, 48000, 0.0, 24000.0);
        assert_eq!(bank.len(), n_mels * n_freqs);
    }

    #[test]
    fn test_mel_filterbank_slaney_norm() {
        // Slaney normalization (post Phase 3.8 Step 2 Wave 0a): each triangular
        // filter is divided by 2/(hz_centers[i+2] - hz_centers[i]). Verify
        // that all weights are non-negative + finite, that at least one
        // non-degenerate filter has a small (sub-unity) Slaney-normalized peak
        // weight, and that the filterbank is deterministic. With high
        // resolution (n_mels=224 over 0..24000 Hz at n_fft=1024, df ≈ 46.9 Hz)
        // the lowest filters can be entirely below the first FFT bin and
        // legitimately have zero weights — same as the pre-fix `area_norm`
        // path which had `if area > 0.0` to skip those.
        let n_mels = 224;
        let n_fft = 1024;
        let n_freqs = n_fft / 2 + 1;
        let bank = mel_filterbank(n_mels, n_fft, 48000, 0.0, 24000.0);
        for &w in &bank {
            assert!(
                w >= 0.0 && w.is_finite(),
                "Slaney filterbank weight must be non-negative + finite, got {w}"
            );
        }
        // Find the first non-degenerate filter and verify its peak is in
        // (0, 1). Slaney normalization scales by 2/Hz_bandwidth, which is
        // small (order 1e-4 for wide upper bands, 1e-3 for narrow lower
        // bands), so the peak weight is well below 1.0.
        let mut saw_peak = false;
        for m in 0..n_mels {
            let row = &bank[m * n_freqs..(m + 1) * n_freqs];
            let peak = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            if peak > 0.0 {
                assert!(
                    peak < 1.0,
                    "Slaney filter {m} peak {peak} unexpectedly >= 1.0; \
                     normalization should produce sub-unity peaks at this resolution"
                );
                saw_peak = true;
                break;
            }
        }
        assert!(
            saw_peak,
            "filterbank had zero non-degenerate filters — bandwidth or layout bug"
        );
    }

    #[test]
    fn test_stft_known_signal() {
        // 1024-sample sine wave at bin 10 frequency (10 * sr / n_fft)
        let n_fft = 1024;
        let sr = 48000.0;
        let freq = 10.0 * sr / n_fft as f32; // bin 10
        let samples: Vec<f32> = (0..n_fft)
            .map(|i| (2.0 * std::f32::consts::PI * freq * i as f32 / sr).sin())
            .collect();

        let frames = stft(&samples, n_fft, n_fft).unwrap();
        assert_eq!(frames.len(), 1);

        // Peak should be at or near bin 10
        let frame = &frames[0];
        let peak_bin = frame
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert!(
            (peak_bin as i32 - 10).unsigned_abs() <= 1,
            "Peak should be at bin ~10, got {peak_bin}"
        );
    }

    #[test]
    fn test_resample_integer_ratio() {
        // 192kHz → 48kHz = factor 4, output length should be ceil(input/4)
        let input: Vec<f32> = (0..19200).map(|i| (i as f32 * 0.001).sin()).collect();
        let output = resample(&input, 192000, 48000).unwrap();
        assert_eq!(output.len(), 19200 / 4);
    }

    #[test]
    fn test_resample_passthrough() {
        let input = vec![1.0, 2.0, 3.0];
        let output = resample(&input, 48000, 48000).unwrap();
        assert_eq!(output, input);
    }

    #[test]
    fn load_audio_rejects_non_finite_raw_samples() {
        let cfg = AudioPreprocessConfig::default();
        for sample in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let err = match load_audio(
                &AudioInput::Samples {
                    data: vec![0.0, sample, 0.0],
                    sample_rate: cfg.sample_rate,
                },
                &cfg,
            ) {
                Ok(_) => panic!("non-finite raw audio sample must fail"),
                Err(err) => err,
            };
            assert!(
                err.to_string().contains("finite"),
                "error should mention finite samples, got {err}"
            );
        }
    }

    #[test]
    fn load_audio_rejects_zero_input_sample_rate() {
        let cfg = AudioPreprocessConfig::default();
        let err = match load_audio(
            &AudioInput::Samples {
                data: vec![0.0; cfg.sample_rate as usize],
                sample_rate: 0,
            },
            &cfg,
        ) {
            Ok(_) => panic!("zero sample_rate must fail"),
            Err(err) => err,
        };
        assert!(
            err.to_string().contains("sample_rate"),
            "error should mention sample_rate, got {err}"
        );
    }

    #[test]
    fn validate_audio_window_params_rejects_invalid_runtime_overrides() {
        assert!(validate_audio_window_params(f32::NAN, 1.0, 0.5, 48_000, 1024).is_err());
        assert!(validate_audio_window_params(1.0, 0.0, 0.5, 48_000, 1024).is_err());
        assert!(validate_audio_window_params(1.0, 1.0, 1.1, 48_000, 1024).is_err());
        assert!(validate_audio_window_params(0.001, 0.001, 0.5, 48_000, 1024).is_err());
        assert!(validate_audio_window_params(1.0, 1.0, 0.5, 0, 1024).is_err());
        assert!(validate_audio_window_params(1.0, 1.0, 0.5, 48_000, 0).is_err());
        assert!(validate_audio_window_params(1.0, 1.0, 0.5, 48_000, 1).is_err());
    }

    #[test]
    fn preprocess_segment_rejects_filterbank_config_mismatch() {
        let config = AudioPreprocessConfig {
            sample_rate: 16_000,
            n_fft: 64,
            hop_length: 32,
            n_mels: 2,
            fmin: 0.0,
            fmax: 8_000.0,
            top_db: 80.0,
            fill_highfreq: false,
        };
        let samples = vec![0.0f32; 128];
        let wrong_mels = MelFilterbank {
            data: vec![0.0; 3 * 33],
            n_mels: 3,
            n_freqs: 33,
            bands: Vec::new(),
        };
        assert!(mel_spectrogram(&samples, config.sample_rate, &config, &wrong_mels).is_err());

        let wrong_freqs = MelFilterbank {
            data: vec![0.0; 2 * 32],
            n_mels: 2,
            n_freqs: 32,
            bands: Vec::new(),
        };
        assert!(mel_spectrogram(&samples, config.sample_rate, &config, &wrong_freqs).is_err());
    }

    #[test]
    fn test_power_to_db_clamp() {
        // Values: 1.0, 0.01, 1e-12
        // dB:     0.0, -20.0, -100.0 (clamped by epsilon to -100)
        // With top_db=80: floor = 0.0 - 80.0 = -80.0
        // After clamp: [0.0, -20.0, -80.0]
        let mut values = vec![1.0, 0.01, 1e-12];
        power_to_db(&mut values, 80.0);

        assert!(
            (values[0] - 0.0).abs() < 0.01,
            "Expected ~0dB, got {}",
            values[0]
        );
        assert!(
            (values[1] - (-20.0)).abs() < 0.01,
            "Expected ~-20dB, got {}",
            values[1]
        );
        assert!(
            (values[2] - (-80.0)).abs() < 0.01,
            "Expected -80dB (clamped), got {}",
            values[2]
        );
    }

    #[test]
    fn test_mel_spectrogram_output_shape() {
        // 48000 samples (1 second at 48kHz), default config (n_fft=2048).
        // Frames: (48000 - 2048) / 512 + 1 = 90
        let config = AudioPreprocessConfig::default();
        let fb = MelFilterbank::new(&config).expect("MelFilterbank::new");
        let samples = vec![0.0f32; 48000];
        let tensor = mel_spectrogram(&samples, config.sample_rate, &config, &fb).unwrap();
        assert_eq!(tensor.shape(), &[1, 1, 224, 90]);
    }

    #[test]
    fn test_mel_spectrogram_short_segment_error() {
        let config = AudioPreprocessConfig::default();
        let fb = MelFilterbank::new(&config).expect("MelFilterbank::new");
        let samples = vec![0.0f32; 1024]; // Too short for n_fft=2048
        let result = mel_spectrogram(&samples, config.sample_rate, &config, &fb);
        assert!(result.is_err());
    }

    #[test]
    fn test_decimate_basic() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let output = decimate(&input, 4);
        // Average of [1,2,3,4]=2.5, [5,6,7,8]=6.5
        assert_eq!(output.len(), 2);
        assert!((output[0] - 2.5).abs() < 1e-6);
        assert!((output[1] - 6.5).abs() < 1e-6);
    }

    /// Phase 3.8 Step 2 perf-fix B (post Wave-4 triage): the i16 fast
    /// path in `decode_wav_int16_fast` MUST be bit-identical to the
    /// hound-iterator slow path. Both use `1.0 / 32768.0 = 2^-15`,
    /// which is exactly representable in f32, so multiplication by
    /// the inverse and division by 32768.0 produce identical results.
    /// This test writes a 16-bit PCM mono WAV to a tempfile, decodes
    /// it through `decode_wav_reader` (fast path) and a clone through
    /// the equivalent slow-path formula, and asserts byte-for-byte
    /// equality across all samples.
    #[test]
    fn test_decode_wav_int16_fast_path_bit_exact_vs_slow() {
        use std::io::Cursor;
        // Synthesize a non-trivial 16-bit PCM signal — full range
        // including extremes (-32768, 32767, 0, ±100, ±10000) so the
        // sign-extension + scale path is exercised.
        let pcm: Vec<i16> = (0..2048i16)
            .flat_map(|i| [i16::MIN, i16::MAX, 0, -100, 100, -10_000, 10_000, i])
            .collect();

        // Write WAV header + samples to an in-memory buffer.
        let mut buf = Cursor::new(Vec::<u8>::new());
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).expect("WavWriter::new");
            for &s in &pcm {
                writer.write_sample(s).expect("write_sample");
            }
            writer.finalize().expect("WavWriter::finalize");
        }
        let wav_bytes = buf.into_inner();

        // Fast path: route through decode_wav_reader (which dispatches
        // to decode_wav_int16_fast for 16-bit PCM).
        let reader_fast =
            hound::WavReader::new(Cursor::new(wav_bytes.clone())).expect("WavReader::new (fast)");
        let (samples_fast, sr_fast) = decode_wav_reader(reader_fast).expect("decode fast");

        // Slow-path equivalent: hound iterator + (i32 as f32 / 32768).
        let reader_slow =
            hound::WavReader::new(Cursor::new(wav_bytes)).expect("WavReader::new (slow)");
        let max_val_slow = (1i64 << 15) as f32; // = 32768
        let samples_slow: Vec<f32> = reader_slow
            .into_samples::<i32>()
            .map(|s| (s.expect("sample") as f32) / max_val_slow)
            .collect();

        assert_eq!(sr_fast, 48_000);
        assert_eq!(samples_fast.len(), pcm.len());
        assert_eq!(samples_slow.len(), pcm.len());
        // Byte-for-byte equality (the i16 fast-path divisor 1/32768 is
        // exactly representable in f32 → no rounding delta vs the slow
        // path's division by 32768.0).
        for (i, (&a, &b)) in samples_fast.iter().zip(&samples_slow).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "sample {i}: fast {a} (bits {:#x}) ≠ slow {b} (bits {:#x}) — fast path \
                 must be bit-identical to slow path",
                a.to_bits(),
                b.to_bits()
            );
        }
    }

    /// Stereo 16-bit PCM should downmix to mono identically through
    /// both paths. Locks the channel-averaging step in the fast path.
    #[test]
    fn test_decode_wav_int16_fast_path_stereo_downmix() {
        use std::io::Cursor;
        // Interleaved stereo: L = +1000, R = -1000 → mono = 0.
        // L = +20000, R = -19000 → mono = 500.
        let pcm_lr: Vec<i16> = vec![1000, -1000, 20_000, -19_000, -32_768, 32_767];
        // Slow-path expected mono (channel-averaged).
        let max_val = 32_768.0_f32;
        let expected: Vec<f32> = pcm_lr
            .chunks_exact(2)
            .map(|ch| ((ch[0] as f32) / max_val + (ch[1] as f32) / max_val) / 2.0)
            .collect();

        let mut buf = Cursor::new(Vec::<u8>::new());
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        {
            let mut writer = hound::WavWriter::new(&mut buf, spec).expect("WavWriter::new");
            for &s in &pcm_lr {
                writer.write_sample(s).expect("write_sample");
            }
            writer.finalize().expect("WavWriter::finalize");
        }
        let wav_bytes = buf.into_inner();
        let reader = hound::WavReader::new(Cursor::new(wav_bytes)).expect("WavReader::new");
        let (mono, sr) = decode_wav_reader(reader).expect("decode stereo");
        assert_eq!(sr, 48_000);
        assert_eq!(mono.len(), expected.len());
        // Allow 1 ulp because the downmix path does (a + b) / 2 in
        // either order; for symmetrical pairs the result is exact.
        for (i, (a, b)) in mono.iter().zip(&expected).enumerate() {
            assert!(
                (a - b).abs() < 1e-7,
                "stereo mono sample {i}: fast {a} vs expected {b}"
            );
        }
    }

    #[test]
    fn test_interpolate_basic() {
        let input = vec![0.0, 1.0, 2.0];
        let output = interpolate(&input, 2);
        // [0, 0.5, 1, 1.5, 2, 2]
        assert_eq!(output.len(), 6);
        assert!((output[0] - 0.0).abs() < 1e-6);
        assert!((output[1] - 0.5).abs() < 1e-6);
        assert!((output[2] - 1.0).abs() < 1e-6);
        assert!((output[3] - 1.5).abs() < 1e-6);
        assert!((output[4] - 2.0).abs() < 1e-6);
        assert!((output[5] - 2.0).abs() < 1e-6);
    }
}

#[cfg(test)]
mod phase_a_r1_preprocess_audio {
    use super::*;
    use sparrow_engine_types::AudioInput;

    /// Tiny mel filterbank (n_mels=2, n_fft=64, sr=16000). Exercises the
    /// triangular-filter construction at a size where every step is
    /// hand-verifiable. Locks: shape == n_mels * (n_fft/2+1), all weights are
    /// non-negative + finite, and the result is deterministic across two
    /// construction calls.
    ///
    /// Phase 3.8 Step 2 Wave 0a (2026-05-04): updated for Slaney normalization
    /// (was area normalization). Slaney divides each filter by `2 / (Hz bandwidth)`
    /// so equal-loudness filters get equivalent energy weighting; the area-sum
    /// invariant `sum(filter * df) ≈ 1.0` no longer holds.
    #[test]
    fn mel_filterbank_tiny_config_deterministic_and_nonneg() {
        let cfg = AudioPreprocessConfig {
            sample_rate: 16000,
            n_fft: 64,
            hop_length: 16,
            n_mels: 2,
            fmin: 0.0,
            fmax: 8000.0,
            top_db: 80.0,
            fill_highfreq: false,
        };
        let fb1 = MelFilterbank::new(&cfg).expect("MelFilterbank::new");
        let fb2 = MelFilterbank::new(&cfg).expect("MelFilterbank::new");
        assert_eq!(fb1.n_mels, 2);
        assert_eq!(fb1.n_freqs, 64 / 2 + 1);
        assert_eq!(fb1.data.len(), fb1.n_mels * fb1.n_freqs);
        assert_eq!(
            fb1.data, fb2.data,
            "mel filterbank must be deterministic across calls with identical config"
        );
        for &w in &fb1.data {
            assert!(
                w >= 0.0 && w.is_finite(),
                "all filterbank weights must be ≥ 0 and finite, got {w}"
            );
        }
    }

    /// Resample 16k → 16k is identity (passthrough early-return at line 247).
    /// Existing `test_resample_passthrough` covers a 3-element vec; we widen
    /// to a non-trivial 16384-sample buffer with non-monotone content to lock
    /// "exact byte-for-byte clone, not a reconstruction".
    #[test]
    fn resample_identity_returns_clone() {
        let cfg = AudioPreprocessConfig {
            sample_rate: 16000,
            n_fft: 1024,
            hop_length: 256,
            n_mels: 64,
            fmin: 0.0,
            fmax: 8000.0,
            top_db: 80.0,
            fill_highfreq: false,
        };
        let samples: Vec<f32> = (0..16384).map(|i| (i as f32 / 32.0).sin() * 0.5).collect();
        let loaded = load_audio(
            &AudioInput::Samples {
                data: samples.clone(),
                sample_rate: 16000,
            },
            &cfg,
        )
        .unwrap();
        assert_eq!(
            loaded.data, samples,
            "16k → 16k resample must be byte-identical (no DSP run)"
        );
        assert_eq!(loaded.sample_rate, 16000);
    }

    /// Mel spectrogram on all-zero input must produce all-zero (or near-zero)
    /// output before the dB clamp. After dB-conversion + top_db clamp, the
    /// values are *clamped to floor*, not zero. So we assert: the entire
    /// tensor is finite, all values equal a single floor (within 1e-3), and
    /// the floor is `0.0 - top_db` (since max log10(epsilon)*10 ≈ -100 → max
    /// dB is -100 + 0 = -100, and floor = max - top_db = -180; but per-segment
    /// max of all-epsilon is what `power_to_db` uses).
    #[test]
    fn mel_spectrogram_zero_input_yields_constant_floor() {
        let cfg = AudioPreprocessConfig::default();
        let fb = MelFilterbank::new(&cfg).expect("MelFilterbank::new");
        let samples = vec![0.0f32; 48000];
        let tensor = mel_spectrogram(&samples, cfg.sample_rate, &cfg, &fb).unwrap();
        let slice = tensor.as_slice().unwrap();
        // All entries must be finite (no NaN from log10 thanks to .max(epsilon)).
        for v in slice {
            assert!(v.is_finite(), "mel of zero input must be finite, got {v}");
        }
        // All entries should be identical (all hit the floor).
        let first = slice[0];
        for v in slice {
            assert!(
                (v - first).abs() < 1e-3,
                "all-zero input must produce a flat mel (got values spanning {first} and {v})"
            );
        }
    }

    /// `top_db` saturation: a large impulse plus mostly-quiet samples must
    /// have its quiet floor clamped to `max - top_db`. Drives the second loop
    /// in `power_to_db` (line 477) — the floor clamp. Existing
    /// `test_power_to_db_clamp` does this in isolation; we drive it through
    /// the public `mel_spectrogram` pipeline so a refactor that bypasses the
    /// clamp would surface here.
    #[test]
    fn mel_spectrogram_top_db_saturation_caps_floor() {
        // tighter floor than the default 80.0 dB
        let cfg = AudioPreprocessConfig {
            top_db: 40.0,
            ..Default::default()
        };
        let fb = MelFilterbank::new(&cfg).expect("MelFilterbank::new");
        // 48000 samples: one burst at 0.5 amplitude in the middle, silence
        // elsewhere. The burst dominates the per-segment max; everything quiet
        // is clamped to (max - 40).
        let mut samples = vec![0.0f32; 48000];
        for sample in samples.iter_mut().skip(24000).take(100) {
            *sample = 0.5;
        }
        let tensor = mel_spectrogram(&samples, cfg.sample_rate, &cfg, &fb).unwrap();
        let slice = tensor.as_slice().unwrap();
        let max = slice.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let min = slice.iter().cloned().fold(f32::INFINITY, f32::min);
        assert!(max.is_finite());
        assert!(min.is_finite());
        // top_db is the maximum allowed range from peak to floor.
        assert!(
            (max - min) <= cfg.top_db + 1e-3,
            "max-min ({}) must not exceed top_db ({})",
            max - min,
            cfg.top_db
        );
    }
}

// =============================================================================
// (Optional) Integration test idea — not strictly needed, but adds a per-crate
// integration scenario.
// === SAVE AS sparrow-engine/sparrow-engine-core/tests/integration_phase_a_r1.rs ===
// (Sketch only — not pasted in this draft. The existing audio_heatmap_e2e.rs +
// integration_viz_dispatch.rs already cover the cross-module integration
// surface; the in-module phase_a_r1_* blocks above are sufficient for Round 01.)
// =============================================================================
