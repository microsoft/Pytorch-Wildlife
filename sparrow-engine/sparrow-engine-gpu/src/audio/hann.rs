//! W1.6 — Hann window + mel filterbank constants on GPU (bit-exact upload).
//!
//! The CPU pipeline computes `hann_window(n_fft)` and `mel_filterbank(...)`
//! once per engine init and reuses across segments. On GPU we replicate the
//! exact same `Vec<f32>` from `sparrow-engine-core::preprocess_audio` and `clone_htod`
//! it. No GPU re-derivation — that would risk introducing FP drift and
//! invalidate the W1.6 G0d "bit-exact = 0.0" gate.
//!
//! # Layouts
//!
//! - `hann_d`: `[n_fft]` row-major `f32`. n_fft = 2048 from the manifest.
//! - `mel_filterbank_d`: `[n_mels * n_freqs]` row-major `f32` with
//!   `n_mels = 224`, `n_freqs = 1025`. Same row-major layout as
//!   `sparrow_engine_core::preprocess_audio::MelFilterbank::data` — row `m` covers
//!   `data[m * n_freqs .. (m + 1) * n_freqs]`.
//!
//! Both buffers are immutable for the engine's lifetime; treat as
//! device-resident constants.
//!
//! # Bit-exact correctness gate
//!
//! `tests::hann_filterbank_bit_exact` (and the companion mel_gemm parity
//! tests) DtoH the GPU buffer and assert byte-equal against the CPU output
//! from `sparrow_engine_core::preprocess_audio::MelFilterbank::new` (Slaney mel scale
//! + Slaney norm post-Wave 0a F0.8 fix). Per `arch-perf_proposal_r2.md
//! §R2.1` G0d, max-abs Δ MUST be `0.0`.

use std::sync::Arc;

use sparrow_engine_core::preprocess_audio::{AudioPreprocessConfig, MelFilterbank};
use sparrow_engine_types::error::{SparrowEngineError, Result};
use cudarc::driver::{CudaSlice, CudaStream};

// ---------------------------------------------------------------------------
// Hann window — symmetric, matches `sparrow_engine_core::preprocess_audio::hann_window`.
// ---------------------------------------------------------------------------

/// Symmetric Hann window: `w[n] = 0.5 * (1 - cos(2*pi*n / (N-1)))`.
///
/// Mirror of the private `hann_window` in `sparrow_engine_core::preprocess_audio`
/// (verified at `sparrow-engine-core/src/preprocess_audio.rs:447-453` post-Wave-0a;
/// computed on host and uploaded). Exposed here so the parity test can
/// re-compute the reference without depending on sparrow-engine-core's private
/// helper.
pub fn hann_window_cpu(n: usize) -> Vec<f32> {
    if n < 2 {
        return Vec::new();
    }
    let denom = (n - 1) as f32;
    (0..n)
        .map(|i| 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / denom).cos()))
        .collect()
}

/// Upload the Hann window to GPU. Returns `[n_fft]` row-major `CudaSlice<f32>`.
///
/// Bit-exact: the GPU buffer holds the literal output of [`hann_window_cpu`],
/// no re-derivation.
pub fn upload_hann_window(stream: &Arc<CudaStream>, n_fft: usize) -> Result<CudaSlice<f32>> {
    if n_fft < 2 {
        return Err(SparrowEngineError::AudioPreprocess(format!(
            "audio n_fft must be at least 2, got {n_fft}"
        )));
    }
    let host = hann_window_cpu(n_fft);
    stream
        .clone_htod(&host)
        .map_err(|e| SparrowEngineError::Ort(format!("clone_htod (hann_window): {e}")))
}

// ---------------------------------------------------------------------------
// Mel filterbank upload.
// ---------------------------------------------------------------------------

/// Upload the mel filterbank to GPU. Returns `[n_mels * n_freqs]` row-major
/// `CudaSlice<f32>`. Source: `sparrow_engine_core::preprocess_audio::MelFilterbank`
/// (Slaney scale + Slaney norm).
pub fn upload_mel_filterbank(
    stream: &Arc<CudaStream>,
    config: &AudioPreprocessConfig,
) -> Result<UploadedMelFilterbank> {
    let fb = MelFilterbank::new(config)?;
    let data = stream
        .clone_htod(&fb.data)
        .map_err(|e| SparrowEngineError::Ort(format!("clone_htod (mel_filterbank): {e}")))?;
    Ok(UploadedMelFilterbank {
        data,
        n_mels: fb.n_mels,
        n_freqs: fb.n_freqs,
    })
}

/// Mel filterbank residing in GPU memory.
///
/// Lifetime invariant: while this struct exists the underlying
/// `CudaSlice<f32>` is alive. Consumers (e.g. `mel_gemm::run`) borrow
/// `data` for cuBLAS GEMM input.
pub struct UploadedMelFilterbank {
    /// Row-major `[n_mels * n_freqs]` filterbank weights.
    pub data: CudaSlice<f32>,
    pub n_mels: usize,
    pub n_freqs: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_core::preprocess_audio::AudioPreprocessConfig;
    use cudarc::driver::CudaContext;

    #[test]
    fn hann_window_bit_exact_upload() {
        let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
        let stream = ctx.default_stream();
        let n_fft = 2048usize;

        let cpu = hann_window_cpu(n_fft);
        let dev = upload_hann_window(&stream, n_fft).expect("upload_hann_window");
        stream.synchronize().expect("stream.synchronize");
        let gpu_back: Vec<f32> = stream.clone_dtoh(&dev).expect("clone_dtoh hann");

        assert_eq!(cpu.len(), gpu_back.len(), "length mismatch");
        let mut max_abs_delta = 0.0f32;
        for (i, (c, g)) in cpu.iter().zip(gpu_back.iter()).enumerate() {
            let d = (c - g).abs();
            if d > max_abs_delta {
                max_abs_delta = d;
            }
            assert_eq!(
                c.to_bits(),
                g.to_bits(),
                "Hann mismatch at i={i}: cpu={c} gpu={g}"
            );
        }
        assert_eq!(
            max_abs_delta, 0.0,
            "G0d gate exceeded: max-abs Δ = {max_abs_delta}"
        );
    }

    #[test]
    fn mel_filterbank_bit_exact_upload() {
        let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
        let stream = ctx.default_stream();
        let config = AudioPreprocessConfig::default();

        let cpu = MelFilterbank::new(&config).expect("MelFilterbank::new");
        let dev = upload_mel_filterbank(&stream, &config).expect("upload_mel_filterbank");
        stream.synchronize().expect("stream.synchronize");
        let gpu_back: Vec<f32> = stream.clone_dtoh(&dev.data).expect("clone_dtoh filterbank");

        assert_eq!(cpu.data.len(), gpu_back.len(), "length mismatch");
        assert_eq!(cpu.n_mels, dev.n_mels);
        assert_eq!(cpu.n_freqs, dev.n_freqs);

        let mut max_abs_delta = 0.0f32;
        for (i, (c, g)) in cpu.data.iter().zip(gpu_back.iter()).enumerate() {
            let d = (c - g).abs();
            if d > max_abs_delta {
                max_abs_delta = d;
            }
            assert_eq!(
                c.to_bits(),
                g.to_bits(),
                "filterbank mismatch at i={i}: cpu={c} gpu={g}"
            );
        }
        assert_eq!(
            max_abs_delta, 0.0,
            "G0d gate exceeded: max-abs Δ = {max_abs_delta}"
        );
    }
}
