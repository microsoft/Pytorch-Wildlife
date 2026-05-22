//! W1.1 — Batched cuFFT R2C plan for the audio mel pipeline. **Tier B.**
//!
//! Wave 0 measured CPU FFT cost at **29.29 ms / 60 s clip** (median, 5
//! fresh-process runs). cuFFT R2C with a batched plan is expected to
//! compress to ~3 ms (~10× speedup). Required to keep data GPU-resident
//! between the windowing kernel and `mel_gemm` — without it the cuBLAS
//! GEMM input must come from a CPU FFT, breaking the GPU pipeline.
//!
//! # Algorithm
//!
//! For `total_frames = batch_segments * 90` time frames, each `n_fft = 2048`
//! samples wide, we run a batched 1-D R2C transform.
//!
//! Input layout: `[total_frames, n_fft]` row-major `f32` in device memory
//! (one window per frame, post Hann-multiply by the windowing kernel —
//! Wave 1 W1.6 / future W1.2-window-frame variant).
//!
//! Output layout: `[total_frames, n_freqs]` row-major complex `f32 + f32`
//! (`cudarc::cufft::sys::float2`) in device memory, where
//! `n_freqs = n_fft / 2 + 1 = 1025` (R2C compact representation).
//!
//! # Plan reuse
//!
//! `CudaFft` plans can be reused across multiple `exec_r2c` calls as long
//! as the input shape doesn't change (verified: `vendor/cudarc/src/cufft/
//! safe.rs:185-199`). Phase 3.8 Step 2 keeps the plan pinned per
//! `(n_fft, total_frames)` combination. Wave 1's `BatchedR2cPlan` caches
//! one plan per construction; multi-batch-size cache lives in Wave 2's
//! orchestrator (post-streaming-mode adoption).
//!
//! # Parity gate
//!
//! `tests/cufft_parity.rs` asserts max-abs Δ ≤ 2e-4 in complex output
//! magnitude vs the realfft-backed CPU reference on a 1 kHz tone with
//! `n_fft = 2048` (`docs/design/phase3.8/step2/round_02/
//! arch-perf_proposal_r2.md §R2.1` G0a; ULP derivation in arch-par §4).

use std::sync::Arc;

use sparrow_engine_types::error::{SparrowEngineError, Result};
use cudarc::cufft::{CudaFft, sys as cufft_sys};
use cudarc::driver::{CudaSlice, CudaStream};

/// Owns one cuFFT R2C plan sized for `(n_fft, total_frames)` batch.
///
/// Plan creation is one-shot at engine init; `exec` is called per inference.
pub struct BatchedR2cPlan {
    plan: CudaFft,
    n_fft: usize,
    total_frames: usize,
}

impl BatchedR2cPlan {
    /// Construct a batched 1-D R2C plan.
    ///
    /// - `n_fft`: FFT size (2048 for the production manifest).
    /// - `total_frames`: number of time frames in the batch (e.g.
    ///   `batch_segments * 90` for the manifest's 1 s segments).
    pub fn new(stream: Arc<CudaStream>, n_fft: usize, total_frames: usize) -> Result<Self> {
        if total_frames == 0 {
            return Err(SparrowEngineError::Ort(
                "BatchedR2cPlan::new: total_frames must be > 0".into(),
            ));
        }
        let nx = i32::try_from(n_fft)
            .map_err(|e| SparrowEngineError::Ort(format!("cuFFT n_fft {n_fft} > i32::MAX: {e}")))?;
        let batch = i32::try_from(total_frames)
            .map_err(|e| SparrowEngineError::Ort(format!("cuFFT total_frames {total_frames} > i32::MAX: {e}")))?;
        let plan = CudaFft::plan_1d(nx, cufft_sys::cufftType::CUFFT_R2C, batch, stream)
            .map_err(|e| SparrowEngineError::Ort(format!("cuFFT plan_1d R2C: {e}")))?;
        Ok(Self {
            plan,
            n_fft,
            total_frames,
        })
    }

    /// Execute the batched R2C transform.
    ///
    /// - `input`: `[total_frames * n_fft]` row-major `f32`.
    /// - `output`: `[total_frames * n_freqs]` row-major `cudarc::cufft::sys::float2`,
    ///   where `n_freqs = n_fft / 2 + 1`.
    ///
    /// The `n_fft` and `total_frames` are pinned to the construction values;
    /// caller must allocate the output buffer with the matching size.
    pub fn exec(
        &self,
        input: &CudaSlice<f32>,
        output: &mut CudaSlice<cufft_sys::float2>,
    ) -> Result<()> {
        self.plan
            .exec_r2c(input, output)
            .map_err(|e| SparrowEngineError::Ort(format!("cuFFT exec_r2c: {e}")))?;
        Ok(())
    }

    /// Number of output complex bins per frame (= `n_fft / 2 + 1`).
    pub fn n_freqs(&self) -> usize {
        self.n_fft / 2 + 1
    }

    pub fn n_fft(&self) -> usize {
        self.n_fft
    }

    pub fn total_frames(&self) -> usize {
        self.total_frames
    }
}

impl std::fmt::Debug for BatchedR2cPlan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchedR2cPlan")
            .field("n_fft", &self.n_fft)
            .field("total_frames", &self.total_frames)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Helpers exposed for the bench harness + parity test (non-public).
// ---------------------------------------------------------------------------

/// Build the batched-FFT *input* tensor on host: stack `total_frames`
/// windowed frames of length `n_fft` row-major. This mirrors what the
/// future GPU window-frame kernel will produce; for Wave 1 it lets the
/// parity test feed deterministic frames into cuFFT without having to
/// also write the windowing kernel.
///
/// Each input window is `samples[start..start+n_fft] * hann(i)`. The
/// caller chooses `starts`. Returns `[total_frames * n_fft]` row-major.
///
/// **NOT a production primitive.** This function is a parity-test
/// helper consumed only by the sparrow-engine-gpu integration tests
/// (`tests/{cufft,power}_parity.rs`) and benchmark examples
/// (`examples/bench_{cufft,power}.rs`). Production code uses the
/// `window_frame_kernel` GPU implementation in
/// `sparrow-engine-gpu/src/audio/window_frame.rs`.
///
/// `#[doc(hidden)]` removes this from the public rustdoc surface so
/// downstream consumers don't depend on it. (Strict `pub(crate)` would
/// cleanly signal "test-only" but breaks the integration tests +
/// examples that import it as `sparrow_engine_gpu::audio::cufft_plan::frames_with_hann_cpu`,
/// which compile against `sparrow-engine-gpu`'s public API only.) Phase 3.8 Step 2
/// audit-fix R2 / R1-F7 (2026-05-05).
#[doc(hidden)]
pub fn frames_with_hann_cpu(
    samples: &[f32],
    starts: &[usize],
    n_fft: usize,
    hann: &[f32],
) -> Vec<f32> {
    assert_eq!(hann.len(), n_fft);
    let mut out = vec![0.0f32; starts.len() * n_fft];
    for (frame_idx, &start) in starts.iter().enumerate() {
        let dst = &mut out[frame_idx * n_fft..(frame_idx + 1) * n_fft];
        for (i, sample) in dst.iter_mut().enumerate() {
            *sample = samples[start + i] * hann[i];
        }
    }
    out
}

/// Allocate `total_frames * n_freqs` complex `f32` zeros on GPU. Helper for
/// callers + tests.
pub fn alloc_complex_output(
    stream: &Arc<CudaStream>,
    total_frames: usize,
    n_freqs: usize,
) -> Result<CudaSlice<cufft_sys::float2>> {
    stream
        .alloc_zeros::<cufft_sys::float2>(total_frames * n_freqs)
        .map_err(|e| SparrowEngineError::Ort(format!("alloc_zeros (cuFFT output): {e}")))
}
