//! W1.3 — Mel filterbank GEMM via cuBLAS sgemm. **Tier A: highest perf win.**
//!
//! Wave 0 measured the CPU mel-filterbank multiply at **1481.8 ms / 60 s clip**
//! (median, 5 fresh-process runs) — 61.8 % of the engine total.
//! cuBLAS sgemm on the same inputs is expected to compress this to ~50 ms
//! (~30× speedup); the bench harness in `examples/bench_mel_gemm.rs` /
//! `scripts/bench_mel_gemm.py` produces the actual measurement.
//!
//! # Algorithm
//!
//! For each segment we compute:
//!
//! ```text
//! mel[m, t] = sum_k filterbank[m, k] * power[k, t]
//! ```
//!
//! where `filterbank` is `[n_mels=224, n_freqs=1025]` (row-major; uploaded
//! by `crate::audio::hann::upload_mel_filterbank`) and `power` is
//! `[n_freqs=1025, n_frames=batch * 90]` (column-major view of
//! `[batch * 90, n_freqs]` from cuFFT).
//!
//! cuBLAS expects column-major. We arrange operands so the GEMM uses
//! `transa = OP_N`, `transb = OP_N`:
//!
//! - In cuBLAS column-major view, `A` is the filterbank with shape
//!   `[n_freqs, n_mels]` (i.e., row-major `[n_mels, n_freqs]` viewed as
//!   transposed column-major). With `OP_T` on `A`, cuBLAS reads A as
//!   `[n_mels, n_freqs]`.
//! - `B` is the power spectrum with shape `[n_freqs, total_frames]`
//!   (column-major), which is what we'd produce from a cuFFT power layout
//!   `[total_frames, n_freqs]` by treating it as transposed.
//!
//! Concretely, our row-major numpy/CPU view is
//!
//! ```text
//! filterbank: [M=n_mels, K=n_freqs]      (row-major)
//! power:      [K=n_freqs, N=total_frames] (column-major in mem; equivalent
//!                                          to row-major [total_frames, n_freqs]
//!                                          which is what cuFFT writes)
//! mel:        [M=n_mels, N=total_frames] (column-major in mem; row-major
//!                                          per-frame mel[m, t] retrievable
//!                                          via gpu[m + M * t])
//! ```
//!
//! In cuBLAS column-major: `A` (filterbank, row-major `[M,K]`) has
//! column-major shape `[K,M]` with `lda = K`. We use `OP_T` to read it as
//! `[M,K]` for the multiply. `B` (power) is column-major `[K,N]` with
//! `ldb = K`, used `OP_N`. Result `C` is column-major `[M,N]` with
//! `ldc = M`. Map to GEMM: `m = M`, `n = N`, `k = K`.
//!
//! # Per-segment vs whole-batch
//!
//! Wave 0 measures the per-segment cost. The Tier-A claim is "compress
//! the whole 60 s clip's mel multiply to ~50 ms". The simplest way to
//! realise that is one cuBLAS sgemm covering all frames in the clip
//! (`total_frames = num_segments * 90` for the manifest's 1 s / 0.3 s
//! sliding window). The per-batch (16-segment) variant is also benched
//! to inform the streaming-callback sliding-window shape arch-prag's R2
//! adopts (whole-clip mel + chunk-of-16 ORT).
//!
//! # Parity gate
//!
//! `tests/mel_gemm_parity.rs` asserts `max_abs(gpu - cpu_scalar) ≤ 5e-5`
//! on a deterministic synthetic input (`docs/design/phase3.8/step2/round_02/
//! arch-perf_proposal_r2.md §R2.1` G0b).

use std::sync::Arc;

use sparrow_engine_types::error::{SparrowEngineError, Result};
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig};
use cudarc::driver::{CudaSlice, CudaStream};

/// Mel-filterbank GEMM helper.
///
/// Owns a cuBLAS handle bound to a CUDA stream. Reuse one instance per
/// engine (`mel_gemm` is called once per audio batch, ~13 times for a 60 s
/// clip when running per-batch; or once for the whole-clip variant).
pub struct MelGemm {
    blas: CudaBlas,
    /// Same Arc the cuBLAS handle is bound to. Held here as a public
    /// accessor since `CudaBlas::stream` is `pub(crate)` upstream.
    stream: Arc<CudaStream>,
    /// Cached `n_mels` (filterbank rows). Validated against caller-provided
    /// shapes on every `run` to catch shape drift.
    n_mels: usize,
    /// Cached `n_freqs` (filterbank columns). Same validation contract.
    n_freqs: usize,
}

impl MelGemm {
    /// Construct from a stream + filterbank dimensions.
    pub fn new(stream: Arc<CudaStream>, n_mels: usize, n_freqs: usize) -> Result<Self> {
        let blas = CudaBlas::new(stream.clone())
            .map_err(|e| SparrowEngineError::Ort(format!("cuBLAS handle init: {e}")))?;
        Ok(Self {
            blas,
            stream,
            n_mels,
            n_freqs,
        })
    }

    /// Compute `mel = filterbank @ power` on GPU.
    ///
    /// - `filterbank`: row-major `[n_mels, n_freqs]` `f32`. Uploaded once
    ///   by `crate::audio::hann::upload_mel_filterbank` and reused across
    ///   calls.
    /// - `power`: row-major `[total_frames, n_freqs]` `f32`. Output of the
    ///   power-spectrum kernel (W1.2). cuBLAS will read it column-major
    ///   `[n_freqs, total_frames]` via OP_N — the row-major `[total_frames,
    ///   n_freqs]` layout in memory IS column-major `[n_freqs,
    ///   total_frames]`.
    /// - `mel_out`: pre-allocated `[n_mels * total_frames]` `f32`. Result is
    ///   written column-major `[n_mels, total_frames]`. To read entry
    ///   `mel[m, t]`, index as `mel_out[m + n_mels * t]`. Callers consuming
    ///   per-frame mel output can either feed this directly to the
    ///   `power_to_db` kernel + ORT (which expects NCHW
    ///   `[batch, 1, n_mels=224, n_frames]` — see `crate::audio::power_to_db`
    ///   for the layout rotation) or DtoH if testing.
    pub fn run(
        &self,
        filterbank: &CudaSlice<f32>,
        power: &CudaSlice<f32>,
        mel_out: &mut CudaSlice<f32>,
        total_frames: usize,
    ) -> Result<()> {
        if total_frames == 0 {
            return Ok(());
        }

        // GEMM dims (column-major):
        //   m = n_mels         (filterbank rows; output rows)
        //   n = total_frames   (output columns)
        //   k = n_freqs        (inner dim)
        //
        // A (filterbank): row-major [M, K] = column-major [K, M] with lda=K.
        //                 Use OP_T so cuBLAS reads the M×K view we want.
        // B (power):      row-major [N, K] = column-major [K, N] with ldb=K.
        //                 Use OP_N. (In CPU code `power[t][k]` is at
        //                 power[t * n_freqs + k]; column-major [K, N] reads
        //                 the same byte at (k, t) → power[k + K*t]. The CPU
        //                 layout is row-major [N, K] which is byte-identical
        //                 to column-major [K, N].)
        // C (mel):        column-major [M, N] with ldc=M. mel[m, t] =
        //                 mel_out[m + M*t].
        let m: i32 = i32::try_from(self.n_mels)
            .map_err(|e| SparrowEngineError::Ort(format!("mel_gemm: n_mels {} > i32::MAX: {e}", self.n_mels)))?;
        let n: i32 = i32::try_from(total_frames).map_err(|e| {
            SparrowEngineError::Ort(format!("mel_gemm: total_frames {total_frames} > i32::MAX: {e}"))
        })?;
        let k: i32 = i32::try_from(self.n_freqs).map_err(|e| {
            SparrowEngineError::Ort(format!("mel_gemm: n_freqs {} > i32::MAX: {e}", self.n_freqs))
        })?;

        let cfg = GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_T,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m,
            n,
            k,
            alpha: 1.0_f32,
            lda: k, // A column-major [K, M], stride between cols = K
            ldb: k, // B column-major [K, N], stride between cols = K
            beta: 0.0_f32,
            ldc: m, // C column-major [M, N], stride between cols = M
        };

        // SAFETY: cuBLAS sgemm is unsafe in cudarc (improper shapes can
        // segfault). We've sized `cfg` against the actual buffer dims:
        //   - `filterbank` is allocated as `n_mels * n_freqs` f32 by
        //     `upload_mel_filterbank`. Indexed by cuBLAS as `[k, m]` with
        //     `lda = k`, which produces `(m-1) * k + (k-1) = m*k - 1`, i.e.
        //     reads exactly `n_mels * n_freqs` elements. ✓
        //   - `power` must be `n_freqs * total_frames`. Caller's contract
        //     (panics here if violated via cuBLAS-side OOB read).
        //   - `mel_out` must be `n_mels * total_frames`. Same contract.
        unsafe {
            self.blas
                .gemm(cfg, filterbank, power, mel_out)
                .map_err(|e| SparrowEngineError::Ort(format!("cuBLAS sgemm: {e}")))?;
        }
        Ok(())
    }

    /// Stream the GEMM is bound to. Use to synchronize after `run` if the
    /// caller needs the result host-side before launching the next op.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

/// Extract the cuBLAS handle's stream so external Drop / DtoH waits use the
/// same stream context.
//
// (No `Send`/`Sync` impl needed: `CudaBlas` is `Send + Sync` already; this
// struct's only field besides `blas` is two `usize`s.)
impl std::fmt::Debug for MelGemm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MelGemm")
            .field("n_mels", &self.n_mels)
            .field("n_freqs", &self.n_freqs)
            .finish_non_exhaustive()
    }
}

/// CPU reference: scalar mel filterbank multiply, identical algorithm to
/// `sparrow_engine_core::preprocess_audio::mel_spectrogram`'s inner GEMM
/// (`sparrow-engine-core/src/preprocess_audio.rs:198-217`). Used by the parity test
/// and the bench's CPU baseline.
///
/// Inputs are row-major:
/// - `filterbank: [n_mels, n_freqs]` row-major
/// - `power: [n_frames, n_freqs]` row-major
///
/// Output: `mel_row_major: [n_mels, n_frames]` row-major (i.e.,
/// `mel[m, t] = mel_row_major[m * n_frames + t]`). Note: the GPU run
/// produces COLUMN-major `[n_mels, n_frames]`. The parity test transposes
/// one of them to compare bit-similar values; this CPU helper exposes
/// the row-major layout the existing `sparrow-engine-core` test corpus uses.
pub fn cpu_mel_gemm_row_major(
    filterbank: &[f32],
    power: &[f32],
    n_mels: usize,
    n_freqs: usize,
    n_frames: usize,
) -> Vec<f32> {
    assert_eq!(filterbank.len(), n_mels * n_freqs);
    assert_eq!(power.len(), n_frames * n_freqs);
    let mut mel = vec![0.0f32; n_mels * n_frames];
    for t in 0..n_frames {
        let p = &power[t * n_freqs..(t + 1) * n_freqs];
        for m in 0..n_mels {
            let f = &filterbank[m * n_freqs..(m + 1) * n_freqs];
            let s: f32 = f.iter().zip(p.iter()).map(|(a, b)| a * b).sum();
            mel[m * n_frames + t] = s;
        }
    }
    mel
}

/// Convert the GPU column-major `[n_mels, n_frames]` mel output to row-major
/// `[n_mels, n_frames]` for comparison with the CPU reference.
pub fn col_major_to_row_major(col: &[f32], n_mels: usize, n_frames: usize) -> Vec<f32> {
    assert_eq!(col.len(), n_mels * n_frames);
    let mut row = vec![0.0f32; n_mels * n_frames];
    for t in 0..n_frames {
        for m in 0..n_mels {
            row[m * n_frames + t] = col[m + n_mels * t];
        }
    }
    row
}
