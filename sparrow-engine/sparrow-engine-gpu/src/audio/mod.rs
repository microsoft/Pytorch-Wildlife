//! GPU primitives for the MD_AudioBirds_V1 mel-spectrogram pipeline.
//!
//! Phase 3.8 Step 2 Wave 1 (2026-05-05). Wave 0 verdict (`docs/research/
//! phase3.8/step2/audio_breakdown.md` headline at 60 s clip):
//!
//! | Stage | CPU median |
//! | --- | --- |
//! | `audio.preprocess.mel_gemm`     | 1481.8 ms |
//! | `audio.ort`                     |  800.7 ms |
//! | `audio.preprocess.fft`          |   29.29 ms |
//! | `audio.preprocess.power_to_db`  |   23.50 ms |
//! | `audio.preprocess.window_frame` |    2.60 ms |
//!
//! Tier A (highest leverage): [`mel_gemm`] (cuBLAS sgemm) + [`ort_io`] (ORT
//! CUDA EP IoBinding). Tier B (keeps data GPU-resident between primitives):
//! [`cufft_plan`] + [`power_kernel`] + [`power_to_db`]. Tier C (constants):
//! [`hann`] (Hann window + filterbank upload — bit-exact from CPU).
//!
//! # Constants
//!
//! All Wave 1 primitives are sized for the production manifest
//! (`sparrow-engine/models/audiobirds.toml` §preprocessing):
//!
//! - `n_fft = 2048`
//! - `n_freqs = n_fft / 2 + 1 = 1025`
//! - `hop_length = 512`
//! - `n_mels = 224`
//! - 1 second segment @ 48 kHz with hop=512 ⇒ 90 frames per segment
//! - Default ORT batch = 16 segments per `Session::run` (matches
//!   `sparrow-engine-cpu/src/detect_audio.rs::DEFAULT_BATCH_SIZE`).
//!
//! # Parity contract
//!
//! Every primitive is paired with a `tests/<name>_parity.rs` integration
//! test that compares the GPU output against the CPU reference in
//! `sparrow-engine-core::preprocess_audio`. Gates per `docs/design/phase3.8/step2/
//! round_02/arch-perf_proposal_r2.md §R2.1`:
//!
//! - `cufft_plan` (W1.1): max-abs Δ ≤ 2e-4 in complex output magnitude
//! - `power_kernel` (W1.2): max-abs Δ ≤ 1e-5 vs CPU `re² + im²`
//! - `mel_gemm` (W1.3): max-abs Δ ≤ 5e-5 on FP32 vs scalar inner-product
//! - `power_to_db` (W1.4): max-abs Δ ≤ 5e-3 dB vs CPU
//! - `ort_io` (W1.5): max-abs Δ = 0.0 (bind-once vs bind-per-call bit-exact)
//! - `hann` + filterbank (W1.6): max-abs Δ = 0.0 (bit-exact upload from CPU)

pub mod cufft_plan;
pub mod hann;
pub mod mel_gemm;
pub mod ort_io;
pub mod power_kernel;
pub mod power_to_db;
// Phase 3.8 Step 2 Wave 2 (e2e orchestrator additions, 2026-05-05):
pub mod transpose;
pub mod window_frame;

use std::sync::Arc;

use sparrow_engine_types::error::{SparrowEngineError, Result};
use cudarc::driver::{CudaContext, CudaFunction};
use cudarc::nvrtc::compile_ptx;

/// Compile a NVRTC kernel source + load it into the given CUDA context,
/// returning the entry-point `CudaFunction` ready for `stream.launch_builder`.
///
/// S5 extract (R2 audit-fix 2026-05-05): the four `power_kernel` /
/// `power_to_db` / `transpose` / `window_frame` modules each duplicated
/// the same three-step `compile_ptx → load_module → load_function`
/// dance with three near-identical `SparrowEngineError::Ort` formatters. This
/// helper centralises the dance; the four kernel structs now construct
/// via `compile_audio_kernel(ctx, KERNEL_SRC, KERNEL_NAME, "<label>")`.
///
/// `label` is a short kernel name used to disambiguate failure messages
/// (e.g. `"power_kernel"`, `"power_to_db"`, `"transpose"`,
/// `"window_frame"`). It is reused in three error contexts (compile /
/// load_module / load_function) so a failure in any stage points at the
/// right kernel.
pub(crate) fn compile_audio_kernel(
    ctx: &Arc<CudaContext>,
    src: &str,
    kernel_name: &str,
    label: &str,
) -> Result<CudaFunction> {
    let ptx = compile_ptx(src)
        .map_err(|e| SparrowEngineError::Ort(format!("nvrtc compile {label}: {e}")))?;
    let module = ctx
        .load_module(ptx)
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_module {label}: {e}")))?;
    let func = module
        .load_function(kernel_name)
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_function {label}: {e}")))?;
    Ok(func)
}
