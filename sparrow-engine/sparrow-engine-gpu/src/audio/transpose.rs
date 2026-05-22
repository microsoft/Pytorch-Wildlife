//! W2.2 — Per-segment column-major → row-major mel transpose.
//!
//! cuBLAS sgemm in [`crate::audio::mel_gemm`] writes mel as column-major
//! `[n_mels, total_frames]`. ORT consumes NCHW row-major
//! `[batch, 1, n_mels, time_steps]`, so each per-segment slab needs a
//! col→row transpose before binding into the ORT IoBinding input.
//!
//! Out-of-place transpose (separate input + output buffer) so we don't
//! incur the in-place transpose cycle-detection cost on a non-square
//! `[224 × 90]` tile.

use std::sync::Arc;

use sparrow_engine_types::error::{SparrowEngineError, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use crate::audio::compile_audio_kernel;

const KERNEL_SRC: &str = include_str!("transpose.cu");
const KERNEL_NAME: &str = "transpose_per_segment_kernel";
const BLOCK: u32 = 256;

/// Compiled per-segment transpose kernel.
#[derive(Clone)]
pub struct TransposeKernel {
    func: CudaFunction,
}

impl TransposeKernel {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let func = compile_audio_kernel(ctx, KERNEL_SRC, KERNEL_NAME, "transpose")?;
        Ok(Self { func })
    }
}

/// Launch the per-segment col→row transpose.
///
/// Reads `mel_col[seg * n_mels * frames_per_seg + m + n_mels * t]` and
/// writes `mel_row[seg * n_mels * frames_per_seg + m * frames_per_seg + t]`
/// for every `(seg, m, t)` in
/// `[0..n_segments) × [0..n_mels) × [0..frames_per_seg)`.
pub fn transpose_per_segment_gpu(
    stream: &Arc<CudaStream>,
    kernel: &TransposeKernel,
    mel_col: &CudaSlice<f32>,
    mel_row_out: &mut CudaSlice<f32>,
    n_segments: usize,
    n_mels: usize,
    frames_per_seg: usize,
) -> Result<()> {
    if n_segments == 0 {
        return Ok(());
    }
    let n_mels_i: i32 = i32::try_from(n_mels)
        .map_err(|e| SparrowEngineError::Ort(format!("transpose: n_mels {n_mels} > i32::MAX: {e}")))?;
    let frames_i: i32 = i32::try_from(frames_per_seg).map_err(|e| {
        SparrowEngineError::Ort(format!(
            "transpose: frames_per_seg {frames_per_seg} > i32::MAX: {e}"
        ))
    })?;
    let segments_u: u32 = u32::try_from(n_segments)
        .map_err(|e| SparrowEngineError::Ort(format!("transpose: n_segments > u32: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (segments_u, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&kernel.func);
    launch
        .arg(mel_col)
        .arg(mel_row_out)
        .arg(&n_mels_i)
        .arg(&frames_i);
    // SAFETY: input + output are sized for `n_segments * n_mels *
    // frames_per_seg` f32s by caller contract; kernel uses block-stride
    // loops over each per-segment slab and never indexes outside it.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch transpose: {e}")))?;
    Ok(())
}
