//! W2.1 — GPU window-frame kernel (Phase 3.8 Step 2 Wave 2).
//!
//! Replaces sparrow-engine-cpu's per-frame `samples[start + i] * hann[i]` inner
//! loop (`sparrow-engine-core/src/preprocess_audio.rs:412-416`). Wave 0 measured
//! the CPU window-frame cost at **2.60 ms / 60 s clip** (median, 5
//! fresh-process runs). The custom NVRTC-compiled CUDA kernel runs one
//! block per frame, block-stride over `n_fft` samples — for the Wave 2
//! Strategy A whole-clip path this fuses 17,820 frames × 2,048 samples
//! into one launch.
//!
//! # Algorithm
//!
//! `windowed_out[f * n_fft + i] = samples[frame_starts[f] + i] * hann[i]`
//!
//! Out-of-range source samples are zero-padded so callers can request
//! frames whose starts run past the resampled audio buffer (matches the
//! tail-segment zero-pad in `sparrow-engine-cpu/src/detect_audio.rs:243-245`).
//!
//! # Inputs
//!
//! - `samples`: `[total_samples]` f32 row-major (post-resample mono).
//! - `frame_starts`: `[total_frames]` i32 absolute sample offsets, one
//!   entry per frame. Negative values are treated as out-of-range (zero
//!   reads). Caller computes these on host based on segment offsets +
//!   hop length; hop arithmetic does not run on GPU.
//! - `hann`: `[n_fft]` f32 (uploaded once via
//!   [`crate::audio::hann::upload_hann_window`]).
//!
//! # Output
//!
//! - `windowed_out`: `[total_frames * n_fft]` f32 row-major. Layout
//!   matches what [`crate::audio::cufft_plan::BatchedR2cPlan::exec`]
//!   expects.
//!
//! # Parity
//!
//! Bit-equivalent to CPU per the W1.6 hann-upload bit-exact path: same
//! Hann coefficients (uploaded byte-for-byte from CPU) × same samples =
//! same product up to FP32 mul rounding (which IS IEEE-deterministic for
//! a single multiply). End-to-end mel parity is the load-bearing
//! verification (W2 §2.1 gates).

use std::sync::Arc;

use sparrow_engine_types::error::{SparrowEngineError, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use crate::audio::compile_audio_kernel;

const KERNEL_SRC: &str = include_str!("window_frame.cu");
const KERNEL_NAME: &str = "window_frame_kernel";
const BLOCK: u32 = 256;

/// Compiled window-frame kernel.
#[derive(Clone)]
pub struct WindowFrameKernel {
    func: CudaFunction,
}

impl WindowFrameKernel {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let func = compile_audio_kernel(ctx, KERNEL_SRC, KERNEL_NAME, "window_frame")?;
        Ok(Self { func })
    }
}

/// Launch the window-frame kernel.
///
/// - `samples`: `[total_samples]` f32 device buffer.
/// - `frame_starts`: `[total_frames]` i32 device buffer with absolute
///   sample offsets per frame.
/// - `hann`: `[n_fft]` f32 device buffer (Hann window).
/// - `windowed_out`: `[total_frames * n_fft]` f32 device buffer (caller-
///   allocated).
#[allow(clippy::too_many_arguments)]
pub fn window_frame_gpu(
    stream: &Arc<CudaStream>,
    kernel: &WindowFrameKernel,
    samples: &CudaSlice<f32>,
    frame_starts: &CudaSlice<i32>,
    hann: &CudaSlice<f32>,
    windowed_out: &mut CudaSlice<f32>,
    n_fft: usize,
    total_frames: usize,
    total_samples: usize,
) -> Result<()> {
    if total_frames == 0 {
        return Ok(());
    }
    let n_fft_i: i32 = i32::try_from(n_fft)
        .map_err(|e| SparrowEngineError::Ort(format!("window_frame: n_fft {n_fft} > i32::MAX: {e}")))?;
    let total_samples_i: i32 = i32::try_from(total_samples).map_err(|e| {
        SparrowEngineError::Ort(format!(
            "window_frame: total_samples {total_samples} > i32::MAX: {e}"
        ))
    })?;
    let blocks_u: u32 = u32::try_from(total_frames)
        .map_err(|e| SparrowEngineError::Ort(format!("window_frame: total_frames > u32: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (blocks_u, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&kernel.func);
    launch
        .arg(samples)
        .arg(frame_starts)
        .arg(hann)
        .arg(windowed_out)
        .arg(&n_fft_i)
        .arg(&total_samples_i);
    // SAFETY: `samples`/`frame_starts`/`hann`/`windowed_out` are sized per
    // caller's contract above. Kernel performs explicit bounds checks
    // against `total_samples` (zero-pads OOB reads) and uses block-stride
    // loops over `n_fft`, so per-frame OOB writes cannot occur.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch window_frame: {e}")))?;
    Ok(())
}
