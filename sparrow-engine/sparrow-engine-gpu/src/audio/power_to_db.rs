//! W1.4 — `power_to_db` kernel: log10 + per-segment max + clamp. **Tier B.**
//!
//! Wave 0 measured the CPU `power_to_db` cost at **23.50 ms / 60 s clip**
//! (median, 5 fresh-process runs). The custom CUDA kernel fuses log10
//! conversion + per-segment max reduction + floor clamp into a single
//! block-per-segment launch.
//!
//! # Algorithm
//!
//! Per `sparrow_engine_core::preprocess_audio::power_to_db`, the CPU sequence is:
//!
//! ```text
//! for x in mel: x = 10 * log10(max(x, 1e-10))
//! max_db = max(x for x in mel)
//! floor = max_db - top_db
//! for x in mel: x = max(x, floor)
//! ```
//!
//! This is applied **per segment** (each segment's spectrogram clamped
//! against its own max). `sparrow-engine-cpu`'s caller invokes
//! `sparrow_engine_core::preprocess_audio::power_to_db` once per `[1, 1, n_mels,
//! n_frames]` mel block (`sparrow-engine-core/src/preprocess_audio.rs:218-225` calls
//! `power_to_db(&mut mel, config.top_db)`).
//!
//! On GPU we run with grid `total_segments × 1`, `block_dim = 256`, one
//! block per segment. The block does:
//!
//! 1. Block-stride pass over the segment's `n_mels * frames_per_segment`
//!    elements: in-place log10 conversion + accumulate per-thread local max.
//! 2. Block-level max reduction in shared memory.
//! 3. `floor = seg_max - top_db`.
//! 4. Block-stride second pass: clamp.
//!
//! # Parity gate
//!
//! `tests/power_to_db_parity.rs` asserts max-abs Δ ≤ 5e-3 dB on
//! post-mel-GEMM data vs CPU `power_to_db` (`docs/design/phase3.8/step2/
//! round_02/arch-perf_proposal_r2.md §R2.1` G0c).
//!
//! # NVRTC preprocessor limitation (Wave 1 finding 2026-05-05)
//!
//! `power_to_db.cu` cannot use the C99 `INFINITY` macro: NVRTC (used here
//! for runtime PTX compilation via `cudarc::nvrtc::compile_ptx`) does NOT
//! preprocess `<math.h>` by default — the `INFINITY` macro is undefined
//! in the NVRTC translation unit. We use `-1.0e30f` as the per-thread
//! `local_max` sentinel instead. The sentinel is below any post-log10
//! mel-dB value the kernel ever produces (real-audio max +60 dB to floor
//! -100 dB are both within ±1e30); a sub-INFINITY sentinel is safe by
//! construction. Initial development hit this bug as
//! `NVRTC_ERROR_COMPILATION: identifier "INFINITY" is undefined`; the
//! same constraint applies to any future `power_to_db.cu`-style kernel
//! in this crate.

use std::sync::Arc;

use sparrow_engine_types::error::{SparrowEngineError, Result};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use crate::audio::compile_audio_kernel;

const KERNEL_SRC: &str = include_str!("power_to_db.cu");
const KERNEL_NAME: &str = "power_to_db_kernel";
const BLOCK: u32 = 256;

/// Compiled `power_to_db` kernel.
#[derive(Clone)]
pub struct PowerToDbKernel {
    func: CudaFunction,
}

impl PowerToDbKernel {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let func = compile_audio_kernel(ctx, KERNEL_SRC, KERNEL_NAME, "power_to_db")?;
        Ok(Self { func })
    }
}

/// Apply log10 + per-segment max-relative floor in place.
///
/// - `mel`: `CudaSlice<f32>` of length `total_segments * n_mels *
///   frames_per_segment`. The per-segment slab is `n_mels *
///   frames_per_segment` contiguous elements. The kernel writes back into
///   this same buffer.
/// - `total_segments`: number of segments.
/// - `n_mels`, `frames_per_segment`: per-segment dimensions.
/// - `top_db`: clamp range below the per-segment max (manifest default 80.0).
///
/// Concrete `CudaSlice<f32>` (not generic `DevicePtrMut`) because
/// `LaunchArgs::arg` only implements `PushKernelArg<&mut CudaSlice<f32>>`
/// directly (sealed trait — see `power_kernel::power_gpu` for the same
/// rationale).
pub fn power_to_db_gpu(
    stream: &Arc<CudaStream>,
    kernel: &PowerToDbKernel,
    mel: &mut CudaSlice<f32>,
    total_segments: usize,
    n_mels: usize,
    frames_per_segment: usize,
    top_db: f32,
) -> Result<()> {
    if total_segments == 0 {
        return Ok(());
    }
    let n_mels_i: i32 = i32::try_from(n_mels)
        .map_err(|e| SparrowEngineError::Ort(format!("power_to_db: n_mels {n_mels} > i32::MAX: {e}")))?;
    let frames_i: i32 = i32::try_from(frames_per_segment).map_err(|e| {
        SparrowEngineError::Ort(format!(
            "power_to_db: frames_per_segment {frames_per_segment} > i32::MAX: {e}"
        ))
    })?;
    let segments_u: u32 = u32::try_from(total_segments)
        .map_err(|e| SparrowEngineError::Ort(format!("power_to_db: total_segments > u32: {e}")))?;

    let cfg = LaunchConfig {
        grid_dim: (segments_u, 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0, // kernel uses a fixed-size __shared__ s_max[BLOCK]
    };

    let mut launch = stream.launch_builder(&kernel.func);
    launch
        .arg(mel)
        .arg(&n_mels_i)
        .arg(&frames_i)
        .arg(&top_db);
    // SAFETY: `mel` is sized for `total_segments * n_mels * frames_per_segment`
    // f32s by caller contract. Block-stride loops are bounds-checked
    // against `seg_size = n_mels * frames_per_segment`.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch power_to_db_kernel: {e}")))?;
    Ok(())
}

/// CPU reference: in-place `power_to_db` matching
/// `sparrow_engine_core::preprocess_audio::power_to_db`. Used by the parity test.
pub fn cpu_power_to_db(values: &mut [f32], top_db: f32) {
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
