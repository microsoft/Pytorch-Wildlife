//! W1.2 — Fused `re² + im²` power-spectrum kernel. **Tier B.**
//!
//! Wave 0 measured the CPU power-spectrum loop (rolled into `audio.preprocess.fft`,
//! ~5 ms of the 29.29 ms FFT total at 60 s). Per `sparrow-engine-core/src/preprocess_audio.rs:422-425`
//! the CPU computes `re*re + im*im` per complex bin in a tight Rust loop.
//!
//! The custom CUDA kernel fuses this into one launch over the
//! `[total_frames, n_freqs]` complex output of `cufft_plan::BatchedR2cPlan`.
//! Compiled at runtime via `cudarc::nvrtc::compile_ptx`, mirroring the
//! Step 1 image kernels (`crate::kernels::letterbox`).
//!
//! # Parity gate
//!
//! `tests/power_parity.rs` asserts max-abs Δ ≤ 1e-5 vs CPU `re² + im²`
//! (`docs/design/phase3.8/step2/round_02/arch-perf_proposal_r2.md §R2.1`
//! G0b' — derived from the same unit FMA precision argument as G0c).
//!
//! Bit-equivalence-NOT-required because GPU FMA can fuse `re*re + im*im`
//! differently than the CPU two-step `mul; add`. Practical Δ at FP32 has
//! been < 1 ULP (~1.2e-7 absolute on inputs of magnitude ~1.0) in initial
//! sandbox runs; gate is set 100× looser to absorb rare edge cases.

use std::sync::Arc;

use sparrow_engine_types::error::{SparrowEngineError, Result};
use cudarc::cufft::sys as cufft_sys;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};

use crate::audio::compile_audio_kernel;

const KERNEL_SRC: &str = include_str!("power_kernel.cu");
const KERNEL_NAME: &str = "power_kernel";

/// Compiled power-spectrum kernel module + entry.
#[derive(Clone)]
pub struct PowerKernel {
    func: CudaFunction,
}

impl PowerKernel {
    /// Compile and load the kernel into the given CUDA context.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let func = compile_audio_kernel(ctx, KERNEL_SRC, KERNEL_NAME, "power_kernel")?;
        Ok(Self { func })
    }
}

/// Launch the power-spectrum kernel.
///
/// - `complex_in`: `CudaSlice<float2>` of length `total_frames * n_freqs`.
/// - `power_out`: `CudaSlice<f32>` of length `total_frames * n_freqs`.
///
/// Concrete types (not `DevicePtr` / `DevicePtrMut` generics) because
/// `cudarc::driver::LaunchArgs::arg` is implemented for the concrete
/// `&CudaSlice<T>` / `&mut CudaSlice<T>` types via a sealed trait
/// (`PushKernelArg`); using a generic `DevicePtr` bound at the call site
/// would require also bounding `LaunchArgs<'_>: PushKernelArg<&I>`, which
/// is not part of cudarc's public API surface as of 0.19.4.
pub fn power_gpu(
    stream: &Arc<CudaStream>,
    kernel: &PowerKernel,
    complex_in: &CudaSlice<cufft_sys::float2>,
    power_out: &mut CudaSlice<f32>,
    total_frames: usize,
    n_freqs: usize,
) -> Result<()> {
    let total = total_frames * n_freqs;
    if total == 0 {
        return Ok(());
    }
    let total_i: i32 = i32::try_from(total)
        .map_err(|e| SparrowEngineError::Ort(format!("power kernel total {total} > i32::MAX: {e}")))?;

    const TX: u32 = 256;
    let blocks = u32::try_from(total.div_ceil(TX as usize))
        .map_err(|e| SparrowEngineError::Ort(format!("power kernel grid > u32: {e}")))?;
    let cfg = LaunchConfig {
        grid_dim: (blocks, 1, 1),
        block_dim: (TX, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&kernel.func);
    launch.arg(complex_in).arg(power_out).arg(&total_i);
    // SAFETY: `complex_in` and `power_out` are sized for `total` elements
    // (caller's contract). Kernel does an explicit `idx >= total` bounds
    // check.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch power_kernel: {e}")))?;
    Ok(())
}

/// CPU reference: scalar `re² + im²` per element. Used by the parity test.
pub fn cpu_power(complex_in: &[(f32, f32)]) -> Vec<f32> {
    complex_in
        .iter()
        .map(|(re, im)| re * re + im * im)
        .collect()
}
