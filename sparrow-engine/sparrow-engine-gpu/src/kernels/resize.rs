//! Rust wrapper around `resize.cu`. Plain bilinear resize, then per-channel
//! `(px/255 - mean) / std` normalize, then NCHW transpose. Mirrors
//! sparrow-engine-cpu's `resize_direct` (manifest method is "resize"). Used by the
//! GPU classifier path when the manifest opts into plain resize rather
//! than center-crop+resize.
//!
//! Phase 3.8 Step 1 follow-up — Amazon Camera Trap v2 onboarding extends
//! the kernel from `/255` only to `(px/255 - mean) / std`. The Unit case
//! (`mean=[0,0,0], std=[1,1,1]`) is bit-exact under IEEE 754 (`x-0=x`,
//! `x/1=x` are exact ops), so SpeciesNet's bit-tightness against
//! sparrow-engine-cpu's `resize_simd` is preserved.
//!
//! Reuses the `NormalizeStats` enum already defined in
//! `sparrow_engine_gpu::kernels::tiled_preprocess` — same `(mean, std)` shape used
//! by HerdNet / OWL-T tiled preprocess.

use std::sync::Arc;

use sparrow_engine_core::preprocess::checked_tensor_len_3hw;
use sparrow_engine_types::error::{SparrowEngineError, Result};
use sparrow_engine_types::manifest::ChannelOrder;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::decode::GpuImage;
use crate::kernels::tiled_preprocess::NormalizeStats;

const KERNEL_SRC: &str = include_str!("resize.cu");
const KERNEL_NAME: &str = "resize_kernel";

#[derive(Clone)]
pub struct ResizeKernel {
    func: CudaFunction,
}

impl ResizeKernel {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| SparrowEngineError::Ort(format!("nvrtc compile resize.cu: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_module resize: {e}")))?;
        let func = module
            .load_function(KERNEL_NAME)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_function resize: {e}")))?;
        Ok(Self { func })
    }
}

/// Convolutional bilinear resize + per-channel normalize + NCHW preprocess on GPU.
///
/// Direct resize from `src` (HWC u8 RGB) to (tgt_w × tgt_h × 3) NCHW f32,
/// no aspect preservation, no crop. Honours the manifest `channel_order`:
/// RGB → plane order [R, G, B]; BGR → plane order [B, G, R].
///
/// Bit-tight against `fast_image_resize::Resizer` with
/// `ResizeAlg::Convolution(FilterType::Bilinear)` — the exact algorithm
/// `sparrow-engine-cpu/src/preprocess.rs::resize_simd` uses, which the SpeciesNet
/// manifest method `"resize"` dispatches into. See `resize.cu` for the
/// algorithm derivation and the per-axis weight-computation loop.
///
/// `stats` controls the post-resize normalization. `NormalizeStats::UNIT`
/// reproduces the pre-Amazon `/255` behaviour bit-exactly.
///
/// Window size is `ceil(filter_radius) * 2 + 1`; for SpeciesNet
/// 1280×960 → 480×480 this is 7 (so up to 49 source-pixel reads per
/// output pixel). The kernel sizes its weight arrays to 16 to handle
/// up to ~7× downsample without spilling.
pub fn resize_gpu(
    stream: &Arc<CudaStream>,
    kernel: &ResizeKernel,
    src: &GpuImage,
    tgt_w: u32,
    tgt_h: u32,
    channel_order: ChannelOrder,
    stats: NormalizeStats,
) -> Result<CudaSlice<f32>> {
    let total = checked_tensor_len_3hw(tgt_h, tgt_w)?;
    let mut dst: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(total)
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc alloc_zeros (resize dst): {e}")))?;

    let bgr_flag: i32 = match channel_order {
        ChannelOrder::Rgb => 0,
        ChannelOrder::Bgr => 1,
    };

    const TX: u32 = 16;
    const TY: u32 = 16;
    let cfg = LaunchConfig {
        grid_dim: (tgt_w.div_ceil(TX), tgt_h.div_ceil(TY), 1),
        block_dim: (TX, TY, 1),
        shared_mem_bytes: 0,
    };

    let mut launch = stream.launch_builder(&kernel.func);
    let src_w_i: i32 = src.width as i32;
    let src_h_i: i32 = src.height as i32;
    let tgt_w_i: i32 = tgt_w as i32;
    let tgt_h_i: i32 = tgt_h as i32;
    let mean_r = stats.mean[0];
    let mean_g = stats.mean[1];
    let mean_b = stats.mean[2];
    let std_r = stats.std[0];
    let std_g = stats.std[1];
    let std_b = stats.std[2];
    // Unit-identity guard: when stats == NormalizeStats::UNIT, take the
    // fast-path branch in the kernel that emits `px/255` directly.
    // Belt-and-braces against any FMA / contraction-rewrite the NVRTC
    // backend might apply to the general `(px/255 - 0)/1` form. The
    // comparison is exact-equal against the public UNIT constants
    // (mean=[0,0,0], std=[1,1,1]) — any other Unit-equivalent stats
    // (e.g., mean=[0,0,0], std=[1.0,1.0,1.0001]) take the general path,
    // which is the correct behaviour.
    let unit_flag: i32 = if stats == NormalizeStats::UNIT { 1 } else { 0 };

    launch
        .arg(&src.data)
        .arg(&src_w_i)
        .arg(&src_h_i)
        .arg(&mut dst)
        .arg(&tgt_w_i)
        .arg(&tgt_h_i)
        .arg(&mean_r)
        .arg(&mean_g)
        .arg(&mean_b)
        .arg(&std_r)
        .arg(&std_g)
        .arg(&std_b)
        .arg(&unit_flag)
        .arg(&bgr_flag);

    // SAFETY: kernel signature matches args; bounds check inside kernel.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch resize_kernel: {e}")))?;

    Ok(dst)
}
