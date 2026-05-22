//! Rust wrapper around `letterbox.cu`.
//!
//! Compiles the CUDA source via NVRTC at first call, caches the loaded
//! function for the lifetime of the [`LetterboxKernel`] instance.
//!
//! Output is `[3 * tgt_h * tgt_w]` FP32 NCHW, ready to feed to the ORT
//! IoBinding wrapper without further reshape.

use std::sync::Arc;

use sparrow_engine_core::preprocess::checked_tensor_len_3hw;
use sparrow_engine_types::error::{SparrowEngineError, Result};
use sparrow_engine_types::manifest::ChannelOrder;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::decode::GpuImage;

const KERNEL_SRC: &str = include_str!("letterbox.cu");
const KERNEL_NAME: &str = "letterbox_kernel";

/// Loaded letterbox kernel module + entry point. Cheap to clone: holds Arcs.
#[derive(Clone)]
pub struct LetterboxKernel {
    func: CudaFunction,
}

impl LetterboxKernel {
    /// Compile + load the kernel. Cache the result if you call this in a hot loop.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| SparrowEngineError::Ort(format!("nvrtc compile letterbox.cu: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_module letterbox: {e}")))?;
        let func = module
            .load_function(KERNEL_NAME)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_function letterbox: {e}")))?;
        Ok(Self { func })
    }
}

/// Letterbox + normalize + NCHW preprocess on GPU.
///
/// `pad_value` is in the closed range 0 to 1 (post-`/255` normalized).
/// 114/255 ≈ 0.447 is the PW/Ultralytics convention.
///
/// `channel_order` controls plane ordering at the output: `Rgb` →
/// plane0=R, plane1=G, plane2=B; `Bgr` → plane0=B, plane1=G, plane2=R.
///
/// Returns a `CudaSlice<f32>` of length `3 * tgt_h * tgt_w` ready to feed
/// to the ORT IoBinding wrapper.
#[allow(clippy::too_many_arguments)]
pub fn letterbox_gpu(
    stream: &Arc<CudaStream>,
    kernel: &LetterboxKernel,
    src: &GpuImage,
    tgt_w: u32,
    tgt_h: u32,
    pad_value: f32,
    channel_order: ChannelOrder,
) -> Result<(CudaSlice<f32>, LetterboxMeta)> {
    let (img_w, img_h) = (src.width as f32, src.height as f32);
    let scale = (tgt_w as f32 / img_w).min(tgt_h as f32 / img_h);

    let new_w = (img_w * scale).round().max(1.0).min(tgt_w as f32) as u32;
    let new_h = (img_h * scale).round().max(1.0).min(tgt_h as f32) as u32;
    let pad_x = (tgt_w as f32 - new_w as f32) / 2.0;
    let pad_y = (tgt_h as f32 - new_h as f32) / 2.0;
    let pad_x_left = pad_x.floor() as u32;
    // PW compatibility: extra pixel on TOP (mirrors sparrow_engine_cpu::preprocess::letterbox).
    let pad_y_top = pad_y.ceil() as u32;

    let total = checked_tensor_len_3hw(tgt_h, tgt_w)?;
    // Pre-zeroed: pad regions are filled by the kernel anyway, but a clean
    // baseline removes any UB from a botched kernel launch.
    let mut dst: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(total)
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc alloc_zeros (letterbox dst): {e}")))?;

    let bgr_flag: i32 = match channel_order {
        ChannelOrder::Rgb => 0,
        ChannelOrder::Bgr => 1,
    };

    // 16x16 thread block ⇒ 256 threads/block. tgt_w/tgt_h % 16 ≠ 0 is
    // handled by the kernel's bounds check.
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
    let new_w_i: i32 = new_w as i32;
    let new_h_i: i32 = new_h as i32;
    let pad_x_i: i32 = pad_x_left as i32;
    let pad_y_i: i32 = pad_y_top as i32;

    launch
        .arg(&src.data)
        .arg(&src_w_i)
        .arg(&src_h_i)
        .arg(&mut dst)
        .arg(&tgt_w_i)
        .arg(&tgt_h_i)
        .arg(&new_w_i)
        .arg(&new_h_i)
        .arg(&pad_x_i)
        .arg(&pad_y_i)
        .arg(&scale)
        .arg(&pad_value)
        .arg(&bgr_flag);

    // SAFETY: kernel signature matches the args bound above; bounds check
    // inside the kernel guards out-of-grid threads.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch letterbox_kernel: {e}")))?;

    let meta = LetterboxMeta {
        scale,
        pad_x,
        pad_y,
        original_width: src.width,
        original_height: src.height,
    };
    Ok((dst, meta))
}

/// Geometric metadata for un-mapping detections back to original-image space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LetterboxMeta {
    pub scale: f32,
    pub pad_x: f32,
    pub pad_y: f32,
    pub original_width: u32,
    pub original_height: u32,
}
