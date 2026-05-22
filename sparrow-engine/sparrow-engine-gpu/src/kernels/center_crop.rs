//! Rust wrapper around `center_crop.cu`. SpeciesNet (and other classifier)
//! preprocess: square center crop → bilinear resize → /255 normalize →
//! NCHW transpose. Default RGB channel order (manifest may override to BGR).

use std::sync::Arc;

use sparrow_engine_core::preprocess::checked_tensor_len_3hw;
use sparrow_engine_types::error::{SparrowEngineError, Result};
use sparrow_engine_types::manifest::ChannelOrder;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::decode::GpuImage;

const KERNEL_SRC: &str = include_str!("center_crop.cu");
const KERNEL_NAME: &str = "center_crop_kernel";

#[derive(Clone)]
pub struct CenterCropKernel {
    func: CudaFunction,
}

impl CenterCropKernel {
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| SparrowEngineError::Ort(format!("nvrtc compile center_crop.cu: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_module center_crop: {e}")))?;
        let func = module
            .load_function(KERNEL_NAME)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_function center_crop: {e}")))?;
        Ok(Self { func })
    }
}

/// Center-crop + resize + /255 normalize + NCHW preprocess on GPU.
///
/// Crop window is the largest centered square that fits inside the source
/// image. For non-square inputs this preserves the central content,
/// matching the SpeciesNet reference preprocess.
pub fn center_crop_gpu(
    stream: &Arc<CudaStream>,
    kernel: &CenterCropKernel,
    src: &GpuImage,
    tgt_w: u32,
    tgt_h: u32,
    channel_order: ChannelOrder,
) -> Result<CudaSlice<f32>> {
    let crop_size = src.width.min(src.height);
    let crop_x = (src.width - crop_size) / 2;
    let crop_y = (src.height - crop_size) / 2;

    let total = checked_tensor_len_3hw(tgt_h, tgt_w)?;
    let mut dst: CudaSlice<f32> = stream
        .alloc_zeros::<f32>(total)
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc alloc_zeros (center_crop dst): {e}")))?;

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
    let crop_x_i: i32 = crop_x as i32;
    let crop_y_i: i32 = crop_y as i32;
    let crop_size_i: i32 = crop_size as i32;

    launch
        .arg(&src.data)
        .arg(&src_w_i)
        .arg(&src_h_i)
        .arg(&mut dst)
        .arg(&tgt_w_i)
        .arg(&tgt_h_i)
        .arg(&crop_x_i)
        .arg(&crop_y_i)
        .arg(&crop_size_i)
        .arg(&bgr_flag);

    // SAFETY: kernel signature matches args; bounds check inside kernel.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch center_crop_kernel: {e}")))?;

    Ok(dst)
}
