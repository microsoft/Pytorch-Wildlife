//! Rust wrapper around `tiled_preprocess.cu`.
//!
//! Phase 3.8 Step 1 Wave 4 follow-up — closes the GPU-pipeline gap left by
//! the initial Wave 4 MVP (which did CPU preprocess + GPU inference).
//!
//! Compiles the CUDA source via NVRTC at first call, caches the loaded
//! function for the lifetime of the [`TiledPreprocessKernel`] instance.
//!
//! Output is `[3 * tgt_h * tgt_w]` FP32 NCHW, ready to feed to ORT.
//! Today the FP32 buffer is `clone_dtoh`'d to a host `Array4<f32>` for
//! `Session::run` because `ort 2.0.0-rc.12` does not expose a clean public
//! `Value` constructor backed by GPU memory (same wall coder-w2 hit on the
//! YOLO path; see `models/yolo.rs` Wave 2 commentary). True IoBinding
//! wiring is the same follow-up across yolo.rs + tiled.rs.

use std::sync::Arc;

use sparrow_engine_core::preprocess::checked_tensor_len_3hw;
use sparrow_engine_types::error::{SparrowEngineError, Result};
use sparrow_engine_types::manifest::ChannelOrder;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::compile_ptx;

use crate::decode::GpuImage;

const KERNEL_SRC: &str = include_str!("tiled_preprocess.cu");
const KERNEL_NAME: &str = "tiled_preprocess_kernel";

/// Per-channel normalization stats: `out = (px / 255 - mean[c]) / std[c]`.
///
/// The kernel dispatches every variant through this single formula so we
/// don't need a separate Unit kernel + ImageNet kernel:
///
/// - **Unit** (OWL-T):  `mean = [0, 0, 0]`, `std = [1, 1, 1]` → `out = px / 255`.
/// - **ImageNet** (HerdNet): `mean = [0.485, 0.456, 0.406]`, `std = [0.229, 0.224, 0.225]`.
/// - **None** (raw 0..=255 passthrough): NOT representable here — sparrow-engine-cpu's
///   `Normalization::None` is unused for the tiled HerdNet/OWL-T MVP and
///   is rejected by `tiled.rs::TiledModel::load`. Add a kernel branch if a
///   future model requires it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NormalizeStats {
    pub mean: [f32; 3],
    pub std: [f32; 3],
}

impl NormalizeStats {
    /// `out = px / 255` (OWL-T).
    pub const UNIT: Self = NormalizeStats {
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    };
    /// ImageNet mean/std (HerdNet, SpeciesNet, most TF-vision pipelines).
    pub const IMAGENET: Self = NormalizeStats {
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };
}

/// Loaded tiled-preprocess kernel module + entry point.
///
/// Cheap to clone: holds Arcs internally via `CudaFunction`. Mirrors the
/// `LetterboxKernel` / `CenterCropKernel` shape so engine wiring (Wave 5+)
/// can treat all three preprocess kernels uniformly.
#[derive(Clone)]
pub struct TiledPreprocessKernel {
    func: CudaFunction,
}

impl TiledPreprocessKernel {
    /// Compile + load the kernel via NVRTC. Cache the result for the lifetime
    /// of `sparrow_engine_gpu::Engine` (or, in tests, hold one per-test).
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self> {
        let ptx = compile_ptx(KERNEL_SRC)
            .map_err(|e| SparrowEngineError::Ort(format!("nvrtc compile tiled_preprocess.cu: {e}")))?;
        let module = ctx
            .load_module(ptx)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_module tiled_preprocess: {e}")))?;
        let func = module
            .load_function(KERNEL_NAME)
            .map_err(|e| SparrowEngineError::Ort(format!("cudarc load_function tiled_preprocess: {e}")))?;
        Ok(Self { func })
    }
}

/// GPU per-tile preprocess: crop + zero-pad (edge tiles) + per-channel
/// `(px/255 - mean)/std` normalize + NCHW transpose + optional RGB↔BGR plane
/// swap.
///
/// `crop_w` ≤ `tgt_w` and `crop_h` ≤ `tgt_h`. The (crop_w, crop_h)
/// region maps to the top-left of the output canvas; remaining cells are
/// zero-padded (matching `sparrow_engine_cpu::preprocess::resize_direct` + the
/// `image::RgbImage::new(...)` black-canvas behavior in
/// `sparrow_engine_cpu::detect::detect_tiled` for edge tiles).
///
/// For HerdNet / OWL-T the manifest forces `tile_size == input_size`, so
/// `(crop_w, crop_h)` only differs from `(tgt_w, tgt_h)` for edge tiles.
///
/// Returns a `CudaSlice<f32>` of length `3 * tgt_h * tgt_w` ready for
/// ORT (after a `clone_dtoh` host-roundtrip to construct the `Value`; see
/// module doc).
#[allow(clippy::too_many_arguments)]
pub fn tiled_preprocess_gpu(
    stream: &Arc<CudaStream>,
    kernel: &TiledPreprocessKernel,
    src: &GpuImage,
    tile_x: u32,
    tile_y: u32,
    crop_w: u32,
    crop_h: u32,
    tgt_w: u32,
    tgt_h: u32,
    stats: NormalizeStats,
    channel_order: ChannelOrder,
) -> Result<CudaSlice<f32>> {
    if crop_w > tgt_w || crop_h > tgt_h {
        return Err(SparrowEngineError::Ort(format!(
            "tiled_preprocess_gpu: crop ({crop_w}x{crop_h}) exceeds target ({tgt_w}x{tgt_h}) — \
             this kernel does not resize, only crop+pad"
        )));
    }

    let total = checked_tensor_len_3hw(tgt_h, tgt_w)?;
    // Pre-zero is defensive — the kernel writes every output slot, but
    // pre-zero leaves a clean baseline if a future kernel-launch failure
    // returns early. Mirrors letterbox.cu's caller convention.
    let mut dst = stream
        .alloc_zeros::<f32>(total)
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc alloc_zeros (tiled_preprocess dst): {e}")))?;

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

    let img_w_i: i32 = src.width as i32;
    let img_h_i: i32 = src.height as i32;
    let tile_x_i: i32 = tile_x as i32;
    let tile_y_i: i32 = tile_y as i32;
    let crop_w_i: i32 = crop_w as i32;
    let crop_h_i: i32 = crop_h as i32;
    let tgt_w_i: i32 = tgt_w as i32;
    let tgt_h_i: i32 = tgt_h as i32;
    let mean_r = stats.mean[0];
    let mean_g = stats.mean[1];
    let mean_b = stats.mean[2];
    let std_r = stats.std[0];
    let std_g = stats.std[1];
    let std_b = stats.std[2];

    let mut launch = stream.launch_builder(&kernel.func);
    launch
        .arg(&src.data)
        .arg(&img_w_i)
        .arg(&img_h_i)
        .arg(&tile_x_i)
        .arg(&tile_y_i)
        .arg(&crop_w_i)
        .arg(&crop_h_i)
        .arg(&mut dst)
        .arg(&tgt_w_i)
        .arg(&tgt_h_i)
        .arg(&mean_r)
        .arg(&mean_g)
        .arg(&mean_b)
        .arg(&std_r)
        .arg(&std_g)
        .arg(&std_b)
        .arg(&bgr_flag);

    // SAFETY: kernel signature matches the args bound above; bounds check
    // inside the kernel guards out-of-grid threads.
    unsafe { launch.launch(cfg) }
        .map_err(|e| SparrowEngineError::Ort(format!("cudarc launch tiled_preprocess_kernel: {e}")))?;

    Ok(dst)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_stats_constants() {
        assert_eq!(NormalizeStats::UNIT.mean, [0.0; 3]);
        assert_eq!(NormalizeStats::UNIT.std, [1.0; 3]);
    }

    #[test]
    fn imagenet_stats_constants() {
        // Lock the constants so a typo doesn't silently shift parity.
        // Source: torchvision.transforms.Normalize defaults.
        assert_eq!(NormalizeStats::IMAGENET.mean, [0.485, 0.456, 0.406]);
        assert_eq!(NormalizeStats::IMAGENET.std, [0.229, 0.224, 0.225]);
    }
}
