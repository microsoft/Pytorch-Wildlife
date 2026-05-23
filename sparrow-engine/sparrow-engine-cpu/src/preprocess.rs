//! Image preprocessing: decode, letterbox/resize, normalize, layout conversion.
//!
//! Transforms an [`ImageInput`] into an `ndarray::Array4<f32>` tensor ready for
//! ONNX Runtime inference. Also returns geometric metadata (scale, padding) so
//! that postprocessing can map detections back to original image coordinates.

use image::RgbImage;
use ndarray::Array4;

use sparrow_engine_core::preprocess::checked_tensor_len_3hw;
use sparrow_engine_types::ImageInput;

use crate::error::{SparrowEngineError, Result};

// ---------------------------------------------------------------------------
// Configuration types
// ---------------------------------------------------------------------------

// Preprocessing enums are defined in manifest.rs (single source of truth for
// TOML-driven config). They live in sparrow-engine-types after Phase 3.8 Phase A;
// re-export for convenience and to keep the `sparrow_engine::preprocess::ChannelOrder`
// consumer path working (lib name is now "sparrow_engine" after the rename).
pub use sparrow_engine_types::manifest::{ChannelOrder, Layout, Normalization, PreprocessMethod};

// Phase 3.8 Phase A: PreprocessMeta + PreprocessConfig moved to
// sparrow-engine-types/src/preprocess_meta.rs (pure POD types, dep-direction-clean).
// Re-export so existing `sparrow_engine::preprocess::PreprocessMeta` /
// `sparrow_engine::preprocess::PreprocessConfig` paths in the rest of sparrow-engine-cpu
// (detect.rs, postprocess.rs) keep working.
pub use sparrow_engine_types::{PreprocessConfig, PreprocessMeta};

// ---------------------------------------------------------------------------
// Result type
// ---------------------------------------------------------------------------

/// Output of preprocessing — tensor plus geometric metadata for postprocessing.
#[derive(Debug, Clone)]
pub struct PreprocessResult {
    /// Image tensor. Shape is `[1, C, H, W]` (NCHW) or `[1, H, W, C]` (NHWC).
    pub tensor: Array4<f32>,
    /// Geometric metadata (scale, padding, original dims) for postprocess undo.
    pub meta: PreprocessMeta,
}

// ---------------------------------------------------------------------------
// ImageNet constants
// ---------------------------------------------------------------------------

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Preprocess an image for model inference.
///
/// Steps: decode → resize/letterbox → normalize → layout conversion → tensor.
pub fn preprocess(image: &ImageInput, config: &PreprocessConfig) -> Result<PreprocessResult> {
    // 1. Decode to RGB
    let rgb = decode_to_rgb(image)?;
    let (orig_w, orig_h) = (rgb.width(), rgb.height());

    if orig_w == 0 || orig_h == 0 {
        return Err(SparrowEngineError::ImageDecode(
            "Image has zero width or height".into(),
        ));
    }

    let target_w = config.input_size[0];
    let target_h = config.input_size[1];

    // 2. Resize / letterbox
    let (canvas, scale, pad_x, pad_y) = match config.method {
        PreprocessMethod::Letterbox => letterbox(
            &rgb,
            target_w,
            target_h,
            config.pad_value,
            &config.normalization,
        )?,
        PreprocessMethod::Resize => resize_direct(&rgb, target_w, target_h)?,
        PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. } => {
            return Err(crate::error::SparrowEngineError::InvalidManifest(format!(
                "{} preprocessing cannot be used with image preprocess()",
                config.method.as_str()
            )));
        }
    };

    // Letterbox canvas is already normalized; resize canvas is raw 0-255.
    // Pass Normalization::None for letterbox so build_tensor doesn't double-normalize.
    let tensor_norm = match config.method {
        PreprocessMethod::Letterbox => Normalization::None,
        PreprocessMethod::Resize => config.normalization,
        PreprocessMethod::MelSpectrogram { .. } | PreprocessMethod::RawAudio { .. } => {
            unreachable!()
        }
    };

    // 3. Layout conversion (+ normalization for resize path)
    let tensor = build_tensor(
        &canvas,
        target_w,
        target_h,
        &tensor_norm,
        config.layout,
        config.channel_order,
    );

    Ok(PreprocessResult {
        tensor,
        meta: PreprocessMeta {
            original_width: orig_w,
            original_height: orig_h,
            scale,
            pad_x,
            pad_y,
        },
    })
}

// ---------------------------------------------------------------------------
// Image decoding
// ---------------------------------------------------------------------------

/// Decode any `ImageInput` variant into an `RgbImage` (8-bit RGB).
///
/// Phase 3.8 Phase C W1 audit-fix R2 (CR-1): the body lives in
/// [`sparrow_engine_core::preprocess::decode_to_rgb`] so `sparrow-engine-cpu` and
/// `sparrow-engine-gpu` share one implementation. This wrapper preserves the
/// `sparrow_engine_cpu::preprocess::decode_to_rgb` call path used by
/// `sparrow_engine_cpu::detect::decode_image`.
pub(crate) fn decode_to_rgb(input: &ImageInput) -> Result<RgbImage> {
    sparrow_engine_core::preprocess::decode_to_rgb(input)
}

// ---------------------------------------------------------------------------
// Letterbox
// ---------------------------------------------------------------------------

/// Resize preserving aspect ratio, center on a padded canvas.
///
/// Returns `(canvas, scale, pad_x, pad_y)` where canvas pixels are already
/// normalized according to `norm` (so padding uses the post-normalization pad_value).
fn letterbox(
    img: &RgbImage,
    target_w: u32,
    target_h: u32,
    pad_value: f32,
    norm: &Normalization,
) -> Result<(Vec<f32>, f32, f32, f32)> {
    let (img_w, img_h) = (img.width() as f32, img.height() as f32);
    let scale = (target_w as f32 / img_w).min(target_h as f32 / img_h);

    let new_w = (img_w * scale).round().max(1.0).min(target_w as f32) as u32;
    let new_h = (img_h * scale).round().max(1.0).min(target_h as f32) as u32;

    // Resize using bilinear interpolation (SIMD via fast_image_resize, thread_local cached Resizer)
    let resized = resize_fast_bilinear(img, new_w, new_h)?;

    let pad_x = (target_w as f32 - new_w as f32) / 2.0;
    let pad_y = (target_h as f32 - new_h as f32) / 2.0;

    let pad_x_left = pad_x.floor() as u32;
    let pad_y_top = pad_y.ceil() as u32; // PW compatibility: extra pixel on TOP, not bottom

    // Build flat [H * W * 3] canvas filled with pad_value
    let total = checked_tensor_len_3hw(target_h, target_w)?;
    let mut canvas = vec![pad_value; total];

    // Place resized image onto canvas (normalized)
    for y in 0..new_h {
        for x in 0..new_w {
            let px = resized.get_pixel(x, y);
            let cy = (pad_y_top + y) as usize;
            let cx = (pad_x_left + x) as usize;
            if cy < target_h as usize && cx < target_w as usize {
                let base = (cy * target_w as usize + cx) * 3;
                let (r, g, b) = normalize_pixel(px[0], px[1], px[2], norm);
                canvas[base] = r;
                canvas[base + 1] = g;
                canvas[base + 2] = b;
            }
        }
    }

    Ok((canvas, scale, pad_x, pad_y))
}

// ---------------------------------------------------------------------------
// SIMD-accelerated resize (fast_image_resize, thread_local Resizer cache)
// ---------------------------------------------------------------------------

/// Bilinear resize using fast_image_resize's SIMD-accelerated convolution.
/// Resizer is thread-local to avoid per-call SIMD-detection + allocation overhead.
fn resize_fast_bilinear(img: &RgbImage, new_w: u32, new_h: u32) -> Result<RgbImage> {
    use fast_image_resize::images::Image as FirImage;
    use fast_image_resize::{
        FilterType as FirFilter, PixelType, ResizeAlg, ResizeOptions, Resizer,
    };
    use std::cell::RefCell;

    thread_local! {
        static RESIZER: RefCell<Resizer> = RefCell::new(Resizer::new());
    }

    let src = FirImage::from_vec_u8(
        img.width(),
        img.height(),
        img.as_raw().to_vec(),
        PixelType::U8x3,
    )
    .map_err(|e| {
        SparrowEngineError::ImageDecode(format!("fast_image_resize source allocation failed: {e}"))
    })?;
    let mut dst = FirImage::new(new_w, new_h, PixelType::U8x3);
    let opts = ResizeOptions::new().resize_alg(ResizeAlg::Convolution(FirFilter::Bilinear));
    RESIZER.with(|r| {
        r.borrow_mut()
            .resize(&src, &mut dst, &opts)
            .map_err(|e| SparrowEngineError::ImageDecode(format!("fast_image_resize failed: {e}")))
    })?;
    RgbImage::from_raw(new_w, new_h, dst.into_vec()).ok_or_else(|| {
        SparrowEngineError::ImageDecode(format!(
            "RgbImage::from_raw failed for resized image {new_w}x{new_h}"
        ))
    })
}

// ---------------------------------------------------------------------------
// Direct resize
// ---------------------------------------------------------------------------

/// Resize directly to target dimensions (no padding, may distort).
///
/// Returns `(canvas, scale=1.0, pad_x=0.0, pad_y=0.0)` with raw u8-as-f32 pixels
/// (normalization applied later in `build_tensor`).
///
/// NOTE: The returned scale/pad values are dummy placeholders (1.0/0.0). This function
/// is only used by heatmap models (HerdNet, OWL-T), which perform their own coordinate
/// mapping in `heatmap_peaks` (heatmap-space to [0,1]). These metadata values must NOT
/// be passed to `denormalize_and_normalize`, which assumes letterbox-based preprocessing.
fn resize_direct(
    img: &RgbImage,
    target_w: u32,
    target_h: u32,
) -> Result<(Vec<f32>, f32, f32, f32)> {
    let resized = resize_fast_bilinear(img, target_w, target_h)?;

    // Store as raw f32 (u8 cast) — normalization happens in build_tensor
    let total = checked_tensor_len_3hw(target_h, target_w)?;
    let mut canvas = Vec::with_capacity(total);
    for y in 0..target_h {
        for x in 0..target_w {
            let px = resized.get_pixel(x, y);
            canvas.push(px[0] as f32);
            canvas.push(px[1] as f32);
            canvas.push(px[2] as f32);
        }
    }

    Ok((canvas, 1.0, 0.0, 0.0))
}

// ---------------------------------------------------------------------------
// Normalization
// ---------------------------------------------------------------------------

/// Normalize a single pixel (R, G, B) according to the scheme.
fn normalize_pixel(r: u8, g: u8, b: u8, norm: &Normalization) -> (f32, f32, f32) {
    match norm {
        Normalization::Unit => (r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0),
        Normalization::Imagenet => (
            (r as f32 / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
            (g as f32 / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
            (b as f32 / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
        ),
        Normalization::None => (r as f32, g as f32, b as f32),
    }
}

/// Normalize a float-valued pixel (already in 0-255 range) according to the scheme.
fn normalize_pixel_f32(r: f32, g: f32, b: f32, norm: &Normalization) -> (f32, f32, f32) {
    match norm {
        Normalization::Unit => (r / 255.0, g / 255.0, b / 255.0),
        Normalization::Imagenet => (
            (r / 255.0 - IMAGENET_MEAN[0]) / IMAGENET_STD[0],
            (g / 255.0 - IMAGENET_MEAN[1]) / IMAGENET_STD[1],
            (b / 255.0 - IMAGENET_MEAN[2]) / IMAGENET_STD[2],
        ),
        Normalization::None => (r, g, b),
    }
}

// ---------------------------------------------------------------------------
// Tensor construction
// ---------------------------------------------------------------------------

/// Build an `Array4<f32>` tensor from a flat `[H*W*3]` canvas.
///
/// For letterbox: canvas pixels are already normalized → `norm` is ignored.
/// For resize: canvas pixels are raw 0-255 → `norm` is applied here.
///
/// `channel_order` controls plane order at the output:
/// - `Rgb`: plane 0 = R, plane 1 = G, plane 2 = B (sparrow-engine's pre-3.8 behaviour).
/// - `Bgr`: plane 0 = B, plane 1 = G, plane 2 = R (Ultralytics / YOLO convention).
///
/// The canvas is always RGB-ordered (produced by `decode_to_rgb`); the swap
/// only happens at this final step when emitting tensor planes.
fn build_tensor(
    canvas: &[f32],
    width: u32,
    height: u32,
    norm: &Normalization,
    layout: Layout,
    channel_order: ChannelOrder,
) -> Array4<f32> {
    let h = height as usize;
    let w = width as usize;

    match layout {
        Layout::Nchw => {
            // [1, 3, H, W] — build flat Vec then wrap with from_shape_vec to bypass
            // ndarray per-element bounds-checked indexing (R4 io-pipeline §5).
            let plane_size = h * w;
            let total = 3 * plane_size;
            // Zero-init: clippy::uninit_vec deny-by-default rejects with_capacity
            // + set_len even when every slot is written, because reading uninit
            // f32 is UB. Cost is <1% of preprocess wall time (tens of µs vs the
            // ms-scale JPEG decode + resize that dominate). Brings NCHW into
            // symmetry with the Nhwc branch's Array4::<f32>::zeros below.
            let mut buf: Vec<f32> = vec![0.0f32; total];
            let (plane0, rest) = buf.split_at_mut(plane_size);
            let (plane1, plane2) = rest.split_at_mut(plane_size);

            // For RGB: plane0=R, plane1=G, plane2=B
            // For BGR: plane0=B, plane1=G, plane2=R
            match channel_order {
                ChannelOrder::Rgb => {
                    for (i, chunk) in canvas.chunks_exact(3).enumerate() {
                        let (r, g, b) = normalize_pixel_f32(chunk[0], chunk[1], chunk[2], norm);
                        plane0[i] = r;
                        plane1[i] = g;
                        plane2[i] = b;
                    }
                }
                ChannelOrder::Bgr => {
                    for (i, chunk) in canvas.chunks_exact(3).enumerate() {
                        let (r, g, b) = normalize_pixel_f32(chunk[0], chunk[1], chunk[2], norm);
                        plane0[i] = b;
                        plane1[i] = g;
                        plane2[i] = r;
                    }
                }
            }

            Array4::from_shape_vec((1, 3, h, w), buf)
                .expect("from_shape_vec invariant: total = 1 * 3 * h * w")
        }
        Layout::Nhwc => {
            // [1, H, W, 3]
            let mut tensor = Array4::<f32>::zeros([1, h, w, 3]);
            for y in 0..h {
                for x in 0..w {
                    let base = (y * w + x) * 3;
                    let (r, g, b) = (canvas[base], canvas[base + 1], canvas[base + 2]);
                    let (r, g, b) = normalize_pixel_f32(r, g, b, norm);
                    let (c0, c1, c2) = match channel_order {
                        ChannelOrder::Rgb => (r, g, b),
                        ChannelOrder::Bgr => (b, g, r),
                    };
                    tensor[[0, y, x, 0]] = c0;
                    tensor[[0, y, x, 1]] = c1;
                    tensor[[0, y, x, 2]] = c2;
                }
            }
            tensor
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use sparrow_engine_types::PixelFormat;

    /// Helper: create a 4x3 red RGB image.
    fn red_image(w: u32, h: u32) -> RgbImage {
        let mut img = RgbImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                img.put_pixel(x, y, image::Rgb([255, 0, 0]));
            }
        }
        img
    }

    // Note: `decode_raw` / `bytes_per_pixel` tests moved to
    // `sparrow-engine-core/src/preprocess.rs::tests` per Phase 3.8 Phase C W1
    // audit-fix R2 CR-1 (decode_to_rgb hoist). The sparrow-engine-cpu
    // `decode_to_rgb` wrapper is exercised by `test_preprocess_*` below
    // (via `preprocess()` → `decode_to_rgb`).

    #[test]
    fn test_normalize_unit() {
        let (r, g, b) = normalize_pixel(255, 128, 0, &Normalization::Unit);
        assert!((r - 1.0).abs() < 1e-6);
        assert!((g - 128.0 / 255.0).abs() < 1e-6);
        assert!(b.abs() < 1e-6);
    }

    #[test]
    fn test_normalize_imagenet() {
        let (r, _g, _b) = normalize_pixel(255, 128, 0, &Normalization::Imagenet);
        let expected_r = (1.0 - 0.485) / 0.229;
        assert!((r - expected_r).abs() < 1e-4);
    }

    #[test]
    fn test_normalize_none() {
        let (r, g, b) = normalize_pixel(100, 200, 50, &Normalization::None);
        assert!((r - 100.0).abs() < 1e-6);
        assert!((g - 200.0).abs() < 1e-6);
        assert!((b - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_resize_direct_shape() {
        let img = red_image(100, 50);
        let (canvas, scale, pad_x, pad_y) = resize_direct(&img, 64, 64).unwrap();
        assert_eq!(canvas.len(), 64 * 64 * 3);
        assert!((scale - 1.0).abs() < 1e-6);
        assert!(pad_x.abs() < 1e-6);
        assert!(pad_y.abs() < 1e-6);
    }

    #[test]
    fn test_letterbox_preserves_aspect() {
        // 200x100 image → 640x640 letterbox
        let img = red_image(200, 100);
        let (_canvas, scale, pad_x, pad_y) =
            letterbox(&img, 640, 640, 0.0, &Normalization::Unit).unwrap();

        // scale = min(640/200, 640/100) = min(3.2, 6.4) = 3.2
        assert!((scale - 3.2).abs() < 1e-4);

        // new_w = 200*3.2 = 640, new_h = 100*3.2 = 320
        // pad_x = (640-640)/2 = 0, pad_y = (640-320)/2 = 160
        assert!(pad_x.abs() < 1e-4);
        assert!((pad_y - 160.0).abs() < 1e-4);
    }

    #[test]
    fn test_preprocess_nchw_shape() {
        let img = ImageInput::Raw {
            data: vec![128; 30 * 20 * 3],
            width: 30,
            height: 20,
            stride: 30 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [64, 64],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
        };
        let result = preprocess(&img, &config).unwrap();
        assert_eq!(result.tensor.shape(), &[1, 3, 64, 64]);
        assert_eq!(result.meta.original_width, 30);
        assert_eq!(result.meta.original_height, 20);
    }

    #[test]
    fn test_preprocess_nhwc_shape() {
        let img = ImageInput::Raw {
            data: vec![128; 30 * 20 * 3],
            width: 30,
            height: 20,
            stride: 30 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Letterbox,
            input_size: [128, 128],
            layout: Layout::Nhwc,
            normalization: Normalization::Unit,
            pad_value: 0.447,
            channel_order: ChannelOrder::Rgb,
        };
        let result = preprocess(&img, &config).unwrap();
        assert_eq!(result.tensor.shape(), &[1, 128, 128, 3]);
    }

    #[test]
    fn test_preprocess_unit_normalization_range() {
        // All-white pixel image through unit normalization should give 1.0
        let img = ImageInput::Raw {
            data: vec![255; 4 * 4 * 3],
            width: 4,
            height: 4,
            stride: 4 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [4, 4],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
        };
        let result = preprocess(&img, &config).unwrap();
        // All values should be 1.0
        for val in result.tensor.iter() {
            assert!((*val - 1.0).abs() < 1e-5, "Expected 1.0, got {val}");
        }
    }

    #[test]
    fn test_preprocess_letterbox_metadata() {
        // 640x480 → 1280x1280 letterbox
        let img = ImageInput::Raw {
            data: vec![128; 640 * 480 * 3],
            width: 640,
            height: 480,
            stride: 640 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Letterbox,
            input_size: [1280, 1280],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 114.0 / 255.0,
            channel_order: ChannelOrder::Rgb,
        };
        let result = preprocess(&img, &config).unwrap();

        assert_eq!(result.meta.original_width, 640);
        assert_eq!(result.meta.original_height, 480);

        // scale = min(1280/640, 1280/480) = min(2.0, 2.667) = 2.0
        assert!((result.meta.scale - 2.0).abs() < 1e-4);

        // new_w = 1280, new_h = 960
        // pad_x = 0, pad_y = (1280 - 960) / 2 = 160
        assert!(result.meta.pad_x.abs() < 1e-4);
        assert!((result.meta.pad_y - 160.0).abs() < 1e-4);
    }

    #[test]
    fn test_preprocess_meta_in_result() {
        // Verify that PreprocessResult.meta carries correct geometric values.
        let img = ImageInput::Raw {
            data: vec![128; 200 * 100 * 3],
            width: 200,
            height: 100,
            stride: 200 * 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Letterbox,
            input_size: [640, 640],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.447,
            channel_order: ChannelOrder::Rgb,
        };
        let result = preprocess(&img, &config).unwrap();

        assert_eq!(result.meta.original_width, 200);
        assert_eq!(result.meta.original_height, 100);
        // scale = min(640/200, 640/100) = min(3.2, 6.4) = 3.2
        assert!((result.meta.scale - 3.2).abs() < 1e-4);
        // new_w = 640, new_h = 320 → pad_x = 0, pad_y = (640-320)/2 = 160
        assert!(result.meta.pad_x.abs() < 1e-4);
        assert!((result.meta.pad_y - 160.0).abs() < 1e-4);
    }

    #[test]
    fn test_zero_size_image_guard() {
        // 0x0 image should be rejected.
        let img = ImageInput::Raw {
            data: vec![],
            width: 0,
            height: 0,
            stride: 0,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [64, 64],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
        };
        let err = preprocess(&img, &config).unwrap_err();
        match err {
            SparrowEngineError::ImageDecode(msg) => {
                assert!(msg.contains("zero"), "Expected zero-size error, got: {msg}");
            }
            other => panic!("Expected ImageDecode error, got: {other:?}"),
        }
    }

    #[test]
    fn test_extreme_aspect_ratio_no_zero_dim() {
        // 1x1281 image → 640x640 target would give new_w=0 without clamp.
        let pixels = vec![128u8; 3 * 1281];
        // Build a 1×1281 raw RGB image
        let img = ImageInput::Raw {
            data: pixels,
            width: 1,
            height: 1281,
            stride: 3,
            format: PixelFormat::Rgb,
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Letterbox,
            input_size: [640, 640],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
        };
        let result = preprocess(&img, &config);
        assert!(
            result.is_ok(),
            "Extreme aspect ratio should not fail: {:?}",
            result.err()
        );
        let prep = result.unwrap();
        assert_eq!(prep.tensor.shape(), &[1, 3, 640, 640]);
    }

    #[test]
    fn test_u32_overflow_stride_validation() {
        // width * bpp would overflow u32 for adversarial inputs
        let data = vec![0u8; 100];
        let img = ImageInput::Raw {
            data,
            width: u32::MAX,
            height: 1,
            stride: u32::MAX,
            format: PixelFormat::Rgba, // bpp=4, u32::MAX * 4 overflows
        };
        let config = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [640, 640],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
        };
        let result = preprocess(&img, &config);
        assert!(result.is_err(), "Should fail on u32 overflow stride");
    }

    #[test]
    fn test_channel_order_swap_rgb_vs_bgr() {
        // 1x1 pixel, R=200, G=100, B=50 (raw RGB).
        let img = ImageInput::Raw {
            data: vec![200, 100, 50],
            width: 1,
            height: 1,
            stride: 3,
            format: PixelFormat::Rgb,
        };

        // RGB ordering: plane 0 = R, plane 1 = G, plane 2 = B.
        let cfg_rgb = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [1, 1],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
        };
        let r_rgb = preprocess(&img, &cfg_rgb).unwrap();
        // After Unit normalization: R=200/255, G=100/255, B=50/255.
        assert!((r_rgb.tensor[[0, 0, 0, 0]] - 200.0 / 255.0).abs() < 1e-5);
        assert!((r_rgb.tensor[[0, 1, 0, 0]] - 100.0 / 255.0).abs() < 1e-5);
        assert!((r_rgb.tensor[[0, 2, 0, 0]] - 50.0 / 255.0).abs() < 1e-5);

        // BGR ordering: plane 0 = B, plane 1 = G, plane 2 = R.
        let cfg_bgr = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [1, 1],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 0.0,
            channel_order: ChannelOrder::Bgr,
        };
        let r_bgr = preprocess(&img, &cfg_bgr).unwrap();
        assert!((r_bgr.tensor[[0, 0, 0, 0]] - 50.0 / 255.0).abs() < 1e-5);
        assert!((r_bgr.tensor[[0, 1, 0, 0]] - 100.0 / 255.0).abs() < 1e-5);
        assert!((r_bgr.tensor[[0, 2, 0, 0]] - 200.0 / 255.0).abs() < 1e-5);
    }
}

#[cfg(test)]
mod phase_a_r2_preprocess {
    //! Regression tests added during Phase 3.8 Phase A audit-fix R2.
    //!
    //! Anchored on B1: `build_tensor` NCHW arm previously used
    //! `Vec::with_capacity(total) + unsafe { buf.set_len(total) }`, which
    //! triggered `clippy::uninit_vec` (deny-by-default). The fix replaces
    //! that pair with `vec![0.0f32; total]`. These tests lock the two
    //! load-bearing properties: (a) deterministic output values for a
    //! known input, and (b) byte-identity across repeated calls — the
    //! strongest guard against a future refactor that re-introduces
    //! uninitialized reads (uninit memory would in principle yield
    //! non-deterministic content; zero-init is fully deterministic).
    use super::*;

    /// 2×2 canvas of (R=255, G=0, B=0) pixels through the NCHW + RGB +
    /// Imagenet path. Asserts: shape, all-finite, exact per-plane values.
    /// Math:
    ///   R'  = (255/255 - 0.485) / 0.229 = 0.515 / 0.229 ≈ 2.2489
    ///   G'  = ( 0/255  - 0.456) / 0.224 = -0.456 / 0.224 ≈ -2.0357
    ///   B'  = ( 0/255  - 0.406) / 0.225 = -0.406 / 0.225 ≈ -1.8044
    #[test]
    fn build_tensor_nchw_rgb_imagenet_deterministic_and_finite() {
        // 2×2 canvas, RGB-interleaved, raw 0-255 (resize-path semantics:
        // build_tensor applies normalization for the resize path).
        let canvas: Vec<f32> = vec![
            255.0, 0.0, 0.0, // (0,0) red
            255.0, 0.0, 0.0, // (0,1) red
            255.0, 0.0, 0.0, // (1,0) red
            255.0, 0.0, 0.0, // (1,1) red
        ];

        let t = build_tensor(
            &canvas,
            2,
            2,
            &Normalization::Imagenet,
            Layout::Nchw,
            ChannelOrder::Rgb,
        );

        assert_eq!(t.shape(), &[1, 3, 2, 2]);
        assert_eq!(t.len(), 12);

        for v in t.iter() {
            assert!(
                v.is_finite(),
                "tensor element must be finite (NaN/inf indicates uninit-leak regression): {v}"
            );
        }

        // Plane 0 (R)
        let expected_r = (1.0 - 0.485) / 0.229;
        for y in 0..2 {
            for x in 0..2 {
                let v = t[[0, 0, y, x]];
                assert!(
                    (v - expected_r).abs() < 1e-4,
                    "R-plane at [0,0,{y},{x}]: expected ≈ {expected_r}, got {v}"
                );
            }
        }
        // Plane 1 (G)
        let expected_g = -0.456 / 0.224;
        for y in 0..2 {
            for x in 0..2 {
                let v = t[[0, 1, y, x]];
                assert!(
                    (v - expected_g).abs() < 1e-4,
                    "G-plane at [0,1,{y},{x}]: expected ≈ {expected_g}, got {v}"
                );
            }
        }
        // Plane 2 (B)
        let expected_b = -0.406 / 0.225;
        for y in 0..2 {
            for x in 0..2 {
                let v = t[[0, 2, y, x]];
                assert!(
                    (v - expected_b).abs() < 1e-4,
                    "B-plane at [0,2,{y},{x}]: expected ≈ {expected_b}, got {v}"
                );
            }
        }
    }

    /// Byte-identity of two `build_tensor` calls on the same input.
    /// Strongest regression guard: `vec![0.0f32; total]` is deterministic;
    /// a re-introduced `unsafe set_len` with uninitialized memory would
    /// (in principle) be non-deterministic.
    #[test]
    fn build_tensor_nchw_byte_deterministic_across_calls() {
        let canvas: Vec<f32> = vec![
            128.0, 64.0, 32.0, // (0,0)
            16.0, 8.0, 4.0, // (0,1)
            2.0, 1.0, 0.5, // (1,0)
            0.25, 0.125, 0.0625, // (1,1)
        ];

        let t1 = build_tensor(
            &canvas,
            2,
            2,
            &Normalization::Imagenet,
            Layout::Nchw,
            ChannelOrder::Rgb,
        );
        let t2 = build_tensor(
            &canvas,
            2,
            2,
            &Normalization::Imagenet,
            Layout::Nchw,
            ChannelOrder::Rgb,
        );

        // Bit-exact equality: both calls produce IEEE-754 identical f32s.
        // Use `to_bits()` rather than `==` to also fail on signed-zero or
        // NaN-payload divergence (NaN != NaN under PartialEq).
        let v1: Vec<u32> = t1.iter().map(|f| f.to_bits()).collect();
        let v2: Vec<u32> = t2.iter().map(|f| f.to_bits()).collect();
        assert_eq!(
            v1, v2,
            "build_tensor must be byte-deterministic across repeated calls"
        );
    }
}
