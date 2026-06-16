//! Image preprocessing for the mobile (LiteRT) flavor.
//!
//! Mirrors the cpu flavor's letterbox geometry + normalization, but packs the
//! tensor in **NHWC** float32 (the layout that onnx2tf-converted `.tflite` image
//! models expect) instead of NCHW, and resizes via the `image` crate's bilinear
//! filter. The cpu flavor uses `fast_image_resize`; the sub-pixel difference
//! between the two bilinear resamplers is immaterial for detection parity and
//! keeps the mobile binary lean (no SIMD-resize dependency on the Pi).
//!
//! Shared with cpu/gpu and reused here: `decode_to_rgb`
//! ([`sparrow_engine_core::preprocess`]) and `PreprocessMeta`
//! ([`sparrow_engine_types`]). Postprocessing is the shared
//! [`sparrow_engine_core::postprocess::yolo_e2e`], which consumes the
//! [`PreprocessMeta`] returned here to undo the letterbox transform.

use image::RgbImage;

use sparrow_engine_types::manifest::{ChannelOrder, Normalization};
use sparrow_engine_types::PreprocessMeta;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Normalize a single `(R, G, B)` pixel per the manifest scheme. Identical to
/// the cpu flavor's `normalize_pixel` so cross-flavor outputs agree.
fn normalize_pixel(r: u8, g: u8, b: u8, norm: Normalization) -> (f32, f32, f32) {
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

/// Letterbox `rgb` into a flat `[1, target_h, target_w, 3]` NHWC f32 buffer plus
/// the [`PreprocessMeta`] needed to undo the transform in postprocessing.
///
/// Geometry matches the cpu flavor exactly: `scale` is the min fit ratio; the
/// resized image is centered with the extra odd pixel placed on **top**
/// (PW/Ultralytics compatibility); padding cells are filled with `pad_value` in
/// post-normalization scale. Channel order honors `channel_order` (RGB default;
/// BGR swaps channels 0 and 2 — the Ultralytics/YOLO convention).
pub(crate) fn letterbox_nhwc(
    rgb: &RgbImage,
    target_w: u32,
    target_h: u32,
    pad_value: f32,
    norm: Normalization,
    channel_order: ChannelOrder,
) -> (Vec<f32>, PreprocessMeta) {
    let orig_w = rgb.width();
    let orig_h = rgb.height();
    let (img_w, img_h) = (orig_w as f32, orig_h as f32);
    let scale = (target_w as f32 / img_w).min(target_h as f32 / img_h);

    let new_w = (img_w * scale).round().max(1.0).min(target_w as f32) as u32;
    let new_h = (img_h * scale).round().max(1.0).min(target_h as f32) as u32;

    // Bilinear resize (Triangle == bilinear in the `image` crate).
    let resized =
        image::imageops::resize(rgb, new_w, new_h, image::imageops::FilterType::Triangle);

    let pad_x = (target_w as f32 - new_w as f32) / 2.0;
    let pad_y = (target_h as f32 - new_h as f32) / 2.0;
    let pad_x_left = pad_x.floor() as usize;
    let pad_y_top = pad_y.ceil() as usize; // PW compat: extra pixel on TOP

    let tw = target_w as usize;
    let th = target_h as usize;

    // NHWC: row-major over (h, w, c). Pre-fill with the (post-norm) pad value.
    let mut buf = vec![pad_value; th * tw * 3];

    for y in 0..new_h as usize {
        let cy = pad_y_top + y;
        if cy >= th {
            continue;
        }
        for x in 0..new_w as usize {
            let cx = pad_x_left + x;
            if cx >= tw {
                continue;
            }
            let px = resized.get_pixel(x as u32, y as u32);
            let (r, g, b) = normalize_pixel(px[0], px[1], px[2], norm);
            let (c0, c1, c2) = match channel_order {
                ChannelOrder::Rgb => (r, g, b),
                ChannelOrder::Bgr => (b, g, r),
            };
            let base = (cy * tw + cx) * 3;
            buf[base] = c0;
            buf[base + 1] = c1;
            buf[base + 2] = c2;
        }
    }

    let meta = PreprocessMeta {
        original_width: orig_w,
        original_height: orig_h,
        scale,
        pad_x,
        pad_y,
    };
    (buf, meta)
}

/// Reinterpret an f32 slice as little-endian bytes for the LiteRT tensor buffer.
pub(crate) fn f32_slice_to_le_bytes(data: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(data.len() * 4);
    for &v in data {
        bytes.extend_from_slice(&v.to_le_bytes());
    }
    bytes
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn solid(w: u32, h: u32, color: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(w, h, Rgb(color))
    }

    #[test]
    fn letterbox_square_no_padding() {
        // Square input -> no padding, scale 1.0 at equal target.
        let img = solid(640, 640, [255, 0, 0]);
        let (buf, meta) = letterbox_nhwc(&img, 640, 640, 0.0, Normalization::Unit, ChannelOrder::Rgb);
        assert_eq!(buf.len(), 640 * 640 * 3);
        assert!((meta.scale - 1.0).abs() < 1e-6);
        assert!(meta.pad_x.abs() < 1e-6 && meta.pad_y.abs() < 1e-6);
        // First pixel is red, unit-normalized: (1.0, 0.0, 0.0).
        assert!((buf[0] - 1.0).abs() < 1e-6);
        assert!(buf[1].abs() < 1e-6);
        assert!(buf[2].abs() < 1e-6);
    }

    #[test]
    fn letterbox_wide_pads_top_bottom() {
        // 1280x640 -> scale 0.5 -> 640x320 centered in 640x640: vertical padding.
        let img = solid(1280, 640, [10, 20, 30]);
        let (buf, meta) =
            letterbox_nhwc(&img, 640, 640, 0.5, Normalization::None, ChannelOrder::Rgb);
        assert!((meta.scale - 0.5).abs() < 1e-6);
        assert!((meta.pad_y - 160.0).abs() < 1e-6); // (640-320)/2
        assert!(meta.pad_x.abs() < 1e-6);
        // Top row (y=0) is in the padded region -> pad_value 0.5.
        assert!((buf[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn channel_order_bgr_swaps() {
        let img = solid(8, 8, [255, 0, 0]); // pure red
        let (rgb, _) = letterbox_nhwc(&img, 8, 8, 0.0, Normalization::Unit, ChannelOrder::Rgb);
        let (bgr, _) = letterbox_nhwc(&img, 8, 8, 0.0, Normalization::Unit, ChannelOrder::Bgr);
        // RGB packs red into channel 0; BGR packs red into channel 2.
        assert!((rgb[0] - 1.0).abs() < 1e-6 && rgb[2].abs() < 1e-6);
        assert!(bgr[0].abs() < 1e-6 && (bgr[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn le_bytes_round_trip() {
        let v = vec![0.0f32, 1.0, -1.5, 3.25];
        let bytes = f32_slice_to_le_bytes(&v);
        assert_eq!(bytes.len(), 16);
        let back: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(v, back);
    }
}
