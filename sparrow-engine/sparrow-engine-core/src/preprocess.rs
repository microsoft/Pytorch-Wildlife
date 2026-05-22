//! Image decode helpers shared between `sparrow-engine-cpu` and `sparrow-engine-gpu`.
//!
//! Phase 3.8 Phase C W1 audit-fix R2 (CR-1): hoist `decode_to_rgb` from
//! `sparrow-engine-cpu::preprocess` to `sparrow-engine-core::preprocess` so both flavors
//! share a single byte-identical implementation. Per
//! `sparrow-engine-gpu/Cargo.toml` invariant ("sparrow-engine-gpu must not depend on
//! sparrow-engine-cpu — both consume sparrow-engine-core"), `sparrow-engine-core` is the only
//! sanctioned home for this shared CPU image-decode logic.
//!
//! The body is verbatim from `sparrow-engine-cpu/src/preprocess.rs` at
//! audit-fix-baseline `f5fb2df`; the only change is moving the pixel
//! manipulation + Raw-buffer handling into a public surface.
//!
//! Letterbox / resize / normalize / tensor build remain in
//! `sparrow-engine-cpu::preprocess` — those depend on `ndarray` + `fast_image_resize`
//! which are CPU-pipeline concerns.

use image::{ImageReader, RgbImage};

use sparrow_engine_types::{SparrowEngineError, ImageInput, PixelFormat, Result};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Decode any [`ImageInput`] variant into an [`RgbImage`] (8-bit RGB).
///
/// Returns:
/// - [`SparrowEngineError::ImageDecode`] on `image` crate decode failures (Encoded /
///   FilePath paths) or buffer-too-small errors (Raw path).
/// - [`SparrowEngineError::ImageFileNotFound`] when the FilePath input does not
///   exist (fast-path check before any decode attempt).
/// - [`SparrowEngineError::InvalidStride`] when a Raw input's stride is smaller
///   than `width * bytes_per_pixel(format)`.
pub fn decode_to_rgb(input: &ImageInput) -> Result<RgbImage> {
    match input {
        ImageInput::Encoded(bytes) => {
            let dyn_img = ImageReader::new(std::io::Cursor::new(bytes))
                .with_guessed_format()
                .map_err(|e| SparrowEngineError::ImageDecode(e.to_string()))?
                .decode()
                .map_err(|e| SparrowEngineError::ImageDecode(e.to_string()))?;
            Ok(dyn_img.to_rgb8())
        }
        ImageInput::FilePath(path) => {
            if !path.exists() {
                return Err(SparrowEngineError::ImageFileNotFound(path.clone()));
            }
            let dyn_img = image::open(path).map_err(|e| SparrowEngineError::ImageDecode(e.to_string()))?;
            Ok(dyn_img.to_rgb8())
        }
        ImageInput::Raw {
            data,
            width,
            height,
            stride,
            format,
        } => decode_raw(data, *width, *height, *stride, *format),
    }
}

// ---------------------------------------------------------------------------
// Tensor-size helpers
// ---------------------------------------------------------------------------

/// Checked element count for an NCHW image tensor with 3 channels.
pub fn checked_tensor_len_3hw(height: u32, width: u32) -> Result<usize> {
    let total = 3usize
        .checked_mul(height as usize)
        .and_then(|v| v.checked_mul(width as usize))
        .ok_or_else(|| {
            SparrowEngineError::ImageDecode(format!(
                "image tensor size overflows usize: 3x{height}x{width}"
            ))
        })?;
    Ok(total)
}

// ---------------------------------------------------------------------------
// Raw-buffer helpers
// ---------------------------------------------------------------------------

/// Construct an `RgbImage` from a raw pixel buffer, handling stride and format conversion.
fn decode_raw(
    data: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    format: PixelFormat,
) -> Result<RgbImage> {
    let bpp = bytes_per_pixel(format);
    let min_stride =
        width
            .checked_mul(bpp)
            .ok_or(SparrowEngineError::InvalidStride { stride, width, bpp })?;
    if stride < min_stride {
        return Err(SparrowEngineError::InvalidStride { stride, width, bpp });
    }

    let expected_len = stride as usize * height as usize;
    if data.len() < expected_len {
        return Err(SparrowEngineError::ImageDecode(format!(
            "Raw buffer too small: got {} bytes, expected at least {} ({}x{} stride={})",
            data.len(),
            expected_len,
            width,
            height,
            stride
        )));
    }

    let mut rgb = RgbImage::new(width, height);

    for y in 0..height {
        let row_start = (y * stride) as usize;
        for x in 0..width {
            let px_start = row_start + (x * bpp) as usize;
            let (r, g, b) = match format {
                PixelFormat::Rgb => (data[px_start], data[px_start + 1], data[px_start + 2]),
                PixelFormat::Rgba => (data[px_start], data[px_start + 1], data[px_start + 2]),
                PixelFormat::Bgr => (data[px_start + 2], data[px_start + 1], data[px_start]),
                PixelFormat::Bgra => (data[px_start + 2], data[px_start + 1], data[px_start]),
            };
            rgb.put_pixel(x, y, image::Rgb([r, g, b]));
        }
    }

    Ok(rgb)
}

/// Bytes per pixel for each pixel format.
fn bytes_per_pixel(format: PixelFormat) -> u32 {
    match format {
        PixelFormat::Rgb | PixelFormat::Bgr => 3,
        PixelFormat::Rgba | PixelFormat::Bgra => 4,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Low-level decode_raw / bytes_per_pixel tests (moved from sparrow-engine-cpu).
    // -----------------------------------------------------------------------

    #[test]
    fn test_bytes_per_pixel() {
        assert_eq!(bytes_per_pixel(PixelFormat::Rgb), 3);
        assert_eq!(bytes_per_pixel(PixelFormat::Rgba), 4);
        assert_eq!(bytes_per_pixel(PixelFormat::Bgra), 4);
        assert_eq!(bytes_per_pixel(PixelFormat::Bgr), 3);
    }

    #[test]
    fn test_decode_raw_rgb() {
        // 2x2 image, no extra stride
        let data = vec![
            255, 0, 0, 0, 255, 0, // row 0: red, green
            0, 0, 255, 128, 128, 128, // row 1: blue, gray
        ];
        let rgb = decode_raw(&data, 2, 2, 6, PixelFormat::Rgb).unwrap();
        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
        assert_eq!(rgb.get_pixel(1, 0), &image::Rgb([0, 255, 0]));
        assert_eq!(rgb.get_pixel(0, 1), &image::Rgb([0, 0, 255]));
    }

    #[test]
    fn test_decode_raw_bgra() {
        // 1x1 BGRA pixel: B=10, G=20, R=30, A=255
        let data = vec![10, 20, 30, 255];
        let rgb = decode_raw(&data, 1, 1, 4, PixelFormat::Bgra).unwrap();
        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([30, 20, 10]));
    }

    #[test]
    fn test_decode_raw_invalid_stride() {
        let data = vec![0; 12];
        let err = decode_raw(&data, 4, 1, 4, PixelFormat::Rgb).unwrap_err();
        match err {
            SparrowEngineError::InvalidStride { stride, width, bpp } => {
                assert_eq!(stride, 4);
                assert_eq!(width, 4);
                assert_eq!(bpp, 3);
            }
            _ => panic!("Expected InvalidStride, got: {err:?}"),
        }
    }

    #[test]
    fn test_decode_raw_with_stride_padding() {
        // 2x1 RGB image with stride=8 (2 bytes padding per row)
        let data = vec![
            255, 0, 0, 0, 255, 0, 0, 0, // row 0: red, green, 2 pad bytes
        ];
        let rgb = decode_raw(&data, 2, 1, 8, PixelFormat::Rgb).unwrap();
        assert_eq!(rgb.get_pixel(0, 0), &image::Rgb([255, 0, 0]));
        assert_eq!(rgb.get_pixel(1, 0), &image::Rgb([0, 255, 0]));
    }

    // -----------------------------------------------------------------------
    // Public-API decode_to_rgb tests (moved from sparrow-engine-gpu/pipeline::tests
    // — `raw_rgb_round_trip`, `raw_bgr_swaps_channels`). Translated from
    // `raw_to_dynamic_image` (which produced `DynamicImage`) to
    // `decode_to_rgb` via `ImageInput::Raw` (which produces `RgbImage`).
    // -----------------------------------------------------------------------

    #[test]
    fn raw_rgb_round_trip() {
        // 2x2 RGB image, tight stride.
        let data: Vec<u8> = vec![
            255, 0, 0, // (0,0) red
            0, 255, 0, // (1,0) green
            0, 0, 255, // (0,1) blue
            255, 255, 255, // (1,1) white
        ];
        let img = ImageInput::Raw {
            data,
            width: 2,
            height: 2,
            stride: 6,
            format: PixelFormat::Rgb,
        };
        let rgb = decode_to_rgb(&img).unwrap();
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(rgb.get_pixel(1, 0).0, [0, 255, 0]);
        assert_eq!(rgb.get_pixel(0, 1).0, [0, 0, 255]);
        assert_eq!(rgb.get_pixel(1, 1).0, [255, 255, 255]);
    }

    #[test]
    fn raw_bgr_swaps_channels() {
        let data: Vec<u8> = vec![
            0, 0, 255, // BGR red → R=255, G=0, B=0
            0, 255, 0, // BGR green → R=0, G=255, B=0
            255, 0, 0, // BGR blue → R=0, G=0, B=255
            255, 255, 255,
        ];
        let img = ImageInput::Raw {
            data,
            width: 2,
            height: 2,
            stride: 6,
            format: PixelFormat::Bgr,
        };
        let rgb = decode_to_rgb(&img).unwrap();
        assert_eq!(rgb.get_pixel(0, 0).0, [255, 0, 0]);
        assert_eq!(rgb.get_pixel(1, 0).0, [0, 255, 0]);
    }

    // -----------------------------------------------------------------------
    // Round-2 regression tests for reviewer F1-F3 (subsumed by CR-1 hoist).
    // Authored by reviewer; absorbed here so a future regression that
    // re-introduces the wrong `SparrowEngineError::Ort(...)` variant on these
    // decode paths fails fast at this single source-of-truth location.
    // F4 (`ImageBuffer::from_raw` size mismatch) is unreachable past
    // the upstream stride/length checks — no separate test per round-1
    // reviewer plan.
    // -----------------------------------------------------------------------

    #[test]
    fn raw_buffer_too_small_returns_image_decode() {
        // F1 — undersized Raw buffer must yield SparrowEngineError::ImageDecode, NOT Ort.
        let data: Vec<u8> = vec![255, 0, 0, 0, 255, 0]; // need 12, give 6
        let input = ImageInput::Raw {
            data,
            width: 2,
            height: 2,
            stride: 6,
            format: PixelFormat::Rgb,
        };
        let err = decode_to_rgb(&input).unwrap_err();
        assert!(
            matches!(err, SparrowEngineError::ImageDecode(_)),
            "expected ImageDecode, got {err:?}"
        );
    }

    #[test]
    fn decode_filepath_missing_returns_image_file_not_found() {
        // F2 — non-existent FilePath must yield ImageFileNotFound.
        let path = std::path::PathBuf::from("/tmp/__bongo_core_nonexistent_test_F2");
        let input = ImageInput::FilePath(path.clone());
        let err = decode_to_rgb(&input).unwrap_err();
        assert!(
            matches!(&err, SparrowEngineError::ImageFileNotFound(p) if p == &path),
            "expected ImageFileNotFound({path:?}), got {err:?}"
        );
    }

    #[test]
    fn decode_encoded_garbage_returns_image_decode() {
        // F3 — Encoded with non-image bytes must yield ImageDecode, not Ort.
        let garbage: Vec<u8> = vec![0xFF; 32];
        let input = ImageInput::Encoded(garbage);
        let err = decode_to_rgb(&input).unwrap_err();
        assert!(
            matches!(err, SparrowEngineError::ImageDecode(_)),
            "expected ImageDecode, got {err:?}"
        );
    }
}
