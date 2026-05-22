//! Day/night classification via BT.709 brightness heuristic.
//!
//! Ports Sparrow Studio Local `DetermineTimeOfDay()` exactly.

use sparrow_engine_types::{SparrowEngineError, Result};

/// Day or night classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DayNight {
    Day,
    Night,
}

/// Result of day/night classification.
#[derive(Debug, Clone)]
pub struct DayNightResult {
    pub classification: DayNight,
    pub mean_brightness: f32,
}

/// Classify image as day or night from encoded image bytes (JPEG/PNG).
///
/// BT.709 luma, samples every 8th pixel, threshold > 85 on [0,255].
pub fn day_night(image_data: &[u8]) -> Result<DayNightResult> {
    let img = decode_rgb8(image_data)?;
    let b = image_brightness_rgb(&img);
    Ok(DayNightResult {
        classification: if b > 85.0 {
            DayNight::Day
        } else {
            DayNight::Night
        },
        mean_brightness: b,
    })
}

/// Mean brightness on [0,255] from encoded image bytes.
pub fn image_brightness(image_data: &[u8]) -> Result<f32> {
    let img = decode_rgb8(image_data)?;
    Ok(image_brightness_rgb(&img))
}

fn decode_rgb8(data: &[u8]) -> Result<image::RgbImage> {
    Ok(image::load_from_memory(data)
        .map_err(|e| SparrowEngineError::ImageDecode(e.to_string()))?
        .into_rgb8())
}

fn image_brightness_rgb(img: &image::RgbImage) -> f32 {
    let (w, h) = img.dimensions();
    let mut sum = 0.0f64;
    let mut count = 0u64;
    for y in (0..h).step_by(8) {
        for x in (0..w).step_by(8) {
            let p = img.get_pixel(x, y);
            sum += 0.2126 * p[0] as f64 + 0.7152 * p[1] as f64 + 0.0722 * p[2] as f64;
            count += 1;
        }
    }
    if count == 0 {
        0.0
    } else {
        (sum / count as f64) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_solid_png(r: u8, g: u8, b: u8) -> Vec<u8> {
        let mut img = image::RgbImage::new(64, 64);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([r, g, b]);
        }
        let mut buf = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut buf);
        image::ImageEncoder::write_image(
            encoder,
            img.as_raw(),
            64,
            64,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();
        buf
    }

    #[test]
    fn white_is_day() {
        let data = make_solid_png(255, 255, 255);
        let result = day_night(&data).unwrap();
        assert_eq!(result.classification, DayNight::Day);
        assert!((result.mean_brightness - 255.0).abs() < 1.0);
    }

    #[test]
    fn black_is_night() {
        let data = make_solid_png(0, 0, 0);
        let result = day_night(&data).unwrap();
        assert_eq!(result.classification, DayNight::Night);
        assert!(result.mean_brightness < 1.0);
    }

    #[test]
    fn threshold_boundary() {
        // Brightness exactly at 85 should be Night (threshold is >85, not >=85).
        // BT.709: 0.2126*R + 0.7152*G + 0.0722*B = 85
        // Use a gray where all channels = 85: luma = 0.2126*85 + 0.7152*85 + 0.0722*85 = 85
        let data = make_solid_png(85, 85, 85);
        let result = day_night(&data).unwrap();
        assert_eq!(result.classification, DayNight::Night);
    }

    #[test]
    fn just_above_threshold() {
        let data = make_solid_png(86, 86, 86);
        let result = day_night(&data).unwrap();
        assert_eq!(result.classification, DayNight::Day);
    }

    #[test]
    fn brightness_function() {
        let data = make_solid_png(100, 100, 100);
        let b = image_brightness(&data).unwrap();
        assert!((b - 100.0).abs() < 1.0);
    }

    #[test]
    fn invalid_image_data() {
        let result = day_night(b"not an image");
        assert!(result.is_err());
    }
}

#[cfg(test)]
mod phase_a_r1_daynight {
    use super::*;

    /// JPEG round-trip: existing tests only cover PNG. This drives the JPEG
    /// branch in `image::load_from_memory` (transparent multi-codec dispatch).
    /// JPEG is lossy, so confidence threshold is wide on brightness equality.
    #[test]
    fn jpeg_input_works() {
        let mut img = image::RgbImage::new(64, 64);
        for pixel in img.pixels_mut() {
            *pixel = image::Rgb([200, 200, 200]);
        }
        let mut buf = Vec::new();
        // JpegEncoder needs a writer that implements Write. Quality 90 is high.
        let encoder = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 90);
        image::ImageEncoder::write_image(
            encoder,
            img.as_raw(),
            64,
            64,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();

        let result = day_night(&buf).unwrap();
        assert_eq!(result.classification, DayNight::Day);
        // JPEG quantisation can drift up to ~5 LSB on flat regions.
        assert!(
            (result.mean_brightness - 200.0).abs() < 5.0,
            "JPEG brightness drifted too far from source (got {})",
            result.mean_brightness
        );
    }

    /// 1×1 image: only one pixel exists, the stride-8 sampler must still hit
    /// it (loop covers `0..1 step 8 == [0]`) and `count == 1` so the divide
    /// has a non-zero divisor. Without the `count == 0` guard at line 60, this
    /// would div-by-zero; the test pins that the guard does NOT trigger here.
    #[test]
    fn single_pixel_image() {
        let mut img = image::RgbImage::new(1, 1);
        *img.get_pixel_mut(0, 0) = image::Rgb([255, 255, 255]);
        let mut buf = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut buf);
        image::ImageEncoder::write_image(
            encoder,
            img.as_raw(),
            1,
            1,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();

        let result = day_night(&buf).unwrap();
        assert_eq!(result.classification, DayNight::Day);
        assert!(
            (result.mean_brightness - 255.0).abs() < 1.0,
            "single-pixel sample should equal the only pixel's luma, got {}",
            result.mean_brightness
        );
    }

    /// Strict `> 85.0` boundary: gray(85) → Night, gray(86) → Day.
    /// `threshold_boundary` already covers the 85 → Night case but we expand
    /// it here to lock the boundary in one test (any flip to `>= 85` breaks).
    #[test]
    fn threshold_strict_inequality_85_vs_86() {
        let mut buf85 = Vec::new();
        let mut buf86 = Vec::new();
        for (rgb, buf) in [(85u8, &mut buf85), (86u8, &mut buf86)] {
            let mut img = image::RgbImage::new(8, 8);
            for p in img.pixels_mut() {
                *p = image::Rgb([rgb, rgb, rgb]);
            }
            let encoder = image::codecs::png::PngEncoder::new(&mut *buf);
            image::ImageEncoder::write_image(
                encoder,
                img.as_raw(),
                8,
                8,
                image::ExtendedColorType::Rgb8,
            )
            .unwrap();
        }
        assert_eq!(day_night(&buf85).unwrap().classification, DayNight::Night);
        assert_eq!(day_night(&buf86).unwrap().classification, DayNight::Day);
    }

    /// Half-black / half-white split (16×16, top half black). With BT.709
    /// luma 0.2126R + 0.7152G + 0.0722B, both halves of a [255,255,255] split
    /// average to 127.5. The stride-8 sampler hits y∈{0,8} and x∈{0,8}, so we
    /// see 1 black row and 1 white row → mean ≈ 127.5. Tolerance accounts for
    /// f64→f32 cast at line 63.
    #[test]
    fn non_uniform_pattern_bt709_weighted_mean() {
        let mut img = image::RgbImage::new(16, 16);
        for y in 0..16u32 {
            for x in 0..16u32 {
                let v = if y < 8 { 0u8 } else { 255u8 };
                *img.get_pixel_mut(x, y) = image::Rgb([v, v, v]);
            }
        }
        let mut buf = Vec::new();
        let encoder = image::codecs::png::PngEncoder::new(&mut buf);
        image::ImageEncoder::write_image(
            encoder,
            img.as_raw(),
            16,
            16,
            image::ExtendedColorType::Rgb8,
        )
        .unwrap();

        let b = image_brightness(&buf).unwrap();
        // Sampler at y ∈ {0, 8} hits one black row and one white row, two
        // x-samples each → 4 samples: 0, 0, 255, 255 → mean = 127.5.
        assert!(
            (b - 127.5).abs() < 1.0,
            "half-black/half-white BT.709 mean must be ~127.5, got {}",
            b
        );
    }
}
