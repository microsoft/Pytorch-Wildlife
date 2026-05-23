//! End-to-end test for `engine_dispatch::viz::render_audio_heatmap`.
//!
//! Phase 3.5 Wave 3 / S9 (item #12). Exercises the audio heatmap renderer
//! against three real WAV fixtures (short / medium / long). ORT-free: builds
//! a synthetic spectrogram image and synthetic `AudioSegment` list per
//! fixture and checks that the renderer produces dimensionally-correct output
//! with the inferno colormap and monotonic confidence→alpha mapping.
//!
//! Visual inspection checklist: `docs/review/phase3.5-manual-test/manual_test_plan.md` §8.2.
//! Output PNGs (for manual inspection) land under
//! `sparrow-engine/test_outputs/libsparrow_engine/audio_heatmap_e2e/`.
//!
//! Run:
//! ```sh
//! ./scripts/test.sh -p sparrow-engine-core --test audio_heatmap_e2e -- --test-threads=1
//! ```

use std::fs;
use std::path::{Path, PathBuf};

use sparrow_engine_core::viz::{render_audio_heatmap, HeatmapOpts};
use sparrow_engine_types::AudioSegment;
use image::{DynamicImage, Rgba, RgbaImage};

const FIXTURES: &[(&str, f32, u32)] = &[
    ("short_2s.wav", 2.0, 200),      // 100 px/s
    ("medium_10s.wav", 10.0, 1000),  // 100 px/s
    ("long_30s.wav", 30.0, 3000),    // 100 px/s
];

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/audio")
}

fn output_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../test_outputs/libsparrow_engine/audio_heatmap_e2e")
}

/// Decode a WAV via hound and return its duration in seconds.
/// Used to assert fixture-file sanity, not rendering behaviour.
fn wav_duration_s(path: &Path) -> f32 {
    let reader = hound::WavReader::open(path)
        .unwrap_or_else(|e| panic!("open {}: {}", path.display(), e));
    let spec = reader.spec();
    let num_samples = reader.len() as f32;
    num_samples / spec.sample_rate as f32
}

/// Build a synthetic spectrogram background: gray gradient, not colormap-colored,
/// so heatmap overlay is visually distinguishable.
fn synthetic_spectrogram(w: u32, h: u32) -> DynamicImage {
    let mut img = RgbaImage::new(w, h);
    for y in 0..h {
        let v = (y as f32 / h as f32 * 128.0) as u8 + 32;
        for x in 0..w {
            img.put_pixel(x, y, Rgba([v, v, v, 255]));
        }
    }
    DynamicImage::ImageRgba8(img)
}

/// Build a monotonic confidence ramp across the fixture duration.
/// Three segments: low (0.3), medium (0.6), high (0.95). Used to test
/// that confidence-to-alpha mapping is monotonic.
fn ramp_segments(duration_s: f32) -> Vec<AudioSegment> {
    let third = duration_s / 3.0;
    vec![
        AudioSegment { start_time_s: 0.0, end_time_s: third, confidence: 0.3, classes: Vec::new() },
        AudioSegment { start_time_s: third, end_time_s: 2.0 * third, confidence: 0.6, classes: Vec::new() },
        AudioSegment { start_time_s: 2.0 * third, end_time_s: duration_s, confidence: 0.95, classes: Vec::new() },
    ]
}

/// Mean luminance (R+G+B / 3) across rows in a horizontal band.
///
/// Output alpha is always 255 (the renderer blends into an opaque canvas
/// and stamps alpha=255), so we cannot observe the confidence→alpha
/// internal mapping by reading the output alpha channel. Instead we observe
/// the *effective* blend strength via the shift in luminance relative to
/// the gray spectrogram base — higher confidence → stronger inferno
/// contribution → higher luminance at the brightest end of the colormap.
fn band_mean_luma(img: &RgbaImage, x_start: u32, x_end: u32) -> f32 {
    let (_, h) = img.dimensions();
    let mut sum = 0u64;
    let mut count = 0u64;
    for y in 0..h {
        for x in x_start..x_end {
            let p = img.get_pixel(x, y).0;
            sum += (p[0] as u64 + p[1] as u64 + p[2] as u64) / 3;
            count += 1;
        }
    }
    sum as f32 / count as f32
}

/// Core e2e: load fixture, render heatmap, run assertions, save PNG.
fn run_fixture(name: &str, expected_duration_s: f32, spec_width: u32) {
    let fixture_path = fixtures_dir().join(name);
    assert!(
        fixture_path.exists(),
        "fixture missing: {}",
        fixture_path.display()
    );

    // Fixture sanity: duration matches expectation within 5%.
    let actual = wav_duration_s(&fixture_path);
    let drift = (actual - expected_duration_s).abs() / expected_duration_s;
    assert!(
        drift < 0.05,
        "{}: duration {:.3}s does not match expected {:.3}s",
        name,
        actual,
        expected_duration_s
    );

    // Synthetic spectrogram: width proportional to duration, height fixed.
    let spec = synthetic_spectrogram(spec_width, 200);
    let segments = ramp_segments(actual);
    let opts = HeatmapOpts::default();

    let out = render_audio_heatmap(&spec, &segments, actual, &opts);
    let out_rgba = out.to_rgba8();

    // Dimensional correctness.
    assert_eq!(
        out_rgba.dimensions(),
        (spec_width, 200),
        "{}: output dimensions must match spectrogram",
        name
    );

    // Confidence→heat monotonicity (via output luminance shift — see
    // band_mean_luma doc). The high-conf band receives a brighter inferno
    // contribution than the low-conf band. Sample the center ±10% of each
    // band so blur smearing near band boundaries doesn't contaminate.
    let band_w = spec_width / 3;
    let margin = band_w / 5; // 10% margin each side = 20% center window
    let low_band = band_mean_luma(&out_rgba, margin, band_w - margin);
    let med_band = band_mean_luma(&out_rgba, band_w + margin, 2 * band_w - margin);
    let high_band = band_mean_luma(&out_rgba, 2 * band_w + margin, spec_width - margin);
    assert!(
        high_band > low_band + 10.0,
        "{}: confidence→heat not monotonic (low-luma={:.1}, med-luma={:.1}, high-luma={:.1})",
        name,
        low_band,
        med_band,
        high_band
    );
    assert!(
        med_band >= low_band,
        "{}: mid-band luma below low-band (low-luma={:.1}, med-luma={:.1})",
        name,
        low_band,
        med_band
    );
    assert!(
        high_band >= med_band,
        "{}: high-band luma below mid-band (med-luma={:.1}, high-luma={:.1})",
        name,
        med_band,
        high_band
    );

    // Inferno colormap sanity: the high-conf band must contain at least one
    // warm-tone pixel (R > 180, R >= G >= B). This matches inferno in the
    // 0.5 < t < 0.993 interpolation range, which blur smearing from the
    // high band's interior produces abundantly even after alpha-blend with
    // the gray base. It does NOT match the exact peak at t=1.0 = [252,255,164]
    // (where G > R by 3 LSB); the peak always has warm neighbors due to blur.
    // Weak check — intentionally avoids asserting exact colormap values.
    let mut saw_warm = false;
    for y in 0..out_rgba.height() {
        for x in 2 * band_w..spec_width {
            let p = out_rgba.get_pixel(x, y).0;
            // Warm tone: red dominates green dominates blue, and red > 180.
            if p[0] > 180 && p[0] >= p[1] && p[1] >= p[2] {
                saw_warm = true;
                break;
            }
        }
        if saw_warm {
            break;
        }
    }
    assert!(
        saw_warm,
        "{}: high-confidence band should contain inferno-warm pixels",
        name
    );

    // Save output PNG for manual visual inspection.
    let dir = output_dir();
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("mkdir {}: {}", dir.display(), e));
    let out_path = dir.join(format!("{}.png", Path::new(name).file_stem().unwrap().to_str().unwrap()));
    out.save(&out_path)
        .unwrap_or_else(|e| panic!("save {}: {}", out_path.display(), e));
}

#[test]
fn e2e_audio_heatmap_short() {
    run_fixture(FIXTURES[0].0, FIXTURES[0].1, FIXTURES[0].2);
}

#[test]
fn e2e_audio_heatmap_medium() {
    run_fixture(FIXTURES[1].0, FIXTURES[1].1, FIXTURES[1].2);
}

#[test]
fn e2e_audio_heatmap_long() {
    run_fixture(FIXTURES[2].0, FIXTURES[2].1, FIXTURES[2].2);
}
