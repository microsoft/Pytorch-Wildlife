//
// Phase 3.8 Phase A: the letterbox math lives in `sparrow-engine-cpu/src/preprocess.rs`
// and is exercised through the public `preprocess()` function. The src-level
// `mod tests` already covers the 200x100 → 640x640 wide-aspect case and the
// metadata round-trip. This file adds 4 NEW edge cases at the integration
// boundary, using ONLY the public API (no `pub(crate)` helpers).
//
// All 4 tests are ORT-free.

use sparrow_engine::manifest::{ChannelOrder, Layout, Normalization, PreprocessMethod};
use sparrow_engine::{ImageInput, PixelFormat, PreprocessConfig};

fn raw_image(w: u32, h: u32) -> ImageInput {
    ImageInput::Raw {
        data: vec![128; (w * h * 3) as usize],
        width: w,
        height: h,
        stride: w * 3,
        format: PixelFormat::Rgb,
    }
}

fn letterbox_cfg(target: u32) -> PreprocessConfig {
    PreprocessConfig {
        method: PreprocessMethod::Letterbox,
        input_size: [target, target],
        layout: Layout::Nchw,
        normalization: Normalization::Unit,
        pad_value: 114.0 / 255.0,
        channel_order: ChannelOrder::Rgb,
    }
}

// -----------------------------------------------------------------------------
// Test 1: 1x1 source → 640x640 letterbox produces well-formed tensor.
// Edge case: minimum-area input must not panic; output tensor shape correct;
// scale and pad values are sensible.
// -----------------------------------------------------------------------------

#[test]
fn letterbox_1x1_source_produces_valid_tensor() {
    let img = raw_image(1, 1);
    let cfg = letterbox_cfg(640);
    let result = sparrow_engine::preprocess::preprocess(&img, &cfg).expect("preprocess(1x1)");

    assert_eq!(result.tensor.shape(), &[1, 3, 640, 640]);
    assert!(result.meta.scale > 0.0, "scale must be positive");
    assert!(result.meta.pad_x >= 0.0, "pad_x must be non-negative");
    assert!(result.meta.pad_y >= 0.0, "pad_y must be non-negative");
    assert_eq!(result.meta.original_width, 1);
    assert_eq!(result.meta.original_height, 1);
}

// -----------------------------------------------------------------------------
// Test 2: square source = target (640x640) → scale = 1.0, pad_x = pad_y = 0.
// -----------------------------------------------------------------------------

#[test]
fn letterbox_square_source_equals_target_zero_pad() {
    let img = raw_image(640, 640);
    let cfg = letterbox_cfg(640);
    let result = sparrow_engine::preprocess::preprocess(&img, &cfg).expect("preprocess(640x640)");

    assert!(
        (result.meta.scale - 1.0).abs() < 1e-5,
        "scale should be 1.0 for square match, got {}",
        result.meta.scale
    );
    assert!(
        result.meta.pad_x.abs() < 1e-5,
        "pad_x should be 0 for square match, got {}",
        result.meta.pad_x
    );
    assert!(
        result.meta.pad_y.abs() < 1e-5,
        "pad_y should be 0 for square match, got {}",
        result.meta.pad_y
    );
}

// -----------------------------------------------------------------------------
// Test 3: wide source (800x400) → 640x640. After letterbox: scale = 0.8,
// new_w = 640, new_h = 320; pad_y > 0, pad_x = 0.
// -----------------------------------------------------------------------------

#[test]
fn letterbox_wide_source_pads_vertically() {
    let img = raw_image(800, 400);
    let cfg = letterbox_cfg(640);
    let result = sparrow_engine::preprocess::preprocess(&img, &cfg).expect("preprocess(800x400)");

    // scale = min(640/800, 640/400) = min(0.8, 1.6) = 0.8
    assert!(
        (result.meta.scale - 0.8).abs() < 1e-4,
        "scale should be 0.8 for 800x400 → 640, got {}",
        result.meta.scale
    );
    // pad_x should be ~0 (resized width = 640, target = 640).
    assert!(
        result.meta.pad_x.abs() < 1e-4,
        "pad_x should be 0 for wide source, got {}",
        result.meta.pad_x
    );
    // pad_y should be (640 - 320) / 2 = 160.
    assert!(
        (result.meta.pad_y - 160.0).abs() < 1e-4,
        "pad_y should be 160 for wide source, got {}",
        result.meta.pad_y
    );
}

// -----------------------------------------------------------------------------
// Test 4: tall source (400x800) → 640x640. After letterbox: scale = 0.8,
// new_w = 320, new_h = 640; pad_x > 0, pad_y = 0.
// -----------------------------------------------------------------------------

#[test]
fn letterbox_tall_source_pads_horizontally() {
    let img = raw_image(400, 800);
    let cfg = letterbox_cfg(640);
    let result = sparrow_engine::preprocess::preprocess(&img, &cfg).expect("preprocess(400x800)");

    // scale = min(640/400, 640/800) = min(1.6, 0.8) = 0.8
    assert!(
        (result.meta.scale - 0.8).abs() < 1e-4,
        "scale should be 0.8 for 400x800 → 640, got {}",
        result.meta.scale
    );
    // pad_x should be (640 - 320) / 2 = 160.
    assert!(
        (result.meta.pad_x - 160.0).abs() < 1e-4,
        "pad_x should be 160 for tall source, got {}",
        result.meta.pad_x
    );
    // pad_y should be ~0.
    assert!(
        result.meta.pad_y.abs() < 1e-4,
        "pad_y should be 0 for tall source, got {}",
        result.meta.pad_y
    );
}
