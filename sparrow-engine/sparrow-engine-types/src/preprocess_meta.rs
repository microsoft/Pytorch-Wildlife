//! Preprocessing configuration + geometric metadata, surgically extracted from
//! the legacy monolithic preprocess module for Phase 3.8 Phase A. Pure POD types; the
//! actual `preprocess()` function (which depends on `image` + `ndarray` + the
//! letterbox/normalize pipeline) stays in sparrow-engine-cpu.
//!
//! Postprocess functions accept `&PreprocessMeta` to reverse the spatial
//! transform — that contract holds across the sparrow-engine-cpu / sparrow-engine-gpu split
//! because `PreprocessMeta` is pure POD and lives in this leaf crate.

use crate::manifest::{ChannelOrder, Layout, Normalization, PreprocessMethod};

/// Full preprocessing configuration, typically derived from a model manifest.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub method: PreprocessMethod,
    /// Target [width, height] for the model input.
    pub input_size: [u32; 2],
    pub layout: Layout,
    pub normalization: Normalization,
    /// Fill value for letterbox padding, in **post-normalization** scale.
    /// e.g., for unit normalization: 114.0/255.0 ≈ 0.447.
    pub pad_value: f32,
    /// Channel order at the model input. RGB is the sparrow-engine default (decode_to_rgb).
    /// BGR is required for YOLO-family models trained via Ultralytics; setting this
    /// causes `build_tensor` to swap channels 0 and 2 when emitting NCHW planes.
    pub channel_order: ChannelOrder,
}

/// Geometric metadata from preprocessing, needed by postprocessing to undo letterbox.
///
/// This is the single source of truth for the preprocessing transform parameters.
/// Postprocess functions accept `&PreprocessMeta` to reverse the spatial transform.
#[derive(Debug, Clone, Copy)]
pub struct PreprocessMeta {
    /// Original image width before any transforms.
    pub original_width: u32,
    /// Original image height before any transforms.
    pub original_height: u32,
    /// Resize scale factor applied to the original image.
    pub scale: f32,
    /// Horizontal padding in model-input pixel space (letterbox only).
    pub pad_x: f32,
    /// Vertical padding in model-input pixel space (letterbox only).
    pub pad_y: f32,
}

#[cfg(test)]
mod phase_a_r1_preprocess_meta_tests {
    use super::*;
    use crate::manifest::{ChannelOrder, Layout, Normalization, PreprocessMethod};

    #[test]
    fn preprocess_meta_is_copy_via_assignment() {
        // PreprocessMeta derives Copy — two assignments from one source must
        // not move ownership. This test compiles only if Copy is wired up.
        let meta = PreprocessMeta {
            original_width: 1920,
            original_height: 1080,
            scale: 0.6667,
            pad_x: 0.0,
            pad_y: 60.0,
        };
        let copy_a = meta;
        let copy_b = meta; // would not compile if Copy were removed
        assert_eq!(copy_a.original_width, 1920);
        assert_eq!(copy_b.original_height, 1080);
        assert!((copy_a.scale - 0.6667).abs() < 1e-6);
        assert!((copy_a.pad_x - copy_b.pad_x).abs() < f32::EPSILON);
        assert!((copy_a.pad_y - 60.0).abs() < f32::EPSILON);
    }

    #[test]
    fn preprocess_config_clone_preserves_fields() {
        // Clone + Debug are required for log emission inside the engine crates.
        let cfg = PreprocessConfig {
            method: PreprocessMethod::Letterbox,
            input_size: [640, 640],
            layout: Layout::Nchw,
            normalization: Normalization::Unit,
            pad_value: 114.0 / 255.0,
            channel_order: ChannelOrder::Bgr,
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.method, cfg.method);
        assert_eq!(cloned.input_size, cfg.input_size);
        assert_eq!(cloned.layout, cfg.layout);
        assert_eq!(cloned.normalization, cfg.normalization);
        assert!((cloned.pad_value - cfg.pad_value).abs() < f32::EPSILON);
        assert_eq!(cloned.channel_order, cfg.channel_order);
    }

    #[test]
    fn preprocess_config_debug_renders_without_panic() {
        // Smoke test — Debug is auto-derived but easy to lose if a non-Debug
        // field is added later. This guards against that regression.
        let cfg = PreprocessConfig {
            method: PreprocessMethod::Resize,
            input_size: [224, 224],
            layout: Layout::Nchw,
            normalization: Normalization::Imagenet,
            pad_value: 0.0,
            channel_order: ChannelOrder::Rgb,
        };
        let debug_str = format!("{cfg:?}");
        assert!(debug_str.contains("PreprocessConfig"));
        assert!(debug_str.contains("Resize"));
    }
}
