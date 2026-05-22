//! Integration coverage for Phase 3.5 S3 — ModelType-driven viz dispatch.
//!
//! These tests exercise the public API boundary:
//! - Manifest loading populates `ModelManifest.subtype`.
//! - `catalog::list_available_models` reports `ModelType::OverheadDetector`
//!   for `subtype = "overhead"` detectors and `ModelType::Detector` for
//!   Standard / absent-subtype manifests.
//! - `viz::render` dispatches to the dot path for `OverheadDetector` and the
//!   bbox path for everything else, independent of bbox pixel size.
//!
//! The ONNX model files are NOT required for these tests; `list_available_models`
//! and `load_manifest` work from TOML alone. `viz::render` operates on a
//! synthetic in-memory image.

use std::io::Write;

use image::DynamicImage;
use sparrow_engine_core::catalog;
use sparrow_engine_core::viz::{render, BboxAnnotation, RenderOpts};
use sparrow_engine_types::manifest::load_manifest;
use sparrow_engine_types::{ModelSubtype, ModelType};

/// Write an overhead HerdNet-like manifest to `{dir}/{id}/manifest.toml`.
fn write_model_dir(
    parent: &std::path::Path,
    id: &str,
    postprocess: &str,
    subtype_line: &str,
) -> std::path::PathBuf {
    let model_dir = parent.join(id);
    std::fs::create_dir_all(&model_dir).unwrap();
    let tile_block = match postprocess {
        "heatmap_peaks" => {
            "[inference]\nstrategy = \"tiled\"\ntile_size = [512, 512]\ntile_overlap = 0\n"
        }
        _ => "[inference]\nstrategy = \"single\"\n",
    };
    let (method, input_size, norm, post_extras) = match postprocess {
        "heatmap_peaks" => (
            "resize",
            "[512, 512]",
            "imagenet",
            "peak_threshold = 0.2\nadaptive = false\npoint_to_box_half_size = 10",
        ),
        _ => ("letterbox", "[640, 640]", "unit", "confidence_threshold = 0.2"),
    };
    let toml = format!(
        r#"
[model]
id = "{id}"
format = "onnx"
file = "model.onnx"
{subtype_line}

[preprocessing]
method = "{method}"
input_size = {input_size}
layout = "nchw"
normalization = "{norm}"

{tile_block}
[postprocessing]
method = "{postprocess}"
{post_extras}
"#
    );
    let path = model_dir.join("manifest.toml");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(toml.as_bytes()).unwrap();
    path
}

#[test]
fn overhead_manifest_promotes_to_overhead_detector_via_catalog() {
    let dir = tempfile::tempdir().unwrap();
    write_model_dir(
        dir.path(),
        "herdnet-overhead",
        "heatmap_peaks",
        r#"subtype = "overhead""#,
    );
    write_model_dir(
        dir.path(),
        "herdnet-legacy-no-subtype",
        "heatmap_peaks",
        "",
    );
    write_model_dir(
        dir.path(),
        "mdv6-standard",
        "yolo_e2e",
        r#"subtype = "standard""#,
    );

    let models = catalog::list_available_models(dir.path());
    assert_eq!(models.len(), 3, "three manifests written, three expected");

    let overhead = models
        .iter()
        .find(|m| m.id == "herdnet-overhead")
        .expect("overhead model listed");
    let legacy = models
        .iter()
        .find(|m| m.id == "herdnet-legacy-no-subtype")
        .expect("legacy model listed");
    let standard = models
        .iter()
        .find(|m| m.id == "mdv6-standard")
        .expect("standard model listed");

    assert_eq!(
        overhead.model_type,
        ModelType::OverheadDetector,
        "subtype = overhead must promote to OverheadDetector"
    );
    assert_eq!(
        legacy.model_type,
        ModelType::Detector,
        "missing subtype must default to Detector (backward compat)"
    );
    assert_eq!(
        standard.model_type,
        ModelType::Detector,
        "subtype = standard must resolve to Detector"
    );
}

#[test]
fn overhead_manifest_loads_with_overhead_subtype() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_model_dir(
        dir.path(),
        "owlt-overhead",
        "heatmap_peaks",
        r#"subtype = "overhead""#,
    );
    let m = load_manifest(&path).expect("manifest parses");
    assert_eq!(m.subtype, ModelSubtype::Overhead);
}

/// Dot-path render differs from bbox-path render for the same annotation.
///
/// This locks in the removal of the pixel-size heuristic: the ONLY dispatch
/// signal is `RenderOpts.model_type`.
#[test]
fn render_dispatch_differs_by_model_type() {
    let img = DynamicImage::new_rgb8(1000, 1000);
    let ann = BboxAnnotation {
        bbox: [0.45, 0.45, 0.55, 0.55], // 100×100 px centroid
        label: "animal".to_string(),
        confidence: 0.9,
    };

    let det_opts = RenderOpts {
        model_type: ModelType::Detector,
        ..Default::default()
    };
    let ovh_opts = RenderOpts {
        model_type: ModelType::OverheadDetector,
        point_radius: 8,
        ..Default::default()
    };

    let det = render(&img, std::slice::from_ref(&ann), &det_opts);
    let ovh = render(&img, std::slice::from_ref(&ann), &ovh_opts);
    assert_ne!(
        det.to_rgba8().as_raw(),
        ovh.to_rgba8().as_raw(),
        "bbox and dot renders must differ"
    );
}

/// High-resolution overhead use-case: a 0.5% bbox at 4K used to trigger the
/// pixel-size heuristic's bbox path (false negative for overhead models). With
/// ModelType dispatch, an overhead model renders a dot regardless of canvas
/// resolution.
#[test]
fn render_overhead_dot_at_4k_is_not_bbox() {
    let img = DynamicImage::new_rgb8(3840, 2160);
    let ann = BboxAnnotation {
        bbox: [0.498, 0.498, 0.503, 0.503], // 19×11 px — too big for old is_point
        label: "animal".to_string(),
        confidence: 0.9,
    };

    let ovh_opts = RenderOpts {
        model_type: ModelType::OverheadDetector,
        point_radius: 4,
        ..Default::default()
    };
    let det_opts = RenderOpts {
        model_type: ModelType::Detector,
        point_radius: 4,
        ..Default::default()
    };

    let ovh = render(&img, std::slice::from_ref(&ann), &ovh_opts).to_rgba8();
    let det = render(&img, std::slice::from_ref(&ann), &det_opts).to_rgba8();

    // Count non-black pixels as a cheap render-activity proxy.
    let nonblack = |canvas: &image::RgbaImage| {
        canvas
            .pixels()
            .filter(|p| p.0[0] != 0 || p.0[1] != 0 || p.0[2] != 0)
            .count()
    };
    let ovh_px = nonblack(&ovh);
    let det_px = nonblack(&det);
    assert!(
        ovh_px > 0,
        "overhead dot must paint visible pixels at 4K resolution"
    );
    assert!(det_px > 0, "detector bbox must paint visible pixels");
    // A filled radius-4 circle ≈ 49 pixels. A 19×11 bbox outline at line_width=2
    // paints ≈ 2*(19+11)*2 - 4*2*2 = 104 pixels. The exact numbers don't matter;
    // what matters is that the renders differ.
    assert_ne!(
        ovh_px, det_px,
        "dot vs bbox paint counts must differ (got dot={ovh_px}, bbox={det_px})"
    );
}

/// Invalid subtype must surface a clear error.
#[test]
fn invalid_subtype_rejected_at_load() {
    let dir = tempfile::tempdir().unwrap();
    let path = write_model_dir(
        dir.path(),
        "bogus",
        "yolo_e2e",
        r#"subtype = "not-a-real-subtype""#,
    );
    let err = load_manifest(&path).expect_err("invalid subtype must error");
    let msg = format!("{err}");
    assert!(
        msg.contains("subtype") && msg.contains("not-a-real-subtype"),
        "error must name the bad value: {msg}"
    );
}
