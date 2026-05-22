//! Integration tests for tiled detection models (HerdNet).
//!
//! Require ORT runtime and model files — run with:
//! ```sh
//! ORT_LIB_LOCATION=/tmp/ort-lib ORT_PREFER_DYNAMIC_LINK=1 LD_LIBRARY_PATH=/tmp/ort-lib \
//!   cargo test -p sparrow-engine-cpu --test integration_tiled -- --ignored --test-threads=1
//! ```

mod common;

use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{DetectOpts, Engine, ImageInput};

/// Single-image HerdNet tiled detection test.
///
/// Uses S_11_05_16_DSC01556.JPG (6000x4000) which has buffalo visible in the
/// upper-left waterhole area. The golden reference has 19 detections.
/// This image exercises edge tiles (6000 % 512 != 0, 4000 % 512 != 0).
#[test]
#[ignore] // Requires ORT + model files
fn test_herdnet_tiled_detection() {
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 4,
        model_dir: common::onnx_dir(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");

    let manifest_path = common::onnx_dir().join("herdnet_manifest.toml");
    let model = engine
        .load_model(&manifest_path)
        .expect("load HerdNet manifest");

    // Single test image with known detections in edge tile regions.
    let img_path = common::test_overhead_dir().join("S_11_05_16_DSC01556.JPG");
    assert!(img_path.exists(), "Test image not found: {:?}", img_path);

    let image_name = "S_11_05_16_DSC01556.JPG";
    let image_data = std::fs::read(&img_path).expect("read image file");
    let input = ImageInput::Encoded(image_data);
    let opts = DetectOpts::default();

    let result = sparrow_engine::detect::detect(&model, &input, &opts)
        .unwrap_or_else(|e| panic!("detect failed on {}: {}", image_name, e));

    // Save libsparrow_engine output for visual comparison.
    let output_dir = common::libsparrow_engine_output_dir();
    common::save_detection_json(
        &output_dir,
        "herdnet",
        image_name,
        result.image_width,
        result.image_height,
        &result.detections,
    );

    // Compare against golden reference (19 detections).
    let golden = common::load_golden_detections("herdnet", image_name);
    if let Err(msg) = common::compare_detections(&golden, &result.detections, image_name, "herdnet")
    {
        drop(model);
        drop(engine);
        panic!("HerdNet tiled detection mismatch:\n{}", msg);
    }

    println!(
        "HerdNet: {} detections (golden: {}), image {}x{}",
        result.detections.len(),
        golden.detections.len(),
        result.image_width,
        result.image_height,
    );

    drop(model);
    drop(engine);
}

/// OWL-T single-output heatmap tiled detection test.
///
/// Uses S_11_05_16_DSC01556.JPG (6000x4000, same overhead image as HerdNet).
/// OWL-T has a single heatmap output (no cls_map), tile_overlap=160, adaptive threshold.
/// This exercises the single-output codepath in detect_tiled.
#[test]
#[ignore] // Requires ORT + model files
fn test_owl_tiled_detection() {
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 4,
        model_dir: common::onnx_dir(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");

    let manifest_path = common::onnx_dir().join("owl_manifest.toml");
    let model = engine
        .load_model(&manifest_path)
        .expect("load OWL-T manifest");

    // Verify model loaded correctly.
    assert_eq!(model.model_id(), "owl-t");

    // Test image: same overhead scene used for HerdNet.
    let img_path = common::test_overhead_dir().join("S_11_05_16_DSC01556.JPG");
    assert!(img_path.exists(), "Test image not found: {:?}", img_path);

    let image_name = "S_11_05_16_DSC01556.JPG";
    let image_data = std::fs::read(&img_path).expect("read image file");
    let input = ImageInput::Encoded(image_data);
    let opts = DetectOpts::default();

    let result = sparrow_engine::detect::detect(&model, &input, &opts)
        .unwrap_or_else(|e| panic!("OWL-T detect failed on {}: {}", image_name, e));

    // Save output for visual inspection.
    let output_dir = common::libsparrow_engine_output_dir();
    common::save_detection_json(
        &output_dir,
        "owl-t",
        image_name,
        result.image_width,
        result.image_height,
        &result.detections,
    );

    // Basic sanity: image dimensions match.
    assert_eq!(result.image_width, 6000);
    assert_eq!(result.image_height, 4000);

    // OWL-T should produce detections on this overhead wildlife image.
    // A vacuously-passing test on 0 detections would hide model/manifest bugs.
    assert!(
        !result.detections.is_empty(),
        "OWL-T should detect animals in overhead image"
    );

    // Verify deduplication: no two detections should have identical bboxes.
    // Exact duplicates indicate that cross-tile dedup failed.
    for i in 0..result.detections.len() {
        for j in (i + 1)..result.detections.len() {
            assert!(
                result.detections[i].bbox != result.detections[j].bbox,
                "Duplicate detection at index {} and {}: bbox {:?}",
                i,
                j,
                result.detections[i].bbox
            );
        }
    }

    // All detections should have label_id = 0 (single-class model).
    for det in &result.detections {
        assert_eq!(
            det.label_id, 0,
            "OWL-T is single-class: all detections should have label_id=0, got {}",
            det.label_id
        );
    }

    // All bboxes should be normalized [0,1].
    for det in &result.detections {
        assert!(
            det.bbox.x_min >= 0.0 && det.bbox.x_min <= 1.0,
            "x_min out of range"
        );
        assert!(
            det.bbox.y_min >= 0.0 && det.bbox.y_min <= 1.0,
            "y_min out of range"
        );
        assert!(
            det.bbox.x_max >= 0.0 && det.bbox.x_max <= 1.0,
            "x_max out of range"
        );
        assert!(
            det.bbox.y_max >= 0.0 && det.bbox.y_max <= 1.0,
            "y_max out of range"
        );
        assert!(det.bbox.x_max >= det.bbox.x_min, "x_max < x_min");
        assert!(det.bbox.y_max >= det.bbox.y_min, "y_max < y_min");
    }

    // Confidences should be within (0, 1] — single-output confidence is raw heatmap value.
    for det in &result.detections {
        assert!(
            det.confidence > 0.0 && det.confidence <= 1.0,
            "Confidence out of (0,1]: {}",
            det.confidence
        );
    }

    // Detections should be sorted by confidence (descending).
    for w in result.detections.windows(2) {
        assert!(
            w[0].confidence >= w[1].confidence,
            "Detections not sorted: {} < {}",
            w[0].confidence,
            w[1].confidence
        );
    }

    println!(
        "OWL-T: {} detections, image {}x{}, processing_time={:.1}ms",
        result.detections.len(),
        result.image_width,
        result.image_height,
        result.processing_time_ms,
    );

    drop(model);
    drop(engine);
}
