//! Integration tests for single-shot detection models (MDV6, deepfaune).
//!
//! Require ORT runtime and model files — run with:
//! ```sh
//! ORT_LIB_LOCATION=/tmp/ort-lib ORT_PREFER_DYNAMIC_LINK=1 LD_LIBRARY_PATH=/tmp/ort-lib \
//!   cargo test -p sparrow-engine-cpu --test integration_detect -- --ignored --test-threads=1
//! ```

mod common;

use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{DetectOpts, Engine, ImageInput};

#[test]
#[ignore] // Requires ORT + model files
fn test_mdv6_detection() {
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 4,
        model_dir: common::onnx_dir(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");

    let manifest_path = common::onnx_dir().join("mdv6_manifest.toml");
    let model = engine
        .load_model(&manifest_path)
        .expect("load MDV6 manifest");

    let images = common::image_paths_from(&common::test_cameratrap_dir(), 10);
    let output_dir = common::libsparrow_engine_output_dir();
    let opts = DetectOpts::default();

    let mut failures = Vec::new();

    for img_path in &images {
        let image_name = img_path.file_name().unwrap().to_str().unwrap();
        let image_data = std::fs::read(img_path).expect("read image file");
        let input = ImageInput::Encoded(image_data);

        let result = sparrow_engine::detect::detect(&model, &input, &opts)
            .unwrap_or_else(|e| panic!("detect failed on {}: {}", image_name, e));

        // Save libsparrow_engine output for visualization comparison
        common::save_detection_json(
            &output_dir,
            "mdv6",
            image_name,
            result.image_width,
            result.image_height,
            &result.detections,
        );

        // Compare against golden reference
        let golden = common::load_golden_detections("mdv6", image_name);
        if let Err(msg) =
            common::compare_detections(&golden, &result.detections, image_name, "mdv6")
        {
            failures.push(msg);
        }
    }

    // Drop model + engine before asserting (singleton cleanup)
    drop(model);
    drop(engine);

    if !failures.is_empty() {
        panic!(
            "MDV6 detection mismatches ({}/{}):\n{}",
            failures.len(),
            images.len(),
            failures.join("\n")
        );
    }
}

#[test]
#[ignore] // Requires ORT + model files
fn test_deepfaune_detection() {
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 4,
        model_dir: common::onnx_dir(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");

    let manifest_path = common::onnx_dir().join("deepfaune_manifest.toml");
    let model = engine
        .load_model(&manifest_path)
        .expect("load deepfaune manifest");

    let images = common::image_paths_from(&common::test_cameratrap_dir(), 10);
    let output_dir = common::libsparrow_engine_output_dir();
    let opts = DetectOpts::default();

    let mut failures = Vec::new();

    for img_path in &images {
        let image_name = img_path.file_name().unwrap().to_str().unwrap();
        let image_data = std::fs::read(img_path).expect("read image file");
        let input = ImageInput::Encoded(image_data);

        let result = sparrow_engine::detect::detect(&model, &input, &opts)
            .unwrap_or_else(|e| panic!("detect failed on {}: {}", image_name, e));

        common::save_detection_json(
            &output_dir,
            "deepfaune",
            image_name,
            result.image_width,
            result.image_height,
            &result.detections,
        );

        let golden = common::load_golden_detections("deepfaune", image_name);
        if let Err(msg) =
            common::compare_detections(&golden, &result.detections, image_name, "deepfaune")
        {
            failures.push(msg);
        }
    }

    drop(model);
    drop(engine);

    if !failures.is_empty() {
        panic!(
            "Deepfaune detection mismatches ({}/{}):\n{}",
            failures.len(),
            images.len(),
            failures.join("\n")
        );
    }
}
