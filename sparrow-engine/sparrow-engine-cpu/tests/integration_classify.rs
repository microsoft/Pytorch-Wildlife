mod common;

use sparrow_engine::classify;
use sparrow_engine::engine::Device;
use sparrow_engine::{ClassifyOpts, Engine, EngineConfig, ImageInput};

#[test]
#[ignore]
fn test_speciesnet_classification() {
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 4,
        model_dir: common::onnx_dir(),
    };
    let engine = Engine::new(config).expect("Engine::new");

    let manifest_path = common::onnx_dir().join("speciesnet_manifest.toml");
    let model = engine.load_model(&manifest_path).expect("load SpeciesNet");

    let images = common::image_paths_from(&common::test_cameratrap_dir(), 10);
    let output_dir = common::libsparrow_engine_output_dir();
    let opts = ClassifyOpts { top_k: Some(5) };

    let mut failures = Vec::new();

    for img_path in &images {
        let image_name = img_path.file_name().unwrap().to_str().unwrap();
        let image_data = std::fs::read(img_path).expect("read image");
        let input = ImageInput::Encoded(image_data);

        let result = classify::classify(&model, &input, &opts).expect("classify failed");

        // Save libsparrow_engine output for visualization comparison
        common::save_classification_json(
            &output_dir,
            "speciesnet",
            image_name,
            result.image_width,
            result.image_height,
            &result.classifications,
        );

        // Compare against golden reference
        let golden = common::load_golden_classifications("speciesnet", image_name);
        if let Err(msg) = common::compare_classifications(
            &golden,
            &result.classifications,
            image_name,
            "speciesnet",
        ) {
            failures.push(msg);
        }
    }

    drop(model);
    drop(engine);

    if !failures.is_empty() {
        panic!(
            "SpeciesNet classification mismatches:\n{}",
            failures.join("\n")
        );
    }
}
