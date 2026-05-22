//! MT-17 regression: spe pipeline GPU teardown stress test.
//!
//! Exercises the detect→crop→classify adhoc pipeline path that historically
//! aborts with `corrupted double-linked list` (SIGABRT) on process exit
//! (~10-33% of runs). See `docs/bugs.md` MT-17 and
//! `docs/tech_report/06_gotchas_and_constraints.md`.
//!
//! One `cargo test` invocation covers intra-run pipeline stress. Combine
//! with `scripts/mt17_stress.sh` to exercise process-exit teardown across
//! N independent processes.
//!
//! Run:
//! ```sh
//! source scripts/ort-env.sh
//! cargo test --release -p sparrow-engine-cpu --test integration_pipeline_gpu_stress -- \
//!   --ignored --test-threads=1
//! ```
//!
//! Env overrides:
//! - `MT17_ITERATIONS` — inner pipeline iterations (default 20)
//! - `MT17_IMAGES` — images per iteration (default 5)
//! - `SPARROW_ENGINE_MODEL_DIR` — model dir (default
//!   `/home/miao/repos/PW_refactor/test_files/sparrow_engine_models_test/sparrow_engine_models`)

mod common;

use std::path::PathBuf;

use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{ClassifyOpts, DetectOpts, Engine, ImageInput};

const DETECTOR_ID: &str = "megadetector-v6-yolov10e";
const CLASSIFIER_ID: &str = "deepfaune-yolo8s";

fn pipeline_model_dir() -> PathBuf {
    std::env::var_os("SPARROW_ENGINE_MODEL_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            PathBuf::from("/home/miao/repos/PW_refactor/test_files/sparrow_engine_models_test/sparrow_engine_models")
        })
}

fn usize_env(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|v| v.parse::<usize>().ok())
        .unwrap_or(default)
}

/// Repeatedly exercise the detect→classify adhoc pipeline on a real image
/// set, then let the engine drop and the process exit. Historical MT-17
/// crashes surface at process teardown, not during inference, so the test
/// passes if (a) every pipeline call returns Ok, and (b) the test binary
/// exits with status 0.
#[test]
#[ignore] // Requires ORT + GPU + sparrow_engine_models_test manifests
fn test_pipeline_gpu_teardown_stress() {
    let iterations = usize_env("MT17_ITERATIONS", 20);
    let per_iter = usize_env("MT17_IMAGES", 5);

    let model_dir = pipeline_model_dir();
    assert!(
        model_dir.join(DETECTOR_ID).join("manifest.toml").exists(),
        "MT-17 regression requires manifest at {:?}/{}/manifest.toml",
        model_dir,
        DETECTOR_ID
    );
    assert!(
        model_dir.join(CLASSIFIER_ID).join("manifest.toml").exists(),
        "MT-17 regression requires manifest at {:?}/{}/manifest.toml",
        model_dir,
        CLASSIFIER_ID
    );

    let config = EngineConfig {
        device: Device::Cuda(0),
        inter_threads: 1,
        intra_threads: 4,
        model_dir,
    };
    let engine = Engine::new(config).expect("Engine::new failed");

    let images = common::image_paths_from(&common::test_cameratrap_dir(), per_iter);
    assert!(
        !images.is_empty(),
        "MT-17 regression requires images in {:?}",
        common::test_cameratrap_dir()
    );

    let detect_opts = DetectOpts::default();
    let classify_opts = ClassifyOpts::default();

    for iter in 0..iterations {
        for img_path in &images {
            let image_data = std::fs::read(img_path)
                .unwrap_or_else(|e| panic!("read {:?}: {}", img_path, e));
            let input = ImageInput::Encoded(image_data);

            let result = sparrow_engine::pipeline::run_pipeline_adhoc(
                &engine,
                &input,
                DETECTOR_ID,
                CLASSIFIER_ID,
                &detect_opts,
                &classify_opts,
            )
            .unwrap_or_else(|e| {
                panic!("pipeline iter {iter} on {:?}: {}", img_path, e)
            });

            assert_eq!(
                result.pipeline_id,
                format!("adhoc:{DETECTOR_ID}+{CLASSIFIER_ID}"),
                "unexpected adhoc pipeline id"
            );
        }
    }

    // Explicit drop so teardown-ordering bugs surface before the test binary
    // exit handlers would hide the cause.
    drop(engine);
}
