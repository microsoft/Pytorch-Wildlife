//! RP-42 image-detection integration test for the mobile (LiteRT) flavor.
//!
//! Loads a `.tflite` `yolo_e2e` image detector (e.g. the converted
//! MegaDetector v6 `MDV6-yolov10-c-tflite`) through the generic
//! [`sparrow_engine::engine::Engine`] and runs single-shot detection over a
//! fixture image, asserting the output is structurally valid (normalized
//! bounding boxes, in-range scores, non-degenerate geometry).
//!
//! Env-gated (skips with a message when the model / fixture / LiteRT lib are
//! absent, e.g. in CI). Run on host:
//!
//! ```text
//! LITERT_LIB_DIR=<x86_64 ai_edge_litert dir> \
//! LD_LIBRARY_PATH=<same dir> \
//! SPE_MOBILE_IMAGE_MODEL_DIR=<catalog with MDV6-yolov10-c-tflite/> \
//! SPE_MOBILE_IMAGE_FIXTURE=<path to a jpg/png> \
//!   cargo test -p sparrow-engine-mobile --test image_detect -- --nocapture
//! ```
//!
//! `SPE_MOBILE_IMAGE_MODEL` (default `MDV6-yolov10-c-tflite`) and
//! `SPE_MOBILE_THREADS` (default `0` = LiteRT default; keep `0` on a host whose
//! stock LiteRT lacks the custom `Lrt*CpuOptions*` thread symbols) are optional.

use std::path::PathBuf;

use sparrow_engine::engine::Engine;
use sparrow_engine::{DetectOpts, Device, EngineConfig, ImageInput};

#[test]
fn mobile_image_detect_yolo_e2e_is_structurally_valid() {
    let model_dir = match std::env::var("SPE_MOBILE_IMAGE_MODEL_DIR") {
        Ok(d) => PathBuf::from(d),
        Err(_) => {
            eprintln!(
                "SKIP mobile_image_detect_yolo_e2e_is_structurally_valid: \
                 set SPE_MOBILE_IMAGE_MODEL_DIR (+ SPE_MOBILE_IMAGE_FIXTURE) to run."
            );
            return;
        }
    };
    let fixture = match std::env::var("SPE_MOBILE_IMAGE_FIXTURE") {
        Ok(f) => PathBuf::from(f),
        Err(_) => {
            eprintln!(
                "SKIP mobile_image_detect_yolo_e2e_is_structurally_valid: \
                 set SPE_MOBILE_IMAGE_FIXTURE to a jpg/png to run."
            );
            return;
        }
    };
    let model_id =
        std::env::var("SPE_MOBILE_IMAGE_MODEL").unwrap_or_else(|_| "MDV6-yolov10-c-tflite".into());
    let threads: u32 = std::env::var("SPE_MOBILE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);

    if !model_dir.join(&model_id).join("manifest.toml").exists() || !fixture.exists() {
        eprintln!(
            "SKIP mobile_image_detect_yolo_e2e_is_structurally_valid: missing model/fixture \
             (model_dir={}, model={model_id}, fixture={}).",
            model_dir.display(),
            fixture.display()
        );
        return;
    }

    let engine = Engine::new(EngineConfig {
        device: Device::Cpu,
        inter_threads: 0,
        intra_threads: threads,
        model_dir: model_dir.clone(),
    })
    .expect("create mobile engine");

    let model = engine
        .load_model_by_id(&model_id)
        .unwrap_or_else(|e| panic!("load model '{model_id}': {e:#}"));

    let result = model
        .detect(&ImageInput::FilePath(fixture.clone()), &DetectOpts::default())
        .unwrap_or_else(|e| panic!("detect on {}: {e:#}", fixture.display()));

    // The original image dimensions are reported back.
    assert!(
        result.image_width > 0 && result.image_height > 0,
        "image dims must be positive, got {}x{}",
        result.image_width,
        result.image_height
    );

    // A real camera-trap fixture produces at least one detection; if a caller
    // points at an empty scene this still must not panic — but we assert > 0
    // here because the default fixture is expected to contain objects.
    assert!(
        !result.detections.is_empty(),
        "expected at least one detection for fixture {}",
        fixture.display()
    );

    for d in &result.detections {
        assert!(
            (0.0..=1.0).contains(&d.confidence),
            "confidence out of [0,1]: {}",
            d.confidence
        );
        let b = &d.bbox;
        for (name, v) in [
            ("x_min", b.x_min),
            ("y_min", b.y_min),
            ("x_max", b.x_max),
            ("y_max", b.y_max),
        ] {
            assert!(
                (0.0..=1.0).contains(&v),
                "bbox {name} not normalized to [0,1]: {v}"
            );
        }
        assert!(
            b.x_min < b.x_max && b.y_min < b.y_max,
            "degenerate bbox: [{}, {}, {}, {}]",
            b.x_min,
            b.y_min,
            b.x_max,
            b.y_max
        );
        assert!(!d.label.is_empty(), "detection label must be non-empty");
    }

    eprintln!(
        "mobile detect OK: {} detection(s) on {}x{} image",
        result.detections.len(),
        result.image_width,
        result.image_height
    );
}
