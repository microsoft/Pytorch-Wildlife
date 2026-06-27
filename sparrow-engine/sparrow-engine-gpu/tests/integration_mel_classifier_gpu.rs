//! GPU mel-input softmax audio-classifier integration test (RP-39).
//!
//! Skips when the tiny shared fixture is absent in this branch, and skips
//! when a CUDA GPU cannot be opened. The fixture is produced by the CPU RP-39
//! branch and is exercised post-merge.

use std::path::PathBuf;

use cudarc::driver::CudaContext;
use sparrow_engine::Engine;
use sparrow_engine_types::types::{AudioDetectOpts, AudioInput};
use sparrow_engine_types::{Device, EngineConfig};

fn mel_classifier_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sparrow-engine-core/tests/fixtures/audio/mel_classifier_tiny")
}

fn core_audio_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sparrow-engine-core/tests/fixtures/audio")
}

fn cuda_available() -> bool {
    match CudaContext::new(0) {
        Ok(ctx) => {
            drop(ctx);
            true
        }
        Err(e) => {
            eprintln!("SKIP: CUDA GPU unavailable for mel classifier GPU test: {e}");
            false
        }
    }
}

#[test]
fn mel_classifier_gpu_emits_top3_segment_per_window_when_fixture_present() {
    let fixture_dir = mel_classifier_fixture_dir();
    if !fixture_dir.join("manifest.toml").exists() {
        eprintln!(
            "SKIP: mel_classifier_tiny fixture not found at {}. It is created by the CPU RP-39 branch and runs post-merge.",
            fixture_dir.display()
        );
        return;
    }
    if !cuda_available() {
        return;
    }

    let manifest_path = fixture_dir.join("manifest.toml");
    let audio_path = core_audio_fixtures_dir().join("short_2s.wav");
    assert!(
        audio_path.exists(),
        "expected audio fixture at {}",
        audio_path.display()
    );

    let config = EngineConfig {
        device: Device::Cuda(0),
        inter_threads: 1,
        intra_threads: 4,
        model_dir: fixture_dir.clone(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");
    let model = engine
        .load_model(&manifest_path)
        .expect("load mel softmax classifier manifest on GPU");
    assert_eq!(model.labels().len(), 3, "fixture should expose 3 labels");

    let result = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_path.clone()),
        &AudioDetectOpts::default(),
    )
    .unwrap_or_else(|e| panic!("detect_audio on {} failed: {e}", audio_path.display()));

    assert!(
        !result.segments.is_empty(),
        "mel softmax classifier should emit one segment per sliding window"
    );
    for (seg_idx, seg) in result.segments.iter().enumerate() {
        assert_eq!(
            seg.classes.len(),
            3,
            "segment {seg_idx}: expected top-3 classes for 3-class fixture"
        );
        assert!(
            (seg.confidence - seg.classes[0].probability).abs() < f32::EPSILON,
            "segment {seg_idx}: confidence must equal top-1 probability"
        );
        let mut prev = f32::INFINITY;
        for (rank, class) in seg.classes.iter().enumerate() {
            assert!(
                class.probability >= 0.0 && class.probability <= 1.0,
                "segment {seg_idx} rank {rank}: probability {} not in [0,1]",
                class.probability
            );
            assert!(
                class.probability <= prev,
                "segment {seg_idx} rank {rank}: probability {} > previous {}",
                class.probability,
                prev
            );
            prev = class.probability;
            assert!(
                (class.class_idx as usize) < 3,
                "segment {seg_idx} rank {rank}: class_idx {} out of range",
                class.class_idx
            );
            assert!(
                class.label.as_ref().is_some_and(|label| !label.is_empty()),
                "segment {seg_idx} rank {rank}: expected non-empty label"
            );
        }
    }

    drop(model);
    drop(engine);
}
