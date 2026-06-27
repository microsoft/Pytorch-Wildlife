//! Integration tests for mel-spectrogram + softmax audio classifiers.
//!
//! This exercises the RP-39 CPU ORT path: shared mel preprocessing feeds a
//! multi-class ONNX classifier, then the audio path applies softmax + top-K.

mod common;

use std::path::PathBuf;

use serial_test::serial;
use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{AudioDetectOpts, AudioInput, Engine, ModelType, SparrowEngineError};

fn ort_runtime_configured() -> bool {
    std::env::var_os("ORT_LIB_LOCATION").is_some()
        || std::env::var_os("ORT_DYLIB_PATH").is_some()
        || std::env::var_os("ORT_CAPI").is_some()
}

fn mel_classifier_bundle_dir() -> Option<PathBuf> {
    if !ort_runtime_configured() {
        eprintln!("SKIP: ORT runtime env not configured; run through ./scripts/test.sh");
        return None;
    }
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../sparrow-engine-core/tests/fixtures/audio/mel_classifier_tiny");
    if p.join("manifest.toml").exists() && p.join("model.onnx").exists() {
        Some(p)
    } else {
        None
    }
}

fn core_audio_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sparrow-engine-core/tests/fixtures/audio")
}

#[test]
#[serial]
fn mel_softmax_manifest_loads_as_audio_classifier() {
    let Some(bundle_dir) = mel_classifier_bundle_dir() else {
        eprintln!("SKIP: mel_classifier_tiny fixture not found");
        return;
    };
    let manifest_path = bundle_dir.join("manifest.toml");
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 1,
        model_dir: bundle_dir.clone(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");
    let model = engine
        .load_model(&manifest_path)
        .expect("MelSpectrogram + Softmax manifest should load");

    assert_eq!(model.model_type(), ModelType::AudioClassifier);
    assert_eq!(model.labels().len(), 3);

    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn mel_softmax_detect_audio_emits_top3_class_segment_per_window() {
    let Some(bundle_dir) = mel_classifier_bundle_dir() else {
        eprintln!("SKIP: mel_classifier_tiny fixture not found");
        return;
    };
    let manifest_path = bundle_dir.join("manifest.toml");
    let audio_path = core_audio_fixtures_dir().join("short_2s.wav");
    assert!(
        audio_path.exists(),
        "expected audio fixture at {}",
        audio_path.display()
    );

    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 1,
        model_dir: bundle_dir.clone(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");
    let model = engine
        .load_model(&manifest_path)
        .expect("load mel classifier manifest");

    let result = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_path.clone()),
        &AudioDetectOpts::default(),
    )
    .unwrap_or_else(|e| panic!("detect_audio on {} failed: {}", audio_path.display(), e));

    assert!(!result.segments.is_empty(), "expected at least one segment");
    assert_eq!(result.sample_rate, 24_000);
    for (i, segment) in result.segments.iter().enumerate() {
        assert_eq!(
            segment.classes.len(),
            3,
            "segment {i}: expected top-K to include all 3 classes"
        );
        assert!(
            (segment.confidence - segment.classes[0].probability).abs() < f32::EPSILON,
            "segment {i}: confidence must equal top-1 probability"
        );
        let mut prev = f32::INFINITY;
        for (rank, class) in segment.classes.iter().enumerate() {
            assert!(
                (class.class_idx as usize) < 3,
                "segment {i} rank {rank}: class_idx {} out of range",
                class.class_idx
            );
            assert!(
                class.probability >= 0.0 && class.probability <= 1.0,
                "segment {i} rank {rank}: probability {} not in [0, 1]",
                class.probability
            );
            assert!(
                class.probability <= prev,
                "segment {i} rank {rank}: probability order is not descending"
            );
            prev = class.probability;
            assert!(
                matches!(
                    class.label.as_deref(),
                    Some("class_a" | "class_b" | "class_c")
                ),
                "segment {i} rank {rank}: unexpected label {:?}",
                class.label
            );
        }
    }

    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn mel_softmax_detect_audio_no_longer_returns_invalid_manifest_guard() {
    let Some(bundle_dir) = mel_classifier_bundle_dir() else {
        eprintln!("SKIP: mel_classifier_tiny fixture not found");
        return;
    };
    let manifest_path = bundle_dir.join("manifest.toml");
    let audio_path = core_audio_fixtures_dir().join("short_2s.wav");
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 1,
        model_dir: bundle_dir.clone(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");
    let model = engine
        .load_model(&manifest_path)
        .expect("load mel classifier manifest");

    let result = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_path.clone()),
        &AudioDetectOpts::default(),
    );
    if let Err(SparrowEngineError::InvalidManifest(msg)) = &result {
        panic!("old MelSpectrogram + Softmax reject guard is still active: {msg}");
    }
    result.unwrap_or_else(|e| panic!("detect_audio failed with non-guard error: {e}"));

    drop(model);
    drop(engine);
}
