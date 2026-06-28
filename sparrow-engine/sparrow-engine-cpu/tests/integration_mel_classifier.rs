//! Integration tests for mel-spectrogram + softmax audio classifiers.
//!
//! This exercises the RP-39 CPU ORT path: shared mel preprocessing feeds a
//! multi-class ONNX classifier, then the audio path applies softmax + top-K.

use std::path::PathBuf;

use serial_test::serial;
use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{AudioDetectOpts, AudioInput, Engine, ModelHandle, ModelType, SparrowEngineError};

const EXPECTED_SEGMENT_RANGES: [(f32, f32); 2] = [(0.0, 1.0), (1.0, 2.0)];
const SOFTMAX_SUM_TOLERANCE: f32 = 1e-3;
const TIME_TOLERANCE: f32 = 1e-6;

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
    assert!(
        p.join("manifest.toml").exists(),
        "expected committed mel_classifier_tiny manifest at {}",
        p.join("manifest.toml").display()
    );
    assert!(
        p.join("model.onnx").exists(),
        "expected committed mel_classifier_tiny model at {}",
        p.join("model.onnx").display()
    );
    Some(p)
}

fn core_audio_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sparrow-engine-core/tests/fixtures/audio")
}

fn load_mel_classifier() -> Option<(Engine, ModelHandle)> {
    let bundle_dir = mel_classifier_bundle_dir()?;
    let manifest_path = bundle_dir.join("manifest.toml");
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 1,
        model_dir: bundle_dir,
    };
    let engine = Engine::new(config).expect("Engine::new failed");
    let model = engine
        .load_model(&manifest_path)
        .expect("load mel classifier manifest");
    Some((engine, model))
}

#[test]
#[serial]
fn mel_softmax_manifest_loads_as_audio_classifier() {
    let Some((engine, model)) = load_mel_classifier() else {
        return;
    };

    assert_eq!(model.model_type(), ModelType::AudioClassifier);
    assert_eq!(model.labels().len(), 3);

    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn mel_softmax_detect_audio_emits_top3_class_segment_per_window() {
    let Some((engine, model)) = load_mel_classifier() else {
        return;
    };
    let audio_path = core_audio_fixtures_dir().join("short_2s.wav");
    assert!(
        audio_path.exists(),
        "expected audio fixture at {}",
        audio_path.display()
    );

    let result = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_path.clone()),
        &AudioDetectOpts::default(),
    )
    .unwrap_or_else(|e| panic!("detect_audio on {} failed: {}", audio_path.display(), e));

    assert_eq!(
        result.segments.len(),
        EXPECTED_SEGMENT_RANGES.len(),
        "expected one classifier segment per 1s sliding window"
    );
    assert_eq!(result.sample_rate, 24_000);
    for (i, segment) in result.segments.iter().enumerate() {
        let (expected_start_s, expected_end_s) = EXPECTED_SEGMENT_RANGES[i];
        assert!(
            (segment.start_time_s - expected_start_s).abs() <= TIME_TOLERANCE,
            "segment {i}: expected start {expected_start_s}, got {}",
            segment.start_time_s
        );
        assert!(
            (segment.end_time_s - expected_end_s).abs() <= TIME_TOLERANCE,
            "segment {i}: expected end {expected_end_s}, got {}",
            segment.end_time_s
        );
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
        let mut probability_sum = 0.0f32;
        for (rank, class) in segment.classes.iter().enumerate() {
            assert!(
                (class.class_idx as usize) < 3,
                "segment {i} rank {rank}: class_idx {} out of range",
                class.class_idx
            );
            assert!(
                class.probability.is_finite(),
                "segment {i} rank {rank}: probability {} is not finite",
                class.probability
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
            probability_sum += class.probability;
            assert!(
                matches!(
                    class.label.as_deref(),
                    Some("class_a" | "class_b" | "class_c")
                ),
                "segment {i} rank {rank}: unexpected label {:?}",
                class.label
            );
        }
        assert!(
            (probability_sum - 1.0).abs() <= SOFTMAX_SUM_TOLERANCE,
            "segment {i}: top-3 probabilities should cover full softmax distribution, sum={probability_sum}"
        );
    }

    drop(model);
    drop(engine);
}

#[test]
#[serial]
fn mel_softmax_detect_audio_no_longer_returns_invalid_manifest_guard() {
    let Some((engine, model)) = load_mel_classifier() else {
        return;
    };
    let audio_path = core_audio_fixtures_dir().join("short_2s.wav");

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
