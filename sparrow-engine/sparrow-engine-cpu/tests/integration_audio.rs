//! Integration tests for audio detection (MD_AudioBirds_V1).
//!
//! Require ORT runtime and audio model — run with:
//! ```sh
//! ORT_LIB_LOCATION=/tmp/ort-lib ORT_PREFER_DYNAMIC_LINK=1 LD_LIBRARY_PATH=/tmp/ort-lib \
//!   cargo test -p sparrow-engine-cpu --test integration_audio -- --ignored --test-threads=1
//! ```

mod common;

use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{AudioDetectOpts, AudioInput, Engine};

const MODEL_NAME: &str = "audio_birds_v1";
const MANIFEST_FILE: &str = "audiobirds_manifest.toml";

// DUNAS field recordings for integration test
const TEST_WAV: &str = "DUNAS_20230925_090000.wav";

#[test]
#[ignore] // Requires ORT + audio model
fn test_audiobirds_detection() {
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 4,
        model_dir: common::onnx_dir(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");

    let manifest_path = common::onnx_dir().join(MANIFEST_FILE);
    let model = engine
        .load_model(&manifest_path)
        .expect("load audiobirds manifest");

    let audio_path = common::test_audio_dir().join(TEST_WAV);
    assert!(audio_path.exists(), "Test WAV not found: {:?}", audio_path);

    let input = AudioInput::FilePath(audio_path);
    let opts = AudioDetectOpts::default();

    let result = sparrow_engine::detect_audio::detect_audio(&model, &input, &opts)
        .unwrap_or_else(|e| panic!("detect_audio failed on {}: {}", TEST_WAV, e));

    println!(
        "AudioBirds: {} segments detected, duration={:.1}s, sr={}",
        result.segments.len(),
        result.duration_s,
        result.sample_rate,
    );

    // Save libsparrow_engine output for visualization comparison
    common::save_audio_json(
        &common::libsparrow_engine_output_dir(),
        MODEL_NAME,
        TEST_WAV,
        &result,
    );

    // Verify basic sanity: binary bird detector should find birds in wildlife audio.
    // DUNAS is a field recording site.
    //
    // Phase 3.8 Step 2 Wave 0a (F0.8 corrective regression, 2026-05-04): the
    // pre-Slaney pipeline (HTK mel scale + area normalization) produced an
    // artificially-saturated confidence distribution — most segments hit
    // confidence ≈ 1.0 because the wrong filterbank shape was over-amplifying
    // bird-band energy. The original >80% gate was tuned to that broken
    // distribution. Post-Slaney (matching MD_AudioBirds_V1 training) the
    // detection rate at the manifest threshold (0.9) drops to ~53% on the
    // DUNAS clip, which is consistent with a healthy binary detector seeing
    // real bird activity. Gate lowered to >30% (still strong signal, well
    // above noise floor). See `docs/research/phase3.8/step2/cpu_pre_fix_log.md`
    // for the per-clip pre-vs-post drift table.
    let total_possible = (result.duration_s / 0.3).floor() as usize; // approx segments at 0.3s stride
    let detection_rate = result.segments.len() as f64 / total_possible as f64;
    println!(
        "Detection rate: {:.1}% ({}/{} segments)",
        detection_rate * 100.0,
        result.segments.len(),
        total_possible
    );
    assert!(
        detection_rate > 0.30,
        "Expected >30% detection rate on wildlife audio (post-Slaney baseline), got {:.1}%",
        detection_rate * 100.0
    );
    assert!(
        result.segments.iter().all(|s| s.confidence >= 0.5),
        "All returned segments should be above threshold"
    );

    drop(model);
    drop(engine);
}

#[test]
#[ignore] // Requires ORT + audio model; sanity-checks exact-logit hardening.
fn audiobirds_default_model_emits_exactly_one_logit_per_segment() {
    let config = EngineConfig {
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 4,
        model_dir: common::onnx_dir(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");
    let manifest_path = common::onnx_dir().join(MANIFEST_FILE);
    let model = engine
        .load_model(&manifest_path)
        .expect("load audiobirds manifest");
    let audio_path = common::test_audio_dir().join(TEST_WAV);
    assert!(audio_path.exists(), "Test WAV not found: {:?}", audio_path);

    let result = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_path),
        &AudioDetectOpts::default(),
    )
    .expect("MD_AudioBirds_V1 must satisfy the exact one-logit-per-segment contract");

    assert!(
        !result.segments.is_empty(),
        "sanity fixture must produce at least one segment so exact-logit validation is exercised"
    );
    assert_eq!(
        result.sample_rate, 48_000,
        "MD_AudioBirds_V1 fixture should run at the manifest sample rate"
    );
}

/// Smoke test: verify golden audio files are loadable and have expected structure.
#[test]
fn audio_golden_loadable() {
    let golden_dir = common::golden_dir().join(MODEL_NAME);
    if !golden_dir.exists() {
        eprintln!(
            "SKIP: golden audio dir not found at {:?}. \
             Run tools/generate_audio_golden.py first.",
            golden_dir
        );
        return;
    }

    // Synthetic fixture golden
    let synth_path = golden_dir.join("synthetic_10s_audio.json");
    if synth_path.exists() {
        let golden: common::GoldenAudioResult =
            serde_json::from_str(&std::fs::read_to_string(&synth_path).unwrap())
                .expect("parse synthetic golden JSON");

        assert_eq!(golden.model, "md-audiobirds-v1");
        assert_eq!(golden.sample_rate, 48000);
        assert_eq!(
            golden.n_fft, 2048,
            "n_fft must be 2048 (matches Sparrow Studio's DefaultNfft and the saturation-fixing manifest update)"
        );
        assert_eq!(
            golden.time_steps_per_segment, 90,
            "time_steps must be 90 for n_fft=2048 at sr=48000, 1.0 s window, hop=512"
        );
        assert_eq!(
            golden.num_segments as usize,
            golden.segments.len(),
            "num_segments vs actual segment count"
        );
        assert_eq!(golden.preprocessing.filter_norm, "slaney");
        assert_eq!(golden.preprocessing.db_reference, "absolute (ref=1.0)");

        // Validate all segments have sane values
        for seg in &golden.segments {
            assert!(
                seg.start_s >= 0.0 && seg.start_s <= golden.duration_s,
                "Seg #{}: start_s={} out of range [0, {}]",
                seg.index,
                seg.start_s,
                golden.duration_s
            );
            assert!(
                seg.confidence >= 0.0 && seg.confidence <= 1.0,
                "Seg #{}: confidence={} out of [0,1]",
                seg.index,
                seg.confidence
            );
        }
    }
}

/// Smoke test: verify test audio directory and files exist.
#[test]
fn audio_test_files_exist() {
    let dir = common::test_audio_dir();
    assert!(dir.exists(), "Test audio dir not found: {:?}", dir);
    let wavs = common::audio_paths_from(&dir, 1);
    assert!(!wavs.is_empty(), "No WAV files in {:?}", dir);
}

/// Smoke test: verify audiobirds manifest exists and is parseable.
#[test]
fn audio_manifest_exists() {
    let path = common::onnx_dir().join(MANIFEST_FILE);
    assert!(path.exists(), "Manifest not found: {:?}", path);
    let content = std::fs::read_to_string(&path).expect("read manifest");
    assert!(content.contains("md-audiobirds-v1"));
    assert!(content.contains("n_fft = 2048"));
    assert!(content.contains("filter_norm = \"slaney\""));
    assert!(content.contains("method = \"sigmoid\""));
}
