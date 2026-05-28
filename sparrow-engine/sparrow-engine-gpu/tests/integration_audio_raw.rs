//! GPU raw-audio integration test — Phase D round 2 B-08.
//!
//! Mirrors `sparrow-engine-cpu/tests/integration_perch2.rs` against the GPU
//! flavor's `RawAudioModel` path. Validates that:
//!   * `manifest.toml` with `method = "raw_audio"` loads on the GPU
//!     dispatcher (now routes to `RawAudioModel` rather than rejecting
//!     up-front).
//!   * `detect_audio` runs against the CUDA EP, producing per-window
//!     softmax + top-5 classes with sensible probabilities.
//!   * Label resolution from the engine's `inner.labels` works.
//!
//! Skipped unless the staged Perch 2 bundle is available AND a CUDA GPU
//! is present. Resolve order for the bundle dir mirrors the CPU test:
//!   1. `$SPARROW_ENGINE_PERCH2_BUNDLE`
//!   2. `$SPARROW_ENGINE_DEV_ROOT/.zenodo-staging/perch-v2`
//!   3. Hardcoded fallback `/home/miao/repos/PW_refactor/sparrow-engine-dev/.zenodo-staging/perch-v2`
//!
//! Run:
//! ```sh
//! ./scripts/test.sh -p sparrow-engine-gpu --test integration_audio_raw -- --ignored --test-threads=1
//! ```

use std::path::PathBuf;

use sparrow_engine::Engine;
use sparrow_engine_types::types::{AudioDetectOpts, AudioInput};
use sparrow_engine_types::{Device, EngineConfig};

fn perch2_bundle_dir() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("SPARROW_ENGINE_PERCH2_BUNDLE") {
        let p = PathBuf::from(p);
        if p.join("manifest.toml").exists() {
            return Some(p);
        }
    }
    if let Ok(root) = std::env::var("SPARROW_ENGINE_DEV_ROOT") {
        let p = PathBuf::from(root).join(".zenodo-staging").join("perch-v2");
        if p.join("manifest.toml").exists() {
            return Some(p);
        }
    }
    let fallback =
        PathBuf::from("/home/miao/repos/PW_refactor/sparrow-engine-dev/.zenodo-staging/perch-v2");
    if fallback.join("manifest.toml").exists() {
        return Some(fallback);
    }
    None
}

fn core_audio_fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sparrow-engine-core/tests/fixtures/audio")
}

#[test]
#[ignore] // Requires a CUDA GPU + the 409 MB Perch 2 ONNX bundle.
fn perch2_gpu_detects_two_5s_windows_with_top5_classes_on_10s_clip() {
    let Some(bundle_dir) = perch2_bundle_dir() else {
        eprintln!(
            "SKIP: Perch 2 bundle not found. Set SPARROW_ENGINE_PERCH2_BUNDLE or \
             SPARROW_ENGINE_DEV_ROOT, or stage the bundle at the documented path."
        );
        return;
    };
    let manifest_path = bundle_dir.join("manifest.toml");
    let audio_path = core_audio_fixtures_dir().join("medium_10s.wav");
    assert!(
        audio_path.exists(),
        "expected audio fixture at {}",
        audio_path.display()
    );

    let config = EngineConfig {
        device: Device::Cuda(0),
        inter_threads: 1,
        intra_threads: 4,
        model_dir: bundle_dir.clone(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");
    let model = engine
        .load_model(&manifest_path)
        .expect("load Perch 2 manifest (GPU)");

    assert_eq!(
        model.labels().len(),
        14_795,
        "Perch 2 manifest should resolve 14795 label lines"
    );

    let opts = AudioDetectOpts::default();
    let result = sparrow_engine::detect_audio::detect_audio(
        &model,
        &AudioInput::FilePath(audio_path.clone()),
        &opts,
    )
    .unwrap_or_else(|e| panic!("detect_audio on {} failed: {}", audio_path.display(), e));

    // 10 s clip @ default stride (manifest carries 5 s) / 5 s window = 2 non-overlapping windows.
    assert_eq!(
        result.segments.len(),
        2,
        "expected 2 windows for 10s @ 5s stride; got {}",
        result.segments.len()
    );
    assert_eq!(result.sample_rate, 32_000);
    assert!((result.duration_s - 10.0).abs() < 0.05);

    let s0 = &result.segments[0];
    let s1 = &result.segments[1];
    assert!((s0.start_time_s - 0.0).abs() < 0.01);
    assert!((s0.end_time_s - 5.0).abs() < 0.01);
    assert!((s1.start_time_s - 5.0).abs() < 0.01);
    assert!((s1.end_time_s - 10.0).abs() < 0.01);

    for (i, seg) in result.segments.iter().enumerate() {
        assert_eq!(
            seg.classes.len(),
            5,
            "seg {}: expected top-K = 5 classes, got {}",
            i,
            seg.classes.len()
        );
        assert!(
            (seg.confidence - seg.classes[0].probability).abs() < f32::EPSILON,
            "seg {}: confidence must equal top-1 probability",
            i,
        );
        let mut prev = f32::INFINITY;
        let mut topk_sum = 0.0f32;
        for (k, c) in seg.classes.iter().enumerate() {
            assert!(
                c.probability >= 0.0 && c.probability <= 1.0,
                "seg {} class {}: probability {} not in [0,1]",
                i,
                k,
                c.probability
            );
            assert!(
                c.probability <= prev,
                "seg {} class {}: probability {} > previous {} (not sorted desc)",
                i,
                k,
                c.probability,
                prev
            );
            prev = c.probability;
            topk_sum += c.probability;
            assert!(
                (c.class_idx as usize) < 14_795,
                "seg {} class {}: class_idx {} out of range",
                i,
                k,
                c.class_idx
            );
            let label = c
                .label
                .as_ref()
                .unwrap_or_else(|| panic!("seg {} class {}: expected label", i, k));
            assert!(!label.is_empty(), "seg {} class {}: label is empty", i, k);
        }
        assert!(
            topk_sum > 0.0 && topk_sum <= 1.0001,
            "seg {}: top-5 sum {} not in (0, 1]",
            i,
            topk_sum
        );
    }

    eprintln!(
        "Perch 2 (GPU): {} segments, duration {:.2}s, sr={} Hz, process={:.0} ms",
        result.segments.len(),
        result.duration_s,
        result.sample_rate,
        result.processing_time_ms,
    );

    drop(model);
    drop(engine);
}

/// Regression test for the up-front reject — was: "GPU raw_audio inference
/// is not yet implemented". With B-08 closed, the dispatcher MUST NOT
/// return InvalidManifest for a RawAudio model. The test is gated on the
/// bundle availability but does NOT need a GPU at load-time (Engine init
/// itself needs CUDA, hence #[ignore]).
#[test]
#[ignore]
fn perch2_gpu_load_no_longer_rejects_raw_audio() {
    let Some(bundle_dir) = perch2_bundle_dir() else {
        eprintln!("SKIP: Perch 2 bundle not found.");
        return;
    };
    let manifest_path = bundle_dir.join("manifest.toml");
    let config = EngineConfig {
        device: Device::Cuda(0),
        inter_threads: 1,
        intra_threads: 4,
        model_dir: bundle_dir.clone(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");
    let result = engine.load_model(&manifest_path);
    match result {
        Ok(_) => { /* expected */ }
        Err(e) => {
            let msg = format!("{e}");
            assert!(
                !msg.contains("not yet implemented"),
                "regressed B-08: GPU raw_audio still rejected with: {msg}"
            );
            panic!("load_model failed for a non-B-08 reason: {msg}");
        }
    }
}
