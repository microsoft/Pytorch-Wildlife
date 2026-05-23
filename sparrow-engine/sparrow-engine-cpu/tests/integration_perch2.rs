//! Integration test for Perch 2 (Google's global bird-vocalization classifier).
//!
//! This is the first model that exercises `PreprocessMethod::RawAudio` + multi-class
//! `PostprocessMethod::Softmax` end-to-end. It validates that:
//!   * `manifest.toml` with `method = "raw_audio"` loads
//!   * `prepare_audio_detection` resolves the `label` output by name on a 4-head
//!     model (`embedding`, `spatial_embedding`, `spectrogram`, `label`)
//!   * `detect_audio_loop_raw` packs `(batch, 160000)` raw audio at 32 kHz and
//!     runs ORT inference correctly
//!   * Softmax + top-K = 5 produces a finite, sane probability distribution
//!   * Class labels from `labels.txt` (one Latin binomial per line, 14795 lines)
//!     are correctly mapped to the top-K indices
//!
//! Skipped unless the staged Perch 2 bundle is available. Resolve order:
//!   1. `$SPARROW_ENGINE_PERCH2_BUNDLE` env var (absolute path to bundle dir)
//!   2. `$SPARROW_ENGINE_DEV_ROOT/.zenodo-staging/perch-v2` (sparrow-engine-dev convention)
//!   3. Hardcoded fallback `/home/miao/repos/PW_refactor/sparrow-engine-dev/.zenodo-staging/perch-v2`
//!
//! Run with:
//! ```sh
//! ./scripts/test.sh -p sparrow-engine-cpu --test integration_perch2 -- --ignored --test-threads=1
//! ```

mod common;

use std::path::PathBuf;

#[cfg(feature = "ffi")]
use std::ffi::{CStr, CString};

use sparrow_engine::engine::{Device, EngineConfig};
use sparrow_engine::{AudioDetectOpts, AudioInput, Engine};

/// Resolve the staged Perch 2 bundle directory, returning `None` if not present.
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
#[ignore] // Requires ORT runtime + the 409 MB Perch 2 ONNX bundle (un-staged in CI).
fn perch2_detects_two_5s_windows_with_top5_classes_on_10s_clip() {
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
        device: Device::Cpu,
        inter_threads: 1,
        intra_threads: 4,
        // model_dir is only used by the legacy resolver; we pass an explicit
        // manifest path so this can be the bundle dir itself.
        model_dir: bundle_dir.clone(),
    };
    let engine = Engine::new(config).expect("Engine::new failed");
    let model = engine
        .load_model(&manifest_path)
        .expect("load Perch 2 manifest");

    // Sanity: 14795 species labels per MODEL_CARD.md
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

    // 10 s clip @ 5 s stride / 5 s window = exactly 2 non-overlapping windows.
    assert_eq!(
        result.segments.len(),
        2,
        "expected 2 windows for 10s @ 5s stride; got {}",
        result.segments.len()
    );
    assert_eq!(result.sample_rate, 32_000);
    assert!((result.duration_s - 10.0).abs() < 0.05);

    // Window time bounds:
    //   seg 0: [0, 5]
    //   seg 1: [5, 10]
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

        // Top-1 confidence is denormalized.
        assert!(
            (seg.confidence - seg.classes[0].probability).abs() < f32::EPSILON,
            "seg {}: confidence ({}) must equal classes[0].probability ({})",
            i,
            seg.confidence,
            seg.classes[0].probability
        );

        // Softmax invariants on the top-K:
        //   * each probability in [0, 1]
        //   * top-K sorted descending
        //   * top-K sum strictly between 0 and 1 (5 of 14795 → small but non-zero)
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
        "Perch 2: {} segments, duration {:.2}s, sr={} Hz, process={:.0} ms",
        result.segments.len(),
        result.duration_s,
        result.sample_rate,
        result.processing_time_ms,
    );
    eprintln!("Window 0 top-5:");
    for c in &result.segments[0].classes {
        eprintln!(
            "  idx={:>5}  p={:.4}  label={}",
            c.class_idx,
            c.probability,
            c.label.as_deref().unwrap_or("<none>")
        );
    }
    eprintln!("Window 1 top-5:");
    for c in &result.segments[1].classes {
        eprintln!(
            "  idx={:>5}  p={:.4}  label={}",
            c.class_idx,
            c.probability,
            c.label.as_deref().unwrap_or("<none>")
        );
    }

    drop(model);
    drop(engine);
}

#[cfg(feature = "ffi")]
unsafe fn ffi_last_error_string() -> String {
    let ptr = sparrow_engine::ffi::sparrow_engine_last_error();
    if ptr.is_null() {
        "<no last error>".to_string()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}

#[test]
#[ignore] // Requires ORT runtime + the 409 MB Perch 2 ONNX bundle (un-staged in CI).
#[cfg(feature = "ffi")]
fn perch2_detect_audio_v2_preserves_top5_classes_over_ffi() {
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

    let config_json = serde_json::json!({
        "device": "cpu",
        "inter_threads": 1,
        "intra_threads": 4,
        "model_dir": bundle_dir,
    })
    .to_string();
    let config_c = CString::new(config_json).unwrap();
    let manifest_c = CString::new(manifest_path.to_str().expect("manifest path UTF-8")).unwrap();
    let audio_c = CString::new(audio_path.to_str().expect("audio path UTF-8")).unwrap();

    unsafe {
        let engine = sparrow_engine::ffi::sparrow_engine_engine_new(config_c.as_ptr());
        assert!(
            !engine.is_null(),
            "Engine::new failed: {}",
            ffi_last_error_string()
        );

        let model = sparrow_engine::ffi::sparrow_engine_load_model(engine, manifest_c.as_ptr());
        assert!(
            !model.is_null(),
            "load model failed: {}",
            ffi_last_error_string()
        );

        let result = sparrow_engine::ffi::sparrow_engine_detect_audio_v2(
            model,
            audio_c.as_ptr(),
            std::ptr::null(),
        );
        assert!(
            !result.is_null(),
            "detect_audio_v2 failed: {}",
            ffi_last_error_string()
        );

        let header = &*result;
        assert_eq!(header.len, 2, "expected two 5s windows");
        assert!(!header.data.is_null());
        assert_eq!(header.sample_rate, 32_000);

        let segments = std::slice::from_raw_parts(header.data, header.len);
        for (i, segment) in segments.iter().enumerate() {
            assert_eq!(
                segment.classes_len, 5,
                "seg {i}: expected top-K = 5 classes, got {}",
                segment.classes_len
            );
            assert!(
                !segment.classes.is_null(),
                "seg {i}: classes pointer is null"
            );
            let classes = std::slice::from_raw_parts(segment.classes, segment.classes_len);
            assert!(
                (segment.confidence - classes[0].probability).abs() < f32::EPSILON,
                "seg {i}: confidence must equal top-1 probability"
            );
            assert!(classes[0].probability >= 0.0 && classes[0].probability <= 1.0);
            assert!(!classes[0].label.is_null(), "seg {i}: expected top-1 label");
            let label = CStr::from_ptr(classes[0].label)
                .to_str()
                .expect("label UTF-8");
            assert!(!label.is_empty(), "seg {i}: top-1 label is empty");
        }

        sparrow_engine::ffi::sparrow_engine_audio_result_v2_free(result);
        sparrow_engine::ffi::sparrow_engine_unload_model(model);
        sparrow_engine::ffi::sparrow_engine_engine_free(engine);
    }
}

#[test]
fn perch2_bundle_is_well_formed_if_present() {
    let Some(bundle_dir) = perch2_bundle_dir() else {
        return;
    };
    assert!(bundle_dir.join("manifest.toml").exists());
    assert!(bundle_dir.join("labels.txt").exists());
    assert!(bundle_dir.join("LICENSE.md").exists());
    assert!(bundle_dir.join("MODEL_CARD.md").exists());
    assert!(bundle_dir.join("1/model.onnx").exists());

    let labels = std::fs::read_to_string(bundle_dir.join("labels.txt")).expect("read labels.txt");
    assert!(
        !labels.contains('\r'),
        "labels.txt must be LF-only (CRLF detected); run sed -i 's/\\r$//'"
    );
    let n_lines = labels.lines().filter(|l| !l.is_empty()).count();
    assert_eq!(n_lines, 14_795, "expected 14795 non-empty label lines");

    let manifest =
        std::fs::read_to_string(bundle_dir.join("manifest.toml")).expect("read manifest");
    assert!(manifest.contains("method = \"raw_audio\""));
    assert!(manifest.contains("sample_rate = 32000"));
    assert!(manifest.contains("window_samples = 160000"));
    assert!(manifest.contains("format = \"one_per_line\""));
    assert!(manifest.contains("method = \"softmax\""));
}
