//! Phase 3.8 Phase C Wave 1 integration tests for the
//! `sparrow_engine::Engine` dispatch glue.
//!
//! These tests exercise the engine + free-fn surface that
//! `sparrow-engine-server` / `sparrow-engine-cli` / `sparrow-engine-python` consume in Phase C
//! waves 2-5. They mirror the gating pattern used by
//! `tests/integration_yolo.rs` etc. — every test that needs a real GPU
//! plus a model fixture is `#[ignore]`d so a no-fixture clean checkout
//! still sees `cargo test -p sparrow-engine-gpu` PASS.
//!
//! Run them explicitly:
//!
//! ```bash
//! SPARROW_ENGINE_GPU_TEST_MODELS=/path/to/sparrow_engine_models \
//! SPARROW_ENGINE_GPU_TEST_CORPUS=/path/to/test_cameratrap \
//! SPARROW_ENGINE_GPU_TEST_AUDIO=/path/to/audio.wav \
//!   cargo test -p sparrow-engine-gpu --release --test integration_engine_dispatch -- --ignored
//! ```
//!
//! Default fixture paths mirror `integration_yolo.rs`:
//! - `SPARROW_ENGINE_GPU_TEST_MODELS` →
//!   `/home/miao/repos/PW_refactor/test_files/sparrow_engine_models`
//! - `SPARROW_ENGINE_GPU_TEST_CORPUS` →
//!   `/home/miao/repos/PW_refactor/test_files/test_cameratrap`
//! - `SPARROW_ENGINE_GPU_TEST_AUDIO` →
//!   `/home/miao/repos/PW_refactor/test_files/audio/DUNAS_20230925_090000.wav`

use std::path::{Path, PathBuf};

use sparrow_engine::{classify, detect, detect_audio, Engine};
use sparrow_engine_types::types::{
    AudioDetectOpts, AudioInput, ClassifyOpts, DetectOpts, ImageInput,
};
use sparrow_engine_types::{Device, EngineConfig};
use serial_test::serial;

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

fn model_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("SPARROW_ENGINE_GPU_TEST_MODELS")
            .unwrap_or_else(|_| "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models".into()),
    )
}

fn corpus_dir() -> PathBuf {
    PathBuf::from(
        std::env::var("SPARROW_ENGINE_GPU_TEST_CORPUS")
            .unwrap_or_else(|_| "/home/miao/repos/PW_refactor/test_files/test_cameratrap".into()),
    )
}

fn audio_path() -> PathBuf {
    PathBuf::from(std::env::var("SPARROW_ENGINE_GPU_TEST_AUDIO").unwrap_or_else(|_| {
        "/home/miao/repos/PW_refactor/test_files/test_audio/DUNAS_20230925_090000.wav".into()
    }))
}

fn first_image_in(dir: &Path) -> Option<PathBuf> {
    std::fs::read_dir(dir).ok()?.find_map(|e| {
        let p = e.ok()?.path();
        let ext = p.extension()?.to_str()?.to_ascii_lowercase();
        if matches!(ext.as_str(), "jpg" | "jpeg" | "png") {
            Some(p)
        } else {
            None
        }
    })
}

/// Skip helper: returns true when the per-test fixture is missing AND
/// the SPARROW_ENGINE_GPU_TEST_FORCE guardrail is unset. Mirrors the pattern in
/// integration_yolo.rs.
fn skip_if_missing(label: &str, condition: bool) -> bool {
    let force = std::env::var("SPARROW_ENGINE_GPU_TEST_FORCE").as_deref() == Ok("1");
    if condition {
        return false;
    }
    if force {
        panic!("{label}: required fixture missing, SPARROW_ENGINE_GPU_TEST_FORCE=1");
    } else {
        eprintln!("{label}: required fixture missing, skipping");
        true
    }
}

/// Build an `EngineConfig` rooted at the model fixture dir.
fn make_config() -> EngineConfig {
    EngineConfig::new(Device::Auto, model_dir())
}

/// One-shot setup output for ignored dispatch tests: a freshly-constructed
/// engine, the manifest path under test, and any optional inference inputs
/// the test asked for.
struct TestEnv {
    engine: Engine,
    manifest_path: PathBuf,
    image_path: Option<PathBuf>,
    audio_path: Option<PathBuf>,
}

/// Skip-aware setup helper for ignored dispatch tests. Returns `Some(env)`
/// only when every required fixture is present; otherwise `None` after
/// emitting a `skip` notice via [`skip_if_missing`]. With
/// `SPARROW_ENGINE_GPU_TEST_FORCE=1` the panicking branch of `skip_if_missing` still
/// applies.
fn try_setup_with(
    model_subdir: &str,
    want_image: bool,
    want_audio: bool,
) -> Option<TestEnv> {
    if skip_if_missing("model dir", model_dir().exists()) {
        return None;
    }
    if want_image && skip_if_missing("corpus dir", corpus_dir().exists()) {
        return None;
    }
    if want_audio && skip_if_missing("audio file", audio_path().exists()) {
        return None;
    }
    let manifest_path = model_dir().join(model_subdir).join("manifest.toml");
    if skip_if_missing(
        &format!("{model_subdir} manifest"),
        manifest_path.exists(),
    ) {
        return None;
    }
    let image = if want_image {
        match first_image_in(&corpus_dir()) {
            Some(p) => Some(p),
            None => {
                eprintln!("{model_subdir}: corpus dir empty, skipping");
                return None;
            }
        }
    } else {
        None
    };
    let audio = if want_audio { Some(audio_path()) } else { None };
    let engine = Engine::new(make_config()).expect("engine");
    Some(TestEnv {
        engine,
        manifest_path,
        image_path: image,
        audio_path: audio,
    })
}

// ---------------------------------------------------------------------------
// Engine construction (no GPU work — safe to run by default).
// ---------------------------------------------------------------------------

#[test]
#[serial]
fn engine_construct_singleton() {
    // Only assert if CUDA is reachable; otherwise the GPU engine cannot
    // even allocate a context.
    let probe = cudarc::driver::CudaContext::new(0);
    if probe.is_err() {
        eprintln!("engine_construct_singleton: no CUDA, skipping");
        return;
    }
    drop(probe);

    let engine = Engine::new(make_config()).expect("first engine");
    let res = Engine::new(make_config());
    assert!(
        matches!(res, Err(sparrow_engine_types::SparrowEngineError::EngineAlreadyExists)),
        "second engine must fail with EngineAlreadyExists"
    );
    drop(res);
    drop(engine);
}

// ---------------------------------------------------------------------------
// Dispatch tests — GPU + model fixtures required.
// ---------------------------------------------------------------------------

#[test]
#[serial]
#[ignore = "needs GPU + sparrow_engine_models fixture (SPARROW_ENGINE_GPU_TEST_MODELS)"]
fn load_unload_yolo_model() {
    let Some(env) = try_setup_with("megadetector-v6-yolov10e", false, false) else {
        return;
    };
    let handle = env.engine.load_model(&env.manifest_path).expect("load mdv6");
    assert_eq!(env.engine.loaded_models().len(), 1);
    env.engine.unload_model(&handle).expect("unload mdv6");
    assert_eq!(env.engine.loaded_models().len(), 0);
}

#[test]
#[serial]
#[ignore = "needs GPU + speciesnet fixture"]
fn load_unload_classifier_model() {
    let Some(env) = try_setup_with("speciesnet-crop", false, false) else {
        return;
    };
    let handle = env.engine.load_model(&env.manifest_path).expect("load speciesnet");
    assert_eq!(handle.model_type(), sparrow_engine_types::ModelType::Classifier);
    env.engine.unload_model(&handle).expect("unload speciesnet");
}

#[test]
#[serial]
#[ignore = "needs GPU + audiobirds fixture"]
fn load_unload_audio_model() {
    let Some(env) = try_setup_with("md-audiobirds-v1", false, false) else {
        return;
    };
    let handle = env.engine.load_model(&env.manifest_path).expect("load audio");
    assert_eq!(handle.model_type(), sparrow_engine_types::ModelType::AudioDetector);
    env.engine.unload_model(&handle).expect("unload audio");
}

#[test]
#[serial]
#[ignore = "needs GPU + sparrow_engine_models fixture"]
fn get_or_load_caches_handle() {
    let Some(env) = try_setup_with("megadetector-v6-yolov10e", false, false) else {
        return;
    };
    let h1 = env
        .engine
        .get_or_load_model("megadetector-v6-yolov10e")
        .expect("get_or_load 1");
    let h2 = env
        .engine
        .get_or_load_model("megadetector-v6-yolov10e")
        .expect("get_or_load 2");
    // Same handle = same Arc<LoadedModel>.
    assert!(
        std::sync::Arc::ptr_eq(h1.manifest(), h2.manifest()),
        "second get_or_load must return the cached handle"
    );
}

#[test]
#[serial]
#[ignore = "needs GPU + sparrow_engine_models + cameratrap corpus"]
fn dispatch_detect_yolo() {
    let Some(env) = try_setup_with("megadetector-v6-yolov10e", true, false) else {
        return;
    };
    let img = env.image_path.expect("image path required");
    let handle = env.engine.load_model(&env.manifest_path).expect("load mdv6");
    let opts = DetectOpts::default();
    let result = detect::detect(&handle, &ImageInput::FilePath(img), &opts).expect("detect");
    // The fixture is a non-empty cameratrap; expect at least 1 detection
    // at default thresholds. If 0, the dispatch is plumbed but the model
    // returned no above-threshold detections — caller should investigate
    // by lowering the threshold.
    eprintln!(
        "dispatch_detect_yolo: {} detections in {:.2} ms",
        result.detections.len(),
        result.processing_time_ms
    );
    assert!(result.image_width > 0 && result.image_height > 0);
    assert!(!result.detections.is_empty(), "YOLO dispatch must return at least one detection on the fixture image");
    for det in &result.detections {
        assert!(det.confidence.is_finite());
        assert!((0.0..=1.0).contains(&det.confidence));
        assert!((0.0..=1.0).contains(&det.bbox.x_min));
        assert!((0.0..=1.0).contains(&det.bbox.x_max));
        assert!((0.0..=1.0).contains(&det.bbox.y_min));
        assert!((0.0..=1.0).contains(&det.bbox.y_max));
        assert!(det.bbox.x_min <= det.bbox.x_max);
        assert!(det.bbox.y_min <= det.bbox.y_max);
    }
}

#[test]
#[serial]
#[ignore = "needs GPU + speciesnet + cameratrap corpus"]
fn dispatch_classify_speciesnet() {
    let Some(env) = try_setup_with("speciesnet-crop", true, false) else {
        return;
    };
    let img = env.image_path.expect("image path required");
    let handle = env.engine.load_model(&env.manifest_path).expect("load speciesnet");
    let opts = ClassifyOpts {
        top_k: Some(3),
    };
    let result =
        classify::classify(&handle, &ImageInput::FilePath(img), &opts).expect("classify");
    assert_eq!(result.classifications.len(), 3, "top_k=3 must yield exactly 3 classifications");
    for pair in result.classifications.windows(2) {
        assert!(pair[0].confidence >= pair[1].confidence, "classifications must be sorted by confidence");
    }
    for class in &result.classifications {
        assert!(class.confidence.is_finite());
        assert!((0.0..=1.0).contains(&class.confidence));
    }
}

#[test]
#[serial]
#[ignore = "needs GPU + audiobirds model + DUNAS audio fixture"]
fn dispatch_detect_audio() {
    let Some(env) = try_setup_with("md-audiobirds-v1", false, true) else {
        return;
    };
    let audio = env.audio_path.expect("audio path required");
    let handle = env.engine.load_model(&env.manifest_path).expect("load audio");
    let opts = AudioDetectOpts::default();
    let result = detect_audio::detect_audio(&handle, &AudioInput::FilePath(audio), &opts)
        .expect("detect_audio");
    eprintln!(
        "dispatch_detect_audio: {} segments over {:.1} s in {:.2} ms",
        result.segments.len(),
        result.duration_s,
        result.processing_time_ms
    );
    assert!(result.duration_s > 0.0);
    assert!(result.sample_rate > 0);
    assert!(!result.segments.is_empty(), "audio dispatch must return at least one segment for the fixture clip");
    for seg in &result.segments {
        assert!(seg.start_time_s.is_finite());
        assert!(seg.end_time_s.is_finite());
        assert!(seg.start_time_s <= seg.end_time_s);
        assert!(seg.confidence.is_finite());
        assert!((0.0..=1.0).contains(&seg.confidence));
    }
}

#[test]
#[serial]
#[ignore = "needs GPU + audiobirds model + DUNAS audio fixture"]
fn dispatch_detect_audio_streaming_callback_fires() {
    let Some(env) = try_setup_with("md-audiobirds-v1", false, true) else {
        return;
    };
    let audio = env.audio_path.expect("audio path required");
    let handle = env.engine.load_model(&env.manifest_path).expect("load audio");
    let opts = AudioDetectOpts::default();
    let mut count = 0usize;
    let result = detect_audio::detect_audio_streaming(
        &handle,
        &AudioInput::FilePath(audio),
        &opts,
        |_seg| {
            count += 1;
        },
    )
    .expect("detect_audio_streaming");
    // Per Phase C Wave 1 dispatch glue: the streaming callback fires
    // post-detect on the GPU side. count and result.segments.len() must
    // match (both reflect post-threshold detections).
    assert_eq!(
        count,
        result.segments.len(),
        "streaming callback count must equal returned segment count"
    );
}
