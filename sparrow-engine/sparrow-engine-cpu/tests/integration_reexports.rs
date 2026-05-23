//
// Verifies the Phase 3.8 Phase A glob re-exports (S2 closure) keep every
// public path in `sparrow_engine::*` resolvable. If a downstream consumer (sparrow-engine-cli,
// sparrow-engine-python, sparrow-engine-server) imports `sparrow_engine::Detection` and a future refactor
// accidentally drops the re-export, this catches it at compile time.
//
// All assertions are `let _ = ...` style on values constructed with default /
// literal data — there is no behavioural verification here, just path /
// type-system reachability. ORT is NOT involved.

#![allow(unused_imports)]
#![allow(dead_code)]

// -----------------------------------------------------------------------------
// Test 1: type re-exports from sparrow-engine-types via `pub use sparrow_engine_types::*;`
// -----------------------------------------------------------------------------

#[test]
fn types_reexports_resolve_at_crate_root() {
    use std::path::PathBuf;

    // Bbox / detection / classification / pipeline result types
    let _bbox: sparrow_engine::BBox = sparrow_engine::BBox {
        x_min: 0.0,
        y_min: 0.0,
        x_max: 1.0,
        y_max: 1.0,
    };
    let _det: sparrow_engine::Detection = sparrow_engine::Detection {
        bbox: _bbox,
        label: "animal".to_string(),
        label_id: 1,
        confidence: 0.9,
    };
    let _detr: sparrow_engine::DetectResult = sparrow_engine::DetectResult {
        detections: vec![],
        image_width: 0,
        image_height: 0,
        processing_time_ms: 0.0,
    };
    let _cls: sparrow_engine::Classification = sparrow_engine::Classification {
        label: "cat".to_string(),
        label_id: 0,
        confidence: 0.7,
    };
    let _clr: sparrow_engine::ClassifyResult = sparrow_engine::ClassifyResult {
        classifications: vec![],
        image_width: 0,
        image_height: 0,
        processing_time_ms: 0.0,
    };
    let _pipe: sparrow_engine::PipelineResult = sparrow_engine::PipelineResult {
        pipeline_id: "p".to_string(),
        detections: vec![],
        image_width: 0,
        image_height: 0,
        processing_time_ms: 0.0,
    };

    // Enums
    let _mt: sparrow_engine::ModelType = sparrow_engine::ModelType::Detector;
    let _ms: sparrow_engine::ModelSubtype = sparrow_engine::ModelSubtype::Standard;
    let _pf: sparrow_engine::PixelFormat = sparrow_engine::PixelFormat::Rgb;

    // ImageInput + opts
    let _ii: sparrow_engine::ImageInput = sparrow_engine::ImageInput::Encoded(vec![]);
    let _do: sparrow_engine::DetectOpts = sparrow_engine::DetectOpts::default();
    let _co: sparrow_engine::ClassifyOpts = sparrow_engine::ClassifyOpts::default();
}

// -----------------------------------------------------------------------------
// Test 2: audio + manifest + engine-config re-exports
// -----------------------------------------------------------------------------

#[test]
fn audio_manifest_engine_config_reexports_resolve_at_crate_root() {
    use std::path::PathBuf;

    // Audio
    let _ar: sparrow_engine::AudioRange = sparrow_engine::AudioRange {
        start_time_s: 0.0,
        end_time_s: 1.0,
        max_confidence: 0.5,
        class: None,
    };
    let _ai: sparrow_engine::AudioInput = sparrow_engine::AudioInput::Samples {
        data: vec![0.0],
        sample_rate: 16000,
    };
    let _ac: sparrow_engine::AudioClass = sparrow_engine::AudioClass {
        class_idx: 0,
        label: None,
        probability: 0.0,
    };
    let _aseg: sparrow_engine::AudioSegment = sparrow_engine::AudioSegment {
        start_time_s: 0.0,
        end_time_s: 1.0,
        confidence: 0.5,
        classes: vec![_ac],
    };
    let _adr: sparrow_engine::AudioDetectResult = sparrow_engine::AudioDetectResult {
        segments: vec![],
        duration_s: 0.0,
        sample_rate: 16000,
        processing_time_ms: 0.0,
    };
    let _ado: sparrow_engine::AudioDetectOpts = sparrow_engine::AudioDetectOpts::default();

    // ModelInfo / EngineConfig / Device
    let _mi: sparrow_engine::ModelInfo = sparrow_engine::ModelInfo {
        id: "x".to_string(),
        path: PathBuf::from("/tmp"),
        model_type: sparrow_engine::ModelType::Detector,
        default: false,
        version: None,
        description: None,
        onnx_sha256: None,
        onnx_size_bytes: None,
    };
    let _dev: sparrow_engine::Device = sparrow_engine::Device::Auto;
    let _ec: sparrow_engine::EngineConfig = sparrow_engine::EngineConfig::new(_dev, "/tmp");

    // Preprocess POD types (lifted from preprocess.rs in Phase A).
    let _pp_meta: sparrow_engine::PreprocessMeta = sparrow_engine::PreprocessMeta {
        original_width: 100,
        original_height: 100,
        scale: 1.0,
        pad_x: 0.0,
        pad_y: 0.0,
    };
    let _pp_cfg: sparrow_engine::PreprocessConfig = sparrow_engine::PreprocessConfig {
        method: sparrow_engine::manifest::PreprocessMethod::Letterbox,
        input_size: [640, 640],
        layout: sparrow_engine::manifest::Layout::Nchw,
        normalization: sparrow_engine::manifest::Normalization::Unit,
        pad_value: 0.447,
        channel_order: sparrow_engine::manifest::ChannelOrder::Rgb,
    };

    // Manifest types reachable at the sparrow_engine:: root (re-exported from
    // sparrow-engine-types/src/lib.rs `pub use manifest::{ModelManifest, PipelineManifest};`).
    let _mm: Option<sparrow_engine::ModelManifest> = None;
    let _pm: Option<sparrow_engine::PipelineManifest> = None;

    // Error type
    let _err: sparrow_engine::SparrowEngineError = sparrow_engine::SparrowEngineError::EngineFreed;
}

// -----------------------------------------------------------------------------
// Test 3: free function re-exports — derive_model_type
// -----------------------------------------------------------------------------

#[test]
fn derive_model_type_reachable_at_crate_root() {
    // C2 closure: `derive_model_type` lives in sparrow-engine-types/src/model_type.rs,
    // re-exported via the glob `pub use sparrow_engine_types::*;` in sparrow-engine-cpu. Verify
    // the path `sparrow_engine::derive_model_type` resolves and produces the expected
    // ModelType for a known input.
    let mt = sparrow_engine::derive_model_type(
        &sparrow_engine::manifest::PreprocessMethod::Letterbox,
        &sparrow_engine::manifest::PostprocessMethod::YoloE2e,
        sparrow_engine::ModelSubtype::Overhead,
    );
    assert_eq!(mt, sparrow_engine::ModelType::OverheadDetector);
}

// -----------------------------------------------------------------------------
// Test 4: submodule path re-exports — sparrow_engine::manifest::load_manifest
// -----------------------------------------------------------------------------

#[test]
fn manifest_submodule_path_resolves() {
    use std::path::PathBuf;

    // The submodule path `sparrow_engine::manifest::*` is reachable because
    // `pub use sparrow_engine_types::*;` re-exports the `manifest` module item too,
    // not just the named type aliases. Verify by calling `load_manifest` on
    // a path that doesn't exist — we only care that the symbol resolves and
    // returns `ManifestNotFound`.
    let res = sparrow_engine::manifest::load_manifest(&PathBuf::from(
        "/nonexistent/manifest_path_unique_to_this_test.toml",
    ));
    match res {
        Err(sparrow_engine::SparrowEngineError::ManifestNotFound(_)) => {} // expected
        other => panic!("Expected ManifestNotFound, got: {other:?}"),
    }
}

// -----------------------------------------------------------------------------
// Test 5: sparrow-engine-core submodule paths — hash, daynight, viz, export, catalog,
// stats, postprocess, preprocess_audio
// -----------------------------------------------------------------------------

#[test]
fn bongo_core_submodule_paths_resolve() {
    use std::path::Path;

    // hash::hash_file — sha256 of empty byte sequence is known. Use any
    // existing file for reachability; the assertion is "function exists
    // and returns Result". We use Cargo.toml because every Rust crate has
    // one and tests run from the crate manifest dir.
    let cargo_toml = Path::new("Cargo.toml");
    let _ = sparrow_engine::hash::hash_file(cargo_toml); // compiles == reachable

    // daynight: pure function on encoded image bytes — pass garbage bytes
    // and let it error; we only test the path resolves.
    let _ = sparrow_engine::daynight::day_night(b"not-an-image");

    // viz::render — construct a 4x4 image and an empty annotation slice, run
    // the path, assert it returns SOMETHING (we only test reachability + no
    // panic on the empty case).
    let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 4));
    let opts = sparrow_engine::viz::RenderOpts::default();
    let _out = sparrow_engine::viz::render(&img, &[], &opts);

    // export::to_megadet — we only need to verify the symbol resolves;
    // call with an empty slice into a Vec writer.
    let mut out: Vec<u8> = Vec::new();
    let empty: &[(&Path, &sparrow_engine::DetectResult)] = &[];
    let _ = sparrow_engine::export::to_megadet(empty, "test_model", &mut out);

    // catalog::list_available_models on a bogus path — returns empty Vec.
    let models = sparrow_engine::catalog::list_available_models(Path::new(
        "/nonexistent/dir/for/this/test_only",
    ));
    assert!(
        models.is_empty(),
        "Bogus dir should yield empty model list, got {} entries",
        models.len()
    );

    // stats::summarize_detections on empty input
    let _summary = sparrow_engine::stats::summarize_detections(&[]);

    // postprocess::apply_max_detections on empty Vec — must not panic
    // (sort_desc_and_cap is pub(crate) per F4 audit; apply_max_detections is the
    //  pub equivalent for cross-crate consumers.)
    let mut empty_dets: Vec<sparrow_engine::Detection> = Vec::new();
    sparrow_engine::postprocess::apply_max_detections(&mut empty_dets, Some(10));
    assert!(empty_dets.is_empty());

    // preprocess_audio submodule reachable (pull in one type symbol)
    let _ap_cfg_default: Option<sparrow_engine::preprocess_audio::AudioPreprocessConfig> = None;
}
