//! Parity test: the generic manifest-driven cascade (`Engine::run_pipeline`)
//! must produce the same per-window result as the proven hardcoded
//! [`OrcaCascade`] reference on the orca cascade fixtures.
//!
//! Env-gated (skips with a message when the models / fixtures / LiteRT lib are
//! absent, e.g. in CI). Run on host:
//!
//! ```text
//! LITERT_LIB_DIR=<x86_64 ai_edge_litert dir> \
//! LD_LIBRARY_PATH=<same dir> \
//! SPE_MOBILE_MODEL_DIR=<model catalog with orca-cascade/pipeline.toml> \
//! SPE_MOBILE_FIXTURES=<fixtures dir> \
//!   cargo test -p sparrow-engine-mobile --test cascade_parity -- --nocapture
//! ```

use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

use sparrow_engine::cascade::OrcaCascade;
use sparrow_engine::engine::Engine;
use sparrow_engine::pipeline::CascadeOpts;
use sparrow_engine::{AudioInput, Device, EngineConfig};

const DEFAULT_MODEL_DIR: &str =
    "/home/miao/repos/PW_refactor/sparrow-engine-dev/.zenodo-staging/sparrow-engine-models-v0.6.0";
const DEFAULT_FIXTURES: &str =
    "/home/miao/repos/PW_refactor/sparrow-engine-dev/bench-binaries/artifacts/fixtures";
const DETECTOR_REL: &str = "orca-detector-fp16-tflite/orca-detector-fp16.tflite";
const ECOTYPE_REL: &str = "orca-ecotype-melinput-fp16-tflite/orca-ecotype-melinput-fp16.tflite";
const PIPELINE_ID: &str = "orca-cascade";

#[test]
fn generic_cascade_matches_orca_reference() {
    let model_dir = PathBuf::from(
        std::env::var("SPE_MOBILE_MODEL_DIR").unwrap_or_else(|_| DEFAULT_MODEL_DIR.into()),
    );
    let fixtures_root = PathBuf::from(
        std::env::var("SPE_MOBILE_FIXTURES").unwrap_or_else(|_| DEFAULT_FIXTURES.into()),
    );

    let pipeline_toml = model_dir.join(PIPELINE_ID).join("pipeline.toml");
    let detector_path = model_dir.join(DETECTOR_REL);
    let ecotype_path = model_dir.join(ECOTYPE_REL);

    if !pipeline_toml.exists()
        || !detector_path.exists()
        || !ecotype_path.exists()
        || !fixtures_root.exists()
    {
        eprintln!(
            "SKIP generic_cascade_matches_orca_reference: missing models/fixtures \
             (model_dir={}, fixtures={}). Set SPE_MOBILE_MODEL_DIR / SPE_MOBILE_FIXTURES.",
            model_dir.display(),
            fixtures_root.display()
        );
        return;
    }

    // Generic manifest-driven path.
    let engine = Engine::new(EngineConfig {
        device: Device::Cpu,
        inter_threads: 0,
        intra_threads: 0,
        model_dir: model_dir.clone(),
    })
    .expect("engine new");
    engine
        .load_pipeline_by_id(PIPELINE_ID)
        .expect("load orca-cascade pipeline");

    // Proven reference path (same fp16 models loaded directly).
    let mut reference =
        OrcaCascade::load(&detector_path, &ecotype_path, 0).expect("load OrcaCascade reference");

    let fixtures = fixture_dirs(&fixtures_root);
    assert!(
        !fixtures.is_empty(),
        "no fixtures under {}",
        fixtures_root.display()
    );

    let mut checked = 0usize;
    for fixture in &fixtures {
        let audio = load_npy_f32_flat(&fixture.join("ecotype_audio.npy"));
        let sample_rate = load_npy_i64_first(&fixture.join("ecotype_sample_rate.npy")) as u32;

        let generic = engine
            .run_pipeline(
                PIPELINE_ID,
                &AudioInput::Samples {
                    data: audio.clone(),
                    sample_rate,
                },
                &CascadeOpts::default(),
            )
            .expect("run_pipeline");
        let reference_seg = reference
            .run_segment(&audio, sample_rate)
            .expect("OrcaCascade run_segment");

        // A single-segment fixture (72 000 samples @ 24 kHz) → exactly one window.
        assert_eq!(
            generic.segments.len(),
            1,
            "{}: expected 1 window, got {}",
            fixture.display(),
            generic.segments.len()
        );
        let seg = &generic.segments[0];

        assert!(
            (seg.detector_logit - reference_seg.detector_logit).abs() < 1e-4,
            "{}: detector_logit {} vs ref {}",
            fixture.display(),
            seg.detector_logit,
            reference_seg.detector_logit
        );
        assert_eq!(
            seg.is_detected,
            reference_seg.is_orca,
            "{}: gating mismatch",
            fixture.display()
        );
        assert_eq!(
            seg.stage2_argmax,
            reference_seg.ecotype_argmax,
            "{}: ecotype argmax mismatch",
            fixture.display()
        );
        if let Some(ref_probs) = &reference_seg.ecotype_probabilities {
            assert_eq!(
                seg.stage2_probabilities.len(),
                ref_probs.len(),
                "{}: ecotype prob count",
                fixture.display()
            );
            for (a, b) in seg.stage2_probabilities.iter().zip(ref_probs) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "{}: ecotype prob {} vs ref {}",
                    fixture.display(),
                    a,
                    b
                );
            }
        } else {
            assert!(
                !seg.stage2_ran,
                "{}: reference skipped stage 2 but generic ran it",
                fixture.display()
            );
        }
        checked += 1;
    }
    eprintln!("generic_cascade_matches_orca_reference: {checked} fixtures matched");
}

fn fixture_dirs(root: &Path) -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)
        .expect("read fixtures dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    dirs
}

fn load_npy_f32_flat(path: &Path) -> Vec<f32> {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let npy = npyz::NpyFile::new(BufReader::new(file))
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    npy.into_vec::<f32>().expect("npy f32 vec")
}

fn load_npy_i64_first(path: &Path) -> i64 {
    let file = File::open(path).unwrap_or_else(|e| panic!("open {}: {e}", path.display()));
    let npy = npyz::NpyFile::new(BufReader::new(file))
        .unwrap_or_else(|e| panic!("parse {}: {e}", path.display()));
    npy.into_vec::<i64>()
        .expect("npy i64 vec")
        .into_iter()
        .next()
        .expect("empty i64 npy")
}
