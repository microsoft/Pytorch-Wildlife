use anyhow::{bail, Context, Result};
use ndarray::{ArrayD, Ix4};
use serde_json::Value;
use sparrow_engine::cascade::{
    argmax, nchw_mel_to_nhwc_le_bytes, orca_audio_config, orca_mel_spectrogram, OrcaCascade,
};
use sparrow_engine::preprocess_audio::MelFilterbank;
use sparrow_engine::sys;
use sparrow_engine::tflite::LiteRtRuntime;
use std::env;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};

const DEFAULT_FIXTURES_DIR: &str =
    "/home/miao/repos/PW_refactor/sparrow-engine-dev/bench-binaries/artifacts/fixtures";
const DEFAULT_MODELS_DIR: &str =
    "/home/miao/repos/PW_refactor/sparrow-engine-dev/bench-binaries/artifacts";
const DETECTOR_MODEL_NAME: &str = "orca-detector-fp32.tflite";
const ECOTYPE_MODEL_NAME: &str = "orca-ecotype-melinput-fp32.tflite";

fn main() -> Result<()> {
    let paths = ExamplePaths::from_env();
    let fixtures = fixture_dirs(&paths.fixtures)?;
    let config = orca_audio_config();
    let filterbank = MelFilterbank::new(&config)?;

    let mut mel_max_diff = 0.0f32;
    let mut mel_mean_sum = 0.0f64;
    let mut mel_count = 0usize;
    let mut detector_ref_max_diff = 0.0f32;
    let mut detector_ref_mean_sum = 0.0f64;
    let mut detector_ref_count = 0usize;

    let runtime = LiteRtRuntime::new()?;
    let mut detector = runtime.load(&paths.detector, 0)?;
    let mut ecotype = runtime.load(&paths.ecotype, 0)?;

    let mut litert_max_diff = 0.0f32;
    let mut litert_mean_sum = 0.0f64;
    let mut litert_count = 0usize;
    let mut litert_argmax_matches = 0usize;

    let mut cascade = OrcaCascade::load(&paths.detector, &paths.ecotype, 0)?;
    let mut ungated_core_ecotype_argmax_matches = 0usize;
    let mut gated_cascade_argmax_matches = 0usize;
    let mut gated_cascade_count = 0usize;

    for fixture in &fixtures {
        let name = fixture
            .file_name()
            .and_then(|s| s.to_str())
            .context("fixture dir has no utf-8 name")?;
        let audio = flatten_f32(load_npy_f32(&fixture.join("ecotype_audio.npy"))?);
        let sample_rate =
            first_i64(load_npy_i64(&fixture.join("ecotype_sample_rate.npy"))?)? as u32;
        let detector_input =
            load_npy_f32(&fixture.join("detector_input.npy"))?.into_dimensionality::<Ix4>()?;
        let ecotype_mel =
            load_npy_f32(&fixture.join("ecotype_mel.npy"))?.into_dimensionality::<Ix4>()?;
        let expected = load_expected_logits(&fixture.join("expected_logits.json"))?;

        let core_mel = orca_mel_spectrogram(&audio, sample_rate, &config, &filterbank)?;
        let ecotype_stats = diff_stats(
            core_mel.as_slice().unwrap(),
            ecotype_mel.as_slice().unwrap(),
        );
        mel_max_diff = mel_max_diff.max(ecotype_stats.max_abs);
        mel_mean_sum += ecotype_stats.sum_abs;
        mel_count += ecotype_stats.count;

        let detector_stats = diff_stats(
            core_mel.as_slice().unwrap(),
            detector_input.as_slice().unwrap(),
        );
        detector_ref_max_diff = detector_ref_max_diff.max(detector_stats.max_abs);
        detector_ref_mean_sum += detector_stats.sum_abs;
        detector_ref_count += detector_stats.count;

        let det_outputs = detector.invoke_named(&[(
            "input",
            nchw_mel_to_nhwc_le_bytes(&detector_input)?,
            sys::LiteRtElementType::kLiteRtElementTypeFloat32,
        )])?;
        let det_logits = det_outputs.first().context("missing detector output")?;
        accumulate_diff(
            det_logits,
            &expected.detector,
            &mut litert_max_diff,
            &mut litert_mean_sum,
            &mut litert_count,
        )?;

        let eco_outputs = ecotype.invoke_named(&[(
            "mel",
            nchw_mel_to_nhwc_le_bytes(&ecotype_mel)?,
            sys::LiteRtElementType::kLiteRtElementTypeFloat32,
        )])?;
        let eco_logits = eco_outputs.first().context("missing ecotype output")?;
        accumulate_diff(
            eco_logits,
            &expected.ecotype,
            &mut litert_max_diff,
            &mut litert_mean_sum,
            &mut litert_count,
        )?;
        let expected_argmax = argmax(&expected.ecotype).context("expected ecotype logits empty")?;
        let litert_argmax = argmax(eco_logits).context("LiteRT ecotype logits empty")?;
        if litert_argmax == expected_argmax {
            litert_argmax_matches += 1;
        }

        let result = cascade.run_segment(&audio, sample_rate)?;
        let core_ecotype_outputs = ecotype.invoke_named(&[(
            "mel",
            nchw_mel_to_nhwc_le_bytes(&core_mel)?,
            sys::LiteRtElementType::kLiteRtElementTypeFloat32,
        )])?;
        let core_ecotype_logits = core_ecotype_outputs
            .first()
            .context("missing core-mel ecotype output")?;
        let core_ecotype_argmax =
            argmax(core_ecotype_logits).context("core-mel ecotype logits empty")?;
        if core_ecotype_argmax == expected_argmax {
            ungated_core_ecotype_argmax_matches += 1;
        }
        if result.ecotype_argmax == Some(expected_argmax) {
            gated_cascade_argmax_matches += 1;
        }
        if result.ecotype_argmax.is_some() {
            gated_cascade_count += 1;
        }
        println!(
            "SEGMENT {name}: detector_logit={:.6} detector_prob={:.6} is_orca={} gated_ecotype_argmax={:?} core_ecotype_argmax={core_ecotype_argmax} expected_argmax={expected_argmax}",
            result.detector_logit,
            result.detector_probability,
            result.is_orca,
            result.ecotype_argmax
        );
    }

    println!(
        "MEL_PARITY ecotype_mel max_abs={:.9} mean_abs={:.9}; detector_input max_abs={:.9} mean_abs={:.9}",
        mel_max_diff,
        mel_mean_sum / mel_count as f64,
        detector_ref_max_diff,
        detector_ref_mean_sum / detector_ref_count as f64,
    );
    println!(
        "LITERT_PARITY fp32_logits max_abs={:.9} mean_abs={:.9} ecotype_argmax={}/{}",
        litert_max_diff,
        litert_mean_sum / litert_count as f64,
        litert_argmax_matches,
        fixtures.len()
    );
    println!(
        "FULL_CASCADE core_ecotype_argmax={}/{} gated_ecotype_argmax={}/{} gated_segments={}/{}",
        ungated_core_ecotype_argmax_matches,
        fixtures.len(),
        gated_cascade_argmax_matches,
        gated_cascade_count,
        gated_cascade_count,
        fixtures.len()
    );
    Ok(())
}

struct ExamplePaths {
    fixtures: PathBuf,
    detector: PathBuf,
    ecotype: PathBuf,
}

impl ExamplePaths {
    fn from_env() -> Self {
        let fixtures = PathBuf::from(
            env::var("SPE_MOBILE_FIXTURES").unwrap_or_else(|_| DEFAULT_FIXTURES_DIR.into()),
        );
        let models_dir = PathBuf::from(
            env::var("SPE_MOBILE_MODELS").unwrap_or_else(|_| DEFAULT_MODELS_DIR.into()),
        );
        let detector = env::var("SPE_MOBILE_DETECTOR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| models_dir.join(DETECTOR_MODEL_NAME));
        let ecotype = env::var("SPE_MOBILE_ECOTYPE")
            .map(PathBuf::from)
            .unwrap_or_else(|_| models_dir.join(ECOTYPE_MODEL_NAME));
        Self {
            fixtures,
            detector,
            ecotype,
        }
    }
}

struct ExpectedLogits {
    detector: Vec<f32>,
    ecotype: Vec<f32>,
}

struct DiffStats {
    max_abs: f32,
    sum_abs: f64,
    count: usize,
}

fn fixture_dirs(root: &Path) -> Result<Vec<PathBuf>> {
    let mut dirs: Vec<PathBuf> = std::fs::read_dir(root)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| path.is_dir())
        .collect();
    dirs.sort();
    Ok(dirs)
}

fn load_npy_f32(path: &Path) -> Result<ArrayD<f32>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let npy = npyz::NpyFile::new(BufReader::new(file))
        .with_context(|| format!("parse npy {}", path.display()))?;
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let data: Vec<f32> = npy.into_vec::<f32>()?;
    Ok(ArrayD::from_shape_vec(shape, data)?)
}

fn load_npy_i64(path: &Path) -> Result<ArrayD<i64>> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    let npy = npyz::NpyFile::new(BufReader::new(file))
        .with_context(|| format!("parse npy {}", path.display()))?;
    let shape: Vec<usize> = npy.shape().iter().map(|&d| d as usize).collect();
    let data: Vec<i64> = npy.into_vec::<i64>()?;
    Ok(ArrayD::from_shape_vec(shape, data)?)
}

fn flatten_f32(array: ArrayD<f32>) -> Vec<f32> {
    array.iter().copied().collect()
}

fn first_i64(array: ArrayD<i64>) -> Result<i64> {
    array.iter().copied().next().context("empty i64 npy")
}

fn load_expected_logits(path: &Path) -> Result<ExpectedLogits> {
    let value: Value = serde_json::from_reader(File::open(path)?)?;
    Ok(ExpectedLogits {
        detector: logits(&value, "detector")?,
        ecotype: logits(&value, "ecotype")?,
    })
}

fn logits(value: &Value, key: &str) -> Result<Vec<f32>> {
    let arr = value
        .get(key)
        .and_then(|v| v.get("fp32"))
        .and_then(Value::as_array)
        .with_context(|| format!("expected_logits missing {key}.fp32"))?;
    arr.iter()
        .map(|v| {
            v.as_f64()
                .map(|x| x as f32)
                .with_context(|| format!("{key}.fp32 contains a non-number"))
        })
        .collect()
}

fn diff_stats(actual: &[f32], expected: &[f32]) -> DiffStats {
    assert_eq!(actual.len(), expected.len());
    let mut max_abs = 0.0f32;
    let mut sum_abs = 0.0f64;
    for (&a, &e) in actual.iter().zip(expected) {
        let d = (a - e).abs();
        max_abs = max_abs.max(d);
        sum_abs += d as f64;
    }
    DiffStats {
        max_abs,
        sum_abs,
        count: actual.len(),
    }
}

fn accumulate_diff(
    actual: &[f32],
    expected: &[f32],
    max_abs: &mut f32,
    sum_abs: &mut f64,
    count: &mut usize,
) -> Result<()> {
    if actual.len() != expected.len() {
        bail!(
            "logit length mismatch: actual {} expected {}",
            actual.len(),
            expected.len()
        );
    }
    let stats = diff_stats(actual, expected);
    *max_abs = (*max_abs).max(stats.max_abs);
    *sum_abs += stats.sum_abs;
    *count += stats.count;
    Ok(())
}
