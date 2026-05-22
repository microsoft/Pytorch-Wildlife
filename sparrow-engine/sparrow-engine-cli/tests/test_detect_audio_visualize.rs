//! Phase 3.8 Step 2 — `spe detect-audio --visualize --output-dir` CLI
//! integration test.
//!
//! Spawns the actual `spe` binary against a real WAV fixture and
//! verifies that the `--visualize` flag produces non-empty PNGs in the
//! requested output directory. The visualization content is rendered by
//! `engine_dispatch::viz::render_audio_heatmap()` (Phase 3) + `render_range_overlay()`
//! and produces up to four layered PNGs:
//!
//! - `{stem}_01_spec.png` — raw mel spectrogram backdrop.
//! - `{stem}_02_segments.png` — spectrogram + per-window discrete confidence.
//! - `{stem}_03_heatmap.png` — spectrogram + smoothed confidence heatmap.
//! - `{stem}_04_full.png` — heatmap + cyan merged-range overlay.
//!
//! ## Why `#[ignore]`
//!
//! This test is gated behind `#[ignore]` for the same reasons as the
//! `audio_e2e_parity_*` tests:
//!
//! 1. It requires the `MD_AudioBirds_V1.onnx` model file (and the FP16
//!    sibling per the post-STRETCH FLIP) at the canonical path under
//!    `/home/miao/repos/PW_refactor/test_files/`. Not bundled.
//! 2. It requires the DUNAS WAV fixture. Not bundled.
//! 3. It requires a working CUDA GPU + ORT runtime stack initialised
//!    via `sparrow-engine/scripts/ort-env.sh`. The default device for
//!    `spe detect-audio` is `auto`; this test pins `--device gpu` so
//!    the failure mode (no GPU) surfaces as a fixture skip, not a CPU
//!    fall-through that takes 3 seconds per detect.
//! 4. It runs `spe detect-audio` end-to-end on a 60 s clip — too slow
//!    for the default `cargo test` smoke loop.
//!
//! The test asserts:
//!
//! - Exit code 0.
//! - At least one `.png` file produced in the output dir.
//! - Each `.png` file is non-empty (> 1 KiB; PNGs from the viz pipeline
//!   are several MB each at the default ~5600×224 resolution).
//! - Filename follows the `{stem}_NN_{layer}.png` Phase 3 convention.

use std::path::{Path, PathBuf};
use std::process::Command;

const FIXTURE: &str =
    "/home/miao/repos/PW_refactor/test_files/test_audio/DUNAS_20230925_090000.wav";
const MODEL_DIR: &str = "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models";
const ONNX_DIR: &str = "/home/miao/repos/PW_refactor/test_files/onnx";
// PNG min size sanity floor. The smallest layer (the discrete segments
// PNG) is typically ~2 MB at default resolution; 1 KiB is a safe
// non-empty-file gate that flags any zero-size or tiny error output.
const MIN_PNG_BYTES: u64 = 1024;

fn fixture_path() -> PathBuf {
    PathBuf::from(FIXTURE)
}

fn force_fixture() -> bool {
    std::env::var("SPARROW_ENGINE_AUDIO_FORCE_FIXTURE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn require_path(p: &Path, label: &str) -> Option<()> {
    if p.exists() {
        return Some(());
    }
    if force_fixture() {
        panic!(
            "missing {label} {} (SPARROW_ENGINE_AUDIO_FORCE_FIXTURE=1)",
            p.display()
        );
    }
    eprintln!(
        "skipping: missing {label} {} (set SPARROW_ENGINE_AUDIO_FORCE_FIXTURE=1 to make this fatal)",
        p.display()
    );
    None
}

#[test]
#[ignore]
fn detect_audio_visualize_produces_layered_pngs() {
    // Gate on fixtures so the test skips cleanly on a fresh box.
    let fx = fixture_path();
    if require_path(&fx, "DUNAS WAV fixture").is_none() {
        return;
    }
    let model_dir = PathBuf::from(MODEL_DIR);
    if require_path(&model_dir, "model dir").is_none() {
        return;
    }
    let onnx_dir = PathBuf::from(ONNX_DIR);
    if require_path(&onnx_dir, "onnx dir").is_none() {
        return;
    }

    // Use a tempdir so the test doesn't leak files between runs.
    let tmp = tempfile::tempdir().expect("tempdir");
    let out_dir = tmp.path();

    // Phase 3.8 Phase C W3: Cargo sets `CARGO_BIN_EXE_<bin-name>` only
    // for binaries actually built. Under `--features cpu` (default),
    // the `spe` bin is built; under `--features gpu`, the
    // `spe-gpu` bin is built. Each test crate compilation only sees
    // one of them, so the env! lookup is cfg-gated to match.
    #[cfg(feature = "cpu")]
    let spe_bin = env!("CARGO_BIN_EXE_spe");
    #[cfg(feature = "gpu")]
    let spe_bin = env!("CARGO_BIN_EXE_spe-gpu");

    let mut cmd = Command::new(spe_bin);
    cmd.arg("--device")
        .arg("gpu")
        .arg("--model-dir")
        .arg(&model_dir)
        .arg("detect-audio")
        .arg(&fx)
        .arg("--visualize")
        .arg("--output-dir")
        .arg(out_dir);

    eprintln!("running: {cmd:?}");
    let output = cmd.output().expect("failed to spawn spe binary");
    eprintln!(
        "stdout (tail):\n{}",
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n")
    );
    eprintln!(
        "stderr (tail):\n{}",
        String::from_utf8_lossy(&output.stderr)
            .lines()
            .rev()
            .take(20)
            .collect::<Vec<_>>()
            .into_iter()
            .rev()
            .collect::<Vec<_>>()
            .join("\n")
    );
    assert!(
        output.status.success(),
        "spe detect-audio --visualize exited non-zero ({}): see stderr above",
        output.status
    );

    // Enumerate produced PNGs.
    let mut pngs: Vec<PathBuf> = std::fs::read_dir(out_dir)
        .expect("read_dir output_dir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .and_then(|s| s.to_str())
                .map(|s| s.eq_ignore_ascii_case("png"))
                .unwrap_or(false)
        })
        .collect();
    pngs.sort();
    assert!(
        !pngs.is_empty(),
        "no PNGs produced in {} — viz pipeline silently skipped",
        out_dir.display()
    );

    // Validate each PNG: filename pattern, non-empty.
    let stem = fx.file_stem().unwrap().to_str().unwrap();
    for p in &pngs {
        let name = p.file_name().unwrap().to_str().unwrap();
        assert!(
            name.starts_with(stem),
            "PNG filename {name:?} does not start with fixture stem {stem:?}"
        );
        // Phase 3 convention: {stem}_NN_{layer}.png
        let suffix = name
            .strip_prefix(stem)
            .and_then(|s| s.strip_prefix('_'))
            .expect("PNG filename should follow {stem}_{layer}.png Phase 3 convention");
        // Exhaustive layer match — pinning the contract so a viz
        // refactor that drops or renames a layer breaks this test.
        assert!(
            matches!(
                suffix,
                "01_spec.png"
                    | "02_segments.png"
                    | "02_segments_windows.png"
                    | "03_heatmap.png"
                    | "04_full.png"
            ),
            "unexpected viz layer suffix {suffix:?} (PNG file {name})"
        );
        let size = p.metadata().expect("PNG metadata").len();
        assert!(
            size >= MIN_PNG_BYTES,
            "PNG {} is suspiciously small ({size} bytes < {MIN_PNG_BYTES} floor) — likely empty / error output",
            p.display()
        );
    }

    // Sanity: the four canonical layers should all be present (Phase 3
    // default-mode visualization without --raw-segments emits 01, 02,
    // 03, 04).
    let names: Vec<String> = pngs
        .iter()
        .map(|p| p.file_name().unwrap().to_str().unwrap().to_string())
        .collect();
    for required in [
        format!("{stem}_01_spec.png"),
        format!("{stem}_02_segments.png"),
        format!("{stem}_03_heatmap.png"),
        format!("{stem}_04_full.png"),
    ] {
        assert!(
            names.contains(&required),
            "expected layer PNG {required} not produced; got {names:?}"
        );
    }

    eprintln!(
        "OK — produced {} PNG(s) in {}: {:?}",
        pngs.len(),
        out_dir.display(),
        names
    );
}

/// Phase 3.8 Step 2 audit-fix R2 / R1-F4 (2026-05-05): regression
/// guard for the JSON byte-identity property documented in commit
/// `649a7b5` ("JSON output (merged ranges) is byte-identical with vs
/// without `--visualize`").
///
/// `cmd_detect_audio` (`sparrow-engine-cli/src/main.rs`) runs inference at
/// `threshold=0` when `--visualize` is set so the heatmap layer renders
/// the FULL per-window confidence distribution, then post-filters the
/// JSON / CSV / merged-range output back to production semantics
/// (`output_threshold = args.threshold OR manifest.confidence_threshold
/// OR 0.5`). The post-filter dance is only correct if the JSON output
/// is byte-identical to a non-viz baseline run — otherwise downstream
/// pipelines that compare CLI output across runs would silently drift
/// when `--visualize` is toggled. This test asserts that property.
#[test]
#[ignore]
fn detect_audio_json_byte_identical_with_and_without_visualize() {
    // Same fixture-skip dance as above.
    let fx = fixture_path();
    if require_path(&fx, "DUNAS WAV fixture").is_none() {
        return;
    }
    let model_dir = PathBuf::from(MODEL_DIR);
    if require_path(&model_dir, "model dir").is_none() {
        return;
    }
    let onnx_dir = PathBuf::from(ONNX_DIR);
    if require_path(&onnx_dir, "onnx dir").is_none() {
        return;
    }

    // Phase 3.8 Phase C W3: cfg-gated bin lookup mirrors line 105.
    #[cfg(feature = "cpu")]
    let spe_bin = env!("CARGO_BIN_EXE_spe");
    #[cfg(feature = "gpu")]
    let spe_bin = env!("CARGO_BIN_EXE_spe-gpu");

    // Run 1: WITHOUT --visualize (production semantics, threshold from
    // manifest default).
    let out_baseline = {
        let mut cmd = Command::new(spe_bin);
        cmd.arg("--device")
            .arg("gpu")
            .arg("--model-dir")
            .arg(&model_dir)
            .arg("detect-audio")
            .arg(&fx)
            .arg("--print")
            .arg("--format")
            .arg("json");
        eprintln!("baseline run: {cmd:?}");
        let out = cmd.output().expect("spawn baseline spe");
        assert!(
            out.status.success(),
            "baseline detect-audio exited non-zero ({}): stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    };

    // Run 2: WITH --visualize + --output-dir (threshold=0 inference,
    // post-filter dance reapplies production threshold to JSON output).
    let out_viz = {
        let tmp = tempfile::tempdir().expect("tempdir for viz run");
        let viz_out = tmp.path();
        let mut cmd = Command::new(spe_bin);
        cmd.arg("--device")
            .arg("gpu")
            .arg("--model-dir")
            .arg(&model_dir)
            .arg("detect-audio")
            .arg(&fx)
            .arg("--print")
            .arg("--format")
            .arg("json")
            .arg("--visualize")
            .arg("--output-dir")
            .arg(viz_out);
        eprintln!("visualize run: {cmd:?}");
        let out = cmd.output().expect("spawn visualize spe");
        assert!(
            out.status.success(),
            "visualize detect-audio exited non-zero ({}): stderr={}",
            out.status,
            String::from_utf8_lossy(&out.stderr)
        );
        out.stdout
    };

    if out_baseline != out_viz {
        // Surface the first diverging line for diagnostic clarity.
        let baseline_str = String::from_utf8_lossy(&out_baseline);
        let viz_str = String::from_utf8_lossy(&out_viz);
        let mut first_diff: Option<(usize, String, String)> = None;
        for (idx, (b_line, v_line)) in baseline_str.lines().zip(viz_str.lines()).enumerate() {
            if b_line != v_line {
                first_diff = Some((idx, b_line.to_string(), v_line.to_string()));
                break;
            }
        }
        let summary = first_diff
            .map(|(idx, b, v)| {
                format!("first divergence at line {idx}:\n  baseline: {b}\n  viz    : {v}")
            })
            .unwrap_or_else(|| "(diverges only in trailing length)".to_string());
        panic!(
            "JSON byte-identity broken: --visualize changed --print json output.\n  baseline len={}, viz len={}\n  {summary}",
            out_baseline.len(),
            out_viz.len()
        );
    }

    eprintln!(
        "OK — JSON byte-identity holds; {} bytes of stdout from both runs",
        out_baseline.len()
    );
}
