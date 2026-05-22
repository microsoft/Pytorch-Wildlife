//! Phase 3.8 Step 2 Wave 3 — FP16 audio parity audit.
//!
//! Compares `sparrow_engine::models::audio::AudioModel` running the **FP16**
//! `MD_AudioBirds_V1_fp16.onnx` against the **FP32** `MD_AudioBirds_V1.onnx`
//! on the same DUNAS corpus. The reference is the FP32 GPU path (NOT
//! sparrow-engine-cpu) — this is an FP16-vs-FP32 audit, not a CPU-vs-GPU audit.
//! The CPU-vs-GPU FP32 audit lives in `audio_e2e_parity.rs`.
//!
//! ## Gates (per `arch-par_proposal_r2.md` § 2.2 R2 + lead brief 2026-05-05)
//!
//! | Gate | Threshold | Source |
//! | --- | --- | --- |
//! | Class-label flip count at threshold 0.9 | = 0 across all corpus segments | semantic gate |
//! | Per-segment confidence max-abs Δ vs FP32 | ≤ 0.01 | arch-par §2.2 |
//! | Per-segment confidence mean-abs Δ | ≤ 0.002 | arch-par §2.2 |
//! | Per-segment confidence relative Δ (where FP32 conf > 0.1) | ≤ 5 % | arch-par §2.2 |
//!
//! Latency-win is measured by `scripts/bench_audio_fp16_audit.py` (this
//! parity test is correctness-only).
//!
//! Per `feedback_no_soft_tolerance_framing_on_gates.md`: every gate's
//! exact integer/decimal Δ is reported with a `met` / `exceeded` verdict;
//! the test asserts at the gate value (no permissive multiplier);
//! exceedance triggers a STOP-and-ping panic.
//!
//! ## Why `#[ignore]`
//!
//! The two FP16 audit tests below are gated `#[ignore]` because they
//! depend on resources that aren't always present and aren't desired
//! in default `cargo test` runs:
//!
//! 1. **Both `MD_AudioBirds_V1.onnx` and `MD_AudioBirds_V1_fp16.onnx`**
//!    at `/home/miao/repos/PW_refactor/test_files/onnx/`. Not bundled
//!    with sparrow_engine; FP16 ONNX is generated via
//!    `sparrow-engine/tools/convert_fp16.py`.
//! 2. **The DUNAS WAV fixtures** at
//!    `/home/miao/repos/PW_refactor/test_files/test_audio/`. Not
//!    bundled.
//! 3. **A working CUDA GPU** with the cudarc + ORT runtime stack
//!    initialized via `sparrow-engine/scripts/ort-env.sh`. The test compares
//!    GPU FP32 vs GPU FP16 — both sides require a GPU.
//! 4. **Two ORT session loads per test** — each test loads the FP32
//!    ONNX, runs `compute_mel_per_segment` + `run_ort_logits_on_host_mel`,
//!    then loads the FP16 ONNX and repeats. Not appropriate for the
//!    default `cargo test` smoke loop.
//!
//! Setting `SPARROW_ENGINE_AUDIO_FORCE_FIXTURE=1` makes a missing fixture or
//! manifest fatal (instead of a skip).
//!
//! ## Running the tests
//!
//! Easiest: use the one-command runner
//!
//! ```bash
//! ./scripts/run_audio_parity.sh    # from repo root (sparrow-engine-dev/)
//! ```
//!
//! Manual: from the sparrow-engine crate workspace:
//!
//! ```bash
//! cd sparrow-engine
//! source scripts/ort-env.sh
//! cargo test --release -p sparrow-engine-gpu --test audio_e2e_parity_fp16 \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ## What each test asserts
//!
//! | Test | FP32 manifest | FP16 manifest | Fixture |
//! | --- | --- | --- | --- |
//! | `audio_fp16_audit_dunas_20230925` | `manifest_fp32.toml` | `manifest_fp16.toml` | DUNAS_20230925_090000.wav |
//! | `audio_fp16_audit_dunas_20230314` | `manifest_fp32.toml` | `manifest_fp16.toml` | DUNAS_20230314_090000.wav |
//!
//! Both tests assert the same four §2.2 R2 gates (max-abs Δ ≤ 0.01,
//! mean-abs Δ ≤ 0.002, relative Δ ≤ 5 %, label flips = 0 @ 0.9), report
//! exact measured values, and panic with the gate name + Δ on any
//! exceedance. Override the FP32/FP16 manifest paths via the
//! `SPARROW_ENGINE_AUDIO_FP32_MANIFEST` / `SPARROW_ENGINE_AUDIO_FP16_MANIFEST` env vars.

use std::path::{Path, PathBuf};

// Keep the GPU flavor import live; the earlier E0433 at this line was the
// shared-target dual-cdylib collision, not a missing sparrow-engine-gpu module.
use sparrow_engine::models::audio::{AudioModel, GpuAudioDetectOpts, Strategy};
use sparrow_engine_types::{AudioDetectOpts, AudioInput, AudioSegment};
use cudarc::driver::CudaContext;

// ---------------------------------------------------------------------------
// Wave 3 §2.2 FP16 audit gates
// ---------------------------------------------------------------------------

const CONF_MAX_ABS_GATE: f32 = 0.01;
const CONF_MEAN_ABS_GATE: f32 = 0.002;
const CONF_REL_GATE: f32 = 0.05; // 5% relative, only where FP32 conf > 0.1
const CONF_REL_FLOOR: f32 = 0.1; // segments with FP32 conf below this skip rel gate
const FLIP_THRESHOLD: f32 = 0.9;

// ---------------------------------------------------------------------------
// Fixture discovery (same DUNAS corpus as `audio_e2e_parity.rs`)
// ---------------------------------------------------------------------------

const FIXTURE_20230925: &str =
    "/home/miao/repos/PW_refactor/test_files/test_audio/DUNAS_20230925_090000.wav";
const FIXTURE_20230314: &str =
    "/home/miao/repos/PW_refactor/test_files/test_audio/DUNAS_20230314_090000.wav";
// Post-STRETCH re-audit (2026-05-05): the production `manifest.toml`
// flipped to `precision = "fp16"` after the FP16-vs-FP32 latency gate
// inverted. This test loads the explicit `manifest_fp32.toml` (FP32 reference,
// added during the flip) and `manifest_fp16.toml` (audit sibling, identical
// content to current `manifest.toml`) — keeping the test self-contained
// against future precision-default flips.
const FP32_MANIFEST: &str =
    "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models/md-audiobirds-v1/manifest_fp32.toml";
const FP16_MANIFEST: &str =
    "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models/md-audiobirds-v1/manifest_fp16.toml";

fn fp32_manifest_path() -> PathBuf {
    PathBuf::from(
        std::env::var("SPARROW_ENGINE_AUDIO_FP32_MANIFEST").unwrap_or_else(|_| FP32_MANIFEST.into()),
    )
}

fn fp16_manifest_path() -> PathBuf {
    PathBuf::from(
        std::env::var("SPARROW_ENGINE_AUDIO_FP16_MANIFEST").unwrap_or_else(|_| FP16_MANIFEST.into()),
    )
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

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn assert_detect_segment_parity(strategy: Strategy, fp32: &[AudioSegment], fp16: &[AudioSegment]) {
    assert_eq!(
        fp32.len(),
        fp16.len(),
        "detect({}) FP32/FP16 segment count mismatch",
        strategy.short_label()
    );
    for (idx, (a, b)) in fp32.iter().zip(fp16.iter()).enumerate() {
        assert!(
            (a.start_time_s - b.start_time_s).abs() <= 1e-6,
            "detect({}) segment {idx} start mismatch fp32={} fp16={}",
            strategy.short_label(),
            a.start_time_s,
            b.start_time_s
        );
        assert!(
            (a.end_time_s - b.end_time_s).abs() <= 1e-6,
            "detect({}) segment {idx} end mismatch fp32={} fp16={}",
            strategy.short_label(),
            a.end_time_s,
            b.end_time_s
        );
        assert!(
            (a.confidence - b.confidence).abs() <= CONF_MAX_ABS_GATE,
            "detect({}) segment {idx} confidence Δ={} exceeds gate {} (fp32={} fp16={})",
            strategy.short_label(),
            (a.confidence - b.confidence).abs(),
            CONF_MAX_ABS_GATE,
            a.confidence,
            b.confidence
        );
    }
}

/// Phase 3.8 Step 2 audit-fix R2 / R1-F3 (2026-05-05): exercise
/// `detect()` with the given strategy on both precisions and report
/// segment-list length parity. The per-segment-logit gates exercised by
/// `collect_per_segment_logits` are strategy-independent (the underlying
/// ORT IoBinding output is deterministic regardless of how the input mel
/// buffer is sub-batched per chunk size — see
/// `docs/research/phase3.8/step2/perf_triage_report.md` § "Parity
/// (G-D / W1.7-anchored)"), so this lightweight detect() exercise
/// closes the production-path coverage gap by ensuring the strategy
/// runs end-to-end on FP16 and FP32 without crashing or returning
/// length-mismatched segment lists.
fn report_detect_segment_count_parity(
    fp32_model: &AudioModel,
    fp16_model: &AudioModel,
    fixture: &Path,
    strategy: Strategy,
) {
    let opts = GpuAudioDetectOpts {
        base: AudioDetectOpts::default(),
        strategy,
    };
    let fp32 = fp32_model
        .detect(&AudioInput::FilePath(fixture.to_path_buf()), &opts)
        .expect("FP32 detect");
    let fp16 = fp16_model
        .detect(&AudioInput::FilePath(fixture.to_path_buf()), &opts)
        .expect("FP16 detect");
    eprintln!(
        "detect({}) segment counts: fp32={}, fp16={} (threshold={:?})",
        strategy.short_label(),
        fp32.segments.len(),
        fp16.segments.len(),
        opts.base.confidence_threshold,
    );
    assert_detect_segment_parity(strategy, &fp32.segments, &fp16.segments);
}

/// Run AudioModel on the fixture; collect raw per-segment logits for ALL
/// segments (not just above-threshold). Returns `Vec<f32>` of logits +
/// `n_segments_total`.
///
/// Approach: build the row-major per-segment mel buffer via
/// `compute_mel_per_segment`, then run ORT IoBinding on it via
/// `run_ort_logits_on_host_mel`. This avoids relying on `detect()`'s
/// above-threshold filtering and gives the FP16-vs-FP32 audit access to
/// every segment's logit (mid-range segments dominate the worst-case Δ
/// per the W1.7 amp curve).
fn collect_per_segment_logits(model: &AudioModel, fixture: &Path) -> (Vec<f32>, usize) {
    let opts = AudioDetectOpts::default();
    let snap = model
        .compute_mel_per_segment(&AudioInput::FilePath(fixture.to_path_buf()), &opts)
        .expect("compute_mel_per_segment");
    let n_segments = snap.segments.len();
    if n_segments == 0 {
        return (Vec::new(), 0);
    }
    let mel_concat: Vec<f32> = snap.segments.iter().flatten().copied().collect();
    let logits = model
        .run_ort_logits_on_host_mel(&mel_concat, n_segments)
        .expect("run_ort_logits_on_host_mel");
    assert_eq!(logits.len(), n_segments);
    (logits, n_segments)
}

/// Verify FP16 model loads + runs detect (for sanity).
#[allow(dead_code)]
fn smoke_detect(model: &AudioModel, fixture: &Path) {
    let opts = GpuAudioDetectOpts {
        base: AudioDetectOpts::default(),
        strategy: Strategy::HybridA {
            ort_chunk_segments: 16,
        },
    };
    let _ = model
        .detect(&AudioInput::FilePath(fixture.to_path_buf()), &opts)
        .expect("FP16 detect");
}

/// Per-fixture FP16 audit — reports all four §2.2 gates.
///
/// Defaults to `Strategy::HybridA{16}` for the production-path
/// segment-count parity exercise. Use [`run_fp16_audit_for_fixture_with_strategy`]
/// to specify a different strategy (e.g., `Strategy::SingleCall` —
/// production default for non-streaming detect, exercised by the
/// `audio_fp16_audit_strategy_single_*` tests added in audit-fix R2).
fn run_fp16_audit_for_fixture(fx: &Path, fp32_manifest: &Path, fp16_manifest: &Path, label: &str) {
    run_fp16_audit_for_fixture_with_strategy(
        fx,
        fp32_manifest,
        fp16_manifest,
        label,
        Strategy::HybridA {
            ort_chunk_segments: 16,
        },
    );
}

/// Strategy-parameterized variant — covers both the perf-default
/// `Strategy::SingleCall` (post Wave-4 perf triage) and the Wave 2
/// `HybridA{16}` reference path. The four §2.2 R2 gates use
/// `compute_mel_per_segment` + `run_ort_logits_on_host_mel` and are
/// strategy-independent (the underlying ORT IoBinding output is
/// deterministic regardless of chunk size — see `audio_e2e_parity.rs`
/// `run_parity_for_fixture_with_strategy` doc); the strategy
/// parameterization adds a `detect()` exercise that confirms the chosen
/// strategy runs end-to-end on FP16 + FP32. Phase 3.8 Step 2 audit-fix
/// R2 (R1-F3, 2026-05-05).
fn run_fp16_audit_for_fixture_with_strategy(
    fx: &Path,
    fp32_manifest: &Path,
    fp16_manifest: &Path,
    label: &str,
    strategy: Strategy,
) {
    eprintln!(
        "\n=== Phase 3.8 Step 2 Wave 3 — FP16 audit ({label}, {}) ===",
        strategy.short_label()
    );
    eprintln!("fixture       : {}", fx.display());
    eprintln!("fp32 manifest : {}", fp32_manifest.display());
    eprintln!("fp16 manifest : {}", fp16_manifest.display());

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");

    // FP32 reference.
    let fp32_model = AudioModel::load(&ctx, fp32_manifest).expect("AudioModel::load FP32 manifest");
    let (fp32_logits, n_seg) = collect_per_segment_logits(&fp32_model, fx);

    // FP16 candidate.
    let fp16_model = AudioModel::load(&ctx, fp16_manifest).expect("AudioModel::load FP16 manifest");
    let (fp16_logits, n_seg_16) = collect_per_segment_logits(&fp16_model, fx);

    // Production-path strategy exercise — confirms the chosen strategy
    // runs end-to-end on both precisions. The §2.2 gates above measure
    // strategy-independent output (`compute_mel_per_segment` +
    // `run_ort_logits_on_host_mel`), so this call is a structural
    // sanity check that closes the production code-path coverage gap.
    report_detect_segment_count_parity(&fp32_model, &fp16_model, fx, strategy);

    assert_eq!(
        n_seg, n_seg_16,
        "segment count mismatch: fp32={n_seg} fp16={n_seg_16}"
    );

    // Convert to confidences.
    let fp32_conf: Vec<f32> = fp32_logits.iter().map(|&l| sigmoid(l)).collect();
    let fp16_conf: Vec<f32> = fp16_logits.iter().map(|&l| sigmoid(l)).collect();

    // Per-segment max-abs Δ.
    let mut conf_max_abs = 0.0f32;
    let mut conf_max_idx = 0usize;
    let mut conf_sum_abs = 0.0f64;
    for (i, (a, b)) in fp32_conf.iter().zip(&fp16_conf).enumerate() {
        let d = (a - b).abs();
        conf_sum_abs += d as f64;
        if d > conf_max_abs {
            conf_max_abs = d;
            conf_max_idx = i;
        }
    }
    let conf_mean_abs = (conf_sum_abs / n_seg as f64) as f32;

    // Relative Δ on segments with FP32 conf > floor.
    let mut rel_max = 0.0f32;
    let mut rel_max_idx = 0usize;
    let mut rel_n = 0usize;
    for (i, (a, b)) in fp32_conf.iter().zip(&fp16_conf).enumerate() {
        if *a > CONF_REL_FLOOR {
            let r = ((a - b).abs()) / *a;
            rel_n += 1;
            if r > rel_max {
                rel_max = r;
                rel_max_idx = i;
            }
        }
    }

    // Class-label flip count at threshold 0.9.
    let mut flips = 0usize;
    let mut flip_examples: Vec<(usize, f32, f32)> = Vec::new();
    for (i, (a, b)) in fp32_conf.iter().zip(&fp16_conf).enumerate() {
        let aa = *a >= FLIP_THRESHOLD;
        let bb = *b >= FLIP_THRESHOLD;
        if aa != bb {
            flips += 1;
            if flip_examples.len() < 5 {
                flip_examples.push((i, *a, *b));
            }
        }
    }

    eprintln!("\nn_segments   : {n_seg}\nrel-eligible : {rel_n} (FP32 conf > {CONF_REL_FLOOR})");

    // Verdicts (using the canonical "met" / "exceeded" framing per
    // `feedback_no_soft_tolerance_framing_on_gates.md`).
    let max_v = if conf_max_abs <= CONF_MAX_ABS_GATE {
        "met"
    } else {
        "exceeded"
    };
    eprintln!(
        "max-abs Δ    = {:.6e} (gate ≤ {:.0e}) — {}  (seg={}, fp32={:.6}, fp16={:.6})",
        conf_max_abs,
        CONF_MAX_ABS_GATE,
        max_v,
        conf_max_idx,
        fp32_conf[conf_max_idx],
        fp16_conf[conf_max_idx]
    );

    let mean_v = if conf_mean_abs <= CONF_MEAN_ABS_GATE {
        "met"
    } else {
        "exceeded"
    };
    eprintln!(
        "mean-abs Δ   = {:.6e} (gate ≤ {:.0e}) — {}",
        conf_mean_abs, CONF_MEAN_ABS_GATE, mean_v
    );

    let rel_v = if rel_max <= CONF_REL_GATE {
        "met"
    } else {
        "exceeded"
    };
    if rel_n > 0 {
        eprintln!(
            "rel Δ (max)  = {:.4e} (gate ≤ {:.0e}) — {}  (seg={}, fp32={:.6}, fp16={:.6})",
            rel_max,
            CONF_REL_GATE,
            rel_v,
            rel_max_idx,
            fp32_conf[rel_max_idx],
            fp16_conf[rel_max_idx]
        );
    } else {
        eprintln!("rel Δ        = (no segments with FP32 conf > {CONF_REL_FLOOR}) — skipped");
    }

    let flip_v = if flips == 0 { "met" } else { "exceeded" };
    eprintln!(
        "label flips  = {} @ {:.2} threshold (gate = 0) — {}",
        flips, FLIP_THRESHOLD, flip_v
    );
    if !flip_examples.is_empty() {
        eprintln!("  flip examples (idx, fp32, fp16): {flip_examples:?}");
    }

    // Final assertions (STOP-and-ping on any exceedance).
    let mut failures: Vec<String> = Vec::new();
    if conf_max_abs > CONF_MAX_ABS_GATE {
        failures.push(format!(
            "max-abs Δ {conf_max_abs:.4e} exceeded gate {CONF_MAX_ABS_GATE:.0e}"
        ));
    }
    if conf_mean_abs > CONF_MEAN_ABS_GATE {
        failures.push(format!(
            "mean-abs Δ {conf_mean_abs:.4e} exceeded gate {CONF_MEAN_ABS_GATE:.0e}"
        ));
    }
    if rel_n > 0 && rel_max > CONF_REL_GATE {
        failures.push(format!(
            "rel Δ {rel_max:.4e} exceeded gate {CONF_REL_GATE:.0e} (eligible n={rel_n})"
        ));
    }
    if flips != 0 {
        failures.push(format!(
            "label-flip count {flips} exceeded gate 0 at threshold {FLIP_THRESHOLD}"
        ));
    }

    if !failures.is_empty() {
        panic!(
            "FP16 audit FAILED on {label} ({} gate(s) exceeded). STOP-and-ping lead. Details:\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }

    eprintln!("\nALL FOUR §2.2 GATES MET on {label}.\n");
}

// ---------------------------------------------------------------------------
// Test functions — both DUNAS clips
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn audio_fp16_audit_dunas_20230925() {
    let fx = PathBuf::from(FIXTURE_20230925);
    if require_path(&fx, "fixture").is_none() {
        return;
    }
    let fp32 = fp32_manifest_path();
    if require_path(&fp32, "fp32 manifest").is_none() {
        return;
    }
    let fp16 = fp16_manifest_path();
    if require_path(&fp16, "fp16 manifest").is_none() {
        return;
    }
    run_fp16_audit_for_fixture(&fx, &fp32, &fp16, "DUNAS_20230925_090000");
}

#[test]
#[ignore]
fn audio_fp16_audit_dunas_20230314() {
    let fx = PathBuf::from(FIXTURE_20230314);
    if require_path(&fx, "fixture").is_none() {
        return;
    }
    let fp32 = fp32_manifest_path();
    if require_path(&fp32, "fp32 manifest").is_none() {
        return;
    }
    let fp16 = fp16_manifest_path();
    if require_path(&fp16, "fp16 manifest").is_none() {
        return;
    }
    run_fp16_audit_for_fixture(&fx, &fp32, &fp16, "DUNAS_20230314_090000");
}

// ---------------------------------------------------------------------------
// Strategy::SingleCall coverage — Phase 3.8 Step 2 audit-fix R2 (R1-F3)
//
// `Strategy::SingleCall` is the production default for non-streaming
// detect (locked by the `default_strategies_split_streaming_vs_non_streaming`
// test in `audio.rs`). The four §2.2 R2 numerical-accuracy gates are
// strategy-independent, so these tests share the same harness as the
// HybridA{16} cases above but exercise `detect(Strategy::SingleCall)`
// end-to-end on both FP16 and FP32. Closes the production code-path
// coverage gap raised by audit-fix R1 finding F3.
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn audio_fp16_audit_strategy_single_dunas_20230925() {
    let fx = PathBuf::from(FIXTURE_20230925);
    if require_path(&fx, "fixture").is_none() {
        return;
    }
    let fp32 = fp32_manifest_path();
    if require_path(&fp32, "fp32 manifest").is_none() {
        return;
    }
    let fp16 = fp16_manifest_path();
    if require_path(&fp16, "fp16 manifest").is_none() {
        return;
    }
    run_fp16_audit_for_fixture_with_strategy(
        &fx,
        &fp32,
        &fp16,
        "DUNAS_20230925_090000 (SingleCall)",
        Strategy::SingleCall,
    );
}

#[test]
#[ignore]
fn audio_fp16_audit_strategy_single_dunas_20230314() {
    let fx = PathBuf::from(FIXTURE_20230314);
    if require_path(&fx, "fixture").is_none() {
        return;
    }
    let fp32 = fp32_manifest_path();
    if require_path(&fp32, "fp32 manifest").is_none() {
        return;
    }
    let fp16 = fp16_manifest_path();
    if require_path(&fp16, "fp16 manifest").is_none() {
        return;
    }
    run_fp16_audit_for_fixture_with_strategy(
        &fx,
        &fp32,
        &fp16,
        "DUNAS_20230314_090000 (SingleCall)",
        Strategy::SingleCall,
    );
}
