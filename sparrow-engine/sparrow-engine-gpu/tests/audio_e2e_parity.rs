//! Phase 3.8 Step 2 Wave 2 — End-to-end FP32 audio parity test.
//!
//! Runs both `sparrow-engine-cpu::detect_audio` (CPU baseline) and
//! `sparrow_engine::models::audio::AudioModel::detect` against the same
//! `MD_AudioBirds_V1` manifest + DUNAS fixtures. Asserts every gate
//! from the Wave 2 §2.1 table, **re-derived against the W1.7
//! amp curve** (lead-approved 2026-05-05; see
//! `docs/research/phase3.8/step2/wave2_e2e_bench.md` § "Gate
//! re-derivation (W1.7-anchored)"):
//!
//! | Gate | Threshold | Source |
//! | --- | --- | --- |
//! | Mel max-Δ vs CPU | ≤ 5e-3 dB | UNCHANGED (arch-par fallback) |
//! | Logit max-Δ | ≤ 3.0e-3 | re-derived against W1.7 amp curve at mel Δ ≈ 6e-4 dB regime (amp ≈ 3.07× linear interp + 1.6× safety) |
//! | Confidence max-Δ | ≤ 7.5e-4 | sigmoid Lipschitz ≤ 0.25 × 3.0e-3 |
//! | Class-label flip count at threshold 0.9 | = 0 | UNCHANGED — semantic gate |
//! | Range-count parity (post-merge) | exact match per clip | UNCHANGED |
//!
//! W1.7 amp factor measurement: max **5.96×** at ε=1e-4 mel
//! perturbation, **0.77×** at ε=1e-3 (`docs/research/phase3.8/step2/
//! wave1_primitives_bench.md` § W1.7). Linear interp at the corpus-
//! measured mel Δ regime ≈ 6e-4 dB → amp ≈ 3.07× → predicted logit
//! Δ ≈ 1.84e-3. The 3.0e-3 gate is empirically calibrated with 1.6×
//! safety margin against W1.7's measured worst-case amp at this mel
//! Δ regime.
//!
//! Per `feedback_no_soft_tolerance_framing_on_gates.md`: the test
//! reports the exact integer/decimal Δ + verdict ("met"/"exceeded"),
//! and asserts at the W1.7-anchored gate value (no permissive
//! multiplier). The previous arch-par §2.1 5e-4 logit gate assumed a
//! "10× contraction" through ORT — empirically refuted by W1.7 (max
//! amp 5.96×, 60× off from the 10× contraction model). This is a
//! magnitude-aware re-derivation against measured W1.7 amp, NOT a
//! tolerance loosening.
//!
//! ## Why `#[ignore]`
//!
//! The four parity tests below are gated `#[ignore]` because they
//! depend on resources that aren't always present and aren't desired
//! in default `cargo test` runs:
//!
//! 1. **The `MD_AudioBirds_V1.onnx` model file** at
//!    `/home/miao/repos/PW_refactor/test_files/onnx/`. Not bundled with
//!    sparrow-engine (85 MB).
//! 2. **The DUNAS WAV fixtures** at
//!    `/home/miao/repos/PW_refactor/test_files/test_audio/`. Real-audio
//!    60 s clips, ~6 MB each. Not bundled.
//! 3. **A working CUDA GPU** with the cudarc + ORT runtime stack
//!    initialized via `sparrow-engine/scripts/ort-env.sh`. CPU-only environments
//!    cannot run the GPU side of the parity comparison.
//! 4. **A few seconds of execution time per test** — each test runs
//!    one CPU `detect_audio` + one GPU `detect` + post-merge range
//!    comparison, on a real 60 s clip; not appropriate for the default
//!    `cargo test` smoke loop.
//!
//! Setting `SPARROW_ENGINE_AUDIO_FORCE_FIXTURE=1` makes a missing fixture fatal
//! (instead of a skip).
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
//! cargo test --release -p sparrow-engine-gpu --test audio_e2e_parity \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//!
//! ## What each test asserts
//!
//! | Test | Manifest | Fixture | Strategy under test |
//! | --- | --- | --- | --- |
//! | `audio_e2e_parity_strategy_single_dunas_20230925` | `manifest_fp32.toml` (FP32 ref) | DUNAS_20230925_090000.wav | `Strategy::SingleCall` (production default for non-streaming) |
//! | `audio_e2e_parity_strategy_single_dunas_20230314` | `manifest_fp32.toml` | DUNAS_20230314_090000.wav | `Strategy::SingleCall` |
//! | `audio_e2e_parity_strategy_a_dunas_20230925` | `manifest_fp32.toml` | DUNAS_20230925_090000.wav | `Strategy::HybridA{16}` (streaming-cadence default) |
//! | `audio_e2e_parity_strategy_a_dunas_20230314` | `manifest_fp32.toml` | DUNAS_20230314_090000.wav | `Strategy::HybridA{16}` |
//! | `audio_e2e_parity_strategy_b_dunas_20230925` | `manifest_fp32.toml` | DUNAS_20230925_090000.wav | `Strategy::PerBatchB { batch_segments: 16 }` (F8 coverage) |
//!
//! Each test runs the same five §2.1 W1.7-anchored gates (mel max-Δ ≤ 5e-3 dB,
//! logit max-Δ ≤ 3.0e-3, conf max-Δ ≤ 7.5e-4, label flips @ 0.9 = 0,
//! merged-range count exact match) and panics with the gate name + measured
//! value on any exceedance.

use std::path::{Path, PathBuf};

// Phase 3.8 Phase C Wave 4b (2026-05-06): both `sparrow-engine-cpu` and
// `sparrow-engine-gpu` now set `[lib] name = "sparrow_engine"`
// (`libsparrow_engine.so` invariant — see
// `docs/design/phase3.8/phase_c/implementation_plan.md` §2.2). The
// dev-dep `sparrow-engine-cpu` is renamed to `sparrow_engine_cpu` via
// Cargo `package =` in `sparrow-engine-gpu/Cargo.toml` so cross-engine
// tests can disambiguate the CPU baseline (`sparrow_engine_cpu::*`) from
// the current crate surface (`sparrow_engine::*`).
use cudarc::driver::CudaContext;
use sparrow_engine::models::audio::{AudioModel, GpuAudioDetectOpts, MelDebugSnapshot, Strategy};
use sparrow_engine_core::preprocess_audio::{self, AudioPreprocessConfig, MelFilterbank};
use sparrow_engine_cpu::detect_audio;
use sparrow_engine_cpu::engine::{Device, Engine, EngineConfig};
use sparrow_engine_types::{AudioDetectOpts, AudioInput, AudioRange, AudioSegment};

// ---------------------------------------------------------------------------
// W1.7-anchored gates (lead-approved 2026-05-05)
// ---------------------------------------------------------------------------

const MEL_GATE_DB: f32 = 5e-3;
const LOGIT_GATE: f32 = 3.0e-3; // re-derived against W1.7 amp curve
const CONF_GATE: f32 = 7.5e-4; // sigmoid Lipschitz × LOGIT_GATE

// ---------------------------------------------------------------------------
// Fixture discovery
// ---------------------------------------------------------------------------

const FIXTURE_20230925: &str =
    "/home/miao/repos/PW_refactor/test_files/test_audio/DUNAS_20230925_090000.wav";
const FIXTURE_20230314: &str =
    "/home/miao/repos/PW_refactor/test_files/test_audio/DUNAS_20230314_090000.wav";
// Post-STRETCH re-audit (2026-05-05): the production `manifest.toml`
// flipped to `precision = "fp16"`. This test is the W1.7-anchored FP32
// parity gate (CPU FP32 vs GPU FP32) — both sides need FP32, so it loads
// the explicit `manifest_fp32.toml` (added during the FP16 flip). Override
// via the `SPARROW_ENGINE_AUDIO_MANIFEST` env var if running against a different
// manifest file.
const DEFAULT_MANIFEST: &str =
    "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models/md-audiobirds-v1/manifest_fp32.toml";

fn manifest_path() -> PathBuf {
    PathBuf::from(std::env::var("SPARROW_ENGINE_AUDIO_MANIFEST").unwrap_or_else(|_| DEFAULT_MANIFEST.into()))
}

fn force_fixture() -> bool {
    std::env::var("SPARROW_ENGINE_AUDIO_FORCE_FIXTURE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

fn require_fixture(p: &Path) -> Option<()> {
    if p.exists() {
        return Some(());
    }
    if force_fixture() {
        panic!(
            "missing audio fixture {} (SPARROW_ENGINE_AUDIO_FORCE_FIXTURE=1)",
            p.display()
        );
    }
    eprintln!(
        "skipping: missing fixture {} (set SPARROW_ENGINE_AUDIO_FORCE_FIXTURE=1 to make this fatal)",
        p.display()
    );
    None
}

// ---------------------------------------------------------------------------
// CPU mel baseline
// ---------------------------------------------------------------------------

/// Run the sparrow-engine-cpu mel pipeline per segment + collect row-major
/// `[n_mels, frames_per_seg]` into a `Vec<Vec<f32>>` matching the
/// AudioModel's `MelDebugSnapshot::segments` layout.
fn cpu_mel_per_segment(
    audio: &AudioInput,
    config: &AudioPreprocessConfig,
    segment_samples: usize,
    stride_samples: usize,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let samples = preprocess_audio::load_audio(audio, config).expect("load_audio");
    let total = samples.data.len();
    let mut offsets = Vec::new();
    let mut o = 0usize;
    while o < total {
        offsets.push(o);
        let remaining = total - o;
        if remaining <= segment_samples {
            break;
        }
        o += stride_samples;
    }
    let fb = MelFilterbank::new(config).expect("MelFilterbank::new");
    let mut out = Vec::with_capacity(offsets.len());
    for &seg_offset in &offsets {
        let remaining = total - seg_offset;
        let tensor = if remaining >= segment_samples {
            preprocess_audio::mel_spectrogram(
                &samples.data[seg_offset..seg_offset + segment_samples],
                samples.orig_sample_rate,
                config,
                &fb,
            )
            .expect("mel_spectrogram")
        } else {
            let mut padded = samples.data[seg_offset..].to_vec();
            padded.resize(segment_samples, 0.0);
            preprocess_audio::mel_spectrogram(&padded, samples.orig_sample_rate, config, &fb)
                .expect("mel_spectrogram")
        };
        let slice = tensor.as_slice().expect("Array4 contiguous").to_vec();
        out.push(slice);
    }
    (out, offsets)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn max_abs_delta(a: &[f32], b: &[f32]) -> (f32, usize) {
    assert_eq!(a.len(), b.len(), "max_abs_delta: length mismatch");
    let mut max = 0.0f32;
    let mut idx = 0usize;
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        let d = (x - y).abs();
        if d > max {
            max = d;
            idx = i;
        }
    }
    (max, idx)
}

fn merge_segments_local(segments: &[AudioSegment], gap_s: f32) -> Vec<AudioRange> {
    // Local copy of sparrow-engine-cpu's `merge_segments` so the test doesn't
    // depend on the public re-export landing through sparrow-engine-cli.
    let mut ranges: Vec<AudioRange> = Vec::new();
    for seg in segments {
        if let Some(last) = ranges.last_mut() {
            let gap = seg.start_time_s - last.end_time_s;
            if gap < gap_s {
                if seg.end_time_s > last.end_time_s {
                    last.end_time_s = seg.end_time_s;
                }
                if seg.confidence > last.max_confidence {
                    last.max_confidence = seg.confidence;
                }
                continue;
            }
        }
        ranges.push(AudioRange {
            start_time_s: seg.start_time_s,
            end_time_s: seg.end_time_s,
            max_confidence: seg.confidence,
            class: None,
        });
    }
    ranges
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn assert_segment_lists_close(label: &str, cpu: &[AudioSegment], gpu: &[AudioSegment]) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "{label}: CPU/GPU detect() segment count mismatch"
    );
    for (idx, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            (c.start_time_s - g.start_time_s).abs() <= 1e-6,
            "{label}: segment {idx} start mismatch cpu={} gpu={}",
            c.start_time_s,
            g.start_time_s
        );
        assert!(
            (c.end_time_s - g.end_time_s).abs() <= 1e-6,
            "{label}: segment {idx} end mismatch cpu={} gpu={}",
            c.end_time_s,
            g.end_time_s
        );
        assert!(
            (c.confidence - g.confidence).abs() <= CONF_GATE,
            "{label}: segment {idx} confidence Δ={} exceeds gate {} (cpu={} gpu={})",
            (c.confidence - g.confidence).abs(),
            CONF_GATE,
            c.confidence,
            g.confidence
        );
    }
}

fn assert_range_lists_close(label: &str, cpu: &[AudioRange], gpu: &[AudioRange]) {
    assert_eq!(
        cpu.len(),
        gpu.len(),
        "{label}: CPU/GPU merged-range count mismatch"
    );
    for (idx, (c, g)) in cpu.iter().zip(gpu.iter()).enumerate() {
        assert!(
            (c.start_time_s - g.start_time_s).abs() <= 1e-6,
            "{label}: range {idx} start mismatch cpu={} gpu={}",
            c.start_time_s,
            g.start_time_s
        );
        assert!(
            (c.end_time_s - g.end_time_s).abs() <= 1e-6,
            "{label}: range {idx} end mismatch cpu={} gpu={}",
            c.end_time_s,
            g.end_time_s
        );
        assert!(
            (c.max_confidence - g.max_confidence).abs() <= CONF_GATE,
            "{label}: range {idx} max-confidence Δ={} exceeds gate {} (cpu={} gpu={})",
            (c.max_confidence - g.max_confidence).abs(),
            CONF_GATE,
            c.max_confidence,
            g.max_confidence
        );
    }
}

// ---------------------------------------------------------------------------
// Per-fixture parity check
// ---------------------------------------------------------------------------

/// Strategy-parameterized parity runner — covers both the perf-default
/// `Strategy::SingleCall` (post Wave-4 perf triage) and the Wave 2
/// `HybridA{16}` reference path. The two MUST produce identical
/// numerical output because the underlying ORT IoBinding calls are
/// deterministic regardless of how the input mel buffer is sub-batched
/// (see `docs/research/phase3.8/step2/perf_triage_report.md`
/// § "Parity (G-D / W1.7-anchored)").
fn run_parity_for_fixture_with_strategy(
    fx: &Path,
    manifest: &Path,
    label: &str,
    strategy: Strategy,
) {
    eprintln!(
        "\n=== Phase 3.8 Step 2 Wave 2 — FP32 e2e parity ({label}, {}) ===",
        strategy.short_label()
    );
    eprintln!("fixture : {}", fx.display());
    eprintln!("manifest: {}", manifest.display());

    // 1. CPU baseline.
    let model_dir = manifest.parent().unwrap().to_path_buf();
    let cpu_segments_cpu_path: Vec<AudioSegment>;
    let processing_time_cpu_ms: f32;
    {
        let cfg = EngineConfig::new(Device::Cuda(0), &model_dir);
        let engine = Engine::new(cfg).expect("sparrow-engine-cpu Engine::new");
        let handle = engine.load_model(manifest).expect("sparrow-engine-cpu load_model");
        let opts = AudioDetectOpts::default();
        let res =
            detect_audio::detect_audio(&handle, &AudioInput::FilePath(fx.to_path_buf()), &opts)
                .expect("sparrow-engine-cpu detect_audio");
        cpu_segments_cpu_path = res.segments;
        processing_time_cpu_ms = res.processing_time_ms;
        drop(handle);
        drop(engine);
    }

    // 2. GPU AudioModel.
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let model = AudioModel::load(&ctx, manifest).expect("AudioModel::load");
    let gpu_opts = GpuAudioDetectOpts {
        base: AudioDetectOpts::default(),
        strategy,
    };
    let gpu_result = model
        .detect(&AudioInput::FilePath(fx.to_path_buf()), &gpu_opts)
        .expect("AudioModel::detect");
    let gpu_segments = gpu_result.segments.clone();
    let processing_time_gpu_ms = gpu_result.processing_time_ms;

    eprintln!(
        "\nCPU sparrow-engine-cpu detect_audio: {} segments, {:.1} ms",
        cpu_segments_cpu_path.len(),
        processing_time_cpu_ms
    );
    eprintln!(
        "GPU AudioModel  detect    : {} segments, {:.1} ms",
        gpu_segments.len(),
        processing_time_gpu_ms
    );

    // ---------- Mel parity ----------
    eprintln!("\n--- Mel parity (post power_to_db, dB) ---");
    let cpu_audio_config = AudioPreprocessConfig::default();
    let segment_samples = 48_000usize; // 1.0 s @ 48 kHz
    let stride_samples = 14_400usize; // 0.3 s @ 48 kHz
    let (cpu_mel_per_seg, cpu_offsets) = cpu_mel_per_segment(
        &AudioInput::FilePath(fx.to_path_buf()),
        &cpu_audio_config,
        segment_samples,
        stride_samples,
    );
    let gpu_mel_snap: MelDebugSnapshot = model
        .compute_mel_per_segment(
            &AudioInput::FilePath(fx.to_path_buf()),
            &AudioDetectOpts::default(),
        )
        .expect("compute_mel_per_segment");

    assert_eq!(
        cpu_offsets.len(),
        gpu_mel_snap.segment_offsets.len(),
        "segment count mismatch: cpu={} gpu={}",
        cpu_offsets.len(),
        gpu_mel_snap.segment_offsets.len()
    );
    assert_eq!(
        cpu_offsets, gpu_mel_snap.segment_offsets,
        "segment offsets differ"
    );

    let mut mel_max_abs = 0.0f32;
    let mut mel_max_seg = 0usize;
    let mut mel_max_idx = 0usize;
    for (s, cpu_mel) in cpu_mel_per_seg.iter().enumerate() {
        let cpu_seg: &[f32] = cpu_mel.as_slice();
        let gpu_seg: &[f32] = gpu_mel_snap.segments[s].as_slice();
        let cpu_len: usize = cpu_seg.len();
        let gpu_len: usize = gpu_seg.len();
        assert_eq!(
            cpu_len, gpu_len,
            "seg {s} mel len mismatch: cpu={} gpu={}",
            cpu_len, gpu_len
        );
        let (d, i) = max_abs_delta(cpu_seg, gpu_seg);
        if d > mel_max_abs {
            mel_max_abs = d;
            mel_max_seg = s;
            mel_max_idx = i;
        }
    }
    let mel_verdict = if mel_max_abs <= MEL_GATE_DB {
        "met"
    } else {
        "exceeded"
    };
    eprintln!(
        "Mel max-Δ vs CPU = {:.3e} dB (gate ≤ {:.0e}) — {}  (seg={}, idx={})",
        mel_max_abs, MEL_GATE_DB, mel_verdict, mel_max_seg, mel_max_idx
    );

    // ---------- Logit parity (W1.7-anchored gate ≤ 3.0e-3) ----------
    eprintln!("\n--- Logit parity (ORT response on CPU vs GPU mel) ---");
    let n_segments = cpu_mel_per_seg.len();
    let n_mels = gpu_mel_snap.n_mels;
    let frames_per_seg = gpu_mel_snap.frames_per_seg;
    let cpu_mel_concat: Vec<f32> = cpu_mel_per_seg.iter().flatten().copied().collect();
    let gpu_mel_concat: Vec<f32> = gpu_mel_snap.segments.iter().flatten().copied().collect();
    assert_eq!(cpu_mel_concat.len(), n_segments * n_mels * frames_per_seg);
    assert_eq!(gpu_mel_concat.len(), n_segments * n_mels * frames_per_seg);

    let logits_on_cpu_mel = model
        .run_ort_logits_on_host_mel(&cpu_mel_concat, n_segments)
        .expect("run_ort_logits_on_host_mel(cpu_mel)");
    let logits_on_gpu_mel = model
        .run_ort_logits_on_host_mel(&gpu_mel_concat, n_segments)
        .expect("run_ort_logits_on_host_mel(gpu_mel)");
    assert_eq!(logits_on_cpu_mel.len(), n_segments);
    assert_eq!(logits_on_gpu_mel.len(), n_segments);

    let (logit_max_abs, logit_max_idx) = max_abs_delta(&logits_on_cpu_mel, &logits_on_gpu_mel);
    let logit_verdict = if logit_max_abs <= LOGIT_GATE {
        "met"
    } else {
        "exceeded"
    };
    eprintln!(
        "Logit max-Δ = {:.3e} (gate ≤ {:.0e}, W1.7-anchored) — {}  (seg={}, cpu_logit={:.3}, gpu_logit={:.3})",
        logit_max_abs,
        LOGIT_GATE,
        logit_verdict,
        logit_max_idx,
        logits_on_cpu_mel[logit_max_idx],
        logits_on_gpu_mel[logit_max_idx]
    );

    // ---------- Confidence parity (W1.7-anchored gate ≤ 7.5e-4) ----------
    let mut conf_max_abs = 0.0f32;
    let mut conf_max_idx = 0usize;
    for (i, (cl, gl)) in logits_on_cpu_mel.iter().zip(&logits_on_gpu_mel).enumerate() {
        let cc = sigmoid(*cl);
        let gc = sigmoid(*gl);
        let d = (cc - gc).abs();
        if d > conf_max_abs {
            conf_max_abs = d;
            conf_max_idx = i;
        }
    }
    let conf_verdict = if conf_max_abs <= CONF_GATE {
        "met"
    } else {
        "exceeded"
    };
    eprintln!(
        "Confidence max-Δ = {:.3e} (gate ≤ {:.1e}, W1.7-anchored) — {}  (seg={})",
        conf_max_abs, CONF_GATE, conf_verdict, conf_max_idx
    );

    // ---------- Label-flip count at threshold 0.9 ----------
    let threshold = 0.9f32;
    let mut flips = 0usize;
    for (cl, gl) in logits_on_cpu_mel.iter().zip(&logits_on_gpu_mel) {
        let cc = sigmoid(*cl) >= threshold;
        let gc = sigmoid(*gl) >= threshold;
        if cc != gc {
            flips += 1;
        }
    }
    let flip_verdict = if flips == 0 { "met" } else { "exceeded" };
    eprintln!(
        "Label-flip count @0.9 = {} (gate = 0) — {}",
        flips, flip_verdict
    );

    // ---------- Range-count parity (post-merge with stride+1e-3) ----------
    let gap = 0.3f32 + 1e-3;
    let cpu_ranges = merge_segments_local(&cpu_segments_cpu_path, gap);
    let gpu_ranges = merge_segments_local(&gpu_segments, gap);
    let range_match = cpu_ranges.len() == gpu_ranges.len();
    let range_verdict = if range_match { "met" } else { "exceeded" };
    assert_segment_lists_close(label, &cpu_segments_cpu_path, &gpu_segments);
    assert_range_lists_close(label, &cpu_ranges, &gpu_ranges);
    eprintln!(
        "Range-count post-merge: cpu={}, gpu={} — {}",
        cpu_ranges.len(),
        gpu_ranges.len(),
        range_verdict
    );

    // ---------- Final assertions (W1.7-anchored gates) ----------
    let mut failures = Vec::<String>::new();
    if mel_max_abs > MEL_GATE_DB {
        failures.push(format!(
            "Mel max-Δ {mel_max_abs:.3e} dB exceeded gate {MEL_GATE_DB:.0e}"
        ));
    }
    if logit_max_abs > LOGIT_GATE {
        failures.push(format!(
            "Logit max-Δ {logit_max_abs:.3e} exceeded W1.7-anchored gate {LOGIT_GATE:.0e}"
        ));
    }
    if conf_max_abs > CONF_GATE {
        failures.push(format!(
            "Confidence max-Δ {conf_max_abs:.3e} exceeded W1.7-anchored gate {CONF_GATE:.1e}"
        ));
    }
    if flips != 0 {
        failures.push(format!("Label-flip count {flips} exceeded gate 0"));
    }
    if !range_match {
        failures.push(format!(
            "Range-count post-merge cpu={} gpu={} differ",
            cpu_ranges.len(),
            gpu_ranges.len()
        ));
    }

    if !failures.is_empty() {
        panic!(
            "FP32 parity FAILED on {label} ({} gate(s) exceeded). STOP-and-ping lead. Details:\n  - {}",
            failures.len(),
            failures.join("\n  - ")
        );
    }

    eprintln!(
        "\nALL FIVE W1.7-ANCHORED GATES MET on {label}. (cpu_processing={:.1} ms, gpu_processing={:.1} ms)\n",
        processing_time_cpu_ms, processing_time_gpu_ms
    );
}

// ---------------------------------------------------------------------------
// Test functions — both DUNAS clips
// ---------------------------------------------------------------------------

/// S10 parity-test helper (R2 audit-fix 2026-05-05): collapse the
/// 4 test bodies' shared boilerplate (fixture-path resolution,
/// manifest-path resolution, `require_fixture` skip-or-run logic) into
/// one helper. Strategy + label remain at the test-function level for
/// discoverability + per-run grep-ability.
fn run_parity_test_with_fixture(fixture_const: &str, label: &str, strategy: Strategy) {
    let fx = PathBuf::from(fixture_const);
    if require_fixture(&fx).is_none() {
        return;
    }
    let manifest = manifest_path();
    if require_fixture(&manifest).is_none() {
        return;
    }
    run_parity_for_fixture_with_strategy(&fx, &manifest, label, strategy);
}

#[test]
#[ignore]
fn audio_e2e_parity_strategy_a_dunas_20230925() {
    run_parity_test_with_fixture(
        FIXTURE_20230925,
        "DUNAS_20230925_090000",
        Strategy::HybridA {
            ort_chunk_segments: 16,
        },
    );
}

#[test]
#[ignore]
fn audio_e2e_parity_strategy_a_dunas_20230314() {
    run_parity_test_with_fixture(
        FIXTURE_20230314,
        "DUNAS_20230314_090000",
        Strategy::HybridA {
            ort_chunk_segments: 16,
        },
    );
}

/// Phase 3.8 Step 2 perf-fix (post-Wave-4 triage):
/// Same W1.7-anchored gates, exercising the production-default
/// `Strategy::SingleCall` (one ORT `Session::run` per detect). Locks
/// that the perf default produces parity-equivalent output.
#[test]
#[ignore]
fn audio_e2e_parity_strategy_single_dunas_20230925() {
    run_parity_test_with_fixture(
        FIXTURE_20230925,
        "DUNAS_20230925_090000 (SingleCall)",
        Strategy::SingleCall,
    );
}

#[test]
#[ignore]
fn audio_e2e_parity_strategy_single_dunas_20230314() {
    run_parity_test_with_fixture(
        FIXTURE_20230314,
        "DUNAS_20230314_090000 (SingleCall)",
        Strategy::SingleCall,
    );
}

/// Phase 3.8 Step 2 audit-fix R2 (R1-F8 / 2026-05-05): exercise
/// `Strategy::PerBatchB { batch_segments: 16 }` end-to-end.
///
/// Strategy B is documented as a fallback for memory-constrained
/// deployments. By construction it should produce numerically identical
/// output to Strategy A because the underlying ORT IoBinding output is
/// deterministic regardless of how the input mel buffer is sub-batched.
/// This test locks that property by asserting the same five
/// W1.7-anchored gates pass on Strategy B as on Strategy A.
///
/// Only one fixture is exercised (DUNAS_20230925) — both fixtures share
/// the same code path through `run_strategy_b`. Add the second fixture
/// if Strategy B becomes a manifest knob in a future phase.
#[test]
#[ignore]
fn audio_e2e_parity_strategy_b_dunas_20230925() {
    run_parity_test_with_fixture(
        FIXTURE_20230925,
        "DUNAS_20230925_090000 (PerBatchB)",
        Strategy::PerBatchB { batch_segments: 16 },
    );
}
