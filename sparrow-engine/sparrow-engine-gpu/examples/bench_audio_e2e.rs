//! Phase 3.8 Step 2 Wave 2 — End-to-end audio bench harness.
//!
//! Single-cell driver: loads MD_AudioBirds_V1 + runs detect on a fixed
//! audio clip; prints per-iteration timing JSON to stdout.
//!
//! Spawned by `scripts/bench_audio_e2e.py` 5× per cell to satisfy
//! `feedback_perf_claims_need_variance.md` (≥ 5 fresh-process runs).
//!
//! ## Env vars
//!
//! Required:
//! - `SPARROW_ENGINE_AUDIO_BENCH_MANIFEST` — path to manifest.toml.
//! - `SPARROW_ENGINE_AUDIO_BENCH_FIXTURE`  — path to a WAV file (DUNAS clip).
//! - `SPARROW_ENGINE_AUDIO_BENCH_STRATEGY` — `A`, `B`, or `S` (SingleCall —
//!   production default for non-streaming detect; ignores T).
//! - `SPARROW_ENGINE_AUDIO_BENCH_T`        — T value (1..=N segments). Ignored
//!   when STRATEGY=S.
//!
//! Optional:
//! - `SPARROW_ENGINE_AUDIO_BENCH_INNER_ITERS` — warm iterations to time (default 10).
//! - `SPARROW_ENGINE_AUDIO_BENCH_WARMUP`      — warmup iterations (default 2).
//! - `SPARROW_ENGINE_AUDIO_BENCH_THRESHOLD`   — confidence threshold override.
//!
//! ## Output
//!
//! One JSON line on stdout (newline-terminated) with the following shape:
//!
//! ```text
//! {
//!   "strategy": "A" | "B",
//!   "t": <usize>,
//!   "fixture": "...",
//!   "inner_iters": <usize>,
//!   "warmup": <usize>,
//!   "n_segments_above_threshold": <usize>,
//!   "n_segments_total": <usize>,
//!   "per_iter_ms": [...],
//!   "p50_ms": <f64>,
//!   "p95_ms": <f64>,
//!   "stddev_ms": <f64>,
//!   "max_ms": <f64>
//! }
//! ```

use std::env;
use std::path::PathBuf;
use std::time::Instant;

use sparrow_engine::models::audio::{AudioModel, GpuAudioDetectOpts, Strategy};
use sparrow_engine_types::{AudioDetectOpts, AudioInput};
use cudarc::driver::CudaContext;

fn env_var(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| panic!("env var {key} required"))
}

fn env_var_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    match env::var(key) {
        Ok(v) => v.parse().unwrap_or(default),
        Err(_) => default,
    }
}

fn percentile(samples_ms: &[f64], pct: f64) -> f64 {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let idx = ((pct / 100.0) * (n as f64 - 1.0)).round() as usize;
    sorted[idx.min(n - 1)]
}

fn median(samples_ms: &[f64]) -> f64 {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

fn stddev(samples_ms: &[f64]) -> f64 {
    if samples_ms.len() < 2 {
        return 0.0;
    }
    let mean = samples_ms.iter().sum::<f64>() / samples_ms.len() as f64;
    let var = samples_ms.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
        / (samples_ms.len() - 1) as f64;
    var.sqrt()
}

fn main() {
    let manifest = PathBuf::from(env_var("SPARROW_ENGINE_AUDIO_BENCH_MANIFEST"));
    let fixture = PathBuf::from(env_var("SPARROW_ENGINE_AUDIO_BENCH_FIXTURE"));
    let strategy_str = env_var("SPARROW_ENGINE_AUDIO_BENCH_STRATEGY");
    // T is ignored when STRATEGY=S (SingleCall); default to 0 so the
    // env var is no longer required for that cell.
    let t: usize = env::var("SPARROW_ENGINE_AUDIO_BENCH_T")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let inner_iters: usize = env_var_or("SPARROW_ENGINE_AUDIO_BENCH_INNER_ITERS", 10);
    let warmup: usize = env_var_or("SPARROW_ENGINE_AUDIO_BENCH_WARMUP", 2);
    let threshold_override: Option<f32> = env::var("SPARROW_ENGINE_AUDIO_BENCH_THRESHOLD")
        .ok()
        .and_then(|s| s.parse().ok());

    let strategy = match strategy_str.as_str() {
        "A" => Strategy::HybridA { ort_chunk_segments: t },
        "B" => Strategy::PerBatchB { batch_segments: t },
        "S" => Strategy::SingleCall,
        other => panic!("SPARROW_ENGINE_AUDIO_BENCH_STRATEGY must be A, B, or S, got {other:?}"),
    };

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let model = AudioModel::load(&ctx, &manifest).expect("AudioModel::load");

    let base = AudioDetectOpts {
        confidence_threshold: threshold_override,
        ..Default::default()
    };
    let opts = GpuAudioDetectOpts {
        base,
        strategy,
    };

    // Warmup (untimed).
    for _ in 0..warmup {
        let _ = model
            .detect(&AudioInput::FilePath(fixture.clone()), &opts)
            .expect("detect (warmup)");
    }

    // Timed iterations.
    let mut per_iter_ms = Vec::with_capacity(inner_iters);
    let mut last_segments_above: usize = 0;
    let mut last_n_total: usize = 0;
    for _ in 0..inner_iters {
        let t0 = Instant::now();
        let res = model
            .detect(&AudioInput::FilePath(fixture.clone()), &opts)
            .expect("detect");
        let elapsed = t0.elapsed();
        per_iter_ms.push(elapsed.as_secs_f64() * 1000.0);
        last_segments_above = res.segments.len();
        // Total segment count, matching `compute_segment_offsets`:
        // every offset where `remaining > segment_samples` plus the
        // final tail-padded offset. For 1.0 s segment + 0.3 s stride at
        // 48 kHz that's `ceil((duration - 1.0) / 0.3) + 1` (e.g.
        // 60 s → 198 segments, NOT the 197 the previous `floor() + 1`
        // formula reported — see perf_triage_report.md "Step 3 —
        // Prototype + measurement"). The bench harness's bench cells
        // were paying chunk-overhead for this off-by-one before
        // SingleCall landed.
        let segment_dur_s = 1.0_f64;
        let stride_s = 0.3_f64;
        let usable = (res.duration_s as f64 - segment_dur_s) / stride_s;
        last_n_total = if usable < 0.0 {
            0
        } else {
            usable.ceil() as usize + 1
        };
    }

    let p50 = median(&per_iter_ms);
    let p95 = percentile(&per_iter_ms, 95.0);
    let sd = stddev(&per_iter_ms);
    let mx = per_iter_ms
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);

    // JSON line.
    let per_iter_json = per_iter_ms
        .iter()
        .map(|x| format!("{x:.6}"))
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "{{\"strategy\":\"{strategy_str}\",\"t\":{t},\"fixture\":\"{fixture}\",\"inner_iters\":{inner_iters},\"warmup\":{warmup},\"n_segments_above_threshold\":{last_segments_above},\"n_segments_total\":{last_n_total},\"per_iter_ms\":[{per_iter_json}],\"p50_ms\":{p50:.6},\"p95_ms\":{p95:.6},\"stddev_ms\":{sd:.6},\"max_ms\":{mx:.6}}}",
        strategy_str = strategy_str,
        t = t,
        fixture = fixture.display(),
        inner_iters = inner_iters,
        warmup = warmup,
        last_segments_above = last_segments_above,
        last_n_total = last_n_total,
        per_iter_json = per_iter_json,
        p50 = p50,
        p95 = p95,
        sd = sd,
        mx = mx,
    );
}
