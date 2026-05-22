//! Phase 3.8 Step 2 perf-triage — ORT IoBinding cost vs buffer churn.
//!
//! Hypothesis: ORT's CUDA EP IoBinding pays a per-pointer rebind cost
//! (cuDNN algo cache miss / memory-plan refresh) on the first call against
//! any new device pointer. The standalone `bench_ort_io` reuses the SAME
//! `mel_d` buffer across all iters, hiding this cost. The e2e detect
//! call (`AudioModel::run_strategy_a`) freshly allocates `mel_row_d`
//! every call, paying the cost every iter.
//!
//! This bench measures both modes:
//! 1. `static` — same buffer reused for all iters (matches `bench_ort_io`)
//! 2. `fresh` — fresh-allocated buffer per iter (matches e2e detect)
//!
//! If the hypothesis is right, `fresh` p50 will be ≫ `static` p50.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use sparrow_engine::audio::ort_io::AudioOrtSession;
use cudarc::driver::CudaContext;

const N_MELS: usize = 224;
const TIME_STEPS: usize = 90;

fn parse_str_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter().position(|a| a == flag).and_then(|p| args.get(p + 1)).cloned()
}
fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    parse_str_arg(args, flag).and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn lcg_rand_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let f = (z >> 40) as f32 / (1u64 << 24) as f32;
        out.push(f * 80.0 - 80.0);
    }
    out
}
fn percentile(samples: &[f64], pct: f64) -> f64 {
    let mut s = samples.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    let idx = ((pct / 100.0) * (n as f64 - 1.0)).round() as usize;
    s[idx.min(n - 1)]
}
fn median(s: &[f64]) -> f64 {
    let mut sorted = s.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) }
}
fn stddev(s: &[f64]) -> f64 {
    if s.len() < 2 { return 0.0; }
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (s.len() - 1) as f64;
    var.sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = PathBuf::from(parse_str_arg(&args, "--model").unwrap_or_else(|| {
        "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models/md-audiobirds-v1/MD_AudioBirds_V1.onnx".to_string()
    }));
    let batch: usize = parse_arg(&args, "--batch", 197);
    let inner_iters: usize = parse_arg(&args, "--inner-iters", 30);
    let warmup: usize = parse_arg(&args, "--warmup", 5);
    if !model_path.exists() {
        panic!("model not found: {model_path:?}");
    }

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    // Phase 3.8 Step 2 perf-fix Fix D: AudioOrtSession is bound to a
    // dedicated non-default stream.
    let stream = ctx.new_stream().expect("ctx.new_stream");
    let session =
        AudioOrtSession::load(&ctx, &stream, &model_path).expect("AudioOrtSession::load");

    let total = batch * N_MELS * TIME_STEPS;
    let mel_host = lcg_rand_vec(0xBEEF_F00D, total);

    // ---------- MODE 1: static buffer, same `mel_d` reused ----------
    let mel_d_static = stream.clone_htod(&mel_host).expect("clone_htod static");
    for _ in 0..warmup {
        let _ = session.run_iobinding(&Arc::clone(&stream), &mel_d_static, batch, N_MELS, TIME_STEPS)
            .expect("warmup static");
    }
    let mut static_ms = Vec::with_capacity(inner_iters);
    for _ in 0..inner_iters {
        let t0 = Instant::now();
        let _ = session.run_iobinding(&Arc::clone(&stream), &mel_d_static, batch, N_MELS, TIME_STEPS)
            .expect("iobinding static");
        static_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // ---------- MODE 2: fresh buffer per iter ----------
    // Warmup with N different fresh buffers — should reveal whether the
    // first call after a fresh pointer is more expensive.
    for _ in 0..warmup {
        let mel_d_fresh = stream.clone_htod(&mel_host).expect("clone_htod fresh warmup");
        let _ = session.run_iobinding(&Arc::clone(&stream), &mel_d_fresh, batch, N_MELS, TIME_STEPS)
            .expect("warmup fresh");
        // mel_d_fresh dropped at end of scope.
    }
    let mut fresh_ms = Vec::with_capacity(inner_iters);
    let mut fresh_alloc_ms = Vec::with_capacity(inner_iters);
    for _ in 0..inner_iters {
        let t_alloc = Instant::now();
        let mel_d_fresh = stream.clone_htod(&mel_host).expect("clone_htod fresh");
        fresh_alloc_ms.push(t_alloc.elapsed().as_secs_f64() * 1000.0);
        let t0 = Instant::now();
        let _ = session.run_iobinding(&Arc::clone(&stream), &mel_d_fresh, batch, N_MELS, TIME_STEPS)
            .expect("iobinding fresh");
        fresh_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // ---------- MODE 3: alloc_zeros + memset per iter (matches e2e exactly) ----------
    // The e2e detect uses `alloc_zeros` (cudaMalloc + cudaMemset) for mel_row_d.
    // The contents of mel_row_d are then computed by the mel pipeline (so the
    // initial zeros are overwritten before ORT). For this microbench, we
    // allocate a zero-filled buffer + run ORT directly on it (the buffer is
    // all-zeros — the model still runs, just produces -log(0)-ish output).
    // What matters is the cost of the IoBinding call itself.
    for _ in 0..warmup {
        let mel_d_zeros = stream.alloc_zeros::<f32>(total).expect("alloc_zeros warmup");
        let _ = session.run_iobinding(&Arc::clone(&stream), &mel_d_zeros, batch, N_MELS, TIME_STEPS)
            .expect("warmup zeros");
    }
    let mut zeros_ms = Vec::with_capacity(inner_iters);
    for _ in 0..inner_iters {
        let mel_d_zeros = stream.alloc_zeros::<f32>(total).expect("alloc_zeros");
        let t0 = Instant::now();
        let _ = session.run_iobinding(&Arc::clone(&stream), &mel_d_zeros, batch, N_MELS, TIME_STEPS)
            .expect("iobinding zeros");
        zeros_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    eprintln!();
    eprintln!("# ORT IoBinding cost: static buffer vs fresh-per-iter");
    eprintln!("Model: {}", model_path.display());
    eprintln!("Batch: {batch}, n_mels: {N_MELS}, time_steps: {TIME_STEPS}");
    eprintln!("Iters: {inner_iters} timed (after {warmup} warmup)");
    eprintln!();
    eprintln!("| Mode | p50 (ms) | p95 (ms) | stddev (ms) | max (ms) |");
    eprintln!("| --- | ---: | ---: | ---: | ---: |");
    for (label, samples) in [
        ("static buffer reused", static_ms.as_slice()),
        ("fresh clone_htod per iter (ORT only)", fresh_ms.as_slice()),
        ("    fresh clone_htod alloc cost (separate)", fresh_alloc_ms.as_slice()),
        ("alloc_zeros per iter (ORT only) — matches e2e", zeros_ms.as_slice()),
    ] {
        let p50 = median(samples);
        let p95 = percentile(samples, 95.0);
        let sd = stddev(samples);
        let mx = samples.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        eprintln!("| {} | {:.4} | {:.4} | {:.4} | {:.4} |", label, p50, p95, sd, mx);
    }

    println!(
        "{{\"primitive\":\"ort_io_fresh\",\"batch\":{batch},\
         \"static_p50_ms\":{},\"static_p95_ms\":{},\
         \"fresh_p50_ms\":{},\"fresh_p95_ms\":{},\
         \"fresh_alloc_p50_ms\":{},\
         \"zeros_p50_ms\":{},\"zeros_p95_ms\":{},\
         \"inner_iters\":{inner_iters},\"warmup\":{warmup}}}",
        median(&static_ms), percentile(&static_ms, 95.0),
        median(&fresh_ms), percentile(&fresh_ms, 95.0),
        median(&fresh_alloc_ms),
        median(&zeros_ms), percentile(&zeros_ms, 95.0),
    );
}
