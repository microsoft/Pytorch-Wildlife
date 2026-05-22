//! W1.5 bench helper: time ORT inference via IoBinding vs host-roundtrip
//! on the live `MD_AudioBirds_V1.onnx` model.
//!
//! Single fresh-process invocation: warm runs first, then `--inner-iters`
//! timed runs of each path. Emits one JSON line on stdout. Python harness
//! `scripts/bench_ort_iobinding.py` orchestrates 5 fresh processes × 3
//! batch sizes (1, 4, 16) and aggregates median/p95/stddev/max.
//!
//! Usage:
//!     cargo run --release --example bench_ort_io -- \
//!         --model /home/miao/repos/PW_refactor/test_files/sparrow_engine_models_test/md-audiobirds-v1/MD_AudioBirds_V1.onnx \
//!         --batch 16 --inner-iters 50

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use sparrow_engine::audio::ort_io::AudioOrtSession;
use cudarc::driver::CudaContext;

const N_MELS: usize = 224;
const TIME_STEPS: usize = 90;

fn parse_str_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|p| args.get(p + 1))
        .cloned()
}

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    parse_str_arg(args, flag)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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

fn percentile(samples_ms: &[f64], pct: f64) -> f64 {
    let mut sorted = samples_ms.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let idx = ((pct / 100.0) * (n as f64 - 1.0)).round() as usize;
    sorted[idx.min(n - 1)]
}

fn median(s: &[f64]) -> f64 {
    let mut sorted = s.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        0.5 * (sorted[n / 2 - 1] + sorted[n / 2])
    }
}

fn stddev(s: &[f64]) -> f64 {
    if s.len() < 2 {
        return 0.0;
    }
    let mean = s.iter().sum::<f64>() / s.len() as f64;
    let var = s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (s.len() - 1) as f64;
    var.sqrt()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = PathBuf::from(
        parse_str_arg(&args, "--model").unwrap_or_else(|| {
            "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models_test/md-audiobirds-v1/MD_AudioBirds_V1.onnx".to_string()
        }),
    );
    let batch: usize = parse_arg(&args, "--batch", 16);
    let inner_iters: usize = parse_arg(&args, "--inner-iters", 50);
    let warmup: usize = parse_arg(&args, "--warmup", 5);

    if !model_path.exists() {
        panic!("model not found at {model_path:?}; pass --model <path>");
    }

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    // Phase 3.8 Step 2 perf-fix Fix D: AudioOrtSession is bound to a
    // dedicated non-default stream via `with_compute_stream`; pass that
    // stream into `load`.
    let stream = ctx.new_stream().expect("ctx.new_stream");
    let session =
        AudioOrtSession::load(&ctx, &stream, &model_path).expect("AudioOrtSession::load");

    let total = batch * N_MELS * TIME_STEPS;
    let mel_host = lcg_rand_vec(0xBEEF_F00D, total);
    let mel_d = stream.clone_htod(&mel_host).expect("clone_htod mel");

    // Warmup both paths.
    for _ in 0..warmup {
        let _ = session
            .run_iobinding(&Arc::clone(&stream), &mel_d, batch, N_MELS, TIME_STEPS)
            .expect("warmup iobinding");
        let _ = session
            .run_host_roundtrip(&Arc::clone(&stream), &mel_d, batch, N_MELS, TIME_STEPS)
            .expect("warmup host_roundtrip");
    }

    // Bench IoBinding.
    let mut iob_ms = Vec::with_capacity(inner_iters);
    for _ in 0..inner_iters {
        let t0 = Instant::now();
        let _ = session
            .run_iobinding(&Arc::clone(&stream), &mel_d, batch, N_MELS, TIME_STEPS)
            .expect("iobinding run");
        iob_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // Bench host-roundtrip.
    let mut host_ms = Vec::with_capacity(inner_iters);
    for _ in 0..inner_iters {
        let t0 = Instant::now();
        let _ = session
            .run_host_roundtrip(&Arc::clone(&stream), &mel_d, batch, N_MELS, TIME_STEPS)
            .expect("host_roundtrip run");
        host_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let iob_p50 = median(&iob_ms);
    let iob_p95 = percentile(&iob_ms, 95.0);
    let iob_sd = stddev(&iob_ms);
    let iob_mx = iob_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let host_p50 = median(&host_ms);
    let host_p95 = percentile(&host_ms, 95.0);
    let host_sd = stddev(&host_ms);
    let host_mx = host_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!(
        "{{\"primitive\":\"ort_io\",\"batch\":{batch},\
         \"iobinding_p50_ms\":{iob_p50},\"iobinding_p95_ms\":{iob_p95},\"iobinding_stddev_ms\":{iob_sd},\"iobinding_max_ms\":{iob_mx},\
         \"host_roundtrip_p50_ms\":{host_p50},\"host_roundtrip_p95_ms\":{host_p95},\"host_roundtrip_stddev_ms\":{host_sd},\"host_roundtrip_max_ms\":{host_mx},\
         \"inner_iters\":{inner_iters},\"warmup\":{warmup}}}"
    );
}
