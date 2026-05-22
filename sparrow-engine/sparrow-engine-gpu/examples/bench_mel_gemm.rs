//! W1.3 bench helper: time cuBLAS sgemm mel-filterbank GEMM on GPU vs the
//! CPU scalar baseline, for a single fresh-process run.
//!
//! Emits one JSON line on stdout (newline-terminated) with measured
//! median/p95/stddev/max from `--inner-iters` warm runs of each path.
//! The Python harness `scripts/bench_mel_gemm.py` orchestrates 5
//! fresh-process runs across batches of 16, 64, ~199 frames per segment.
//!
//! Usage:
//!     cargo run --release --example bench_mel_gemm -- \
//!         --batch-segments 64 --frames-per-segment 90 --inner-iters 50

use std::time::Instant;

use sparrow_engine_core::preprocess_audio::AudioPreprocessConfig;
use sparrow_engine::audio::hann::upload_mel_filterbank;
use sparrow_engine::audio::mel_gemm::{cpu_mel_gemm_row_major, MelGemm};
use cudarc::driver::CudaContext;

const N_MELS: usize = 224;
const N_FREQS: usize = 1025;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    if let Some(pos) = args.iter().position(|a| a == flag) {
        if let Some(val) = args.get(pos + 1) {
            if let Ok(parsed) = val.parse() {
                return parsed;
            }
        }
    }
    default
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
        out.push(f * 4.0);
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
    let args: Vec<String> = std::env::args().collect();
    let batch_segments: usize = parse_arg(&args, "--batch-segments", 64);
    let frames_per_segment: usize = parse_arg(&args, "--frames-per-segment", 90);
    let inner_iters: usize = parse_arg(&args, "--inner-iters", 50);
    let warmup: usize = parse_arg(&args, "--warmup", 5);
    let do_cpu: bool = !args.iter().any(|a| a == "--no-cpu");

    let total_frames = batch_segments * frames_per_segment;

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let config = AudioPreprocessConfig::default();

    let fb_d = upload_mel_filterbank(&stream, &config).expect("upload_mel_filterbank");
    let fb_host: Vec<f32> = stream.clone_dtoh(&fb_d.data).expect("clone_dtoh fb");

    let power_host = lcg_rand_vec(0xBEEF_F00D, total_frames * N_FREQS);
    let power_d = stream.clone_htod(&power_host).expect("clone_htod power");

    let mut mel_d = stream
        .alloc_zeros::<f32>(N_MELS * total_frames)
        .expect("alloc mel_out");

    let gemm = MelGemm::new(stream.clone(), N_MELS, N_FREQS).expect("MelGemm::new");

    // GPU warmup.
    for _ in 0..warmup {
        gemm.run(&fb_d.data, &power_d, &mut mel_d, total_frames)
            .expect("warmup run");
    }
    stream.synchronize().expect("warmup sync");

    let mut gpu_ms = Vec::with_capacity(inner_iters);
    for _ in 0..inner_iters {
        let t0 = Instant::now();
        gemm.run(&fb_d.data, &power_d, &mut mel_d, total_frames)
            .expect("inner run");
        stream.synchronize().expect("inner sync");
        gpu_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    let gpu_p50 = median(&gpu_ms);
    let gpu_p95 = percentile(&gpu_ms, 95.0);
    let gpu_stddev = stddev(&gpu_ms);
    let gpu_max = gpu_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    // CPU baseline (single iteration to keep the bench tractable; the CPU
    // path is deterministic so a single iter is representative within a
    // few %, and Wave 0 already established the variance distribution).
    let mut cpu_p50 = -1.0_f64;
    let mut cpu_max = -1.0_f64;
    let mut cpu_stddev = -1.0_f64;
    if do_cpu {
        // CPU iters: fewer than GPU because each takes ~150 ms+. 3 is
        // enough for median + max bounded by Wave 0's measured baseline stddev.
        let cpu_iters = 3.min(inner_iters);
        let mut cpu_ms = Vec::with_capacity(cpu_iters);
        for _ in 0..cpu_iters {
            let t0 = Instant::now();
            let _ = cpu_mel_gemm_row_major(&fb_host, &power_host, N_MELS, N_FREQS, total_frames);
            cpu_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }
        cpu_p50 = median(&cpu_ms);
        cpu_max = cpu_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        cpu_stddev = stddev(&cpu_ms);
    }

    println!(
        "{{\"primitive\":\"mel_gemm\",\"batch_segments\":{batch_segments},\
         \"frames_per_segment\":{frames_per_segment},\"total_frames\":{total_frames},\
         \"gpu_p50_ms\":{gpu_p50},\"gpu_p95_ms\":{gpu_p95},\"gpu_stddev_ms\":{gpu_stddev},\"gpu_max_ms\":{gpu_max},\
         \"cpu_p50_ms\":{cpu_p50},\"cpu_max_ms\":{cpu_max},\"cpu_stddev_ms\":{cpu_stddev},\
         \"inner_iters\":{inner_iters},\"warmup\":{warmup}}}"
    );
}
