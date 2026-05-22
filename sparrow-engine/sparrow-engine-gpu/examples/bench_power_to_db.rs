//! W1.4 bench helper: time the `power_to_db` kernel + CPU baseline.

use std::sync::Arc;
use std::time::Instant;

use sparrow_engine_core::preprocess_audio::AudioPreprocessConfig;
use sparrow_engine::audio::hann::upload_mel_filterbank;
use sparrow_engine::audio::mel_gemm::MelGemm;
use sparrow_engine::audio::power_to_db::{cpu_power_to_db, power_to_db_gpu, PowerToDbKernel};
use cudarc::driver::CudaContext;

const N_MELS: usize = 224;
const N_FREQS: usize = 1025;
const FRAMES_PER_SEGMENT: usize = 90;
const TOP_DB: f32 = 80.0;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter().position(|a| a == flag).and_then(|p| args.get(p + 1)).and_then(|v| v.parse().ok()).unwrap_or(default)
}
fn percentile(s: &[f64], pct: f64) -> f64 {
    let mut sorted = s.to_vec(); sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len(); let idx = ((pct / 100.0) * (n as f64 - 1.0)).round() as usize;
    sorted[idx.min(n - 1)]
}
fn median(s: &[f64]) -> f64 { let mut sorted = s.to_vec(); sorted.sort_by(|a, b| a.partial_cmp(b).unwrap()); let n = sorted.len(); if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) } }
fn stddev(s: &[f64]) -> f64 { if s.len() < 2 { return 0.0; } let mean = s.iter().sum::<f64>() / s.len() as f64; (s.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (s.len() - 1) as f64).sqrt() }

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

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_segments: usize = parse_arg(&args, "--n-segments", 64);
    let inner_iters: usize = parse_arg(&args, "--inner-iters", 50);
    let warmup: usize = parse_arg(&args, "--warmup", 5);

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let config = AudioPreprocessConfig::default();
    let total_frames = n_segments * FRAMES_PER_SEGMENT;

    // Set up mel input via cuBLAS GEMM so the bench has realistic post-GEMM data.
    let fb_d = upload_mel_filterbank(&stream, &config).expect("upload");
    let power_host = lcg_rand_vec(0xCAFE, total_frames * N_FREQS);
    let power_d = stream.clone_htod(&power_host).expect("clone_htod");
    let mut mel_d = stream.alloc_zeros::<f32>(N_MELS * total_frames).expect("alloc mel");
    let gemm = MelGemm::new(stream.clone(), N_MELS, N_FREQS).expect("MelGemm");
    gemm.run(&fb_d.data, &power_d, &mut mel_d, total_frames).expect("gemm");
    stream.synchronize().expect("sync gemm");

    // Snapshot the GEMM output so each bench iter starts from the same state
    // (the kernel mutates mel_d in place).
    let mel_seed: Vec<f32> = stream.clone_dtoh(&mel_d).expect("seed dtoh");

    let kernel = PowerToDbKernel::new(&ctx).expect("PowerToDbKernel");

    for _ in 0..warmup {
        let mut tmp = stream.clone_htod(&mel_seed).expect("warmup htod");
        power_to_db_gpu(&Arc::clone(&stream), &kernel, &mut tmp, n_segments, N_MELS, FRAMES_PER_SEGMENT, TOP_DB)
            .expect("warmup");
    }
    stream.synchronize().expect("warmup sync");

    let mut gpu_ms = Vec::with_capacity(inner_iters);
    for _ in 0..inner_iters {
        let mut tmp = stream.clone_htod(&mel_seed).expect("htod");
        stream.synchronize().expect("htod sync");
        let t0 = Instant::now();
        power_to_db_gpu(&Arc::clone(&stream), &kernel, &mut tmp, n_segments, N_MELS, FRAMES_PER_SEGMENT, TOP_DB)
            .expect("run");
        stream.synchronize().expect("sync");
        gpu_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // CPU reference: per-segment cpu_power_to_db on the seed.
    let cpu_iters = 3.min(inner_iters);
    let mut cpu_ms = Vec::with_capacity(cpu_iters);
    let seg_size = N_MELS * FRAMES_PER_SEGMENT;
    for _ in 0..cpu_iters {
        let mut buf = mel_seed.clone();
        let t0 = Instant::now();
        for s in 0..n_segments {
            cpu_power_to_db(&mut buf[s * seg_size..(s + 1) * seg_size], TOP_DB);
        }
        cpu_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    println!(
        "{{\"primitive\":\"power_to_db\",\"n_segments\":{n_segments},\"total_frames\":{total_frames},\
         \"gpu_p50_ms\":{},\"gpu_p95_ms\":{},\"gpu_stddev_ms\":{},\"gpu_max_ms\":{},\
         \"cpu_p50_ms\":{},\"cpu_max_ms\":{},\"cpu_stddev_ms\":{},\
         \"inner_iters\":{inner_iters},\"warmup\":{warmup}}}",
        median(&gpu_ms), percentile(&gpu_ms, 95.0), stddev(&gpu_ms),
        gpu_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        median(&cpu_ms),
        cpu_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        stddev(&cpu_ms),
    );
}
