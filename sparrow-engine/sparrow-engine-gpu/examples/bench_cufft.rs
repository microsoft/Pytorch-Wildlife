//! W1.1 bench helper: time cuFFT R2C batched plan vs realfft serial CPU
//! reference at varying total_frames. Single fresh-process run.
//!
//! Usage: `cargo run --release --example bench_cufft -- --batch-segments 64 --inner-iters 50`

use std::sync::Arc;
use std::time::Instant;

use sparrow_engine::audio::cufft_plan::{alloc_complex_output, frames_with_hann_cpu, BatchedR2cPlan};
use sparrow_engine::audio::hann::hann_window_cpu;
use cudarc::driver::CudaContext;
use realfft::RealFftPlanner;

const N_FFT: usize = 2048;
const HOP: usize = 512;
const FRAMES_PER_SEGMENT: usize = 90;
const SAMPLE_RATE: f32 = 48_000.0;

fn parse_arg<T: std::str::FromStr>(args: &[String], flag: &str, default: T) -> T {
    args.iter()
        .position(|a| a == flag)
        .and_then(|p| args.get(p + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn synth_tone(n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
        })
        .collect()
}

fn percentile(s: &[f64], pct: f64) -> f64 {
    let mut sorted = s.to_vec();
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
    let batch_segments: usize = parse_arg(&args, "--batch-segments", 64);
    let inner_iters: usize = parse_arg(&args, "--inner-iters", 50);
    let warmup: usize = parse_arg(&args, "--warmup", 5);

    let total_frames = batch_segments * FRAMES_PER_SEGMENT;
    let n_samples = (total_frames - 1) * HOP + N_FFT;

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    let tone = synth_tone(n_samples);
    let starts: Vec<usize> = (0..total_frames).map(|f| f * HOP).collect();
    let hann = hann_window_cpu(N_FFT);
    let frames_host = frames_with_hann_cpu(&tone, &starts, N_FFT, &hann);
    let frames_d = stream.clone_htod(&frames_host).expect("clone_htod");

    let plan = BatchedR2cPlan::new(Arc::clone(&stream), N_FFT, total_frames)
        .expect("BatchedR2cPlan::new");
    let mut output_d = alloc_complex_output(&stream, total_frames, plan.n_freqs())
        .expect("alloc complex out");

    for _ in 0..warmup {
        plan.exec(&frames_d, &mut output_d).expect("warmup exec");
    }
    stream.synchronize().expect("warmup sync");

    let mut gpu_ms = Vec::with_capacity(inner_iters);
    for _ in 0..inner_iters {
        let t0 = Instant::now();
        plan.exec(&frames_d, &mut output_d).expect("exec");
        stream.synchronize().expect("sync");
        gpu_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    // CPU realfft baseline.
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(N_FFT);
    let mut buf = fft.make_input_vec();
    let mut out_complex = fft.make_output_vec();

    let cpu_iters = 3.min(inner_iters);
    let mut cpu_ms = Vec::with_capacity(cpu_iters);
    for _ in 0..cpu_iters {
        let t0 = Instant::now();
        for f in 0..total_frames {
            buf.copy_from_slice(&frames_host[f * N_FFT..(f + 1) * N_FFT]);
            fft.process(&mut buf, &mut out_complex).expect("realfft");
        }
        cpu_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
    }

    println!(
        "{{\"primitive\":\"cufft\",\"batch_segments\":{batch_segments},\"total_frames\":{total_frames},\
         \"gpu_p50_ms\":{},\"gpu_p95_ms\":{},\"gpu_stddev_ms\":{},\"gpu_max_ms\":{},\
         \"cpu_p50_ms\":{},\"cpu_max_ms\":{},\"cpu_stddev_ms\":{},\
         \"inner_iters\":{inner_iters},\"warmup\":{warmup}}}",
        median(&gpu_ms),
        percentile(&gpu_ms, 95.0),
        stddev(&gpu_ms),
        gpu_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        median(&cpu_ms),
        cpu_ms.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        stddev(&cpu_ms),
    );
}
