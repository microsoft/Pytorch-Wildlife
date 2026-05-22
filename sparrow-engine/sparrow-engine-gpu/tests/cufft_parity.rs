//! W1.1 parity test — cuFFT R2C batched plan vs realfft (RustFFT-backed)
//! on a deterministic 1 kHz tone at n_fft = 2048.
//!
//! Gate: max-abs Δ ≤ 2e-4 in complex output magnitude
//! (`docs/design/phase3.8/step2/round_02/arch-perf_proposal_r2.md §R2.1`
//! G0a; ULP derivation in arch-par §4: ~5.3 ULP at n=2048 from
//! Cooley-Tukey √N scaling on the n=512 NVIDIA-forum 2.65 ULP datapoint.
//! 5.3 ULP × FP32 magnitude ≈ 2e-4 at unit-amplitude input).
//!
//! Test signal: 1 kHz sine wave at 48 kHz sample rate, Hann-windowed,
//! n_fft = 2048. The realfft and cuFFT outputs are both unnormalized R2C
//! (per `arch-par_proposal_r2.md §5` — RustFFT is its own implementation
//! but converges to the same unnormalized DFT as cuFFT).

use std::sync::Arc;

use sparrow_engine::audio::cufft_plan::{alloc_complex_output, frames_with_hann_cpu, BatchedR2cPlan};
use sparrow_engine::audio::hann::hann_window_cpu;
use cudarc::cufft::sys as cufft_sys;
use cudarc::driver::CudaContext;
use realfft::RealFftPlanner;

const N_FFT: usize = 2048;
const HOP: usize = 512;
const SAMPLE_RATE: f32 = 48_000.0;
const TONE_HZ: f32 = 1_000.0;
const EPSILON: f32 = 2e-4;

/// Generate a deterministic 1 kHz tone at 48 kHz sample rate, length
/// `n_samples`. Amplitude 1.0 so the tone exercises the unit-magnitude
/// regime the gate is calibrated for.
fn synth_tone(n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            (2.0 * std::f32::consts::PI * TONE_HZ * t).sin()
        })
        .collect()
}

/// CPU realfft on each frame; returns row-major `[total_frames * n_freqs]`
/// complex `(re, im)` pairs.
fn cpu_realfft(input_frames: &[f32], n_fft: usize, total_frames: usize) -> Vec<(f32, f32)> {
    let n_freqs = n_fft / 2 + 1;
    let mut planner = RealFftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(n_fft);
    let mut buf = fft.make_input_vec();
    let mut out_complex = fft.make_output_vec();
    let mut all = Vec::with_capacity(total_frames * n_freqs);
    for f in 0..total_frames {
        buf.copy_from_slice(&input_frames[f * n_fft..(f + 1) * n_fft]);
        fft.process(&mut buf, &mut out_complex).expect("realfft.process");
        for c in &out_complex {
            all.push((c.re, c.im));
        }
    }
    all
}

#[test]
fn cufft_r2c_parity_single_frame_1khz_tone() {
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    // 90 frames × hop=512 + n_fft=2048 = 47,648 samples needed; round up.
    let total_frames = 90usize;
    let n_samples = (total_frames - 1) * HOP + N_FFT;
    let tone = synth_tone(n_samples);
    let starts: Vec<usize> = (0..total_frames).map(|f| f * HOP).collect();
    let hann = hann_window_cpu(N_FFT);

    let frames_host = frames_with_hann_cpu(&tone, &starts, N_FFT, &hann);

    // CPU reference.
    let cpu_complex = cpu_realfft(&frames_host, N_FFT, total_frames);

    // GPU.
    let frames_d = stream
        .clone_htod(&frames_host)
        .expect("clone_htod frames");
    let plan = BatchedR2cPlan::new(Arc::clone(&stream), N_FFT, total_frames)
        .expect("BatchedR2cPlan::new");
    let mut output_d =
        alloc_complex_output(&stream, total_frames, plan.n_freqs()).expect("alloc complex out");
    plan.exec(&frames_d, &mut output_d).expect("plan.exec");
    stream.synchronize().expect("synchronize");
    let gpu_complex_raw: Vec<cufft_sys::float2> =
        stream.clone_dtoh(&output_d).expect("clone_dtoh complex");

    // Convert to (re, im) pairs.
    let gpu_complex: Vec<(f32, f32)> = gpu_complex_raw.iter().map(|c| (c.x, c.y)).collect();

    assert_eq!(cpu_complex.len(), gpu_complex.len(), "length mismatch");

    // Compare per-bin magnitudes (the gate is on complex magnitude per
    // arch-perf_proposal_r2.md §R2.1 G0a).
    let mut max_abs_mag = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (cpu_c, gpu_c)) in cpu_complex.iter().zip(gpu_complex.iter()).enumerate() {
        let cpu_mag = (cpu_c.0 * cpu_c.0 + cpu_c.1 * cpu_c.1).sqrt();
        let gpu_mag = (gpu_c.0 * gpu_c.0 + gpu_c.1 * gpu_c.1).sqrt();
        let d = (cpu_mag - gpu_mag).abs();
        if d > max_abs_mag {
            max_abs_mag = d;
            max_idx = i;
        }
    }

    eprintln!(
        "cuFFT parity (1 kHz tone, {total_frames} frames, n_fft={N_FFT}): \
         max-abs Δ in complex magnitude = {max_abs_mag:.3e} at i={max_idx}; \
         gate = {EPSILON:.0e}"
    );

    if max_abs_mag > EPSILON {
        panic!(
            "G0a gate EXCEEDED: max-abs complex-magnitude Δ = {max_abs_mag:.3e} > {EPSILON:.0e} \
             at i={max_idx}. STOP — do not commit. Hypothesised cause: cuFFT vs realfft \
             butterfly accumulation order at n=2048 produces drift exceeding the 5.3 ULP \
             extrapolation. Diagnostic plan: bin-by-bin histogram of per-bin Δ; check whether \
             drift concentrates at the tone bin (algorithm divergence) or spreads across all \
             bins (accumulation-order divergence). If accumulation-order: the gate is calibrated \
             correctly and a higher empirical ULP is the finding to report; do NOT loosen \
             the gate without lead OK."
        );
    }
}
