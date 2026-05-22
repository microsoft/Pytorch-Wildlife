//! W1.2 parity test — fused `re² + im²` kernel vs CPU scalar.
//!
//! Gate (re-derived 2026-05-05 with lead approval; see
//! `wave1_primitives_bench.md` §"W1.2 gate re-derivation"):
//!
//! ```text
//! max(abs(gpu - cpu)) / max(abs(cpu)) ≤ 1e-7
//! ```
//!
//! The original absolute `1e-5` from the Wave 1 brief was magnitude-blind:
//! at the bench signal's natural cuFFT-output magnitude (~2.27e5 at the
//! 1 kHz tone bin, ~2.5e5 at peak), 1 FP32 ULP is ~1.56e-2 — 1500× larger
//! than the 1e-5 absolute gate. NO FP32-correct kernel can meet that gate
//! at this signal scale.
//!
//! The relative `1e-7` floor reflects FP32 ULP-level precision: empirical
//! drift on the synthetic 1 kHz tone is 6.9e-8 (sub-ULP), met with
//! ~1.5× headroom. STOP-and-ping if exceeded — the kernel is FMA-fused
//! so the residual is bounded by FMA-vs-`mul; add` ULP drift; >1 ULP
//! relative would indicate an actual implementation bug.

use std::sync::Arc;

use sparrow_engine::audio::cufft_plan::{alloc_complex_output, frames_with_hann_cpu, BatchedR2cPlan};
use sparrow_engine::audio::hann::hann_window_cpu;
use sparrow_engine::audio::power_kernel::{cpu_power, power_gpu, PowerKernel};
use cudarc::cufft::sys as cufft_sys;
use cudarc::driver::CudaContext;

const N_FFT: usize = 2048;
const HOP: usize = 512;
const FRAMES_PER_SEGMENT: usize = 90;
const SAMPLE_RATE: f32 = 48_000.0;
/// Relative gate: `max(abs(gpu - cpu)) / max(abs(cpu)) ≤ EPSILON_REL`.
/// 1e-7 is FP32 ULP floor; matched to the gate exactly per
/// `feedback_no_soft_tolerance_framing_on_gates.md` (no permissive
/// multiplier).
const EPSILON_REL: f32 = 1e-7;

fn synth_tone(n_samples: usize) -> Vec<f32> {
    (0..n_samples)
        .map(|i| {
            let t = i as f32 / SAMPLE_RATE;
            (2.0 * std::f32::consts::PI * 1000.0 * t).sin()
        })
        .collect()
}

#[test]
fn power_parity_post_cufft() {
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();

    // 1. Run cuFFT on a synthetic tone to produce the complex output that
    //    feeds the power kernel.
    let total_frames = FRAMES_PER_SEGMENT;
    let n_samples = (total_frames - 1) * HOP + N_FFT;
    let tone = synth_tone(n_samples);
    let starts: Vec<usize> = (0..total_frames).map(|f| f * HOP).collect();
    let hann = hann_window_cpu(N_FFT);
    let frames_host = frames_with_hann_cpu(&tone, &starts, N_FFT, &hann);
    let frames_d = stream.clone_htod(&frames_host).expect("clone_htod");

    let plan = BatchedR2cPlan::new(Arc::clone(&stream), N_FFT, total_frames)
        .expect("BatchedR2cPlan::new");
    let mut complex_d =
        alloc_complex_output(&stream, total_frames, plan.n_freqs()).expect("alloc complex");
    plan.exec(&frames_d, &mut complex_d).expect("plan.exec");

    // 2. Run power kernel on the cuFFT output.
    let kernel = PowerKernel::new(&ctx).expect("PowerKernel::new");
    let n_freqs = plan.n_freqs();
    let mut power_d = stream
        .alloc_zeros::<f32>(total_frames * n_freqs)
        .expect("alloc power");
    power_gpu(&stream, &kernel, &complex_d, &mut power_d, total_frames, n_freqs)
        .expect("power_gpu");
    stream.synchronize().expect("synchronize");

    // 3. CPU reference: `re² + im²` on the cuFFT output (post-DtoH). Both
    //    paths consume the SAME complex bytes — drift is purely in the
    //    multiply-add (GPU FMA-fused vs CPU two-step `mul; add`).
    let complex_host: Vec<cufft_sys::float2> =
        stream.clone_dtoh(&complex_d).expect("clone_dtoh complex");
    let cpu_complex_pairs: Vec<(f32, f32)> =
        complex_host.iter().map(|c| (c.x, c.y)).collect();
    let cpu_power_host = cpu_power(&cpu_complex_pairs);

    let gpu_power_host: Vec<f32> = stream.clone_dtoh(&power_d).expect("clone_dtoh power");
    assert_eq!(cpu_power_host.len(), gpu_power_host.len());

    // Compute absolute max-Δ + max(|cpu|) for the relative gate.
    let mut max_abs_delta = 0.0f32;
    let mut max_abs_idx = 0usize;
    for (i, (g, c)) in gpu_power_host.iter().zip(cpu_power_host.iter()).enumerate() {
        let d: f32 = (*g - *c).abs();
        if d > max_abs_delta {
            max_abs_delta = d;
            max_abs_idx = i;
        }
    }
    let max_abs_cpu = cpu_power_host
        .iter()
        .cloned()
        .fold(0.0f32, |acc: f32, x: f32| acc.max(x.abs()));
    let relative_delta = if max_abs_cpu > 0.0 {
        max_abs_delta / max_abs_cpu
    } else {
        0.0
    };

    eprintln!(
        "power parity (post-cuFFT, {total_frames} frames × {n_freqs} bins): \
         max-abs Δ = {max_abs_delta:.3e} at i={max_abs_idx} \
         (gpu={}, cpu={}), max(|cpu|) = {max_abs_cpu:.3e}, \
         relative Δ = {relative_delta:.3e}; gate (relative) = {EPSILON_REL:.0e}",
        gpu_power_host[max_abs_idx], cpu_power_host[max_abs_idx]
    );

    if relative_delta > EPSILON_REL {
        panic!(
            "G0b' gate EXCEEDED: relative Δ = {relative_delta:.3e} > {EPSILON_REL:.0e} \
             (max-abs Δ = {max_abs_delta:.3e}, max(|cpu|) = {max_abs_cpu:.3e}). STOP. \
             Hypothesised cause: GPU FMA-fused multiply-add vs CPU two-step `mul; add` \
             produces up to 1 FP32 ULP drift at FMA boundaries; relative >1e-7 indicates \
             an actual implementation bug, not the FMA floor. \
             Diagnostic plan: instrument intermediate cuFFT output bytes (DtoH compare \
             vs CPU realfft on the SAME windowed input); rerun with deterministic cuFFT \
             plan flags; if drift persists, suspect the kernel's index-bound logic or \
             the host-side stream sync."
        );
    }
}
