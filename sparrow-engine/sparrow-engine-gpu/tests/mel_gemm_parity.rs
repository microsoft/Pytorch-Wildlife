//! W1.3 parity test — cuBLAS sgemm mel-filterbank GEMM vs CPU scalar
//! inner-product on the production-manifest shapes.
//!
//! Gate: max-abs Δ ≤ 5e-5 (`docs/design/phase3.8/step2/round_02/
//! arch-perf_proposal_r2.md §R2.1` G0b). Hard STOP on exceeded.
//!
//! Production-manifest shapes (`sparrow-engine/models/audiobirds.toml`):
//! - n_mels = 224
//! - n_freqs = n_fft / 2 + 1 = 1025
//! - frames_per_segment = 90
//!
//! The test runs the cuBLAS GEMM on a deterministic seed-controlled
//! random `f32` power-spectrum and compares vs `cpu_mel_gemm_row_major`.

use sparrow_engine_core::preprocess_audio::AudioPreprocessConfig;
use sparrow_engine::audio::hann::upload_mel_filterbank;
use sparrow_engine::audio::mel_gemm::{col_major_to_row_major, cpu_mel_gemm_row_major, MelGemm};
use cudarc::driver::CudaContext;

const N_MELS: usize = 224;
const N_FREQS: usize = 1025;
const FRAMES_PER_SEGMENT: usize = 90;
const EPSILON: f32 = 5e-5;

/// Deterministic linear-congruential PRNG so the parity test does not
/// depend on `rand` or any external crate.
fn lcg_rand_vec(seed: u64, n: usize) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        // SplitMix64 step (well-distributed, deterministic).
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // Map to [0, 1) f32 then to [0, 4) (positive-only; power values are
        // always ≥ 0 in the real pipeline).
        let f = (z >> 40) as f32 / (1u64 << 24) as f32;
        out.push(f * 4.0);
    }
    out
}

#[test]
fn mel_gemm_parity_single_segment() {
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let config = AudioPreprocessConfig::default();

    // 1. Upload the Slaney mel filterbank.
    let fb_d = upload_mel_filterbank(&stream, &config).expect("upload_mel_filterbank");
    let fb_host: Vec<f32> = stream.clone_dtoh(&fb_d.data).expect("clone_dtoh fb");

    // 2. Build a deterministic synthetic power spectrum
    //    [n_frames=90, n_freqs=1025] row-major.
    let total_frames = FRAMES_PER_SEGMENT;
    let power_host = lcg_rand_vec(0xDEAD_BEEF, total_frames * N_FREQS);
    let power_d = stream.clone_htod(&power_host).expect("clone_htod power");

    // 3. Allocate GPU output buffer for [n_mels=224, n_frames=90]
    //    column-major.
    let mut mel_d_col = stream
        .alloc_zeros::<f32>(N_MELS * total_frames)
        .expect("alloc mel_out");

    // 4. Run cuBLAS sgemm.
    let gemm = MelGemm::new(stream.clone(), N_MELS, N_FREQS).expect("MelGemm::new");
    gemm.run(&fb_d.data, &power_d, &mut mel_d_col, total_frames)
        .expect("MelGemm::run");
    stream.synchronize().expect("synchronize");

    // 5. DtoH GPU output and convert to row-major.
    let mel_d_col_host: Vec<f32> = stream.clone_dtoh(&mel_d_col).expect("clone_dtoh mel_col");
    let gpu_row = col_major_to_row_major(&mel_d_col_host, N_MELS, total_frames);

    // 6. CPU reference.
    let cpu_row = cpu_mel_gemm_row_major(&fb_host, &power_host, N_MELS, N_FREQS, total_frames);

    // 7. Max-abs delta.
    assert_eq!(gpu_row.len(), cpu_row.len(), "length mismatch");
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (g, c)) in gpu_row.iter().zip(cpu_row.iter()).enumerate() {
        let d: f32 = (*g - *c).abs();
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
    }

    eprintln!(
        "mel_gemm parity (single segment): max-abs Δ = {max_abs:.3e} at i={max_idx} \
         (gpu={}, cpu={}); gate = {EPSILON:.0e}",
        gpu_row[max_idx], cpu_row[max_idx]
    );

    if max_abs > EPSILON {
        panic!(
            "G0b gate EXCEEDED: max-abs Δ = {max_abs:.3e} > {EPSILON:.0e} at i={max_idx} \
             (gpu={}, cpu={}). STOP — do not commit. Hypothesised cause: \
             cuBLAS tile-accumulation order vs scalar serial accumulation. \
             Diagnostic plan: re-run with cuBLAS pointer mode = HOST + alpha/beta \
             at f32 zero/one literals; if drift persists check fb upload bit-exactness \
             via the W1.6 hann_filterbank parity test.",
            gpu_row[max_idx], cpu_row[max_idx]
        );
    }
}

#[test]
fn mel_gemm_parity_batch_of_16() {
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let config = AudioPreprocessConfig::default();

    // Same shape as one DEFAULT_BATCH_SIZE=16 batch from the CPU code.
    let batch = 16usize;
    let total_frames = batch * FRAMES_PER_SEGMENT;

    let fb_d = upload_mel_filterbank(&stream, &config).expect("upload_mel_filterbank");
    let fb_host: Vec<f32> = stream.clone_dtoh(&fb_d.data).expect("clone_dtoh fb");

    let power_host = lcg_rand_vec(0xCAFE_F00D, total_frames * N_FREQS);
    let power_d = stream.clone_htod(&power_host).expect("clone_htod power");

    let mut mel_d_col = stream
        .alloc_zeros::<f32>(N_MELS * total_frames)
        .expect("alloc mel_out");

    let gemm = MelGemm::new(stream.clone(), N_MELS, N_FREQS).expect("MelGemm::new");
    gemm.run(&fb_d.data, &power_d, &mut mel_d_col, total_frames)
        .expect("MelGemm::run");
    stream.synchronize().expect("synchronize");

    let mel_d_col_host: Vec<f32> = stream.clone_dtoh(&mel_d_col).expect("clone_dtoh mel_col");
    let gpu_row = col_major_to_row_major(&mel_d_col_host, N_MELS, total_frames);
    let cpu_row = cpu_mel_gemm_row_major(&fb_host, &power_host, N_MELS, N_FREQS, total_frames);

    let mut max_abs = 0.0f32;
    for (g, c) in gpu_row.iter().zip(cpu_row.iter()) {
        let d: f32 = (*g - *c).abs();
        if d > max_abs {
            max_abs = d;
        }
    }

    eprintln!(
        "mel_gemm parity (batch=16): max-abs Δ = {max_abs:.3e}; gate = {EPSILON:.0e}"
    );

    assert!(
        max_abs <= EPSILON,
        "G0b gate EXCEEDED on batch=16: max-abs Δ = {max_abs:.3e} > {EPSILON:.0e}"
    );
}

#[test]
fn mel_gemm_parity_whole_60s_clip() {
    // 60 s clip = 199 segments per the manifest (0.3 s stride, 1.0 s window).
    // We round to 200 to keep the bench bookkeeping simple — this exercises
    // the same per-frame GEMM math, just with a larger N. Wave 2's whole-clip
    // path will use exactly this layout.
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let config = AudioPreprocessConfig::default();

    let n_segments = 200usize;
    let total_frames = n_segments * FRAMES_PER_SEGMENT;

    let fb_d = upload_mel_filterbank(&stream, &config).expect("upload_mel_filterbank");
    let fb_host: Vec<f32> = stream.clone_dtoh(&fb_d.data).expect("clone_dtoh fb");

    let power_host = lcg_rand_vec(0x12345678, total_frames * N_FREQS);
    let power_d = stream.clone_htod(&power_host).expect("clone_htod power");

    let mut mel_d_col = stream
        .alloc_zeros::<f32>(N_MELS * total_frames)
        .expect("alloc mel_out");

    let gemm = MelGemm::new(stream.clone(), N_MELS, N_FREQS).expect("MelGemm::new");
    gemm.run(&fb_d.data, &power_d, &mut mel_d_col, total_frames)
        .expect("MelGemm::run");
    stream.synchronize().expect("synchronize");

    let mel_d_col_host: Vec<f32> = stream.clone_dtoh(&mel_d_col).expect("clone_dtoh mel_col");
    let gpu_row = col_major_to_row_major(&mel_d_col_host, N_MELS, total_frames);
    let cpu_row = cpu_mel_gemm_row_major(&fb_host, &power_host, N_MELS, N_FREQS, total_frames);

    let mut max_abs = 0.0f32;
    for (g, c) in gpu_row.iter().zip(cpu_row.iter()) {
        let d: f32 = (*g - *c).abs();
        if d > max_abs {
            max_abs = d;
        }
    }

    eprintln!(
        "mel_gemm parity (60 s, {n_segments} segments × {FRAMES_PER_SEGMENT} frames): \
         max-abs Δ = {max_abs:.3e}; gate = {EPSILON:.0e}"
    );

    assert!(
        max_abs <= EPSILON,
        "G0b gate EXCEEDED on whole-clip: max-abs Δ = {max_abs:.3e} > {EPSILON:.0e}"
    );
}
