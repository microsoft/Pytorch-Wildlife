//! W1.4 parity test — `power_to_db` kernel vs CPU `sparrow_engine_core::preprocess_audio::power_to_db`.
//!
//! Gate: max-abs Δ ≤ 5e-3 dB on post-mel-GEMM data
//! (`docs/design/phase3.8/step2/round_02/arch-perf_proposal_r2.md §R2.1` G0c).
//!
//! Inputs are the GPU mel output from W1.3 (cuBLAS sgemm), reshaped to
//! the `[n_segments * n_mels * n_frames]` slab the kernel expects.
//! Compare against the CPU reference applied per-segment (matches
//! `sparrow_engine_core::preprocess_audio::power_to_db` semantics).


use sparrow_engine_core::preprocess_audio::AudioPreprocessConfig;
use sparrow_engine::audio::hann::upload_mel_filterbank;
use sparrow_engine::audio::mel_gemm::MelGemm;
use sparrow_engine::audio::power_to_db::{cpu_power_to_db, power_to_db_gpu, PowerToDbKernel};
use cudarc::driver::CudaContext;

const N_MELS: usize = 224;
const N_FREQS: usize = 1025;
const FRAMES_PER_SEGMENT: usize = 90;
const TOP_DB: f32 = 80.0;
const EPSILON: f32 = 5e-3;

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
        // Mel-pow magnitude range (post-cuBLAS): roughly [0, 4]; the
        // post-power_to_db output range is [-TOP_DB, +max_dB].
        out.push(f * 4.0);
    }
    out
}

#[test]
fn power_to_db_parity_post_mel_gemm() {
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let stream = ctx.default_stream();
    let config = AudioPreprocessConfig::default();

    // Test on a 16-segment batch (matches DEFAULT_BATCH_SIZE).
    let n_segments = 16usize;
    let total_frames = n_segments * FRAMES_PER_SEGMENT;

    // 1. Set up mel input via W1.3 (cuBLAS sgemm) so the per-segment
    //    layout matches the production pipeline. Random power input, then
    //    GEMM produces the mel slab.
    let fb_d = upload_mel_filterbank(&stream, &config).expect("upload_mel_filterbank");
    let power_host = lcg_rand_vec(0x4242, total_frames * N_FREQS);
    let power_d = stream.clone_htod(&power_host).expect("clone_htod power");
    let mut mel_col = stream
        .alloc_zeros::<f32>(N_MELS * total_frames)
        .expect("alloc mel");
    let gemm = MelGemm::new(stream.clone(), N_MELS, N_FREQS).expect("MelGemm::new");
    gemm.run(&fb_d.data, &power_d, &mut mel_col, total_frames)
        .expect("MelGemm::run");

    // 2. The GPU mel layout is column-major [n_mels, total_frames]:
    //    `mel[m, t] = mel_col[m + N_MELS * t]`. The `power_to_db` kernel
    //    operates on contiguous per-segment slabs of `n_mels *
    //    frames_per_segment`. The current column-major layout makes a
    //    segment's slab the `frames_per_segment` columns at offset
    //    `seg_start * N_MELS .. (seg_start + frames_per_segment) * N_MELS`,
    //    which IS contiguous (each column is `n_mels` elements), so the
    //    kernel reads the right data.
    //
    //    For the CPU reference we'll work on the same column-major layout:
    //    DtoH the column-major mel buffer, slice per segment, run the
    //    CPU power_to_db on each slab.
    let mel_col_host: Vec<f32> = stream.clone_dtoh(&mel_col).expect("clone_dtoh mel");
    let mut cpu_dbg = mel_col_host.clone();
    let seg_size = N_MELS * FRAMES_PER_SEGMENT;
    for seg in 0..n_segments {
        let slab = &mut cpu_dbg[seg * seg_size..(seg + 1) * seg_size];
        cpu_power_to_db(slab, TOP_DB);
    }

    // 3. Run the GPU kernel.
    let kernel = PowerToDbKernel::new(&ctx).expect("PowerToDbKernel::new");
    power_to_db_gpu(
        &stream,
        &kernel,
        &mut mel_col,
        n_segments,
        N_MELS,
        FRAMES_PER_SEGMENT,
        TOP_DB,
    )
    .expect("power_to_db_gpu");
    stream.synchronize().expect("synchronize");

    let gpu_dbg: Vec<f32> = stream.clone_dtoh(&mel_col).expect("clone_dtoh out");

    // 4. Compare.
    assert_eq!(gpu_dbg.len(), cpu_dbg.len());
    let mut max_abs = 0.0f32;
    let mut max_idx = 0usize;
    for (i, (g, c)) in gpu_dbg.iter().zip(cpu_dbg.iter()).enumerate() {
        let d = (g - c).abs();
        if d > max_abs {
            max_abs = d;
            max_idx = i;
        }
    }

    eprintln!(
        "power_to_db parity (post-mel-GEMM, {n_segments} × {N_MELS} × {FRAMES_PER_SEGMENT}): \
         max-abs Δ = {max_abs:.3e} dB at i={max_idx} \
         (gpu={}, cpu={}); gate = {EPSILON:.0e} dB",
        gpu_dbg[max_idx], cpu_dbg[max_idx]
    );

    if max_abs > EPSILON {
        panic!(
            "G0c gate EXCEEDED: max-abs Δ = {max_abs:.3e} > {EPSILON:.0e} dB \
             at i={max_idx} (gpu={}, cpu={}). STOP — do not commit. \
             Hypothesised cause: __log10f (CUDA libdevice) vs f32::log10 \
             (Rust libm) drift up to ~2 ULP, plus block-reduction max vs serial \
             max ordering at FP32 (max is bit-exact at FP32 — the only divergence \
             must come from log10). Diagnostic plan: snapshot per-bin Δ before \
             vs after the per-segment max reduction; if drift is purely in the \
             log10 step, log10 is the named root cause.",
            gpu_dbg[max_idx], cpu_dbg[max_idx]
        );
    }
}
