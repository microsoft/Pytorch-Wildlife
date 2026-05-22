//! W1.5 parity test — ORT CUDA EP IoBinding (`run_iobinding`) vs the
//! legacy host-roundtrip path (`run_host_roundtrip`) on the live
//! `MD_AudioBirds_V1.onnx` model.
//!
//! Gate: max-abs Δ on logits = 0.0 (`docs/design/phase3.8/step2/round_02/
//! arch-perf_proposal_r2.md §R2.1` G0e — "bind-once vs bind-per-call must
//! produce bit-exact logits"). Both paths feed ORT identical bytes;
//! divergence implies a binding-side bug or non-determinism in CUDA EP
//! that needs investigation BEFORE relying on IoBinding for production
//! perf. Hard STOP on exceeded.
//!
//! The test resolves the model from the same `sparrow_engine_models_test` tree as
//! the Step 1 image-model integration tests (`test_files/sparrow_engine_models_test`).
//! Skipped if the model file is missing (CI without the test corpus).

use std::path::PathBuf;

use sparrow_engine::audio::ort_io::AudioOrtSession;
use cudarc::driver::CudaContext;

const N_MELS: usize = 224;
const TIME_STEPS: usize = 90;
const MODEL_RELATIVE: &str = "test_files/sparrow_engine_models_test/md-audiobirds-v1/MD_AudioBirds_V1.onnx";

/// Locate the ONNX model. Returns `Some(path)` if found; `None` if the
/// PW_refactor test_files corpus is not present (CI without corpus).
fn locate_model() -> Option<PathBuf> {
    let candidates = [
        "/home/miao/repos/PW_refactor/".to_string() + MODEL_RELATIVE,
        std::env::var("SPARROW_ENGINE_TEST_FILES_DIR")
            .map(|d| format!("{d}/sparrow_engine_models_test/md-audiobirds-v1/MD_AudioBirds_V1.onnx"))
            .unwrap_or_default(),
    ];
    for c in candidates.iter().filter(|s| !s.is_empty()) {
        let p = PathBuf::from(c);
        if p.exists() {
            return Some(p);
        }
    }
    None
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
        // Mel-dB-like distribution: roughly uniform in [-80, 0] dB, which
        // is the post-`power_to_db` range for the production manifest.
        let f = (z >> 40) as f32 / (1u64 << 24) as f32;
        out.push(f * 80.0 - 80.0);
    }
    out
}

#[test]
fn ort_io_bind_once_vs_host_roundtrip_bit_exact() {
    let model_path = match locate_model() {
        Some(p) => p,
        None => {
            eprintln!(
                "ort_io_parity: skipping — MD_AudioBirds_V1.onnx not found at \
                 /home/miao/repos/PW_refactor/{MODEL_RELATIVE} \
                 (set SPARROW_ENGINE_TEST_FILES_DIR to override)"
            );
            return;
        }
    };

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    // Phase 3.8 Step 2 perf-fix Fix D: AudioOrtSession is bound to a
    // dedicated non-default stream via `with_compute_stream`.
    let stream = ctx.new_stream().expect("ctx.new_stream");

    let session =
        AudioOrtSession::load(&ctx, &stream, &model_path).expect("AudioOrtSession::load");

    for &batch in &[1usize, 4, 16] {
        let total = batch * N_MELS * TIME_STEPS;
        let mel_host = lcg_rand_vec(0xA5A5_A5A5_A5A5_A5A5_u64.wrapping_add(batch as u64), total);
        let mel_d = stream
            .clone_htod(&mel_host)
            .expect("clone_htod mel input");

        let logits_iob = session
            .run_iobinding(&stream, &mel_d, batch, N_MELS, TIME_STEPS)
            .expect("run_iobinding");

        let logits_host = session
            .run_host_roundtrip(&stream, &mel_d, batch, N_MELS, TIME_STEPS)
            .expect("run_host_roundtrip");

        assert_eq!(
            logits_iob.len(),
            batch,
            "iobinding output length mismatch at batch={batch}"
        );
        assert_eq!(
            logits_host.len(),
            batch,
            "host_roundtrip output length mismatch at batch={batch}"
        );

        let mut max_abs = 0.0f32;
        let mut max_idx = 0usize;
        for (i, (a, b)) in logits_iob.iter().zip(logits_host.iter()).enumerate() {
            let d: f32 = (*a - *b).abs();
            if d > max_abs {
                max_abs = d;
                max_idx = i;
            }
        }

        eprintln!(
            "ort_io parity (batch={batch}): max-abs Δ = {max_abs:.6e} at i={max_idx} \
             (iobinding={}, host={})",
            logits_iob[max_idx], logits_host[max_idx]
        );

        if max_abs != 0.0 {
            panic!(
                "G0e gate EXCEEDED: bind-once vs bind-per-call max-abs Δ = {max_abs:.6e} ≠ 0.0 \
                 at batch={batch}, i={max_idx} (iobinding={}, host={}). STOP — do not commit. \
                 Hypothesised cause: ORT CUDA EP non-determinism between TensorRefMut::from_raw \
                 (device-pointer binding) and TensorRef::from_array_view (host-uploaded binding). \
                 Diagnostic plan: enable ORT verbose logging (ORT_LOG_LEVEL=Verbose), check whether \
                 the two paths exercise different cuDNN algos at the conv layers, and verify the \
                 device pointer's MemoryInfo correctly identifies AllocationDevice::CUDA + \
                 device_id matching the session EP.",
                logits_iob[max_idx], logits_host[max_idx]
            );
        }
    }
}
