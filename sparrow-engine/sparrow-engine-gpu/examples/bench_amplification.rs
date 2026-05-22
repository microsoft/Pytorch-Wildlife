//! W1.7 mel-Δ-to-logit-Δ amplification bench. Sets the production
//! G-D logit gate stringency per `arch-prag_proposal_r2.md §D.2`.
//!
//! For each ε ∈ {1e-4, 1e-3, 1e-2, 1e-1} dB:
//!   1. Generate baseline mel `M` (deterministic SplitMix64 random).
//!   2. Perturb: `M' = M + ε * random_uniform(-1, 1)`.
//!   3. Run `M` and `M'` through the same MD_AudioBirds_V1.onnx session.
//!   4. Record max-abs Δ in pre-sigmoid logits; compute amplification
//!      factor = max-abs Δ logit / ε.
//!
//! Decision rule (per Wave 1 brief Tier C):
//!   - Amplification < 5× → G-D = 1e-4 (tight)
//!   - 5×–10×          → G-D = 5e-4 (current arch-par/arch-prag converged)
//!   - > 10×           → tighten G-C dB gate to compensate; ML-quality investigation.

use std::path::PathBuf;
use std::sync::Arc;

use sparrow_engine::audio::ort_io::AudioOrtSession;
use cudarc::driver::CudaContext;

const N_MELS: usize = 224;
const TIME_STEPS: usize = 90;
const BATCH: usize = 16;

fn parse_str_arg(args: &[String], flag: &str) -> Option<String> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|p| args.get(p + 1))
        .cloned()
}

fn lcg_rand_vec(seed: u64, n: usize, lo: f32, hi: f32) -> Vec<f32> {
    let mut state = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut out = Vec::with_capacity(n);
    let span = hi - lo;
    for _ in 0..n {
        state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        let f = (z >> 40) as f32 / (1u64 << 24) as f32;
        out.push(lo + f * span);
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = PathBuf::from(parse_str_arg(&args, "--model").unwrap_or_else(|| {
        "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models_test/md-audiobirds-v1/MD_AudioBirds_V1.onnx".to_string()
    }));
    if !model_path.exists() {
        panic!("model not found at {model_path:?}");
    }

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    // Phase 3.8 Step 2 perf-fix Fix D: AudioOrtSession is bound to a
    // dedicated non-default stream.
    let stream = ctx.new_stream().expect("ctx.new_stream");
    let session =
        AudioOrtSession::load(&ctx, &stream, &model_path).expect("AudioOrtSession::load");

    // Baseline mel: production-range [-80, 0] dB.
    let total = BATCH * N_MELS * TIME_STEPS;
    let mel_baseline = lcg_rand_vec(0xBA5E_BA11, total, -80.0, 0.0);

    // Three repeats per ε for stddev across perturbation seeds.
    let epsilons: [f32; 4] = [1e-4, 1e-3, 1e-2, 1e-1];
    let perturbation_seeds: [u64; 3] = [0x1111, 0x2222, 0x3333];

    let mut results: Vec<(f32, Vec<f32>)> = Vec::new();
    for &eps in &epsilons {
        let mut amp_per_seed = Vec::with_capacity(perturbation_seeds.len());
        for &seed in &perturbation_seeds {
            // Perturbation: ε × uniform(-1, 1) in dB.
            let pert = lcg_rand_vec(seed, total, -1.0, 1.0);
            let mel_perturbed: Vec<f32> = mel_baseline
                .iter()
                .zip(pert.iter())
                .map(|(b, p)| b + eps * p)
                .collect();

            let mel_d_base = stream.clone_htod(&mel_baseline).expect("htod base");
            let logits_base = session
                .run_iobinding(&Arc::clone(&stream), &mel_d_base, BATCH, N_MELS, TIME_STEPS)
                .expect("base inference");

            let mel_d_pert = stream.clone_htod(&mel_perturbed).expect("htod pert");
            let logits_pert = session
                .run_iobinding(&Arc::clone(&stream), &mel_d_pert, BATCH, N_MELS, TIME_STEPS)
                .expect("pert inference");

            let max_logit_delta: f32 = logits_base
                .iter()
                .zip(logits_pert.iter())
                .map(|(a, b): (&f32, &f32)| (*a - *b).abs())
                .fold(0.0_f32, f32::max);
            let amp = max_logit_delta / eps;
            amp_per_seed.push(amp);
        }
        results.push((eps, amp_per_seed));
    }

    // Print as CSV-ish JSON.
    print!("{{\"primitive\":\"amplification\",\"batch\":{BATCH},\"results\":[");
    for (i, (eps, amps)) in results.iter().enumerate() {
        let med = {
            let mut v = amps.clone();
            v.sort_by(|a, b| a.partial_cmp(b).unwrap());
            v[v.len() / 2]
        };
        let mn = amps.iter().cloned().fold(f32::INFINITY, f32::min);
        let mx = amps.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        if i > 0 {
            print!(",");
        }
        print!(
            "{{\"epsilon_db\":{eps},\"amp_factor_median\":{med},\"amp_factor_min\":{mn},\
             \"amp_factor_max\":{mx},\"n_seeds\":{}}}",
            amps.len()
        );
    }
    println!("]}}");
}
