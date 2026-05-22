//! Phase 3.8 Step 2 perf-triage — kernel-level profile of the audio pipeline.
//!
//! Captures per-stage timings emitted by `sparrow-engine-core::preprocess_audio`
//! and `sparrow-engine-gpu::models::audio` via tracing events. Runs the whole-clip
//! Strategy A pipeline `--inner-iters` times after `--warmup` warmup
//! iters and prints the per-stage median + p95.
//!
//! Stages collected (event names):
//! - `audio.decode`        (CPU; from preprocess_audio)
//! - `audio.resample`      (CPU; from preprocess_audio)
//! - `audio.gpu.h2d`       (single H2D for samples + frame_starts)
//! - `audio.gpu.mel`       (whole mel pipeline incl. allocs + 6 kernels + sync)
//! - `audio.gpu.ort`       (ORT IoBinding loop)
//! - `audio.gpu.post`      (sigmoid + threshold + collect)
//! - `total e2e`           (wall-clock around model.detect)
//!
//! Set `SPARROW_ENGINE_AUDIO_BENCH_MANIFEST` and `SPARROW_ENGINE_AUDIO_BENCH_FIXTURE`. Optional
//! `SPARROW_ENGINE_AUDIO_BENCH_INNER_ITERS` and `SPARROW_ENGINE_AUDIO_BENCH_WARMUP`.

use std::collections::HashMap;
use std::env;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use sparrow_engine::models::audio::{AudioModel, GpuAudioDetectOpts, Strategy};
use sparrow_engine_types::AudioInput;
use cudarc::driver::CudaContext;
use tracing::field::{Field, Visit};
use tracing::span::Attributes;
use tracing::subscriber::with_default;
use tracing::{Event, Id};
use tracing_subscriber::Layer;
use tracing_subscriber::layer::Context;
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::registry::LookupSpan;

/// Custom layer that picks up `stage = ".."` + `duration_ns = N` from each
/// `tracing::info!` event and accumulates ns per stage label.
#[derive(Default, Clone)]
struct StageCollector {
    by_stage: Arc<Mutex<HashMap<String, Vec<u64>>>>,
}

impl StageCollector {
    fn snapshot(&self) -> HashMap<String, Vec<u64>> {
        self.by_stage.lock().unwrap().clone()
    }
    fn clear(&self) {
        self.by_stage.lock().unwrap().clear();
    }
}

struct PairVisitor {
    stage: Option<String>,
    duration_ns: Option<u64>,
}

impl Visit for PairVisitor {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == "stage" {
            self.stage = Some(value.to_string());
        }
    }
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        if field.name() == "stage" {
            // Sometimes `stage = "..."` round-trips through Debug.
            let s = format!("{value:?}");
            self.stage = Some(s.trim_matches('"').to_string());
        }
    }
    fn record_u64(&mut self, field: &Field, value: u64) {
        if field.name() == "duration_ns" {
            self.duration_ns = Some(value);
        }
    }
    fn record_i64(&mut self, field: &Field, value: i64) {
        if field.name() == "duration_ns" && value >= 0 {
            self.duration_ns = Some(value as u64);
        }
    }
}

impl<S> Layer<S> for StageCollector
where
    S: tracing::Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, _attrs: &Attributes<'_>, _id: &Id, _ctx: Context<'_, S>) {}
    fn on_event(&self, event: &Event<'_>, _ctx: Context<'_, S>) {
        let mut v = PairVisitor { stage: None, duration_ns: None };
        event.record(&mut v);
        if let (Some(stage), Some(ns)) = (v.stage, v.duration_ns) {
            self.by_stage.lock().unwrap().entry(stage).or_default().push(ns);
        }
    }
}

fn env_var(key: &str) -> String {
    env::var(key).unwrap_or_else(|_| panic!("env var {key} required"))
}

fn env_var_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    match env::var(key) {
        Ok(v) => v.parse().unwrap_or(default),
        Err(_) => default,
    }
}

fn median_f64(s: &[f64]) -> f64 {
    let mut v = s.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 { return 0.0; }
    if n % 2 == 1 { v[n / 2] } else { 0.5 * (v[n / 2 - 1] + v[n / 2]) }
}

fn p95_f64(s: &[f64]) -> f64 {
    let mut v = s.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 { return 0.0; }
    let idx = ((0.95 * (n as f64 - 1.0)).round() as usize).min(n - 1);
    v[idx]
}

fn ns_to_ms_vec(v: &[u64]) -> Vec<f64> {
    v.iter().map(|&n| (n as f64) / 1_000_000.0).collect()
}

fn main() {
    let manifest_path = PathBuf::from(env_var("SPARROW_ENGINE_AUDIO_BENCH_MANIFEST"));
    let fixture = PathBuf::from(env_var("SPARROW_ENGINE_AUDIO_BENCH_FIXTURE"));
    let inner_iters: usize = env_var_or("SPARROW_ENGINE_AUDIO_BENCH_INNER_ITERS", 10);
    let warmup: usize = env_var_or("SPARROW_ENGINE_AUDIO_BENCH_WARMUP", 2);

    let collector = StageCollector::default();
    let subscriber = tracing_subscriber::registry().with(collector.clone());

    // Run the entire bench under our subscriber.
    with_default(subscriber, || {
        let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
        let model = AudioModel::load(&ctx, &manifest_path).expect("AudioModel::load");

        let ort_chunk: usize = env_var_or("SPARROW_ENGINE_PROFILE_ORT_CHUNK", 197);
        let opts = GpuAudioDetectOpts {
            base: Default::default(),
            strategy: Strategy::HybridA { ort_chunk_segments: ort_chunk },
        };
        eprintln!("Using ort_chunk_segments = {ort_chunk}");

        // Warmup (drop the events).
        for _ in 0..warmup {
            let _ = model.detect(
                &AudioInput::FilePath(fixture.clone()), &opts,
            ).expect("warmup");
        }
        collector.clear();

        // Timed iterations.
        let mut t_total_ms: Vec<f64> = Vec::with_capacity(inner_iters);
        for _ in 0..inner_iters {
            let t0 = Instant::now();
            let _res = model.detect(
                &AudioInput::FilePath(fixture.clone()), &opts,
            ).expect("detect");
            t_total_ms.push(t0.elapsed().as_secs_f64() * 1000.0);
        }

        let snap = collector.snapshot();
        let mut stages: Vec<String> = snap.keys().cloned().collect();
        stages.sort();

        // Print raw per-chunk samples if available.
        if let Some(chunks) = snap.get("audio.gpu.ort.chunk") {
            let ms_list = ns_to_ms_vec(chunks);
            eprintln!("\nRaw per-chunk ORT samples (ms): {ms_list:?}\n");
        }

        eprintln!();
        eprintln!("# Phase 3.8 Step 2 perf-triage — per-stage profile");
        eprintln!("Manifest : {}", manifest_path.display());
        eprintln!("Fixture  : {}", fixture.display());
        eprintln!("Iters    : {inner_iters} timed (after {warmup} warmup)");
        eprintln!();
        eprintln!("| Stage | n events | median (ms) | p95 (ms) | sum (ms) |");
        eprintln!("| --- | ---: | ---: | ---: | ---: |");
        for stage in &stages {
            let ns_list = &snap[stage];
            let ms_list = ns_to_ms_vec(ns_list);
            let med = median_f64(&ms_list);
            let p95 = p95_f64(&ms_list);
            let sum: f64 = ms_list.iter().sum();
            eprintln!(
                "| {} | {} | {:.4} | {:.4} | {:.4} |",
                stage, ms_list.len(), med, p95, sum,
            );
        }

        let total_med = median_f64(&t_total_ms);
        let total_p95 = p95_f64(&t_total_ms);
        eprintln!("| total e2e (model.detect) | {} | {:.4} | {:.4} | {:.4} |",
                 t_total_ms.len(), total_med, total_p95, t_total_ms.iter().sum::<f64>());

        // Compute the residual = total - sum(known stages PER ITER)
        // For per-iter accounting we need per-iter sums; assume stage events
        // fire in sequence, so sum of all events from one iter ~ sum of all
        // events / inner_iters when each event fires once per iter. Some
        // events (audio.gpu.ort) may fire MORE than once per iter (one per
        // ORT chunk in T<197 case); we're running T=197 so each fires once.
        // We compute total_summed = sum_of_per_event_medians and check
        // it accounts for total_med.
        let mut sum_of_medians_per_iter = 0.0f64;
        for stage in &stages {
            let ms_list = ns_to_ms_vec(&snap[stage]);
            sum_of_medians_per_iter += median_f64(&ms_list);
        }
        eprintln!();
        eprintln!("Sum of per-stage medians (each fires once per iter at T=whole): {:.3} ms",
                 sum_of_medians_per_iter);
        eprintln!("Total e2e median: {:.3} ms", total_med);
        eprintln!("Residual (total - stage-sum): {:.3} ms", total_med - sum_of_medians_per_iter);

        // JSON line
        print!("{{\"profile\":\"audio_e2e\",\"fixture\":\"{}\",\"inner_iters\":{},\"warmup\":{},\"total_e2e_ms_p50\":{:.6},\"total_e2e_ms_p95\":{:.6},\"stages\":{{",
              fixture.display(), inner_iters, warmup, total_med, total_p95);
        let mut first = true;
        for stage in &stages {
            let ms_list = ns_to_ms_vec(&snap[stage]);
            let med = median_f64(&ms_list);
            let p95 = p95_f64(&ms_list);
            if !first { print!(","); }
            first = false;
            print!("\"{}\":{{\"p50_ms\":{:.6},\"p95_ms\":{:.6},\"n\":{}}}",
                  stage, med, p95, ms_list.len());
        }
        println!("}}}}");
    });
}
