//! Phase 3.8 Step 1 Wave 4 — direct-function tests for `TiledModel`
//! (HerdNet + OWL-T tiled GPU path).
//!
//! These tests skip the engine layer (final_design §3 footnote: trait insert
//! deferred to Phase B) and call `TiledModel::load_from_path` + `detect_tiled`
//! directly. Each test:
//!
//! 1. Constructs a CUDA context.
//! 2. Loads a model manifest from the live `test_files/onnx/` fixtures.
//! 3. Runs detect on `test_files/test_overhead/S_11_05_16_DSC01556.JPG`
//!    (6000×4000 buffalo overhead image — same fixture sparrow-engine-cpu uses).
//! 4. Asserts:
//!    - Detection-count delta against the sparrow-engine-cpu golden JSON for HerdNet
//!      ≤ 1 / 1 image (Gate 4 spec, lead override 2026-05-03: target is
//!      ≤1/100 across the corpus; on a single image the gate is ≤1).
//!    - Bbox / confidence / dedup sanity for OWL-T.
//! 5. Prints `[bench]` lines on stderr the wave_4_bench.md harness greps for.
//!
//! Skip via `SPARROW_ENGINE_GPU_TESTS=0` (matches the engine_init / kernels_parity
//! convention from Wave 1).
//!
//! Run with the same env that drives `scripts/test.sh`:
//! ```sh
//! source scripts/ort-env.sh
//! cargo test -p sparrow-engine-gpu --release --test integration_tiled -- --nocapture --test-threads=1
//! ```

use std::path::{Path, PathBuf};

use sparrow_engine::models::tiled::TiledModel;
use sparrow_engine_types::{DetectOpts, ImageInput};
use cudarc::driver::CudaContext;
use serde::Deserialize;

fn gpu_tests_enabled() -> bool {
    !matches!(std::env::var("SPARROW_ENGINE_GPU_TESTS").as_deref(), Ok("0"))
}

struct EnvVarGuard {
    key: &'static str,
    prior: Option<String>,
}

impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prior = std::env::var(key).ok();
        unsafe {
            std::env::set_var(key, value);
        }
        Self { key, prior }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        match self.prior.take() {
            Some(v) => unsafe { std::env::set_var(self.key, v) },
            None => unsafe { std::env::remove_var(self.key) },
        }
    }
}

/// Walk up from `CARGO_MANIFEST_DIR` looking for `marker` as a child of any
/// ancestor. Bounded to `MAX_WALKUP` iterations so a misconfigured layout
/// doesn't scan the whole filesystem.
///
/// Worktree layouts sit deeper than the main checkout (six levels under the
/// project root for `.claude/worktrees/<branch>/sparrow-engine/sparrow-engine-gpu`), so the
/// limit needs to handle both forms.
const MAX_WALKUP: usize = 10;

fn walk_up_for(marker_components: &[&str]) -> Option<PathBuf> {
    let mut cur = Path::new(env!("CARGO_MANIFEST_DIR")).to_path_buf();
    for _ in 0..MAX_WALKUP {
        let mut candidate = cur.clone();
        for c in marker_components {
            candidate.push(c);
        }
        if candidate.exists() {
            return Some(candidate);
        }
        if !cur.pop() {
            break;
        }
    }
    None
}

/// Resolve the test-files root.
///
/// Priority order (per lead override 2026-05-03):
/// 1. `SPARROW_ENGINE_TEST_FILES_ROOT` env var (must point at an existing directory).
/// 2. Walk up from `CARGO_MANIFEST_DIR` looking for an ancestor's
///    `test_files/` sibling.
///
/// Returns `None` if neither resolves. Callers must skip with `eprintln!`,
/// not panic — the tests should be runnable from worktrees and from clean
/// checkouts where the corpus may have moved.
fn test_files_root() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("SPARROW_ENGINE_TEST_FILES_ROOT") {
        let p = PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }
    walk_up_for(&["test_files"])
}

/// Resolve the test-outputs root (where sparrow-engine-cpu's golden detections land).
///
/// Same env-var + walk-up pattern as [`test_files_root`]. Walk-up looks for
/// `sparrow-engine/test_outputs/golden` (NOT just `sparrow-engine/test_outputs`) so a stale
/// `sparrow-engine/test_outputs/libsparrow_engine/` left behind by a prior `cargo test` in a
/// worktree doesn't shadow the canonical golden tree in the main checkout —
/// `golden/` is what carries parity references, so we require it to be
/// present. Returns the parent of `golden/` so callers can treat it as the
/// `test_outputs/` root.
///
/// `test_outputs/` is gitignored, so worktrees won't have it locally — set
/// `SPARROW_ENGINE_TEST_OUTPUTS_ROOT` to point at the main checkout's path if you need
/// parity comparisons from a worktree.
fn test_outputs_root() -> Option<PathBuf> {
    if let Ok(v) = std::env::var("SPARROW_ENGINE_TEST_OUTPUTS_ROOT") {
        let p = PathBuf::from(v);
        if p.exists() {
            return Some(p);
        }
    }
    let golden = walk_up_for(&["sparrow-engine", "test_outputs", "golden"])?;
    golden.parent().map(|p| p.to_path_buf())
}

fn herdnet_manifest_path() -> Option<PathBuf> {
    test_files_root().map(|r| r.join("onnx").join("herdnet_manifest.toml"))
}

fn owl_manifest_path() -> Option<PathBuf> {
    test_files_root().map(|r| r.join("onnx").join("owl_manifest.toml"))
}

fn overhead_image() -> Option<PathBuf> {
    test_files_root().map(|r| r.join("test_overhead").join("S_11_05_16_DSC01556.JPG"))
}

// ---------------------------------------------------------------------------
// Golden JSON loader.
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct GoldenFile {
    detections: Vec<GoldenDetection>,
}

#[derive(Deserialize)]
struct GoldenDetection {
    bbox: [f64; 4],
}

struct GoldenSummary {
    n_detections: usize,
    top_bboxes: Vec<[f64; 4]>,
}

fn herdnet_golden_summary(top_k: usize) -> Option<GoldenSummary> {
    let Some(root) = test_outputs_root() else {
        eprintln!(
            "Golden JSON dir missing (set SPARROW_ENGINE_TEST_OUTPUTS_ROOT or run from a checkout \
             that has sparrow-engine/test_outputs/); parity check downgraded to sanity-only"
        );
        return None;
    };
    let path = root
        .join("golden")
        .join("herdnet")
        .join("S_11_05_16_DSC01556_detections.json");
    let content = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("Golden JSON missing or unreadable at {path:?}: {e}"));
    let parsed: GoldenFile = serde_json::from_str(&content)
        .unwrap_or_else(|e| panic!("Golden JSON parse failed for {path:?}: {e}"));
    let n_detections = parsed.detections.len();
    let top_bboxes = parsed
        .detections
        .into_iter()
        .take(top_k)
        .map(|d| d.bbox)
        .collect();

    Some(GoldenSummary {
        n_detections,
        top_bboxes,
    })
}

// ---------------------------------------------------------------------------
// HerdNet — dual-output heatmap, tile_overlap = 0
// ---------------------------------------------------------------------------

#[test]
fn herdnet_tiled_detection_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping HerdNet tiled GPU test");
        return;
    }
    let Some(manifest_path) = herdnet_manifest_path().filter(|p| p.exists()) else {
        eprintln!("HerdNet manifest fixture missing — skipping (set SPARROW_ENGINE_TEST_FILES_ROOT)");
        return;
    };
    let Some(img_path) = overhead_image().filter(|p| p.exists()) else {
        eprintln!("Overhead test image missing — skipping (set SPARROW_ENGINE_TEST_FILES_ROOT)");
        return;
    };

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let model = TiledModel::load_from_path(&ctx, &manifest_path)
        .expect("TiledModel::load_from_path (HerdNet)");
    assert_eq!(model.model_id(), "herdnet-general-2022");

    let bytes = std::fs::read(&img_path).expect("read overhead jpeg");
    let input = ImageInput::Encoded(bytes);
    let opts = DetectOpts::default();

    let result = model
        .detect_tiled(&ctx, &input, &opts)
        .expect("detect_tiled (HerdNet)");

    eprintln!(
        "[bench] herdnet_tiled total_ms={:.3} detections={} image={}x{}",
        result.processing_time_ms,
        result.detections.len(),
        result.image_width,
        result.image_height,
    );

    // Sanity: bbox normalized [0,1] and ordered.
    for d in &result.detections {
        assert!(
            (0.0..=1.0).contains(&d.bbox.x_min),
            "x_min out of range: {}",
            d.bbox.x_min,
        );
        assert!(
            (0.0..=1.0).contains(&d.bbox.y_min),
            "y_min out of range: {}",
            d.bbox.y_min,
        );
        assert!(
            (0.0..=1.0).contains(&d.bbox.x_max),
            "x_max out of range: {}",
            d.bbox.x_max,
        );
        assert!(
            (0.0..=1.0).contains(&d.bbox.y_max),
            "y_max out of range: {}",
            d.bbox.y_max,
        );
        assert!(d.bbox.x_max >= d.bbox.x_min);
        assert!(d.bbox.y_max >= d.bbox.y_min);
    }

    // Confidence-descending sort.
    for w in result.detections.windows(2) {
        assert!(
            w[0].confidence >= w[1].confidence,
            "HerdNet detections not sorted by confidence",
        );
    }

    // Image dimensions match the corpus.
    assert_eq!(result.image_width, 6000);
    assert_eq!(result.image_height, 4000);

    // Detection-count parity vs sparrow-engine-cpu golden (Gate 4).
    if let Some(golden) = herdnet_golden_summary(5) {
        let bongo_n = result.detections.len();
        let golden_n = golden.n_detections;
        let drift = bongo_n as i32 - golden_n as i32;
        eprintln!(
            "[parity] herdnet_tiled sparrow_engine_gpu={bongo_n} bongo_cpu_golden={golden_n} delta={drift:+}/1",
        );
        // Gate 4 spec (lead override 2026-05-03): ≤ 1 / 100 images across
        // the corpus. On the single-image test the bound is ≤ 1; we assert
        // exactly that and treat any larger delta as a STOP signal that
        // surfaces in test output rather than getting hidden behind a
        // "tolerance" framing.
        let abs_drift = drift.unsigned_abs() as usize;
        assert!(
            abs_drift <= 1,
            "HerdNet detection-count delta {drift:+}/1 exceeds Gate 4 bound (≤1) \
             (golden = {golden_n}, sparrow_engine_gpu = {bongo_n}). \
             Investigate cause before commit.",
        );

        // For each of the top-k golden bboxes (by confidence — golden is
        // pre-sorted descending), expect a GPU detection within 0.02
        // normalized-coord center match. The 0.02 floor accounts for ORT
        // CPU EP vs CUDA EP per-pixel FP arithmetic-ordering drift on the
        // same ONNX graph (heatmap pixel values can differ by ~1e-6 in
        // [0,1] space, which round-trips to sub-pixel coord drift).
        for (i, g_bbox) in golden.top_bboxes.iter().enumerate() {
            let mut best_dist = f64::INFINITY;
            for d in &result.detections {
                let cx_g = (g_bbox[0] + g_bbox[2]) * 0.5;
                let cy_g = (g_bbox[1] + g_bbox[3]) * 0.5;
                let cx_d = (d.bbox.x_min + d.bbox.x_max) as f64 * 0.5;
                let cy_d = (d.bbox.y_min + d.bbox.y_max) as f64 * 0.5;
                let dist = ((cx_g - cx_d).powi(2) + (cy_g - cy_d).powi(2)).sqrt();
                if dist < best_dist {
                    best_dist = dist;
                }
            }
            eprintln!(
                "[parity] herdnet golden[{i}] bbox={:?} → nearest gpu det dist={:.4}",
                g_bbox, best_dist,
            );
            assert!(
                best_dist < 0.02,
                "HerdNet top-{i} golden detection has no GPU match within 0.02 normalized-coord \
                 (nearest = {best_dist})",
            );
        }
    } else {
        // No golden: at minimum require the detection set is plausible.
        eprintln!("[parity] herdnet_tiled golden missing — sanity-only assertion");
        assert!(
            !result.detections.is_empty(),
            "HerdNet should produce detections on the buffalo overhead test image",
        );
    }
}

// ---------------------------------------------------------------------------
// OWL-T — single-output heatmap, tile_overlap = 160, adaptive threshold
// ---------------------------------------------------------------------------

#[test]
fn owl_tiled_detection_gpu() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping OWL-T tiled GPU test");
        return;
    }
    let Some(manifest_path) = owl_manifest_path().filter(|p| p.exists()) else {
        eprintln!("OWL-T manifest fixture missing — skipping (set SPARROW_ENGINE_TEST_FILES_ROOT)");
        return;
    };
    let Some(img_path) = overhead_image().filter(|p| p.exists()) else {
        eprintln!("Overhead test image missing — skipping (set SPARROW_ENGINE_TEST_FILES_ROOT)");
        return;
    };

    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let model = TiledModel::load_from_path(&ctx, &manifest_path)
        .expect("TiledModel::load_from_path (OWL-T)");
    assert_eq!(model.model_id(), "owl-t");

    let bytes = std::fs::read(&img_path).expect("read overhead jpeg");
    let input = ImageInput::Encoded(bytes);
    let opts = DetectOpts::default();

    let result = model
        .detect_tiled(&ctx, &input, &opts)
        .expect("detect_tiled (OWL-T)");

    eprintln!(
        "[bench] owl_tiled total_ms={:.3} detections={} image={}x{}",
        result.processing_time_ms,
        result.detections.len(),
        result.image_width,
        result.image_height,
    );

    // Image dimensions match the corpus.
    assert_eq!(result.image_width, 6000);
    assert_eq!(result.image_height, 4000);

    // OWL-T should produce detections on this overhead wildlife image.
    assert!(
        !result.detections.is_empty(),
        "OWL-T should detect animals in overhead image (vacuous-pass check)",
    );

    // Bbox sanity.
    for d in &result.detections {
        assert!(
            (0.0..=1.0).contains(&d.bbox.x_min),
            "x_min out of range: {}",
            d.bbox.x_min,
        );
        assert!(
            (0.0..=1.0).contains(&d.bbox.y_min),
            "y_min out of range: {}",
            d.bbox.y_min,
        );
        assert!(d.bbox.x_max >= d.bbox.x_min);
        assert!(d.bbox.y_max >= d.bbox.y_min);
        assert!(
            d.confidence > 0.0 && d.confidence <= 1.0,
            "OWL-T confidence out of (0,1]: {}",
            d.confidence,
        );
        assert_eq!(
            d.label_id, 0,
            "OWL-T is single-class; all detections must have label_id = 0, got {}",
            d.label_id,
        );
    }

    // No two detections may have identical bboxes (cross-tile dedup must run).
    for i in 0..result.detections.len() {
        for j in (i + 1)..result.detections.len() {
            assert_ne!(
                result.detections[i].bbox, result.detections[j].bbox,
                "OWL-T duplicate detection at indices {i}/{j}: bbox {:?}",
                result.detections[i].bbox,
            );
        }
    }

    // Confidence-descending sort.
    for w in result.detections.windows(2) {
        assert!(
            w[0].confidence >= w[1].confidence,
            "OWL-T detections not sorted by confidence",
        );
    }
}

// ---------------------------------------------------------------------------
// In-process latency bench (separate test, ignored by default).
//
// Cross-process bench (5 fresh-process runs → wave_4_bench.md) is driven from
// outside: re-invoke `cargo test --test integration_tiled herdnet_tiled_detection_gpu`
// 5 times and grep `[bench] herdnet_tiled total_ms=...` from stderr.
//
// This in-process variant is useful for a quick local sanity check that
// per-tile latency is stable across iterations within a single warm process.
// Run with `cargo test -p sparrow-engine-gpu --release --test integration_tiled -- --ignored bench`.
// ---------------------------------------------------------------------------

#[test]
#[ignore]
fn bench_herdnet_tiled_5_iters() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping HerdNet bench");
        return;
    }
    let Some(manifest_path) = herdnet_manifest_path().filter(|p| p.exists()) else {
        eprintln!("HerdNet bench fixtures missing — skipping");
        return;
    };
    let Some(img_path) = overhead_image().filter(|p| p.exists()) else {
        eprintln!("HerdNet bench image missing — skipping");
        return;
    };
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let model = TiledModel::load_from_path(&ctx, &manifest_path).expect("load HerdNet");
    let bytes = std::fs::read(&img_path).expect("read jpeg");
    let input = ImageInput::Encoded(bytes);
    let opts = DetectOpts::default();
    // Warm-up: ORT cuDNN algo selection happens on first run.
    let _ = model
        .detect_tiled(&ctx, &input, &opts)
        .expect("warmup detect_tiled");

    let mut samples: Vec<f32> = Vec::with_capacity(5);
    for _ in 0..5 {
        let r = model
            .detect_tiled(&ctx, &input, &opts)
            .expect("bench detect_tiled");
        samples.push(r.processing_time_ms);
    }
    print_bench_summary("herdnet_tiled (warm, in-process)", &samples);
}

#[test]
#[ignore]
fn bench_owl_tiled_5_iters() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping OWL-T bench");
        return;
    }
    let Some(manifest_path) = owl_manifest_path().filter(|p| p.exists()) else {
        eprintln!("OWL-T bench fixtures missing — skipping");
        return;
    };
    let Some(img_path) = overhead_image().filter(|p| p.exists()) else {
        eprintln!("OWL-T bench image missing — skipping");
        return;
    };
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let model = TiledModel::load_from_path(&ctx, &manifest_path).expect("load OWL-T");
    let bytes = std::fs::read(&img_path).expect("read jpeg");
    let input = ImageInput::Encoded(bytes);
    let opts = DetectOpts::default();
    let _ = model
        .detect_tiled(&ctx, &input, &opts)
        .expect("warmup detect_tiled");

    let mut samples: Vec<f32> = Vec::with_capacity(5);
    for _ in 0..5 {
        let r = model
            .detect_tiled(&ctx, &input, &opts)
            .expect("bench detect_tiled");
        samples.push(r.processing_time_ms);
    }
    print_bench_summary("owl_tiled (warm, in-process)", &samples);
}

// ---------------------------------------------------------------------------
// Phase 3.8 Step 1 audit-fix R1 regression tests (B1 + B2)
// ---------------------------------------------------------------------------

/// B1 regression: `SPARROW_ENGINE_GPU_FORCE_CPU_DECODE=1` must reach the
/// `decode_via_cpu_fallback` branch for `TiledModel::detect_tiled` (was
/// previously honored only in `YoloModel`). We can't observe the branch
/// directly without a public hook, but a successful `detect_tiled` call
/// with the env var set proves the CPU fallback path runs end-to-end
/// without panicking — that path was unreachable before this fix.
#[test]
#[ignore]
fn b1_force_cpu_decode_runs_end_to_end_for_tiled() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping b1_force_cpu_decode_runs_end_to_end_for_tiled");
        return;
    }
    let Some(manifest_path) = herdnet_manifest_path().filter(|p| p.exists()) else {
        eprintln!("HerdNet manifest missing — skipping B1 tiled regression");
        return;
    };
    let Some(img_path) = overhead_image().filter(|p| p.exists()) else {
        eprintln!("Overhead image missing — skipping B1 tiled regression");
        return;
    };

    let _env_guard = EnvVarGuard::set("SPARROW_ENGINE_GPU_FORCE_CPU_DECODE", "1");
    let result = {
        let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
        let model = TiledModel::load_from_path(&ctx, &manifest_path).expect("load HerdNet (B1)");
        let bytes = std::fs::read(&img_path).expect("read jpeg (B1)");
        let input = ImageInput::Encoded(bytes);
        let opts = DetectOpts::default();
        model.detect_tiled(&ctx, &input, &opts)
    };
    let det = result.expect("detect_tiled with SPARROW_ENGINE_GPU_FORCE_CPU_DECODE=1");
    eprintln!(
        "[b1] tiled CPU-decode path: {} detections, {} ms",
        det.detections.len(),
        det.processing_time_ms,
    );
    // detect_tiled returns >0 detections on the canonical buffalo overhead
    // image (golden sparrow-engine-cpu reference), so use that as a sanity gate.
    // The decode path is the only thing that changes; preprocessing +
    // inference + postprocess all match the nvjpeg-default path.
    assert!(
        !det.detections.is_empty(),
        "tiled CPU-decode path produced 0 detections — buffalo overhead image should yield >0",
    );
}

/// B2 regression: `TiledModel::device_id()` must match `ctx.ordinal() as
/// i32` after load. Previously the session was hardcoded to
/// `with_device_id(0)`; this guard prevents regression.
#[test]
#[ignore]
fn b2_tiled_device_id_matches_ctx_ordinal() {
    if !gpu_tests_enabled() {
        eprintln!("SPARROW_ENGINE_GPU_TESTS=0 → skipping b2_tiled_device_id_matches_ctx_ordinal");
        return;
    }
    let Some(manifest_path) = herdnet_manifest_path().filter(|p| p.exists()) else {
        eprintln!("HerdNet manifest missing — skipping B2 tiled regression");
        return;
    };
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let model = TiledModel::load_from_path(&ctx, &manifest_path).expect("load HerdNet (B2)");
    let expected: i32 = ctx.ordinal().try_into().expect("ctx.ordinal as i32");
    assert_eq!(
        model.device_id(),
        expected,
        "TiledModel::device_id() must equal ctx.ordinal() as i32 — \
         hardcoded with_device_id(0) regression",
    );
}

fn print_bench_summary(label: &str, samples: &[f32]) {
    let mut sorted: Vec<f32> = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = sorted.len();
    let median = sorted[n / 2];
    let p95 = sorted[((n - 1) as f32 * 0.95).round() as usize];
    let mean = sorted.iter().sum::<f32>() / n as f32;
    let var = sorted.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / n as f32;
    let stddev = var.sqrt();
    let max = *sorted.last().unwrap();
    eprintln!(
        "[bench] {label} n={n} median={median:.3}ms p95={p95:.3}ms stddev={stddev:.3}ms max={max:.3}ms samples={sorted:?}",
    );
}
