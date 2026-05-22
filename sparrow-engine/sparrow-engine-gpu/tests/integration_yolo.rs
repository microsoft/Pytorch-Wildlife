//! Phase 3.8 Step 1 Wave 2 integration tests: MDv6 + DeepFaune YOLO E2E.
//!
//! Tests are gated on environment variables:
//! - `SPARROW_ENGINE_GPU_TEST_CORPUS` — directory of JPEGs to run inference on
//!   (default `/home/miao/repos/PW_refactor/test_files/test_cameratrap`).
//! - `SPARROW_ENGINE_GPU_TEST_MODELS` — root directory containing
//!   `megadetector-v6-yolov10e/` and `deepfaune-yolo8s/` manifest folders
//!   (default `/home/miao/repos/PW_refactor/test_files/sparrow_engine_models`).
//! - `SPARROW_ENGINE_GPU_TEST_FORCE` — when set to `1`, missing fixtures cause the
//!   test to fail rather than skip (CI guardrail).
//!
//! All inference tests are `#[ignore]` so a no-fixture clean checkout still
//! sees `cargo test` PASS. Run them explicitly:
//!
//! ```bash
//! cargo test -p sparrow-engine-gpu --release --test integration_yolo -- --ignored
//! ```
//!
//! Parity reference: `sparrow-engine-cpu` (Engine + detect path) on the same image
//! corpus. Detection-count parity ±1 / 100, IoU ≥ 0.99, score Δ ≤ 0.005
//! mean / 0.01 max — Gate G2 in `docs/design/phase3.8/step1/implementation_plan.md §4`.
//!
//! Algorithmic divergence note: sparrow-engine-cpu uses `fast_image_resize` Triangle
//! bilinear (convolutional with 2-pixel kernel taps); sparrow-engine-gpu's letterbox
//! kernel is a 2-tap bilinear sampler. For >2× downsampling the two filter
//! responses differ — this is documented in `docs/ideas.md` (P3.8-4) and
//! is expected to surface as small score drift on a minority of detections.
//! When parity exceeds the Gate G2 tolerance, the residual is recorded in
//! `docs/research/phase3.8/step1/wave_2_bench.md` instead of failing the
//! test (the test asserts only the headline gate values; per-image residual
//! tables live in the bench doc).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

// Phase 3.8 Phase C Wave 4b: see audio_e2e_parity.rs for the dev-dep
// rename rationale. `sparrow_engine_cpu` is the CPU baseline used for parity;
// `sparrow-engine-gpu` (current crate) is the sparrow-engine-gpu surface under test.
use sparrow_engine::kernels::letterbox::LetterboxKernel;
use sparrow_engine::models::yolo::YoloModel;
use sparrow_engine_cpu::engine::{Device, Engine, EngineConfig};
use sparrow_engine_types::manifest::load_manifest;
use sparrow_engine_types::{DetectOpts, DetectResult, Detection, ImageInput, PixelFormat};
use cudarc::driver::CudaContext;

// ---------------------------------------------------------------------------
// Fixture discovery
// ---------------------------------------------------------------------------

const DEFAULT_CORPUS: &str = "/home/miao/repos/PW_refactor/test_files/test_cameratrap";
const DEFAULT_MODELS: &str = "/home/miao/repos/PW_refactor/test_files/sparrow_engine_models";

fn corpus_dir() -> PathBuf {
    PathBuf::from(std::env::var("SPARROW_ENGINE_GPU_TEST_CORPUS").unwrap_or_else(|_| DEFAULT_CORPUS.into()))
}

fn models_dir() -> PathBuf {
    PathBuf::from(std::env::var("SPARROW_ENGINE_GPU_TEST_MODELS").unwrap_or_else(|_| DEFAULT_MODELS.into()))
}

fn force_fixtures() -> bool {
    std::env::var("SPARROW_ENGINE_GPU_TEST_FORCE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Skip the test (return None) if `path` is missing and `SPARROW_ENGINE_GPU_TEST_FORCE`
/// is not set; otherwise panic.
fn require_fixture(path: &Path) -> Option<()> {
    if path.exists() {
        return Some(());
    }
    if force_fixtures() {
        panic!(
            "missing fixture: {} (SPARROW_ENGINE_GPU_TEST_FORCE=1)",
            path.display()
        );
    }
    eprintln!(
        "skipping: missing fixture {} (set SPARROW_ENGINE_GPU_TEST_FORCE=1 to make this fatal)",
        path.display()
    );
    None
}

/// Collect the first `limit` JPEGs from a directory, sorted by filename.
fn collect_jpegs(dir: &Path, limit: usize) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|entry: std::io::Result<std::fs::DirEntry>| {
                entry.ok().map(|entry: std::fs::DirEntry| entry.path())
            })
            .filter(|p: &PathBuf| {
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg" | "png")
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    out.sort();
    out.truncate(limit);
    out
}

// ---------------------------------------------------------------------------
// Engine setup helpers (CPU baseline + GPU under test)
// ---------------------------------------------------------------------------

/// Build the sparrow-engine-cpu engine + load the model. Returns (engine, model_id).
/// Wraps the engine in `Box` so the caller drops it before constructing the
/// sparrow-engine-gpu engine (singleton constraint inside each crate; here the two
/// crates' singletons are independent but we still serialize for hygiene).
fn cpu_baseline_detect(manifest_path: &Path, images: &[PathBuf]) -> Vec<DetectResult> {
    let model_dir = manifest_path.parent().unwrap().to_path_buf();
    let cfg = EngineConfig::new(Device::Cuda(0), &model_dir);
    let engine = Engine::new(cfg).expect("sparrow-engine-cpu Engine::new");
    let handle = engine
        .load_model(manifest_path)
        .expect("sparrow-engine-cpu load_model");
    let opts = DetectOpts::default();
    let mut results = Vec::with_capacity(images.len());
    for p in images {
        let r = sparrow_engine_cpu::detect::detect(&handle, &ImageInput::FilePath(p.clone()), &opts)
            .expect("sparrow-engine-cpu detect");
        results.push(r);
    }
    drop(handle);
    drop(engine);
    results
}

/// Build a fresh CudaContext + LetterboxKernel + load the YoloModel. Returns
/// (ctx, kernel, model). Caller must keep them alive for the test.
fn gpu_load_yolo(manifest_path: &Path) -> (Arc<CudaContext>, LetterboxKernel, YoloModel) {
    let manifest = load_manifest(manifest_path).expect("load_manifest");
    let manifest_dir = manifest_path.parent().unwrap();
    let ctx = CudaContext::new(0).expect("CudaContext::new(0)");
    let kernel = LetterboxKernel::new(&ctx).expect("LetterboxKernel::new");
    let model = YoloModel::load(&ctx, &manifest, manifest_dir).expect("YoloModel::load");
    (ctx, kernel, model)
}

/// Run YoloModel::detect on each image; returns the detect results.
fn gpu_yolo_detect(
    ctx: &Arc<CudaContext>,
    kernel: &LetterboxKernel,
    model: &YoloModel,
    images: &[PathBuf],
) -> Vec<DetectResult> {
    let opts = DetectOpts::default();
    let mut out = Vec::with_capacity(images.len());
    for p in images {
        let r = model
            .detect(ctx, kernel, &ImageInput::FilePath(p.clone()), &opts)
            .expect("YoloModel::detect");
        out.push(r);
    }
    out
}

// ---------------------------------------------------------------------------
// Parity helpers
// ---------------------------------------------------------------------------

/// Greedy IoU matching: for each cpu detection, find the best gpu detection
/// (same class, highest IoU). Reports per-image counts + per-match IoU /
/// score deltas.
struct ParityReport {
    cpu_count: usize,
    gpu_count: usize,
    matched: usize,
    iou_min: f32,
    iou_mean: f32,
    score_abs_max: f32,
    score_abs_mean: f32,
    /// Per-image rows: (image_path, cpu_count, gpu_count, max_iou_per_match,
    /// score_delta_per_match) for all 100 images. Used to surface outliers.
    per_image: Vec<PerImageRow>,
}

#[derive(Debug)]
struct PerImageRow {
    image: PathBuf,
    cpu_count: usize,
    gpu_count: usize,
    /// Min IoU across matched detections for this image. NaN when matched=0.
    iou_min_match: f32,
    /// Max score Δ across matched detections for this image. 0.0 when matched=0.
    score_max_delta: f32,
    matched: usize,
}

fn iou(a: &Detection, b: &Detection) -> f32 {
    let ax1 = a.bbox.x_min;
    let ay1 = a.bbox.y_min;
    let ax2 = a.bbox.x_max;
    let ay2 = a.bbox.y_max;
    let bx1 = b.bbox.x_min;
    let by1 = b.bbox.y_min;
    let bx2 = b.bbox.x_max;
    let by2 = b.bbox.y_max;
    let ix1 = ax1.max(bx1);
    let iy1 = ay1.max(by1);
    let ix2 = ax2.min(bx2);
    let iy2 = ay2.min(by2);
    let iw = (ix2 - ix1).max(0.0);
    let ih = (iy2 - iy1).max(0.0);
    let inter = iw * ih;
    let area_a = ((ax2 - ax1) * (ay2 - ay1)).max(0.0);
    let area_b = ((bx2 - bx1) * (by2 - by1)).max(0.0);
    let denom = area_a + area_b - inter;
    if denom <= 0.0 {
        0.0
    } else {
        inter / denom
    }
}

fn parity_report(images: &[PathBuf], cpu: &[DetectResult], gpu: &[DetectResult]) -> ParityReport {
    let mut cpu_count = 0usize;
    let mut gpu_count = 0usize;
    let mut matched = 0usize;
    let mut iou_sum = 0.0f64;
    let mut iou_min = f32::INFINITY;
    let mut score_abs_sum = 0.0f64;
    let mut score_abs_max = 0.0f32;
    let mut per_image = Vec::with_capacity(cpu.len().min(gpu.len()));
    let n = cpu.len().min(gpu.len()).min(images.len());
    for i in 0..n {
        cpu_count += cpu[i].detections.len();
        gpu_count += gpu[i].detections.len();

        // IoU-descending greedy matcher: enumerate every same-class pair,
        // sort by IoU (largest first), and claim pairs while neither side
        // is already taken. This avoids the "early CPU detection steals a
        // mid-quality GPU match" pathology that the by-CPU-iteration matcher
        // had — see image 69267e43 in earlier runs (4-vs-4 with one IoU=0.42
        // false low caused by a single mis-pairing).
        let mut pairs: Vec<(usize, usize, f32)> =
            Vec::with_capacity(cpu[i].detections.len() * gpu[i].detections.len());
        for (ci, c) in cpu[i].detections.iter().enumerate() {
            for (gi, g) in gpu[i].detections.iter().enumerate() {
                if g.label_id != c.label_id {
                    continue;
                }
                let v = iou(c, g);
                if v > 0.0 {
                    pairs.push((ci, gi, v));
                }
            }
        }
        pairs.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

        let mut c_used = vec![false; cpu[i].detections.len()];
        let mut g_used = vec![false; gpu[i].detections.len()];
        let mut img_iou_min = f32::INFINITY;
        let mut img_score_max = 0.0f32;
        let mut img_matched = 0usize;
        for (ci, gi, v) in pairs {
            if c_used[ci] || g_used[gi] {
                continue;
            }
            c_used[ci] = true;
            g_used[gi] = true;
            matched += 1;
            img_matched += 1;
            iou_sum += v as f64;
            if v < iou_min {
                iou_min = v;
            }
            if v < img_iou_min {
                img_iou_min = v;
            }
            let score_d =
                (cpu[i].detections[ci].confidence - gpu[i].detections[gi].confidence).abs();
            score_abs_sum += score_d as f64;
            if score_d > score_abs_max {
                score_abs_max = score_d;
            }
            if score_d > img_score_max {
                img_score_max = score_d;
            }
        }
        per_image.push(PerImageRow {
            image: images[i].clone(),
            cpu_count: cpu[i].detections.len(),
            gpu_count: gpu[i].detections.len(),
            iou_min_match: if img_iou_min.is_finite() {
                img_iou_min
            } else {
                f32::NAN
            },
            score_max_delta: img_score_max,
            matched: img_matched,
        });
    }
    let iou_mean = if matched > 0 {
        (iou_sum / matched as f64) as f32
    } else {
        0.0
    };
    let score_abs_mean = if matched > 0 {
        (score_abs_sum / matched as f64) as f32
    } else {
        0.0
    };
    ParityReport {
        cpu_count,
        gpu_count,
        matched,
        iou_min: if iou_min.is_finite() { iou_min } else { 0.0 },
        iou_mean,
        score_abs_max,
        score_abs_mean,
        per_image,
    }
}

/// Dump a full per-image CSV to `<dir>/<model>_per_image.csv`.
fn dump_per_image_csv(report: &ParityReport, dir: &str, model: &str) -> std::io::Result<()> {
    use std::io::Write as _;
    std::fs::create_dir_all(dir)?;
    let path = std::path::PathBuf::from(dir).join(format!("{model}_per_image.csv"));
    let mut f = std::fs::File::create(&path)?;
    writeln!(
        f,
        "image,cpu_count,gpu_count,matched,iou_min_match,score_max_delta"
    )?;
    for r in &report.per_image {
        let name = r
            .image
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| r.image.display().to_string());
        let iou_str = if r.iou_min_match.is_finite() {
            format!("{:.6}", r.iou_min_match)
        } else {
            String::new()
        };
        writeln!(
            f,
            "{name},{},{},{},{iou_str},{:.6}",
            r.cpu_count, r.gpu_count, r.matched, r.score_max_delta
        )?;
    }
    Ok(())
}

/// Print the worst N per-image rows by both count-drift magnitude and IoU min /
/// score-delta. Used after a parity assertion to localize outliers.
fn print_outlier_table(report: &ParityReport, top_n: usize, label: &str) {
    eprintln!(
        "[{label}] per-image outliers (worst {top_n} by score Δ, then by IoU drop, then by count drift)"
    );
    let mut rows: Vec<&PerImageRow> = report.per_image.iter().collect();
    // Order: largest score delta first, then smallest IoU min, then largest |count diff|.
    rows.sort_by(|a, b| {
        let a_score = a.score_max_delta;
        let b_score = b.score_max_delta;
        match b_score
            .partial_cmp(&a_score)
            .unwrap_or(std::cmp::Ordering::Equal)
        {
            std::cmp::Ordering::Equal => {
                let a_iou = if a.iou_min_match.is_finite() {
                    a.iou_min_match
                } else {
                    1.0
                };
                let b_iou = if b.iou_min_match.is_finite() {
                    b.iou_min_match
                } else {
                    1.0
                };
                a_iou
                    .partial_cmp(&b_iou)
                    .unwrap_or(std::cmp::Ordering::Equal)
            }
            other => other,
        }
    });
    eprintln!(
        "  {:<60} | cpu | gpu | matched | iou_min  | score_max",
        "image"
    );
    for r in rows.iter().take(top_n) {
        let name = r
            .image
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| r.image.display().to_string());
        let iou_str = if r.iou_min_match.is_finite() {
            format!("{:.4}", r.iou_min_match)
        } else {
            "n/a".to_string()
        };
        eprintln!(
            "  {name:<60} | {:>3} | {:>3} | {:>7} | {:>8} | {:>9.5}",
            r.cpu_count, r.gpu_count, r.matched, iou_str, r.score_max_delta
        );
    }
}

// ---------------------------------------------------------------------------
// Stub-level test (always runs; verifies the module API surface)
// ---------------------------------------------------------------------------

/// Smoke test: manifest loading reports a sensible error when the manifest
/// is missing. Always runs (no GPU required).
#[test]
fn manifest_loader_missing_manifest_errors() {
    let bogus =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("__bongo_gpu_nonexistent_manifest.toml");
    let res = sparrow_engine_types::manifest::load_manifest(&bogus);
    assert!(res.is_err(), "missing manifest must error");
}

// ---------------------------------------------------------------------------
// MDv6 parity + bench
// ---------------------------------------------------------------------------

/// Wave 2 Gate G2: MDv6 GPU vs CPU parity on the 100-image corpus.
///
/// Records:
/// - Detection count drift.
/// - IoU min / mean.
/// - Score Δ mean / max.
///
/// Asserts a sanity floor (count drift ≤ 5 / 100, IoU mean ≥ 0.95, score
/// Δ mean ≤ 0.05). The strict Gate G2 thresholds (count ±1, IoU ≥ 0.99,
/// score Δ ≤ 0.005) are reported but NOT asserted here — the algorithmic
/// divergence between sparrow-engine-cpu's `fast_image_resize` Triangle bilinear
/// (4-pixel kernel taps) and sparrow-engine-gpu's 2-tap bilinear sampler is documented
/// as P3.8-4 in `docs/ideas.md` and produces small score / count residuals.
/// Actual numbers land in `docs/research/phase3.8/step1/wave_2_bench.md`.
///
/// Skips when `SPARROW_ENGINE_GPU_TEST_CORPUS` / `SPARROW_ENGINE_GPU_TEST_MODELS` are missing
/// unless `SPARROW_ENGINE_GPU_TEST_FORCE=1`.
#[test]
#[ignore]
fn mdv6_parity_100_images() {
    run_yolo_parity("megadetector-v6-yolov10e", 100);
}

/// Wave 2 Gate G2: DeepFaune GPU vs CPU parity on the 100-image corpus.
#[test]
#[ignore]
fn deepfaune_parity_100_images() {
    run_yolo_parity("deepfaune-yolo8s", 100);
}

// Gate G2 re-spec (team-lead, 2026-05-03; superseded the Phase-A-inherited
// numbers in `docs/design/phase3.8/step1/implementation_plan.md §4`).
//
// Phase A's Gate-0 thresholds (count ≤1, IoU min ≥0.99, score Δ ≤0.005/0.01)
// assumed a pure-refactor architecture where sparrow-engine-gpu and sparrow-engine-cpu shared
// the same JPEG decoder + same ORT EP. Step 1 Wave 2 introduces TWO
// independent divergence sources on top of the preprocess axis:
//
//   - sparrow-engine-gpu uses nvjpeg + ORT CUDA EP.
//   - sparrow-engine-cpu uses zune-jpeg (via `image` crate) + ORT CPU EP.
//
// Each axis is now attributed to a measurable cause; thresholds are set
// to the measured baseline post-multi-tap-letterbox-fix:
//
//   count drift  ≤ 1 / 100  — preprocess parity (must close; closed by
//                              the multi-tap letterbox + pad_value fix).
//   IoU min     ≥ 0.91     — TWO mechanisms (the threshold accommodates
//                              both):
//                              (a) nvjpeg (GPU) vs zune-jpeg (CPU) IDCT
//                                  divergence on large-bbox detections.
//                                  Verifiable by re-running with
//                                  SPARROW_ENGINE_GPU_FORCE_CPU_DECODE=1 — if the
//                                  IoU min tightens, decoder is the
//                                  variable for that case.
//                              (b) EP-side cuDNN conv precision
//                                  propagated through bbox regression,
//                                  amplified proportionally on detections
//                                  with bbox dim < 5% (small-animal
//                                  cases in busy frames). Verifiable by
//                                  the same force_cpu_decode flip — if
//                                  IoU min DOESN'T tighten, the residual
//                                  is EP-side, not decoder-side.
//                              Empirical floor on the 100-image corpus:
//                              DeepFaune 0.9148 on `5dd96738-...`
//                              (4-detection busy image; tiny-animal pair
//                              at bbox w=0.04 h=0.14, force_cpu IoU min
//                              also 0.9151 → confirms (b) cause).
//   score Δ mean ≤ 0.01    — cuDNN session-mean reduction-order
//                              divergence (Phase 3.7 R2 finding).
//   score Δ max  ≤ 0.50    — TWO mechanisms (a) and (b):
//                              (a) cuDNN EXHAUSTIVE re-roll outliers
//                                  (per-session algo selection produces
//                                  ~0.05-0.1 score deltas on busy images).
//                              (b) YOLO E2E NMS keeping 2-vs-1 on
//                                  borderline detections (MDv6 image
//                                  52608121: cpu 1 det @ 0.97 vs gpu 2
//                                  dets @ 0.61 + 0.56 on identical
//                                  bbox — classification-head
//                                  perturbation crosses NMS suppression
//                                  threshold, splitting one CPU
//                                  detection into two GPU detections).
//
// Per-image diff CSV: `SPARROW_ENGINE_GPU_TEST_DUMP_PER_IMAGE=<dir>` env var on
// `mdv6_parity_100_images` / `deepfaune_parity_100_images` writes a
// 100-row CSV (image, cpu_count, gpu_count, matched, iou_min_match,
// score_max_delta) to that dir. Use it when an axis is exceeded to
// localize the offending image.
//
// Re-spec is binding. Do NOT raise these thresholds without re-running
// the variable-isolation experiment first (multi-tap default vs
// `SPARROW_ENGINE_GPU_FORCE_CPU_DECODE=1` → narrows down whether the residual is
// JPEG-decoder-side or EP-side).
// Gate G2 spec — re-spec'd 2026-05-04 after manifest-flip to FP16 default
// for both sparrow-engine-cpu and sparrow-engine-gpu (per user directive: "default both gpu
// and cpu mode to fp16, the borderline detection is fine"):
//
// COUNT_DRIFT_MAX raised 1 → 2 because ORT CPU EP's FP16 path is
// software-emulated (no CPU FP16 hardware) while ORT CUDA EP's FP16 path
// uses Tensor Core hardware. The two paths round FP16 ops differently.
// On DeepFaune specifically (densest borderline detections at the 0.2
// threshold of the 5 image models), this lands as +1 detection appearing
// on sparrow-engine-cpu AND -1 detection disappearing on sparrow-engine-gpu, on two
// different images — net cross-engine drift = 2, not 1. Both are real
// detections shifting at the FP16 quantization boundary on different
// hardware. Documented in CHANGELOG.md (2026-05-04 FP16-by-default note)
// and `docs/research/phase3.8/step1/full_bench.md` § DeepFaune.
const GATE_G2_COUNT_DRIFT_MAX: i32 = 2;
const GATE_G2_IOU_MIN: f32 = 0.90;
const GATE_G2_SCORE_MEAN_MAX: f32 = 0.01;
const GATE_G2_SCORE_MAX_MAX: f32 = 0.50;

fn run_yolo_parity(model_subdir: &str, image_limit: usize) {
    let corpus = corpus_dir();
    let models = models_dir();
    let manifest_path = models.join(model_subdir).join("manifest.toml");
    if require_fixture(&corpus).is_none() || require_fixture(&manifest_path).is_none() {
        return;
    }
    let images = collect_jpegs(&corpus, image_limit);
    if images.is_empty() {
        if force_fixtures() {
            panic!("no images found in {}", corpus.display());
        }
        eprintln!("skipping: no images found in {}", corpus.display());
        return;
    }

    eprintln!(
        "[{model_subdir}] running parity on {} images from {}",
        images.len(),
        corpus.display()
    );

    // Run CPU baseline first; engine drops before GPU engine builds.
    let cpu = cpu_baseline_detect(&manifest_path, &images);

    // GPU under test.
    let (ctx, kernel, model) = gpu_load_yolo(&manifest_path);
    let gpu = gpu_yolo_detect(&ctx, &kernel, &model, &images);
    drop(model);
    drop(kernel);
    drop(ctx);

    // Report.
    let r = parity_report(&images, &cpu, &gpu);
    eprintln!(
        "[{model_subdir}] cpu_count={} gpu_count={} matched={} iou_min={:.4} iou_mean={:.4} \
         score_abs_max={:.5} score_abs_mean={:.5}",
        r.cpu_count,
        r.gpu_count,
        r.matched,
        r.iou_min,
        r.iou_mean,
        r.score_abs_max,
        r.score_abs_mean
    );
    // Always emit the worst-10 table — useful both at PASS (sanity check)
    // and at FAIL (localize the offending images).
    print_outlier_table(&r, 10, model_subdir);

    // If `SPARROW_ENGINE_GPU_TEST_DUMP_PER_IMAGE` points at a directory, write a
    // full per-image CSV so team-lead can see all 100 rows. Useful when
    // the assertion fails and the worst-10 table isn't enough.
    if let Ok(dir) = std::env::var("SPARROW_ENGINE_GPU_TEST_DUMP_PER_IMAGE") {
        if let Err(e) = dump_per_image_csv(&r, &dir, model_subdir) {
            eprintln!("[{model_subdir}] WARN: failed to write per-image CSV: {e}");
        } else {
            eprintln!(
                "[{model_subdir}] per-image CSV: {}/{}_per_image.csv",
                dir, model_subdir
            );
        }
    }

    let count_diff = (r.cpu_count as i32 - r.gpu_count as i32).abs();

    // Gate G2 strict assertions, per the re-spec'd thresholds. Each axis
    // names its measurable cause-source so a future reader can re-run the
    // variable-isolation experiment if the threshold needs revisiting.
    assert!(
        count_diff <= GATE_G2_COUNT_DRIFT_MAX,
        "[{model_subdir}] Gate G2 count drift {} > {} (cpu={}, gpu={}); \
         named source if exceeded: preprocess divergence — re-run with \
         multi-tap letterbox kernel + pad_value fix BEFORE re-specing.",
        count_diff,
        GATE_G2_COUNT_DRIFT_MAX,
        r.cpu_count,
        r.gpu_count
    );
    let min_expected_matches = r
        .cpu_count
        .min(r.gpu_count)
        .saturating_sub(GATE_G2_COUNT_DRIFT_MAX as usize);
    assert!(
        r.matched >= min_expected_matches,
        "[{model_subdir}] Gate G2 matched detections {} < expected floor {}          (cpu={}, gpu={}, allowed_count_drift={}); count parity alone cannot          prove bbox/label parity. Inspect unmatched rows in the outlier table.",
        r.matched,
        min_expected_matches,
        r.cpu_count,
        r.gpu_count,
        GATE_G2_COUNT_DRIFT_MAX
    );

    if r.matched > 0 {
        assert!(
            r.iou_min >= GATE_G2_IOU_MIN,
            "[{model_subdir}] Gate G2 iou_min {} < {} (worst-case matched pair); \
             two named-source mechanisms accommodated by this threshold: \
             (a) nvjpeg (GPU) vs zune-jpeg (CPU) IDCT divergence on \
             large-bbox detections; (b) EP-side cuDNN conv precision \
             propagated through bbox regression and amplified on small-\
             animal detections (bbox dim < 5%). Verify which mechanism \
             with SPARROW_ENGINE_GPU_FORCE_CPU_DECODE=1: if IoU min tightens, \
             cause is (a); if unchanged, cause is (b).",
            r.iou_min,
            GATE_G2_IOU_MIN
        );
        assert!(
            r.score_abs_mean <= GATE_G2_SCORE_MEAN_MAX,
            "[{model_subdir}] Gate G2 score_abs_mean {} > {}; \
             named source if exceeded: cuDNN session-mean reduction-order \
             divergence (Phase 3.7 R2 finding). Test SPARROW_ENGINE_GPU_FORCE_CPU_DECODE=1 \
             to confirm decoder is not the variable; if that doesn't tighten, \
             escalate to ORT EP determinism investigation.",
            r.score_abs_mean,
            GATE_G2_SCORE_MEAN_MAX
        );
        assert!(
            r.score_abs_max <= GATE_G2_SCORE_MAX_MAX,
            "[{model_subdir}] Gate G2 score_abs_max {} > {}; two mechanisms \
             accommodated: (a) cuDNN EXHAUSTIVE re-roll outliers; (b) YOLO \
             E2E NMS keeping 2-vs-1 on borderline detections (bbox identical, \
             classification-head perturbation crosses NMS suppression \
             threshold). Empirical example: MDv6 image 52608121 — cpu 1 det \
             @ 0.97 vs gpu 2 dets @ 0.61 + 0.56 on identical bbox.",
            r.score_abs_max,
            GATE_G2_SCORE_MAX_MAX
        );
    }

    eprintln!(
        "[{model_subdir}] Gate G2 PASS (re-spec): count drift={} (≤{}); \
         iou_min={:.4} (≥{:.2}); score Δ mean={:.5} (≤{}); score Δ max={:.5} (≤{})",
        count_diff,
        GATE_G2_COUNT_DRIFT_MAX,
        r.iou_min,
        GATE_G2_IOU_MIN,
        r.score_abs_mean,
        GATE_G2_SCORE_MEAN_MAX,
        r.score_abs_max,
        GATE_G2_SCORE_MAX_MAX,
    );
}

// ---------------------------------------------------------------------------
// Latency bench (informational; documented in wave_2_bench.md)
// ---------------------------------------------------------------------------

/// Run YoloModel::detect 30 times warmup + 100 timed iters, print median /
/// p95 / stddev / max in ms. Informational only — no assertions; latency
/// targets are tracked in the bench doc.
#[test]
#[ignore]
fn mdv6_latency_bench() {
    run_yolo_bench("megadetector-v6-yolov10e", 30, 100);
}

#[test]
#[ignore]
fn deepfaune_latency_bench() {
    run_yolo_bench("deepfaune-yolo8s", 30, 100);
}

/// MDv6 FP16 bench. Reads manifest from `SPARROW_ENGINE_GPU_TEST_FP16_MANIFEST`
/// (a full `manifest.toml` path including the FP16 file_fp16 + precision
/// fields). Used during Wave 2 to capture the FP16 baseline that the
/// production manifest doesn't yet expose.
#[test]
#[ignore]
fn mdv6_fp16_latency_bench() {
    let manifest_env = std::env::var("SPARROW_ENGINE_GPU_TEST_FP16_MANIFEST").ok();
    let manifest_path = match manifest_env {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!(
                "skipping: SPARROW_ENGINE_GPU_TEST_FP16_MANIFEST not set (point at a manifest.toml \
                 with precision = fp16 + file_fp16)"
            );
            return;
        }
    };
    if require_fixture(&manifest_path).is_none() {
        return;
    }
    let corpus = corpus_dir();
    let images = collect_jpegs(&corpus, 1);
    if images.is_empty() {
        eprintln!("skipping: no images found in {}", corpus.display());
        return;
    }
    let img = ImageInput::FilePath(images[0].clone());

    let (ctx, kernel, model) = gpu_load_yolo(&manifest_path);
    let opts = DetectOpts::default();
    let warmup = 30usize;
    let iters = 100usize;

    for _ in 0..warmup {
        let _ = model
            .detect(&ctx, &kernel, &img, &opts)
            .expect("warmup detect (fp16)");
    }
    let mut timings = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = model
            .detect(&ctx, &kernel, &img, &opts)
            .expect("bench detect (fp16)");
        timings.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = timings[timings.len() / 2];
    let p95 = timings[(timings.len() as f64 * 0.95) as usize];
    let max = *timings.last().unwrap();
    let mean: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
    let var: f64 = timings.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / timings.len() as f64;
    let stddev = var.sqrt();
    eprintln!(
        "[mdv6-fp16] median={median:.2} ms  p95={p95:.2} ms  stddev={stddev:.2} ms  max={max:.2} ms  ({iters} iters, single image: {})",
        images[0].display()
    );

    drop(model);
    drop(kernel);
    drop(ctx);
}

fn run_yolo_bench(model_subdir: &str, warmup: usize, iters: usize) {
    // Initialize tracing if RUST_LOG is set (so internal stage timings surface).
    let _ = tracing_subscriber::fmt::try_init();

    let corpus = corpus_dir();
    let models = models_dir();
    let manifest_path = models.join(model_subdir).join("manifest.toml");
    if require_fixture(&corpus).is_none() || require_fixture(&manifest_path).is_none() {
        return;
    }
    let images = collect_jpegs(&corpus, 1);
    if images.is_empty() {
        eprintln!("skipping: no images found in {}", corpus.display());
        return;
    }
    let img = ImageInput::FilePath(images[0].clone());

    let (ctx, kernel, model) = gpu_load_yolo(&manifest_path);
    let opts = DetectOpts::default();

    // Warmup.
    for _ in 0..warmup {
        let _ = model
            .detect(&ctx, &kernel, &img, &opts)
            .expect("warmup detect");
    }

    // Timed.
    let mut timings = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t0 = Instant::now();
        let _ = model
            .detect(&ctx, &kernel, &img, &opts)
            .expect("bench detect");
        timings.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    timings.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = timings[timings.len() / 2];
    let p95 = timings[(timings.len() as f64 * 0.95) as usize];
    let max = *timings.last().unwrap();
    let mean: f64 = timings.iter().sum::<f64>() / timings.len() as f64;
    let var: f64 = timings.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / timings.len() as f64;
    let stddev = var.sqrt();
    eprintln!(
        "[{model_subdir}] median={median:.2} ms  p95={p95:.2} ms  stddev={stddev:.2} ms  max={max:.2} ms  ({iters} iters, single image: {})",
        images[0].display()
    );

    drop(model);
    drop(kernel);
    drop(ctx);
}

/// Diagnostic: dump CPU + GPU bboxes side by side for a single image.
/// Used to investigate the IoU=0.42 outlier on `69267e43-...jpg` for MDv6.
/// Run with:
/// ```
/// cargo test -p sparrow-engine-gpu --release --test integration_yolo -- --ignored \
///     --test-threads=1 mdv6_inspect_outlier_69267e43 --nocapture
/// ```
#[test]
#[ignore]
fn mdv6_inspect_outlier_69267e43() {
    let corpus = corpus_dir();
    let models = models_dir();
    let manifest_path = models
        .join("megadetector-v6-yolov10e")
        .join("manifest.toml");
    if require_fixture(&manifest_path).is_none() {
        return;
    }
    let img_path = corpus.join("69267e43-cf73-4e10-b4cf-ab64877a83cb.jpg");
    if require_fixture(&img_path).is_none() {
        return;
    }

    // CPU baseline.
    let model_dir = manifest_path.parent().unwrap().to_path_buf();
    let cpu_engine =
        Engine::new(EngineConfig::new(Device::Cuda(0), &model_dir)).expect("sparrow-engine-cpu Engine::new");
    let cpu_handle = cpu_engine
        .load_model(&manifest_path)
        .expect("cpu load_model");
    let cpu_res = sparrow_engine_cpu::detect::detect(
        &cpu_handle,
        &ImageInput::FilePath(img_path.clone()),
        &DetectOpts::default(),
    )
    .expect("sparrow-engine-cpu detect");
    drop(cpu_handle);
    drop(cpu_engine);

    // GPU under test.
    let (ctx, kernel, model) = gpu_load_yolo(&manifest_path);
    let gpu_res = model
        .detect(
            &ctx,
            &kernel,
            &ImageInput::FilePath(img_path),
            &DetectOpts::default(),
        )
        .expect("gpu detect");

    eprintln!("CPU detections ({}):", cpu_res.detections.len());
    for (i, d) in cpu_res.detections.iter().enumerate() {
        eprintln!(
            "  cpu[{i}] class={} conf={:.5} bbox=[{:.5},{:.5},{:.5},{:.5}] (w={:.5},h={:.5})",
            d.label_id,
            d.confidence,
            d.bbox.x_min,
            d.bbox.y_min,
            d.bbox.x_max,
            d.bbox.y_max,
            d.bbox.x_max - d.bbox.x_min,
            d.bbox.y_max - d.bbox.y_min
        );
    }
    eprintln!("GPU detections ({}):", gpu_res.detections.len());
    for (i, d) in gpu_res.detections.iter().enumerate() {
        eprintln!(
            "  gpu[{i}] class={} conf={:.5} bbox=[{:.5},{:.5},{:.5},{:.5}] (w={:.5},h={:.5})",
            d.label_id,
            d.confidence,
            d.bbox.x_min,
            d.bbox.y_min,
            d.bbox.x_max,
            d.bbox.y_max,
            d.bbox.x_max - d.bbox.x_min,
            d.bbox.y_max - d.bbox.y_min
        );
    }
}

#[test]
fn parity_helpers_iou_basic() {
    // Two identical bboxes → IoU = 1.0.
    let a = Detection {
        bbox: sparrow_engine_types::BBox {
            x_min: 0.1,
            y_min: 0.2,
            x_max: 0.5,
            y_max: 0.6,
        },
        label: "x".into(),
        label_id: 0,
        confidence: 0.9,
    };
    let b = a.clone();
    assert!((iou(&a, &b) - 1.0).abs() < 1e-6);
}

// ---------------------------------------------------------------------------
// Phase 3.8 Step 1 audit-fix R1 regression test (B2 MODIFY)
// ---------------------------------------------------------------------------

/// B2 MODIFY regression: `YoloModel::device_id()` must match
/// `ctx.ordinal() as i32` after load. Previously `build_session` used
/// `CUDA::default()` with no `with_device_id`, which silently resolved to
/// device 0 in ORT's EP factory regardless of the caller's actual context.
/// The per-call ordinal guard at the top of `detect()` would catch a
/// later mismatch, but the SESSION itself was still mis-pinned.
#[test]
#[ignore]
fn b2_yolo_device_id_matches_ctx_ordinal() {
    let manifest_path = models_dir()
        .join("megadetector-v6-yolov10e")
        .join("manifest.toml");
    if require_fixture(&manifest_path).is_none() {
        return;
    }
    let (ctx, _kernel, model) = gpu_load_yolo(&manifest_path);
    let expected: i32 = ctx.ordinal().try_into().expect("ctx.ordinal as i32");
    assert_eq!(
        model.device_id(),
        expected,
        "YoloModel::device_id() must equal ctx.ordinal() as i32 — \
         build_session CUDA::default() (no with_device_id pin) regression",
    );
}

// ---------------------------------------------------------------------------
// Phase 3.8 Step 1 audit-fix R3 M4 (B9 integration coverage)
// ---------------------------------------------------------------------------

/// Exercises YoloModel's `ImageInput::Raw` arm at `yolo.rs::detect`, which
/// routes through `crate::decode::raw_to_gpu` (B9 hoist). Before B9, the
/// `raw_to_png` helper rejected non-RGB pixel formats; BGRA was unsupported.
///
/// `#[ignore]` to match the rest of this file's inference tests (module doc
/// l12-17 mandates this); opt-in via `cargo test --ignored` once
/// `SPARROW_ENGINE_GPU_TEST_MODELS` resolves.
#[test]
#[ignore]
fn detect_with_raw_input_bgra_succeeds() {
    let manifest_path = models_dir()
        .join("megadetector-v6-yolov10e")
        .join("manifest.toml");
    if require_fixture(&manifest_path).is_none() {
        return;
    }
    let images = collect_jpegs(&corpus_dir(), 1);
    if images.is_empty() {
        eprintln!("skipping: no images in {}", corpus_dir().display());
        return;
    }

    // Decode JPEG to RGB then re-pack as BGRA (alpha = 0xFF). The classifier-side
    // RGB test exercises the 3-channel arms of `crate::decode::raw_to_gpu`; this
    // YOLO-side BGRA test exercises the 4-channel + byte-shuffle arm and the
    // dispatch site at `yolo.rs::detect` (B9).
    let dyn_img = image::ImageReader::open(&images[0])
        .expect("open jpeg")
        .with_guessed_format()
        .expect("guess fmt")
        .decode()
        .expect("decode");
    let rgb = dyn_img.to_rgb8();
    let (w, h) = (rgb.width(), rgb.height());
    let stride = w * 4;
    let mut bgra: Vec<u8> = Vec::with_capacity((w as usize) * (h as usize) * 4);
    for px in rgb.pixels() {
        bgra.push(px[2]); // B
        bgra.push(px[1]); // G
        bgra.push(px[0]); // R
        bgra.push(0xFF); // A
    }
    let input = ImageInput::Raw {
        data: bgra,
        width: w,
        height: h,
        stride,
        format: PixelFormat::Bgra,
    };

    let (ctx, kernel, model) = gpu_load_yolo(&manifest_path);
    let opts = DetectOpts::default();
    let encoded_input = ImageInput::Encoded(std::fs::read(&images[0]).expect("read encoded jpeg"));
    let encoded = model
        .detect(&ctx, &kernel, &encoded_input, &opts)
        .expect("detect(ImageInput::Encoded) baseline must succeed");
    let result = model
        .detect(&ctx, &kernel, &input, &opts)
        .expect("detect(ImageInput::Raw BGRA) must succeed post-B9");
    drop(model);
    drop(kernel);
    drop(ctx);

    assert_eq!(result.image_width, w);
    assert_eq!(result.image_height, h);
    assert!(
        !encoded.detections.is_empty(),
        "Encoded baseline should produce detections on the selected image"
    );
    assert!(
        !result.detections.is_empty(),
        "Raw BGRA detect expected non-empty detections"
    );
    let raw_top = &result.detections[0];
    let encoded_top = &encoded.detections[0];
    assert_eq!(
        raw_top.label_id, encoded_top.label_id,
        "Raw BGRA top detection label_id must match encoded baseline"
    );
    let bbox_l1 = (raw_top.bbox.x_min - encoded_top.bbox.x_min).abs()
        + (raw_top.bbox.y_min - encoded_top.bbox.y_min).abs()
        + (raw_top.bbox.x_max - encoded_top.bbox.x_max).abs()
        + (raw_top.bbox.y_max - encoded_top.bbox.y_max).abs();
    assert!(
        bbox_l1 <= 0.05,
        "Raw BGRA top bbox drifted from encoded baseline: raw={:?} encoded={:?} l1={bbox_l1}",
        raw_top.bbox,
        encoded_top.bbox
    );
    assert!(
        (raw_top.confidence - encoded_top.confidence).abs() <= 0.10,
        "Raw BGRA top confidence drifted from encoded baseline: raw={:.4} encoded={:.4}",
        raw_top.confidence,
        encoded_top.confidence
    );
    for det in &result.detections {
        assert!(
            det.confidence.is_finite(),
            "confidence must be finite: {det:?}"
        );
        assert!(
            (0.0..=1.0).contains(&det.confidence),
            "confidence out of range: {det:?}"
        );
        assert!(
            det.bbox.x_min <= det.bbox.x_max,
            "bbox x ordering invalid: {det:?}"
        );
        assert!(
            det.bbox.y_min <= det.bbox.y_max,
            "bbox y ordering invalid: {det:?}"
        );
        assert!(
            (0.0..=1.0).contains(&det.bbox.x_min),
            "bbox x_min out of range: {det:?}"
        );
        assert!(
            (0.0..=1.0).contains(&det.bbox.x_max),
            "bbox x_max out of range: {det:?}"
        );
        assert!(
            (0.0..=1.0).contains(&det.bbox.y_min),
            "bbox y_min out of range: {det:?}"
        );
        assert!(
            (0.0..=1.0).contains(&det.bbox.y_max),
            "bbox y_max out of range: {det:?}"
        );
    }
    eprintln!(
        "detect_with_raw_input_bgra_succeeds: {}x{} via Raw BGRA → {} detections",
        result.image_width,
        result.image_height,
        result.detections.len(),
    );
}
