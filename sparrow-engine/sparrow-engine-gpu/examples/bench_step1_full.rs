//! Phase 3.8 Step 1 final bench harness — sparrow-engine-gpu side.
//!
//! Single-cell driver: loads ONE model + runs the configured corpus, prints
//! per-image timing + detection counts as JSON to stdout. The Python parent
//! (`scripts/bench_step1_full.py`) spawns this binary 5× per (engine,
//! precision, model) cell to satisfy the variance discipline mandate from
//! `feedback_perf_claims_need_variance.md`.
//!
//! ## Why an example, not a test
//!
//! Existing `sparrow-engine-gpu/tests/` benches run 100 iters on 1 image (single-image
//! latency). This binary runs N≥100 different images so we can compare apples
//! to apples against PW PyTorch (which is benched per-image over the same
//! corpus). It is an `examples/` binary so:
//! - it can `use sparrow_engine_cpu` (already a dev-dep) for cross-engine bench,
//! - it doesn't enter `tests/` (avoids file-conflict with coder-fp16's
//!   in-flight FP16 manifest work on `tests/integration_*.rs`),
//! - `cargo run --release -p sparrow-engine-gpu --example bench_step1_full` keeps the
//!   compile cache between fresh invocations (cargo test re-discovers tests
//!   each time).
//!
//! ## Env vars
//!
//! Required:
//! - `SPARROW_ENGINE_BENCH_MODEL` — `mdv6` | `deepfaune` | `herdnet` | `owl-t` | `speciesnet` | `amazon`
//! - `SPARROW_ENGINE_BENCH_MANIFEST` — path to `manifest.toml`
//! - `SPARROW_ENGINE_BENCH_CORPUS` — directory of JPEG images
//! - `SPARROW_ENGINE_BENCH_OUTFILE` — path where JSON results are written
//!
//! Optional:
//! - `SPARROW_ENGINE_BENCH_N_IMAGES` — how many images to run (default 100)
//! - `SPARROW_ENGINE_BENCH_WARMUP` — warmup iters on the first image (default 5)
//!
//! Single-output bench-record JSON shape:
//! ```text
//! {
//!   "model_kind": "yolo|tiled|classifier",
//!   "model_id": "...",
//!   "manifest_path": "...",
//!   "corpus_dir": "...",
//!   "n_images": 100,
//!   "warmup_iters": 5,
//!   "per_image_ms": [...],
//!   "detection_counts": [...],
//!   "total_detections": 243,
//!   "image_paths": [...]
//! }
//! ```
//!
//! Run:
//! ```bash
//! SPARROW_ENGINE_BENCH_MODEL=mdv6 \
//!   SPARROW_ENGINE_BENCH_MANIFEST=/.../sparrow_engine_models/megadetector-v6-yolov10e/manifest.toml \
//!   SPARROW_ENGINE_BENCH_CORPUS=/.../test_cameratrap \
//!   SPARROW_ENGINE_BENCH_OUTFILE=/tmp/cell.json \
//!   cargo run --release -p sparrow-engine-gpu --example bench_step1_full
//! ```

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use sparrow_engine::kernels::center_crop::CenterCropKernel;
use sparrow_engine::kernels::letterbox::LetterboxKernel;
use sparrow_engine::kernels::resize::ResizeKernel;
use sparrow_engine::models::classifier::{ClassifierModel, JpegDecoder};
use sparrow_engine::models::tiled::TiledModel;
use sparrow_engine::models::yolo::YoloModel;
use sparrow_engine_types::manifest::load_manifest;
use sparrow_engine_types::{ClassifyOpts, DetectOpts, ImageInput};
use cudarc::driver::CudaContext;

/// Collect JPEGs from `dir`, sorted alphabetically. Truncates to `limit`.
fn collect_jpegs(dir: &Path, limit: usize) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|r| r.ok())
            .map(|e| e.path())
            .filter(|p| {
                let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("");
                matches!(ext.to_lowercase().as_str(), "jpg" | "jpeg")
            })
            .collect(),
        Err(_) => Vec::new(),
    };
    out.sort();
    out.truncate(limit);
    out
}

fn req_env(key: &str) -> Result<String, String> {
    env::var(key).map_err(|_| format!("missing required env var: {key}"))
}

fn opt_usize(key: &str, default: usize) -> usize {
    env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

#[derive(Debug, Clone, Copy)]
enum ModelKind {
    Yolo,      // mdv6, deepfaune
    Tiled,     // herdnet, owl-t
    Classifier, // speciesnet, amazon
}

fn model_kind_of(s: &str) -> Result<ModelKind, String> {
    match s {
        "mdv6" | "deepfaune" => Ok(ModelKind::Yolo),
        "herdnet" | "owl-t" => Ok(ModelKind::Tiled),
        "speciesnet" | "amazon" => Ok(ModelKind::Classifier),
        other => Err(format!(
            "unknown SPARROW_ENGINE_BENCH_MODEL '{other}': must be one of \
             mdv6, deepfaune, herdnet, owl-t, speciesnet, amazon"
        )),
    }
}

/// Bench result row: per-image latency in ms + detection / classification count.
///
/// `top1_label` and `top1_conf` are populated for classifier cells (Amazon)
/// and left empty for yolo/tiled cells. The Python parent uses them to
/// compute per-image top-1 match rate vs PW (PW returns top-1 only; sparrow-engine-gpu
/// returns top-5, so a raw count comparison is the wrong metric for classifier
/// cells — top-1 match rate + score Δ is the apples-to-apples comparison).
struct PerImage {
    latency_ms: f64,
    count: usize,
    top1_label: String,
    top1_conf: f32,
}

fn bench_yolo(
    manifest_path: &Path,
    images: &[PathBuf],
    warmup: usize,
) -> Result<Vec<PerImage>, String> {
    let manifest = load_manifest(manifest_path)
        .map_err(|e| format!("load_manifest({}): {e}", manifest_path.display()))?;
    let manifest_dir = manifest_path.parent().ok_or("manifest has no parent")?;
    let ctx = CudaContext::new(0).map_err(|e| format!("CudaContext::new: {e}"))?;
    let kernel = LetterboxKernel::new(&ctx).map_err(|e| format!("LetterboxKernel::new: {e}"))?;
    let model = YoloModel::load(&ctx, &manifest, manifest_dir)
        .map_err(|e| format!("YoloModel::load: {e}"))?;
    let opts = DetectOpts::default();

    // Lever C (Path 2 follow-up): pre-load all JPEG bytes outside the timed
    // loop. Mirrors the Phase 3.7 prototype's methodology
    // (`results.md § "Phase 3.8 viability check"`: "Read JPEG bytes ...
    // CPU | excluded from per-image timing (pre-loaded all 100)").
    // Removes ~0.1 ms / image of fs::read overhead from per-image readings
    // so sparrow-engine-gpu numbers compare apples-to-apples against the prototype
    // 11.11 ms reference.
    let preloaded: Vec<Vec<u8>> = images
        .iter()
        .map(|p| std::fs::read(p).map_err(|e| format!("preload {}: {e}", p.display())))
        .collect::<Result<_, _>>()?;

    // Lever A (Path 2 follow-up): opt-in pipelined-batch mode via env var
    // SPARROW_ENGINE_GPU_YOLO_BATCH_PIPELINE=1. Calls detect_batch_pipelined() which
    // overlaps nvjpeg + letterbox of image N+1 with ORT.run of image N on
    // a dedicated decode_stream. Per-image timing is captured from the
    // method's per-call boundary (start of detect_consume_and_prepare_next
    // through return) — same per-call boundary semantic as detect().
    if std::env::var("SPARROW_ENGINE_GPU_YOLO_BATCH_PIPELINE").as_deref() == Ok("1") {
        // Warmup on first image via the regular detect() (pipelined path
        // needs a min of 2 images to actually pipeline, and we want a
        // warmup phase that doesn't pollute the main timed batch).
        let first = ImageInput::Encoded(preloaded[0].clone());
        for _ in 0..warmup {
            let _ = model
                .detect(&ctx, &kernel,&first, &opts)
                .map_err(|e| format!("warmup detect (pipelined mode): {e}"))?;
        }
        // Pipelined batch run. Per-image timings are captured INSIDE
        // detect_batch_pipelined via the profile module (when
        // SPARROW_ENGINE_GPU_PROFILE_DUMP is set). For the bench summary's
        // per_image_ms we use the DetectResult.processing_time_ms.
        let bytes_refs: Vec<&[u8]> = preloaded.iter().map(|v| v.as_slice()).collect();
        let results = model
            .detect_batch_pipelined(&ctx, &kernel, bytes_refs.iter().copied(), &opts)
            .map_err(|e| format!("detect_batch_pipelined: {e}"))?;
        let out: Vec<PerImage> = results
            .into_iter()
            .map(|r| PerImage {
                latency_ms: r.processing_time_ms as f64,
                count: r.detections.len(),
                top1_label: String::new(),
                top1_conf: 0.0,
            })
            .collect();
        return Ok(out);
    }

    // Warmup on first image (also using the pre-loaded bytes).
    let first = ImageInput::Encoded(preloaded[0].clone());
    for _ in 0..warmup {
        let _ = model
            .detect(&ctx, &kernel,&first, &opts)
            .map_err(|e| format!("warmup detect: {e}"))?;
    }

    let mut out = Vec::with_capacity(images.len());
    for (p, bytes) in images.iter().zip(preloaded.iter()) {
        let input = ImageInput::Encoded(bytes.clone());
        let t0 = Instant::now();
        let r = model
            .detect(&ctx, &kernel,&input, &opts)
            .map_err(|e| format!("detect {}: {e}", p.display()))?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        out.push(PerImage {
            latency_ms: dt,
            count: r.detections.len(),
            top1_label: String::new(),
            top1_conf: 0.0,
        });
    }
    Ok(out)
}

fn bench_tiled(
    manifest_path: &Path,
    images: &[PathBuf],
    warmup: usize,
) -> Result<Vec<PerImage>, String> {
    let ctx = CudaContext::new(0).map_err(|e| format!("CudaContext::new: {e}"))?;
    let model = TiledModel::load_from_path(&ctx, manifest_path)
        .map_err(|e| format!("TiledModel::load_from_path: {e}"))?;
    let opts = DetectOpts::default();

    // Warmup on first image (single iteration; tiled is already 30+ tiles
    // worth of inference per image so per-image warmup is heavy enough to
    // settle cuDNN algo selection).
    let first = ImageInput::FilePath(images[0].clone());
    for _ in 0..warmup {
        let _ = model
            .detect_tiled(&ctx, &first, &opts)
            .map_err(|e| format!("warmup detect_tiled: {e}"))?;
    }

    let mut out = Vec::with_capacity(images.len());
    for p in images {
        let input = ImageInput::FilePath(p.clone());
        let t0 = Instant::now();
        let r = model
            .detect_tiled(&ctx, &input, &opts)
            .map_err(|e| format!("detect_tiled {}: {e}", p.display()))?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        out.push(PerImage {
            latency_ms: dt,
            count: r.detections.len(),
            top1_label: String::new(),
            top1_conf: 0.0,
        });
    }
    Ok(out)
}

fn bench_classifier(
    manifest_path: &Path,
    images: &[PathBuf],
    warmup: usize,
) -> Result<Vec<PerImage>, String> {
    let manifest = load_manifest(manifest_path)
        .map_err(|e| format!("load_manifest({}): {e}", manifest_path.display()))?;
    let manifest_dir = manifest_path.parent().ok_or("manifest has no parent")?;
    let ctx: Arc<CudaContext> = CudaContext::new(0).map_err(|e| format!("CudaContext::new: {e}"))?;
    let center_crop = CenterCropKernel::new(&ctx)
        .map_err(|e| format!("CenterCropKernel::new: {e}"))?;
    let resize = ResizeKernel::new(&ctx).map_err(|e| format!("ResizeKernel::new: {e}"))?;
    let mut decoder = JpegDecoder::new(&ctx).map_err(|e| format!("JpegDecoder::new: {e}"))?;
    let model = ClassifierModel::load(&ctx, &manifest, manifest_dir)
        .map_err(|e| format!("ClassifierModel::load: {e}"))?;
    let opts = ClassifyOpts { top_k: Some(5) };

    let first = ImageInput::FilePath(images[0].clone());
    for _ in 0..warmup {
        let _ = model
            .classify(&ctx, &center_crop, &resize, &mut decoder, &first, &opts)
            .map_err(|e| format!("warmup classify: {e}"))?;
    }

    let mut out = Vec::with_capacity(images.len());
    for p in images {
        let input = ImageInput::FilePath(p.clone());
        let t0 = Instant::now();
        let r = model
            .classify(&ctx, &center_crop, &resize, &mut decoder, &input, &opts)
            .map_err(|e| format!("classify {}: {e}", p.display()))?;
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        // For classifier, "count" is the number of classifications returned
        // (top-k is typically 5; we use this as a sanity sentinel).
        // top1_label / top1_conf carry the per-image top-1 prediction so the
        // Python parent can compute top-1 match rate + score Δ vs PW (PW
        // returns top-1 only; sparrow-engine-gpu returns top-5, so raw-count parity is
        // the wrong metric for classifier cells).
        let (top1_label, top1_conf) = match r.classifications.first() {
            Some(c) => (c.label.clone(), c.confidence),
            None => (String::new(), 0.0),
        };
        out.push(PerImage {
            latency_ms: dt,
            count: r.classifications.len(),
            top1_label,
            top1_conf,
        });
    }
    Ok(out)
}

fn run() -> Result<(), String> {
    let model_str = req_env("SPARROW_ENGINE_BENCH_MODEL")?;
    let manifest_path = PathBuf::from(req_env("SPARROW_ENGINE_BENCH_MANIFEST")?);
    let corpus_dir = PathBuf::from(req_env("SPARROW_ENGINE_BENCH_CORPUS")?);
    let outfile = PathBuf::from(req_env("SPARROW_ENGINE_BENCH_OUTFILE")?);
    let n_images = opt_usize("SPARROW_ENGINE_BENCH_N_IMAGES", 100);
    let warmup = opt_usize("SPARROW_ENGINE_BENCH_WARMUP", 5);

    let kind = model_kind_of(&model_str)?;
    if !manifest_path.is_file() {
        return Err(format!("manifest not found: {}", manifest_path.display()));
    }
    if !corpus_dir.is_dir() {
        return Err(format!("corpus not found: {}", corpus_dir.display()));
    }

    let images = collect_jpegs(&corpus_dir, n_images);
    if images.is_empty() {
        return Err(format!("no JPEGs in corpus {}", corpus_dir.display()));
    }
    eprintln!(
        "[bench_step1_full] model={model_str} kind={:?} images={} warmup={} manifest={}",
        kind,
        images.len(),
        warmup,
        manifest_path.display()
    );

    let per = match kind {
        ModelKind::Yolo => bench_yolo(&manifest_path, &images, warmup)?,
        ModelKind::Tiled => bench_tiled(&manifest_path, &images, warmup)?,
        ModelKind::Classifier => bench_classifier(&manifest_path, &images, warmup)?,
    };

    let total_dets: usize = per.iter().map(|x| x.count).sum();
    eprintln!(
        "[bench_step1_full] DONE n={} total_count={}",
        per.len(),
        total_dets
    );

    // Hand-rolled JSON (no serde_json dep needed at example level — keep deps lean).
    let mut json = String::new();
    json.push('{');
    json.push_str(&format!(r#""model_kind":{:?},"#, format!("{kind:?}").to_lowercase()));
    json.push_str(&format!(r#""model":{:?},"#, model_str));
    json.push_str(&format!(r#""manifest_path":{:?},"#, manifest_path.display().to_string()));
    json.push_str(&format!(r#""corpus_dir":{:?},"#, corpus_dir.display().to_string()));
    json.push_str(&format!(r#""n_images":{},"#, per.len()));
    json.push_str(&format!(r#""warmup_iters":{},"#, warmup));
    json.push_str(&format!(r#""total_count":{},"#, total_dets));
    json.push_str(r#""per_image_ms":["#);
    for (i, p) in per.iter().enumerate() {
        if i > 0 { json.push(','); }
        json.push_str(&format!("{:.4}", p.latency_ms));
    }
    json.push_str(r#"],"detection_counts":["#);
    for (i, p) in per.iter().enumerate() {
        if i > 0 { json.push(','); }
        json.push_str(&p.count.to_string());
    }
    // Per-image top-1 label + confidence (populated for classifier cells;
    // empty + 0.0 for yolo/tiled cells). Used by the Python parent to compute
    // per-image top-1 match rate + score Δ for Amazon classify cells.
    json.push_str(r#"],"top1_labels":["#);
    for (i, p) in per.iter().enumerate() {
        if i > 0 { json.push(','); }
        let s = &p.top1_label;
        if s.contains('\\') || s.contains('"') {
            return Err(format!("top1_label contains \\ or \": {s}"));
        }
        json.push('"');
        json.push_str(s);
        json.push('"');
    }
    json.push_str(r#"],"top1_confs":["#);
    for (i, p) in per.iter().enumerate() {
        if i > 0 { json.push(','); }
        json.push_str(&format!("{:.6}", p.top1_conf));
    }
    json.push_str(r#"],"image_paths":["#);
    for (i, p) in images.iter().enumerate() {
        if i > 0 { json.push(','); }
        // Escape backslashes/quotes the lazy way: bail if any image path
        // contains them. Camera-trap fixtures don't have such names.
        let s = p.display().to_string();
        if s.contains('\\') || s.contains('"') {
            return Err(format!("image path contains \\ or \": {s}"));
        }
        json.push('"');
        json.push_str(&s);
        json.push('"');
    }
    json.push_str("]}");

    std::fs::write(&outfile, json)
        .map_err(|e| format!("write {}: {e}", outfile.display()))?;

    // Dump per-stage profile records if SPARROW_ENGINE_GPU_PROFILE_DUMP is set.
    sparrow_engine::profile::dump_to_path();

    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("[bench_step1_full] ERROR: {e}");
            ExitCode::from(1)
        }
    }
}
