//! spe CLI — Command-line interface for the sparrow-engine wildlife inference engine.
//!
//! 8 commands matching the Phase 2.5 MVP function list:
//! detect, classify, detect-audio, pipeline, models list, models info, device, init.
//!
//! # Phase 3.8 Phase C Wave 3 — engine flavor
//!
//! This crate produces two binaries from one source via Cargo features:
//! - `spe` — CPU flavor (`--features cpu`, default).
//! - `spe-gpu` — GPU flavor (`--features gpu`).
//!
//! The active engine is selected by the `engine_dispatch` shim
//! ([`crate::engine_dispatch`]). The rest of this file is engine-agnostic
//! — it speaks to `engine_dispatch::*` paths (an alias for `engine_dispatch`) and
//! cargo compile-time dispatch resolves them to either `sparrow-engine-cpu` or
//! `sparrow-engine-gpu`.

mod engine_dispatch;
mod ort_resolver;

// Mutual exclusivity + presence guard: cargo's `required-features`
// keys on each `[[bin]]` already prevent builds with neither feature,
// but a workspace-wide `cargo build --features cpu --features gpu`
// would otherwise compile both arms of `engine_dispatch` and fail with
// a duplicate-glob error. The `compile_error!` makes the failure mode
// explicit.
#[cfg(all(feature = "cpu", feature = "gpu"))]
compile_error!("sparrow-engine-cli: features `cpu` and `gpu` are mutually exclusive");

#[cfg(not(any(feature = "cpu", feature = "gpu")))]
compile_error!("sparrow-engine-cli: one of `cpu` or `gpu` must be enabled (default = cpu)");

// Flavor-driven program name + version string for clap's --version output.
// Per `phase_c/implementation_plan.md` §4 Wave 3 deliverables, the CPU
// binary identifies itself as `spe` and the GPU binary as `spe-gpu`,
// each appending `(CPU flavor)` / `(GPU flavor)` to the version line.
#[cfg(feature = "cpu")]
const PROG_NAME: &str = "spe";
#[cfg(feature = "gpu")]
const PROG_NAME: &str = "spe-gpu";

#[cfg(feature = "cpu")]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (CPU flavor)");
#[cfg(feature = "gpu")]
const VERSION: &str = concat!(env!("CARGO_PKG_VERSION"), " (GPU flavor)");

use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

// Pull engine-side types/functions in via the dispatch shim. Under
// either feature set they come from the `sparrow_engine` crate link,
// backed by `sparrow-engine-cpu` or `sparrow-engine-gpu`. The rest of
// this file uses `engine_dispatch::*` paths directly — no backward-compat
// alias.
use crate::engine_dispatch::{
    classify, detect, detect_audio, AudioDetectOpts, AudioDetectResult, AudioInput, ClassifyOpts,
    ClassifyResult, DetectOpts, DetectResult, Device, Engine, EngineConfig, ImageInput, ModelInfo,
    ModelType, PipelineResult,
};
use clap::{Parser, Subcommand};
use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use serde::Serialize;

// ---------------------------------------------------------------------------
// CLI argument definitions
// ---------------------------------------------------------------------------

#[derive(Parser)]
#[command(name = PROG_NAME, version = VERSION, about = "Wildlife inference engine CLI")]
struct Cli {
    /// Compute device: auto, cpu, or cuda:N
    #[arg(long, global = true, default_value = "auto")]
    device: String,

    /// Base directory for model manifests
    #[arg(long, global = true)]
    model_dir: Option<PathBuf>,

    /// Suppress the progress bar on batch commands (detect, classify,
    /// detect-audio, pipeline). The bar is also suppressed automatically
    /// when stderr is not a TTY.
    #[arg(long, global = true)]
    quiet: bool,

    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run object detection on images
    Detect(DetectArgs),
    /// Run classification on images
    Classify(ClassifyArgs),
    /// Run audio detection on audio files
    DetectAudio(DetectAudioArgs),
    /// Run detect -> classify pipeline on images
    Pipeline(PipelineArgs),
    /// Model management commands
    Models {
        #[command(subcommand)]
        action: ModelsAction,
    },
    /// Show active compute device
    Device,
    /// Initialize the engine (uses global --device and --model-dir)
    Init,
    /// Compute SHA-256 hash of a file
    Hash(HashArgs),
    /// Classify image as day or night
    DayNight(DayNightArgs),
}

#[derive(clap::Args)]
struct DetectArgs {
    /// Input files or directories
    #[arg(required = true)]
    input: Vec<PathBuf>,
    /// Model ID to use
    #[arg(long)]
    model: Option<String>,
    /// Confidence threshold
    #[arg(long)]
    threshold: Option<f32>,
    /// Maximum detections per image
    #[arg(long)]
    max_detections: Option<u32>,
    /// Print per-file results to stdout (one JSON object or CSV row block per file).
    /// Default off — only the progress bar shows. Use --export-format for
    /// consolidated batch output.
    #[arg(long)]
    print: bool,
    /// Output format for --print (no effect without --print)
    #[arg(long, default_value = "json")]
    format: OutputFormat,
    /// Recurse into subdirectories
    #[arg(long)]
    recursive: bool,
    /// Print detection summary statistics after processing
    #[arg(long)]
    summary: bool,
    /// Export format: megadet, coco, or csv
    #[arg(long)]
    export_format: Option<ExportFormat>,
    /// Output path for export (defaults to stdout)
    #[arg(long)]
    export_output: Option<PathBuf>,
    /// Visualize results on source images
    #[arg(long, requires = "output_dir")]
    visualize: bool,
    /// Output directory for visualization images (required with --visualize)
    #[arg(long, requires = "visualize")]
    output_dir: Option<PathBuf>,
    /// Render `"{label} {conf:.2}"` text above each bbox (default off).
    /// No effect on overhead-detector dot output.
    #[arg(long, requires = "visualize")]
    show_labels: bool,
}

#[derive(Clone, clap::ValueEnum)]
enum ExportFormat {
    Megadet,
    Coco,
    Csv,
}

#[derive(clap::Args)]
struct ClassifyArgs {
    /// Input files or directories
    #[arg(required = true)]
    input: Vec<PathBuf>,
    /// Model ID to use
    #[arg(long)]
    model: Option<String>,
    /// Number of top predictions to return
    #[arg(long)]
    top_k: Option<u32>,
    /// Print per-file results to stdout (one JSON object or CSV row block per file).
    /// Default off — only the progress bar shows.
    #[arg(long)]
    print: bool,
    /// Output format for --print (no effect without --print)
    #[arg(long, default_value = "json")]
    format: OutputFormat,
    /// Recurse into subdirectories
    #[arg(long)]
    recursive: bool,
    /// Visualize results on source images
    #[arg(long, requires = "output_dir")]
    visualize: bool,
    /// Output directory for visualization images (required with --visualize)
    #[arg(long, requires = "visualize")]
    output_dir: Option<PathBuf>,
    /// Render `"{label} {conf:.2}"` text above each bbox (default off).
    #[arg(long, requires = "visualize")]
    show_labels: bool,
}

#[derive(clap::Args)]
struct DetectAudioArgs {
    /// Input audio files or directories
    #[arg(required = true)]
    input: Vec<PathBuf>,
    /// Model ID to use
    #[arg(long)]
    model: Option<String>,
    /// Confidence threshold
    #[arg(long)]
    threshold: Option<f32>,
    /// Print per-file results to stdout (one JSON object or CSV row block per file).
    /// Default off — only the progress bar shows.
    #[arg(long)]
    print: bool,
    /// Output format for --print (no effect without --print)
    #[arg(long, default_value = "json")]
    format: OutputFormat,
    /// Recurse into subdirectories
    #[arg(long)]
    recursive: bool,
    /// Emit one row per sliding-window segment (the pre-Phase-3.5 default
    /// output). Without this flag, consecutive above-threshold windows are
    /// merged into `AudioRange`s (`start_time_s`, `end_time_s`,
    /// `max_confidence`, `class`). Set this when a downstream script
    /// parses the old per-window format.
    #[arg(long)]
    raw_segments: bool,
    /// Render audio confidence heatmap (with merged-range bars overlaid by
    /// default; plain heatmap under `--raw-segments`) per input file as PNG.
    #[arg(long)]
    visualize: bool,
    /// Output directory for visualization images (required with --visualize)
    #[arg(long)]
    output_dir: Option<PathBuf>,
    /// Apply Gaussian blur smoothing to layer 03 / 04 of the visualization.
    /// Off by default — layer 03 renders the same discrete per-slot pattern
    /// as layer 02 unless this flag is set, so the user sees the actual
    /// per-slot confidence values without smoothing artefacts.
    #[arg(long)]
    smooth: bool,
    /// Emit `_02_segments_windows.png`: the segments image with a "window
    /// lanes" band appended below — each sliding-window as a thin horizontal
    /// line at its `[start_time_s, end_time_s]` x range, staggered into
    /// `ceil(window_s / stride_s)` lanes, coloured by `inferno(confidence)`.
    /// Diagnostic for verifying window placement and per-window confidence.
    #[arg(long)]
    show_windows: bool,
    /// Override the sliding-window stride (in seconds). The manifest provides
    /// a default; this flag is the runtime override. Stride is engine policy:
    /// it never has to match a model architecture constraint, so any positive
    /// value is accepted (validated > 0).
    #[arg(long)]
    stride: Option<f32>,
    /// Override the sliding-window segment duration (in seconds). The
    /// manifest provides a default; this flag is the runtime override.
    /// Honored by mel-spectrogram audio models with dynamic ONNX time-axis
    /// (e.g. md-audiobirds-v1). Silently ignored by raw-audio classifiers
    /// whose ONNX input is fixed-size (e.g. perch-v2's `[batch, 160000]`) —
    /// the window is an upstream architecture constraint for those models.
    #[arg(long = "segment-duration")]
    segment_duration_s: Option<f32>,
}

#[derive(clap::Args)]
struct PipelineArgs {
    /// Input files or directories
    #[arg(required = true)]
    input: Vec<PathBuf>,
    /// Detector model ID
    #[arg(long, required = true)]
    detector: String,
    /// Classifier model ID
    #[arg(long, required = true)]
    classifier: String,
    /// Detection confidence threshold
    #[arg(long)]
    threshold: Option<f32>,
    /// Top-k classifications per detection
    #[arg(long)]
    top_k: Option<u32>,
    /// Print per-file results to stdout (one JSON object or CSV row block per file).
    /// Default off — only the progress bar shows. Use --export-format for
    /// consolidated batch output.
    #[arg(long)]
    print: bool,
    /// Output format for --print (no effect without --print)
    #[arg(long, default_value = "json")]
    format: OutputFormat,
    /// Recurse into subdirectories
    #[arg(long)]
    recursive: bool,
    /// Visualize results on source images
    #[arg(long, requires = "output_dir")]
    visualize: bool,
    /// Output directory for visualization images (required with --visualize)
    #[arg(long, requires = "visualize")]
    output_dir: Option<PathBuf>,
    /// Render `"{label} {conf:.2}"` text above each bbox (default off).
    /// No effect on overhead-detector dot output.
    #[arg(long, requires = "visualize")]
    show_labels: bool,
    /// Export format: megadet, coco, or csv
    #[arg(long)]
    export_format: Option<ExportFormat>,
    /// Output path for export (defaults to stdout)
    #[arg(long)]
    export_output: Option<PathBuf>,
}

#[derive(clap::Args)]
struct HashArgs {
    /// File to hash
    file: PathBuf,
}

#[derive(clap::Args)]
struct DayNightArgs {
    /// Image file to classify
    image: PathBuf,
}

#[derive(Subcommand)]
enum ModelsAction {
    /// List loaded models
    List,
    /// Show info for a specific loaded model
    Info {
        /// Model ID
        model_id: String,
    },
    /// Verify model integrity against manifest checksums
    Verify {
        /// Model ID (optional; verifies all if omitted)
        model_id: Option<String>,
        /// Compute and write checksums to manifest.
        /// Warning: not safe to run concurrently — manifest.toml writes have no cross-process lock.
        #[arg(long)]
        write: bool,
    },
}

#[derive(Clone, clap::ValueEnum)]
enum OutputFormat {
    Json,
    Csv,
}

// ---------------------------------------------------------------------------
// JSON output types (serde)
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct DetectOutput {
    file: String,
    model_id: String,
    image_size: [u32; 2],
    detections: Vec<DetectionOutput>,
}

#[derive(Serialize)]
struct DetectionOutput {
    label: String,
    confidence: f32,
    bbox: BBoxOutput,
}

#[derive(Serialize)]
struct BBoxOutput {
    x_min: f32,
    y_min: f32,
    x_max: f32,
    y_max: f32,
}

#[derive(Serialize)]
struct ClassifyOutput {
    file: String,
    model_id: String,
    image_size: [u32; 2],
    classifications: Vec<ClassificationOutput>,
}

#[derive(Serialize)]
struct ClassificationOutput {
    label: String,
    confidence: f32,
}

/// Per-window audio output (pre-Phase-3.5 default; now opt-in via
/// `--raw-segments`). Schema: `segments: [{start_time_s, end_time_s,
/// confidence, classes?}]`; `classes` is emitted only for multi-class
/// classifiers.
#[derive(Serialize)]
struct AudioDetectRawOutput {
    file: String,
    model_id: String,
    duration_s: f32,
    sample_rate: u32,
    segments: Vec<AudioSegmentOutput>,
}

#[derive(Serialize)]
struct AudioClassOutput {
    class_idx: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    label: Option<String>,
    probability: f32,
}

#[derive(Serialize)]
struct AudioSegmentOutput {
    start_time_s: f32,
    end_time_s: f32,
    confidence: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    classes: Option<Vec<AudioClassOutput>>,
}

/// Merged-range audio output (Phase 3.5 default; item #6). Schema:
/// `ranges: [{start_time_s, end_time_s, max_confidence, class}]`.
/// `class` is `null` for binary audio detectors.
#[derive(Serialize)]
struct AudioDetectMergedOutput {
    file: String,
    model_id: String,
    duration_s: f32,
    sample_rate: u32,
    ranges: Vec<AudioRangeOutput>,
}

#[derive(Serialize)]
struct AudioRangeOutput {
    start_time_s: f32,
    end_time_s: f32,
    max_confidence: f32,
    class: Option<String>,
}

#[derive(Serialize)]
struct PipelineOutput {
    file: String,
    pipeline_id: String,
    image_size: [u32; 2],
    detections: Vec<PipelineDetectionOutput>,
}

#[derive(Serialize)]
struct PipelineDetectionOutput {
    label: String,
    confidence: f32,
    bbox: BBoxOutput,
    classification: Option<ClassificationOutput>,
}

#[derive(Serialize)]
struct ModelInfoOutput {
    id: String,
    path: String,
    model_type: String,
    #[serde(skip_serializing_if = "std::ops::Not::not")]
    default: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    version: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    description: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    onnx_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    onnx_size_bytes: Option<u64>,
}

#[derive(Serialize)]
struct DeviceOutput {
    device: String,
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

/// Restore default SIGPIPE handler so `spe ... | head` exits cleanly (141)
/// instead of surfacing EPIPE as a fatal error. Rust disables SIGPIPE at
/// startup; most Unix CLIs want it enabled so downstream pipe-closes
/// (head, less, grep | head) terminate the upstream process silently.
#[cfg(unix)]
fn reset_sigpipe() {
    extern "C" {
        fn signal(signum: i32, handler: usize) -> usize;
    }
    const SIGPIPE: i32 = 13;
    const SIG_DFL: usize = 0;
    unsafe {
        signal(SIGPIPE, SIG_DFL);
    }
}

#[cfg(not(unix))]
fn reset_sigpipe() {}

fn main() {
    reset_sigpipe();
    // RP-4 (Path B, tarball CLI): resolve bundled libonnxruntime from
    // <bundle_root>/lib/ before any engine call. No-op when running from
    // `cargo run` / system install / when ORT_DYLIB_PATH is already set.
    // Must run before init_tracing (which itself reads RUST_LOG, not ORT
    // env, but ordering is cheap insurance) and before any clap or engine
    // touchpoint. Single-threaded program entry — set_var is sound here.
    ort_resolver::init_ort_env();
    init_tracing();
    let cli = Cli::parse();

    if let Err(e) = run(cli) {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

/// Wire a stderr tracing subscriber so warn!/error! events from the engine
/// (e.g. broken manifest skip in `catalog::list_available_models`) surface to
/// the user. Defaults to `warn`; user-configurable via `RUST_LOG`.
fn init_tracing() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(filter)
        .with_target(false) // suppress module-path prefix (e.g. "sparrow_engine_core::catalog: ")
        .try_init();
}

fn run(cli: Cli) -> Result<(), Box<dyn std::error::Error>> {
    match cli.command {
        Commands::Detect(args) => cmd_detect(&cli.device, &cli.model_dir, cli.quiet, args),
        Commands::Classify(args) => cmd_classify(&cli.device, &cli.model_dir, cli.quiet, args),
        Commands::DetectAudio(args) => {
            cmd_detect_audio(&cli.device, &cli.model_dir, cli.quiet, args)
        }
        Commands::Pipeline(args) => cmd_pipeline(&cli.device, &cli.model_dir, cli.quiet, args),
        Commands::Models { action } => cmd_models(&cli.device, &cli.model_dir, action),
        Commands::Device => cmd_device(&cli.device, &cli.model_dir),
        Commands::Init => cmd_init(&cli.device, &cli.model_dir),
        Commands::Hash(args) => cmd_hash(args),
        Commands::DayNight(args) => cmd_day_night(args),
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a progress bar for a batch operation (detect / classify /
/// detect-audio / pipeline).
///
/// Returns a hidden [`ProgressBar`] when any of these is true:
///   - `quiet` is set (user passed `--quiet`);
///   - stderr is not a TTY (piped/redirected output);
///   - `total == 0` (the file loop is a no-op anyway).
///
/// A hidden bar still exposes `inc()` / `finish_and_clear()` as no-ops,
/// so callers do not need to branch.
///
/// Template: `[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta}, {per_sec}) {msg}`.
/// `{msg}` carries the current filename. The bar redraws at up to 10 Hz
/// and, crucially for this CLI, renders to **stderr** — `stdout` is
/// reserved for inference output (JSON / CSV) so callers can redirect it
/// without the bar polluting the data stream.
fn make_progress_bar(total: u64, quiet: bool) -> ProgressBar {
    let hide = quiet || !io::stderr().is_terminal() || total == 0;
    if hide {
        return ProgressBar::hidden();
    }
    let bar = ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::stderr_with_hz(10));
    // Fall back to the default style if the template is malformed (belt-and-suspenders).
    let style = ProgressStyle::with_template(
        "[{elapsed_precise}] [{bar:40}] {pos}/{len} ({eta}, {per_sec}) {msg}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-");
    bar.set_style(style);
    // Force an initial tick so the bar renders even before the first
    // `inc()` on fast batches.
    bar.enable_steady_tick(Duration::from_millis(250));
    bar
}

/// Parse device string ("auto", "cpu", "cuda:0") into Device enum.
fn parse_device(s: &str) -> Result<Device, Box<dyn std::error::Error>> {
    match s.to_lowercase().as_str() {
        "auto" => Ok(Device::Auto),
        "cpu" => Ok(Device::Cpu),
        "gpu" | "cuda" => Ok(Device::Cuda(0)),
        other => {
            if let Some(idx) = other.strip_prefix("cuda:") {
                let n: u32 = idx.parse()?;
                Ok(Device::Cuda(n))
            } else {
                Err(format!("invalid device: '{s}'. Expected: auto, cpu, gpu, cuda, cuda:N").into())
            }
        }
    }
}

/// Resolve model directory: --model-dir > SPARROW_ENGINE_MODEL_DIR env > ~/.sparrow-engine/models.
fn resolve_model_dir(model_dir: &Option<PathBuf>) -> PathBuf {
    if let Some(dir) = model_dir {
        return dir.clone();
    }
    if let Ok(dir) = std::env::var("SPARROW_ENGINE_MODEL_DIR") {
        return PathBuf::from(dir);
    }
    dirs_default_model_dir()
}

fn dirs_default_model_dir() -> PathBuf {
    if let Some(home) = std::env::var_os("HOME") {
        PathBuf::from(home).join(".sparrow-engine").join("models")
    } else {
        PathBuf::from(".sparrow-engine").join("models")
    }
}

/// First-user hint shown when the resolved model directory is missing or empty.
/// Hardcoded URL + DOI (no env var workaround) so the message stays self-contained
/// and copy-pasteable on a fresh box. Bootstrap script lives in the sparrow-engine-dev
/// branch of microsoft/Pytorch-Wildlife.
const BOOTSTRAP_HINT: &str = "First run? Populate the model directory:\n  \
    bash -c \"$(curl -fsSL https://raw.githubusercontent.com/microsoft/Pytorch-Wildlife/sparrow-engine-dev/sparrow-engine/scripts/download_models.sh)\"\n\n\
    Or set SPARROW_ENGINE_MODEL_DIR to an existing model directory.\n\
    (Zenodo v0.4.0 bundle: https://doi.org/10.5281/zenodo.20360316)";

/// Surface an actionable error when the resolved model directory is missing or
/// contains no per-model `manifest.toml`. Intercepts BEFORE `Engine::new`, so the
/// user gets a setup hint instead of a cryptic `Model manifest not found: …`
/// from the engine's typed error.
fn check_model_dir_populated(dir: &Path) -> Result<(), Box<dyn std::error::Error>> {
    if !dir.exists() {
        return Err(format!(
            "No models found: directory does not exist: {}\n\n{}",
            dir.display(),
            BOOTSTRAP_HINT
        )
        .into());
    }
    let has_manifest = std::fs::read_dir(dir)
        .map(|rd| {
            rd.filter_map(|e| e.ok())
                .any(|e| e.path().join("manifest.toml").is_file())
        })
        .unwrap_or(false);
    if !has_manifest {
        return Err(format!(
            "No models found in {}.\n\n{}",
            dir.display(),
            BOOTSTRAP_HINT
        )
        .into());
    }
    Ok(())
}

/// Create an Engine with the given global options.
fn create_engine(
    device_str: &str,
    model_dir: &Option<PathBuf>,
) -> Result<Engine, Box<dyn std::error::Error>> {
    let device = parse_device(device_str)?;
    let dir = resolve_model_dir(model_dir);
    check_model_dir_populated(&dir)?;
    let config = EngineConfig::new(device, dir);
    let engine = Engine::new(config)?;
    Ok(engine)
}

/// Format ModelType for display.
fn model_type_display(mt: &ModelType) -> &'static str {
    match mt {
        ModelType::Detector => "detector",
        ModelType::OverheadDetector => "overhead_detector",
        ModelType::Classifier => "classifier",
        ModelType::AudioDetector => "audio_detector",
        ModelType::AudioClassifier => "audio_classifier",
    }
}

/// Resolve input paths: expand directories, collect files.
fn resolve_inputs(inputs: &[PathBuf], recursive: bool) -> Vec<PathBuf> {
    let image_exts: &[&str] = &["jpg", "jpeg", "png", "bmp", "tiff", "tif"];
    let mut files = Vec::new();
    let mut visited = std::collections::HashSet::new();

    for input in inputs {
        if input.is_file() {
            files.push(input.clone());
        } else if input.is_dir() {
            collect_files_from_dir(input, image_exts, recursive, &mut files, &mut visited);
        } else {
            eprintln!("warning: skipping non-existent path: {}", input.display());
        }
    }

    files.sort();
    files
}

/// Resolve audio inputs (WAV files only).
fn resolve_audio_inputs(inputs: &[PathBuf], recursive: bool) -> Vec<PathBuf> {
    let audio_exts: &[&str] = &["wav"];
    let mut files = Vec::new();
    let mut visited = std::collections::HashSet::new();

    for input in inputs {
        if input.is_file() {
            files.push(input.clone());
        } else if input.is_dir() {
            collect_files_from_dir(input, audio_exts, recursive, &mut files, &mut visited);
        } else {
            eprintln!("warning: skipping non-existent path: {}", input.display());
        }
    }

    files.sort();
    files
}

fn collect_files_from_dir(
    dir: &PathBuf,
    extensions: &[&str],
    recursive: bool,
    out: &mut Vec<PathBuf>,
    visited: &mut std::collections::HashSet<PathBuf>,
) {
    // Resolve symlinks to detect cycles.
    let canonical = match std::fs::canonicalize(dir) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("warning: cannot resolve directory {}: {e}", dir.display());
            return;
        }
    };
    if !visited.insert(canonical) {
        return; // cycle detected — already visited this directory
    }

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("warning: cannot read directory {}: {e}", dir.display());
            return;
        }
    };

    for entry in entries {
        // ReadDir::next() returns io::Result<DirEntry>; the previous
        // entries.flatten() silently dropped Err arms (EBADF or other
        // corrupted-DIR* errors). Surface them as a warn-and-skip so the
        // operator sees the dirent went missing. Mirrors R3 catalog.rs B5.
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!(
                    "warning: skipping unreadable entry in {}: {e}",
                    dir.display()
                );
                continue;
            }
        };
        let path = entry.path();
        // Path::is_dir() / Path::is_file() silently coerce EACCES/ELOOP/etc
        // to false (rustdoc: "convenience function that coerces errors to
        // false"). Use std::fs::metadata (= Path::metadata, FOLLOWS symlinks)
        // so chmod-000 child entries surface a warn line instead of being
        // silently dropped. Important: NOT entry.metadata() — DirEntry::
        // metadata is symlink_metadata-equivalent on Unix (rustdoc: "this
        // function will not traverse symlinks") and would silently drop
        // both dirsymlinks AND filesymlinks. CLI users curate test sets via
        // `ln -s /raw/IMG_*.jpg ./review/` constantly — and setup.sh:75-77
        // produces /tmp/sparrow_engine_test_10/ with 10 image symlinks; under the
        // wrong DirEntry::metadata, `spe detect /tmp/sparrow_engine_test_10` would
        // return 0 images. std::fs::metadata follows symlinks like the
        // original Path::is_file()/is_dir() did.
        let md = match std::fs::metadata(&path) {
            Ok(m) => m,
            Err(e) => {
                eprintln!(
                    "warning: skipping entry with unreadable metadata {}: {e}",
                    path.display()
                );
                continue;
            }
        };
        if md.is_dir() && recursive {
            collect_files_from_dir(&path, extensions, recursive, out, visited);
        } else if md.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if extensions.iter().any(|x| x.eq_ignore_ascii_case(ext)) {
                    out.push(path);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Visualization helpers
// ---------------------------------------------------------------------------

/// Validate that --output-dir is provided when --visualize is set.
fn validate_viz_args(
    visualize: bool,
    output_dir: &Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    if visualize && output_dir.is_none() {
        return Err("--output-dir is required when --visualize is set\n\
             hint: spe <cmd> <input> --visualize --output-dir ./spe_viz"
            .into());
    }
    if let Some(dir) = output_dir {
        std::fs::create_dir_all(dir)
            .map_err(|e| format!("cannot create output directory '{}': {e}", dir.display()))?;
    }
    Ok(())
}

/// Compute the longest common directory prefix of the given paths.
fn longest_common_prefix(paths: &[PathBuf]) -> PathBuf {
    if paths.is_empty() {
        return PathBuf::new();
    }
    let parents: Vec<PathBuf> = paths
        .iter()
        .filter_map(|p| p.parent())
        .map(|p| p.to_path_buf())
        .collect();
    if parents.is_empty() {
        return PathBuf::new();
    }
    let first: Vec<_> = parents[0].components().collect();
    let mut prefix_len = first.len();
    for parent in &parents[1..] {
        let other: Vec<_> = parent.components().collect();
        prefix_len = prefix_len.min(other.len());
        for i in 0..prefix_len {
            if first[i] != other[i] {
                prefix_len = i;
                break;
            }
        }
    }
    first[..prefix_len].iter().collect()
}

/// Create parent directories for `path` if missing. No-op when path has no
/// parent or parent is the empty path (e.g., bare filename "out.json").
/// Used before opening a user-provided output file so nested paths like
/// `./nested/out.json` work even when `./nested/` does not exist.
fn ensure_parent_dir(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    Ok(())
}

/// Compute the output path for a visualization file.
fn viz_output_path(image_path: &Path, output_dir: &Path, common_prefix: &Path) -> PathBuf {
    let relative = match image_path.strip_prefix(common_prefix) {
        Ok(r) => r.to_path_buf(),
        Err(_) => {
            // Cross-root fallback: use filename only to avoid joining absolute paths.
            PathBuf::from(image_path.file_name().unwrap_or_default())
        }
    };
    let stem = relative.file_stem().unwrap_or_default().to_string_lossy();
    // Lowercase the extension so JPG/JPEG/PNG inputs all yield jpg/jpeg/png
    // outputs. Mirrors `sparrow-engine-python::viz_output_extension`'s lowercase
    // guarantee — keeps CLI and Python visualisation paths in sync per the
    // functionality-consistency rule.
    let ext = image_path
        .extension()
        .unwrap_or_default()
        .to_string_lossy()
        .to_lowercase();
    let parent = relative.parent().unwrap_or(Path::new(""));
    output_dir.join(parent).join(format!("{stem}_viz.{ext}"))
}

/// Load source image, render annotations, save to output path with directory mirroring.
fn save_visualization(
    image_path: &Path,
    annotations: &[engine_dispatch::viz::BboxAnnotation],
    output_dir: &Path,
    common_prefix: &Path,
    model_type: engine_dispatch::types::ModelType,
    show_labels: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    let img = image::open(image_path)?;
    let opts = engine_dispatch::viz::RenderOpts {
        model_type,
        show_labels,
        ..Default::default()
    };
    let rendered = engine_dispatch::viz::render(&img, annotations, &opts);
    let out_path = viz_output_path(image_path, output_dir, common_prefix);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    rendered.save(&out_path)?;
    eprintln!("viz: {}", out_path.display());
    Ok(())
}

/// Returns true iff every visualization failed and there was at least one
/// input. Used to decide if the CLI should exit with an error after viz.
/// Partial failure prints per-file warnings + summary but still exits 0.
fn is_all_viz_failed(viz_fail: usize, collected_len: usize) -> bool {
    viz_fail == collected_len && collected_len > 0
}

/// Compute the audio visualization output path with a layer suffix. Always
/// `.png`; mirrors directory structure relative to `common_prefix`.
///
/// Layered viz produces multiple PNGs per audio input (spec / segments /
/// heatmap / full) so the tester can isolate where a perceived bug lives.
fn audio_viz_output_path(
    audio_path: &Path,
    output_dir: &Path,
    common_prefix: &Path,
    layer_suffix: &str,
) -> PathBuf {
    let relative = match audio_path.strip_prefix(common_prefix) {
        Ok(r) => r.to_path_buf(),
        Err(_) => PathBuf::from(audio_path.file_name().unwrap_or_default()),
    };
    let stem = relative.file_stem().unwrap_or_default().to_string_lossy();
    let parent = relative.parent().unwrap_or(Path::new(""));
    output_dir
        .join(parent)
        .join(format!("{stem}_{layer_suffix}.png"))
}

/// Build a synthetic gray-gradient spectrogram backdrop sized for the given
/// audio duration. Fallback when the real mel-spectrogram render fails (e.g.,
/// non-WAV input or audio shorter than `n_fft`).
fn build_synthetic_spectrogram(duration_s: f32) -> image::DynamicImage {
    const PIXELS_PER_SECOND: f32 = 100.0;
    const HEIGHT: u32 = 120;
    let width = ((duration_s * PIXELS_PER_SECOND).max(50.0)) as u32;
    let mut img = image::RgbaImage::new(width, HEIGHT);
    for y in 0..HEIGHT {
        let v = (y as f32 / HEIGHT as f32 * 128.0) as u8 + 32;
        for x in 0..width {
            img.put_pixel(x, y, image::Rgba([v, v, v, 255]));
        }
    }
    image::DynamicImage::ImageRgba8(img)
}

/// Save one layer PNG. Creates parent directory if needed. Returns the
/// saved path so the caller can log it through the progress bar (raw
/// `eprintln!` interleaves badly with `indicatif`'s ANSI redraws — paths
/// get clobbered).
fn save_layer_png(
    img: &image::DynamicImage,
    audio_path: &Path,
    output_dir: &Path,
    common_prefix: &Path,
    layer_suffix: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let out_path = audio_viz_output_path(audio_path, output_dir, common_prefix, layer_suffix);
    if let Some(parent) = out_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    img.save(&out_path)?;
    Ok(out_path)
}

/// Configuration knobs for the layered audio visualization.
///
/// Bundled into a struct so `save_audio_visualization` stays under
/// clippy's `too_many_arguments` threshold and the CLI viz config can
/// flow through one type rather than ten positional args.
struct AudioVizParams<'a> {
    ranges: Option<&'a [engine_dispatch::detect_audio::AudioRange]>,
    audio_config: Option<&'a engine_dispatch::preprocess_audio::AudioPreprocessConfig>,
    window_s: f32,
    stride_s: f32,
    smooth: bool,
    show_windows: bool,
}

/// Render and save the layered audio visualization for one file.
///
/// Produces up to five PNGs so the tester can isolate which stage of the
/// pipeline a perceived bug lives in:
///
/// - `{stem}_01_spec.png` — raw mel spectrogram only (no overlays). If this
///   doesn't reflect the actual audio energy, the spectrogram render or the
///   manifest's preprocessing config is wrong.
/// - `{stem}_02_segments.png` — spectrogram + per-window confidence as
///   discrete inferno-coloured bars (`blur_passes=0`, no smoothing). Each
///   bar corresponds to one sliding-window segment from the model. If
///   segments don't track audio energy, the model is wrong.
/// - `{stem}_02_segments_windows.png` (only when `show_windows`) —
///   `_02_segments.png` with a "window lanes" band appended below. Each
///   sliding-window is a thin horizontal line spanning its `[start_time_s,
///   end_time_s]` x range, staggered across `ceil(window_s / stride_s)`
///   lanes by index, and filled with `inferno(seg.confidence)`. Time-aligned
///   with the spectrogram above; reads as a per-lane confidence chart over
///   time.
/// - `{stem}_03_heatmap.png` — spectrogram + smoothed inferno heatmap
///   (default blur). Same data as `_02_segments.png` but smoothed for
///   readability. Useful for spotting overall confidence patterns.
/// - `{stem}_04_full.png` — `_03_heatmap.png` + cyan merged-range bars.
///   Only emitted when `ranges` is `Some` (default merged mode). Lets the
///   tester see whether the merged ranges line up with where the model
///   actually places confidence.
fn save_audio_visualization(
    audio_path: &Path,
    result: &AudioDetectResult,
    params: &AudioVizParams<'_>,
    output_dir: &Path,
    common_prefix: &Path,
) -> Result<Vec<PathBuf>, Box<dyn std::error::Error>> {
    let spec = match params.audio_config {
        Some(cfg) => match engine_dispatch::viz::render_mel_spectrogram(audio_path, cfg) {
            Ok(img) => img,
            Err(e) => {
                eprintln!(
                    "viz: mel render failed for {}: {e} — falling back to synthetic backdrop",
                    audio_path.display()
                );
                build_synthetic_spectrogram(result.duration_s)
            }
        },
        None => build_synthetic_spectrogram(result.duration_s),
    };

    let opts = engine_dispatch::viz::AudioLayersOpts {
        smooth: params.smooth,
        show_windows: params.show_windows,
        window_s: params.window_s,
        stride_s: params.stride_s,
    };
    let layers = engine_dispatch::viz::render_audio_layers(
        &spec,
        &result.segments,
        params.ranges,
        result.duration_s,
        &opts,
    );

    let mut paths = Vec::with_capacity(layers.len());
    for (layer_suffix, img) in layers {
        paths.push(save_layer_png(
            &img,
            audio_path,
            output_dir,
            common_prefix,
            layer_suffix,
        )?);
    }

    Ok(paths)
}

/// Run visualization for a batch of collected results, printing a summary on failure.
fn run_visualization<R>(
    collected: &[(PathBuf, R)],
    to_annotations: impl Fn(&R) -> Vec<engine_dispatch::viz::BboxAnnotation>,
    output_dir: &Path,
    files: &[PathBuf],
    model_type: engine_dispatch::types::ModelType,
    show_labels: bool,
) -> usize {
    let common_prefix = longest_common_prefix(files);
    let mut viz_ok = 0usize;
    let mut viz_fail = 0usize;
    for (file, result) in collected {
        let annotations = to_annotations(result);
        match save_visualization(
            file,
            &annotations,
            output_dir,
            &common_prefix,
            model_type,
            show_labels,
        ) {
            Ok(()) => viz_ok += 1,
            Err(e) => {
                eprintln!("warning: visualization failed for {}: {e}", file.display());
                viz_fail += 1;
            }
        }
    }
    if viz_fail > 0 {
        eprintln!("{viz_ok} visualized, {viz_fail} failed");
    }
    viz_fail
}

// ---------------------------------------------------------------------------
// Command: detect
// ---------------------------------------------------------------------------

fn cmd_detect(
    device_str: &str,
    model_dir: &Option<PathBuf>,
    quiet: bool,
    args: DetectArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_viz_args(args.visualize, &args.output_dir)?;

    let files = resolve_inputs(&args.input, args.recursive);
    if files.is_empty() {
        return Err("No image files found.".into());
    }

    let engine = create_engine(device_str, model_dir)?;
    let model_id = args.model.as_deref().unwrap_or("megadetector-v6-yolov10e");
    let handle = engine.get_or_load_model(model_id)?;

    let opts = DetectOpts {
        confidence_threshold: args.threshold,
        max_detections: args.max_detections,
    };

    let total = files.len();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut errors = 0usize;
    let needs_collect = args.summary || args.export_format.is_some() || args.visualize;
    let mut collected: Vec<(PathBuf, DetectResult)> = Vec::new();

    if args.print && matches!(args.format, OutputFormat::Csv) {
        writeln!(
            out,
            "file,model_id,idx,label,confidence,x_min,y_min,x_max,y_max"
        )?;
    }

    let bar = make_progress_bar(total as u64, quiet);
    for file in &files {
        bar.set_message(file.display().to_string());
        let image = ImageInput::FilePath(file.clone());

        match detect::detect(&handle, &image, &opts) {
            Ok(result) => {
                if args.print {
                    write_detect_output(&mut out, file, model_id, &result, &args.format)?;
                }
                if needs_collect {
                    collected.push((file.clone(), result));
                }
            }
            Err(e) => {
                // Preserve a newline above the error line so it doesn't
                // blend into the progress bar under `--quiet` off-cases.
                bar.println(format!("error: {}: {e}", file.display()));
                errors += 1;
            }
        }
        bar.inc(1);
    }
    bar.finish_and_clear();

    if errors == total {
        return Err("All files failed processing.".into());
    }

    // Export if requested.
    if let Some(ref export_fmt) = args.export_format {
        let entries: Vec<(&std::path::Path, &DetectResult)> =
            collected.iter().map(|(p, r)| (p.as_path(), r)).collect();
        let mut export_writer: Box<dyn Write> = if let Some(ref path) = args.export_output {
            ensure_parent_dir(path)?;
            Box::new(std::fs::File::create(path)?)
        } else {
            Box::new(io::stdout().lock())
        };
        match export_fmt {
            ExportFormat::Megadet => {
                engine_dispatch::export::to_megadet(&entries, model_id, &mut export_writer)?
            }
            ExportFormat::Coco => engine_dispatch::export::to_coco(&entries, &mut export_writer)?,
            ExportFormat::Csv => engine_dispatch::export::to_csv(&entries, &mut export_writer)?,
        }
    }

    // Summary if requested.
    if args.summary {
        let detect_results: Vec<DetectResult> = collected.iter().map(|(_, r)| r.clone()).collect();
        let summary = engine_dispatch::stats::summarize_detections(&detect_results);
        eprintln!("--- Summary ---");
        eprintln!("  Total images: {}", summary.total_images);
        eprintln!("  With detections: {}", summary.images_with_detections);
        eprintln!("  Empty: {}", summary.empty_images);
        eprintln!("  Total detections: {}", summary.total_detections);
        eprintln!(
            "  Confidence: min={:.4} max={:.4} mean={:.4}",
            summary.confidence_min, summary.confidence_max, summary.confidence_mean
        );
        for (cat, stats) in &summary.per_category {
            eprintln!(
                "  {cat}: count={} conf min={:.4} max={:.4} mean={:.4}",
                stats.count, stats.confidence_min, stats.confidence_max, stats.confidence_mean
            );
        }
    }

    // Visualize if requested.
    if args.visualize {
        let output_dir = args.output_dir.as_ref().unwrap();
        let viz_fail = run_visualization(
            &collected,
            engine_dispatch::viz::detections_to_annotations,
            output_dir,
            &files,
            handle.model_type(),
            args.show_labels,
        );
        if is_all_viz_failed(viz_fail, collected.len()) {
            return Err("All visualizations failed".into());
        }
    }

    Ok(())
}

fn write_detect_output(
    out: &mut impl Write,
    file: &Path,
    model_id: &str,
    result: &DetectResult,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        OutputFormat::Json => {
            let output = DetectOutput {
                file: file.display().to_string(),
                model_id: model_id.to_string(),
                image_size: [result.image_width, result.image_height],
                detections: result
                    .detections
                    .iter()
                    .map(|d| DetectionOutput {
                        label: d.label.clone(),
                        confidence: d.confidence,
                        bbox: BBoxOutput {
                            x_min: d.bbox.x_min,
                            y_min: d.bbox.y_min,
                            x_max: d.bbox.x_max,
                            y_max: d.bbox.y_max,
                        },
                    })
                    .collect(),
            };
            serde_json::to_writer(&mut *out, &output)?;
            writeln!(out)?;
        }
        OutputFormat::Csv => {
            let file_str = engine_dispatch::export::csv_escape(&file.display().to_string());
            for (idx, d) in result.detections.iter().enumerate() {
                let label_str = engine_dispatch::export::csv_escape(&d.label);
                writeln!(
                    out,
                    "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
                    file_str,
                    engine_dispatch::export::csv_escape(model_id),
                    idx,
                    label_str,
                    d.confidence,
                    d.bbox.x_min,
                    d.bbox.y_min,
                    d.bbox.x_max,
                    d.bbox.y_max,
                )?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Command: classify
// ---------------------------------------------------------------------------

fn cmd_classify(
    device_str: &str,
    model_dir: &Option<PathBuf>,
    quiet: bool,
    args: ClassifyArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_viz_args(args.visualize, &args.output_dir)?;

    let files = resolve_inputs(&args.input, args.recursive);
    if files.is_empty() {
        return Err("No image files found.".into());
    }

    let engine = create_engine(device_str, model_dir)?;
    let model_id = args.model.as_deref().unwrap_or("speciesnet");
    let handle = engine.get_or_load_model(model_id)?;

    let opts = ClassifyOpts { top_k: args.top_k };

    let total = files.len();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut errors = 0usize;
    let needs_collect = args.visualize;
    let mut collected: Vec<(PathBuf, ClassifyResult)> = Vec::new();

    if args.print && matches!(args.format, OutputFormat::Csv) {
        writeln!(out, "file,model_id,idx,label,confidence")?;
    }

    let bar = make_progress_bar(total as u64, quiet);
    for file in &files {
        bar.set_message(file.display().to_string());
        let image = ImageInput::FilePath(file.clone());

        match classify::classify(&handle, &image, &opts) {
            Ok(result) => {
                if args.print {
                    write_classify_output(&mut out, file, model_id, &result, &args.format)?;
                }
                if needs_collect {
                    collected.push((file.clone(), result));
                }
            }
            Err(e) => {
                bar.println(format!("error: {}: {e}", file.display()));
                errors += 1;
            }
        }
        bar.inc(1);
    }
    bar.finish_and_clear();

    // Visualize if requested.
    if args.visualize {
        let output_dir = args.output_dir.as_ref().unwrap();
        let viz_fail = run_visualization(
            &collected,
            engine_dispatch::viz::classifications_to_annotations,
            output_dir,
            &files,
            handle.model_type(),
            args.show_labels,
        );
        if is_all_viz_failed(viz_fail, collected.len()) {
            return Err("All visualizations failed".into());
        }
    }

    if errors == total {
        return Err("All files failed processing.".into());
    }
    Ok(())
}

fn write_classify_output(
    out: &mut impl Write,
    file: &Path,
    model_id: &str,
    result: &ClassifyResult,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        OutputFormat::Json => {
            let output = ClassifyOutput {
                file: file.display().to_string(),
                model_id: model_id.to_string(),
                image_size: [result.image_width, result.image_height],
                classifications: result
                    .classifications
                    .iter()
                    .map(|c| ClassificationOutput {
                        label: c.label.clone(),
                        confidence: c.confidence,
                    })
                    .collect(),
            };
            serde_json::to_writer(&mut *out, &output)?;
            writeln!(out)?;
        }
        OutputFormat::Csv => {
            let file_str = engine_dispatch::export::csv_escape(&file.display().to_string());
            for (idx, c) in result.classifications.iter().enumerate() {
                let label_str = engine_dispatch::export::csv_escape(&c.label);
                writeln!(
                    out,
                    "{},{},{},{},{:.6}",
                    file_str,
                    engine_dispatch::export::csv_escape(model_id),
                    idx,
                    label_str,
                    c.confidence,
                )?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Command: detect-audio
// ---------------------------------------------------------------------------

/// Default sliding-window length (seconds) used by `cmd_detect_audio`
/// when a manifest does not expose `audio_window_stride()`. Anchored on
/// `MD_AudioBirds_V1` (the only Phase 1 audio model with sliding-window
/// semantics — `[audio.window] window_s = 1.0` in the production
/// manifest at 2026-05-05). A future audio model with no sliding-window
/// params (e.g., a single-shot detector) will silently inherit this
/// fallback for `--visualize` overlap-mean slot widths and merge-gap
/// computations; flagged by Phase 3.8 Step 2 audit-fix R1-F9.
const MD_AUDIOBIRDS_DEFAULT_WINDOW_S: f32 = 1.0;
/// Default sliding-window stride (seconds) used by `cmd_detect_audio`.
/// See [`MD_AUDIOBIRDS_DEFAULT_WINDOW_S`] for anchor + caveats. The
/// production `MD_AudioBirds_V1` manifest sets
/// `[audio.window] stride_s = 0.3` (2026-05-05).
const MD_AUDIOBIRDS_DEFAULT_STRIDE_S: f32 = 0.3;

fn audio_visualize_output_filter_threshold(
    cli_threshold: Option<f32>,
    manifest_threshold: Option<f32>,
) -> Option<f32> {
    manifest_threshold.map(|threshold| cli_threshold.unwrap_or(threshold))
}

fn cmd_detect_audio(
    device_str: &str,
    model_dir: &Option<PathBuf>,
    quiet: bool,
    args: DetectAudioArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    let files = resolve_audio_inputs(&args.input, args.recursive);
    if files.is_empty() {
        return Err("No audio files found.".into());
    }

    validate_viz_args(args.visualize, &args.output_dir)?;
    // Validate user-supplied stride / segment-duration overrides. Match the
    // sparrow-engine-server policy (`handlers/audio.rs`): finite + positive.
    if let Some(s) = args.stride {
        if !s.is_finite() || s <= 0.0 {
            return Err("--stride must be a finite positive number".into());
        }
    }
    if let Some(d) = args.segment_duration_s {
        if !d.is_finite() || d <= 0.0 {
            return Err("--segment-duration must be a finite positive number".into());
        }
    }
    let viz_common_prefix = if args.visualize {
        longest_common_prefix(&files)
    } else {
        PathBuf::new()
    };

    let engine = create_engine(device_str, model_dir)?;
    let model_id = args.model.as_deref().unwrap_or("md-audiobirds-v1");
    let handle = engine.get_or_load_model(model_id)?;
    let audio_config = handle.audio_preprocess_config();

    // Resolve window + stride from the manifest, then apply CLI overrides.
    // The manifest provides defaults; `--stride` / `--segment-duration`
    // override them at runtime. Falls back to MD_AudioBirds defaults if the
    // model doesn't expose sliding-window params (e.g., a future single-shot
    // audio model). The merge-gap is `stride + 1ms` so strictly-adjacent
    // windows merge while a true silence gap ≥ stride splits the range.
    let (manifest_window_s, manifest_stride_s) = handle.audio_window_stride().unwrap_or((
        MD_AUDIOBIRDS_DEFAULT_WINDOW_S,
        MD_AUDIOBIRDS_DEFAULT_STRIDE_S,
    ));
    let window_s = args.segment_duration_s.unwrap_or(manifest_window_s);
    let stride_s = args.stride.unwrap_or(manifest_stride_s);
    let merge_gap_s = stride_s + 1e-3;

    // When --visualize is set for thresholded sigmoid detectors, layers 02
    // (segments) and 03 (heatmap) need the full per-window confidence
    // distribution. Run those detectors at threshold=0, then post-filter
    // JSON / CSV / merged-range output back to the user's intended detector
    // threshold (CLI override > manifest default). Thresholdless softmax
    // classifiers such as Perch 2 have no production threshold to restore, so
    // visualization must not add a CLI-only 0.5 output filter.
    let output_filter_threshold = audio_visualize_output_filter_threshold(
        args.threshold,
        handle.audio_confidence_threshold(),
    );
    let inference_threshold = if args.visualize && output_filter_threshold.is_some() {
        Some(0.0)
    } else {
        args.threshold
    };

    let opts = AudioDetectOpts {
        confidence_threshold: inference_threshold,
        segment_duration_s: args.segment_duration_s,
        stride_s: args.stride,
    };

    let total = files.len();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut errors = 0usize;
    let mut audio_viz_attempts = 0usize;
    let mut audio_viz_fail = 0usize;

    // CSV header depends on --raw-segments: merged ranges emit a 4-column
    // schema (start, end, max_confidence, class); raw segments retain
    // the pre-Phase-3.5 3-column schema.
    if args.print && matches!(args.format, OutputFormat::Csv) {
        if args.raw_segments {
            writeln!(out, "file,model_id,idx,start_time_s,end_time_s,confidence")?;
        } else {
            writeln!(
                out,
                "file,model_id,idx,start_time_s,end_time_s,max_confidence,class"
            )?;
        }
    }

    let bar = make_progress_bar(total as u64, quiet);
    for file in &files {
        bar.set_message(file.display().to_string());
        let audio = AudioInput::FilePath(file.clone());

        match detect_audio::detect_audio(&handle, &audio, &opts) {
            Ok(result) => {
                // For thresholded detectors, --visualize lowers inference to
                // 0 and this view restores the intended machine-readable
                // output threshold. Thresholdless classifiers skip the filter
                // so visualization does not change printed output cardinality.
                let output_result;
                let output_view: &AudioDetectResult = if args.visualize {
                    if let Some(output_threshold) = output_filter_threshold {
                        output_result = AudioDetectResult {
                            segments: result
                                .segments
                                .iter()
                                .filter(|s| s.confidence >= output_threshold)
                                .cloned()
                                .collect(),
                            duration_s: result.duration_s,
                            sample_rate: result.sample_rate,
                            processing_time_ms: result.processing_time_ms,
                        };
                        &output_result
                    } else {
                        &result
                    }
                } else {
                    &result
                };
                if args.print {
                    write_audio_output(
                        &mut out,
                        file,
                        model_id,
                        output_view,
                        &args.format,
                        args.raw_segments,
                        merge_gap_s,
                    )?;
                }
                if args.visualize {
                    let output_dir = args.output_dir.as_deref().expect("validated above");
                    // Layer 04's cyan range overlay should always reflect
                    // high-confidence merged ranges, independent of the
                    // --threshold flag (which lets the user lower the
                    // filter for diagnostic raw-segment viewing in
                    // layer 02). Hardcoded to 0.9 (matches the manifest
                    // production default in `sparrow-engine/models/audiobirds.toml`).
                    //
                    // Stride pulled from the manifest above (not a constant)
                    // so the slot/merge resolution works for any
                    // sliding-window model, not just MD_AudioBirds_V1.
                    const VIZ_MERGE_THRESHOLD: f32 = 0.9;
                    let ranges_owned = if args.raw_segments {
                        None
                    } else {
                        let slots = engine_dispatch::viz::segments_to_overlap_mean_slots(
                            &result.segments,
                            result.duration_s,
                            stride_s,
                        );
                        let high_conf_slots: Vec<engine_dispatch::types::AudioSegment> = slots
                            .into_iter()
                            .filter(|s| s.confidence >= VIZ_MERGE_THRESHOLD)
                            .collect();
                        Some(detect_audio::merge_segments(&high_conf_slots, merge_gap_s))
                    };
                    let ranges_ref = ranges_owned.as_deref();
                    let viz_params = AudioVizParams {
                        ranges: ranges_ref,
                        audio_config: audio_config.as_ref(),
                        window_s,
                        stride_s,
                        smooth: args.smooth,
                        show_windows: args.show_windows,
                    };
                    audio_viz_attempts += 1;
                    match save_audio_visualization(
                        file,
                        &result,
                        &viz_params,
                        output_dir,
                        &viz_common_prefix,
                    ) {
                        Ok(paths) => {
                            for p in paths {
                                bar.println(format!("viz: {}", p.display()));
                            }
                        }
                        Err(e) => {
                            audio_viz_fail += 1;
                            bar.println(format!("viz error: {}: {e}", file.display()));
                        }
                    }
                }
            }
            Err(e) => {
                bar.println(format!("error: {}: {e}", file.display()));
                errors += 1;
            }
        }
        bar.inc(1);
    }
    bar.finish_and_clear();

    if errors == total {
        return Err("All files failed processing.".into());
    }
    if is_all_viz_failed(audio_viz_fail, audio_viz_attempts) {
        return Err("All visualizations failed".into());
    }
    Ok(())
}

fn write_audio_output(
    out: &mut impl Write,
    file: &Path,
    model_id: &str,
    result: &AudioDetectResult,
    format: &OutputFormat,
    raw_segments: bool,
    merge_gap_s: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    if raw_segments {
        write_audio_output_raw(out, file, model_id, result, format)
    } else {
        write_audio_output_merged(out, file, model_id, result, format, merge_gap_s)
    }
}

/// Raw per-window output (pre-Phase-3.5 default; `--raw-segments` opt-in).
fn write_audio_output_raw(
    out: &mut impl Write,
    file: &Path,
    model_id: &str,
    result: &AudioDetectResult,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        OutputFormat::Json => {
            let output = AudioDetectRawOutput {
                file: file.display().to_string(),
                model_id: model_id.to_string(),
                duration_s: result.duration_s,
                sample_rate: result.sample_rate,
                segments: result
                    .segments
                    .iter()
                    .map(|s| AudioSegmentOutput {
                        start_time_s: s.start_time_s,
                        end_time_s: s.end_time_s,
                        confidence: s.confidence,
                        classes: (s.classes.len() > 1).then(|| {
                            s.classes
                                .iter()
                                .map(|c| AudioClassOutput {
                                    class_idx: c.class_idx,
                                    label: c.label.clone(),
                                    probability: c.probability,
                                })
                                .collect()
                        }),
                    })
                    .collect(),
            };
            serde_json::to_writer(&mut *out, &output)?;
            writeln!(out)?;
        }
        OutputFormat::Csv => {
            let file_str = engine_dispatch::export::csv_escape(&file.display().to_string());
            for (idx, s) in result.segments.iter().enumerate() {
                writeln!(
                    out,
                    "{},{},{},{:.6},{:.6},{:.6}",
                    file_str,
                    engine_dispatch::export::csv_escape(model_id),
                    idx,
                    s.start_time_s,
                    s.end_time_s,
                    s.confidence,
                )?;
            }
        }
    }
    Ok(())
}

/// Merged-range output (Phase 3.5 default; item #6).
fn write_audio_output_merged(
    out: &mut impl Write,
    file: &Path,
    model_id: &str,
    result: &AudioDetectResult,
    format: &OutputFormat,
    merge_gap_s: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let ranges = detect_audio::merge_segments_with_class(&result.segments, merge_gap_s, |s| {
        s.classes.first().and_then(|c| c.label.clone())
    });
    match format {
        OutputFormat::Json => {
            let output = AudioDetectMergedOutput {
                file: file.display().to_string(),
                model_id: model_id.to_string(),
                duration_s: result.duration_s,
                sample_rate: result.sample_rate,
                ranges: ranges
                    .iter()
                    .map(|r| AudioRangeOutput {
                        start_time_s: r.start_time_s,
                        end_time_s: r.end_time_s,
                        max_confidence: r.max_confidence,
                        class: r.class.clone(),
                    })
                    .collect(),
            };
            serde_json::to_writer(&mut *out, &output)?;
            writeln!(out)?;
        }
        OutputFormat::Csv => {
            let file_str = engine_dispatch::export::csv_escape(&file.display().to_string());
            for (idx, r) in ranges.iter().enumerate() {
                let class_str = match &r.class {
                    Some(c) => engine_dispatch::export::csv_escape(c),
                    None => String::new(),
                };
                writeln!(
                    out,
                    "{},{},{},{:.6},{:.6},{:.6},{}",
                    file_str,
                    engine_dispatch::export::csv_escape(model_id),
                    idx,
                    r.start_time_s,
                    r.end_time_s,
                    r.max_confidence,
                    class_str,
                )?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Command: pipeline
// ---------------------------------------------------------------------------

fn validate_pipeline_ids(
    engine: &Engine,
    detector_id: &str,
    classifier_id: &str,
) -> engine_dispatch::Result<()> {
    validate_pipeline_ids_from_available(
        &engine.list_available_models(),
        detector_id,
        classifier_id,
    )
}

fn validate_pipeline_ids_from_available(
    available: &[ModelInfo],
    detector_id: &str,
    classifier_id: &str,
) -> engine_dispatch::Result<()> {
    let detector_type = available
        .iter()
        .find(|m| m.id == detector_id)
        .map(|m| m.model_type);
    let classifier_type = available
        .iter()
        .find(|m| m.id == classifier_id)
        .map(|m| m.model_type);

    match (detector_type, classifier_type) {
        (Some(detector), Some(classifier)) => {
            engine_dispatch::pipeline_compat::validate_pipeline_compat(
                Some(detector),
                Some(classifier),
            )
        }
        _ => Ok(()),
    }
}

fn cmd_pipeline(
    device_str: &str,
    model_dir: &Option<PathBuf>,
    quiet: bool,
    args: PipelineArgs,
) -> Result<(), Box<dyn std::error::Error>> {
    validate_viz_args(args.visualize, &args.output_dir)?;

    let files = resolve_inputs(&args.input, args.recursive);
    if files.is_empty() {
        return Err("No image files found.".into());
    }

    let engine = create_engine(device_str, model_dir)?;
    validate_pipeline_ids(&engine, &args.detector, &args.classifier)?;

    // Pre-load detector to obtain its ModelType for viz dispatch (Phase 3.5
    // S3 / MT-9). Lazy + idempotent: a repeat call (e.g. same id for both
    // detector and classifier) reuses the existing session rather than
    // force-reloading and invalidating prior handles.
    let detector_handle = engine.get_or_load_model(&args.detector)?;
    let detector_model_type = detector_handle.model_type();
    drop(detector_handle);
    let classifier_handle = engine.get_or_load_model(&args.classifier)?;
    drop(classifier_handle);

    let d_opts = DetectOpts {
        confidence_threshold: args.threshold,
        ..Default::default()
    };
    let c_opts = ClassifyOpts { top_k: args.top_k };

    let total = files.len();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut errors = 0usize;
    let needs_collect = args.visualize || args.export_format.is_some();
    let mut collected: Vec<(PathBuf, PipelineResult)> = Vec::new();

    if args.print && matches!(args.format, OutputFormat::Csv) {
        writeln!(out, "file,pipeline_id,idx,label,confidence,x_min,y_min,x_max,y_max,cls_label,cls_confidence")?;
    }

    let bar = make_progress_bar(total as u64, quiet);
    for file in &files {
        bar.set_message(file.display().to_string());
        let image = ImageInput::FilePath(file.clone());

        match engine_dispatch::pipeline::run_pipeline_adhoc(
            &engine,
            &image,
            &args.detector,
            &args.classifier,
            &d_opts,
            &c_opts,
        ) {
            Ok(result) => {
                if args.print {
                    write_pipeline_output(&mut out, file, &result, &args.format)?;
                }
                if needs_collect {
                    collected.push((file.clone(), result));
                }
            }
            Err(e) => {
                bar.println(format!("error: {}: {e}", file.display()));
                errors += 1;
            }
        }
        bar.inc(1);
    }
    bar.finish_and_clear();

    if errors == total {
        return Err("All files failed processing.".into());
    }

    // Export if requested.
    if let Some(ref export_fmt) = args.export_format {
        let pipeline_entries: Vec<(&Path, &PipelineResult)> =
            collected.iter().map(|(p, r)| (p.as_path(), r)).collect();
        let detect_entries =
            engine_dispatch::export::pipeline_results_to_detect_entries(&pipeline_entries);
        let export_refs: Vec<(&Path, &DetectResult)> = detect_entries
            .iter()
            .map(|(p, r)| (p.as_path(), r))
            .collect();
        let mut export_writer: Box<dyn Write> = if let Some(ref path) = args.export_output {
            ensure_parent_dir(path)?;
            Box::new(std::fs::File::create(path)?)
        } else {
            Box::new(io::stdout().lock())
        };
        match export_fmt {
            ExportFormat::Megadet => engine_dispatch::export::to_megadet(
                &export_refs,
                &args.detector,
                &mut export_writer,
            )?,
            ExportFormat::Coco => {
                engine_dispatch::export::to_coco(&export_refs, &mut export_writer)?
            }
            ExportFormat::Csv => engine_dispatch::export::to_csv(&export_refs, &mut export_writer)?,
        }
    }

    // Visualize if requested.
    if args.visualize {
        let output_dir = args.output_dir.as_ref().unwrap();
        let viz_fail = run_visualization(
            &collected,
            engine_dispatch::viz::pipeline_to_annotations,
            output_dir,
            &files,
            detector_model_type,
            args.show_labels,
        );
        if is_all_viz_failed(viz_fail, collected.len()) {
            return Err("All visualizations failed".into());
        }
    }

    Ok(())
}

fn write_pipeline_output(
    out: &mut impl Write,
    file: &Path,
    result: &PipelineResult,
    format: &OutputFormat,
) -> Result<(), Box<dyn std::error::Error>> {
    match format {
        OutputFormat::Json => {
            let output = PipelineOutput {
                file: file.display().to_string(),
                pipeline_id: result.pipeline_id.clone(),
                image_size: [result.image_width, result.image_height],
                detections: result
                    .detections
                    .iter()
                    .map(|pd| PipelineDetectionOutput {
                        label: pd.detection.label.clone(),
                        confidence: pd.detection.confidence,
                        bbox: BBoxOutput {
                            x_min: pd.detection.bbox.x_min,
                            y_min: pd.detection.bbox.y_min,
                            x_max: pd.detection.bbox.x_max,
                            y_max: pd.detection.bbox.y_max,
                        },
                        classification: pd.classification.as_ref().map(|c| ClassificationOutput {
                            label: c.label.clone(),
                            confidence: c.confidence,
                        }),
                    })
                    .collect(),
            };
            serde_json::to_writer(&mut *out, &output)?;
            writeln!(out)?;
        }
        OutputFormat::Csv => {
            let file_str = engine_dispatch::export::csv_escape(&file.display().to_string());
            let pipeline_str = engine_dispatch::export::csv_escape(&result.pipeline_id);
            for (idx, pd) in result.detections.iter().enumerate() {
                let det_label = engine_dispatch::export::csv_escape(&pd.detection.label);
                let (cls_label, cls_conf) = match &pd.classification {
                    Some(c) => (
                        engine_dispatch::export::csv_escape(&c.label),
                        format!("{:.6}", c.confidence),
                    ),
                    None => (String::new(), String::new()),
                };
                writeln!(
                    out,
                    "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
                    file_str,
                    pipeline_str,
                    idx,
                    det_label,
                    pd.detection.confidence,
                    pd.detection.bbox.x_min,
                    pd.detection.bbox.y_min,
                    pd.detection.bbox.x_max,
                    pd.detection.bbox.y_max,
                    cls_label,
                    cls_conf,
                )?;
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Command: models
// ---------------------------------------------------------------------------

fn cmd_models(
    device_str: &str,
    model_dir: &Option<PathBuf>,
    action: ModelsAction,
) -> Result<(), Box<dyn std::error::Error>> {
    // Verify does not need ORT — skip engine creation (design 3.2).
    if let ModelsAction::Verify {
        ref model_id,
        write,
    } = action
    {
        let dir = resolve_model_dir(model_dir);
        let ids: Vec<String> = if let Some(ref id) = model_id {
            vec![id.clone()]
        } else {
            engine_dispatch::catalog::list_available_models(&dir)
                .iter()
                .map(|m| m.id.clone())
                .collect()
        };
        if write {
            let mut had_error = false;
            for id in &ids {
                match engine_dispatch::catalog::write_checksum(&dir, id) {
                    Ok((hash, size)) => {
                        eprintln!("{id}: wrote sha256={hash}, size={size}");
                    }
                    Err(e) => {
                        eprintln!("{id}: error: {e}");
                        had_error = true;
                    }
                }
            }
            if had_error {
                return Err("one or more models failed checksum write".into());
            }
        } else {
            let mut had_failure = false;
            for id in &ids {
                match engine_dispatch::catalog::verify_model(&dir, id) {
                    Ok(engine_dispatch::catalog::VerifyResult::Ok) => {
                        println!("{id}: OK");
                    }
                    Ok(engine_dispatch::catalog::VerifyResult::NoChecksum) => {
                        println!("{id}: no checksum (use --write to generate)");
                    }
                    Ok(engine_dispatch::catalog::VerifyResult::SizeMismatch {
                        expected,
                        actual,
                    }) => {
                        println!("{id}: FAIL size mismatch (expected={expected}, actual={actual})");
                        had_failure = true;
                    }
                    Ok(engine_dispatch::catalog::VerifyResult::ChecksumMismatch {
                        expected,
                        actual,
                    }) => {
                        println!("{id}: FAIL checksum mismatch");
                        println!("  expected: {expected}");
                        println!("  actual:   {actual}");
                        had_failure = true;
                    }
                    Err(e) => {
                        println!("{id}: error: {e}");
                        had_failure = true;
                    }
                }
            }
            if had_failure {
                return Err("one or more models failed verification".into());
            }
        }
        return Ok(());
    }

    let engine = create_engine(device_str, model_dir)?;

    match action {
        ModelsAction::List => {
            let models = engine.list_available_models();
            if models.is_empty() {
                eprintln!(
                    "No models found in {}.",
                    engine.config().model_dir.display()
                );
                return Ok(());
            }
            let stdout = io::stdout();
            let mut out = stdout.lock();
            for m in &models {
                let output = ModelInfoOutput {
                    id: m.id.clone(),
                    path: m.path.display().to_string(),
                    model_type: model_type_display(&m.model_type).to_string(),
                    default: m.default,
                    version: m.version.clone(),
                    description: m.description.clone(),
                    onnx_sha256: m.onnx_sha256.clone(),
                    onnx_size_bytes: m.onnx_size_bytes,
                };
                serde_json::to_writer(&mut out, &output)?;
                writeln!(out)?;
            }
        }
        ModelsAction::Info { model_id } => {
            let models = engine.list_available_models();
            let m = models
                .iter()
                .find(|m| m.id == model_id)
                .ok_or_else(|| format!("Model not found: {model_id}"))?;
            let info = ModelInfoOutput {
                id: m.id.clone(),
                path: m.path.display().to_string(),
                model_type: model_type_display(&m.model_type).to_string(),
                default: m.default,
                version: m.version.clone(),
                description: m.description.clone(),
                onnx_sha256: m.onnx_sha256.clone(),
                onnx_size_bytes: m.onnx_size_bytes,
            };
            let stdout = io::stdout();
            let mut out = stdout.lock();
            serde_json::to_writer(&mut out, &info)?;
            writeln!(out)?;
        }
        ModelsAction::Verify { .. } => unreachable!(),
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Command: device
// ---------------------------------------------------------------------------

fn cmd_device(
    device_str: &str,
    model_dir: &Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_engine(device_str, model_dir)?;
    let output = DeviceOutput {
        device: engine.active_device().to_string(),
    };
    let stdout = io::stdout();
    let mut out = stdout.lock();
    serde_json::to_writer(&mut out, &output)?;
    writeln!(out)?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Command: init
// ---------------------------------------------------------------------------

fn cmd_init(
    device_str: &str,
    model_dir: &Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let engine = create_engine(device_str, model_dir)?;
    let dir = engine.config().model_dir.display();
    eprintln!(
        "Engine initialized. device={}, model_dir={dir}",
        engine.active_device()
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Command: hash
// ---------------------------------------------------------------------------

fn cmd_hash(args: HashArgs) -> Result<(), Box<dyn std::error::Error>> {
    let hash = engine_dispatch::hash::hash_file(&args.file)?;
    println!("{hash}  {}", args.file.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Command: day-night
// ---------------------------------------------------------------------------

fn cmd_day_night(args: DayNightArgs) -> Result<(), Box<dyn std::error::Error>> {
    let data = std::fs::read(&args.image)?;
    let result = engine_dispatch::daynight::day_night(&data)?;
    let class = match result.classification {
        engine_dispatch::daynight::DayNight::Day => "day",
        engine_dispatch::daynight::DayNight::Night => "night",
    };
    println!(
        "{}  classification={}  brightness={:.1}",
        args.image.display(),
        class,
        result.mean_brightness,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn model_info(id: &str, model_type: ModelType) -> ModelInfo {
        ModelInfo {
            id: id.to_string(),
            path: PathBuf::from(format!("/models/{id}/manifest.toml")),
            model_type,
            default: false,
            version: None,
            description: None,
            onnx_sha256: None,
            onnx_size_bytes: None,
        }
    }

    #[test]
    fn audio_visualize_filter_threshold_only_for_thresholded_detectors() {
        assert_eq!(
            audio_visualize_output_filter_threshold(None, Some(0.9)),
            Some(0.9)
        );
        assert_eq!(
            audio_visualize_output_filter_threshold(Some(0.4), Some(0.9)),
            Some(0.4)
        );
        assert_eq!(audio_visualize_output_filter_threshold(None, None), None);
        assert_eq!(
            audio_visualize_output_filter_threshold(Some(0.4), None),
            None
        );
    }

    #[test]
    fn validate_pipeline_ids_rejects_known_incompatible_pair() {
        let available = vec![
            model_info("owl-t", ModelType::OverheadDetector),
            model_info("speciesnet-crop", ModelType::Classifier),
        ];
        let err = validate_pipeline_ids_from_available(&available, "owl-t", "speciesnet-crop")
            .unwrap_err();
        match err {
            engine_dispatch::SparrowEngineError::IncompatiblePipeline { reason, .. } => {
                assert!(
                    reason.contains("point detection"),
                    "unexpected reason: {reason}"
                );
            }
            other => panic!("expected IncompatiblePipeline, got {other:?}"),
        }
    }

    #[test]
    fn validate_pipeline_ids_defers_unknown_ids_to_load_path() {
        let available = vec![model_info("speciesnet-crop", ModelType::Classifier)];
        validate_pipeline_ids_from_available(&available, "missing", "speciesnet-crop").unwrap();
    }

    // -----------------------------------------------------------------------
    // parse_device
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_device_auto() {
        let d = parse_device("auto").unwrap();
        assert!(matches!(d, Device::Auto));
    }

    #[test]
    fn test_parse_device_cpu() {
        let d = parse_device("cpu").unwrap();
        assert!(matches!(d, Device::Cpu));
    }

    #[test]
    fn test_parse_device_gpu() {
        let d = parse_device("gpu").unwrap();
        assert!(matches!(d, Device::Cuda(0)));
    }

    #[test]
    fn test_parse_device_cuda() {
        let d = parse_device("cuda").unwrap();
        assert!(matches!(d, Device::Cuda(0)));
    }

    #[test]
    fn test_parse_device_cuda_0() {
        let d = parse_device("cuda:0").unwrap();
        assert!(matches!(d, Device::Cuda(0)));
    }

    #[test]
    fn test_parse_device_cuda_2() {
        let d = parse_device("cuda:2").unwrap();
        assert!(matches!(d, Device::Cuda(2)));
    }

    #[test]
    fn test_parse_device_case_insensitive() {
        assert!(matches!(parse_device("AUTO").unwrap(), Device::Auto));
        assert!(matches!(parse_device("Cpu").unwrap(), Device::Cpu));
        assert!(matches!(parse_device("GPU").unwrap(), Device::Cuda(0)));
        assert!(matches!(parse_device("CUDA:1").unwrap(), Device::Cuda(1)));
    }

    #[test]
    fn test_parse_device_invalid() {
        assert!(parse_device("invalid").is_err());
    }

    #[test]
    fn test_parse_device_cuda_bad_index() {
        assert!(parse_device("cuda:abc").is_err());
    }

    // -----------------------------------------------------------------------
    // Device::Display (canonical impl in sparrow-engine-types, tested here for CLI contract)
    // -----------------------------------------------------------------------

    #[test]
    fn test_device_display_auto() {
        assert_eq!(Device::Auto.to_string(), "auto");
    }

    #[test]
    fn test_device_display_cpu() {
        assert_eq!(Device::Cpu.to_string(), "cpu");
    }

    #[test]
    fn test_device_display_cuda_0() {
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
    }

    #[test]
    fn test_device_display_cuda_2() {
        assert_eq!(Device::Cuda(2).to_string(), "cuda:2");
    }

    // -----------------------------------------------------------------------
    // model_type_display
    // -----------------------------------------------------------------------

    #[test]
    fn test_model_type_display_detector() {
        assert_eq!(model_type_display(&ModelType::Detector), "detector");
    }

    #[test]
    fn test_model_type_display_overhead_detector() {
        // Phase 3.5 S3 (MT-9 fix): added the `OverheadDetector` variant;
        // pin its CLI string to `"overhead_detector"` to keep parity with
        // `sparrow_engine_cpu::types::ModelType::as_str`, `sparrow-engine-server::model_type_str`,
        // and `sparrow-engine-python::convert_model_type`.
        assert_eq!(
            model_type_display(&ModelType::OverheadDetector),
            "overhead_detector"
        );
    }

    #[test]
    fn test_model_type_display_classifier() {
        assert_eq!(model_type_display(&ModelType::Classifier), "classifier");
    }

    #[test]
    fn test_model_type_display_audio_detector() {
        assert_eq!(
            model_type_display(&ModelType::AudioDetector),
            "audio_detector"
        );
    }

    #[test]
    fn test_model_type_display_audio_classifier() {
        assert_eq!(
            model_type_display(&ModelType::AudioClassifier),
            "audio_classifier"
        );
    }

    // -----------------------------------------------------------------------
    // resolve_model_dir
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_model_dir_explicit() {
        let explicit = PathBuf::from("/tmp/my_models");
        let result = resolve_model_dir(&Some(explicit.clone()));
        assert_eq!(result, explicit);
    }

    #[test]
    fn test_resolve_model_dir_env() {
        // Save and remove any existing --model-dir-like env
        let prev = std::env::var("SPARROW_ENGINE_MODEL_DIR").ok();
        std::env::set_var("SPARROW_ENGINE_MODEL_DIR", "/tmp/env_models");
        let result = resolve_model_dir(&None);
        assert_eq!(result, PathBuf::from("/tmp/env_models"));
        // Restore
        match prev {
            Some(v) => std::env::set_var("SPARROW_ENGINE_MODEL_DIR", v),
            None => std::env::remove_var("SPARROW_ENGINE_MODEL_DIR"),
        }
    }

    #[test]
    fn test_resolve_model_dir_default() {
        let prev = std::env::var("SPARROW_ENGINE_MODEL_DIR").ok();
        std::env::remove_var("SPARROW_ENGINE_MODEL_DIR");
        let result = resolve_model_dir(&None);
        // Should end with .sparrow-engine/models
        assert!(
            result.ends_with(".sparrow-engine/models"),
            "got: {}",
            result.display()
        );
        // Restore
        if let Some(v) = prev {
            std::env::set_var("SPARROW_ENGINE_MODEL_DIR", v);
        }
    }

    // -----------------------------------------------------------------------
    // collect_files_from_dir
    // -----------------------------------------------------------------------

    #[test]
    fn test_collect_files_extension_filtering() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.jpg"), b"").unwrap();
        fs::write(dir.path().join("b.png"), b"").unwrap();
        fs::write(dir.path().join("c.txt"), b"").unwrap();

        let mut out = Vec::new();
        let mut visited = std::collections::HashSet::new();
        collect_files_from_dir(
            &dir.path().to_path_buf(),
            &["jpg", "png"],
            false,
            &mut out,
            &mut visited,
        );
        out.sort();
        assert_eq!(out.len(), 2);
        assert!(out.iter().any(|p| p.ends_with("a.jpg")));
        assert!(out.iter().any(|p| p.ends_with("b.png")));
    }

    #[test]
    fn test_collect_files_case_insensitive_ext() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.JPG"), b"").unwrap();
        fs::write(dir.path().join("b.Png"), b"").unwrap();

        let mut out = Vec::new();
        let mut visited = std::collections::HashSet::new();
        collect_files_from_dir(
            &dir.path().to_path_buf(),
            &["jpg", "png"],
            false,
            &mut out,
            &mut visited,
        );
        assert_eq!(out.len(), 2);
    }

    // -----------------------------------------------------------------------
    // resolve_inputs
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_inputs_single_file() {
        let dir = TempDir::new().unwrap();
        let f = dir.path().join("test.jpg");
        fs::write(&f, b"").unwrap();

        let result = resolve_inputs(std::slice::from_ref(&f), false);
        assert_eq!(result, vec![f]);
    }

    #[test]
    fn test_resolve_inputs_directory_expansion() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.jpg"), b"").unwrap();
        fs::write(dir.path().join("b.png"), b"").unwrap();
        fs::write(dir.path().join("c.txt"), b"").unwrap();

        let result = resolve_inputs(&[dir.path().to_path_buf()], false);
        assert_eq!(result.len(), 2);
        assert!(result.iter().any(|p| p.ends_with("a.jpg")));
        assert!(result.iter().any(|p| p.ends_with("b.png")));
    }

    #[test]
    fn test_resolve_inputs_nonexistent_skipped() {
        let result = resolve_inputs(&[PathBuf::from("/nonexistent/fake.jpg")], false);
        assert!(result.is_empty());
    }

    #[test]
    fn test_resolve_inputs_recursive() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        fs::write(dir.path().join("top.jpg"), b"").unwrap();
        fs::write(sub.join("nested.png"), b"").unwrap();

        let result = resolve_inputs(&[dir.path().to_path_buf()], true);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_resolve_inputs_non_recursive_skips_subdirs() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        fs::write(dir.path().join("top.jpg"), b"").unwrap();
        fs::write(sub.join("nested.png"), b"").unwrap();

        let result = resolve_inputs(&[dir.path().to_path_buf()], false);
        assert_eq!(result.len(), 1);
        assert!(result[0].ends_with("top.jpg"));
    }

    #[test]
    fn test_resolve_inputs_sorted() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("c.jpg"), b"").unwrap();
        fs::write(dir.path().join("a.jpg"), b"").unwrap();
        fs::write(dir.path().join("b.jpg"), b"").unwrap();

        let result = resolve_inputs(&[dir.path().to_path_buf()], false);
        assert_eq!(result.len(), 3);
        assert!(result[0].ends_with("a.jpg"));
        assert!(result[1].ends_with("b.jpg"));
        assert!(result[2].ends_with("c.jpg"));
    }

    // -----------------------------------------------------------------------
    // resolve_audio_inputs
    // -----------------------------------------------------------------------

    #[test]
    fn test_resolve_audio_inputs_wav_only() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("a.wav"), b"").unwrap();
        fs::write(dir.path().join("b.mp3"), b"").unwrap();
        fs::write(dir.path().join("c.flac"), b"").unwrap();

        let result = resolve_audio_inputs(&[dir.path().to_path_buf()], false);
        assert_eq!(result.len(), 1);
        assert!(result[0].ends_with("a.wav"));
    }

    #[test]
    fn test_resolve_audio_inputs_directory() {
        let dir = TempDir::new().unwrap();
        fs::write(dir.path().join("x.wav"), b"").unwrap();
        fs::write(dir.path().join("y.wav"), b"").unwrap();

        let result = resolve_audio_inputs(&[dir.path().to_path_buf()], false);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_resolve_audio_inputs_single_file() {
        let dir = TempDir::new().unwrap();
        let f = dir.path().join("test.wav");
        fs::write(&f, b"").unwrap();

        let result = resolve_audio_inputs(std::slice::from_ref(&f), false);
        assert_eq!(result, vec![f]);
    }

    #[test]
    fn test_resolve_audio_inputs_recursive() {
        let dir = TempDir::new().unwrap();
        let sub = dir.path().join("subdir");
        fs::create_dir(&sub).unwrap();
        fs::write(dir.path().join("top.wav"), b"").unwrap();
        fs::write(sub.join("nested.wav"), b"").unwrap();

        let non_recursive = resolve_audio_inputs(&[dir.path().to_path_buf()], false);
        assert_eq!(non_recursive.len(), 1);

        let recursive = resolve_audio_inputs(&[dir.path().to_path_buf()], true);
        assert_eq!(recursive.len(), 2);
    }

    // -----------------------------------------------------------------------
    // collect_files_from_dir — symlink-follow regression guard (R5 T1)
    //
    // Pre-R5: collect_files_from_dir used DirEntry::metadata() which is
    // symlink_metadata-equivalent on Unix and silently dropped both
    // dirsymlinks and filesymlinks. Original code (pre-R4) used
    // path.is_dir()/path.is_file() which follow symlinks. The fix uses
    // std::fs::metadata(&path) which restores the symlink-follow semantic.
    //
    // Concrete deployed-regression scenario: setup.sh:75-77 creates
    // /tmp/sparrow_engine_test_10/IMG_*.jpg as symlinks to test fixtures. Under the
    // pre-R5 buggy code, `spe detect /tmp/sparrow_engine_test_10` returned 0
    // images and the manual-test §3 (10-image bench) was unreachable.
    //
    // Tests are #[cfg(unix)] because std::os::unix::fs::symlink is
    // Unix-only. Deterministic on tmpfs/ext4/xfs (Bongo's CI + dev hosts).
    // -----------------------------------------------------------------------

    #[cfg(unix)]
    #[test]
    fn test_collect_files_follows_symlink_to_dir() {
        use std::os::unix::fs::symlink;
        let target = TempDir::new().unwrap();
        fs::write(target.path().join("a.jpg"), b"").unwrap();
        let parent = TempDir::new().unwrap();
        symlink(target.path(), parent.path().join("models")).unwrap();

        let mut out = Vec::new();
        let mut visited = std::collections::HashSet::new();
        collect_files_from_dir(
            &parent.path().to_path_buf(),
            &["jpg"],
            true, // recursive — must follow the symlink-to-dir
            &mut out,
            &mut visited,
        );
        assert_eq!(
            out.len(),
            1,
            "symlinked dir should be followed; got {} files: {:?}",
            out.len(),
            out
        );
        assert_eq!(out[0].file_name().unwrap(), "a.jpg");
    }

    #[cfg(unix)]
    #[test]
    fn test_collect_files_follows_symlink_to_file() {
        use std::os::unix::fs::symlink;
        let real = TempDir::new().unwrap();
        fs::write(real.path().join("real.jpg"), b"").unwrap();
        let dir = TempDir::new().unwrap();
        symlink(real.path().join("real.jpg"), dir.path().join("link.jpg")).unwrap();

        let mut out = Vec::new();
        let mut visited = std::collections::HashSet::new();
        collect_files_from_dir(
            &dir.path().to_path_buf(),
            &["jpg"],
            false,
            &mut out,
            &mut visited,
        );
        assert_eq!(
            out.len(),
            1,
            "symlinked file should be followed; got {} files: {:?}",
            out.len(),
            out
        );
        assert_eq!(out[0].file_name().unwrap(), "link.jpg");
    }

    // -----------------------------------------------------------------------
    // validate_viz_args
    // -----------------------------------------------------------------------

    #[test]
    fn test_visualize_requires_output_dir() {
        // --visualize without --output-dir should error.
        assert!(validate_viz_args(true, &None).is_err());
        // --visualize with --output-dir should succeed.
        assert!(validate_viz_args(true, &Some(PathBuf::from("/tmp/viz"))).is_ok());
        // No --visualize should always succeed.
        assert!(validate_viz_args(false, &None).is_ok());
        assert!(validate_viz_args(false, &Some(PathBuf::from("/tmp/viz"))).is_ok());
    }

    // -----------------------------------------------------------------------
    // longest_common_prefix
    // -----------------------------------------------------------------------

    #[test]
    fn test_common_prefix_computation() {
        // Same directory.
        let paths = vec![
            PathBuf::from("/data/cam_01/a.jpg"),
            PathBuf::from("/data/cam_01/b.jpg"),
        ];
        assert_eq!(longest_common_prefix(&paths), PathBuf::from("/data/cam_01"));

        // Different subdirs.
        let paths = vec![
            PathBuf::from("/data/site_A/cam_01/a.jpg"),
            PathBuf::from("/data/site_A/cam_02/b.jpg"),
        ];
        assert_eq!(longest_common_prefix(&paths), PathBuf::from("/data/site_A"));

        // Single file.
        let paths = vec![PathBuf::from("/data/cam_01/a.jpg")];
        assert_eq!(longest_common_prefix(&paths), PathBuf::from("/data/cam_01"));

        // No common prefix (different roots).
        let paths = vec![PathBuf::from("/mnt/a.jpg"), PathBuf::from("/home/b.jpg")];
        assert_eq!(longest_common_prefix(&paths), PathBuf::from("/"));

        // Empty.
        let paths: Vec<PathBuf> = vec![];
        assert_eq!(longest_common_prefix(&paths), PathBuf::new());
    }

    // -----------------------------------------------------------------------
    // viz_output_path
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_path_mirroring() {
        // Basic case: strip prefix, keep original extension.
        let out = viz_output_path(
            Path::new("/data/site_A/cam_01/IMG_0001.jpg"),
            Path::new("/out"),
            Path::new("/data/site_A"),
        );
        assert_eq!(out, PathBuf::from("/out/cam_01/IMG_0001_viz.jpg"));

        // Flat: common prefix == parent dir.
        let out = viz_output_path(
            Path::new("/data/cam_01/IMG_0001.jpg"),
            Path::new("/out"),
            Path::new("/data/cam_01"),
        );
        assert_eq!(out, PathBuf::from("/out/IMG_0001_viz.jpg"));

        // Prefix mismatch: falls back to filename only.
        let out = viz_output_path(
            Path::new("/mnt/traps/IMG_0001.jpg"),
            Path::new("/out"),
            Path::new("/other"),
        );
        assert_eq!(out, PathBuf::from("/out/IMG_0001_viz.jpg"));

        // PNG input preserves PNG extension.
        let out = viz_output_path(
            Path::new("/data/cam/photo.png"),
            Path::new("/out"),
            Path::new("/data/cam"),
        );
        assert_eq!(out, PathBuf::from("/out/photo_viz.png"));
    }

    // -----------------------------------------------------------------------
    // Regression (H4): ensure_parent_dir creates missing nested dirs
    // -----------------------------------------------------------------------

    #[test]
    fn test_ensure_parent_dir_creates_nested() {
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("nested").join("deeper").join("out.json");
        assert!(!target.parent().unwrap().exists());
        ensure_parent_dir(&target).unwrap();
        assert!(target.parent().unwrap().is_dir());
        // Verify File::create works after ensure_parent_dir.
        let _f = std::fs::File::create(&target).unwrap();
        assert!(target.exists());
    }

    #[test]
    fn test_ensure_parent_dir_bare_filename_ok() {
        // Bare filename has Some(Path("")) as parent; must not error.
        ensure_parent_dir(Path::new("out.json")).unwrap();
    }

    #[test]
    fn test_ensure_parent_dir_existing_parent_ok() {
        let tmp = TempDir::new().unwrap();
        let target = tmp.path().join("out.json");
        ensure_parent_dir(&target).unwrap();
        let _f = std::fs::File::create(&target).unwrap();
        assert!(target.exists());
    }

    // -----------------------------------------------------------------------
    // Regression (H3): "All visualizations failed" predicate semantics
    // -----------------------------------------------------------------------
    //
    // The fatal check on viz failure was `viz_fail > 0` before H3, which
    // aborted the entire run on any single failed image. After H3, the
    // check is `viz_fail == collected.len() && !collected.is_empty()`,
    // so partial failure prints a warning + summary but still exits 0.
    // cmd_detect/cmd_classify/cmd_pipeline all call `is_all_viz_failed`
    // (defined in the main module) to decide whether viz failure is fatal.

    #[test]
    fn test_h3_all_fail_is_fatal() {
        assert!(is_all_viz_failed(3, 3));
    }

    #[test]
    fn test_h3_partial_fail_not_fatal() {
        // 1 of 3 failed, 2 succeeded — not fatal (was the H3 bug).
        assert!(!is_all_viz_failed(1, 3));
        // 2 of 3 failed — still not fatal.
        assert!(!is_all_viz_failed(2, 3));
    }

    #[test]
    fn test_h3_zero_fail_not_fatal() {
        assert!(!is_all_viz_failed(0, 3));
    }

    #[test]
    fn test_h3_empty_collected_not_fatal() {
        // Edge case: empty input set must not trip the predicate.
        assert!(!is_all_viz_failed(0, 0));
    }

    // Regression (ITEM-REV-003): cmd_detect_audio tracks one attempt per
    // successful inference that reaches save_audio_visualization, then feeds
    // those counters into the shared is_all_viz_failed predicate.
    #[test]
    fn test_audio_viz_counter_semantics_match_common_predicate() {
        assert!(is_all_viz_failed(2, 2));
        assert!(!is_all_viz_failed(1, 2));
        assert!(!is_all_viz_failed(0, 0));
    }

    // -----------------------------------------------------------------------
    // Regression: CLI --format csv must escape fields (S2 fix)
    // -----------------------------------------------------------------------

    #[test]
    fn test_csv_output_escapes_commas_in_path() {
        let result = DetectResult {
            detections: vec![engine_dispatch::Detection {
                bbox: engine_dispatch::BBox {
                    x_min: 0.1,
                    y_min: 0.2,
                    x_max: 0.3,
                    y_max: 0.4,
                },
                label: "animal".to_string(),
                label_id: 0,
                confidence: 0.95,
            }],
            image_width: 1920,
            image_height: 1080,
            processing_time_ms: 50.0,
        };
        let mut buf = Vec::new();
        let path = Path::new("Smith, John/camera/img.jpg");
        write_detect_output(&mut buf, path, "mdv6", &result, &OutputFormat::Csv).unwrap();
        let output = String::from_utf8(buf).unwrap();
        // Path with comma must be quoted per RFC 4180.
        assert!(
            output.starts_with("\"Smith, John/camera/img.jpg\""),
            "CSV path with comma must be quoted, got: {output}"
        );
    }

    #[test]
    fn test_csv_output_escapes_quotes_in_label() {
        let result = DetectResult {
            detections: vec![engine_dispatch::Detection {
                bbox: engine_dispatch::BBox {
                    x_min: 0.1,
                    y_min: 0.2,
                    x_max: 0.3,
                    y_max: 0.4,
                },
                label: "\"bird\"".to_string(),
                label_id: 0,
                confidence: 0.8,
            }],
            image_width: 640,
            image_height: 480,
            processing_time_ms: 30.0,
        };
        let mut buf = Vec::new();
        let path = Path::new("test.jpg");
        write_detect_output(&mut buf, path, "mdv6", &result, &OutputFormat::Csv).unwrap();
        let output = String::from_utf8(buf).unwrap();
        // Label with quotes must be escaped: "bird" -> """bird"""
        assert!(
            output.contains("\"\"\"bird\"\"\""),
            "CSV label with quotes must be double-escaped, got: {output}"
        );
    }

    #[test]
    fn test_classify_csv_escapes_fields() {
        let result = ClassifyResult {
            classifications: vec![engine_dispatch::Classification {
                label: "has,comma".to_string(),
                label_id: 1,
                confidence: 0.9,
            }],
            image_width: 640,
            image_height: 480,
            processing_time_ms: 20.0,
        };
        let mut buf = Vec::new();
        let path = Path::new("test.jpg");
        write_classify_output(&mut buf, path, "model", &result, &OutputFormat::Csv).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(
            output.contains("\"has,comma\""),
            "CSV label with comma must be quoted, got: {output}"
        );
    }

    // -----------------------------------------------------------------------
    // Phase 3.5 S5 (#6) — audio output: raw vs merged
    // -----------------------------------------------------------------------

    fn fake_audio_result(segments: Vec<engine_dispatch::AudioSegment>) -> AudioDetectResult {
        AudioDetectResult {
            segments,
            duration_s: 60.0,
            sample_rate: 48000,
            processing_time_ms: 10.0,
        }
    }

    fn seg(start: f32, end: f32, conf: f32) -> engine_dispatch::AudioSegment {
        engine_dispatch::AudioSegment {
            start_time_s: start,
            end_time_s: end,
            confidence: conf,
            classes: Vec::new(),
        }
    }

    fn audio_class(class_idx: u32, label: &str, probability: f32) -> engine_dispatch::AudioClass {
        engine_dispatch::AudioClass {
            class_idx,
            label: Some(label.to_string()),
            probability,
        }
    }

    fn seg_with_classes(
        start: f32,
        end: f32,
        classes: Vec<engine_dispatch::AudioClass>,
    ) -> engine_dispatch::AudioSegment {
        let confidence = classes.first().map(|c| c.probability).unwrap_or(0.0);
        engine_dispatch::AudioSegment {
            start_time_s: start,
            end_time_s: end,
            confidence,
            classes,
        }
    }

    #[test]
    fn audio_json_default_emits_ranges_not_segments() {
        // Two adjacent windows at 0.3 s stride should merge into one range.
        let result = fake_audio_result(vec![seg(0.0, 1.0, 0.9), seg(0.3, 1.3, 0.95)]);
        let mut buf = Vec::new();
        write_audio_output(
            &mut buf,
            Path::new("bird.wav"),
            "md-audiobirds-v1",
            &result,
            &OutputFormat::Json,
            /* raw_segments = */ false,
            0.31,
        )
        .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(
            output.contains("\"ranges\""),
            "default JSON must emit `ranges`, got: {output}"
        );
        assert!(
            !output.contains("\"segments\""),
            "default JSON must NOT emit `segments`, got: {output}"
        );
        assert!(
            output.contains("max_confidence"),
            "range entry must have max_confidence"
        );
        // Merged: the two inputs collapse into one range; max_confidence = 0.95.
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        let ranges = parsed["ranges"].as_array().unwrap();
        assert_eq!(ranges.len(), 1, "two adjacent windows must merge");
        assert!((ranges[0]["max_confidence"].as_f64().unwrap() - 0.95).abs() < 1e-5);
    }

    #[test]
    fn audio_json_raw_segments_opts_in_to_old_format() {
        let result = fake_audio_result(vec![seg(0.0, 1.0, 0.9), seg(0.3, 1.3, 0.95)]);
        let mut buf = Vec::new();
        write_audio_output(
            &mut buf,
            Path::new("bird.wav"),
            "md-audiobirds-v1",
            &result,
            &OutputFormat::Json,
            /* raw_segments = */ true,
            0.31,
        )
        .unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(
            output.contains("\"segments\""),
            "--raw-segments must emit `segments`"
        );
        assert!(
            !output.contains("\"ranges\""),
            "--raw-segments must NOT emit `ranges`"
        );
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        let segs = parsed["segments"].as_array().unwrap();
        assert_eq!(segs.len(), 2, "raw segments preserved verbatim");
    }

    #[test]
    fn audio_raw_json_classes_and_class_aware_merge() {
        let result = fake_audio_result(vec![
            seg(0.0, 1.0, 0.4),
            seg_with_classes(1.0, 2.0, vec![audio_class(7, "single", 0.7)]),
            seg_with_classes(
                2.0,
                3.0,
                vec![
                    audio_class(11, "species-a", 0.9),
                    audio_class(12, "species-b", 0.08),
                ],
            ),
        ]);
        let mut buf = Vec::new();
        write_audio_output_raw(
            &mut buf,
            Path::new("bird.wav"),
            "perch-v2",
            &result,
            &OutputFormat::Json,
        )
        .unwrap();
        let output = String::from_utf8(buf).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        let segments = parsed["segments"].as_array().unwrap();
        assert!(
            segments[0].get("classes").is_none(),
            "empty class list must omit classes: {output}"
        );
        assert!(
            segments[1].get("classes").is_none(),
            "single-class binary-compatible segment must omit classes: {output}"
        );
        let classes = segments[2]["classes"].as_array().unwrap();
        assert_eq!(classes.len(), 2);
        assert_eq!(classes[0]["class_idx"], 11);
        assert_eq!(classes[0]["label"], "species-a");
        assert!((classes[0]["probability"].as_f64().unwrap() - 0.9).abs() < 1e-6);
        assert_eq!(classes[1]["class_idx"], 12);
        assert_eq!(classes[1]["label"], "species-b");
        assert!((classes[1]["probability"].as_f64().unwrap() - 0.08).abs() < 1e-6);

        let different_top1 = fake_audio_result(vec![
            seg_with_classes(
                0.0,
                1.0,
                vec![
                    audio_class(11, "species-a", 0.9),
                    audio_class(12, "species-b", 0.08),
                ],
            ),
            seg_with_classes(
                1.2,
                2.2,
                vec![
                    audio_class(13, "species-c", 0.85),
                    audio_class(11, "species-a", 0.1),
                ],
            ),
        ]);
        let mut buf = Vec::new();
        write_audio_output_merged(
            &mut buf,
            Path::new("bird.wav"),
            "perch-v2",
            &different_top1,
            &OutputFormat::Json,
            0.31,
        )
        .unwrap();
        let output = String::from_utf8(buf).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        let ranges = parsed["ranges"].as_array().unwrap();
        assert_eq!(
            ranges.len(),
            2,
            "adjacent classifier windows with different top-1 labels must not merge: {output}"
        );
        assert_eq!(ranges[0]["class"], "species-a");
        assert_eq!(ranges[1]["class"], "species-c");

        let same_top1 = fake_audio_result(vec![
            seg_with_classes(
                0.0,
                1.0,
                vec![
                    audio_class(11, "species-a", 0.9),
                    audio_class(12, "species-b", 0.08),
                ],
            ),
            seg_with_classes(
                1.2,
                2.2,
                vec![
                    audio_class(11, "species-a", 0.8),
                    audio_class(13, "species-c", 0.15),
                ],
            ),
        ]);
        let mut buf = Vec::new();
        write_audio_output_merged(
            &mut buf,
            Path::new("bird.wav"),
            "perch-v2",
            &same_top1,
            &OutputFormat::Json,
            0.31,
        )
        .unwrap();
        let output = String::from_utf8(buf).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(output.trim()).unwrap();
        let ranges = parsed["ranges"].as_array().unwrap();
        assert_eq!(
            ranges.len(),
            1,
            "adjacent classifier windows with the same top-1 label must merge: {output}"
        );
        assert_eq!(ranges[0]["class"], "species-a");
        assert!((ranges[0]["end_time_s"].as_f64().unwrap() - 2.2).abs() < 1e-6);
    }

    #[test]
    fn audio_csv_default_uses_merged_schema() {
        let result = fake_audio_result(vec![
            seg(0.0, 1.0, 0.9),
            seg(0.3, 1.3, 0.95),
            seg(5.0, 6.0, 0.88),
        ]);
        let mut buf = Vec::new();
        write_audio_output(
            &mut buf,
            Path::new("bird.wav"),
            "md-audiobirds-v1",
            &result,
            &OutputFormat::Csv,
            false,
            0.31,
        )
        .unwrap();
        let output = String::from_utf8(buf).unwrap();
        // Two ranges (silence gap splits 5.0 s from the 0.0–1.3 s merge).
        let lines: Vec<_> = output.lines().collect();
        assert_eq!(lines.len(), 2, "expect 2 merged-range rows, got: {output}");
        // class column empty for binary detector.
        for line in &lines {
            assert!(
                line.ends_with(','),
                "class column should be empty trailing comma: {line}"
            );
        }
    }

    #[test]
    fn audio_csv_raw_segments_uses_old_schema() {
        let result = fake_audio_result(vec![seg(0.0, 1.0, 0.9), seg(0.3, 1.3, 0.95)]);
        let mut buf = Vec::new();
        write_audio_output(
            &mut buf,
            Path::new("bird.wav"),
            "md-audiobirds-v1",
            &result,
            &OutputFormat::Csv,
            true,
            0.31,
        )
        .unwrap();
        let output = String::from_utf8(buf).unwrap();
        let lines: Vec<_> = output.lines().collect();
        assert_eq!(lines.len(), 2, "expect one row per raw segment");
        // Old schema has exactly 6 commas (file,model,idx,start,end,conf -> 5 separators + 5 commas
        // inside the 6 fields; model-id is empty of commas).
        for line in &lines {
            let comma_count = line.chars().filter(|c| *c == ',').count();
            assert_eq!(
                comma_count, 5,
                "raw segment line should have 5 commas (6 columns), got: {line}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Phase 3.5 S5 (#1-cli) — progress bar helper
    // -----------------------------------------------------------------------

    #[test]
    fn progress_bar_hidden_when_quiet() {
        let bar = make_progress_bar(100, /* quiet = */ true);
        // Hidden bars report is_hidden() as true.
        assert!(bar.is_hidden(), "--quiet must hide the bar");
    }

    #[test]
    fn progress_bar_hidden_when_total_zero() {
        // A total of 0 makes the bar meaningless; must hide.
        let bar = make_progress_bar(0, false);
        assert!(bar.is_hidden());
    }

    #[test]
    fn progress_bar_nontty_hidden() {
        // Under `cargo test`, stderr is typically not a TTY (captured by
        // the harness). Confirm the bar auto-hides in that case even
        // when the caller did not pass --quiet.
        let bar = make_progress_bar(10, false);
        // Under a TTY this would be false; the test is only meaningful
        // when the harness captures stderr. We assert the contract
        // (quiet OR non-TTY OR total=0 → hidden); harness typically
        // triggers the non-TTY branch.
        if !io::stderr().is_terminal() {
            assert!(
                bar.is_hidden(),
                "non-TTY stderr should auto-hide the progress bar"
            );
        }
    }
}
