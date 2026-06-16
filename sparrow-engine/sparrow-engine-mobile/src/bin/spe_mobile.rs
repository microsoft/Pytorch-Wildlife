//! `spe-mobile` — command-line front end for the sparrow-engine mobile flavor.
//!
//! Peer to `spe` / `spe-gpu`. Drives the generic manifest-driven engine
//! ([`sparrow_engine::engine::Engine`]) over a model catalog directory: it loads
//! a config-described audio cascade (e.g. the orca detector → ecotype cascade)
//! and runs it over WAV files. RP-25-FU-1 replaced the previous hardcoded
//! two-model path with this generic engine + pipeline.
//!
//! Built only with `--features cli` (keeps the default cdylib lean for FFI
//! consumers):
//!   cargo build -p sparrow-engine-mobile --features cli --bin spe-mobile --release
//!
//! Example (model catalog with orca-cascade/pipeline.toml + the two model dirs):
//!   spe-mobile detect-audio \
//!     --model-dir /path/to/model_catalog \
//!     --pipeline orca-cascade \
//!     --threads 4 --labels SRKW,TKW,SAR,NRKW,OKW recording.wav

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

use sparrow_engine::engine::Engine;
use sparrow_engine::pipeline::{CascadeOpts, CascadeSegment};
use sparrow_engine::{AudioInput, DetectOpts, Device, EngineConfig, ImageInput};

/// Default ecotype abstention threshold (calibrated; below this -> Unassigned).
const DEFAULT_ABSTENTION: f32 = 0.940_095_8;

#[derive(Parser)]
#[command(
    name = "spe-mobile",
    version,
    about = "sparrow-engine mobile CLI (LiteRT backend)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run a config-described audio cascade over one or more WAV files.
    DetectAudio(DetectAudioArgs),
    /// Run single-shot image detection (yolo_e2e) over one or more images.
    Detect(DetectArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Text,
    Json,
}

#[derive(Parser)]
struct DetectAudioArgs {
    /// Model catalog directory ({model_dir}/{id}/manifest.toml + pipeline.toml).
    #[arg(long)]
    model_dir: PathBuf,
    /// Pipeline id to load + run (a {model_dir}/{id}/pipeline.toml).
    #[arg(long, default_value = "orca-cascade")]
    pipeline: String,
    /// LiteRT CPU inference threads (0 = LiteRT default).
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Sliding-window length in seconds (default: pipeline manifest value).
    #[arg(long)]
    window_sec: Option<f32>,
    /// Sliding-window overlap in seconds (default: pipeline manifest value).
    #[arg(long)]
    overlap_sec: Option<f32>,
    /// Ecotype abstention threshold; max prob below this reports "Unassigned".
    #[arg(long, default_value_t = DEFAULT_ABSTENTION)]
    abstention: f32,
    /// Optional comma-separated ecotype labels (else the class index is shown).
    #[arg(long, value_delimiter = ',')]
    labels: Option<Vec<String>>,
    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
    /// One or more WAV files.
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

#[derive(Parser)]
struct DetectArgs {
    /// Model catalog directory ({model_dir}/{id}/manifest.toml).
    #[arg(long)]
    model_dir: PathBuf,
    /// Model id to load (a {model_dir}/{id}/manifest.toml, format = "tflite").
    #[arg(long)]
    model: String,
    /// LiteRT CPU inference threads (0 = LiteRT default).
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Minimum confidence threshold (default: manifest value).
    #[arg(long)]
    confidence: Option<f32>,
    /// Cap on the number of detections returned (default: unlimited).
    #[arg(long)]
    max_detections: Option<u32>,
    /// Output format.
    #[arg(long, value_enum, default_value_t = Format::Text)]
    format: Format,
    /// One or more image files (JPEG/PNG).
    #[arg(required = true)]
    inputs: Vec<PathBuf>,
}

fn run_detect(args: DetectArgs) -> anyhow::Result<()> {
    let engine = Engine::new(EngineConfig {
        device: Device::Cpu,
        inter_threads: 0,
        intra_threads: args.threads as u32,
        model_dir: args.model_dir.clone(),
    })?;
    let model = engine
        .load_model_by_id(&args.model)
        .map_err(|e| anyhow::anyhow!("load model '{}': {e:#}", args.model))?;

    let opts = DetectOpts {
        confidence_threshold: args.confidence,
        max_detections: args.max_detections,
    };

    let mut json_items: Vec<serde_json::Value> = Vec::new();
    for input in &args.inputs {
        let result = model
            .detect(&ImageInput::FilePath(input.clone()), &opts)
            .map_err(|e| anyhow::anyhow!("detect {}: {e:#}", input.display()))?;
        match args.format {
            Format::Text => {
                println!(
                    "{}: {} detection(s) [{}x{}, {:.1} ms]",
                    input.display(),
                    result.detections.len(),
                    result.image_width,
                    result.image_height,
                    result.processing_time_ms
                );
                for d in &result.detections {
                    println!(
                        "  {} (id {}, conf {:.4}) bbox [{:.4}, {:.4}, {:.4}, {:.4}]",
                        d.label,
                        d.label_id,
                        d.confidence,
                        d.bbox.x_min,
                        d.bbox.y_min,
                        d.bbox.x_max,
                        d.bbox.y_max
                    );
                }
            }
            Format::Json => {
                json_items.push(serde_json::json!({
                    "image": input.display().to_string(),
                    "image_width": result.image_width,
                    "image_height": result.image_height,
                    "processing_time_ms": result.processing_time_ms,
                    "detections": result
                        .detections
                        .iter()
                        .map(|d| {
                            serde_json::json!({
                                "label": d.label,
                                "label_id": d.label_id,
                                "confidence": d.confidence,
                                "bbox": [d.bbox.x_min, d.bbox.y_min, d.bbox.x_max, d.bbox.y_max],
                            })
                        })
                        .collect::<Vec<_>>(),
                }));
            }
        }
    }
    if matches!(args.format, Format::Json) {
        println!("{}", serde_json::to_string_pretty(&json_items)?);
    }
    Ok(())
}

/// File-level aggregate of the per-window cascade results.
struct FileAggregate {
    detected: bool,
    label: String,
    confidence: f32,
    best_start_s: f32,
    best_end_s: f32,
}

fn label_for(idx: usize, labels: &Option<Vec<String>>) -> String {
    match labels {
        Some(l) if idx < l.len() => l[idx].clone(),
        _ => format!("class_{idx}"),
    }
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    match cli.command {
        Commands::DetectAudio(args) => match run_detect_audio(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::FAILURE
            }
        },
        Commands::Detect(args) => match run_detect(args) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("error: {e:#}");
                ExitCode::FAILURE
            }
        },
    }
}

fn run_detect_audio(args: DetectAudioArgs) -> anyhow::Result<()> {
    if let Some(w) = args.window_sec {
        if w <= 0.0 {
            anyhow::bail!("--window-sec must be > 0");
        }
    }
    if let (Some(w), Some(o)) = (args.window_sec, args.overlap_sec) {
        if o >= w {
            anyhow::bail!("--overlap-sec ({o}) must be < --window-sec ({w})");
        }
    }

    let engine = Engine::new(EngineConfig {
        device: Device::Cpu,
        inter_threads: 0,
        intra_threads: args.threads as u32,
        model_dir: args.model_dir.clone(),
    })?;
    engine
        .load_pipeline_by_id(&args.pipeline)
        .map_err(|e| anyhow::anyhow!("load pipeline '{}': {e:#}", args.pipeline))?;

    let opts = CascadeOpts {
        window_sec: args.window_sec,
        overlap_sec: args.overlap_sec,
        detector_threshold: None,
    };

    let mut reports: Vec<(PathBuf, Vec<CascadeSegment>)> = Vec::new();
    for input in &args.inputs {
        let result = engine
            .run_pipeline(&args.pipeline, &AudioInput::FilePath(input.clone()), &opts)
            .map_err(|e| anyhow::anyhow!("run {} : {e:#}", input.display()))?;
        reports.push((input.clone(), result.segments));
    }

    match args.format {
        Format::Text => print_text(&reports, &args.labels, args.abstention),
        Format::Json => print_json(&reports, &args.labels, args.abstention)?,
    }
    Ok(())
}

/// Detected = any orca window; best = highest-confidence orca window (abstention applied).
fn aggregate(
    windows: &[CascadeSegment],
    labels: &Option<Vec<String>>,
    abstention: f32,
) -> FileAggregate {
    let best = windows
        .iter()
        .filter(|w| w.is_detected)
        .map(|w| (w, w.stage2_confidence))
        .reduce(|a, b| if b.1 > a.1 { b } else { a });

    match best {
        None => FileAggregate {
            detected: false,
            label: "NonBio".to_string(),
            confidence: 0.0,
            best_start_s: 0.0,
            best_end_s: 0.0,
        },
        Some((w, conf)) => {
            let label = if conf < abstention {
                "Unassigned".to_string()
            } else {
                label_for(w.stage2_argmax.unwrap_or(0), labels)
            };
            FileAggregate {
                detected: true,
                label,
                confidence: conf,
                best_start_s: w.start_s,
                best_end_s: w.end_s,
            }
        }
    }
}

fn print_text(
    reports: &[(PathBuf, Vec<CascadeSegment>)],
    labels: &Option<Vec<String>>,
    abstention: f32,
) {
    for (path, windows) in reports {
        let agg = aggregate(windows, labels, abstention);
        println!("{}", path.display());
        println!(
            "  {:>7}  {:>9}  {:>5}  {:>10}  {:>8}",
            "win_s", "det_prob", "orca", "ecotype", "max_prob"
        );
        for w in windows {
            let (eco, mp) = if w.stage2_ran {
                (
                    label_for(w.stage2_argmax.unwrap_or(0), labels),
                    w.stage2_confidence,
                )
            } else {
                ("-".to_string(), 0.0)
            };
            println!(
                "  {:>7.1}  {:>9.4}  {:>5}  {:>10}  {:>8.4}",
                w.start_s,
                w.detector_probability,
                if w.is_detected { "yes" } else { "no" },
                eco,
                mp
            );
        }
        if agg.detected {
            println!(
                "  => detected: {} (confidence {:.4}; best window {:.1}-{:.1}s)",
                agg.label, agg.confidence, agg.best_start_s, agg.best_end_s
            );
        } else {
            println!("  => no orca ({} windows)", windows.len());
        }
    }
}

fn print_json(
    reports: &[(PathBuf, Vec<CascadeSegment>)],
    labels: &Option<Vec<String>>,
    abstention: f32,
) -> anyhow::Result<()> {
    let mut files = Vec::new();
    for (path, windows) in reports {
        let agg = aggregate(windows, labels, abstention);
        let wins: Vec<_> = windows
            .iter()
            .map(|w| {
                serde_json::json!({
                    "start_s": w.start_s,
                    "end_s": w.end_s,
                    "detector_probability": w.detector_probability,
                    "is_detected": w.is_detected,
                    "stage2_ran": w.stage2_ran,
                    "stage2_argmax": w.stage2_argmax,
                    "stage2_confidence": w.stage2_confidence,
                    "stage2_probabilities": w.stage2_probabilities,
                })
            })
            .collect();
        files.push(serde_json::json!({
            "file": path.display().to_string(),
            "detected": agg.detected,
            "label": agg.label,
            "confidence": agg.confidence,
            "best_window_start_s": agg.best_start_s,
            "best_window_end_s": agg.best_end_s,
            "windows": wins,
        }));
    }
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({ "files": files }))?
    );
    Ok(())
}
