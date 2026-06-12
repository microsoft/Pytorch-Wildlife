//! `spe-mobile` — command-line front end for the sparrow-engine-mobile orca cascade.
//!
//! Peer to `spe` / `spe-gpu`, scoped to what the mobile flavor currently exposes:
//! the two-stage orca cascade (detector -> ecotype) on the LiteRT backend. Lets an
//! operator run inference on the Pi without the water-sparrow Python app.
//!
//! Built only with `--features cli` (keeps the default cdylib lean for FFI consumers):
//!   cargo build -p sparrow-engine-mobile --features cli --bin spe-mobile --release
//!
//! Example:
//!   spe-mobile detect-audio \
//!     --detector orca-detector-fp16.tflite \
//!     --ecotype  orca-ecotype-melinput-fp16.tflite \
//!     --threads 4 --labels SRKW,TKW,SAR,NRKW,OKW recording.wav

use std::path::PathBuf;
use std::process::ExitCode;

use clap::{Parser, Subcommand, ValueEnum};

use sparrow_engine::cascade::{OrcaCascade, OrcaCascadeResult, ORCA_SAMPLE_RATE};
use sparrow_engine::preprocess_audio::load_audio_at_sample_rate;
use sparrow_engine::AudioInput;

/// Skip degenerate tail windows shorter than this many samples (matches water-sparrow).
const MIN_WINDOW_SAMPLES: usize = 16;
/// Default ecotype abstention threshold (calibrated; below this -> Unassigned).
const DEFAULT_ABSTENTION: f32 = 0.940_095_8;

#[derive(Parser)]
#[command(
    name = "spe-mobile",
    version,
    about = "sparrow-engine mobile orca cascade CLI (LiteRT backend)"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Run the orca two-stage cascade over one or more WAV files.
    DetectAudio(DetectAudioArgs),
}

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Text,
    Json,
}

#[derive(Parser)]
struct DetectAudioArgs {
    /// Orca detector .tflite (stage 1).
    #[arg(long)]
    detector: PathBuf,
    /// Orca ecotype .tflite (stage 2, mel-input).
    #[arg(long)]
    ecotype: PathBuf,
    /// LiteRT CPU inference threads (0 = LiteRT default).
    #[arg(long, default_value_t = 4)]
    threads: usize,
    /// Sliding-window length in seconds.
    #[arg(long, default_value_t = 3.0)]
    window_sec: f32,
    /// Sliding-window overlap in seconds (must be < window-sec).
    #[arg(long, default_value_t = 1.5)]
    overlap_sec: f32,
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

struct WindowResult {
    start_s: f32,
    end_s: f32,
    res: OrcaCascadeResult,
}

/// File-level aggregate of the per-window results.
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

fn ecotype_max_prob(res: &OrcaCascadeResult) -> f32 {
    res.ecotype_probabilities
        .as_ref()
        .map(|p| p.iter().cloned().fold(f32::MIN, f32::max))
        .unwrap_or(res.detector_probability)
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
    }
}

fn run_detect_audio(args: DetectAudioArgs) -> anyhow::Result<()> {
    if args.window_sec <= 0.0 {
        anyhow::bail!("--window-sec must be > 0");
    }
    if args.overlap_sec >= args.window_sec {
        anyhow::bail!(
            "--overlap-sec ({}) must be < --window-sec ({})",
            args.overlap_sec,
            args.window_sec
        );
    }

    let win = (args.window_sec * ORCA_SAMPLE_RATE as f32).round() as usize;
    let step =
        (((args.window_sec - args.overlap_sec) * ORCA_SAMPLE_RATE as f32).round() as usize).max(1);

    let mut cascade =
        OrcaCascade::load(&args.detector, &args.ecotype, args.threads).map_err(|e| {
            anyhow::anyhow!(
                "load cascade (detector={}, ecotype={}): {e:#}",
                args.detector.display(),
                args.ecotype.display()
            )
        })?;

    let mut reports = Vec::new();
    for input in &args.inputs {
        let audio =
            load_audio_at_sample_rate(&AudioInput::FilePath(input.clone()), ORCA_SAMPLE_RATE)
                .map_err(|e| anyhow::anyhow!("decode {}: {e:#}", input.display()))?;
        let samples = audio.data;
        let mut windows = Vec::new();
        if samples.len() >= MIN_WINDOW_SAMPLES {
            let mut start = 0usize;
            loop {
                let end = (start + win).min(samples.len());
                if end - start >= MIN_WINDOW_SAMPLES {
                    let res = cascade
                        .run_segment(&samples[start..end], ORCA_SAMPLE_RATE)
                        .map_err(|e| {
                            anyhow::anyhow!(
                                "cascade {} @ {:.1}s: {e:#}",
                                input.display(),
                                start as f32 / ORCA_SAMPLE_RATE as f32
                            )
                        })?;
                    windows.push(WindowResult {
                        start_s: start as f32 / ORCA_SAMPLE_RATE as f32,
                        end_s: end as f32 / ORCA_SAMPLE_RATE as f32,
                        res,
                    });
                }
                if end >= samples.len() {
                    break;
                }
                start += step;
            }
        }
        reports.push((input.clone(), windows));
    }

    match args.format {
        Format::Text => print_text(&reports, &args.labels, args.abstention),
        Format::Json => print_json(&reports, &args.labels, args.abstention)?,
    }
    Ok(())
}

/// Detected = any orca window; best = highest-confidence orca window (abstention applied).
fn aggregate(
    windows: &[WindowResult],
    labels: &Option<Vec<String>>,
    abstention: f32,
) -> FileAggregate {
    let best = windows
        .iter()
        .filter(|w| w.res.is_orca)
        .map(|w| (w, ecotype_max_prob(&w.res)))
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
                label_for(w.res.ecotype_argmax.unwrap_or(0), labels)
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
    reports: &[(PathBuf, Vec<WindowResult>)],
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
            let (eco, mp) = match (&w.res.ecotype_probabilities, w.res.ecotype_argmax) {
                (Some(_), Some(i)) => (label_for(i, labels), ecotype_max_prob(&w.res)),
                _ => ("-".to_string(), 0.0),
            };
            println!(
                "  {:>7.1}  {:>9.4}  {:>5}  {:>10}  {:>8.4}",
                w.start_s,
                w.res.detector_probability,
                if w.res.is_orca { "yes" } else { "no" },
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
    reports: &[(PathBuf, Vec<WindowResult>)],
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
                    "detector_probability": w.res.detector_probability,
                    "is_orca": w.res.is_orca,
                    "ecotype_argmax": w.res.ecotype_argmax,
                    "ecotype_probabilities": w.res.ecotype_probabilities,
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
