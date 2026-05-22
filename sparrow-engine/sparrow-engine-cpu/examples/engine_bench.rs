//! Direct sparrow-engine inference benchmark — no HTTP, no server.
//!
//! Usage:
//!   cargo run --release --example engine_bench -- --device auto --model-dir /path/to/onnx --image-dir /path/to/images
//!
//! Options:
//!   --device auto|cpu|cuda:0   Compute device (default: auto)
//!   --model-dir PATH           Directory containing mdv6_manifest.toml
//!   --image-dir PATH           Directory with .jpg/.jpeg/.png images
//!   --threshold FLOAT          Confidence threshold (default: 0.2)

use sparrow_engine::detect::detect;
use sparrow_engine::{DetectOpts, Device, Engine, EngineConfig, ImageInput};
use std::path::PathBuf;
use std::time::Instant;

fn parse_args() -> (Device, PathBuf, PathBuf, f32) {
    let args: Vec<String> = std::env::args().collect();

    let mut device_str = "auto".to_string();
    let mut model_dir = None;
    let mut image_dir = None;
    let mut threshold = 0.2_f32;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--device" => {
                i += 1;
                device_str = args[i].clone();
            }
            "--model-dir" => {
                i += 1;
                model_dir = Some(PathBuf::from(&args[i]));
            }
            "--image-dir" => {
                i += 1;
                image_dir = Some(PathBuf::from(&args[i]));
            }
            "--threshold" => {
                i += 1;
                threshold = args[i].parse().expect("invalid threshold");
            }
            other => panic!("unknown argument: {other}"),
        }
        i += 1;
    }

    let device = match device_str.as_str() {
        "auto" => Device::Auto,
        "cpu" => Device::Cpu,
        s if s.starts_with("cuda:") => Device::Cuda(s[5..].parse().expect("invalid CUDA device id")),
        _ => panic!("unknown device: {device_str} (expected auto|cpu|cuda:N)"),
    };

    let model_dir = model_dir.expect("--model-dir is required");
    let image_dir = image_dir.expect("--image-dir is required");

    (device, model_dir, image_dir, threshold)
}

fn collect_images(dir: &PathBuf) -> Vec<PathBuf> {
    let mut images: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("cannot read image dir {}: {e}", dir.display()))
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            match path.extension()?.to_str()?.to_lowercase().as_str() {
                "jpg" | "jpeg" | "png" => Some(path),
                _ => None,
            }
        })
        .collect();
    images.sort();
    images
}

fn main() {
    let (device, model_dir, image_dir, threshold) = parse_args();

    // --- Engine + model ---
    let config = EngineConfig::new(device, &model_dir);
    let engine = Engine::new(config).expect("failed to create engine");
    let manifest_path = model_dir.join("mdv6_manifest.toml");
    let handle = engine
        .load_model(&manifest_path)
        .expect("failed to load model");

    let opts = DetectOpts {
        confidence_threshold: Some(threshold),
        max_detections: None,
    };

    // --- Collect images ---
    let images = collect_images(&image_dir);
    if images.is_empty() {
        panic!("no images found in {}", image_dir.display());
    }

    // Pre-read all files so file I/O is excluded from timing.
    let image_bytes: Vec<Vec<u8>> = images
        .iter()
        .map(|p| std::fs::read(p).unwrap_or_else(|e| panic!("cannot read {}: {e}", p.display())))
        .collect();

    // --- Warmup (discard timing) ---
    let warmup_count = 3.min(image_bytes.len());
    for bytes in &image_bytes[..warmup_count] {
        let _ = detect(&handle, &ImageInput::Encoded(bytes.clone()), &opts);
    }

    // --- Benchmark ---
    let mut times = Vec::with_capacity(image_bytes.len());
    let mut total_dets: usize = 0;
    let start = Instant::now();

    for bytes in &image_bytes {
        let t = Instant::now();
        let result = detect(&handle, &ImageInput::Encoded(bytes.clone()), &opts)
            .expect("detect failed");
        times.push(t.elapsed());
        total_dets += result.detections.len();
    }

    let total = start.elapsed();

    // --- Report ---
    times.sort();
    let n = images.len();
    let mean_ms = total.as_secs_f64() * 1000.0 / n as f64;
    let median = times[n / 2];
    let median_ms = median.as_secs_f64() * 1000.0;
    let total_ms = total.as_secs_f64() * 1000.0;
    let device = engine.active_device();

    println!("=== sparrow-engine direct inference benchmark ===");
    println!("Images:     {n}");
    println!("Device:     {device:?}");
    println!("Threshold:  {threshold}");
    println!("Total:      {total_ms:.1}ms");
    println!("Mean:       {mean_ms:.1}ms/img");
    println!("Median:     {median_ms:.1}ms/img");
    println!("Detections: {total_dets}");
    println!();
    // Machine-parseable line
    println!("RESULT bongo_rust {device:?} {total_ms:.0} {mean_ms:.1} {total_dets}");
}
