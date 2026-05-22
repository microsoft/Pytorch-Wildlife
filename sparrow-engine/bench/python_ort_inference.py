#!/usr/bin/env python3
"""Pure Python ONNX Runtime inference benchmark.

Measures raw inference speed: load model -> preprocess -> infer -> postprocess.
No HTTP overhead. Compares to libsparrow_engine Rust engine.

Usage:
    python bench/python_ort_inference.py --device gpu --model-dir /path/to/onnx --image-dir /path/to/images
    python bench/python_ort_inference.py --device cpu --model-dir /path/to/onnx --image-dir /path/to/images
"""

import argparse
import math
import statistics
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

# ---------------------------------------------------------------------------
# Letterbox preprocessing — matches libsparrow_engine preprocess.rs exactly
# ---------------------------------------------------------------------------

PAD_VALUE = 114.0 / 255.0  # post-normalization pad value
TARGET_SIZE = 1280


def letterbox(img: Image.Image, target_w: int, target_h: int) -> np.ndarray:
    """Letterbox resize matching libsparrow_engine's Rust implementation.

    Steps:
    1. Compute scale = min(target_w/img_w, target_h/img_h)
    2. Resize to (new_w, new_h) with bilinear
    3. Pad symmetrically with 114/255, extra pixel on TOP (PW compat: ceil for top)
    4. Unit normalize to [0,1] float32
    5. NCHW layout: (1, 3, H, W)
    """
    img_w, img_h = img.size
    scale = min(target_w / img_w, target_h / img_h)

    new_w = max(1, min(round(img_w * scale), target_w))
    new_h = max(1, min(round(img_h * scale), target_h))

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0

    pad_x_left = int(math.floor(pad_x))
    pad_y_top = int(math.ceil(pad_y))  # PW compat: extra pixel on TOP

    # Build canvas filled with pad_value
    canvas = np.full((target_h, target_w, 3), PAD_VALUE, dtype=np.float32)

    # Place resized image (unit normalized)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    canvas[pad_y_top : pad_y_top + new_h, pad_x_left : pad_x_left + new_w, :] = arr

    # HWC -> NCHW
    tensor = np.transpose(canvas, (2, 0, 1))  # (3, H, W)
    return np.expand_dims(tensor, axis=0)  # (1, 3, H, W)


# ---------------------------------------------------------------------------
# Postprocessing — YOLOv10 output [1, 300, 6]
# ---------------------------------------------------------------------------


def postprocess(output: np.ndarray, threshold: float) -> int:
    """Filter detections by confidence. Returns count."""
    # output shape: (1, 300, 6) -> each row: [x1, y1, x2, y2, conf, class]
    dets = output[0]  # (300, 6)
    mask = dets[:, 4] >= threshold
    return int(np.sum(mask))


# ---------------------------------------------------------------------------
# ORT session setup
# ---------------------------------------------------------------------------


def create_session(model_path: str, device: str):
    """Create ORT InferenceSession with specified EP."""
    import onnxruntime as ort

    providers = []
    if device == "gpu":
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")

    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_opts, providers=providers)

    # Verify EP
    active_eps = session.get_providers()
    if device == "gpu" and "CUDAExecutionProvider" not in active_eps:
        print(
            f"WARNING: CUDA EP requested but not active. Active: {active_eps}",
            file=sys.stderr,
        )
    print(f"Active providers: {active_eps}", file=sys.stderr)

    return session


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    session,
    image_paths: list[Path],
    threshold: float,
    warmup: int,
) -> tuple[float, list[float], int]:
    """Run inference on all images, return (total_ms, per_image_ms_list, total_detections)."""
    input_name = session.get_inputs()[0].name

    # Warmup
    warmup_imgs = image_paths[:warmup] if len(image_paths) >= warmup else image_paths
    for p in warmup_imgs:
        img = Image.open(p).convert("RGB")
        tensor = letterbox(img, TARGET_SIZE, TARGET_SIZE)
        session.run(None, {input_name: tensor})

    # Timed run
    total_detections = 0
    per_image_ms = []

    t_total_start = time.perf_counter()
    for p in image_paths:
        t_start = time.perf_counter()

        img = Image.open(p).convert("RGB")
        tensor = letterbox(img, TARGET_SIZE, TARGET_SIZE)
        outputs = session.run(None, {input_name: tensor})
        n_dets = postprocess(outputs[0], threshold)

        t_end = time.perf_counter()
        per_image_ms.append((t_end - t_start) * 1000.0)
        total_detections += n_dets

    t_total_end = time.perf_counter()
    total_ms = (t_total_end - t_total_start) * 1000.0

    return total_ms, per_image_ms, total_detections


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DEFAULT_MODEL_DIR = "/home/miao/repos/PW_refactor/test_files/onnx"
DEFAULT_IMAGE_DIR = "/home/miao/repos/PW_refactor/test_files/test_cameratrap"
MODEL_FILENAME = "models_MDV6-yolov10-e_model.onnx"


def main():
    parser = argparse.ArgumentParser(
        description="Pure Python ONNX Runtime inference benchmark"
    )
    parser.add_argument(
        "--device",
        choices=["gpu", "cpu"],
        default="gpu",
        help="Execution provider (default: gpu)",
    )
    parser.add_argument(
        "--model-dir",
        default=DEFAULT_MODEL_DIR,
        help=f"Directory containing ONNX model (default: {DEFAULT_MODEL_DIR})",
    )
    parser.add_argument(
        "--image-dir",
        default=DEFAULT_IMAGE_DIR,
        help=f"Directory containing test images (default: {DEFAULT_IMAGE_DIR})",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.40,
        help="Confidence threshold (default: 0.40)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup images (default: 3)",
    )
    args = parser.parse_args()

    # Resolve paths
    model_path = Path(args.model_dir) / MODEL_FILENAME
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        print(f"ERROR: Image directory not found: {image_dir}", file=sys.stderr)
        sys.exit(1)

    image_paths = sorted(
        p for p in image_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not image_paths:
        print(f"ERROR: No images found in {image_dir}", file=sys.stderr)
        sys.exit(1)

    n_images = len(image_paths)
    print(f"Model: {model_path}", file=sys.stderr)
    print(f"Images: {n_images} from {image_dir}", file=sys.stderr)
    print(f"Device: {args.device}", file=sys.stderr)
    print(f"Threshold: {args.threshold}", file=sys.stderr)
    print(f"Warmup: {args.warmup} images", file=sys.stderr)

    # Create session
    session = create_session(str(model_path), args.device)

    # Run benchmark
    total_ms, per_image_ms, total_detections = run_benchmark(
        session, image_paths, args.threshold, args.warmup
    )

    mean_ms = statistics.mean(per_image_ms)
    median_ms = statistics.median(per_image_ms)

    # Human-readable output
    print(f"\n--- Results ({args.device.upper()}) ---", file=sys.stderr)
    print(f"Total:       {total_ms:.1f} ms", file=sys.stderr)
    print(f"Per-image:   {mean_ms:.1f} ms mean, {median_ms:.1f} ms median", file=sys.stderr)
    print(f"Images:      {n_images}", file=sys.stderr)
    print(f"Detections:  {total_detections}", file=sys.stderr)

    # Machine-parseable RESULT line (stdout)
    print(
        f"RESULT python_ort {args.device} {total_ms:.1f} {mean_ms:.1f} {total_detections}"
    )


if __name__ == "__main__":
    main()
