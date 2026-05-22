#!/usr/bin/env python3
"""Generate golden reference outputs for libsparrow_engine integration tests.

Runs each ONNX model on 10 test images using libsparrow_engine-identical preprocessing,
saves JSON results and visualization overlays.

Usage:
    uv run --no-project --with onnxruntime,pillow,numpy \
        tools/generate_golden_outputs.py

Output:
    test_outputs/golden/{model}/  — per-image JSON + overlay JPEGs
    test_outputs/golden/summary/  — 2x5 grid images per model
"""

import json
import math
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_FILES = REPO_ROOT.parent / "test_files"
ONNX_DIR = TEST_FILES / "onnx"
CAMERATRAP_DIR = TEST_FILES / "test_cameratrap"
OVERHEAD_DIR = TEST_FILES / "test_overhead"
OUTPUT_ROOT = REPO_ROOT / "test_outputs" / "golden"

NUM_IMAGES = 10

# ImageNet constants (must match sparrow-engine-cpu/src/preprocess.rs)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Model configs (from manifest TOMLs)
# ---------------------------------------------------------------------------

MODELS = {
    "mdv6": {
        "onnx": "models_MDV6-yolov10-e_model.onnx",
        "labels": "models_MDV6-yolov10-e_labels.txt",
        "task": "detection",
        "postprocess": "yolo_e2e",
        "input_size": (1280, 1280),
        "preprocess": "letterbox",
        "layout": "nchw",
        "normalization": "unit",
        "pad_value": 0.447,  # 114/255 post-normalization
        "confidence_threshold": 0.2,
    },
    "deepfaune": {
        "onnx": "models_deepfaune-yolo8s_model.onnx",
        "labels": "models_deepfaune-yolo8s_labels.txt",
        "task": "detection",
        "postprocess": "yolo_e2e",
        "input_size": (960, 960),
        "preprocess": "letterbox",
        "layout": "nchw",
        "normalization": "unit",
        "pad_value": 0.447,
        "confidence_threshold": 0.2,
    },
    "herdnet": {
        "onnx": "models_HerdNet_General_Dataset_2022_model.onnx",
        "labels": "models_HerdNet_General_Dataset_2022_labels.txt",
        "task": "detection",
        "postprocess": "heatmap_peaks",
        "input_size": (512, 512),
        "preprocess": "resize",
        "layout": "nchw",
        "normalization": "imagenet",
        "confidence_threshold": 0.2,
        "peak_threshold": 0.2,
        "point_to_box_half_size": 10,
        "tile_size": (512, 512),
        "tile_overlap": 0,
    },
    "speciesnet": {
        "onnx": "models_classification_SpeciesNet-Crop_model.onnx",
        "labels": "models_classification_SpeciesNet-Crop_labels.txt",
        "task": "classification",
        "postprocess": "softmax",
        "input_size": (480, 480),
        "preprocess": "resize",
        "layout": "nhwc",
        "normalization": "unit",
    },
}


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------


def load_labels(path: Path) -> list[str]:
    """Load labels from name,index CSV format."""
    labels_by_idx = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.rsplit(",", 1)
        if len(parts) == 2:
            name, idx = parts[0], int(parts[1])
            labels_by_idx[idx] = name
        else:
            labels_by_idx[len(labels_by_idx)] = parts[0]
    max_idx = max(labels_by_idx.keys()) if labels_by_idx else -1
    return [labels_by_idx.get(i, f"unknown_{i}") for i in range(max_idx + 1)]


# ---------------------------------------------------------------------------
# Preprocessing — matches sparrow-engine-cpu/src/preprocess.rs exactly
# ---------------------------------------------------------------------------


def letterbox_preprocess(
    img: Image.Image,
    target_w: int,
    target_h: int,
    pad_value: float,
) -> tuple[np.ndarray, dict]:
    """Letterbox resize with unit normalization. Matches sparrow-engine-cpu letterbox().

    - Bilinear resize preserving aspect ratio
    - pad_x_left = floor(pad_x), pad_y_top = ceil(pad_y)
    - Pixel / 255.0 normalization
    - Pad with pad_value (post-normalization scale)
    """
    orig_w, orig_h = img.size
    scale = min(target_w / orig_w, target_h / orig_h)

    new_w = max(1, min(target_w, round(orig_w * scale)))
    new_h = max(1, min(target_h, round(orig_h * scale)))

    resized = img.resize((new_w, new_h), Image.BILINEAR)

    pad_x = (target_w - new_w) / 2.0
    pad_y = (target_h - new_h) / 2.0

    pad_x_left = math.floor(pad_x)
    pad_y_top = math.ceil(pad_y)  # sparrow-engine-cpu: ceil for top padding

    # Build canvas filled with pad_value (already in post-norm scale)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.float32)

    # Place resized image, normalized to [0,1]
    resized_arr = np.array(resized, dtype=np.float32) / 255.0
    canvas[pad_y_top : pad_y_top + new_h, pad_x_left : pad_x_left + new_w] = resized_arr

    # HWC -> NCHW
    tensor = np.transpose(canvas, (2, 0, 1))[np.newaxis, ...]

    meta = {
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
        "original_width": orig_w,
        "original_height": orig_h,
    }
    return tensor, meta


def resize_preprocess(
    img: Image.Image,
    target_w: int,
    target_h: int,
    normalization: str,
    layout: str,
) -> tuple[np.ndarray, dict]:
    """Direct resize. Matches sparrow-engine-cpu resize_direct() + build_tensor().

    - Bilinear resize to exact target dims
    - Unit: pixel / 255.0
    - ImageNet: (pixel/255.0 - mean) / std
    """
    orig_w, orig_h = img.size
    resized = img.resize((target_w, target_h), Image.BILINEAR)
    arr = np.array(resized, dtype=np.float32)

    if normalization == "unit":
        arr = arr / 255.0
    elif normalization == "imagenet":
        arr = (arr / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    elif normalization == "none":
        pass  # raw 0-255
    else:
        raise ValueError(f"Unknown normalization: {normalization}")

    if layout == "nchw":
        tensor = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
    elif layout == "nhwc":
        tensor = arr[np.newaxis, ...]
    else:
        raise ValueError(f"Unknown layout: {layout}")

    meta = {
        "scale": 1.0,
        "pad_x": 0.0,
        "pad_y": 0.0,
        "original_width": orig_w,
        "original_height": orig_h,
    }
    return tensor, meta


def preprocess_image(img: Image.Image, config: dict) -> tuple[np.ndarray, dict]:
    tw, th = config["input_size"]
    if config["preprocess"] == "letterbox":
        return letterbox_preprocess(img, tw, th, config["pad_value"])
    else:
        return resize_preprocess(img, tw, th, config["normalization"], config["layout"])


# ---------------------------------------------------------------------------
# Postprocessing — matches sparrow-engine-core/src/postprocess.rs
# ---------------------------------------------------------------------------


def postprocess_yolo_e2e(
    output: np.ndarray,
    meta: dict,
    labels: list[str],
    threshold: float,
) -> list[dict]:
    """YOLO e2e: output [1, N, 6] -> x1,y1,x2,y2,conf,class_id in model-input space."""
    if output.ndim == 3:
        output = output[0]

    ow = meta["original_width"]
    oh = meta["original_height"]
    scale = meta["scale"]
    px = meta["pad_x"]
    py = meta["pad_y"]

    detections = []
    for row in output:
        conf = float(row[4])
        if conf < threshold:
            continue

        # Denormalize: remove padding -> undo scale -> normalize to [0,1]
        x1 = float(np.clip((row[0] - px) / scale / ow, 0, 1))
        y1 = float(np.clip((row[1] - py) / scale / oh, 0, 1))
        x2 = float(np.clip((row[2] - px) / scale / ow, 0, 1))
        y2 = float(np.clip((row[3] - py) / scale / oh, 0, 1))

        cid = int(row[5])
        label = labels[cid] if cid < len(labels) else f"unknown_{cid}"
        detections.append(
            {
                "bbox": [x1, y1, x2, y2],
                "label": label,
                "label_id": cid,
                "confidence": conf,
            }
        )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def postprocess_heatmap_peaks(
    loc_map: np.ndarray,
    cls_map: np.ndarray,
    labels: list[str],
    config: dict,
) -> list[dict]:
    """Heatmap peak finding. loc_map [1,1,H,W], cls_map [1,C,H,W].

    Matches sparrow-engine-core heatmap_peaks() including tie-breaking.
    """
    loc = loc_map[0, 0]  # [H, W]
    cls = cls_map[0]  # [C, H, W]
    h, w = loc.shape
    threshold = config["peak_threshold"]
    half = config["point_to_box_half_size"]

    detections = []
    for y in range(h):
        for x in range(w):
            val = float(loc[y, x])
            if val < threshold:
                continue

            # 8-connected local max with tie-breaking (south/east strict)
            is_max = True
            for dy in range(-1, 2):
                for dx in range(-1, 2):
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        neighbor = float(loc[ny, nx])
                        is_south_east = dy > 0 or (dy == 0 and dx > 0)
                        if is_south_east:
                            if neighbor >= val:
                                is_max = False
                                break
                        else:
                            if neighbor > val:
                                is_max = False
                                break
                    if not is_max:
                        break
                if not is_max:
                    break

            if not is_max:
                continue

            # Classify
            class_scores = cls[:, y, x]
            cid = int(np.argmax(class_scores))
            cscore = float(class_scores[cid])

            conf = val * cscore
            if conf < threshold:
                continue

            hf = float(h)
            wf = float(w)
            x_min = max(0.0, (x - half) / wf)
            y_min = max(0.0, (y - half) / hf)
            x_max = min(1.0, (x + half) / wf)
            y_max = min(1.0, (y + half) / hf)

            label = labels[cid] if cid < len(labels) else f"unknown_{cid}"
            detections.append(
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "label": label,
                    "label_id": cid,
                    "confidence": conf,
                }
            )

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def postprocess_softmax(
    logits: np.ndarray,
    labels: list[str],
    top_k: int = 5,
) -> list[dict]:
    """Softmax classification. logits [1, num_classes]."""
    if logits.ndim > 1:
        logits = logits[0]

    # Numerically stable softmax
    logits = logits - logits.max()
    exps = np.exp(logits)
    probs = exps / exps.sum()

    top_idx = np.argsort(probs)[::-1][:top_k]
    return [
        {
            "label": labels[int(i)] if int(i) < len(labels) else f"unknown_{i}",
            "label_id": int(i),
            "confidence": float(probs[i]),
        }
        for i in top_idx
    ]


# ---------------------------------------------------------------------------
# HerdNet tiling
# ---------------------------------------------------------------------------


def run_herdnet_tiled(
    session: ort.InferenceSession,
    img: Image.Image,
    config: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Tile image into 512x512 patches, run each, assemble heatmaps."""
    tw, th = config["tile_size"]
    orig_w, orig_h = img.size

    # Number of tiles (ceiling division)
    nx = math.ceil(orig_w / tw)
    ny = math.ceil(orig_h / th)

    # Pad image to exact tile grid
    padded_w = nx * tw
    padded_h = ny * th
    padded = Image.new("RGB", (padded_w, padded_h), (0, 0, 0))
    padded.paste(img, (0, 0))

    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]

    # Collect per-tile detections, then merge
    all_tile_locs = []  # list of (tile_x_offset, tile_y_offset, loc_array, cls_array)

    for ty in range(ny):
        for tx in range(nx):
            x0 = tx * tw
            y0 = ty * th
            tile = padded.crop((x0, y0, x0 + tw, y0 + th))
            tensor, _ = resize_preprocess(tile, tw, th, "imagenet", "nchw")
            outputs = session.run(output_names, {input_name: tensor})
            all_tile_locs.append((x0, y0, outputs[0], outputs[1]))

    return all_tile_locs, (padded_w, padded_h), (orig_w, orig_h)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------


def get_font(size: int = 14):
    """Try to load a TrueType font, fall back to default."""
    try:
        return ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
        )
    except (OSError, IOError):
        try:
            return ImageFont.truetype(
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", size
            )
        except (OSError, IOError):
            return ImageFont.load_default()


def draw_detections(img: Image.Image, detections: list[dict]) -> Image.Image:
    """Draw GREEN bboxes with labels on image copy."""
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    font = get_font(16)
    w, h = vis.size

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        # Convert [0,1] normalized coords to pixel coords
        px1, py1, px2, py2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

        # GREEN rectangle
        draw.rectangle([px1, py1, px2, py2], outline=(0, 255, 0), width=2)

        # Label text
        text = f"{det['label']} {det['confidence']:.2f}"
        bbox = draw.textbbox((0, 0), text, font=font)
        tw, th_text = bbox[2] - bbox[0], bbox[3] - bbox[1]

        # Dark background for readability
        text_y = max(0, py1 - th_text - 4)
        draw.rectangle(
            [px1, text_y, px1 + tw + 4, text_y + th_text + 4], fill=(0, 0, 0)
        )
        draw.text((px1 + 2, text_y + 2), text, fill=(255, 255, 255), font=font)

    return vis


def draw_classifications(img: Image.Image, classifications: list[dict]) -> Image.Image:
    """Draw top-3 classification labels with confidence bars."""
    vis = img.copy()
    draw = ImageDraw.Draw(vis)
    font = get_font(18)

    top3 = classifications[:3]
    y_offset = 10
    bar_width = 200
    bar_height = 24

    for cls in top3:
        text = f"{cls['label']}: {cls['confidence']:.3f}"
        # Background bar
        draw.rectangle(
            [10, y_offset, 10 + bar_width, y_offset + bar_height], fill=(0, 0, 0, 180)
        )
        # Confidence fill
        fill_w = int(cls["confidence"] * bar_width)
        draw.rectangle(
            [10, y_offset, 10 + fill_w, y_offset + bar_height], fill=(0, 180, 0)
        )
        # Text
        draw.text((14, y_offset + 2), text, fill=(255, 255, 255), font=font)
        y_offset += bar_height + 6

    return vis


def make_grid(images: list[Image.Image], title: str, cols: int = 5) -> Image.Image:
    """Create a 2x5 grid of images with title."""
    if not images:
        return Image.new("RGB", (100, 100), (0, 0, 0))

    rows = math.ceil(len(images) / cols)

    # Resize all to same size
    cell_w, cell_h = 400, 300
    title_h = 40

    grid_w = cols * cell_w
    grid_h = title_h + rows * cell_h

    grid = Image.new("RGB", (grid_w, grid_h), (30, 30, 30))
    draw = ImageDraw.Draw(grid)
    font = get_font(24)

    # Title
    draw.text((grid_w // 2 - 100, 8), title, fill=(255, 255, 255), font=font)

    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        thumb = img.copy()
        thumb.thumbnail((cell_w, cell_h), Image.BILINEAR)
        # Center in cell
        x_off = col * cell_w + (cell_w - thumb.width) // 2
        y_off = title_h + row * cell_h + (cell_h - thumb.height) // 2
        grid.paste(thumb, (x_off, y_off))

    return grid


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def process_detection_model(
    model_name: str,
    config: dict,
    images: list[Path],
    session: ort.InferenceSession,
    labels: list[str],
    out_dir: Path,
):
    """Process a YOLO detection model (MDV6 or deepfaune)."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    threshold = config["confidence_threshold"]
    overlays = []

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        tensor, meta = preprocess_image(img, config)

        raw = session.run(output_names, {input_name: tensor})
        detections = postprocess_yolo_e2e(raw[0], meta, labels, threshold)

        stem = img_path.stem

        # Save JSON
        result = {
            "image": img_path.name,
            "model": model_name,
            "image_width": meta["original_width"],
            "image_height": meta["original_height"],
            "preprocess_meta": {
                "scale": meta["scale"],
                "pad_x": meta["pad_x"],
                "pad_y": meta["pad_y"],
            },
            "detections": detections,
        }
        json_path = out_dir / f"{stem}_detections.json"
        json_path.write_text(json.dumps(result, indent=2))

        # Save overlay
        vis = draw_detections(img, detections)
        overlay_path = out_dir / f"{stem}_overlay.jpg"
        vis.save(overlay_path, quality=90)
        overlays.append(vis)

        print(f"  {img_path.name}: {len(detections)} detections")

    return overlays


def process_herdnet(
    config: dict,
    images: list[Path],
    session: ort.InferenceSession,
    labels: list[str],
    out_dir: Path,
):
    """Process HerdNet with tiling.

    Each tile produces its own heatmap. We find peaks per-tile,
    then map detections back to full image coordinates.
    """
    tw, th = config["tile_size"]
    half = config["point_to_box_half_size"]
    peak_thresh = config["peak_threshold"]
    overlays = []

    for img_path in images:
        img = Image.open(img_path).convert("RGB")

        tile_results, (padded_w, padded_h), (orig_w, orig_h) = run_herdnet_tiled(
            session,
            img,
            config,
        )

        all_detections = []
        for tile_x0, tile_y0, loc_map, cls_map in tile_results:
            loc = loc_map[0, 0]  # [loc_h, loc_w]
            cls = cls_map[0]  # [C, cls_h, cls_w]
            loc_h, loc_w = loc.shape
            cls_h, cls_w = cls.shape[1], cls.shape[2]

            # Scale from loc heatmap coords to tile pixel coords
            scale_x = tw / loc_w
            scale_y = th / loc_h
            # Scale from loc coords to cls coords (cls may be lower res)
            cls_scale_y = cls_h / loc_h
            cls_scale_x = cls_w / loc_w

            for y in range(loc_h):
                for x in range(loc_w):
                    val = float(loc[y, x])
                    if val < peak_thresh:
                        continue

                    # 8-connected local max with tie-breaking
                    is_max = True
                    for dy in range(-1, 2):
                        for dx in range(-1, 2):
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < loc_h and 0 <= nx < loc_w:
                                neighbor = float(loc[ny, nx])
                                is_se = dy > 0 or (dy == 0 and dx > 0)
                                if is_se:
                                    if neighbor >= val:
                                        is_max = False
                                        break
                                else:
                                    if neighbor > val:
                                        is_max = False
                                        break
                            if not is_max:
                                break
                        if not is_max:
                            break

                    if not is_max:
                        continue

                    # Classify: map loc coords to cls coords (nearest neighbor)
                    cy = min(int(y * cls_scale_y), cls_h - 1)
                    cx = min(int(x * cls_scale_x), cls_w - 1)
                    class_scores = cls[:, cy, cx]
                    cid = int(np.argmax(class_scores))
                    cscore = float(class_scores[cid])
                    conf = val * cscore
                    if conf < peak_thresh:
                        continue

                    # Map heatmap (y,x) to full-image pixel coords
                    px = tile_x0 + x * scale_x
                    py_coord = tile_y0 + y * scale_y

                    # Point to bbox in normalized [0,1] coords relative to original image
                    x_min = max(0.0, (px - half) / orig_w)
                    y_min = max(0.0, (py_coord - half) / orig_h)
                    x_max = min(1.0, (px + half) / orig_w)
                    y_max = min(1.0, (py_coord + half) / orig_h)

                    label = labels[cid] if cid < len(labels) else f"unknown_{cid}"
                    all_detections.append(
                        {
                            "bbox": [x_min, y_min, x_max, y_max],
                            "label": label,
                            "label_id": cid,
                            "confidence": conf,
                        }
                    )

        all_detections.sort(key=lambda d: d["confidence"], reverse=True)

        stem = img_path.stem
        result = {
            "image": img_path.name,
            "model": "herdnet",
            "image_width": orig_w,
            "image_height": orig_h,
            "tile_size": list(config["tile_size"]),
            "detections": all_detections,
        }
        json_path = out_dir / f"{stem}_detections.json"
        json_path.write_text(json.dumps(result, indent=2))

        vis = draw_detections(img, all_detections)
        overlay_path = out_dir / f"{stem}_overlay.jpg"
        vis.save(overlay_path, quality=90)
        overlays.append(vis)

        print(f"  {img_path.name}: {len(all_detections)} detections")

    return overlays


def process_speciesnet(
    config: dict,
    images: list[Path],
    session: ort.InferenceSession,
    labels: list[str],
    out_dir: Path,
):
    """Process SpeciesNet classifier."""
    input_name = session.get_inputs()[0].name
    output_names = [o.name for o in session.get_outputs()]
    overlays = []

    for img_path in images:
        img = Image.open(img_path).convert("RGB")
        tensor, meta = preprocess_image(img, config)

        raw = session.run(output_names, {input_name: tensor})
        classifications = postprocess_softmax(raw[0], labels, top_k=5)

        stem = img_path.stem

        result = {
            "image": img_path.name,
            "model": "speciesnet",
            "image_width": meta["original_width"],
            "image_height": meta["original_height"],
            "classifications": classifications,
        }
        json_path = out_dir / f"{stem}_classifications.json"
        json_path.write_text(json.dumps(result, indent=2))

        vis = draw_classifications(img, classifications)
        overlay_path = out_dir / f"{stem}_overlay.jpg"
        vis.save(overlay_path, quality=90)
        overlays.append(vis)

        top = classifications[0] if classifications else {"label": "?", "confidence": 0}
        print(f"  {img_path.name}: top={top['label']} ({top['confidence']:.3f})")

    return overlays


# Matches torchvision.datasets.folder.IMG_EXTENSIONS
IMG_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".ppm",
    ".bmp",
    ".pgm",
    ".tif",
    ".tiff",
    ".webp",
}


def get_images(directory: Path, n: int):
    """Get first n images from a directory, sorted alphabetically.
    Supports all extensions from torchvision IMG_EXTENSIONS."""
    imgs = sorted(
        p
        for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS
    )[:n]
    if not imgs:
        print(f"WARNING: No images found in {directory}")
    return imgs


def main():
    cameratrap_images = get_images(CAMERATRAP_DIR, NUM_IMAGES)
    # Single overhead image with known buffalo detections for HerdNet.
    overhead_images = [OVERHEAD_DIR / "S_11_05_16_DSC01556.JPG"]
    assert overhead_images[0].exists(), (
        f"Overhead test image not found: {overhead_images[0]}"
    )

    print(f"Camera trap images: {len(cameratrap_images)}")
    print(f"Overhead images: {len(overhead_images)}")
    print(f"Models: {list(MODELS.keys())}")
    print()

    summary_dir = OUTPUT_ROOT / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    for model_name, config in MODELS.items():
        onnx_path = ONNX_DIR / config["onnx"]
        label_path = ONNX_DIR / config["labels"]

        if not onnx_path.exists():
            print(f"SKIP {model_name}: {onnx_path} not found")
            continue

        # HerdNet uses overhead images; everything else uses camera trap images
        images = overhead_images if model_name == "herdnet" else cameratrap_images

        print(f"=== {model_name} ({len(images)} images) ===")
        labels = load_labels(label_path)
        print(f"  Labels: {len(labels)}")

        session = ort.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )

        out_dir = OUTPUT_ROOT / model_name
        out_dir.mkdir(parents=True, exist_ok=True)

        try:
            if model_name in ("mdv6", "deepfaune"):
                overlays = process_detection_model(
                    model_name,
                    config,
                    images,
                    session,
                    labels,
                    out_dir,
                )
            elif model_name == "herdnet":
                overlays = process_herdnet(config, images, session, labels, out_dir)
            elif model_name == "speciesnet":
                overlays = process_speciesnet(config, images, session, labels, out_dir)
            else:
                print("  Unknown model type, skipping")
                continue

            # Summary grid
            grid = make_grid(overlays, model_name.upper())
            grid_path = summary_dir / f"{model_name}_grid.jpg"
            grid.save(grid_path, quality=90)
            print(f"  Grid saved: {grid_path}")

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback

            traceback.print_exc()

        print()

    print(f"Done. Outputs in {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
