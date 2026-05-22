#!/usr/bin/env python3
"""Visualize libsparrow_engine JSON outputs as overlay images with BLUE bounding boxes.

Reads libsparrow_engine output JSONs (same format as golden outputs from
generate_golden_outputs.py) and produces overlay images for visual inspection
and side-by-side comparison.

Usage:
    uv run --no-project --with pillow,numpy tools/visualize_libsparrow_engine_outputs.py \
        --input test_outputs/libsparrow_engine \
        --images <path-to-test-images> \
        --output test_outputs/libsparrow_engine/overlays

Each overlay image has BLUE bounding boxes with labels and confidence scores.
For classifiers, a label table is drawn in the top-left corner.
"""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Colors
BLUE = (66, 133, 244)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Box drawing
LINE_WIDTH = 3
FONT_SIZE = 14
LABEL_PAD = 4


def get_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Try to load a monospace font, fall back to default."""
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ]:
        try:
            return ImageFont.truetype(path, size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def draw_detections(img: Image.Image, detections: list, color: tuple = BLUE) -> Image.Image:
    """Draw bounding boxes with labels on an image copy."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = get_font()
    w, h = img.size

    for det in detections:
        bbox = det["bbox"]  # [x1, y1, x2, y2] normalized [0,1]
        x1 = bbox[0] * w
        y1 = bbox[1] * h
        x2 = bbox[2] * w
        y2 = bbox[3] * h

        # Draw bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_WIDTH)

        # Label text
        label = det.get("label", "?")
        conf = det.get("confidence", 0.0)
        text = f"{label} {conf:.3f}"

        # Label background
        text_bbox = draw.textbbox((0, 0), text, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]
        label_y = max(0, y1 - th - 2 * LABEL_PAD)
        draw.rectangle(
            [x1, label_y, x1 + tw + 2 * LABEL_PAD, label_y + th + 2 * LABEL_PAD],
            fill=color,
        )
        draw.text((x1 + LABEL_PAD, label_y + LABEL_PAD), text, fill=WHITE, font=font)

    return overlay


def draw_classifications(img: Image.Image, classifications: list, color: tuple = BLUE) -> Image.Image:
    """Draw classification results as a label table in the top-left corner."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = get_font()

    lines = ["Classifications:"]
    for i, cls in enumerate(classifications[:5]):
        lines.append(f"  {i+1}. {cls['label']} ({cls['confidence']:.4f})")

    # Compute text block size
    line_heights = []
    max_width = 0
    for line in lines:
        bb = draw.textbbox((0, 0), line, font=font)
        line_heights.append(bb[3] - bb[1])
        max_width = max(max_width, bb[2] - bb[0])

    total_h = sum(line_heights) + len(lines) * 4
    pad = 8

    # Background
    draw.rectangle([pad, pad, pad + max_width + 2 * pad, pad + total_h + 2 * pad], fill=color)

    # Text
    y = pad + pad
    for i, line in enumerate(lines):
        draw.text((pad + pad, y), line, fill=WHITE, font=font)
        y += line_heights[i] + 4

    return overlay


def process_model_json(json_path: Path, image_dir: Path, output_dir: Path) -> int:
    """Process a single model JSON and generate overlay images. Returns count."""
    with open(json_path) as f:
        data = json.load(f)

    model_name = data["model"]
    model_type = data.get("model_type", "detector")
    model_out = output_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_result in data["images"]:
        img_name = img_result["image"]
        img_path = image_dir / img_name

        if not img_path.exists():
            print(f"  WARNING: image not found: {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")

        if model_type == "classifier":
            classifications = img_result.get("classifications", [])
            overlay = draw_classifications(img, classifications)
        else:
            detections = img_result.get("detections", [])
            overlay = draw_detections(img, detections)

        stem = Path(img_name).stem
        out_path = model_out / f"{stem}_overlay.jpg"
        overlay.save(out_path, quality=90)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(
        description="Visualize libsparrow_engine JSON outputs as overlay images"
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Directory containing model JSON files (e.g., test_outputs/libsparrow_engine/)"
    )
    parser.add_argument(
        "--images", required=True, type=Path,
        help="Directory containing original test images"
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output directory for overlay images"
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"ERROR: input directory not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.images.exists():
        print(f"ERROR: images directory not found: {args.images}", file=sys.stderr)
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    json_files = sorted(args.input.glob("*.json"))
    if not json_files:
        print(f"No JSON files found in {args.input}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(json_files)} model JSON files")
    for jf in json_files:
        print(f"Processing {jf.stem}...")
        count = process_model_json(jf, args.images, args.output)
        print(f"  Generated {count} overlay images")

    print(f"Done. Overlays saved to {args.output}/")


if __name__ == "__main__":
    main()
