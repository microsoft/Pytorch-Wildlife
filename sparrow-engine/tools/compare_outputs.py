#!/usr/bin/env python3
"""Compare golden (PW baseline) outputs with libsparrow_engine outputs.

Produces side-by-side visualizations, per-model summary grids, and a
machine-readable JSON report with pass/fail status.

Usage:
    uv run --no-project --with pillow,numpy tools/compare_outputs.py \
        --golden test_outputs/golden \
        --libsparrow-engine test_outputs/libsparrow_engine \
        --output test_outputs/comparison

Input directories should contain per-model JSON files in the format produced
by generate_golden_outputs.py (detections with normalized bboxes, or
classifications with top-k labels and confidences).
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BBOX_TOL = 0.005  # Max allowed bbox coordinate difference
CONF_TOL = 0.01  # Max allowed confidence difference
IOU_THRESHOLD = 0.5  # Min IoU to consider a detection match

GREEN = (52, 168, 83)  # Golden
BLUE = (66, 133, 244)  # libsparrow_engine
RED = (234, 67, 53)  # Mismatch
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
DARK_GRAY = (100, 100, 100)
LIGHT_GREEN = (200, 240, 200)
LIGHT_RED = (255, 210, 210)

LINE_WIDTH = 3
FONT_SIZE = 13
LABEL_PAD = 3
COMPARE_WIDTH = 1200  # Width of each side in per-image comparison
THUMB_SIZE = 240  # Thumbnail size in summary grid


# ---------------------------------------------------------------------------
# Font helper
# ---------------------------------------------------------------------------

_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def get_font(size: int = FONT_SIZE) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if size in _font_cache:
        return _font_cache[size]
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    ]:
        try:
            f = ImageFont.truetype(path, size)
            _font_cache[size] = f
            return f
        except (OSError, IOError):
            continue
    f = ImageFont.load_default()
    _font_cache[size] = f
    return f


# ---------------------------------------------------------------------------
# Detection matching
# ---------------------------------------------------------------------------


def iou(a: list[float], b: list[float]) -> float:
    """Compute IoU between two [x1,y1,x2,y2] normalized bboxes."""
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class MatchResult:
    """Result of matching golden detections to libsparrow_engine detections."""

    matched: list[dict] = field(
        default_factory=list
    )  # [{golden, libsparrow_engine, iou, bbox_diff, conf_diff}]
    missed: list[dict] = field(default_factory=list)  # Golden detections with no match
    extra: list[dict] = field(
        default_factory=list
    )  # libsparrow_engine detections with no match
    max_bbox_diff: float = 0.0
    max_conf_diff: float = 0.0
    has_mismatch: bool = False


def match_detections(
    golden_dets: list[dict], libsparrow_engine_dets: list[dict]
) -> MatchResult:
    """Match golden to libsparrow_engine detections by IoU + label, then compare."""
    result = MatchResult()
    used_libsparrow_engine = set()

    for g_det in golden_dets:
        g_bbox = g_det["bbox"]
        g_label = g_det["label"]
        best_iou = 0.0
        best_idx = -1

        for j, l_det in enumerate(libsparrow_engine_dets):
            if j in used_libsparrow_engine:
                continue
            if l_det["label"] != g_label:
                continue
            score = iou(g_bbox, l_det["bbox"])
            if score > best_iou:
                best_iou = score
                best_idx = j

        if best_idx >= 0 and best_iou >= IOU_THRESHOLD:
            used_libsparrow_engine.add(best_idx)
            l_det = libsparrow_engine_dets[best_idx]

            # Compute diffs
            bbox_diffs = [abs(g - l) for g, l in zip(g_bbox, l_det["bbox"])]
            max_bd = max(bbox_diffs)
            conf_diff = abs(g_det["confidence"] - l_det["confidence"])

            result.max_bbox_diff = max(result.max_bbox_diff, max_bd)
            result.max_conf_diff = max(result.max_conf_diff, conf_diff)

            mismatch = max_bd > BBOX_TOL or conf_diff > CONF_TOL
            if mismatch:
                result.has_mismatch = True

            result.matched.append(
                {
                    "golden": g_det,
                    "libsparrow_engine": l_det,
                    "iou": best_iou,
                    "max_bbox_diff": max_bd,
                    "conf_diff": conf_diff,
                    "within_tolerance": not mismatch,
                }
            )
        else:
            result.missed.append(g_det)
            result.has_mismatch = True

    for j, l_det in enumerate(libsparrow_engine_dets):
        if j not in used_libsparrow_engine:
            result.extra.append(l_det)
            result.has_mismatch = True

    return result


# ---------------------------------------------------------------------------
# Classification comparison
# ---------------------------------------------------------------------------


@dataclass
class ClassCompareResult:
    top1_match: bool = False
    top1_golden: str = ""
    top1_libsparrow_engine: str = ""
    conf_diff: float = 0.0
    within_tolerance: bool = False
    golden_top5: list[dict] = field(default_factory=list)
    libsparrow_engine_top5: list[dict] = field(default_factory=list)


def compare_classifications(
    golden_cls: list[dict], libsparrow_engine_cls: list[dict]
) -> ClassCompareResult:
    result = ClassCompareResult()
    if golden_cls:
        result.top1_golden = golden_cls[0]["label"]
        result.golden_top5 = golden_cls[:5]
    if libsparrow_engine_cls:
        result.top1_libsparrow_engine = libsparrow_engine_cls[0]["label"]
        result.libsparrow_engine_top5 = libsparrow_engine_cls[:5]

    result.top1_match = result.top1_golden == result.top1_libsparrow_engine
    if golden_cls and libsparrow_engine_cls:
        result.conf_diff = abs(
            golden_cls[0]["confidence"] - libsparrow_engine_cls[0]["confidence"]
        )
    result.within_tolerance = result.top1_match and result.conf_diff <= CONF_TOL
    return result


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------


def draw_bboxes(img: Image.Image, detections: list[dict], color: tuple) -> Image.Image:
    """Draw bounding boxes on a copy of img."""
    overlay = img.copy()
    draw = ImageDraw.Draw(overlay)
    font = get_font()
    w, h = img.size

    for det in detections:
        bbox = det["bbox"]
        x1, y1, x2, y2 = bbox[0] * w, bbox[1] * h, bbox[2] * w, bbox[3] * h
        draw.rectangle([x1, y1, x2, y2], outline=color, width=LINE_WIDTH)

        text = f"{det.get('label', '?')} {det.get('confidence', 0):.3f}"
        tb = draw.textbbox((0, 0), text, font=font)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ly = max(0, y1 - th - 2 * LABEL_PAD)
        draw.rectangle(
            [x1, ly, x1 + tw + 2 * LABEL_PAD, ly + th + 2 * LABEL_PAD], fill=color
        )
        draw.text((x1 + LABEL_PAD, ly + LABEL_PAD), text, fill=WHITE, font=font)

    return overlay


def resize_to_width(img: Image.Image, target_w: int) -> Image.Image:
    """Resize maintaining aspect ratio to target width."""
    w, h = img.size
    ratio = target_w / w
    return img.resize((target_w, int(h * ratio)), Image.BILINEAR)


def create_per_image_comparison(
    original_img: Image.Image,
    golden_dets: list[dict] | None,
    libsparrow_engine_dets: list[dict] | None,
    match_result: MatchResult | None,
    class_result: ClassCompareResult | None,
    model_name: str,
    image_name: str,
) -> Image.Image:
    """Create a side-by-side comparison image for one image."""
    side_w = COMPARE_WIDTH // 2

    # Draw golden and libsparrow_engine overlays
    golden_overlay = draw_bboxes(original_img, golden_dets or [], GREEN)
    libsparrow_engine_overlay = draw_bboxes(
        original_img, libsparrow_engine_dets or [], BLUE
    )

    golden_resized = resize_to_width(golden_overlay, side_w)
    libsparrow_engine_resized = resize_to_width(libsparrow_engine_overlay, side_w)

    img_h = golden_resized.size[1]

    # Metrics text area height
    metrics_h = 120
    if class_result:
        metrics_h = 200

    total_h = 30 + img_h + metrics_h  # header + images + metrics
    has_mismatch = False
    if match_result:
        has_mismatch = match_result.has_mismatch
    if class_result:
        has_mismatch = not class_result.within_tolerance

    canvas = Image.new("RGB", (COMPARE_WIDTH, total_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    font = get_font()
    font_sm = get_font(11)

    # Header
    border_color = RED if has_mismatch else GREEN
    draw.rectangle(
        [0, 0, COMPARE_WIDTH - 1, total_h - 1], outline=border_color, width=4
    )
    header = f"{model_name} / {image_name}"
    status = "MISMATCH" if has_mismatch else "MATCH"
    status_color = RED if has_mismatch else GREEN
    draw.text((10, 8), header, fill=BLACK, font=font)
    draw.text((COMPARE_WIDTH - 120, 8), status, fill=status_color, font=font)

    # Side-by-side images
    y_img = 30
    canvas.paste(golden_resized, (0, y_img))
    canvas.paste(libsparrow_engine_resized, (side_w, y_img))

    # Labels above images
    draw.text((10, y_img + 2), "GOLDEN (green)", fill=GREEN, font=font_sm)
    draw.text(
        (side_w + 10, y_img + 2), "LIBSPARROW_ENGINE (blue)", fill=BLUE, font=font_sm
    )

    # Divider line
    draw.line([(side_w, y_img), (side_w, y_img + img_h)], fill=GRAY, width=1)

    # Metrics area
    y_met = y_img + img_h + 10
    if match_result:
        lines = [
            f"Matched: {len(match_result.matched)}  |  Missed: {len(match_result.missed)}  |  Extra: {len(match_result.extra)}",
            f"Max bbox diff: {match_result.max_bbox_diff:.6f}  (tol: {BBOX_TOL})"
            + ("  FAIL" if match_result.max_bbox_diff > BBOX_TOL else "  OK"),
            f"Max conf diff: {match_result.max_conf_diff:.6f}  (tol: {CONF_TOL})"
            + ("  FAIL" if match_result.max_conf_diff > CONF_TOL else "  OK"),
        ]
        for i, line in enumerate(lines):
            color = RED if "FAIL" in line else DARK_GRAY
            draw.text((10, y_met + i * 18), line, fill=color, font=font_sm)

    if class_result:
        lines = [
            f"Top-1 label: golden={class_result.top1_golden} | libsparrow_engine={class_result.top1_libsparrow_engine}"
            + ("  MATCH" if class_result.top1_match else "  MISMATCH"),
            f"Top-1 conf diff: {class_result.conf_diff:.6f}  (tol: {CONF_TOL})"
            + ("  OK" if class_result.conf_diff <= CONF_TOL else "  FAIL"),
            "",
            "Top-5 comparison:",
        ]
        for i in range(5):
            g = (
                class_result.golden_top5[i]
                if i < len(class_result.golden_top5)
                else None
            )
            l = (
                class_result.libsparrow_engine_top5[i]
                if i < len(class_result.libsparrow_engine_top5)
                else None
            )
            g_str = f"{g['label'][:30]} ({g['confidence']:.4f})" if g else "---"
            l_str = f"{l['label'][:30]} ({l['confidence']:.4f})" if l else "---"
            lines.append(f"  {i + 1}. {g_str:45s} vs {l_str}")

        for i, line in enumerate(lines):
            color = RED if "MISMATCH" in line or "FAIL" in line else DARK_GRAY
            draw.text((10, y_met + i * 16), line, fill=color, font=font_sm)

    return canvas


def create_model_summary(
    comparisons: list[Image.Image],
    model_name: str,
    overall_pass: bool,
    num_images: int,
    max_bbox_diff: float,
    max_conf_diff: float,
) -> Image.Image:
    """Create a summary grid of all per-image comparisons for a model."""
    cols = 2
    rows = (len(comparisons) + cols - 1) // cols

    # Resize comparisons to thumbnails
    thumbs = []
    for c in comparisons:
        t = resize_to_width(c, COMPARE_WIDTH // 2)
        thumbs.append(t)

    if not thumbs:
        # Empty placeholder
        canvas = Image.new("RGB", (COMPARE_WIDTH, 100), WHITE)
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, 40),
            f"{model_name}: no images to compare",
            fill=DARK_GRAY,
            font=get_font(),
        )
        return canvas

    thumb_h = max(t.size[1] for t in thumbs)
    header_h = 50
    total_w = COMPARE_WIDTH
    total_h = header_h + rows * (thumb_h + 10) + 10

    canvas = Image.new("RGB", (total_w, total_h), WHITE)
    draw = ImageDraw.Draw(canvas)
    font = get_font(16)

    # Header
    status = "PASS" if overall_pass else "FAIL"
    status_color = GREEN if overall_pass else RED
    border_color = GREEN if overall_pass else RED
    draw.rectangle([0, 0, total_w - 1, total_h - 1], outline=border_color, width=4)
    draw.text((10, 10), f"{model_name}", fill=BLACK, font=font)
    draw.text((300, 10), f"Status: {status}", fill=status_color, font=font)
    draw.text(
        (500, 10),
        f"Images: {num_images}  |  Max bbox diff: {max_bbox_diff:.6f}  |  Max conf diff: {max_conf_diff:.6f}",
        fill=DARK_GRAY,
        font=get_font(11),
    )

    # Grid
    for i, thumb in enumerate(thumbs):
        col = i % cols
        row = i // cols
        x = col * (total_w // cols) + 5
        y = header_h + row * (thumb_h + 10) + 5
        canvas.paste(thumb, (x, y))

    return canvas


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def build_report(model_results: dict) -> dict:
    """Build the machine-readable JSON report."""
    report = {"models": {}}
    for model_name, data in model_results.items():
        report["models"][model_name] = {
            "status": "PASS" if data["pass"] else "FAIL",
            "model_type": data.get("model_type", "detector"),
            "images_tested": data["images_tested"],
            "max_bbox_diff": data.get("max_bbox_diff", 0.0),
            "max_confidence_diff": data.get("max_conf_diff", 0.0),
            "detection_count_matches": data.get("detection_count_matches", True),
            "mismatches": data.get("mismatches", []),
        }
    return report


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def find_image(image_name: str, search_dirs: list[Path]) -> Path | None:
    """Find an original image in one of the search directories."""
    for d in search_dirs:
        p = d / image_name
        if p.exists():
            return p
    return None


def process_model(
    model_name: str,
    golden_data: dict,
    libsparrow_engine_data: dict,
    image_dirs: list[Path],
    output_dir: Path,
) -> dict:
    """Compare one model's golden vs libsparrow_engine outputs. Returns model result dict."""
    model_type = golden_data.get("model_type", "detector")
    model_out = output_dir / model_name
    model_out.mkdir(parents=True, exist_ok=True)

    # Index libsparrow_engine images by name
    libsparrow_engine_by_image = {}
    for img_result in libsparrow_engine_data.get("images", []):
        libsparrow_engine_by_image[img_result["image"]] = img_result

    comparisons = []
    overall_pass = True
    max_bbox_diff = 0.0
    max_conf_diff = 0.0
    detection_count_matches = True
    mismatches = []
    images_tested = 0

    for g_img in golden_data.get("images", []):
        image_name = g_img["image"]
        l_img = libsparrow_engine_by_image.get(image_name)

        if l_img is None:
            mismatches.append(
                {"image": image_name, "reason": "missing from libsparrow_engine output"}
            )
            overall_pass = False
            continue

        images_tested += 1

        # Load original image for overlay
        orig_path = find_image(image_name, image_dirs)
        if orig_path:
            original = Image.open(orig_path).convert("RGB")
        else:
            # Create a placeholder
            w = g_img.get("image_width", 640)
            h = g_img.get("image_height", 480)
            original = Image.new("RGB", (w, h), GRAY)

        match_result = None
        class_result = None

        if model_type == "classifier":
            g_cls = g_img.get("classifications", [])
            l_cls = l_img.get("classifications", [])
            class_result = compare_classifications(g_cls, l_cls)
            if not class_result.within_tolerance:
                overall_pass = False
                mismatches.append(
                    {
                        "image": image_name,
                        "reason": "classification mismatch",
                        "top1_golden": class_result.top1_golden,
                        "top1_libsparrow_engine": class_result.top1_libsparrow_engine,
                        "conf_diff": class_result.conf_diff,
                    }
                )
            max_conf_diff = max(max_conf_diff, class_result.conf_diff)
            golden_dets = None
            libsparrow_engine_dets = None
        else:
            golden_dets = g_img.get("detections", [])
            libsparrow_engine_dets = l_img.get("detections", [])
            match_result = match_detections(golden_dets, libsparrow_engine_dets)

            if len(golden_dets) != len(libsparrow_engine_dets):
                detection_count_matches = False

            max_bbox_diff = max(max_bbox_diff, match_result.max_bbox_diff)
            max_conf_diff = max(max_conf_diff, match_result.max_conf_diff)

            if match_result.has_mismatch:
                overall_pass = False
                mismatch_info = {"image": image_name, "reasons": []}
                if match_result.missed:
                    mismatch_info["reasons"].append(
                        f"{len(match_result.missed)} missed detections"
                    )
                if match_result.extra:
                    mismatch_info["reasons"].append(
                        f"{len(match_result.extra)} extra detections"
                    )
                if match_result.max_bbox_diff > BBOX_TOL:
                    mismatch_info["reasons"].append(
                        f"bbox diff {match_result.max_bbox_diff:.6f} > {BBOX_TOL}"
                    )
                if match_result.max_conf_diff > CONF_TOL:
                    mismatch_info["reasons"].append(
                        f"conf diff {match_result.max_conf_diff:.6f} > {CONF_TOL}"
                    )
                mismatches.append(mismatch_info)

        # Create per-image comparison
        comp = create_per_image_comparison(
            original,
            golden_dets,
            libsparrow_engine_dets,
            match_result,
            class_result,
            model_name,
            image_name,
        )
        stem = Path(image_name).stem
        comp.save(model_out / f"{stem}_compare.jpg", quality=90)
        comparisons.append(comp)

    # Create summary grid
    summary = create_model_summary(
        comparisons,
        model_name,
        overall_pass,
        images_tested,
        max_bbox_diff,
        max_conf_diff,
    )
    summary.save(output_dir / f"{model_name}_summary.jpg", quality=90)

    return {
        "pass": overall_pass,
        "model_type": model_type,
        "images_tested": images_tested,
        "max_bbox_diff": max_bbox_diff,
        "max_conf_diff": max_conf_diff,
        "detection_count_matches": detection_count_matches,
        "mismatches": mismatches,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare golden (PW) vs libsparrow_engine outputs with side-by-side visualizations"
    )
    parser.add_argument(
        "--golden",
        required=True,
        type=Path,
        help="Directory with golden reference JSONs",
    )
    parser.add_argument(
        "--libsparrow-engine",
        required=True,
        type=Path,
        help="Directory with libsparrow_engine output JSONs",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for comparison images and report",
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=None,
        help="Directory with original test images (auto-detected if omitted)",
    )
    args = parser.parse_args()

    if not args.golden.exists():
        print(f"ERROR: golden directory not found: {args.golden}", file=sys.stderr)
        sys.exit(1)
    if not args.libsparrow_engine.exists():
        print(
            f"ERROR: libsparrow_engine directory not found: {args.libsparrow_engine}",
            file=sys.stderr,
        )
        sys.exit(1)

    args.output.mkdir(parents=True, exist_ok=True)

    # Image search directories
    image_dirs = []
    if args.images:
        image_dirs.append(args.images)
    # Auto-detect common locations
    repo_root = Path(__file__).resolve().parent.parent
    for candidate in [
        repo_root.parent / "test_files" / "test_data",
        repo_root / "test_data",
    ]:
        if candidate.exists():
            image_dirs.append(candidate)

    # Load golden JSONs
    golden_files = {f.stem: f for f in sorted(args.golden.glob("*.json"))}
    libsparrow_engine_files = {
        f.stem: f for f in sorted(args.libsparrow_engine.glob("*.json"))
    }

    if not golden_files:
        print(f"No golden JSON files found in {args.golden}", file=sys.stderr)
        sys.exit(1)

    print(f"Golden models:   {list(golden_files.keys())}")
    print(f"libsparrow_engine models: {list(libsparrow_engine_files.keys())}")
    print()

    model_results = {}

    for model_name, golden_path in golden_files.items():
        libsparrow_engine_path = libsparrow_engine_files.get(model_name)
        if libsparrow_engine_path is None:
            print(f"SKIP {model_name}: no libsparrow_engine output found")
            model_results[model_name] = {
                "pass": False,
                "model_type": "unknown",
                "images_tested": 0,
                "max_bbox_diff": 0.0,
                "max_conf_diff": 0.0,
                "detection_count_matches": False,
                "mismatches": [{"reason": "SKIPPED — no libsparrow_engine output"}],
            }
            continue

        print(f"Comparing {model_name}...")
        with open(golden_path) as f:
            golden_data = json.load(f)
        with open(libsparrow_engine_path) as f:
            libsparrow_engine_data = json.load(f)

        result = process_model(
            model_name, golden_data, libsparrow_engine_data, image_dirs, args.output
        )
        model_results[model_name] = result

        status = "PASS" if result["pass"] else "FAIL"
        print(
            f"  {status} — {result['images_tested']} images, "
            f"max bbox diff: {result['max_bbox_diff']:.6f}, "
            f"max conf diff: {result['max_conf_diff']:.6f}"
        )
        if result["mismatches"]:
            for m in result["mismatches"][:3]:
                print(f"    - {m}")
            if len(result["mismatches"]) > 3:
                print(f"    ... and {len(result['mismatches']) - 3} more")
        print()

    # Write report
    report = build_report(model_results)
    report_path = args.output / "report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Report: {report_path}")

    # Overall status
    all_pass = all(r["pass"] for r in model_results.values())
    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
