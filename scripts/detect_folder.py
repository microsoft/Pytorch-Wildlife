"""Detect animals, people and vehicles in a folder of images and export the results.

    python scripts/detect_folder.py path/to/images --format megadet --out results.json
"""

import argparse
from pathlib import Path

import sparrow_engine as spe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="image file or folder")
    parser.add_argument("--model", default="MDV6-yolov10-c", help="detector model id")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--recursive", action="store_true", help="include sub-folders")
    parser.add_argument("--format", default="megadet", choices=["megadet", "coco", "csv"])
    parser.add_argument("--out", default="detections.json", help="output file")
    parser.add_argument("--viz", help="also write annotated images to this folder")
    args = parser.parse_args()
    if not Path(args.input).exists():
        parser.error(f"input not found: {args.input}")

    spe.download_model(args.model)
    results = spe.detect(
        args.input, model=args.model, threshold=args.threshold, recursive=args.recursive
    )
    spe.export(results, args.format, args.out, model_id=args.model)
    n = sum(len(r.detections) for r in results)
    print(f"{len(results)} images, {n} detections -> {args.out}")
    if args.viz:
        spe.visualize(results, output_dir=args.viz)
        print(f"annotated images -> {args.viz}")


if __name__ == "__main__":
    main()
