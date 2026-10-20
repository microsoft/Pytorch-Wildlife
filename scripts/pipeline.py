"""Detect animals, then classify each detected crop to species.

    python scripts/pipeline.py path/to/images --classifier AI4G-Amazon-V2 --out results.csv
"""

import argparse

import sparrow_engine as spe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="image file or folder")
    parser.add_argument("--detector", default="MDV6-yolov10-c")
    parser.add_argument("--classifier", default="AI4G-Amazon-V2")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--format", default="csv", choices=["megadet", "coco", "csv"])
    parser.add_argument("--out", default="pipeline.csv")
    args = parser.parse_args()

    spe.download_models([args.detector, args.classifier])
    results = spe.pipeline(
        args.input,
        detector=args.detector,
        classifier=args.classifier,
        threshold=args.threshold,
        recursive=args.recursive,
    )
    for r in results:
        for d in r.detections:
            c = d.classification
            species = f"{c.label} {c.confidence:.2f}" if c else "-"
            print(f"{r.path}\t{d.detection.label} {d.detection.confidence:.2f}\t{species}")
    spe.export(results, args.format, args.out, model_id=args.detector)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
