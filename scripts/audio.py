"""Run a bioacoustics model over WAV recordings and print the top classes per segment.

    python scripts/audio.py recording.wav --model perch-v2
"""

import argparse
from pathlib import Path

import sparrow_engine as spe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="WAV file or folder of WAV files")
    parser.add_argument("--model", default="perch-v2", help="audio model id")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--top", type=int, default=3, help="classes to print per segment")
    args = parser.parse_args()
    if not Path(args.input).exists():
        parser.error(f"input not found: {args.input}")

    spe.download_model(args.model)
    for result in spe.detect_audio(args.input, model=args.model, threshold=args.threshold):
        print(f"{result.path} ({result.duration_s:.1f} s)")
        for seg in result.segments:
            top = sorted(seg.classes, key=lambda c: c.probability, reverse=True)[: args.top]
            labels = ", ".join(f"{c.label} {c.probability:.2f}" for c in top)
            print(f"  {seg.start_time_s:6.1f}-{seg.end_time_s:6.1f} s  {labels}")


if __name__ == "__main__":
    main()
