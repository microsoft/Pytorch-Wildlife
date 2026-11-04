"""Run detection on a video file and write an annotated copy.

    python scripts/video.py clip.mp4 --fps 2 --out clip_annotated.mp4
"""

import argparse
from pathlib import Path

import cv2
import sparrow_engine as spe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("input", help="video file")
    parser.add_argument("--model", default="MDV6-yolov10-c", help="detector model id")
    parser.add_argument("--classifier", help="optional species classifier model id")
    parser.add_argument("--threshold", type=float, default=0.2)
    parser.add_argument("--fps", type=float, default=2.0, help="frames per second to analyse")
    parser.add_argument("--out", default="annotated.mp4")
    args = parser.parse_args()
    if not Path(args.input).exists():
        parser.error(f"input not found: {args.input}")

    spe.download_models([m for m in (args.model, args.classifier) if m])
    writer = None
    for fr in spe.iter_video(
        args.input,
        args.model,
        classifier=args.classifier,
        threshold=args.threshold,
        fps=args.fps,
        keep_frames=True,
    ):
        frame = spe.draw_detections(fr.frame, fr.result)
        if writer is None:
            h, w = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.out, fourcc, args.fps, (w, h))
        writer.write(frame)
        print(f"t={fr.timestamp_s:6.2f} s  {len(fr.result.detections)} detections")
    if writer is not None:
        writer.release()
        print(f"-> {args.out}")


if __name__ == "__main__":
    main()
