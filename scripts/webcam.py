"""Live detection from a webcam or an RTSP stream, shown in an OpenCV window. Press q to quit.

    python scripts/webcam.py                      # default webcam
    python scripts/webcam.py --source rtsp://...  # network camera

Needs the GUI build of OpenCV (`pip install opencv-python`), not the headless one.
"""

import argparse

import cv2
import sparrow_engine as spe


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="0", help="device index or stream URL")
    parser.add_argument("--model", default="MDV6-yolov10-c", help="detector model id")
    parser.add_argument("--classifier", help="optional species classifier model id")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--every-n", type=int, default=1, help="analyse every n-th frame")
    args = parser.parse_args()

    source = int(args.source) if args.source.isdigit() else args.source
    spe.download_models([m for m in (args.model, args.classifier) if m])
    try:
        for fr in spe.detect_webcam(
            source,
            args.model,
            classifier=args.classifier,
            threshold=args.threshold,
            every_n=args.every_n,
        ):
            cv2.imshow("SPARROW Playground", spe.draw_detections(fr.frame, fr.result))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
