"""Live detection from a webcam or an RTSP stream, shown in an OpenCV window. Press q to quit.

    python scripts/webcam.py                      # default webcam
    python scripts/webcam.py --source rtsp://...  # network camera

Needs the GUI build of OpenCV (`pip install opencv-python`), not the headless one.
"""

import argparse
import sys

import cv2
import sparrow_engine as spe

WINDOW = "SPARROW Playground"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", default="0", help="device index or stream URL")
    parser.add_argument("--model", default="MDV6-yolov10-c", help="detector model id")
    parser.add_argument("--classifier", help="optional species classifier model id")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--every-n", type=int, default=1, help="analyse every n-th frame")
    args = parser.parse_args()

    try:
        cv2.namedWindow(WINDOW)
    except cv2.error:
        sys.exit(
            "This OpenCV build has no window support (opencv-python-headless).\n"
            "Install the GUI build: pip uninstall -y opencv-python-headless && "
            "pip install opencv-python\n"
            "In Jupyter, use notebooks/09_webcam.ipynb instead."
        )

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
            cv2.imshow(WINDOW, spe.draw_detections(fr.frame, fr.result))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
