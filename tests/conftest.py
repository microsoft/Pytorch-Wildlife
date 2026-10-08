"""Shared pytest fixtures for the PytorchWildlife tests.

The fixtures provide small generated images and hand-built detection results in
the same format the detectors return, so the tests run without downloading
model weights.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import supervision as sv
from PIL import Image

IMAGE_WIDTH = 200
IMAGE_HEIGHT = 100
CLASS_NAMES = {0: "animal", 1: "person", 2: "vehicle"}


def make_result(
    img_path: Path,
    boxes: list[list[float]],
    confidences: list[float],
    class_ids: list[int],
) -> dict:
    """Build a detection result with the keys returned by `single_image_detection`."""
    if boxes:
        detections = sv.Detections(
            xyxy=np.array(boxes, dtype=float),
            confidence=np.array(confidences, dtype=float),
            class_id=np.array(class_ids, dtype=int),
        )
    else:
        detections = sv.Detections.empty()
    return {
        "img_id": str(img_path),
        "detections": detections,
        "labels": [
            f"{CLASS_NAMES[class_id]} {confidence:0.2f}"
            for class_id, confidence in zip(class_ids, confidences)
        ],
        "normalized_coords": [
            [x1 / IMAGE_WIDTH, y1 / IMAGE_HEIGHT, x2 / IMAGE_WIDTH, y2 / IMAGE_HEIGHT]
            for x1, y1, x2, y2 in boxes
        ],
    }


@pytest.fixture()
def image_folder(tmp_path: Path) -> Path:
    """A folder with two white 200 x 100 images: `animal.jpg` and `empty.jpg`."""
    folder = tmp_path / "images"
    folder.mkdir()
    for name in ["animal.jpg", "empty.jpg"]:
        Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), "white").save(folder / name)
    return folder


@pytest.fixture()
def results(image_folder: Path) -> list[dict]:
    """Detection results for `image_folder`.

    `animal.jpg` has an animal (confidence 0.9) and a person (confidence 0.5);
    `empty.jpg` has no detections.
    """
    return [
        make_result(
            image_folder / "animal.jpg",
            boxes=[[10, 10, 60, 40], [100, 20, 150, 90]],
            confidences=[0.9, 0.5],
            class_ids=[0, 1],
        ),
        make_result(image_folder / "empty.jpg", boxes=[], confidences=[], class_ids=[]),
    ]
