"""Tests for the image file checks and transforms in `PytorchWildlife.data`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from PytorchWildlife.data import datasets, transforms


@pytest.mark.parametrize(
    "filename", ["photo.jpg", "PHOTO.JPG", "photo.jpeg", "photo.png", "photo.tif"]
)
def test_image_files_are_recognized(filename: str) -> None:
    """Common image extensions are recognized, regardless of case."""
    assert datasets.is_image_file(filename)


@pytest.mark.parametrize("filename", ["notes.txt", "photo.jpg.txt", "video.mp4"])
def test_other_files_are_not_images(filename: str) -> None:
    """Files that are not images are skipped, even if `.jpg` appears in the name."""
    assert not datasets.is_image_file(filename)


@pytest.mark.parametrize("height, width", [(1494, 2048), (2048, 1494), (100, 100)])
def test_letterbox_returns_a_square_of_the_target_size(height: int, width: int) -> None:
    """Landscape, portrait and small images all become `new_shape` x `new_shape`."""
    output = transforms.letterbox(torch.rand(3, height, width), new_shape=1280)
    assert tuple(output.shape) == (3, 1280, 1280)


def test_letterbox_keeps_the_aspect_ratio_and_pads_with_gray() -> None:
    """The image is scaled without distortion and the rest is filled with gray (114).

    A white 2048 x 1494 image scaled to 1280 wide is round(1494 * 1280 / 2048) = 934
    rows high; the remaining rows are padding.
    """
    output = transforms.letterbox(torch.ones(3, 1494, 2048), new_shape=1280)

    white_rows = (output[0] == 1.0).all(dim=1).sum().item()
    assert white_rows == pytest.approx(934, abs=1)
    assert output[:, 0, 0].tolist() == pytest.approx([114 / 255] * 3)


def test_megadetector_transform_output() -> None:
    """The detection transform returns a 3 x 1280 x 1280 tensor scaled to [0, 1]."""
    image = np.full((1494, 2048, 3), 255, dtype=np.uint8)
    output = transforms.MegaDetector_v5_Transform(target_size=1280)(image)

    assert tuple(output.shape) == (3, 1280, 1280)
    assert output.min() >= 0
    assert output.max() <= 1


def test_classification_transform_output() -> None:
    """The classification transform returns a 3 x 224 x 224 tensor."""
    image = Image.new("RGB", (640, 480), "white")
    output = transforms.Classification_Inference_Transform(target_size=224)(image)
    assert tuple(output.shape) == (3, 224, 224)
