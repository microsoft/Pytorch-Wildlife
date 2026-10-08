"""Tests for `PytorchWildlife.utils.post_process`.

The detection results come from the `results` fixture in `conftest.py`:
`animal.jpg` has an animal box [10, 10, 60, 40] (confidence 0.9) and a person box
[100, 20, 150, 90] (confidence 0.5) in a 200 x 100 image; `empty.jpg` has none.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from PytorchWildlife.models.detection import MegaDetectorV6
from PytorchWildlife.utils import post_process

ANIMAL = 0
PERSON = 1


def save_detection_json(results: list[dict], path: Path, **options) -> dict:
    """Save `results` with `save_detection_json` and return the loaded file."""
    post_process.save_detection_json(
        results, str(path), categories=MegaDetectorV6.CLASS_NAMES, **options
    )
    return json.loads(path.read_text())


def sort_into_folders(
    results: list[dict], image_folder: Path, tmp_path: Path, threshold: float
) -> Path:
    """Save `results` with relative paths, sort the images, and return the destination."""
    json_file = tmp_path / "detections.json"
    save_detection_json(results, json_file, exclude_file_path=str(image_folder))
    destination = tmp_path / "sorted"
    post_process.detection_folder_separation(
        str(json_file), str(image_folder), str(destination), threshold
    )
    return destination


# ---------------------------------------------------------------------------
# Detection JSON
# ---------------------------------------------------------------------------


def test_detection_json_contains_every_detection(results, tmp_path) -> None:
    """Every image and detection is saved with its box, category and confidence."""
    saved = save_detection_json(results, tmp_path / "detections.json")

    animal, empty = saved["annotations"]
    assert animal["img_id"] == results[0]["img_id"]
    assert animal["bbox"] == [[10, 10, 60, 40], [100, 20, 150, 90]]
    assert animal["category"] == [ANIMAL, PERSON]
    assert animal["confidence"] == pytest.approx([0.9, 0.5])
    assert empty["bbox"] == empty["category"] == empty["confidence"] == []


def test_detection_json_excludes_categories(results, tmp_path) -> None:
    """`exclude_category_ids` removes those detections, for example people."""
    saved = save_detection_json(
        results, tmp_path / "detections.json", exclude_category_ids=[PERSON]
    )
    animal = saved["annotations"][0]
    assert animal["category"] == [ANIMAL]
    assert animal["bbox"] == [[10, 10, 60, 40]]


def test_detection_json_makes_paths_relative(results, image_folder, tmp_path) -> None:
    """`exclude_file_path` removes the folder from the saved image paths."""
    saved = save_detection_json(
        results, tmp_path / "detections.json", exclude_file_path=str(image_folder)
    )
    names = [annotation["img_id"] for annotation in saved["annotations"]]
    assert names == ["animal.jpg", "empty.jpg"]


# ---------------------------------------------------------------------------
# Timelapse JSON
# ---------------------------------------------------------------------------


def test_timelapse_json_detections(results, tmp_path) -> None:
    """Timelapse boxes are normalized [x, y, width, height], categories are strings,
    and `max_detection_conf` is the highest confidence in the image."""
    path = tmp_path / "timelapse.json"
    post_process.save_detection_timelapse_json(
        results, str(path), categories=MegaDetectorV6.CLASS_NAMES
    )
    image = json.loads(path.read_text())["images"][0]

    animal, person = image["detections"]
    assert animal["bbox"] == pytest.approx([0.05, 0.1, 0.25, 0.3])
    assert person["bbox"] == pytest.approx([0.5, 0.2, 0.25, 0.7])
    assert [animal["category"], person["category"]] == ["0", "1"]
    assert image["max_detection_conf"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# Sorting images into folders
# ---------------------------------------------------------------------------


def test_folder_separation_sorts_images(results, image_folder, tmp_path) -> None:
    """Images with an animal above the threshold go to `Animal/`, others to
    `No_animal/`. The original images are copied, not moved."""
    destination = sort_into_folders(results, image_folder, tmp_path, threshold=0.2)

    assert (destination / "Animal" / "animal.jpg").exists()
    assert (destination / "No_animal" / "empty.jpg").exists()
    assert (image_folder / "animal.jpg").exists()
    assert (image_folder / "empty.jpg").exists()


def test_folder_separation_respects_the_threshold(
    results, image_folder, tmp_path
) -> None:
    """An animal below the confidence threshold does not count as an animal."""
    destination = sort_into_folders(results, image_folder, tmp_path, threshold=0.95)

    assert (destination / "No_animal" / "animal.jpg").exists()
    assert not (destination / "Animal" / "animal.jpg").exists()


# ---------------------------------------------------------------------------
# Crops and annotated images
# ---------------------------------------------------------------------------


def test_crop_images_one_file_per_detection(results, tmp_path) -> None:
    """One crop per detection, named `{class}_{index}_{image}`, sized like its box."""
    output = tmp_path / "crops"
    post_process.save_crop_images(results, str(output))

    crops = {path.name: Image.open(path).size for path in output.iterdir()}
    assert crops == {"0_0_animal.jpg": (50, 30), "1_1_animal.jpg": (50, 70)}


def in_camera_folder(result: dict, tmp_path: Path) -> tuple[dict, Path]:
    """Copy the result's image into `images/camera_1/` and point the result at it.

    Returns the updated result and the `images` folder to pass as `input_dir`.
    """
    images = tmp_path / "images_by_camera"
    camera = images / "camera_1"
    camera.mkdir(parents=True)
    image_path = camera / Path(result["img_id"]).name
    image_path.write_bytes(Path(result["img_id"]).read_bytes())
    return {**result, "img_id": str(image_path)}, images


def test_crop_images_keep_the_subfolder_structure(results, tmp_path) -> None:
    """With `input_dir`, crops are saved in the same subfolders as their images,
    for example one folder per camera."""
    result, images = in_camera_folder(results[0], tmp_path)
    output = tmp_path / "crops"
    post_process.save_crop_images([result], str(output), input_dir=str(images))

    names = sorted(path.name for path in (output / "camera_1").iterdir())
    assert names == ["0_0_animal.jpg", "1_1_animal.jpg"]


def test_annotated_images_keep_the_subfolder_structure(results, tmp_path) -> None:
    """With `input_dir`, annotated images are saved in the same subfolders as their images."""
    result, images = in_camera_folder(results[0], tmp_path)
    output = tmp_path / "annotated"
    post_process.save_detection_images([result], str(output), input_dir=str(images))

    assert (output / "camera_1" / "animal.jpg").exists()


def test_annotated_images_draw_the_detections(results, image_folder, tmp_path) -> None:
    """One annotated image per input image, at the original size. Boxes are drawn on
    images with detections; images without detections are left as they were."""
    output = tmp_path / "annotated"
    post_process.save_detection_images(results, str(output))

    assert sorted(path.name for path in output.iterdir()) == ["animal.jpg", "empty.jpg"]
    for name in ["animal.jpg", "empty.jpg"]:
        original = np.asarray(Image.open(image_folder / name), dtype=int)
        annotated = np.asarray(Image.open(output / name), dtype=int)
        assert annotated.shape == original.shape
        # Saving as JPEG changes pixels slightly, so only clear differences count.
        changed = (np.abs(annotated - original) > 40).any()
        assert changed == (name == "animal.jpg")
