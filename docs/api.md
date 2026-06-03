---
title: "API Overview: PyTorch-Wildlife Framework"
description: "How to use the PyTorch-Wildlife API: load detection and classification models, run single-image and batch inference, and export results, with runnable code."
tags:
  - PyTorch-Wildlife API
  - wildlife AI framework
  - conservation deep learning framework
  - MegaDetector API
  - batch inference
---

# API Overview

PyTorch-Wildlife exposes a small, predictable surface. Once you know the shape of one model, you know them all: detectors and classifiers share the same single-image and batch entry points, and a common set of utilities turns their output into annotated images, crops, and JSON. This page is a guided tour of that API with code you can run, rather than a raw symbol dump. For the full catalog of loadable models, see the [Wildlife Model Zoo](model_zoo.md).

## Package layout

The framework groups everything under a few namespaces:

```python
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife.models import bioacoustics as pw_bioacoustics
from PytorchWildlife import utils as pw_utils
```

- `models.detection`: bounding-box and point detectors (MegaDetector, Deepfaune, HerdNet).
- `models.classification`: species classifiers for crops or whole images.
- `models.bioacoustics`: audio classifiers.
- `utils`: output helpers for saving images, crops, JSON, and processing video.

## Choosing a device

Every model constructor takes a `device` argument. Use a CUDA GPU when one is available; otherwise the model runs on CPU:

```python
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
```

If `torch.cuda.is_available()` returns `False` on a machine with an NVIDIA GPU, the [installation guide](installation.md#gpu-setup) explains how to install a CUDA-enabled PyTorch build.

## Detection

Detectors share two methods. `single_image_detection` runs one image; `batch_image_detection` runs a folder or dataloader. Both accept a `conf_thres` confidence threshold (default `0.2`).

```python
from PytorchWildlife.models import detection as pw_detection

detector = pw_detection.MegaDetectorV6(device=DEVICE, version="MDV6-yolov10-e")

# One image
result = detector.single_image_detection("path/to/image.jpg", conf_thres=0.2)

# A whole folder, batched
results = detector.batch_image_detection("path/to/folder", batch_size=16)
```

The returned dictionary carries the detections (boxes, confidences, and class IDs) alongside the image identifier. Class IDs map through `detector.CLASS_NAMES`.

## Classification

Classifiers mirror the detection interface with `single_image_classification` and `batch_image_classification`. They are most often run on the crops a detector produces, which is the standard two-stage pattern in camera-trap analysis.

```python
from PytorchWildlife.models import classification as pw_classification

classifier = pw_classification.AI4GAmazonRainforest(device=DEVICE)
prediction = classifier.single_image_classification("path/to/crop.jpg")
# prediction["prediction"] holds the species label; prediction["confidence"] the score
```

## Detect, then classify

Because both stages share a consistent API, chaining them is straightforward: detect animals, crop each box, and classify the crop.

```python
import supervision as sv
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

detector = pw_detection.MegaDetectorV6(device=DEVICE, version="MDV6-yolov10-e")
classifier = pw_classification.AI4GOpossum(device=DEVICE)

image = "path/to/image.jpg"
det = detector.single_image_detection(image)

import numpy as np
from PIL import Image
frame = np.array(Image.open(image).convert("RGB"))

for box in det["detections"].xyxy:
    crop = sv.crop_image(image=frame, xyxy=box)
    label = classifier.single_image_classification(crop)
    print(label["prediction"], label["confidence"])
```

## Saving results

The `utils` module turns raw detections into the artifacts conservation workflows expect. These are the functions the demo scripts use:

```python
from PytorchWildlife import utils as pw_utils

# Annotated images with boxes drawn on
pw_utils.save_detection_images(results, "annotated_output", overwrite=False)

# Cropped detections, one image per animal
pw_utils.save_crop_images(results, "crop_output", overwrite=False)

# Plain JSON
pw_utils.save_detection_json(results, "results.json",
                             categories=detector.CLASS_NAMES)

# Timelapse-compatible JSON for ecologists' existing tooling
pw_utils.save_detection_timelapse_json(results, "results_timelapse.json",
                                       categories=detector.CLASS_NAMES,
                                       info={"detector": "MegaDetectorV6"})
```

For point-based detectors such as HerdNet, the dot-style variants `save_detection_images_dots` and `save_detection_json_as_dots` render and export results as points instead of boxes.

## Video

The `process_video` helper runs any per-frame callback across a video and writes an annotated copy, with a progress bar and selectable codec:

```python
from PytorchWildlife import utils as pw_utils

pw_utils.process_video(
    source_path="input.mp4",
    target_path="output.mp4",
    callback=my_frame_callback,   # takes (frame, index), returns an annotated frame
    target_fps=1,
)
```

A complete video pipeline that detects and classifies every frame lives in `demo/video_demo.py`. See [inference examples](inference-examples.md) for the full walkthrough.

## Bioacoustics

Audio classification uses the same package, exposed through the `bioacoustics` namespace. The `ResNetClassifier` supports both binary and multiclass setups:

```python
from PytorchWildlife.models import bioacoustics as pw_bioacoustics

model = pw_bioacoustics.ResNetClassifier(num_classes=2)
```

The framework provides the runtime here; the trained audio models and end-to-end audio pipelines are documented at MegaDetector-Acoustic (documentation coming soon).

## Where to go next

- Browse every loadable model in the [Wildlife Model Zoo](model_zoo.md).
- Follow runnable end-to-end scripts on the [inference examples](inference-examples.md) page.
- Set up your environment with the [installation guide](installation.md).
