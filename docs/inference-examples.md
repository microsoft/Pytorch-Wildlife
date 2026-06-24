---
title: "Inference Examples: Run Wildlife Models at Scale"
description: "Runnable PyTorch-Wildlife inference examples: single image, batch folder, video, and the Gradio web UI. Export annotated images, crops, and Timelapse JSON."
tags:
  - PyTorch-Wildlife inference
  - batch image detection
  - wildlife model zoo
  - conservation deep learning framework
  - MegaDetector batch
---

# Inference Examples

PyTorch-Wildlife ships runnable demo scripts so you can go from a fresh install to real output without writing your own harness first. This page walks through the common ways to run inference: one image, a whole folder, a video, and an interactive web UI. Each example maps to a script under the repository's `demo/` directory.

Before you start, install the framework with the [installation guide](installation.md) and skim the [API overview](api.md) so the method names below feel familiar. New to the framework? The [Getting Started](getting-started.md) guide covers the path up to this point.

> [!TIP]
> The demo scripts are written in cell style (`#%%`), so they run top to bottom as a plain `python demo/<script>.py`, or block by block in an interactive window.

## Single image

The smallest useful run: load a detector, detect one image, and save annotated and cropped output.

```python
import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife import utils as pw_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
detector = pw_detection.MegaDetectorV6(device=DEVICE, version="MDV6-yolov10-e")

results = detector.single_image_detection("demo_data/imgs/10050028_0.JPG")

pw_utils.save_detection_images(results, "demo_output", overwrite=False)
pw_utils.save_crop_images(results, "crop_output", overwrite=False)
```

This mirrors `demo/image_demo.py`.

## Batch a folder of images

For real camera-trap surveys you run thousands of images at once. `batch_image_detection` takes a folder path and a `batch_size`, and the JSON exporters write results in formats downstream tools can read, including a Timelapse-compatible export.

```python
folder = "demo_data/imgs"
results = detector.batch_image_detection(folder, batch_size=16)

# Annotated images and crops
pw_utils.save_detection_images(results, "batch_output", folder, overwrite=False)
pw_utils.save_crop_images(results, "crop_output", folder, overwrite=False)

# Plain and Timelapse JSON
pw_utils.save_detection_json(results, "batch_output.json",
                             categories=detector.CLASS_NAMES)
pw_utils.save_detection_timelapse_json(results, "batch_output_timelapse.json",
                                       categories=detector.CLASS_NAMES,
                                       info={"detector": "MegaDetectorV6"})
```

This mirrors `demo/image_demo.py`.

## Detect and classify video

`demo/video_demo.py` runs a detector and a classifier on every sampled frame and writes an annotated video. The pattern is a per-frame callback handed to `process_video`:

```python
import numpy as np
import supervision as sv
import torch
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification
from PytorchWildlife import utils as pw_utils

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
detector = pw_detection.MegaDetectorV6(device=DEVICE, version="MDV6-yolov10-e")
classifier = pw_classification.AI4GOpossum(device=DEVICE)

box_annotator = sv.BoxAnnotator(thickness=4)

def callback(frame: np.ndarray, index: int) -> np.ndarray:
    det = detector.single_image_detection(frame, img_path=index)
    labels = []
    for box in det["detections"].xyxy:
        crop = sv.crop_image(image=frame, xyxy=box)
        clf = classifier.single_image_classification(crop)
        labels.append("{} {:.2f}".format(clf["prediction"], clf["confidence"]))
    return box_annotator.annotate(scene=frame, detections=det["detections"])

pw_utils.process_video(
    source_path="demo_data/videos/opossum_example.MP4",
    target_path="demo_data/videos/opossum_example_processed.MP4",
    callback=callback,
    target_fps=1,
)
```

## Interactive web UI

`demo/gradio_demo.py` launches a browser-based interface for uploading images and viewing detections, useful for demos and quick checks without writing code:

```bash
python demo/gradio_demo.py
```

The same interface is hosted online if you would rather not run anything locally:

- [Hugging Face demo](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife): upload images in your browser.
- [Google Colab notebook](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing): free cloud GPU.

## More demos

The `demo/` directory also includes notebooks and scripts for the detect-then-classify pipeline (`detection_classification_pipeline_demo.py`), HerdNet aerial detection (`image_demo_herdnet.py`), and image separation by detection result (`image_separation_demo.py`).

## Related Microsoft biodiversity AI projects

- [microsoft/Biodiversity](https://microsoft.github.io/Biodiversity/): the umbrella hub for AI for Good Lab biodiversity tools.
- [MegaDetector](https://microsoft.github.io/MegaDetector/): the camera-trap detection model behind these examples.
- [MegaDetector-Acoustic](https://microsoft.github.io/MegaDetector-Acoustic/): audio-based monitoring models.
- [SPARROW](https://microsoft.github.io/SPARROW/): the solar-powered edge device for running models in the field.
