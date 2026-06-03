---
title: "PyTorch-Wildlife: Conservation Deep Learning Framework"
description: "PyTorch-Wildlife is the open-source conservation deep learning framework and wildlife model zoo from the Microsoft AI for Good Lab. Runs MegaDetector fast."
tags:
  - PyTorch-Wildlife
  - wildlife AI framework
  - conservation deep learning framework
  - wildlife model zoo
  - pytorchwildlife pip install
  - MegaDetector
  - species classification
---

![PyTorch-Wildlife, the open-source conservation deep learning framework from the Microsoft AI for Good Lab](https://zenodo.org/records/15376499/files/Pytorch_Banner_transparentbk.png)

# PyTorch-Wildlife: A Wildlife AI Framework

> [!TIP]
> PyTorch-Wildlife is part of the [microsoft/Biodiversity](https://microsoft.github.io/Biodiversity/) umbrella, the hub for every AI for Good Lab wildlife tool. Looking for the camera-trap detection model on its own? See [MegaDetector](https://microsoft.github.io/MegaDetector/).

**PyTorch-Wildlife is the open-source conservation deep learning framework from the [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/ai/ai-for-good).** One Python package gives you a tested wildlife model zoo, a consistent load-and-run API, and the data utilities that turn a folder of camera-trap images into structured detections. You write a few lines; the framework handles weight downloads, batching, and output formatting.

The goal is a shared foundation that conservation scientists can build on together: common model interfaces, reusable training and inference code, and a place to publish new architectures so the whole community benefits. Every modality-focused project in our ecosystem plugs into this framework rather than reinventing it.

## Why a framework, not just a model

A single detection model solves one problem. Real conservation pipelines need detection, classification, batch processing, video support, and exportable results that downstream tools can read. PyTorch-Wildlife packages all of that behind one import:

- **A unified model zoo.** Detection and classification models load with one line and fetch their own weights. Swap `MegaDetectorV6` for `MegaDetectorV5` or a different classifier without rewriting your pipeline.
- **A consistent inference API.** Every detector exposes `single_image_detection` and `batch_image_detection`; every classifier exposes `single_image_classification` and `batch_image_classification`. Learn it once.
- **Conservation-ready outputs.** Built-in utilities save annotated images, cropped detections, and JSON, including a Timelapse-compatible format for ecologists' existing workflows.
- **Framework support across modalities.** Vision detection, species classification, and bioacoustic classifiers all share the same package, so a multi-modal pipeline is a few imports rather than a few dependencies.

## Quick Start

Install the framework from PyPI:

```bash
pip install PytorchWildlife
```

Run detection and classification in a handful of lines. Model weights download automatically on first use:

```python
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

# Detection with MegaDetector V6
detection_model = pw_detection.MegaDetectorV6()
detection_result = detection_model.single_image_detection("path/to/image.jpg")

# Species classification
classification_model = pw_classification.AI4GAmazonRainforest()
classification_result = classification_model.single_image_classification("path/to/image.jpg")
```

New to the package? The [installation guide](installation.md) covers GPU setup, Docker, and Windows, and the [API overview](api.md) walks through the detection and classification interfaces with runnable examples.

**Try it without installing anything:**

- [Hugging Face demo](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife): upload images in your browser
- [Google Colab notebook](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing): free cloud GPU

## What's Inside

PyTorch-Wildlife ships a modular set of building blocks:

- **Detection models.** MegaDetector V5 and V6 across several architectures, the Deepfaune detector, and HerdNet for aerial imagery. See the [Wildlife Model Zoo](model_zoo.md).
- **Classification models.** Region-specific species classifiers for the Amazon, the Serengeti, Europe, and more.
- **Bioacoustic models.** A ResNet-based audio classifier for sound-based monitoring.
- **Data and output utilities.** Transforms, datasets, batch dataloaders, video processing, and JSON exporters.
- **Demos.** Jupyter notebooks, runnable Python scripts, and a Gradio web UI. See [inference examples](inference-examples.md).

For the complete list with versions and load commands, head to the [Wildlife Model Zoo](model_zoo.md).

## Related Microsoft biodiversity AI projects

PyTorch-Wildlife is the framework layer. The modality-specific tools in the ecosystem each own their domain, and the framework provides support for running them:

- [microsoft/Biodiversity](https://microsoft.github.io/Biodiversity/): the umbrella hub documenting every AI for Good Lab biodiversity tool.
- [MegaDetector](https://microsoft.github.io/MegaDetector/): the camera-trap animal detection model, invoked through this framework.
- [MegaDetector-Acoustic](https://microsoft.github.io/MegaDetector-Acoustic/): bioacoustic models for audio-based wildlife monitoring.
- [SPARROW](https://microsoft.github.io/SPARROW/): the solar-powered edge device that runs these models in the field.

> [!TIP]
> Questions? [Email us](mailto:zhongqimiao@microsoft.com) or join us on Discord: [![Join the PyTorch-Wildlife Discord](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PyTorch-Wildlife)](https://discord.gg/TeEVxzaYtm)
