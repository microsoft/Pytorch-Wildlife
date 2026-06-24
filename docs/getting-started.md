---
title: "Getting Started with PyTorch-Wildlife: Detection and Classification"
description: "Getting started with PyTorch-Wildlife: install the framework, run your first MegaDetector detection, add a species classifier, and scale up to batch."
slug: getting-started
tags:
  - getting started
  - PyTorch-Wildlife
  - wildlife AI framework
  - quickstart
  - detection and classification
---

# Getting Started with PyTorch-Wildlife

New to PyTorch-Wildlife? This page is the short path from a fresh install to real output. It covers four steps: install the framework, run your first detection, add a species classifier on top, and scale up to whole folders. Each step links to the deeper reference page when you want more detail.

If you only need the minimal snippet, the [Overview](index.md#quick-start) has it. If you want the guided version, read on.

## Is PyTorch-Wildlife right for my project?

PyTorch-Wildlife is the unified open-source framework from the Microsoft AI for Good Lab for wildlife monitoring. It bundles detection models (including MegaDetector), species classifiers, bioacoustic models, and the data utilities that tie them together, so you can run a full detect-then-classify pipeline from one package rather than stitching tools together yourself.

Reach for it when you want to:

- Run MegaDetector and a species classifier through one consistent Python API.
- Process camera-trap images one at a time or thousands in a batch.
- Build on a shared foundation instead of maintaining your own inference harness.

If you only need the detector and nothing else, the dedicated [MegaDetector](https://microsoft.github.io/MegaDetector/) site is a narrower starting point. For field hardware, see [SPARROW](https://microsoft.github.io/SPARROW/); for audio, see MegaDetector-Acoustic.

## Step 1: Install the framework

Install from PyPI:

```bash
pip install PytorchWildlife
```

A clean conda environment keeps dependencies tidy:

```bash
conda create -n pytorch-wildlife python=3.10 -y
conda activate pytorch-wildlife
pip install PytorchWildlife
```

A CUDA-capable NVIDIA GPU gives a large speedup but is optional. GPU-enabled PyTorch, a Docker image, and platform notes for Windows, macOS, and Ubuntu are on the [Installation](installation.md) page. To skip installation entirely while you evaluate, try the [Hugging Face demo](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife) in a browser or the [Google Colab notebook](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing) on a free cloud GPU.

## Step 2: Run your first detection

Load MegaDetectorV6 and detect one image. Weights download on first use:

```python
from PytorchWildlife.models import detection as pw_detection

detector = pw_detection.MegaDetectorV6()
results = detector.single_image_detection("path/to/image.jpg")
```

That is the smallest useful run. For annotated images, saved crops, and JSON exports (including a Timelapse-compatible file), the [Inference Examples](inference-examples.md) page has runnable scripts.

## Step 3: Add species classification

MegaDetector finds animals but does not name the species. To get species, pass each detection crop to a classifier. The two stages share the same API, so chaining them is direct:

```python
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

detector = pw_detection.MegaDetectorV6()
classifier = pw_classification.AI4GAmazonRainforest()

detection_result = detector.single_image_detection("image.jpg")
classification_result = classifier.single_image_classification("image.jpg")
```

The framework ships several classifiers (Amazon Rainforest, Snapshot Serengeti, Opossum, Deepfaune, DFNE). The [API overview](api.md#detect-then-classify) shows the full crop-and-classify loop, and the [Model Zoo](model_zoo.md#detection-plus-classification-pipeline) lists every model with benchmarks.

## Step 4: Scale up and explore

Real surveys run many images at once. `batch_image_detection` takes a folder and a batch size, and the JSON exporters write formats downstream tools can read:

```python
results = detector.batch_image_detection("path/to/image_folder/", batch_size=16)
```

From here you can:

- Run on video frames or launch the Gradio web UI, both covered in [Inference Examples](inference-examples.md).
- Browse the full set of detectors, classifiers, and bioacoustic models in the [Model Zoo](model_zoo.md).
- Fine-tune a model on your own data; the fine-tuning modules adapt detection and classification to your datasets.

## Get help

- **Email:** [zhongqimiao@microsoft.com](mailto:zhongqimiao@microsoft.com).
- **Discord:** [the PyTorch-Wildlife community server](https://discord.gg/TeEVxzaYtm).

PyTorch-Wildlife is part of the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) umbrella; the hub links every tool in the ecosystem.
