---
description: "PyTorch-Wildlife: unified open-source AI framework from Microsoft AI for Good Lab for camera-trap detection, species classification, and wildlife monitoring."
tags:
  - PyTorch-Wildlife
  - MegaDetector
  - wildlife AI
  - camera trap detection
  - species classification
  - conservation AI
  - Microsoft AI for Good
---

# PyTorch-Wildlife

> [!TIP]
> PyTorch-Wildlife is part of the [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) umbrella — the hub for all AI for Good Lab wildlife tools. MegaDetector lives at [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector).

**PyTorch-Wildlife is the unified open-source AI framework from the [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/ai/ai-for-good) for wildlife monitoring.** It hosts detection models, species classifiers, and the tools needed to run them — from single-image inference to large-scale batch processing across camera-trap datasets.

Our mission is to create a global community where conservation scientists can collaborate — sharing datasets and deep learning architectures for wildlife conservation. PyTorch-Wildlife provides the shared foundation that every project in our ecosystem builds on.


## Quick Start

```bash
pip install PytorchWildlife
```

```python
import numpy as np
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

# Detection — MegaDetector V6, weights download automatically
detection_model = pw_detection.MegaDetectorV6()
detection_result = detection_model.single_image_detection("path/to/image.jpg")

# Classification
classification_model = pw_classification.AI4GAmazonRainforest()
classification_result = classification_model.single_image_classification("path/to/image.jpg")
```

**Try without installing:**
- [Hugging Face demo](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife) — upload images in your browser
- [Google Colab notebook](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing) — free cloud GPU


## What's Inside

PyTorch-Wildlife provides a modular set of building blocks:

- **Detection models** — MegaDetector V5/V6 (multiple architectures), Deepfaune detector, HerdNet for aerial imagery
- **Classification models** — Amazon Rainforest, Snapshot Serengeti, Opossum, Deepfaune, DFNE (New England)
- **Bioacoustic models** — audio-based wildlife identification
- **Data utilities** — transforms, datasets, batch processing, video support
- **Demo notebooks** — Jupyter notebooks and Gradio web UI for hands-on exploration

See the [Model Zoo](model_zoo.md) for the full list with performance benchmarks.


## Part of the Biodiversity Ecosystem

PyTorch-Wildlife is one project in a larger open-source ecosystem from the AI for Good Lab:

| Repo | Purpose |
|---|---|
| [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) | The umbrella repository — documentation hub for the AI for Good Lab's biodiversity work |
| [microsoft/Pytorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife) | This repo — the unified deep learning framework |
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | Animal detection in camera-trap imagery |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI-enabled edge device |
| [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) | Bioacoustic models for audio-based wildlife monitoring |
| [microsoft/MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) | Camera-trap species classification fine-tuning — adapt classifiers to your own datasets and geographic regions |
| [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) | Point-based detection for overhead and aerial imagery |
| [SPARROW Studio](https://github.com/microsoft/Biodiversity/tree/main/SPARROW-Studio) | Desktop application for running all models with a graphical interface |

> [!TIP]
> If you have any questions, please [email us](mailto:zhongqimiao@microsoft.com) or join us on Discord: [![](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=PyTorch-Wildlife)](https://discord.gg/TeEVxzaYtm)
