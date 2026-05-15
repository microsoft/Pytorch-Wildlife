![A colorful banner illustrating various species of animals and plants in a natural environment, symbolizing biodiversity and the use of AI for conservation purposes.](https://zenodo.org/records/20044680/files/Biodiversity_Banner.png)

# PyTorch-Wildlife

**Unified open-source AI framework for wildlife monitoring and conservation.**  
Microsoft AI for Good Lab — camera-trap detection, species classification, bioacoustic analysis, and more.

<div align="center">
<br>
<a href="https://github.com/microsoft/Pytorch-Wildlife/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue" /></a>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/v/PytorchWildlife?color=limegreen" /></a>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://static.pepy.tech/badge/pytorchwildlife" /></a>
<a href="https://pypi.org/project/PytorchWildlife"><img src="https://img.shields.io/pypi/pyversions/PytorchWildlife" /></a>
<a href="https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue" /></a>
<a href="https://discord.gg/TeEVxzaYtm"><img src="https://img.shields.io/badge/Discord-Join_us-5865F2?logo=discord&logoColor=white" /></a>
<a href="https://microsoft.github.io/Pytorch-Wildlife/"><img src="https://img.shields.io/badge/Docs-526CFE?logo=MaterialForMkDocs&logoColor=white" /></a>
<br><br>
</div>

PyTorch-Wildlife is the collaborative deep learning framework that powers the [Microsoft AI for Good Lab](https://www.microsoft.com/en-us/ai/ai-for-good)'s biodiversity work. It hosts detection models, species classifiers, and the tools needed to run them — from single-image inference to large-scale batch processing.

**MegaDetector**, the most widely used camera-trap detection model in conservation, is invoked through PyTorch-Wildlife. So are the species classifiers for Amazon Rainforest, Snapshot Serengeti, and European ecosystems.


## Quick Start

```bash
pip install PytorchWildlife
```

```python
import numpy as np
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

# Detection — weights download automatically
detection_model = pw_detection.MegaDetectorV6()
detection_result = detection_model.single_image_detection("path/to/image.jpg")

# Classification
classification_model = pw_classification.AI4GAmazonRainforest()
classification_result = classification_model.single_image_classification("path/to/image.jpg")
```

**Try without installing anything:**
- [Hugging Face demo](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife) — upload images in your browser
- [Google Colab notebook](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing) — free cloud GPU


## Available Models

### Detection
| Model | Architecture | Description |
|---|---|---|
| `MegaDetectorV6` | YOLOv10 / YOLOv9 / RT-DETR | Animal detection in camera-trap images |
| `MegaDetectorV5` | YOLOv5 | Previous generation, widely deployed |
| `DeepfauneDetector` | YOLOv8 | European ecosystem detection |
| `HerdNet` | CNN localization | Point-based detection for aerial imagery |

### Classification
| Model | Description |
|---|---|
| `AI4GAmazonRainforest` | Species classification for Amazon Rainforest |
| `AI4GSnapshotSerengeti` | Species classification for African savanna |
| `AI4GOpossum` | Opossum vs. non-opossum classifier |
| `DeepfauneClassifier` | European ecosystem species classifier |
| `DFNE` | Deepfaune fine-tuned for Northeastern North America |

See the [Model Zoo](https://microsoft.github.io/Pytorch-Wildlife/model_zoo/) for full details, performance benchmarks, and version history.


## Part of the Biodiversity Ecosystem

PyTorch-Wildlife is part of the larger open-source ecosystem from the Microsoft AI for Good Lab:

| Repo | Purpose |
|---|---|
| [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity) | The umbrella repository — documentation hub for the AI for Good Lab's biodiversity work |
| [microsoft/Pytorch-Wildlife](https://github.com/microsoft/Pytorch-Wildlife) | This repo — the unified deep learning framework |
| [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) | Animal detection in camera-trap imagery |
| [microsoft/SPARROW](https://github.com/microsoft/SPARROW) | Solar-Powered Acoustic and Remote Recording Observation Watch — AI-enabled edge device |
| [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) | Bioacoustic models for audio-based wildlife monitoring |
| [microsoft/MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) | Camera-trap species classification fine-tuning — adapt classifiers to your own datasets and geographic regions |
| [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) | Point-based detection for overhead and aerial imagery |
| SPARROW Studio | Desktop application for running all models with a graphical interface |

> Questions? [Email us](mailto:zhongqimiao@microsoft.com) or join the [![Discord](https://img.shields.io/badge/any_text-Join_us!-blue?logo=discord&label=Discord)](https://discord.gg/TeEVxzaYtm)
