---
title: "Wildlife Model Zoo: MegaDetector and Species Classifiers"
description: "The PyTorch-Wildlife model zoo: MegaDetector V5 and V6 detectors, region-specific species classifiers, HerdNet, and bioacoustic models. Each loads in one line."
tags:
  - wildlife model zoo
  - PyTorch-Wildlife
  - MegaDetector versions
  - species classification models
  - conservation deep learning framework
---

# Wildlife Model Zoo

The PyTorch-Wildlife model zoo is the catalog of detection, classification, and bioacoustic models that ship with the framework. Every entry loads with a single constructor call and downloads its own weights on first use, so you can compare architectures or swap a model without touching the rest of your pipeline.

If you are new to the package, install it first with the [installation guide](installation.md), then come back here to pick a model. The [API overview](api.md) shows how to feed images into any model below.

## Detection Models

### MegaDetector V6

The current generation of [MegaDetector](https://microsoft.github.io/MegaDetector/), trained on diverse global camera-trap datasets. Several architecture variants let you trade accuracy against speed and licensing. The framework provides three wrapper classes so you can pick the license your project needs.

| Version | Architecture | License | Load with |
|---|---|---|---|
| `MDV6-yolov10-c` | YOLOv10 Compact | AGPL | `MegaDetectorV6(version="MDV6-yolov10-c")` |
| `MDV6-yolov10-e` | YOLOv10 Extra | AGPL | `MegaDetectorV6(version="MDV6-yolov10-e")` |
| `MDV6-yolov9-c` | YOLOv9 Compact | AGPL | `MegaDetectorV6(version="MDV6-yolov9-c")` |
| `MDV6-yolov9-e` | YOLOv9 Extra | AGPL | `MegaDetectorV6(version="MDV6-yolov9-e")` |
| `MDV6-mit-yolov9-c` | YOLOv9 Compact | MIT | `MegaDetectorV6MIT(version="MDV6-mit-yolov9-c")` |
| `MDV6-mit-yolov9-e` | YOLOv9 Extra | MIT | `MegaDetectorV6MIT(version="MDV6-mit-yolov9-e")` |
| `MDV6-apa-rtdetr-c` | RT-DETR Compact | Apache 2.0 | `MegaDetectorV6Apache(version="MDV6-apa-rtdetr-c")` |
| `MDV6-apa-rtdetr-e` | RT-DETR Extra | Apache 2.0 | `MegaDetectorV6Apache(version="MDV6-apa-rtdetr-e")` |

```python
from PytorchWildlife.models import detection as pw_detection

# Default AGPL build (YOLOv10 Extra is the recommended starting point)
detector = pw_detection.MegaDetectorV6(version="MDV6-yolov10-e")

# MIT-licensed YOLOv9 weights
detector = pw_detection.MegaDetectorV6MIT(version="MDV6-mit-yolov9-e")

# Apache-2.0 RT-DETR weights
detector = pw_detection.MegaDetectorV6Apache(version="MDV6-apa-rtdetr-e")
```

### MegaDetector V5

The previous generation, still widely deployed across conservation organizations. Built on YOLOv5.

```python
detector = pw_detection.MegaDetectorV5()
```

For V5 weights and earlier releases, see the [archive branch](https://github.com/microsoft/Biodiversity/tree/archive) of the Biodiversity repository.

### Deepfaune Detector

A detector tuned for European ecosystems, and the first third-party camera-trap model integrated into the framework.

```python
detector = pw_detection.DeepfauneDetector()
```

See the [Deepfaune project](https://www.deepfaune.cnrs.fr/en/) for background on the underlying model.

### HerdNet

A point-based localization model for overhead and aerial imagery, where animals appear as small dots rather than large bounding boxes.

```python
detector = pw_detection.HerdNet()
```

## Classification Models

Classifiers turn a detected animal crop into a species label. Pair any classifier with any detector to build a two-stage detect-then-classify pipeline.

| Model | Class | Geography | Coverage |
|---|---|---|---|
| AI4G Amazon Rainforest | `AI4GAmazonRainforest` | Amazon | ~36 species |
| AI4G Snapshot Serengeti | `AI4GSnapshotSerengeti` | African savanna | ~48 species |
| AI4G Opossum | `AI4GOpossum` | Americas | Opossum vs. non-opossum |
| Deepfaune | `DeepfauneClassifier` | Europe | ~44 species |
| DFNE | `DFNE` | Northeastern North America | Fine-tuned Deepfaune |

```python
from PytorchWildlife.models import classification as pw_classification

classifier = pw_classification.AI4GAmazonRainforest()
classifier = pw_classification.AI4GSnapshotSerengeti()
classifier = pw_classification.DeepfauneClassifier()
classifier = pw_classification.DFNE()
```

> [!TIP]
> Need a classifier for a region or species set that is not listed here? The [MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) project covers fine-tuning a classifier on your own labeled data.

## Detection plus Classification Pipeline

Run a detector to find animals, then send each crop to a classifier for a species label:

```python
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

detection_model = pw_detection.MegaDetectorV6(version="MDV6-yolov10-e")
classification_model = pw_classification.AI4GAmazonRainforest()

detection_result = detection_model.single_image_detection("image.jpg")
classification_result = classification_model.single_image_classification("image.jpg")
```

The `demo/detection_classification_pipeline_demo.py` script shows the full two-stage flow end to end. More runnable walkthroughs live on the [inference examples](inference-examples.md) page.

## Bioacoustic Models

The framework also includes a ResNet-based classifier for audio, so sound-based monitoring shares the same package as vision.

```python
from PytorchWildlife.models import bioacoustics as pw_bioacoustics

model = pw_bioacoustics.ResNetClassifier(num_classes=2)
```

For the dedicated bioacoustic model zoo and audio pipelines, the framework provides support for, but does not own, that modality. See MegaDetector-Acoustic (documentation coming soon).

## Related Microsoft biodiversity AI projects

- [microsoft/Biodiversity](https://microsoft.github.io/Biodiversity/): the umbrella hub for every AI for Good Lab biodiversity tool.
- [MegaDetector](https://microsoft.github.io/MegaDetector/): the camera-trap detection model whose weights this zoo serves.
- MegaDetector-Acoustic (documentation coming soon): the bioacoustic model family for audio monitoring.
- [SPARROW](https://microsoft.github.io/SPARROW/): the solar-powered edge device that runs these models in the field.
