---
description: "PyTorch-Wildlife model zoo: MegaDetector V5/V6 detection models, species classifiers (Amazon, Serengeti, Deepfaune), and bioacoustic models for wildlife monitoring."
tags:
  - PyTorch-Wildlife model zoo
  - MegaDetector versions
  - species classification models
  - wildlife AI models
  - camera trap models
---

# Model Zoo

PyTorch-Wildlife provides a growing library of detection, classification, and bioacoustic models. All models load with a single line and download weights automatically.


## Detection Models

### MegaDetector V6

The latest generation of MegaDetector, trained on diverse global camera-trap datasets. Multiple architecture variants are available to trade off accuracy vs. speed vs. licensing.

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

# Default (AGPL, YOLOv10)
detector = pw_detection.MegaDetectorV6()

# MIT-licensed YOLO
detector = pw_detection.MegaDetectorV6MIT(version="MDV6-mit-yolov9-e")

# Apache RT-DETR
detector = pw_detection.MegaDetectorV6Apache(version="MDV6-apa-rtdetr-e")
```

### MegaDetector V5

The previous generation, widely deployed across conservation organizations. Uses YOLOv5.

```python
detector = pw_detection.MegaDetectorV5()
```

For V5 model weights and earlier versions, see the [archive branch](https://github.com/microsoft/Biodiversity/tree/archive) of the Biodiversity repository.

### Deepfaune Detector

Trained for European ecosystems. The first third-party camera-trap detection model integrated into PyTorch-Wildlife.

```python
detector = pw_detection.DeepfauneDetector()
```

See the [Deepfaune website](https://www.deepfaune.cnrs.fr/en/) for more details.

### HerdNet

Point-based localization model for overhead and aerial imagery.

```python
detector = pw_detection.HerdNet()
```


## Classification Models

All classifiers can be paired with any detection model to build a detection + classification pipeline.

| Model | Class | Geography | Species |
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


## Detection + Classification Pipeline

```python
from PytorchWildlife.models import detection as pw_detection
from PytorchWildlife.models import classification as pw_classification

detection_model = pw_detection.MegaDetectorV6()
classification_model = pw_classification.AI4GAmazonRainforest()

# Detect, then classify crops
detection_result = detection_model.single_image_detection("image.jpg")
classification_result = classification_model.single_image_classification("image.jpg")
```

For a full pipeline demo, see the `demo/detection_classification_pipeline_demo.py` script.


## Bioacoustic Models

```python
from PytorchWildlife.models import bioacoustics as pw_bioacoustics

model = pw_bioacoustics.BioacousticsResnetClassifier()
```

For the full bioacoustic model zoo, see [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic).
