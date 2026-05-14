---
description: "Install PyTorch-Wildlife for camera-trap AI and wildlife detection. Supports pip, conda, and Docker on Windows, macOS, and Linux with optional CUDA GPU acceleration."
tags:
  - PyTorch-Wildlife installation
  - pip install PytorchWildlife
  - conda environment
  - wildlife AI setup
  - MegaDetector install
  - GPU CUDA setup
---

# Installation

## Prerequisites

- Python 3.8+ (3.10+ recommended)
- Optional: NVIDIA GPU with CUDA 12.1 for 10–50x speedup

## pip

```bash
pip install PytorchWildlife
```

## Conda

```bash
conda create -n pytorch-wildlife python=3.10 -y
conda activate pytorch-wildlife
pip install PytorchWildlife
```

> **Windows users:** Use the Anaconda Prompt if using Anaconda, otherwise use PowerShell.


## GPU Setup

### Check if CUDA is available

```python
import torch
print(torch.cuda.is_available())
```

### Install GPU-enabled PyTorch (CUDA 12.1)

If `torch.cuda.is_available()` returns `False` on a CUDA machine:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install PytorchWildlife
```

### Ubuntu — OpenCV dependency

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
```

### macOS — ffmpeg for video decoding

```bash
brew install ffmpeg
```

### Windows

See the [Windows installation guide](https://zenodo.org/records/15376499/files/PytorchWildlife_Windows_installation_tutorial.pdf) for a step-by-step walkthrough.


## Verify Installation

```python
from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6()
print("PyTorch-Wildlife loaded successfully.")
```

Model weights download automatically on first use.


## Try Without Installing

- [Hugging Face demo](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife) — upload images in your browser
- [Google Colab notebook](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing) — free cloud GPU


## Docker

```bash
docker pull andreshdz/pytorchwildlife:1.0.2.3
docker run -p 80:80 andreshdz/pytorchwildlife:1.0.2.3 python demo/gradio_demo.py
```


## Jupyter Notebooks

To use the demo notebooks with Jupyter:

```bash
conda install ipykernel
python -m ipykernel install --user --name pytorch-wildlife --display-name "Python (PytorchWildlife)"
```

Then select the `Python (PytorchWildlife)` kernel when running notebooks.
