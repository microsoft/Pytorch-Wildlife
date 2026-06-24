---
title: "Install PyTorch-Wildlife: pip, Conda, Docker"
description: "Install PyTorch-Wildlife (pytorchwildlife pip install) for camera-trap AI. Covers pip, Conda, and Docker on Windows, macOS, and Linux, plus CUDA GPU setup."
tags:
  - PyTorch-Wildlife installation
  - pytorchwildlife pip install
  - conda environment
  - wildlife AI framework
  - GPU CUDA setup
---

# Installation

PyTorch-Wildlife installs as a single PyPI package, `PytorchWildlife`, on Windows, macOS, and Linux. The fastest path is a one-line `pip install`; the sections below add GPU acceleration, Docker, and platform-specific notes. Once it is installed, the [API overview](api.md) and [inference examples](inference-examples.md) show what to do next. New to the framework? The [Getting Started](getting-started.md) guide puts this install step in context.

## Prerequisites

- Python 3.8 or newer (3.10+ recommended)
- Optional: an NVIDIA GPU with CUDA 12.1 for a large speedup on batch jobs

## pip install PytorchWildlife

The package name on PyPI is `PytorchWildlife`:

```bash
pip install PytorchWildlife
```

That single command pulls in the detection and classification models, the data utilities, and their dependencies. Model weights are not bundled; each model downloads its own weights automatically the first time you load it.

## Conda

A clean Conda environment keeps the install isolated from other projects:

```bash
conda create -n pytorch-wildlife python=3.10 -y
conda activate pytorch-wildlife
pip install PytorchWildlife
```

> **Windows users:** use the Anaconda Prompt with Anaconda, or PowerShell otherwise.

## GPU Setup

### Check whether CUDA is available

```python
import torch
print(torch.cuda.is_available())
```

### Install a GPU-enabled PyTorch (CUDA 12.1)

If `torch.cuda.is_available()` returns `False` on a CUDA-capable machine, reinstall PyTorch from the CUDA wheel index, then reinstall the framework:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install PytorchWildlife
```

### Ubuntu: OpenCV dependency

```bash
sudo apt-get update
sudo apt-get install -y python3-opencv
```

### macOS: ffmpeg for video decoding

```bash
brew install ffmpeg
```

### Windows

The [Windows installation guide](https://zenodo.org/records/15376499/files/PytorchWildlife_Windows_installation_tutorial.pdf) is a step-by-step walkthrough for setting up the environment from scratch.

## Verify the install

```python
from PytorchWildlife.models import detection as pw_detection

model = pw_detection.MegaDetectorV6()
print("PyTorch-Wildlife loaded successfully.")
```

Weights download automatically on first load, so the first call is slower than later ones.

## Docker

A prebuilt image runs the Gradio demo without a local Python setup:

```bash
docker pull andreshdz/pytorchwildlife:1.0.2.3
docker run -p 80:80 andreshdz/pytorchwildlife:1.0.2.3 python demo/gradio_demo.py
```

## Jupyter notebooks

To run the demo notebooks under your environment, register it as a Jupyter kernel:

```bash
conda install ipykernel
python -m ipykernel install --user --name pytorch-wildlife --display-name "Python (PytorchWildlife)"
```

Then select the `Python (PytorchWildlife)` kernel when opening a notebook.

## Try it without installing

- [Hugging Face demo](https://huggingface.co/spaces/ai-for-good-lab/pytorch-wildlife): upload images in your browser.
- [Google Colab notebook](https://colab.research.google.com/drive/1rjqHrTMzEHkMualr4vB55dQWCsCKMNXi?usp=sharing): free cloud GPU.

## Next steps

- Pick a model from the [Wildlife Model Zoo](model_zoo.md).
- Learn the load-and-run interface in the [API overview](api.md).
- Run end-to-end scripts on the [inference examples](inference-examples.md) page.
