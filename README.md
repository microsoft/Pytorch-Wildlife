# Sparrow Engine

A Rust ML inference engine for camera-trap and bioacoustic data.
Drop-in for MegaDetector v6, DeepFaune, HerdNet, OWL-T, SpeciesNet, and
MD_AudioBirds_V1; model-agnostic via TOML manifests.

## Quickstart

### Easiest: Homebrew (macOS arm64 / brew-Linux x86_64)

```bash
brew tap microsoft/sparrow-engine
brew install sparrow-engine            # CPU; works on macOS arm64 + brew-Linux x86_64
brew install sparrow-engine-gpu        # GPU; brew-Linux x86_64 + NVIDIA only

spe device                              # {"device":"cpu"}  or  {"device":"cuda:0"}

# One-time: download a model from the Zenodo bundle (brew doesn't ship models)
mkdir -p ~/.sparrow-engine/models && cd ~/.sparrow-engine/models
curl -fLO https://zenodo.org/records/20360316/files/MDV6-yolov10-e.zip
unzip -q MDV6-yolov10-e.zip && rm MDV6-yolov10-e.zip
cd -

spe detect /path/to/photos --model MDV6-yolov10-e --recursive --export-format megadet --export-output detections.json
```

Both formulas can coexist (separate binaries `spe` + `spe-gpu`; shared model cache at `~/.sparrow-engine/models/`). The example above pulls MegaDetector v6 (general camera-trap detection); see the [Model zoo](#model-zoo) section below for the other 15 models in the Zenodo bundle (image classifiers, audio detectors, overhead-imagery detectors). See `docs/user-manual.md §2.4` for the other install paths.

#### GPU host prerequisites

The `sparrow-engine-gpu` formula ships ~256 MB of `libonnxruntime` + ORT CUDA provider sidecars, but it does **NOT** bundle NVIDIA's runtime libraries (NVIDIA's license forbids redistribution). The host must provide:

| Library | Apt package (Ubuntu/Debian) | pip wheel (no root) | Why |
|---|---|---|---|
| NVIDIA driver ≥550.x | `nvidia-driver-550` (or newer) | — (kernel module; host-only) | GPU access |
| CUDA runtime 12.6 | `nvidia-cuda-toolkit` brings it | `nvidia-cuda-runtime-cu12` | `libcudart.so.12` |
| **cuDNN ≥9.10** (9.8 has Conv bug on sm_89) | `nvidia-cudnn` | `nvidia-cudnn-cu12` | `libcudnn.so.9` — convolutions |
| cuBLAS | bundled with CUDA toolkit | `nvidia-cublas-cu12` | matrix multiplications |
| cuRAND | bundled with CUDA toolkit | `nvidia-curand-cu12` | rand sampling (some models) |
| cuFFT | bundled with CUDA toolkit | `nvidia-cufft-cu12` | audio FFT (MD_AudioBirds_V1) |
| nvJPEG | bundled with CUDA toolkit | `nvidia-nvjpeg-cu12` | GPU JPEG decode |

After installing the libraries (system or pip), the brew-installed `spe-gpu` wrapper auto-discovers them from common host locations — no `LD_LIBRARY_PATH` setup needed for production users. Search order (first hit wins):

1. `SPARROW_ENGINE_CUDA_LIB_DIR` (user override; honored as-is)
2. `~/.sparrow-engine/cuda-sidecars/lib/python*/site-packages/nvidia/*/lib` (the convention if you used pip sidecars)
3. `/usr/lib/python3/dist-packages/torch/lib` (Lambda Stack / system PyTorch — cuDNN comes bundled)
4. `/usr/local/cuda/lib64` (NVIDIA CUDA toolkit)
5. `/usr/lib/x86_64-linux-gnu` (Ubuntu apt nvidia-cudnn)

Full table + remediation appears in `brew info sparrow-engine-gpu`. Quick all-pip install (no root) for a fresh host:

```bash
uv venv ~/.sparrow-engine/cuda-sidecars --python 3.11
~/.sparrow-engine/cuda-sidecars/bin/pip install \
    nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 \
    nvidia-cufft-cu12 nvidia-nvjpeg-cu12 nvidia-cuda-runtime-cu12
```

Verify with `spe-gpu device` — `{"device":"cuda:0"}` means good, any dlopen error in the output names the missing library.

### Alternative install paths

If brew isn't right for your environment (server distro without brew-Linux, Windows, etc.), the install wrapper handles probe-and-install for Linux / macOS / Windows:

```bash
# Linux / macOS — clone the repo and run from its root
bash installer/sparrow-engine-install.sh
```

```powershell
# Windows PowerShell — clone the repo and run from its root
installer\sparrow-engine-install.ps1
```

The wrapper probes hardware once, picks the right CPU or GPU build, and
installs the matching CLI binary plus the Python wheel into `~/.sparrow-engine/`.
Pass `--flavor cpu` or `--flavor gpu` to skip the probe. Pass `--docker`
to install the HTTP-server image instead.

System prerequisites for GPU: NVIDIA driver ≥550.x, CUDA 12.6 runtime,
and **cuDNN ≥9.10** (cuDNN 9.8 has a Conv-engine bug on sm_89).

### Python package only (PyPI)

If you only want the Python wheel — no CLI, no Docker image — install
straight from PyPI. Both wheels target CPython ≥ 3.11 (`cp311-abi3`), so
make sure your venv runs Python 3.11 or newer.

**With `uv` (recommended)**:

```bash
uv venv --python 3.11
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# CPU
uv pip install sparrow-engine

# GPU (Linux x86_64 only; requires CUDA 12.6 runtime on the host)
uv pip install sparrow-engine-gpu
```

`uv venv` does not ship `pip` inside the venv by default, so use `uv pip
install` (uv's pip-compatible wrapper) instead of bare `pip install`.
Calling `pip install …` after `source activate` falls back to the system
pip, which usually targets the wrong Python version and fails with
`No matching distribution found`.

**With stdlib `venv`**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate         # Windows: .venv\Scripts\activate

# CPU
pip install sparrow-engine

# GPU (Linux x86_64 only; requires CUDA 12.6 runtime on the host)
pip install sparrow-engine-gpu
```

Both wheels import as `sparrow_engine`. Never install both into the same
environment. Check the installed version with
`python -c "import sparrow_engine; print(sparrow_engine.__version__)"`.
See [§6 of the user manual](docs/user-manual.md#6-python-package--sparrow-engine)
for the full API surface and GPU sidecar options.

---

> 📖 **[Read the full user manual →](docs/user-manual.md)**
>
> One document covering install, CLI (`spe`), Python wheel (`import sparrow_engine`), HTTP API server, HTTP SDK, native DLL (C ABI), TOML model manifests, the Phase 4 inference-log / drift / provenance surface, cold-start + lazy load, gotchas + edge cases, performance characteristics, and Sparrow Studio integration.

---

## Model zoo

Sparrow Engine doesn't ship the ONNX model weights in the repo. They live in a public Zenodo record so the repo stays small and operators can pull just the models they need.

**Zenodo DOI**: [10.5281/zenodo.20360316](https://doi.org/10.5281/zenodo.20360316) (v0.4.0) — concept DOI [10.5281/zenodo.20348978](https://doi.org/10.5281/zenodo.20348978) always resolves to the latest version.

Download all 16 models to `./models/`:

```bash
bash scripts/download_models.sh
```

Or just specific models:

```bash
bash scripts/download_models.sh MDV6-yolov10-e SpeciesNet-Crop
bash scripts/download_models.sh --list          # list available model IDs
bash scripts/download_models.sh --dest /custom/path
```

Point Sparrow Engine at the directory:

```bash
export SPARROW_ENGINE_MODELS_DIR=$(realpath ./models)
spe models list                                 # confirms catalog discovery
spe detect --model MDV6-yolov10-e --print image.jpg
```

The downloader verifies SHA-256 per model, is idempotent (skip-if-present unless `--force`), and unpacks into the layout Sparrow Engine expects (`<dir>/<model_id>/manifest.toml` + `model.onnx` + `labels.txt`).

### Per-model catalog

This is a **multi-license bundle** — each model ships under its own upstream license. Open each `models/<model_id>/LICENSE.md` after download for the canonical terms.

The catalog splits into four families (detectors, heatmap detectors, classifiers, audio). All detectors emit bounding boxes via in-graph NMS; all classifiers consume crops produced by an upstream detector.

#### Bounding-box detectors

| Model ID | Resolution | Classes | ONNX | License |
|---|---|---|---|---|
| `MDV6-yolov10-c` | 640 × 640 | 3 (animal / person / vehicle) | 9 MB | Ultralytics AGPL-3.0 |
| `MDV6-yolov10-e` | 1280 × 1280 | 3 (animal / person / vehicle) | 113 MB | Ultralytics AGPL-3.0 |
| `Species_Net_MDV5a` | 1280 × 1280 | 3 (animal / person / vehicle) | 535 MB | Ultralytics AGPL-3.0 |
| `deepfaune-yolo8s` | 960 × 960 | 3 (MD-style) | 43 MB | AGPL-3.0 ∩ CC-BY-NC-SA 4.0 |
| `european_mammals` | 640 × 480 | 31 | 113 MB | Ultralytics AGPL-3.0 |
| `north_american_mammals` | 640 × 480 | 14 | 113 MB | Ultralytics AGPL-3.0 |
| `sub_saharan` | 640 × 480 | 35 | 113 MB | Ultralytics AGPL-3.0 |

- MegaDetector v6 (`MDV6-yolov10-c` / `-e`) is the recommended default detector — `-c` for speed, `-e` for accuracy.
- `Species_Net_MDV5a` is the legacy v5a detector; kept for projects validated against v5a outputs.
- `deepfaune-yolo8s` is the DeepFaune detector stage, designed to pair with `Deepfaune-Europe` / `Deepfaune-New-England` classifiers.
- `european_mammals` / `north_american_mammals` / `sub_saharan` are the AI for Good Lab regional YOLO detectors (multi-species per region).

#### Heatmap-based detectors

| Model ID | Resolution | Classes | ONNX | License |
|---|---|---|---|---|
| `HerdNet_General_Dataset_2022` | 512 × 512 | 6 species + background | 70 MB | MIT |
| `OWL` | 512 × 512 (tiled) | 1 (animal) | 114 MB | MIT |

- `HerdNet_General_Dataset_2022` counts large African mammals (elephants, antelopes, zebras, etc.) in low-altitude aerial / drone imagery.
- `OWL` does tiled detection of small wildlife in large camera-trap or aerial scenes; converts heatmap peaks to fixed-size boxes.

#### Image classifiers (consume crops from a detector)

| Model ID | Crop | Classes | ONNX | License |
|---|---|---|---|---|
| `Deepfaune-Europe` | 182 × 182 | 34 | 1.2 GB | CC-BY-NC-SA 4.0 |
| `Deepfaune-New-England` | 182 × 182 | 24 | 1.2 GB | CC-BY-NC-SA 4.0 |
| `SpeciesNet-Crop` | 480 × 480 | 2498 | 214 MB | Apache 2.0 |
| `AI4G-Amazon-V2` | 224 × 224 | 36 | 90 MB | MIT |
| `AI4G-Serengeti` | 224 × 224 | 10 | 43 MB | MIT |

- `Deepfaune-Europe` / `Deepfaune-New-England` are the DeepFaune classifier stage for European and New England (NA) mammals.
- `SpeciesNet-Crop` is Google's SpeciesNet classifier; pairs downstream of a detector (e.g. MDv6).
- `AI4G-Amazon-V2` and `AI4G-Serengeti` are AI for Good Lab regional classifiers for Amazon-basin and Serengeti / East African species.

#### Audio detectors / classifiers

| Model ID | Input window | Classes | ONNX | License |
|---|---|---|---|---|
| `MD_AudioBirds_V1` | 1 s @ 48 kHz, mel spectrogram (0.3 s stride) | 1 (bird vs no-bird) | 81 MB | MIT |
| `perch-v2` | 5 s @ 32 kHz raw audio | 14795 | 391 MB | Apache 2.0 |

- `MD_AudioBirds_V1` is the sparrow-engine default audio detector — a lightweight binary bird-vs-no-bird model used in benchmarks and Phase 4.x manual tests. Sliding-window mel-spectrogram front-end (Slaney mel scale + Slaney filter norm). Ships in the v0.4.0 Zenodo bundle (DOI [10.5281/zenodo.20360316](https://doi.org/10.5281/zenodo.20360316)) as FP32; the FP16 conversion path is in `sparrow-engine/tools/convert_fp16.py` and is parity-verified against the FP32 reference (Phase 3.8 Step 2 post-STRETCH audit, 2026-05-05).
- `perch-v2` is Google Perch 2, a global bird-vocalisation classifier (Conformer encoder) with an in-graph mel front-end. Takes 160000-sample windows of raw audio; emits softmax over 14795 classes (birds + non-bird FSD50K labels).

#### License summary

- **Ultralytics AGPL-3.0** (7 models): MDv6 × 2, MDv5a, the 3 AI4G regional YOLOs, plus `deepfaune-yolo8s` (which also intersects CC-BY-NC-SA 4.0).
- **CC-BY-NC-SA 4.0** (3 models): `deepfaune-yolo8s`, `Deepfaune-Europe`, `Deepfaune-New-England`.
- **Apache 2.0** (2 models): `SpeciesNet-Crop`, `perch-v2`.
- **MIT** (5 models): `AI4G-Amazon-V2`, `AI4G-Serengeti`, `OWL`, `HerdNet_General_Dataset_2022`, `MD_AudioBirds_V1`.

**Commercial users of YOLO-based detectors** should obtain an [Ultralytics Enterprise License](https://www.ultralytics.com/license).

---

## Architecture

Sparrow Engine is engine-only: it loads ONNX models and runs inference.
Annotation, training, data versioning, model registry, drift detection,
and deployment orchestration live in sibling repos.

Core invariants:

- ONNX for all models (vision + audio)
- NCHW layout mandatory
- Normalized bbox `[0,1]` at all public API boundaries
- TOML manifests (one per model)
- NMS in the ONNX graph, never in the Sparrow Engine
- `Engine` is a singleton (ORT is process-global)

## License

See [`LICENSE`](LICENSE).

---

## Internal development

This is the **public** sparrow-engine repo. It carries the shipping code, the install wrapper, models, and one user-facing manual.

Dev/AI artifacts — design rounds, research notes, audit-fix / doc-fix / `/implement` skill rounds, inquisitor reports, scope ledgers, prompt logs, agent instructions, plan / changelog / lessons / ideas — live in the **internal dev companion** repo (`zhmiao/sparrow-engine-dev`), NOT here. See that repo's `docs/design/architecture.md § Internal dev companion convention` for the full rule.
