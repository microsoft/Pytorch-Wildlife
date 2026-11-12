<div align="center">

<img src="assets/sparrow_playground_banner.png" width="100%" alt="SPARROW Playground, Microsoft AI for Good Lab">

<br>

[Sparrow Engine](https://github.com/microsoft/SPARROW-Engine) | [MegaDetector](https://github.com/microsoft/MegaDetector) | [MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) | [SPARROW](https://github.com/microsoft/SPARROW) | [Biodiversity](https://github.com/microsoft/Biodiversity)

<br>

[![sparrow-engine](https://img.shields.io/pypi/v/sparrow-engine?label=sparrow-engine&color=2e8b57)](https://pypi.org/project/sparrow-engine/)
[![license](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![discord](https://img.shields.io/badge/Discord-join_us-5865F2?logo=discord&logoColor=white)](https://discord.gg/TeEVxzaYtm)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen)](#-contributing)

</div>

> [!IMPORTANT]
> **This repository used to be Pytorch-Wildlife.** Old `microsoft/Pytorch-Wildlife` links redirect here.
> If you have Pytorch-Wildlife code, [notebook 10](notebooks/10_from_pytorch_wildlife.ipynb) shows the
> Sparrow Engine call for each old class. The old Pytorch-Wildlife code is kept, unmaintained, on the
> [`archive/pytorch-wildlife`](https://github.com/microsoft/SPARROW-Playground/tree/archive/pytorch-wildlife) branch.

## 👋 What this is

SPARROW Playground is where you try out the biodiversity tools from the Microsoft AI for Good Lab.
It is a set of notebooks and small scripts, written the way a data scientist would work: load some
camera-trap images, a video or a sound file, run a model, look at the results in a table or a plot,
save them, and then copy the code into your own project.

You can use it to:

- find animals, people and vehicles in camera-trap images and videos with MegaDetector;
- classify the animals to species with regional classifiers (Amazon, Serengeti, Europe and others);
- detect birds and identify species in audio recordings;
- compare and search images with BioCLIP embeddings;
- run live detection on a webcam or an RTSP camera;
- export results as MegaDetector JSON, COCO or CSV;
- take any notebook cell or script as a starting point for your own code. They only use the
  public `sparrow_engine` API, so they work outside this repository unchanged.

Nothing here trains models. The playground only runs them.

## 🔄 From Pytorch-Wildlife to SPARROW Playground

Pytorch-Wildlife started in 2023 as a PyTorch framework for running and fine-tuning MegaDetector
and our species classifiers. We changed three things:

1. **The name.** We can no longer use "PyTorch" in a project name, so the project needed a new one.
2. **The engine.** We built [Sparrow Engine](https://github.com/microsoft/SPARROW-Engine), a general
   inference engine written in Rust on ONNX Runtime. It runs on CPU or GPU without PyTorch, and it
   comes with a model zoo of more than 80 models for camera traps, audio, overhead imagery and
   marine imagery. Every example in this repository uses it.
3. **Fine-tuning moved out.** Training code now lives in repositories built for each kind of model:

   | Area | Repository |
   |---|---|
   | Camera-trap detection | [microsoft/MegaDetector](https://github.com/microsoft/MegaDetector) |
   | Species classification | [microsoft/MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier) |
   | Bioacoustics | [microsoft/MegaDetector-Acoustic](https://github.com/microsoft/MegaDetector-Acoustic) |
   | Overhead and aerial imagery | [microsoft/MegaDetector-Overhead](https://github.com/microsoft/MegaDetector-Overhead) |

That leaves this repository with one job: a place to test and play with the SPARROW and AI for
Good biodiversity toolkits.

**Still on the `PytorchWildlife` package?** You can keep using it: `pip install PytorchWildlife`
still works, and version 1.3.1 stays on PyPI. It will not get any more updates or fixes, though,
so we suggest switching to Sparrow Engine now. [Notebook 10](notebooks/10_from_pytorch_wildlife.ipynb)
shows the Sparrow Engine replacement for each Pytorch-Wildlife class and call.

**Looking for the old code?** The full Pytorch-Wildlife codebase, including the PyTorch models,
fine-tuning code and Gradio demo, is on the
[`archive/pytorch-wildlife`](https://github.com/microsoft/SPARROW-Playground/tree/archive/pytorch-wildlife)
branch, and the released 1.3.1 source is at tag
[`v1.3.1`](https://github.com/microsoft/SPARROW-Playground/tree/v1.3.1). Both are kept for reference
only: they will not be maintained, and we will not accept issues or pull requests for them.

## ⚡ Sparrow Engine in five lines

Install the engine (Python 3.11 or later):

```bash
pip install sparrow-engine        # CPU
pip install sparrow-engine-gpu    # NVIDIA GPU, same Python API
```

For video and webcam functions, add the `video` extra, which installs OpenCV:
`pip install "sparrow-engine[video]"`.

Download a model once, then run it on a folder of images:

```python
import sparrow_engine as spe

spe.download_model("MDV6-yolov10-c")                  # MegaDetector v6, into ~/.sparrow-engine/models
results = spe.detect("my_images/", model="MDV6-yolov10-c", threshold=0.3, recursive=True)

for r in results:
    print(r.path, [(d.label, round(d.confidence, 2)) for d in r.detections])
```

Add a species classifier on top of the detector:

```python
spe.download_model("AI4G-Amazon-V2")
results = spe.pipeline("my_images/", detector="MDV6-yolov10-c", classifier="AI4G-Amazon-V2", threshold=0.3)

for r in results:
    for d in r.detections:
        print(r.path, d.detection.label, "->", d.classification.label, round(d.classification.confidence, 2))
```

<details>
<summary>More: video, audio, embeddings, export</summary>

```python
# Video (needs the video extra): one result per sampled frame
frames = spe.detect_video("clip.mp4", model="MDV6-yolov10-c", every_n=30)

# Audio: bird species in a WAV file
spe.download_model("perch-v2")
audio = spe.detect_audio("recording.wav", model="perch-v2")

# Embeddings: one L2-normalized vector per image
spe.download_model("bioclip-2-fp16")
vectors = spe.embed(["a.jpg", "b.jpg"], model="bioclip-2-fp16")

# Export detections
spe.export(results, "coco", "coco.json")
```

</details>

Inference never downloads anything by itself. Set `SPARROW_ENGINE_MODEL_DIR` to keep models
somewhere other than `~/.sparrow-engine/models`.

## 🦁 The model zoo

Installing Sparrow Engine gives you the whole Sparrow Model Zoo: 82 models in zoo v0.31.0, across
camera traps (48), acoustics (13), overhead imagery (10), marine imagery (6) and general use (5).
54 of them download directly; the rest link to their original source.

```python
import sparrow_engine as spe

for m in spe.zoo_models():
    print(m.id, m.domain, m.task, m.license, m.downloadable)
```

- Full catalogue with sources, licences and citations:
  [Sparrow Engine model zoo catalogue](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/model-zoo-catalogue.md)
- Model files: [Sparrow Model Zoo on Zenodo](https://doi.org/10.5281/zenodo.20348978) (this link always opens the latest version)
- Browse it in a notebook: [01 Model zoo](notebooks/01_model_zoo.ipynb)

Each model keeps its own licence. Check it before you use a model, especially for commercial work.

## 📓 Notebooks

Every notebook installs `sparrow-engine` if it is missing and downloads its own demo data.

| Notebook | What it shows |
|---|---|
| [00 Quickstart](notebooks/00_quickstart.ipynb) | Download a detector and find animals in a folder of images |
| [01 Model zoo](notebooks/01_model_zoo.ipynb) | Browse the Sparrow Model Zoo and download models |
| [02 Image detection](notebooks/02_image_detection.ipynb) | Detection results, confidence thresholds, drawing boxes |
| [03 Detection + classification](notebooks/03_detection_classification.ipynb) | Detect animals, then classify each one to species |
| [04 Bioacoustics](notebooks/04_bioacoustics.ipynb) | Bird presence and species from WAV recordings |
| [05 Embeddings](notebooks/05_embeddings.ipynb) | BioCLIP image embeddings and similarity search |
| [06 Export](notebooks/06_export.ipynb) | Save results as MegaDetector JSON, COCO or CSV |
| [07 Day/night and visualization](notebooks/07_day_night_visualization.ipynb) | Day/night tagging and annotated images |
| [08 Video](notebooks/08_video.ipynb) | Detect and classify animals in video, write an annotated clip |
| [09 Webcam](notebooks/09_webcam.ipynb) | Live detection from a webcam or RTSP camera |
| [10 From Pytorch-Wildlife](notebooks/10_from_pytorch_wildlife.ipynb) | Map Pytorch-Wildlife classes and calls to Sparrow Engine |
| [16 HTTP server](notebooks/16_http_server.ipynb) | Run the engine as a Docker service and call it over HTTP (needs Docker) |

Demo data goes into `./data` (set `SPP_DATA_DIR` to change it).

### Receiving field data (sparrow-comm)

Notebooks 11-15 use [sparrow-comm](https://github.com/microsoft/SPARROW-Comm) to receive images
from field hardware, then process them with Sparrow Engine. Every receiver runs inside the
notebook, so no server or cloud account is needed. They need **Python 3.12** or later and install
`sparrow-comm[ftps]` themselves (`pip install -r requirements-comms.txt` does it up front).

| Notebook | What it shows |
|---|---|
| [11 Device API](notebooks/11_comm_device_api.ipynb) | A Sparrow field device onboards and uploads images; person images are discarded |
| [12 FTPS](notebooks/12_comm_ftps.ipynb) | A trail camera uploads over FTPS; the poller stores the files |
| [13 Reconyx](notebooks/13_comm_reconyx.ipynb) | The Reconyx cloud forwards images to the project endpoint |
| [14 E-mail](notebooks/14_comm_email.ipynb) | Poll a camera's IMAP mailbox (offline by default, or a real mailbox) |
| [15 Process on arrival](notebooks/15_comm_process_on_arrival.ipynb) | Run MegaDetector the moment any channel delivers an image |

## 🛠️ Scripts

To run the notebooks locally, or to use the scripts, clone the repository and install the
requirements:

```bash
git clone https://github.com/microsoft/SPARROW-Playground.git
cd SPARROW-Playground
pip install -r requirements.txt
```

| Script | Use |
|---|---|
| `scripts/download_models.py` | List or download zoo models (`--list`, or model IDs) |
| `scripts/detect_folder.py` | Detect in a folder of images, export JSON/COCO/CSV, optional annotated images |
| `scripts/pipeline.py` | Detection plus species classification |
| `scripts/audio.py` | Bioacoustics on WAV files |
| `scripts/video.py` | Detection on a video file, annotated output video |
| `scripts/webcam.py` | Live detection in an OpenCV window (needs `opencv-python`, not the headless build) |

## 🔌 Other interfaces

The notebooks use the Python package. The same engine and models are also available as a command-line
tool, an HTTP server and a C library, for code that is not Python:

| Interface | Example | Good for |
|---|---|---|
| `spe` command line | [`interfaces/cli`](interfaces/cli) | Shell scripts, batch jobs, no Python install |
| HTTP server (Docker) | [notebook 16](notebooks/16_http_server.ipynb) | Other languages, other machines, one model shared by several programs |
| C library | [`interfaces/c`](interfaces/c) | Embedding the engine in C, C++, Go, Rust or C# programs |

## 📡 Coming next: live data

Notebooks 11-15 receive data the way it arrives from the field: [SPARROW](https://github.com/microsoft/SPARROW)
solar-powered edge devices, 4G cellular camera traps (by e-mail or through the Reconyx cloud), and
FTPS uploads. In the notebooks the senders are simulated, so they run anywhere. Next:

- notebooks that connect to a real deployment;
- drones.

Each one follows the same pattern as the rest of the playground: connect, run a model on what
arrives, look at the results.

## 📜 Licenses

The code in this repository is MIT licensed. Each model in the Sparrow Model Zoo carries its own
license; check `license` in `sparrow_engine.zoo_models()` (or notebook 01) before using a model.

Demo data:
- Camera-trap images and the opossum video: Pytorch-Wildlife demo data,
  [Zenodo record 15376499](https://zenodo.org/records/15376499), CC BY 4.0.
- Bird recording: "Vireo flavifrons - Yellow-Throated Vireo Song" by SwampVids,
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vireo_flavifrons_-_Yellow-Throated_Vireo_Song.wav),
  CC BY 4.0.

## 📝 Cite

If SPARROW Playground or Pytorch-Wildlife helped your research, please cite:

```bibtex
@misc{hernandez2024pytorchwildlife,
      title={Pytorch-Wildlife: A Collaborative Deep Learning Framework for Conservation},
      author={Andres Hernandez and Zhongqi Miao and Luisa Vargas and Sara Beery and Rahul Dodhia and Juan Lavista},
      year={2024},
      eprint={2405.12930},
      archivePrefix={arXiv},
}
```

If you use MegaDetector, please also cite:

```bibtex
@misc{beery2019efficient,
      title={Efficient Pipeline for Camera Trap Image Review},
      author={Sara Beery and Dan Morris and Siyu Yang},
      year={2019},
      eprint={1907.06772},
      archivePrefix={arXiv},
}
```

## 💬 Questions

- Email: [zhongqimiao@microsoft.com](mailto:zhongqimiao@microsoft.com)
- Discord: [join us](https://discord.gg/TeEVxzaYtm)
- Hub for the AI for Good Lab's biodiversity work: [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity)

## 🤝 Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## ™️ Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
