# SPARROW Playground

> [!IMPORTANT]
> **This repository was Pytorch-Wildlife. It is now SPARROW Playground.**
> The models are the same (MegaDetector, the AI for Good species classifiers, and more) and now run
> on [Sparrow Engine](https://github.com/microsoft/SPARROW-Engine), which does not need PyTorch.
> Old `microsoft/Pytorch-Wildlife` links redirect here.
>
> - Coming from Pytorch-Wildlife? Start with
>   [`notebooks/10_from_pytorch_wildlife.ipynb`](notebooks/10_from_pytorch_wildlife.ipynb).
> - The `PytorchWildlife` Python package ended at version 1.3.1 and is no longer maintained. Its
>   code is kept at tag [`v1.3.1`](https://github.com/microsoft/SPARROW-Playground/tree/v1.3.1).
> - Fine-tuning moved to [MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier).

SPARROW Playground is a set of notebooks and scripts that show what the SPARROW toolkit can do for
wildlife monitoring: detect animals in camera-trap images and videos, classify them to species,
identify birds from audio, compare images with embeddings, and export the results. Every example
runs on the `sparrow-engine` Python package and models from the Sparrow Model Zoo.

This repository is for inference only. To fine-tune MegaDetector classifiers, use
[microsoft/MegaDetector-Classifier](https://github.com/microsoft/MegaDetector-Classifier).

## Install

Python 3.11 or later.

```bash
pip install -r requirements.txt          # CPU
# NVIDIA GPU: pip install sparrow-engine-gpu  (instead of sparrow-engine)
```

Each notebook also installs `sparrow-engine` itself when it is missing, so the notebooks run
in Google Colab without a separate setup step.

Models are downloaded on first use into `~/.sparrow-engine/models` (set
`SPARROW_ENGINE_MODEL_DIR` to change it):

```bash
python scripts/download_models.py --list
python scripts/download_models.py MDV6-yolov10-c AI4G-Amazon-V2
```

## Notebooks

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

Demo data is downloaded by the notebooks into `./data` (set `SPP_DATA_DIR` to change it).

## Scripts

| Script | Use |
|---|---|
| `scripts/download_models.py` | List or download zoo models |
| `scripts/detect_folder.py` | Detect in a folder of images, export JSON/COCO/CSV, optional annotated images |
| `scripts/pipeline.py` | Detection plus species classification |
| `scripts/audio.py` | Bioacoustics on WAV files |
| `scripts/video.py` | Detection on a video file, annotated output video |
| `scripts/webcam.py` | Live detection in an OpenCV window (needs `opencv-python`, not the headless build) |

## Licenses

The code in this repository is MIT licensed. Each model in the Sparrow Model Zoo carries its own
license; check `license` in `sparrow_engine.zoo_models()` (or notebook 01) before using a model.

Demo data:
- Camera-trap images and the opossum video: Pytorch-Wildlife demo data,
  [Zenodo record 15376499](https://zenodo.org/records/15376499), CC BY 4.0.
- Bird recording: "Vireo flavifrons - Yellow-Throated Vireo Song" by SwampVids,
  [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Vireo_flavifrons_-_Yellow-Throated_Vireo_Song.wav),
  CC BY 4.0.

## Cite

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

## Questions

- Email: [zhongqimiao@microsoft.com](mailto:zhongqimiao@microsoft.com)
- Discord: [join us](https://discord.gg/TeEVxzaYtm)
- Hub for the AI for Good Lab's biodiversity work: [microsoft/Biodiversity](https://github.com/microsoft/Biodiversity)

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit [Contributor License Agreements](https://cla.opensource.microsoft.com).

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
