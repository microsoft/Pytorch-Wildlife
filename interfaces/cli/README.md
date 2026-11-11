# Command line (`spe`)

The `spe` command runs the same engine and models as the Python package, without Python.
Use it in shell scripts, cron jobs, or on machines where you would rather not set up a Python environment.

## Install

| Platform | Command |
|---|---|
| macOS arm64, Linux x86_64 (Homebrew) | `brew tap microsoft/sparrow-engine && brew install sparrow-engine` (`sparrow-engine-gpu` for NVIDIA) |
| Any | download `sparrow-engine-cpu-<version>-<os>.tar.gz` (or `.zip` on Windows) from the [Sparrow Engine releases](https://github.com/microsoft/SPARROW-Engine/releases), unpack it and put `bin/` on your `PATH` |

Check it with `spe --version` and `spe device`.

> `pip install sparrow-engine` installs the Python package only; it does not put `spe` on your `PATH`.

## Models

`spe` reads models from `~/.sparrow-engine/models` (or `--model-dir`, or `SPARROW_ENGINE_MODEL_DIR`).
Each model is a folder named after its id. Ways to fill it:

- `bash interfaces/cli/get_demo_data.sh` downloads the demo images and `MDV6-yolov10-c` with `curl` only.
- Any other model: download its zip from the [Sparrow Model Zoo on Zenodo](https://doi.org/10.5281/zenodo.20348978)
  (files are named `<domain>__<task>__<model-id>.zip`) and unzip it into the model directory.
- If Python is available: `python scripts/download_models.py MDV6-yolov10-c AI4G-Amazon-V2`.

`spe models list` shows what the engine found.

## Scripts

| Script | What it does |
|---|---|
| [`get_demo_data.sh`](get_demo_data.sh) | demo camera-trap images into `./data`, `MDV6-yolov10-c` into the model directory, both checksum-verified |
| [`detect_folder.sh`](detect_folder.sh) | `spe detect` on a folder: COCO and CSV files plus annotated images |
| [`pipeline.sh`](pipeline.sh) | `spe pipeline`: detect, then classify each animal crop (default classifier `AI4G-Amazon-V2`) |

```bash
bash interfaces/cli/get_demo_data.sh
bash interfaces/cli/detect_folder.sh data/data/data_example/images
python scripts/download_models.py AI4G-Amazon-V2      # or unzip it from Zenodo by hand
bash interfaces/cli/pipeline.sh data/data/data_example/images
```

Settings are environment variables, for example `MODEL=MDV6-yolov10-e THRESHOLD=0.3 OUT=results bash interfaces/cli/detect_folder.sh my_images`.

## Other commands

```bash
spe classify image.jpg --model AI4G-Amazon-V2 --print
spe detect-audio recording.wav --model buzzdetect --print
spe embed image.jpg --model bioclip-2 > embeddings.ndjson
spe day-night image.jpg
spe models info MDV6-yolov10-c
spe --help
```

The full command reference is in the [Sparrow Engine user manual](https://github.com/microsoft/SPARROW-Engine/blob/main/docs/user-manual.md).
