#!/usr/bin/env bash
# Download the camera-trap demo images (CC-BY-4.0, https://zenodo.org/records/15376499)
# and one detector model, using only curl, md5sum and unzip.
#
#   bash interfaces/cli/get_demo_data.sh            # images -> ./data, model -> ~/.sparrow-engine/models
#   DATA_DIR=/tmp/spp bash interfaces/cli/get_demo_data.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
MODEL_DIR="${SPARROW_ENGINE_MODEL_DIR:-$HOME/.sparrow-engine/models}"
# Zenodo rejects curl's default user agent.
UA="sparrow-playground"

fetch() {  # url  target  md5
    local url="$1" target="$2" md5="$3"
    if [[ ! -f "$target" ]]; then
        echo "downloading $(basename "$target") ..."
        curl -fsSL -A "$UA" -o "$target.part" "$url"
        mv "$target.part" "$target"
    fi
    if [[ "$(md5sum "$target" | cut -d' ' -f1)" != "$md5" ]]; then
        rm -f "$target"
        echo "checksum mismatch for $target; deleted, run again" >&2
        exit 1
    fi
}

mkdir -p "$DATA_DIR" "$MODEL_DIR"

fetch "https://zenodo.org/records/15376499/files/demo_data_det.zip?download=1" \
    "$DATA_DIR/demo_data_det.zip" 373e15c8e7c8894379a82086a9177763
unzip -q -o "$DATA_DIR/demo_data_det.zip" -d "$DATA_DIR"
echo "images: $DATA_DIR/data/data_example/images"

# MegaDetector V6 compact, from the Sparrow Model Zoo v0.33.0 (https://doi.org/10.5281/zenodo.23992557).
if [[ ! -f "$MODEL_DIR/MDV6-yolov10-c/manifest.toml" ]]; then
    fetch "https://zenodo.org/records/23992557/files/camera_trap__detector__MDV6-yolov10-c.zip?download=1" \
        "$MODEL_DIR/MDV6-yolov10-c.zip" 39cce4d012608e2969ff735f6eefb58f
    unzip -q -o "$MODEL_DIR/MDV6-yolov10-c.zip" -d "$MODEL_DIR"
    rm -f "$MODEL_DIR/MDV6-yolov10-c.zip"
fi
echo "model:  $MODEL_DIR/MDV6-yolov10-c"
