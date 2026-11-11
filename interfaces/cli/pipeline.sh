#!/usr/bin/env bash
# Detect, then classify each animal crop to species, with the spe command.
# Both models must be in the model directory first (see README.md, "Models").
#
#   bash interfaces/cli/pipeline.sh data/data/data_example/images
#   CLASSIFIER=AI4G-Serengeti bash interfaces/cli/pipeline.sh my_images
set -euo pipefail

IMAGES="${1:?usage: pipeline.sh IMAGE_FOLDER}"
DETECTOR="${DETECTOR:-MDV6-yolov10-c}"
CLASSIFIER="${CLASSIFIER:-AI4G-Amazon-V2}"
OUT="${OUT:-out/cli_pipeline}"
mkdir -p "$OUT"

spe pipeline "$IMAGES" --recursive --detector "$DETECTOR" --classifier "$CLASSIFIER" \
    --top-k 3 --quiet \
    --export-format csv --export-output "$OUT/pipeline.csv" \
    --visualize --output-dir "$OUT/annotated" --show-labels

echo "CSV:       $OUT/pipeline.csv"
echo "annotated: $OUT/annotated/"
