#!/usr/bin/env bash
# Detect animals, people and vehicles in a folder of images with the spe command.
# Writes a COCO file, a CSV file and annotated images.
#
#   bash interfaces/cli/detect_folder.sh data/data/data_example/images
#   MODEL=MDV6-yolov10-e OUT=out bash interfaces/cli/detect_folder.sh my_images
set -euo pipefail

IMAGES="${1:?usage: detect_folder.sh IMAGE_FOLDER}"
MODEL="${MODEL:-MDV6-yolov10-c}"
THRESHOLD="${THRESHOLD:-0.2}"
OUT="${OUT:-out/cli_detect}"
mkdir -p "$OUT"

spe detect "$IMAGES" --recursive --model "$MODEL" --threshold "$THRESHOLD" --quiet \
    --export-format coco --export-output "$OUT/detections_coco.json"

spe detect "$IMAGES" --recursive --model "$MODEL" --threshold "$THRESHOLD" --quiet \
    --export-format csv --export-output "$OUT/detections.csv" \
    --visualize --output-dir "$OUT/annotated" --show-labels --summary

echo "COCO:      $OUT/detections_coco.json"
echo "CSV:       $OUT/detections.csv"
echo "annotated: $OUT/annotated/"
