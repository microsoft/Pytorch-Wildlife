#!/usr/bin/env bash
# scripts/download_models.sh — download the sparrow-engine model zoo from Zenodo.
#
# Downloads the ONNX model bundles from the public Zenodo record
# (https://doi.org/10.5281/zenodo.20864372, concept DOI 10.5281/zenodo.20348978
# which always resolves to the latest version), verifies SHA-256 integrity,
# and unpacks each model into a layout directly loadable by sparrow-engine
# (i.e. `<dest>/<model_id>/manifest.toml` + `model.onnx` + `labels.txt`).
#
# Usage:
#   bash scripts/download_models.sh                    # all 18 models to ~/.sparrow-engine/models/
#   bash scripts/download_models.sh --dest /path       # custom destination dir
#   bash scripts/download_models.sh MDV6-yolov10-e ... # specific model(s) only
#   bash scripts/download_models.sh --list             # show available models
#   bash scripts/download_models.sh --force            # re-download even if present
#   bash scripts/download_models.sh --no-verify        # skip SHA-256 check (faster, unsafe)
#
# After the script completes, point sparrow-engine at the directory:
#   export SPARROW_ENGINE_MODEL_DIR=$(realpath ~/.sparrow-engine/models)
#   spe list-models     # lists the 18 catalog entries
#   spe detect --model MDV6-yolov10-e --image /path/to/image.jpg
#
# (No explicit env var is needed if the default ~/.sparrow-engine/models is
# used — the CLI / server / Python wheels all default to that path.)
#
# Override the Zenodo record (e.g. to test a newer version):
#   ZENODO_RECORD=<id> bash scripts/download_models.sh

set -euo pipefail

# ---- Constants ----
ZENODO_RECORD="${ZENODO_RECORD:-20864372}"
ZENODO_DOI="10.5281/zenodo.${ZENODO_RECORD}"
ZENODO_BASE="https://zenodo.org/records/${ZENODO_RECORD}/files"

# The 18 ONNX model entries auto-downloaded by default. The v0.9.0 Zenodo bundle also
# carries spe-mobile .tflite artifacts (the fp16 orca-cascade re-exports, MDV6-yolov10-c-tflite,
# the orca-detector-dclde2026-v2/v3 detectors in fp16/int8, and the int8 ecotype) plus the
# orca-cascade pipeline.toml; those are fetched on demand by spe-mobile consumers (e.g. the
# water-sparrow bundle), not part of this default catalog.
# (v0.1.0 had 14; v0.2.0 added perch-v2 — bird vocalization classifier;
#  v0.3.0 added md-audiobirds-v1 — default audio detector, MIT;
#  v0.4.0 — OWL + HerdNet manifest fix (subtype = "overhead"). No new models;
#  v0.5.0 — added orca-detector-dclde2026-v1 + orca-ecotype-dclde2026-v1
#           cascade for DCLDE 2026 (requires sparrow-engine >= v0.1.16);
#  v0.6.0 — added the two fp16 orca-cascade .tflite re-exports for spe-mobile;
#  v0.7.0 — added MDV6-yolov10-c-tflite (mobile fp16 detector) + orca-cascade pipeline;
#  v0.8.0 — added orca-detector-dclde2026-v2 (SpecAug + hard-neg Stage-1 retrain: onnx +
#           fp16/int8 .tflite) + orca-ecotype-melinput-int8-tflite. v1 detector kept.
#  v0.9.0 — added orca-detector-dclde2026-v3 (fine-tune 3class_sparrow_ft_v3: onnx +
#           fp16/int8 .tflite). v1/v2 detectors kept.)
ALL_MODELS=(
  "MDV6-yolov10-e"
  "MDV6-yolov10-c"
  "deepfaune-yolo8s"
  "HerdNet_General_Dataset_2022"
  "OWL"
  "Species_Net_MDV5a"
  "european_mammals"
  "north_american_mammals"
  "sub_saharan"
  "SpeciesNet-Crop"
  "AI4G-Amazon-V2"
  "AI4G-Serengeti"
  "Deepfaune-Europe"
  "Deepfaune-New-England"
  "perch-v2"
  "md-audiobirds-v1"
  "orca-detector-dclde2026-v1"
  "orca-ecotype-dclde2026-v1"
)

# ---- Defaults ----
# Default destination matches the CLI / server / Python default
# (`dirs_default_model_dir` in sparrow-engine-cli/src/main.rs). Picking the
# same path means a no-arg `download_models.sh` followed by a no-arg `spe
# detect` works without any env-var ceremony. Phase D round-2 D-R2-4 fix.
DEST="${HOME:-.}/.sparrow-engine/models"
VERIFY=1
FORCE=0
SELECTED=()

# ---- Argument parsing ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dest)       DEST="$2"; shift 2 ;;
    --dest=*)     DEST="${1#*=}"; shift ;;
    --no-verify)  VERIFY=0; shift ;;
    --force)      FORCE=1; shift ;;
    --list)
      echo "Available models (Zenodo record ${ZENODO_RECORD}, DOI ${ZENODO_DOI}):"
      for m in "${ALL_MODELS[@]}"; do echo "  ${m}"; done
      exit 0
      ;;
    -h|--help)
      sed -n '/^# /,/^set/p' "$0" | sed '/^set/d; s/^# \?//'
      exit 0
      ;;
    -*)
      echo "ERROR: unknown flag '$1'. Use --help for usage." >&2
      exit 1
      ;;
    *)
      SELECTED+=("$1")
      shift
      ;;
  esac
done

if [[ ${#SELECTED[@]} -eq 0 ]]; then
  SELECTED=("${ALL_MODELS[@]}")
fi

# Validate selected ids against the known catalog.
for m in "${SELECTED[@]}"; do
  ok=0
  for available in "${ALL_MODELS[@]}"; do
    if [[ "$m" == "$available" ]]; then ok=1; break; fi
  done
  if [[ $ok -eq 0 ]]; then
    echo "ERROR: unknown model id '$m'. Run with --list to see available models." >&2
    exit 1
  fi
done

# Tool check.
for tool in curl unzip sha256sum; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "ERROR: required tool '$tool' not found in PATH." >&2
    exit 1
  fi
done

# ---- Prep ----
mkdir -p "$DEST"

echo "Zenodo record: ${ZENODO_RECORD} (DOI ${ZENODO_DOI})"
echo "Destination:   $(realpath "$DEST")"
echo "Models:        ${#SELECTED[@]} of ${#ALL_MODELS[@]}"
echo ""

# ---- Fetch checksums.sha256 once (for SHA-256 integrity) ----
if [[ $VERIFY -eq 1 ]]; then
  echo "Downloading checksums.sha256..."
  if curl -fsSL "$ZENODO_BASE/checksums.sha256" -o "$DEST/checksums.sha256.tmp"; then
    mv "$DEST/checksums.sha256.tmp" "$DEST/checksums.sha256"
  else
    echo "WARN: failed to download checksums.sha256; proceeding without SHA-256 verification" >&2
    VERIFY=0
  fi
fi

# ---- Download + unpack each model ----
for m in "${SELECTED[@]}"; do
  echo ""
  echo "==> ${m}"

  if [[ -f "$DEST/$m/manifest.toml" && $FORCE -eq 0 ]]; then
    echo "  already present (manifest.toml exists); skipping. Use --force to re-download."
    continue
  fi

  ZIP_URL="$ZENODO_BASE/${m}.zip"
  ZIP_PATH="$DEST/${m}.zip"

  echo "  downloading from ${ZIP_URL} ..."
  curl -fL --progress-bar -o "$ZIP_PATH" "$ZIP_URL"

  echo "  unpacking..."
  # -o = overwrite without prompting (idempotent re-runs)
  # -q = quiet (one line per zip otherwise floods)
  unzip -q -o "$ZIP_PATH" -d "$DEST"

  rm "$ZIP_PATH"

  # Per-model SHA-256 verification (against checksums.sha256 entries).
  if [[ $VERIFY -eq 1 ]]; then
    # checksums.sha256 lines look like:
    #   <sha256>  <model_id>/model.onnx
    #   <sha256>  <model_id>/1/model.onnx
    expected=$(grep -E "  ${m}/" "$DEST/checksums.sha256" || true)
    if [[ -z "$expected" ]]; then
      echo "  WARN: no checksum entry found for ${m} in checksums.sha256" >&2
    else
      if (cd "$DEST" && echo "$expected" | sha256sum -c --quiet); then
        echo "  [OK] SHA-256 verified"
      else
        echo "  [FAIL] SHA-256 mismatch — model file is corrupt or tampered" >&2
        exit 1
      fi
    fi
  fi
done

# ---- Summary ----
echo ""
echo "======================================================================"
echo "Downloaded ${#SELECTED[@]} model(s) to: $(realpath "$DEST")"
echo ""
echo "Load with sparrow-engine:"
echo "  export SPARROW_ENGINE_MODEL_DIR=$(realpath "$DEST")"
echo "  spe list-models"
echo "  spe detect --model MDV6-yolov10-e --image /path/to/image.jpg"
echo ""
echo "If you use these models, please cite:"
echo "  Zenodo DOI: ${ZENODO_DOI}"
echo "  URL:        https://doi.org/${ZENODO_DOI}"
echo ""
echo "Per-model LICENSE.md inside each ${DEST}/<model_id>/ directory describes"
echo "the upstream license terms (mix of AGPL-3.0, CC-BY-NC-SA, Apache, MIT)."
echo "======================================================================"
