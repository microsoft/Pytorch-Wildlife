#!/usr/bin/env python3
"""Download sparrow-engine model files from known URLs.

Usage:
    python download_models.py --model megadetector-v6 --output-dir ./models
    python download_models.py --all --output-dir ./models
    python download_models.py --list
"""

import argparse
import hashlib
import os
import sys
import urllib.request

# Model registry: id -> metadata.
# URLs are placeholders until models are hosted (e.g., HuggingFace or GitHub Releases).
# SHA256 checksums are placeholders (set to None) — fill in when URLs are finalized.
MODELS = {
    "megadetector-v6-yolov10e": {
        "description": "MegaDetector v6 (YOLOv10-E) — camera trap animal/person/vehicle detector, 1280x1280",
        "files": {
            "models_MDV6-yolov10-e_model.onnx": {
                "url": "https://PLACEHOLDER/models_MDV6-yolov10-e_model.onnx",
                "sha256": None,
            },
            "mdv6_manifest.toml": {
                "url": "https://PLACEHOLDER/mdv6_manifest.toml",
                "sha256": None,
            },
            "models_MDV6-yolov10-e_labels.txt": {
                "url": "https://PLACEHOLDER/models_MDV6-yolov10-e_labels.txt",
                "sha256": None,
            },
        },
    },
    "deepfaune-yolo8s": {
        "description": "DeepFaune (YOLOv8s) — European wildlife detector, 960x960",
        "files": {
            "models_deepfaune-yolo8s_model.onnx": {
                "url": "https://PLACEHOLDER/models_deepfaune-yolo8s_model.onnx",
                "sha256": None,
            },
            "deepfaune_manifest.toml": {
                "url": "https://PLACEHOLDER/deepfaune_manifest.toml",
                "sha256": None,
            },
            "models_deepfaune-yolo8s_labels.txt": {
                "url": "https://PLACEHOLDER/models_deepfaune-yolo8s_labels.txt",
                "sha256": None,
            },
        },
    },
    "herdnet-general-2022": {
        "description": "HerdNet General 2022 — heatmap-based animal density estimator, 512x512 tiled",
        "files": {
            "models_HerdNet_General_Dataset_2022_model.onnx": {
                "url": "https://PLACEHOLDER/models_HerdNet_General_Dataset_2022_model.onnx",
                "sha256": None,
            },
            "herdnet_manifest.toml": {
                "url": "https://PLACEHOLDER/herdnet_manifest.toml",
                "sha256": None,
            },
            "models_HerdNet_General_Dataset_2022_labels.txt": {
                "url": "https://PLACEHOLDER/models_HerdNet_General_Dataset_2022_labels.txt",
                "sha256": None,
            },
        },
    },
    "speciesnet-crop": {
        "description": "SpeciesNet Crop — species classification from detection crops, 480x480",
        "files": {
            "models_classification_SpeciesNet-Crop_model.onnx": {
                "url": "https://PLACEHOLDER/models_classification_SpeciesNet-Crop_model.onnx",
                "sha256": None,
            },
            "speciesnet_manifest.toml": {
                "url": "https://PLACEHOLDER/speciesnet_manifest.toml",
                "sha256": None,
            },
            "models_classification_SpeciesNet-Crop_labels.txt": {
                "url": "https://PLACEHOLDER/models_classification_SpeciesNet-Crop_labels.txt",
                "sha256": None,
            },
        },
    },
    "md-audiobirds-v1": {
        "description": "MD AudioBirds V1 — binary bird detector from audio (mel spectrogram), 48kHz",
        "files": {
            "MD_AudioBirds_V1.onnx": {
                "url": "https://PLACEHOLDER/MD_AudioBirds_V1.onnx",
                "sha256": None,
            },
            "audiobirds_manifest.toml": {
                "url": "https://PLACEHOLDER/audiobirds_manifest.toml",
                "sha256": None,
            },
            "audio_birds_labels.txt": {
                "url": "https://PLACEHOLDER/audio_birds_labels.txt",
                "sha256": None,
            },
        },
    },
    "owl-t": {
        "description": "OWL-T — single-output heatmap detector, 512x512 tiled with overlap",
        "files": {
            "models_OWL_model.onnx": {
                "url": "https://PLACEHOLDER/models_OWL_model.onnx",
                "sha256": None,
            },
            "owl_manifest.toml": {
                "url": "https://PLACEHOLDER/owl_manifest.toml",
                "sha256": None,
            },
            "models_OWL_labels.txt": {
                "url": "https://PLACEHOLDER/models_OWL_labels.txt",
                "sha256": None,
            },
        },
    },
}


def sha256_file(path: str) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: str, expected_sha256: str | None) -> bool:
    """Download a single file with progress display. Returns True on success."""
    if os.path.exists(dest) and expected_sha256 is not None:
        actual = sha256_file(dest)
        if actual == expected_sha256:
            print(f"  [skip] {os.path.basename(dest)} — already exists, checksum OK")
            return True

    if "PLACEHOLDER" in url:
        print(f"  [skip] {os.path.basename(dest)} — URL not yet configured")
        return False

    print(f"  [download] {os.path.basename(dest)} ...")
    tmp_dest = dest + ".part"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "sparrow-engine-download/1.0"})
        with urllib.request.urlopen(req) as resp:
            total = resp.headers.get("Content-Length")
            total = int(total) if total else None
            downloaded = 0

            with open(tmp_dest, "wb") as out:
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    out.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded * 100 // total
                        mb = downloaded / (1024 * 1024)
                        total_mb = total / (1024 * 1024)
                        print(
                            f"\r    {mb:.1f}/{total_mb:.1f} MB ({pct}%)",
                            end="",
                            flush=True,
                        )
                    else:
                        mb = downloaded / (1024 * 1024)
                        print(f"\r    {mb:.1f} MB", end="", flush=True)
            print()  # newline after progress

        # Verify checksum if known.
        if expected_sha256 is not None:
            actual = sha256_file(tmp_dest)
            if actual != expected_sha256:
                os.unlink(tmp_dest)
                print(f"  [ERROR] checksum mismatch for {os.path.basename(dest)}")
                print(f"    expected: {expected_sha256}")
                print(f"    actual:   {actual}")
                return False

        os.replace(tmp_dest, dest)
        print(f"  [ok] {os.path.basename(dest)}")
        return True

    except Exception as e:
        if os.path.exists(tmp_dest):
            os.unlink(tmp_dest)
        print(f"  [ERROR] {os.path.basename(dest)}: {e}")
        return False


def list_models() -> None:
    """Print available models."""
    print("Available models:\n")
    for model_id, info in MODELS.items():
        files = ", ".join(info["files"].keys())
        print(f"  {model_id}")
        print(f"    {info['description']}")
        print(f"    Files: {files}")
        print()


def download_model(model_id: str, output_dir: str) -> bool:
    """Download all files for a model. Returns True if all succeeded."""
    info = MODELS[model_id]
    model_dir = os.path.join(output_dir, model_id)
    os.makedirs(model_dir, exist_ok=True)

    print(f"Model: {model_id}")
    print(f"  -> {model_dir}")

    all_ok = True
    for filename, file_info in info["files"].items():
        dest = os.path.join(model_dir, filename)
        ok = download_file(file_info["url"], dest, file_info["sha256"])
        if not ok:
            all_ok = False

    return all_ok


def resolve_model_id(name: str) -> str | None:
    """Resolve a model name to its full ID, supporting partial matches."""
    # Exact match.
    if name in MODELS:
        return name
    # Prefix match.
    matches = [mid for mid in MODELS if mid.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    # Substring match.
    matches = [mid for mid in MODELS if name in mid]
    if len(matches) == 1:
        return matches[0]
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download sparrow-engine model files.",
        epilog="URLs are placeholders until models are hosted publicly.",
    )
    parser.add_argument("--list", action="store_true", help="List available models")
    parser.add_argument("--model", type=str, help="Model ID to download (supports partial match)")
    parser.add_argument("--all", action="store_true", help="Download all models")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.expanduser("~/.sparrow-engine/models"),
        help="Target directory (default: ~/.sparrow-engine/models)",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return 0

    if not args.model and not args.all:
        parser.print_help()
        return 1

    if args.model and args.all:
        print("Error: --model and --all are mutually exclusive", file=sys.stderr)
        return 1

    if args.model:
        model_id = resolve_model_id(args.model)
        if model_id is None:
            print(f"Error: unknown model '{args.model}'", file=sys.stderr)
            print(f"Available: {', '.join(MODELS.keys())}", file=sys.stderr)
            return 1
        ok = download_model(model_id, args.output_dir)
        return 0 if ok else 1

    # --all
    print(f"Downloading all {len(MODELS)} models to {args.output_dir}\n")
    all_ok = True
    for model_id in MODELS:
        ok = download_model(model_id, args.output_dir)
        if not ok:
            all_ok = False
        print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
