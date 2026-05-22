# PytorchWildlife Model Manifest — TOML Format Reference

## Design rationale

TOML (Tom's Obvious Minimal Language) was chosen over JSON and YAML for the
following reasons:
- Human-editable, comments supported (unlike JSON)
- No indentation-sensitivity ambiguity (unlike YAML)
- Familiar to Rust/Python ecosystem users (Cargo.toml, pyproject.toml analogy)
- stdlib support in Python 3.11+ (tomllib); backport via tomli for 3.10

## File layout

```
<model-id>/
  megadetector-v6.onnx     # binary model
  megadetector-v6.toml     # manifest (this file)
  labels.txt               # optional label file
```

The manifest TOML lives beside the ONNX file.  The runtime loads the manifest
from the HuggingFace Hub at startup (with a bundled fallback in the package).

---

## Top-level fields

```toml
schema_version = "1.0"
# String. Major version controls runtime compatibility check.
# Library raises ManifestError if major != "1".
```

---

## [model] section — all fields

```toml
[model]

# --- Identity ---
id          = "megadetector-v6"   # string, kebab-case, unique in registry
name        = "MegaDetector v6"   # string, human-readable display name
type        = "detection"         # string, one of:
                                  #   "detection"
                                  #   "classification"
                                  #   "audio_classification"
                                  #   "point_detection"
format      = "onnx"              # string, always "onnx" for v1 manifests
version     = "6.0.0"             # string, semver preferred
license     = "MIT"               # string, SPDX identifier
description = "General-purpose camera trap animal detector (RT-DETR)"
tags        = ["camera-trap", "general", "detection"]  # list[str]

# --- File references ---
file              = "megadetector_v6.onnx"   # string, filename relative to manifest
sha256            = "abc123..."              # string, lowercase hex SHA-256
file_size_bytes   = 85000000                # int, used for progress bars
labels_file       = "labels.txt"            # string, optional

# For multi-file models (e.g. Perch embedding + classifier),
# use [model.files.<role>] tables instead of [model].file.
# See the Perch example below.

# --- ONNX graph metadata (extracted by pw-tools manifest create) ---
input_format    = "NCHW"           # string: "NCHW" | "NHW" | "NL" | "NCL"
input_shape     = [-1, 3, 640, 640] # list[int], -1 = dynamic dimension
opset_version   = 17               # int

# --- Preprocessing (required) ---
[model.preprocessing]
type        = "image_letterbox"    # string, one of:
                                   #   "image_letterbox"
                                   #   "image_resize_normalize"
                                   #   "audio_mel_spectrogram"
                                   #   "herdnet_patches"
resize      = [640, 640]           # list[int], [H, W]
scale       = 255.0                # float, divide pixel values by this
color_space = "RGB"                # string: "RGB" | "BGR" | "L"
pad_value   = 114                  # int, letterbox pad value (pre-scale)

# image_resize_normalize specific:
# mean = [0.485, 0.456, 0.406]
# std  = [0.229, 0.224, 0.225]

# audio_mel_spectrogram specific:
# sample_rate      = 48000
# segment_duration = 3.0   # seconds
# n_fft            = 1024
# hop_length       = 512
# n_mels           = 128

# herdnet_patches specific:
# overlap = 128

# --- Postprocessing (required) ---
[model.postprocessing]
type                    = "yolo_nms"  # string, one of:
                                      #   "yolo_nms"
                                      #   "rtdetr_topk"
                                      #   "softmax"
                                      #   "sigmoid"
                                      #   "audio_softmax"
                                      #   "herdnet_stitch_lmds"
default_conf_threshold  = 0.2         # float

# yolo_nms specific:
# iou_threshold  = 0.45
# max_detections = 300

# rtdetr_topk specific:
# topk = 300

# audio_softmax specific:
# output_names = ["logits"]   # list[str]

# herdnet_stitch_lmds specific:
# lmds_threshold = 100

# --- Labels (inline TOML table, optional) ---
[model.labels]
"0" = "animal"
"1" = "person"
"2" = "vehicle"

# --- Provenance (auto-filled by pw-tools, informational) ---
[model.provenance]
producer_name    = "pytorch"
producer_version = "2.5.0"
ir_version       = 10
```

---

## Complete example: MegaDetector v6 (detection)

```toml
schema_version = "1.0"

[model]
id              = "megadetector-v6"
name            = "MegaDetector v6"
type            = "detection"
format          = "onnx"
version         = "6.0.0"
license         = "MIT"
description     = "General-purpose camera trap animal detector trained with RT-DETR. Detects animals, persons, and vehicles."
tags            = ["camera-trap", "detection", "megadetector", "general"]
file            = "megadetector_v6.onnx"
sha256          = "a1b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2"
file_size_bytes = 87_400_000
input_format    = "NCHW"
input_shape     = [-1, 3, 640, 640]
opset_version   = 17
labels_file     = "labels.txt"

[model.preprocessing]
type        = "image_letterbox"
resize      = [640, 640]
scale       = 255.0
color_space = "RGB"
pad_value   = 114

[model.postprocessing]
type                   = "rtdetr_topk"
topk                   = 300
default_conf_threshold = 0.2

[model.labels]
"0" = "animal"
"1" = "person"
"2" = "vehicle"

[model.provenance]
producer_name    = "pytorch"
producer_version = "2.5.1"
ir_version       = 10
```

---

## Complete example: BirdNET v2.4 (audio classification)

```toml
schema_version = "1.0"

[model]
id              = "birdnet-v2.4"
name            = "BirdNET v2.4"
type            = "audio_classification"
format          = "onnx"
version         = "2.4.0"
license         = "CC-BY-NC-SA-4.0"
description     = "Bird species identifier from audio recordings. Trained on BirdNET dataset (6,522 species). Non-commercial use only."
tags            = ["audio", "bird", "bioacoustics", "species-id"]
file            = "birdnet_v2.4.onnx"
sha256          = "b2c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3"
file_size_bytes = 14_200_000
input_format    = "NCHW"
input_shape     = [-1, 1, 128, 384]    # [batch, channels, mel_bins, time_frames]
opset_version   = 17
labels_file     = "birdnet_labels.txt"

[model.preprocessing]
type             = "audio_mel_spectrogram"
sample_rate      = 48000
segment_duration = 3.0
n_fft            = 1024
hop_length       = 278     # yields 384 time frames at 48kHz for 3s
n_mels           = 128
normalize        = true

[model.postprocessing]
type                   = "audio_softmax"
default_conf_threshold = 0.1
output_names           = ["logits"]

[model.provenance]
producer_name    = "tensorflow"
producer_version = "2.12.0"
ir_version       = 8
```

---

## Multi-file model example: Perch v2

For models with multiple ONNX files (e.g. embedding + classifier),
use `[model.files.<role>]` tables instead of the flat `file` field.

```toml
schema_version = "1.0"

[model]
id              = "perch-v2"
name            = "Perch v2"
type            = "audio_classification"
format          = "onnx"
version         = "2.0.0"
license         = "Apache-2.0"
description     = "Google Research bird species audio classifier with embedding head."
tags            = ["audio", "bird", "embedding", "bioacoustics"]
input_format    = "NL"
input_shape     = [-1, 160000]    # [batch, samples at 32kHz for 5s]
opset_version   = 17
labels_file     = "perch_labels.json"

[model.files.embedding]
filename        = "perch_v2_embedding.onnx"
sha256          = "c3d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4"
file_size_bytes = 48_000_000
role            = "embedding"

[model.files.classifier]
filename        = "perch_v2_classifier.onnx"
sha256          = "d4e5f6a7b8c9d0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a1b2c3d4e5"
file_size_bytes = 350_000_000
role            = "classifier"

[model.preprocessing]
type             = "audio_mel_spectrogram"
sample_rate      = 32000
segment_duration = 5.0

[model.postprocessing]
type                   = "audio_softmax"
default_conf_threshold = 0.1
output_names           = ["logits", "embedding"]

[model.provenance]
producer_name    = "tensorflow"
producer_version = "2.13.0"
ir_version       = 8
```

---

## Field reference summary

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `schema_version` | string | yes | Format version ("1.0") |
| `[model].id` | string | yes | Kebab-case unique identifier |
| `[model].name` | string | yes | Human-readable display name |
| `[model].type` | string | yes | Task type (detection/classification/…) |
| `[model].format` | string | yes | Always "onnx" for v1 |
| `[model].version` | string | yes | Model version (semver preferred) |
| `[model].license` | string | yes | SPDX license identifier |
| `[model].description` | string | yes | One-line description |
| `[model].tags` | list[str] | yes | Searchable tags |
| `[model].file` | string | yes* | ONNX filename (*or use `.files`) |
| `[model].sha256` | string | yes | Lowercase hex SHA-256 of ONNX file |
| `[model].file_size_bytes` | int | yes | File size for progress bars |
| `[model].input_format` | string | yes | NCHW / NHW / NL / NCL |
| `[model].input_shape` | list[int] | yes | Shape with -1 for dynamic dims |
| `[model].opset_version` | int | yes | ONNX opset version |
| `[model].labels_file` | string | no | Label file name (txt or json) |
| `[model.preprocessing].type` | string | yes | Preprocessing pipeline type |
| `[model.postprocessing].type` | string | yes | Postprocessing pipeline type |
| `[model.postprocessing].default_conf_threshold` | float | yes | Default confidence threshold |
| `[model.labels]` | table | no | Inline label map (index → name) |
| `[model.provenance]` | table | no | Auto-filled ONNX export metadata |
| `[model.files.<role>]` | table | no* | Multi-file model components |
