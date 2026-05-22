"""Manifest creation and inspection for pw-tools.

The manifest format is TOML (decided in Design Phase Round 1 — Cargo.toml
analogy, human-editable, no YAML ambiguity, stdlib tomllib in Python 3.11+).

See tools/pw_tools/manifest_schema.md for the full field reference and two
complete examples (MegaDetector v6, BirdNET v2.4).
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

# Python 3.11+ has tomllib in stdlib; tomli is the backport for 3.10.
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef,import-not-found]

# tomli_w / tomllib only reads; we need a TOML writer.
try:
    import tomli_w  # type: ignore[import]
except ImportError:
    tomli_w = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# ONNX metadata extraction
# ---------------------------------------------------------------------------


def extract_onnx_metadata(onnx_path: Path) -> dict[str, Any]:
    """Return a dict of metadata extracted from an ONNX file.

    Fields returned:
        opset_version: int
        ir_version: int
        producer_name: str
        producer_version: str
        inputs: list of {name, shape, dtype}
        outputs: list of {name, shape, dtype}
        custom_metadata: dict[str, str]  (from onnx.ModelProto.metadata_props)
        file_size_bytes: int
    """
    import onnx
    import onnxruntime as ort

    model_proto = onnx.load(str(onnx_path))

    # Opset version (take the first entry — typically the default onnx domain)
    opset_version = 0
    for entry in model_proto.opset_import:
        if entry.domain in ("", "ai.onnx"):
            opset_version = entry.version
            break

    # Custom metadata_props
    custom_metadata = {prop.key: prop.value for prop in model_proto.metadata_props}

    # Use ORT session for input/output shapes (handles dynamic dims cleanly)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])

    def _shape(info: Any) -> list[int | str]:
        shape: list[int | str] = []
        for dim in info.shape:
            if isinstance(dim, int):
                shape.append(dim if dim != 0 else -1)
            elif hasattr(dim, "dim_param") and dim.dim_param:
                # Symbolic / dynamic dimension
                shape.append(-1)
            elif hasattr(dim, "dim_value"):
                shape.append(dim.dim_value if dim.dim_value != 0 else -1)
            else:
                shape.append(-1)
        return shape

    inputs = [
        {"name": inp.name, "shape": _shape(inp), "dtype": inp.type}
        for inp in session.get_inputs()
    ]
    outputs = [
        {"name": out.name, "shape": _shape(out), "dtype": out.type}
        for out in session.get_outputs()
    ]

    return {
        "opset_version": opset_version,
        "ir_version": model_proto.ir_version,
        "producer_name": model_proto.producer_name,
        "producer_version": model_proto.producer_version,
        "inputs": inputs,
        "outputs": outputs,
        "custom_metadata": custom_metadata,
        "file_size_bytes": onnx_path.stat().st_size,
    }


def compute_sha256(path: Path) -> str:
    """Return the lowercase hex SHA-256 digest of *path*."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Interactive prompts
# ---------------------------------------------------------------------------

# Preprocessing type → required fields (name, default)
_PREPROCESSING_FIELDS: dict[str, list[tuple[str, Any]]] = {
    "image_letterbox": [
        ("resize", [640, 640]),
        ("scale", 255.0),
        ("color_space", "RGB"),
        ("pad_value", 114),
    ],
    "image_resize_normalize": [
        ("resize", [640, 640]),
        ("mean", [0.485, 0.456, 0.406]),
        ("std", [0.229, 0.224, 0.225]),
        ("scale", 255.0),
        ("color_space", "RGB"),
    ],
    "audio_mel_spectrogram": [
        ("sample_rate", 48000),
        ("segment_duration", 3.0),
        ("n_fft", 1024),
        ("hop_length", 512),
        ("n_mels", 128),
    ],
    "herdnet_patches": [
        ("resize", [512, 512]),
        ("overlap", 128),
        ("scale", 255.0),
        ("mean", [0.485, 0.456, 0.406]),
        ("std", [0.229, 0.224, 0.225]),
    ],
}

_POSTPROCESSING_FIELDS: dict[str, list[tuple[str, Any]]] = {
    "yolo_nms": [
        ("default_conf_threshold", 0.2),
        ("iou_threshold", 0.45),
        ("max_detections", 300),
    ],
    "rtdetr_topk": [
        ("topk", 300),
        ("default_conf_threshold", 0.2),
    ],
    "softmax": [
        ("default_conf_threshold", 0.1),
    ],
    "sigmoid": [
        ("default_conf_threshold", 0.5),
    ],
    "audio_softmax": [
        ("default_conf_threshold", 0.1),
        ("output_names", ["logits"]),
    ],
    "herdnet_stitch_lmds": [
        ("default_conf_threshold", 0.5),
        ("lmds_threshold", 100),
    ],
}

_TASK_TO_PREPROCESSING = {
    "detection": "image_letterbox",
    "classification": "image_resize_normalize",
    "audio_classification": "audio_mel_spectrogram",
    "point_detection": "herdnet_patches",
}

_TASK_TO_POSTPROCESSING = {
    "detection": "yolo_nms",
    "classification": "softmax",
    "audio_classification": "audio_softmax",
    "point_detection": "herdnet_stitch_lmds",
}


def _prompt(label: str, default: Any) -> Any:
    """Prompt the user, returning *default* on empty input."""
    import rich
    from rich.prompt import Prompt

    if isinstance(default, list):
        default_str = ",".join(str(x) for x in default)
        raw = Prompt.ask(f"  {label}", default=default_str)
        # Parse as list of ints or floats
        parts = [x.strip() for x in raw.split(",")]
        try:
            return [int(x) for x in parts]
        except ValueError:
            try:
                return [float(x) for x in parts]
            except ValueError:
                return parts
    elif isinstance(default, bool):
        from rich.prompt import Confirm

        return Confirm.ask(f"  {label}", default=default)
    elif isinstance(default, float):
        raw = Prompt.ask(f"  {label}", default=str(default))
        return float(raw)
    elif isinstance(default, int):
        raw = Prompt.ask(f"  {label}", default=str(default))
        return int(raw)
    else:
        return Prompt.ask(f"  {label}", default=str(default))


def _interactive_preprocessing(task: str, onnx_meta: dict[str, Any]) -> dict[str, Any]:
    """Interactively collect preprocessing parameters."""
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console()
    console.print("\n[bold]Preprocessing configuration[/]")

    default_type = _TASK_TO_PREPROCESSING.get(task, "image_resize_normalize")
    choices = list(_PREPROCESSING_FIELDS.keys())
    console.print(f"  Available types: {', '.join(choices)}")
    pp_type = Prompt.ask("  type", default=default_type)

    config: dict[str, Any] = {"type": pp_type}
    fields = _PREPROCESSING_FIELDS.get(pp_type, [])
    for name, default in fields:
        config[name] = _prompt(name, default)
    return config


def _interactive_postprocessing(task: str) -> dict[str, Any]:
    """Interactively collect postprocessing parameters."""
    from rich.console import Console
    from rich.prompt import Prompt

    console = Console()
    console.print("\n[bold]Postprocessing configuration[/]")

    default_type = _TASK_TO_POSTPROCESSING.get(task, "softmax")
    choices = list(_POSTPROCESSING_FIELDS.keys())
    console.print(f"  Available types: {', '.join(choices)}")
    pp_type = Prompt.ask("  type", default=default_type)

    config: dict[str, Any] = {"type": pp_type}
    fields = _POSTPROCESSING_FIELDS.get(pp_type, [])
    for name, default in fields:
        config[name] = _prompt(name, default)
    return config


# ---------------------------------------------------------------------------
# Core manifest builder
# ---------------------------------------------------------------------------


def _build_manifest(
    *,
    onnx_path: Path,
    name: str,
    task: str,
    labels_path: str | None,
    version: str,
    license_id: str,
    description: str,
    tags: list[str],
    preprocessing: dict[str, Any],
    postprocessing: dict[str, Any],
    onnx_meta: dict[str, Any],
    sha256: str,
) -> dict[str, Any]:
    """Return the manifest as a Python dict ready for TOML serialisation."""
    # Derive input_format from shape length
    shape = onnx_meta["inputs"][0]["shape"] if onnx_meta["inputs"] else [-1]
    if len(shape) == 4:
        input_format = "NCHW"
    elif len(shape) == 2:
        input_format = "NL"
    elif len(shape) == 3:
        input_format = "NHW"
    else:
        input_format = "unknown"

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        # Flat model section — the TOML layout mirrors what the runtime reads
        "model": {
            "id": name,
            "name": name,
            "type": task,
            "format": "onnx",
            "version": version,
            "license": license_id,
            "description": description,
            "tags": tags,
            # File metadata
            "file": onnx_path.name,
            "sha256": sha256,
            "file_size_bytes": onnx_meta["file_size_bytes"],
            # Shape metadata (from ONNX graph)
            "input_format": input_format,
            "input_shape": shape,
            "opset_version": onnx_meta["opset_version"],
            # Preprocessing / postprocessing
            "preprocessing": preprocessing,
            "postprocessing": postprocessing,
        },
    }

    # Labels
    if labels_path:
        lp = Path(labels_path)
        lines = lp.read_text().splitlines()
        labels = {str(i): lbl.strip() for i, lbl in enumerate(lines) if lbl.strip()}
        manifest["model"]["labels"] = labels
        manifest["model"]["labels_file"] = lp.name

    # ONNX producer provenance (informational)
    manifest["model"]["provenance"] = {
        "producer_name": onnx_meta["producer_name"],
        "producer_version": onnx_meta["producer_version"],
        "ir_version": onnx_meta["ir_version"],
    }

    return manifest


# ---------------------------------------------------------------------------
# TOML write helper
# ---------------------------------------------------------------------------


def _write_toml(data: dict[str, Any], dst: Path) -> None:
    """Write *data* as TOML to *dst*.

    Falls back to a hand-rolled minimal serialiser if tomli_w is not
    installed, rather than failing hard.
    """
    if tomli_w is not None:
        dst.write_bytes(tomli_w.dumps(data).encode())
        return

    # Minimal fallback — sufficient for the simple nested dicts we produce.
    lines: list[str] = []

    def _write_section(d: dict[str, Any], prefix: str = "") -> None:
        scalar_keys = [k for k, v in d.items() if not isinstance(v, dict)]
        table_keys = [k for k, v in d.items() if isinstance(v, dict)]

        for k in scalar_keys:
            v = d[k]
            if isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, (int, float)):
                lines.append(f"{k} = {v}")
            elif isinstance(v, list):
                items = ", ".join(
                    f'"{x}"' if isinstance(x, str) else str(x) for x in v
                )
                lines.append(f"{k} = [{items}]")
            elif isinstance(v, str):
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"')
            elif v is None:
                pass  # skip None values
            else:
                lines.append(f'{k} = "{v}"')

        for k in table_keys:
            section = f"{prefix}.{k}".lstrip(".") if prefix else k
            lines.append(f"\n[{section}]")
            _write_section(d[k], section)

    # Top-level scalars first
    for k, v in data.items():
        if not isinstance(v, dict):
            if isinstance(v, str):
                escaped = v.replace("\\", "\\\\").replace('"', '\\"')
                lines.append(f'{k} = "{escaped}"')
            else:
                lines.append(f"{k} = {v}")

    # Then sections
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"\n[{k}]")
            _write_section(v, k)

    dst.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def create_manifest(
    onnx_path: str,
    name: str,
    task: str,
    labels_path: str | None,
    config_path: str | None,
    output_path: str,
    version: str,
    license_id: str,
    description: str,
    tags: list[str],
    interactive: bool,
) -> Path:
    """Create a TOML manifest for *onnx_path* and write it to *output_path*."""
    from rich.console import Console

    console = Console()
    onnx = Path(onnx_path)

    if not onnx.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx}")

    console.print(f"[bold cyan]Extracting ONNX metadata[/] from {onnx}")
    onnx_meta = extract_onnx_metadata(onnx)

    console.print(f"  Opset version  : {onnx_meta['opset_version']}")
    console.print(f"  IR version     : {onnx_meta['ir_version']}")
    console.print(f"  Producer       : {onnx_meta['producer_name']} {onnx_meta['producer_version']}")
    for inp in onnx_meta["inputs"]:
        console.print(f"  Input  [{inp['name']}]  shape={inp['shape']}  dtype={inp['dtype']}")
    for out in onnx_meta["outputs"]:
        console.print(f"  Output [{out['name']}]  shape={out['shape']}  dtype={out['dtype']}")

    console.print(f"\n[bold cyan]Computing SHA-256[/]")
    sha256 = compute_sha256(onnx)
    console.print(f"  {sha256}")

    # Load preprocessing / postprocessing from config file if provided
    preprocessing: dict[str, Any] | None = None
    postprocessing: dict[str, Any] | None = None

    if config_path:
        cfg_path = Path(config_path)
        if not cfg_path.exists():
            raise FileNotFoundError(f"Config file not found: {cfg_path}")
        with open(cfg_path, "rb") as fh:
            cfg = tomllib.load(fh)
        preprocessing = cfg.get("preprocessing")
        postprocessing = cfg.get("postprocessing")

    if preprocessing is None:
        if interactive:
            preprocessing = _interactive_preprocessing(task, onnx_meta)
        else:
            # Non-interactive: use defaults for the task
            pp_type = _TASK_TO_PREPROCESSING.get(task, "image_resize_normalize")
            preprocessing = {"type": pp_type}
            for field_name, default in _PREPROCESSING_FIELDS.get(pp_type, []):
                preprocessing[field_name] = default

    if postprocessing is None:
        if interactive:
            postprocessing = _interactive_postprocessing(task)
        else:
            pp_type = _TASK_TO_POSTPROCESSING.get(task, "softmax")
            postprocessing = {"type": pp_type}
            for field_name, default in _POSTPROCESSING_FIELDS.get(pp_type, []):
                postprocessing[field_name] = default

    manifest = _build_manifest(
        onnx_path=onnx,
        name=name,
        task=task,
        labels_path=labels_path,
        version=version,
        license_id=license_id,
        description=description,
        tags=tags,
        preprocessing=preprocessing,
        postprocessing=postprocessing,
        onnx_meta=onnx_meta,
        sha256=sha256,
    )

    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_toml(manifest, dst)

    console.print(f"\n[bold green]Manifest written[/] → {dst}")
    return dst


def inspect_manifest(manifest_path: str) -> None:
    """Load and pretty-print a TOML manifest."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    path = Path(manifest_path)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")

    with open(path, "rb") as fh:
        data = tomllib.load(fh)

    model = data.get("model", {})
    table = Table(title=f"Manifest: {path.name}", show_lines=True)
    table.add_column("Field", style="cyan")
    table.add_column("Value")

    for key, value in model.items():
        if isinstance(value, dict):
            table.add_row(key, str(value))
        elif isinstance(value, list) and all(isinstance(x, str) for x in value):
            table.add_row(key, ", ".join(str(v) for v in value))
        else:
            table.add_row(key, str(value))

    console.print(table)

    # Verify the file still exists and SHA256 matches
    onnx_candidate = path.parent / model.get("file", "")
    if onnx_candidate.exists():
        console.print(f"\n[bold cyan]Verifying SHA-256[/] of {onnx_candidate.name}")
        actual = compute_sha256(onnx_candidate)
        expected = model.get("sha256", "")
        if actual == expected:
            console.print(f"[green]SHA-256 OK[/]  {actual}")
        else:
            console.print(f"[red]SHA-256 MISMATCH[/]\n  Expected: {expected}\n  Actual:   {actual}")
    else:
        console.print(f"[yellow]ONNX file not found at {onnx_candidate} — skipping SHA-256 check[/]")
