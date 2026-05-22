"""ONNX conversion logic for pw-tools.

Supports:
  - PyTorch nn.Module via torch.onnx.export(dynamo=True)  [PyTorch 2.5+, recommended]
  - PyTorch nn.Module via torch.onnx.export legacy path    [PyTorch <2.5 fallback]
  - TorchScript (torch.jit.ScriptModule)                   [auto-detected]

Post-conversion:
  - ORT graph optimisation (ORT_ENABLE_ALL saved to disk)
  - Optional FP16 conversion via onnxconverter-common
"""

from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path
from typing import Sequence

import torch
import torch.nn as nn

logger = logging.getLogger("pw-tools.convert")


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def convert_to_onnx(
    source_path: str,
    output_path: str,
    opset: int = 17,
    dynamic_batch: bool = False,
    input_shape: tuple[int, ...] = (1, 3, 640, 640),
    input_names: list[str] | None = None,
    output_names: list[str] | None = None,
    use_dynamo: bool = True,
    optimize: bool = True,
    half: bool = False,
) -> Path:
    """Convert *source_path* to ONNX and write to *output_path*.

    Returns the final output path (may differ from *output_path* when the ORT
    optimisation pass writes to a temp file first).

    Raises:
        FileNotFoundError: if *source_path* does not exist.
        RuntimeError: if conversion or optimisation fails.
    """
    import rich
    from rich.console import Console

    console = Console()
    src = Path(source_path)
    dst = Path(output_path)

    if not src.exists():
        raise FileNotFoundError(f"Source model not found: {src}")

    dst.parent.mkdir(parents=True, exist_ok=True)

    input_names = input_names or ["images"]
    output_names = output_names or ["output0"]

    console.print(f"[bold cyan]Loading model[/] from {src}")
    model, is_torchscript = _load_model(src)

    console.print(f"[bold cyan]Exporting to ONNX[/]  opset={opset}  dynamo={use_dynamo and not is_torchscript}")

    # TorchScript always uses the legacy exporter (dynamo path doesn't support it)
    if is_torchscript:
        _export_torchscript(
            model=model,  # type: ignore[arg-type]
            dst=dst,
            opset=opset,
            dynamic_batch=dynamic_batch,
            input_shape=input_shape,
            input_names=input_names,
            output_names=output_names,
        )
    elif use_dynamo and _pytorch_supports_dynamo():
        _export_dynamo(
            model=model,  # type: ignore[arg-type]
            dst=dst,
            opset=opset,
            dynamic_batch=dynamic_batch,
            input_shape=input_shape,
            input_names=input_names,
            output_names=output_names,
        )
    else:
        if use_dynamo:
            console.print(
                "[yellow]WARNING[/] dynamo=True requires PyTorch>=2.5; "
                f"you have {torch.__version__}. Falling back to legacy exporter."
            )
        _export_legacy(
            model=model,  # type: ignore[arg-type]
            dst=dst,
            opset=opset,
            dynamic_batch=dynamic_batch,
            input_shape=input_shape,
            input_names=input_names,
            output_names=output_names,
        )

    console.print(f"[green]ONNX export complete[/] → {dst}")

    # Validate the exported model with onnx.checker
    _check_onnx(dst)
    console.print("[green]onnx.checker passed[/]")

    if optimize:
        console.print("[bold cyan]Running ORT graph optimisation[/]")
        dst = _ort_optimize(dst)
        console.print(f"[green]Optimisation complete[/] → {dst}")

    if half:
        console.print("[bold cyan]Converting to FP16[/]")
        dst = _convert_fp16(dst)
        console.print(f"[green]FP16 conversion complete[/] → {dst}")

    file_size_mb = dst.stat().st_size / (1024 * 1024)
    console.print(f"[bold green]Done.[/]  File size: {file_size_mb:.1f} MB")
    return dst


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def _load_model(src: Path) -> tuple[nn.Module | torch.jit.ScriptModule, bool]:
    """Return (model, is_torchscript).

    Heuristic: try torch.jit.load first; if it fails with a RuntimeError
    (wrong format), try torch.load with weights_only=False as a plain
    checkpoint.  Models loaded via torch.load are assumed to be
    nn.Module instances already stored in eval mode.
    """
    # Try TorchScript first
    try:
        model = torch.jit.load(str(src), map_location="cpu")
        model.eval()
        logger.info("Loaded as TorchScript model")
        return model, True
    except Exception:
        pass

    # Try plain checkpoint / serialised nn.Module
    try:
        obj = torch.load(str(src), map_location="cpu", weights_only=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to load model from {src}: {exc}") from exc

    if isinstance(obj, nn.Module):
        obj.eval()
        return obj, False

    if isinstance(obj, dict):
        # Common checkpoint formats: {"model": ..., "state_dict": ...}
        for key in ("model", "module", "net"):
            if key in obj and isinstance(obj[key], nn.Module):
                obj[key].eval()
                return obj[key], False
        raise RuntimeError(
            f"Loaded a dict from {src} but found no nn.Module at keys "
            f"'model'/'module'/'net'. Keys found: {list(obj.keys())}. "
            "Please load the model manually and pass an nn.Module."
        )

    raise RuntimeError(
        f"Unsupported object type loaded from {src}: {type(obj)}. "
        "Expected nn.Module, TorchScript, or checkpoint dict."
    )


# ---------------------------------------------------------------------------
# Dynamo export  (PyTorch 2.5+)
# ---------------------------------------------------------------------------


def _pytorch_supports_dynamo() -> bool:
    """Return True if torch.onnx.export supports the dynamo=True kwarg."""
    major, minor, *_ = (int(x) for x in torch.__version__.split(".")[:2])
    return (major, minor) >= (2, 5)


def _export_dynamo(
    model: nn.Module,
    dst: Path,
    opset: int,
    dynamic_batch: bool,
    input_shape: tuple[int, ...],
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Export using the torch.export-based ONNX exporter (PyTorch 2.5+).

    The dynamo path returns an ExportedProgram-like object that must be
    saved separately.  Dynamic shapes are specified via torch.export
    Dim objects.

    Notes on the API (PyTorch 2.5–2.9):
      - torch.onnx.export(..., dynamo=True) returns an ONNXProgram.
      - ONNXProgram.save(path) writes the .onnx file.
      - As of 2.9, dynamo=True is the default.
      - As of 2.7, optimize=True is the default inside the dynamo path.
    """
    example_inputs = (torch.zeros(*input_shape),)

    dynamic_shapes: dict | None = None
    if dynamic_batch:
        # torch.export dynamic shapes API: dict keyed by input name,
        # value is a dict mapping dimension index to a Dim.
        batch_dim = torch.export.Dim("batch", min=1, max=2048)
        dynamic_shapes = {input_names[0]: {0: batch_dim}}

    # torch.onnx.export with dynamo=True returns an ONNXProgram object
    onnx_program = torch.onnx.export(
        model,
        example_inputs,
        dynamo=True,
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes=dynamic_shapes,
        # optimize=True is the default from PT 2.7+; explicit here for clarity
        optimize=True,
    )
    onnx_program.save(str(dst))


# ---------------------------------------------------------------------------
# Legacy export  (pre-2.5 fallback)
# ---------------------------------------------------------------------------


def _export_legacy(
    model: nn.Module,
    dst: Path,
    opset: int,
    dynamic_batch: bool,
    input_shape: tuple[int, ...],
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Export using the traditional TorchScript-based ONNX exporter."""
    example_inputs = torch.zeros(*input_shape)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    torch.onnx.export(
        model,
        (example_inputs,),
        str(dst),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# TorchScript export
# ---------------------------------------------------------------------------


def _export_torchscript(
    model: torch.jit.ScriptModule,
    dst: Path,
    opset: int,
    dynamic_batch: bool,
    input_shape: tuple[int, ...],
    input_names: list[str],
    output_names: list[str],
) -> None:
    """Export a TorchScript model using the legacy exporter (dynamo doesn't
    support ScriptModule)."""
    example_inputs = torch.zeros(*input_shape)

    dynamic_axes: dict[str, dict[int, str]] | None = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}

    torch.onnx.export(
        model,
        (example_inputs,),
        str(dst),
        opset_version=opset,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        do_constant_folding=True,
        export_params=True,
        verbose=False,
    )


# ---------------------------------------------------------------------------
# Post-export passes
# ---------------------------------------------------------------------------


def _check_onnx(path: Path) -> None:
    """Run onnx.checker.check_model and raise RuntimeError on failure."""
    import onnx

    model = onnx.load(str(path))
    onnx.checker.check_model(model)


def _ort_optimize(src: Path) -> Path:
    """Run ORT graph optimisation at ORT_ENABLE_ALL level.

    Writes the optimised model to <stem>_optimized.onnx in the same
    directory and returns the new path.
    """
    import onnxruntime as ort

    dst = src.parent / f"{src.stem}_optimized.onnx"

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.optimized_model_filepath = str(dst)
    # Loading the session triggers the optimisation and writes the file.
    ort.InferenceSession(str(src), so, providers=["CPUExecutionProvider"])

    # Replace the original with the optimised version
    shutil.move(str(dst), str(src))
    return src


def _convert_fp16(src: Path) -> Path:
    """Convert an FP32 ONNX model to FP16 using onnxconverter-common."""
    try:
        from onnxconverter_common import float16  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "FP16 conversion requires onnxconverter-common. "
            "Install it with: pip install onnxconverter-common"
        ) from exc

    import onnx

    fp32_model = onnx.load(str(src))
    fp16_model = float16.convert_float_to_float16(fp32_model, keep_io_types=True)
    dst = src.parent / f"{src.stem}_fp16.onnx"
    onnx.save(fp16_model, str(dst))
    return dst
