"""Manifest validation: load ONNX model, run test images, compare to reference outputs.

Validation steps:
  1. Parse manifest TOML and verify required fields
  2. Confirm ONNX file exists and SHA-256 matches
  3. Load ONNX via ORT and run a smoke test (random input, check output shape)
  4. (Optional) Run inference on test images and compare to .npy reference outputs
  5. Report pass/fail with per-image tolerances
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomllib  # type: ignore[no-redef]
    except ImportError:
        import tomli as tomllib  # type: ignore[no-redef,import-not-found]


# ---------------------------------------------------------------------------
# Manifest loader
# ---------------------------------------------------------------------------


def _load_manifest(manifest_path: Path) -> dict[str, Any]:
    with open(manifest_path, "rb") as fh:
        return tomllib.load(fh)


def _resolve_onnx(manifest_path: Path, model: dict[str, Any]) -> Path:
    """Return the absolute path to the ONNX file referenced by the manifest."""
    file_name = model.get("file", "")
    if not file_name:
        raise ValueError("manifest [model].file field is missing or empty")
    onnx_path = manifest_path.parent / file_name
    if not onnx_path.exists():
        raise FileNotFoundError(
            f"ONNX file '{file_name}' not found at {onnx_path}. "
            "Make sure the .onnx file is in the same directory as the manifest."
        )
    return onnx_path


# ---------------------------------------------------------------------------
# SHA-256 check
# ---------------------------------------------------------------------------


def _verify_sha256(onnx_path: Path, expected: str) -> bool:
    import hashlib

    h = hashlib.sha256()
    with open(onnx_path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest() == expected


# ---------------------------------------------------------------------------
# ORT session builder
# ---------------------------------------------------------------------------


def _build_session(onnx_path: Path) -> Any:
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    providers = ort.get_available_providers()
    return ort.InferenceSession(str(onnx_path), so, providers=providers)


# ---------------------------------------------------------------------------
# Image preprocessing (minimal, independent of the pytorchwildlife runtime)
# ---------------------------------------------------------------------------


def _preprocess_image(image_path: Path, preprocessing: dict[str, Any]) -> np.ndarray:
    """Preprocess a single image according to the manifest preprocessing config.

    Returns an NCHW float32 array with batch=1.
    """
    from PIL import Image

    pp_type = preprocessing.get("type", "image_letterbox")
    resize = preprocessing.get("resize", [640, 640])
    scale = float(preprocessing.get("scale", 255.0))
    color_space = preprocessing.get("color_space", "RGB")

    img = Image.open(image_path).convert("RGB" if color_space == "RGB" else "L")
    img = img.resize((resize[1], resize[0]), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / scale

    if pp_type == "image_resize_normalize":
        mean = np.array(preprocessing.get("mean", [0.485, 0.456, 0.406]), dtype=np.float32)
        std = np.array(preprocessing.get("std", [0.229, 0.224, 0.225]), dtype=np.float32)
        arr = (arr - mean) / std
    elif pp_type == "image_letterbox":
        pad_value = float(preprocessing.get("pad_value", 114)) / scale
        # Letterbox: keep aspect ratio, pad to square
        h_orig, w_orig = arr.shape[:2]
        h_target, w_target = resize[0], resize[1]
        scale_factor = min(h_target / h_orig, w_target / w_orig)
        new_h = int(round(h_orig * scale_factor))
        new_w = int(round(w_orig * scale_factor))
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        arr_resized = np.array(img_resized, dtype=np.float32) / scale
        canvas = np.full((h_target, w_target, 3), pad_value, dtype=np.float32)
        y0 = (h_target - new_h) // 2
        x0 = (w_target - new_w) // 2
        canvas[y0 : y0 + new_h, x0 : x0 + new_w] = arr_resized
        arr = canvas

    # HWC -> NCHW
    arr = arr.transpose(2, 0, 1)[np.newaxis]
    return np.ascontiguousarray(arr)


# ---------------------------------------------------------------------------
# Inference runner
# ---------------------------------------------------------------------------


def _run_inference(session: Any, arr: np.ndarray) -> list[np.ndarray]:
    """Run *arr* through *session* and return all outputs as a list of arrays."""
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: arr})
    return outputs


# ---------------------------------------------------------------------------
# Output comparison
# ---------------------------------------------------------------------------


def _compare_outputs(
    predicted: list[np.ndarray],
    reference: list[np.ndarray],
    atol: float,
    rtol: float,
) -> tuple[bool, list[str]]:
    """Return (all_passed, list_of_error_messages)."""
    errors: list[str] = []
    if len(predicted) != len(reference):
        errors.append(
            f"Output count mismatch: predicted {len(predicted)}, reference {len(reference)}"
        )
        return False, errors

    for i, (pred, ref) in enumerate(zip(predicted, reference)):
        if pred.shape != ref.shape:
            errors.append(f"Output[{i}] shape mismatch: {pred.shape} vs {ref.shape}")
            continue
        if not np.allclose(pred, ref, atol=atol, rtol=rtol):
            max_diff = float(np.max(np.abs(pred - ref)))
            rel_diff = float(np.max(np.abs(pred - ref) / (np.abs(ref) + 1e-9)))
            errors.append(
                f"Output[{i}] values differ: max_abs={max_diff:.6f} (atol={atol}), "
                f"max_rel={rel_diff:.6f} (rtol={rtol})"
            )

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Smoke test (no reference images needed)
# ---------------------------------------------------------------------------


def _smoke_test(session: Any, model: dict[str, Any]) -> tuple[bool, str]:
    """Run a single forward pass with random noise.  Returns (passed, message)."""
    input_shape = model.get("input_shape", [-1, 3, 640, 640])
    # Replace dynamic dims (-1) with concrete values for the smoke test
    concrete_shape = [s if s > 0 else 1 for s in input_shape]
    arr = np.random.rand(*concrete_shape).astype(np.float32)
    try:
        outputs = _run_inference(session, arr)
        out_shapes = [o.shape for o in outputs]
        return True, f"Smoke test passed. Output shapes: {out_shapes}"
    except Exception as exc:
        return False, f"Smoke test FAILED: {exc}"


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_manifest(
    manifest_path: str,
    test_images_dir: str | None,
    reference_outputs_dir: str | None,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> None:
    """Full validation pipeline.  Exits with code 1 on any failure."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    mp = Path(manifest_path)

    if not mp.exists():
        console.print(f"[red]Manifest not found: {mp}[/]")
        raise SystemExit(1)

    console.print(f"[bold cyan]Loading manifest[/] {mp}")
    data = _load_manifest(mp)
    model = data.get("model", {})

    # --- 1. Required fields ---
    required = ["id", "name", "type", "format", "version", "file", "sha256", "preprocessing"]
    missing = [f for f in required if not model.get(f)]
    if missing:
        console.print(f"[red]Manifest missing required fields: {missing}[/]")
        raise SystemExit(1)
    console.print("[green]Required fields: OK[/]")

    # --- 2. ONNX file + SHA-256 ---
    try:
        onnx_path = _resolve_onnx(mp, model)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"[red]{exc}[/]")
        raise SystemExit(1)

    console.print(f"[bold cyan]Verifying SHA-256[/]")
    expected_sha = model.get("sha256", "")
    if not expected_sha:
        console.print("[yellow]Warning: sha256 not set in manifest — skipping checksum[/]")
    elif _verify_sha256(onnx_path, expected_sha):
        console.print(f"[green]SHA-256 OK[/]")
    else:
        console.print("[red]SHA-256 MISMATCH — file may be corrupted or manifest is stale[/]")
        raise SystemExit(1)

    # --- 3. ONNX checker ---
    console.print("[bold cyan]Running onnx.checker[/]")
    try:
        import onnx

        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        console.print("[green]onnx.checker: OK[/]")
    except Exception as exc:
        console.print(f"[red]onnx.checker FAILED: {exc}[/]")
        raise SystemExit(1)

    # --- 4. Load ORT session ---
    console.print("[bold cyan]Loading ORT session[/]")
    try:
        session = _build_session(onnx_path)
        console.print(f"[green]Session loaded (providers: {session.get_providers()})[/]")
    except Exception as exc:
        console.print(f"[red]ORT session load FAILED: {exc}[/]")
        raise SystemExit(1)

    # --- 5. Smoke test ---
    console.print("[bold cyan]Running smoke test[/]")
    passed, msg = _smoke_test(session, model)
    if passed:
        console.print(f"[green]{msg}[/]")
    else:
        console.print(f"[red]{msg}[/]")
        raise SystemExit(1)

    # --- 6. Test images vs references ---
    if test_images_dir is None:
        console.print("[yellow]No --test-images provided — skipping image validation[/]")
        console.print("\n[bold green]All checks passed.[/]")
        return

    images_dir = Path(test_images_dir)
    if not images_dir.is_dir():
        console.print(f"[red]test-images directory not found: {images_dir}[/]")
        raise SystemExit(1)

    image_paths = sorted(
        p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )
    if not image_paths:
        console.print(f"[yellow]No images found in {images_dir} — skipping image validation[/]")
        console.print("\n[bold green]All checks passed.[/]")
        return

    preprocessing = model.get("preprocessing", {})
    refs_dir = Path(reference_outputs_dir) if reference_outputs_dir else None

    results_table = Table(title="Per-Image Validation", show_lines=True)
    results_table.add_column("Image", style="cyan")
    results_table.add_column("Status")
    results_table.add_column("Notes")

    all_passed = True
    for img_path in image_paths:
        try:
            arr = _preprocess_image(img_path, preprocessing)
            outputs = _run_inference(session, arr)

            if refs_dir is not None:
                ref_path = refs_dir / f"{img_path.stem}.npy"
                if ref_path.exists():
                    refs = [np.load(str(ref_path))]
                    # Compare first output only (most common case)
                    ok, errors = _compare_outputs([outputs[0]], refs, atol=atol, rtol=rtol)
                    if ok:
                        results_table.add_row(img_path.name, "[green]PASS[/]", "matches reference")
                    else:
                        results_table.add_row(img_path.name, "[red]FAIL[/]", "; ".join(errors))
                        all_passed = False
                else:
                    out_shapes = [str(o.shape) for o in outputs]
                    results_table.add_row(
                        img_path.name, "[yellow]NO REF[/]", f"output shapes: {', '.join(out_shapes)}"
                    )
            else:
                out_shapes = [str(o.shape) for o in outputs]
                results_table.add_row(
                    img_path.name, "[green]INFER OK[/]", f"output shapes: {', '.join(out_shapes)}"
                )
        except Exception as exc:
            results_table.add_row(img_path.name, "[red]ERROR[/]", str(exc))
            all_passed = False

    console.print(results_table)

    if all_passed:
        console.print("\n[bold green]All checks passed.[/]")
    else:
        console.print("\n[bold red]Some checks failed. See table above.[/]")
        raise SystemExit(1)
