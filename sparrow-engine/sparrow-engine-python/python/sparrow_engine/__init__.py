"""sparrow_engine: Camera trap animal detection powered by sparrow-engine-cpu."""
from __future__ import annotations

import os
import sys
import threading
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from pathlib import Path
from typing import Callable, Optional, Union


# Public package version. Single-sourced from the wheel METADATA (which
# itself comes from sparrow-engine-python/pyproject.toml). Tries the GPU
# distribution name first because the GPU wheel is the more specific
# install — if both were ever resolvable in the same env (Provides-Dist
# advisory only; not a hard guard per Phase 3.8 Phase C `feedback_no_soft_tolerance_framing_on_gates.md`),
# preferring GPU is the safer reflection of what's actually loaded.
def _resolve_version() -> str:
    for dist in ("sparrow-engine-gpu", "sparrow-engine"):
        try:
            return _pkg_version(dist)
        except PackageNotFoundError:
            continue
    return "unknown"


__version__ = _resolve_version()


# S6: per-file progress callback. Invoked once per input file, AFTER the
# file's inference attempt resolves (success or failure), with
# ``(index, total, filename)`` positional args. ``index`` is 0-based.
ProgressCallback = Callable[[int, int, str], None]


def _runtime_flavor() -> str:
    try:
        from sparrow_engine._flavor import FLAVOR as flavor  # type: ignore[import-not-found]
    except ImportError:
        flavor = os.environ.get("SPARROW_ENGINE_FLAVOR", "")
    return str(flavor).lower()


def _preload_nvjpeg_sidecar() -> None:
    """Preload nvidia-nvjpeg-cu12's libnvjpeg for the GPU Linux wheel."""
    if sys.platform != "linux" or _runtime_flavor() != "gpu":
        return

    import ctypes
    from importlib.resources import files

    override = os.environ.get("SPARROW_ENGINE_NVJPEG_LIBRARY_PATH")
    if override:
        try:
            ctypes.CDLL(override, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass
        return

    try:
        lib_dir = files("nvidia.nvjpeg") / "lib"
        candidate = lib_dir / "libnvjpeg.so.12"
        if candidate.is_file():
            ctypes.CDLL(str(candidate), mode=ctypes.RTLD_GLOBAL)
    except (ModuleNotFoundError, FileNotFoundError, OSError, TypeError):
        pass


def _preload_cuda_runtime_sidecars() -> None:
    """Preload CUDA runtime libs (cuDNN, cuBLAS, cuRAND, cuFFT) for the
    GPU Linux wheel from `nvidia-*-cu12` PyPI packages.

    ORT's CUDA EP dlopens `libcudnn.so.9` + `libcublas.so.12` +
    `libcurand.so.10` + `libcufft.so.11` at first inference. Without these
    preloads end users would need to set LD_LIBRARY_PATH manually at every
    launch (the gap surfaced when E.7 of the Phase E manual test ran against
    the TestPyPI wheel in a fresh venv, 2026-05-25). Mirrors the nvjpeg
    sidecar pattern.

    Preload semantics: `ctypes.CDLL(path, mode=RTLD_GLOBAL)` registers the
    library's symbols in the global symbol table, so ORT's subsequent
    `dlopen("libcudnn.so.9")` (by SONAME, no path) finds the already-loaded
    handle without needing the directory on LD_LIBRARY_PATH.

    Override via `SPARROW_ENGINE_CUDA_RUNTIME_PATH` (colon-separated list of
    lib dirs) — useful for system-CUDA installs or non-standard layouts.
    """
    if sys.platform != "linux" or _runtime_flavor() != "gpu":
        return

    import ctypes
    from importlib.resources import files

    def _preload_dir(d: Union[str, Path]) -> None:
        p = Path(d)
        if not p.is_dir():
            return
        # Sort so `libfoo.so.<major>` loads after any sub-libs the resolver
        # needs (alphabetical ordering does this reliably for the cuDNN /
        # cuBLAS family — `libcudnn_adv.so.9` precedes `libcudnn.so.9`).
        for entry in sorted(p.iterdir()):
            name = entry.name
            if name.startswith("lib") and ".so." in name and entry.is_file():
                try:
                    ctypes.CDLL(str(entry), mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    pass

    override = os.environ.get("SPARROW_ENGINE_CUDA_RUNTIME_PATH")
    if override:
        for d in override.split(os.pathsep):
            _preload_dir(d)
        return

    for pkg in ("nvidia.cudnn", "nvidia.cublas", "nvidia.curand", "nvidia.cufft"):
        try:
            _preload_dir(str(files(pkg) / "lib"))
        except (ModuleNotFoundError, FileNotFoundError, OSError, TypeError):
            pass


_preload_nvjpeg_sidecar()
_preload_cuda_runtime_sidecars()


# -------------------------------------------------------------------------
# RP-3 (2026-05-23): ORT dylib discovery shim.
#
# The native `_sparrow_engine_core` cdylib is built with `ort/load-dynamic`,
# which makes the `ort` crate dlopen `libonnxruntime` at first ORT call
# rather than DT_NEEDED-linking it at process load. With no env override
# `ort` falls back to a bare name (`libonnxruntime.so` / `.dylib` /
# `onnxruntime.dll`) — none of which pip wheels for `onnxruntime` ship
# directly (pip ships `libonnxruntime.so.X.Y.Z` only on Linux), so an
# unaided import would dlopen-fail.
#
# Fix: locate the versioned ORT dylib inside the user's pip-installed
# `onnxruntime` (or `onnxruntime-gpu`) and set ``ORT_DYLIB_PATH`` to its
# absolute path BEFORE the native module is imported. This eliminates the
# MT-4.1-15 manual ``ln -sf libonnxruntime.so.X.Y.Z libonnxruntime.so.1``
# workaround that every end user used to have to run by hand.
#
# Respects an explicit user override: if ``ORT_DYLIB_PATH`` is already set
# in the environment (any non-empty value), we leave it alone. This lets
# users point at a custom-built ORT, a system package, or a manylinux
# wheel sitting outside `site-packages`.
# -------------------------------------------------------------------------

def _discover_ort_dylib() -> Optional[str]:
    """Locate the versioned `libonnxruntime` shipped by pip's onnxruntime.

    Returns the absolute path as a string, or ``None`` if discovery fails
    (e.g. ``onnxruntime`` not installed, unknown layout). Caller is
    responsible for the fallback: leaving ``ORT_DYLIB_PATH`` unset lets
    ``ort`` try its platform-default name and surface a clearer error than
    a path we guessed wrong.
    """
    try:
        import onnxruntime  # type: ignore[import-not-found]
    except ImportError:
        return None

    ort_pkg = Path(onnxruntime.__file__).parent  # .../site-packages/onnxruntime
    capi = ort_pkg / "capi"
    if not capi.is_dir():
        return None

    # ort 2.0.0-rc.12 dlopens via libloading; on each platform it expects
    # the platform-specific extension. pip wheels ship versioned files:
    #   Linux   : libonnxruntime.so.X.Y.Z      (e.g. libonnxruntime.so.1.25.1)
    #   macOS   : libonnxruntime.X.Y.Z.dylib   (e.g. libonnxruntime.1.25.1.dylib)
    #   Windows : onnxruntime.dll              (no version suffix; ships as-is)
    if sys.platform == "win32":
        candidate = capi / "onnxruntime.dll"
        return str(candidate) if candidate.is_file() else None

    if sys.platform == "darwin":
        # Match libonnxruntime.<version>.dylib OR libonnxruntime.dylib.
        # pip's onnxruntime ships the versioned form; bare form is rare.
        for pattern in ("libonnxruntime.*.dylib", "libonnxruntime.dylib"):
            matches = sorted(capi.glob(pattern))
            if matches:
                return str(matches[-1])  # highest version
        return None

    # Linux + other ELF platforms.
    # Glob highest-versioned libonnxruntime.so.X.Y.Z. Fall back to bare .so
    # only as a last resort (most pip wheels don't ship the unversioned one).
    matches = sorted(capi.glob("libonnxruntime.so.*"))
    matches = [m for m in matches if not m.is_symlink()]  # prefer real files
    if matches:
        return str(matches[-1])
    bare = capi / "libonnxruntime.so"
    return str(bare) if bare.is_file() else None


def _configure_ort_dylib_path() -> None:
    """Populate ``ORT_DYLIB_PATH`` if unset. Idempotent. Silent on failure."""
    if os.environ.get("ORT_DYLIB_PATH"):
        return  # respect user override
    discovered = _discover_ort_dylib()
    if discovered is not None:
        os.environ["ORT_DYLIB_PATH"] = discovered


_configure_ort_dylib_path()


from sparrow_engine._sparrow_engine_core import (
    AudioClass,
    AudioResult,
    AudioSegment,
    BBox,
    SparrowEngineError,
    Classification,
    ClassifyResult,
    Detection,
    DetectResult,
    ModelInfo,
    PipelineDetection,
    PipelineResult,
    PyEngine,
)
from sparrow_engine._sparrow_engine_core import day_night as _day_night_core
from sparrow_engine._sparrow_engine_core import export_results as _export_core
from sparrow_engine._sparrow_engine_core import hash_file as _hash_file_core
from sparrow_engine._sparrow_engine_core import summarize as _summarize_core
from sparrow_engine._sparrow_engine_core import verify_model as _verify_model_core
from sparrow_engine._sparrow_engine_core import visualize as _visualize_core
from sparrow_engine._sparrow_engine_core import visualize_audio as _visualize_audio_core

__all__ = [
    # Version (single-sourced from wheel METADATA via importlib.metadata)
    "__version__",
    # Functions
    "init",
    "detect",
    "classify",
    "detect_audio",
    "pipeline",
    "list_models",
    "model_info",
    "active_device",
    # Phase 3 standalone functions
    "hash_file",
    "day_night",
    "verify_model",
    "summarize",
    # Phase 3 viz/export
    "visualize",
    "visualize_audio",
    "export",
    # Types (re-exported for isinstance checks and type annotations)
    "BBox",
    "Detection",
    "DetectResult",
    "Classification",
    "ClassifyResult",
    "AudioClass",
    "AudioSegment",
    "AudioResult",
    "PipelineDetection",
    "PipelineResult",
    "ModelInfo",
    "SparrowEngineError",
    # Callback alias
    "ProgressCallback",
]

_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
_AUDIO_EXTS = {".wav"}  # sparrow-engine-core uses hound (WAV only); expand when more codecs are added

_engine: Optional[PyEngine] = None
_engine_lock = threading.Lock()


def _get_engine() -> PyEngine:
    """Return the global engine, creating it lazily with env-var defaults."""
    global _engine
    if _engine is not None:
        return _engine
    with _engine_lock:
        if _engine is not None:
            return _engine
        device = os.environ.get("SPARROW_ENGINE_DEVICE", "auto")
        model_dir = os.environ.get(
            "SPARROW_ENGINE_MODEL_DIR", str(Path.home() / ".sparrow-engine" / "models")
        )
        _engine = PyEngine(device=device, model_dir=model_dir)
        return _engine


def _path_to_str(path: Union[str, Path]) -> str:
    return os.fsdecode(os.fspath(path))


def _resolve_inputs(
    input: Union[str, Path, list[Union[str, Path]]],  # noqa: A002
    extensions: set[str],
    recursive: bool = False,
) -> list[str]:
    """Normalize input to a list of file paths.

    Accepts a single path (str or Path), a directory (expands to matching
    files), or a list of paths. When ``recursive`` is True, directories
    are traversed recursively.
    """
    if isinstance(input, (str, Path)):
        input = [input]
    files: list[str] = []
    for item in input:
        p = Path(item)
        if p.is_dir():
            if recursive:
                files.extend(
                    str(f) for f in p.rglob("*") if f.suffix.lower() in extensions
                )
            else:
                files.extend(
                    str(f) for f in p.iterdir() if f.suffix.lower() in extensions
                )
        else:
            files.append(str(p))
    return sorted(files)


# -------------------------------------------------------------------------
# 8 MVP functions
# -------------------------------------------------------------------------


def init(device: str = "auto", model_dir: Optional[str] = None) -> None:
    """Explicitly initialize the engine.

    Optional — the engine auto-initializes on first inference call using
    ``SPARROW_ENGINE_DEVICE`` (default ``auto``) and ``SPARROW_ENGINE_MODEL_DIR`` (default
    ``~/.sparrow-engine/models``).
    """
    global _engine
    with _engine_lock:
        if model_dir is None:
            model_dir = os.environ.get(
                "SPARROW_ENGINE_MODEL_DIR", str(Path.home() / ".sparrow-engine" / "models")
            )
        _engine = None  # Drop old engine first → ENGINE_EXISTS = false
        _engine = PyEngine(device=device, model_dir=model_dir)


def detect(
    input: Union[str, Path, list[Union[str, Path]]],  # noqa: A002
    model: str,
    threshold: Optional[float] = None,
    max_detections: Optional[int] = None,
    recursive: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[DetectResult]:
    """Run object detection on one or more images.

    ``input`` can be a file path, directory, or list of paths.
    When ``recursive`` is True, directories are traversed recursively.
    Always returns ``list[DetectResult]``, even for a single image.

    ``threshold`` defaults to ``None``, which defers to the manifest's
    ``[postprocessing] confidence_threshold`` (typically 0.2 for YOLO-family
    models). Pass an explicit float to override.

    If ``progress_callback`` is provided, it is called once per file after
    its inference attempt resolves, with ``(index, total, filename)``.
    ``index`` is 0-based. Raising from the callback aborts the batch.
    """
    paths = _resolve_inputs(input, _IMAGE_EXTS, recursive=recursive)
    return _get_engine().detect(
        paths, model, threshold, max_detections, progress_callback
    )


def classify(
    input: Union[str, Path, list[Union[str, Path]]],  # noqa: A002
    model: str,
    top_k: int = 5,
    recursive: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[ClassifyResult]:
    """Run image classification on one or more images.

    ``input`` can be a file path, directory, or list of paths.
    When ``recursive`` is True, directories are traversed recursively.
    Always returns ``list[ClassifyResult]``, even for a single image.

    If ``progress_callback`` is provided, it is called once per file after
    its inference attempt resolves, with ``(index, total, filename)``.
    ``index`` is 0-based. Raising from the callback aborts the batch.
    """
    paths = _resolve_inputs(input, _IMAGE_EXTS, recursive=recursive)
    return _get_engine().classify(paths, model, top_k, progress_callback)


def detect_audio(
    input: Union[str, Path, list[Union[str, Path]]],  # noqa: A002
    model: str,
    threshold: Optional[float] = None,
    recursive: bool = False,
    stride_s: Optional[float] = None,
    segment_duration_s: Optional[float] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[AudioResult]:
    """Run audio detection on one or more audio files.

    ``input`` can be a file path, directory, or list of paths.
    When ``recursive`` is True, directories are traversed recursively.
    Always returns ``list[AudioResult]``, even for a single file.

    ``stride_s`` and ``segment_duration_s`` override the manifest defaults.
    Stride is always engine-controlled. Segment duration is honored by
    mel-spectrogram audio models with a dynamic ONNX time-axis (e.g.
    ``md-audiobirds-v1``); silently ignored by raw-audio classifiers whose
    ONNX input is fixed-size (e.g. ``perch-v2``'s ``[batch, 160000]``) —
    the window is an upstream architecture constraint for those models.

    If ``progress_callback`` is provided, it is called once per file after
    its inference attempt resolves, with ``(index, total, filename)``.
    ``index`` is 0-based. Raising from the callback aborts the batch.
    """
    paths = _resolve_inputs(input, _AUDIO_EXTS, recursive=recursive)
    return _get_engine().detect_audio(
        paths,
        model,
        threshold,
        stride_s,
        segment_duration_s,
        progress_callback,
    )


def pipeline(
    input: Union[str, Path, list[Union[str, Path]]],  # noqa: A002
    detector: str,
    classifier: str,
    threshold: Optional[float] = None,
    top_k: int = 5,
    recursive: bool = False,
    progress_callback: Optional[ProgressCallback] = None,
) -> list[PipelineResult]:
    """Run detect-then-classify pipeline on one or more images.

    Ad-hoc pipeline — no pre-defined TOML required. Detect with
    ``detector``, crop each detection, classify with ``classifier``.
    When ``recursive`` is True, directories are traversed recursively.
    Always returns ``list[PipelineResult]``, even for a single image.

    If ``progress_callback`` is provided, it is called once per file after
    its inference attempt resolves, with ``(index, total, filename)``.
    ``index`` is 0-based. Raising from the callback aborts the batch.
    """
    paths = _resolve_inputs(input, _IMAGE_EXTS, recursive=recursive)
    return _get_engine().pipeline(
        paths, detector, classifier, threshold, top_k, progress_callback
    )


def list_models() -> list[ModelInfo]:
    """List all available models in the model directory."""
    return _get_engine().list_models()


def model_info(model_id: str) -> ModelInfo:
    """Get info for a specific model by ID.

    Raises ``SparrowEngineError`` if the model is not found.
    """
    return _get_engine().model_info(model_id)


def active_device() -> str:
    """Return the active compute device (``"cpu"``, ``"cuda:0"``, etc.)."""
    return _get_engine().active_device()


# -------------------------------------------------------------------------
# Phase 3 standalone functions (no engine initialization)
# -------------------------------------------------------------------------


def hash_file(path: Union[str, Path]) -> str:
    """Compute SHA-256 hash of a file. No engine initialization required."""
    return _hash_file_core(str(path))


def day_night(path: Union[str, Path]) -> dict:
    """Classify an image as day or night. No engine initialization required.

    Returns ``{"classification": "day"|"night", "mean_brightness": float}``.
    """
    return _day_night_core(str(path))


def verify_model(
    model_id: str, model_dir: Optional[Union[str, Path]] = None
) -> dict:
    """Verify a model's integrity against manifest checksums.

    No engine initialization required. Resolves ``model_dir`` from
    ``SPARROW_ENGINE_MODEL_DIR`` env var or ``~/.sparrow-engine/models`` if not provided.

    Returns a dict with ``"status"`` key (``"ok"``, ``"no_checksum"``,
    ``"size_mismatch"``, ``"checksum_mismatch"``).
    """
    if model_dir is None:
        model_dir = os.environ.get(
            "SPARROW_ENGINE_MODEL_DIR", str(Path.home() / ".sparrow-engine" / "models")
        )
    return _verify_model_core(str(model_dir), model_id)


def summarize(results: list[DetectResult]) -> dict:
    """Summarize detection results. No engine initialization required.

    Returns a dict with total_images, images_with_detections, empty_images,
    total_detections, confidence stats (confidence_min / confidence_max /
    confidence_mean), and a per-category breakdown where each entry carries
    count plus confidence_min / confidence_max / confidence_mean.
    """
    return _summarize_core(results)


# -------------------------------------------------------------------------
# Phase 3 viz/export
# -------------------------------------------------------------------------


def visualize(
    items: list[tuple[Union[str, Path], Union[DetectResult, ClassifyResult, PipelineResult]]],
    output_dir: Optional[Union[str, Path]] = None,
    show_labels: bool = False,
) -> list[bytes]:
    """Render bounding box visualizations for a batch of (path, result) pairs.

    No engine initialization required. Returns ``list[bytes]`` with encoded
    image bytes — JPEG for ``.jpg``/``.jpeg`` inputs, PNG for all other
    inputs (including PNG and unknown extensions; PNG is lossless).
    If ``output_dir`` is set, also saves to disk with directory mirroring.

    ``show_labels=True`` renders ``"{label} {conf:.2}"`` text above each
    bbox using the bundled DejaVu Sans font. Default off (clean overlays).
    """
    converted = [(_path_to_str(p), r) for p, r in items]
    out = _path_to_str(output_dir) if output_dir is not None else None
    return _visualize_core(converted, out, show_labels)


def visualize_audio(
    items: list[tuple[Union[str, Path], AudioResult]],
    output_dir: Optional[Union[str, Path]] = None,
    smooth: bool = False,
    show_windows: bool = False,
    show_ranges: bool = True,
) -> list[list[bytes]]:
    """Render audio detection visualization layers for a batch.

    Mirrors :func:`visualize` but for :class:`AudioResult`. Non-empty calls
    automatically initialize the engine on first use via ``_get_engine()``;
    explicit :func:`init` is optional and lets callers choose device/model_dir.
    The engine supplies audio preprocess config via each result's ``model_id``;
    each :class:`AudioResult` carries the effective window/stride used during
    detection, including runtime overrides.

    Returns ``list[list[bytes]]`` — outer list one entry per input item;
    inner list holds encoded PNG bytes for every layer rendered in render
    order (3-5 layers depending on options/ranges):

    * ``01_spec`` — raw spectrogram, no overlays
    * ``02_segments`` — per-slot confidence (no blur)
    * ``02_segments_windows`` — only when ``show_windows=True``
    * ``03_heatmap`` — smoothed heatmap when ``smooth=True``, else identical to 02
    * ``04_full`` — heatmap + cyan range bars (only when ``show_ranges=True``)

    If ``output_dir`` is set, also writes files using directory mirroring
    with filenames ``{stem}_{layer_name}.png``.
    """
    if not items:
        return []
    converted = [(_path_to_str(p), r) for p, r in items]
    out = _path_to_str(output_dir) if output_dir is not None else None
    engine = _get_engine()
    return _visualize_audio_core(engine, converted, out, smooth, show_windows, show_ranges)


def export(
    items: list[tuple[Union[str, Path], Union[DetectResult, PipelineResult]]],
    format: str,  # noqa: A002
    output: Optional[Union[str, Path]] = None,
    model_id: Optional[str] = None,
) -> str:
    """Export detection/pipeline results to megadet, coco, or csv format.

    No engine initialization required. Always returns ``str`` (formatted
    content). If ``output`` is set, also writes to file. ``model_id`` is
    required for megadet format.
    """
    converted = [(str(p), r) for p, r in items]
    out = str(output) if output is not None else None
    return _export_core(converted, format, out, model_id)
