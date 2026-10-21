import importlib.metadata as importlib_metadata
from pathlib import Path

try:
    # When installed (pip install [-e] .), setuptools writes the version
    # from setup.py into the installed distribution metadata.
    __version__ = importlib_metadata.version(__package__ or __name__)
except importlib_metadata.PackageNotFoundError:
    # Source checkout without install — fall back to version.txt at repo root.
    _version_file = Path(__file__).resolve().parent.parent / "version.txt"
    try:
        __version__ = _version_file.read_text().strip()
    except FileNotFoundError:
        __version__ = "development"

import warnings as _warnings

_warnings.warn(
    "PytorchWildlife is no longer maintained and 1.3.1 is its final release. "
    "Switch to sparrow-engine (`pip install sparrow-engine`), which runs the same "
    "MegaDetector and classifier models without PyTorch. Migration guide: "
    "https://github.com/microsoft/SPARROW-Playground/blob/main/notebooks/10_from_pytorch_wildlife.ipynb",
    FutureWarning,
    stacklevel=2,
)

from .data import *
from .models import *
from .utils import *