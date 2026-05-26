"""Tests for the public `sparrow_engine.__version__` attribute.

`__version__` is single-sourced from the wheel METADATA via
`importlib.metadata.version(...)`, which itself comes from
`sparrow-engine-python/pyproject.toml`. The CPU wheel installs as
`sparrow-engine`; the GPU wheel installs as `sparrow-engine-gpu`. Both
import as `sparrow_engine`, so the resolver tries the GPU dist name first
and falls through to the CPU dist name.

This test asserts the public API contract, not the version string itself
(which changes every release). A failure here means a tester's
`python -c "import sparrow_engine; print(sparrow_engine.__version__)"`
will break, which was the gap surfaced by the 2026-05-26 prod-PyPI smoke
attempt.
"""
from __future__ import annotations

import re

import sparrow_engine


def test_version_attribute_exists() -> None:
    """`sparrow_engine.__version__` must be a non-empty string."""
    assert hasattr(sparrow_engine, "__version__"), (
        "sparrow_engine.__version__ is missing — see "
        "sparrow-engine-python/python/sparrow_engine/__init__.py"
    )
    assert isinstance(sparrow_engine.__version__, str)
    assert sparrow_engine.__version__, "version string is empty"


def test_version_listed_in__all__() -> None:
    """`__version__` must be in `__all__` so `from sparrow_engine import *` exports it."""
    assert "__version__" in sparrow_engine.__all__


def test_version_shape_is_pep440_or_unknown() -> None:
    """Version must be either a PEP-440-shaped string or the fallback `"unknown"`.

    PEP-440 covers all current and foreseeable shapes: `0.1.4`, `0.1.4rc1`,
    `0.1.4.dev3+gabc1234`, etc. The fallback `"unknown"` should only fire
    when neither distribution name is resolvable, which would mean
    `pip install` somehow landed the package without metadata — a broken
    install state we still want to handle gracefully rather than crash.
    """
    v = sparrow_engine.__version__
    if v == "unknown":
        return
    # Minimal PEP-440 sniff: starts with a digit, contains only [0-9a-z.+-]
    assert re.match(r"^[0-9][0-9a-z.+\-]*$", v), f"unexpected version shape: {v!r}"
