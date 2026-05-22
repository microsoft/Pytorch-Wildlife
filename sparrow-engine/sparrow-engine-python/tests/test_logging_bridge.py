"""Tests for the tracing → Python logging bridge (S6).

These tests verify that Rust-side `tracing::warn!(target: "sparrow_engine::python", ...)`
events are surfaced via `pyo3-log` under the Python logger
`logging.getLogger("sparrow_engine.python")` (a child of `"sparrow_engine"`), satisfying the
Phase 3.5 S6 acceptance criterion:

    `logging.getLogger("sparrow_engine").setLevel(logging.DEBUG); sparrow_engine.detect(...)`
    → WARN record visible to Python logging.

Before S6 these sites used `eprintln!`, which is invisible under PyO3 /
Jupyter (see PyO3 #2247). The full batch path is not exercised here — it
requires a loaded ONNX model; see docs/design/phase3.5/final_design.md §4
S6 for the end-to-end canonical invocation. These tests use the
`_emit_test_warn` native helper to drive the same `tracing::warn!` code
path without touching the engine.
"""
from __future__ import annotations

import logging

import pytest

import sparrow_engine
from sparrow_engine import _sparrow_engine_core  # noqa: F401 — import triggers module init


def test_module_import_initializes_pyo3_log_without_panic() -> None:
    """Module init calls `pyo3_log::try_init()`. Importing `sparrow_engine` must
    not raise, even when imported multiple times (re-import is a no-op).
    """
    # Re-importing is a no-op in Python; `try_init()` also does nothing
    # on the Rust side since `log::set_logger` already succeeded on the
    # first init. This test pins the "no panic on re-import" property.
    import importlib

    importlib.reload(sparrow_engine)


def test_tracing_warn_reaches_sparrow_engine_logger(sparrow_engine_log_handler) -> None:
    """A `tracing::warn!` on target `"sparrow_engine::python"` must surface as a
    WARNING record at `logging.getLogger("sparrow_engine.python")`, which
    propagates up to the `"sparrow_engine"` logger where the handler is attached.
    """
    _sparrow_engine_core._emit_test_warn("hello from rust")

    # Find the record that originated from our test emission.
    matches = [r for r in sparrow_engine_log_handler.records if "hello from rust" in r.getMessage()]
    assert matches, (
        f"no bridged record captured; saw {len(sparrow_engine_log_handler.records)} "
        f"records total: {[r.getMessage() for r in sparrow_engine_log_handler.records]}"
    )

    record = matches[-1]
    assert record.levelno == logging.WARNING, (
        f"expected WARNING level, got {record.levelname} ({record.levelno})"
    )


def test_tracing_target_becomes_dotted_python_logger(sparrow_engine_log_handler) -> None:
    """`pyo3-log` converts `::` in the Rust log target to `.` in the
    Python logger name. Rust `target: "sparrow_engine::python"` must land on
    Python logger `"sparrow_engine.python"` — making it a child of `"sparrow_engine"` so
    the handler registered on `"sparrow_engine"` captures it via propagation.
    """
    _sparrow_engine_core._emit_test_warn("target probe")

    matches = [r for r in sparrow_engine_log_handler.records if "target probe" in r.getMessage()]
    assert matches, "expected our probe record to be captured"
    record = matches[-1]
    assert record.name == "sparrow_engine.python", (
        f"expected logger name 'sparrow_engine.python', got '{record.name}'"
    )


def test_sparrow_engine_logger_level_filter_blocks_info(sparrow_engine_log_handler) -> None:
    """Honoring the Python logger's level filter: setting the `"sparrow_engine"`
    logger to WARNING should drop DEBUG/INFO events. This confirms we
    integrate with Python logging's effective-level machinery rather
    than side-stepping it.

    We only emit a WARN event here (the only helper we expose), then
    re-filter. DEBUG/INFO emissions would require additional helpers —
    left as follow-up if future code paths need them.
    """
    logger = logging.getLogger("sparrow_engine")
    logger.setLevel(logging.ERROR)  # above WARNING
    before = len(sparrow_engine_log_handler.records)

    _sparrow_engine_core._emit_test_warn("should be filtered")

    after = len(sparrow_engine_log_handler.records)
    # With level ERROR and an emitted WARNING, the Python logger should
    # drop the record before reaching our handler.
    assert after == before, (
        f"expected level-filter to block WARNING at level ERROR; "
        f"captured {after - before} new record(s)"
    )

    # Restore DEBUG so later tests still see events.
    logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize("payload", ["unicode: café", "arg {0} {1}", "braces {{ }}"])
def test_message_payload_roundtrips_verbatim(sparrow_engine_log_handler, payload: str) -> None:
    """The format-string payload from Rust must land in the Python record
    verbatim — no double-interpretation of `{}` braces, no charset loss.
    """
    _sparrow_engine_core._emit_test_warn(payload)
    matches = [r for r in sparrow_engine_log_handler.records if r.getMessage() == payload]
    assert matches, (
        f"payload {payload!r} not found in captured records "
        f"(saw: {[r.getMessage() for r in sparrow_engine_log_handler.records]})"
    )
