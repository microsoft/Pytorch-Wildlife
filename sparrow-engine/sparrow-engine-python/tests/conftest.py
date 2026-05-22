"""Shared pytest fixtures for sparrow-engine-python tests.

Phase 3.5 S6: the tests in this directory target the Rust → Python logging
bridge (`pyo3-log`) and the per-file progress callback. Neither needs a
loaded ONNX model — both drive the code paths via small internal helpers
on the native module (`_emit_test_warn`, `_invoke_test_progress_callback`).
"""
from __future__ import annotations

import logging
from typing import Iterator, List

import pytest


class _RecordCollector(logging.Handler):
    """Minimal handler that stores every `LogRecord` it receives.

    Used by the logging-bridge tests to assert that Rust-side
    `tracing::warn!(target: "sparrow_engine::python", ...)` events reach the
    `"sparrow_engine"` Python logger hierarchy.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture()
def sparrow_engine_log_handler() -> Iterator[_RecordCollector]:
    """Attach a record-collecting handler to `logging.getLogger("sparrow_engine")`.

    Sets the `"sparrow_engine"` logger to DEBUG level, installs a clean handler,
    and (via `propagate=False`) keeps records out of pytest's own
    capturing plumbing. The fixture yields the handler so tests can
    inspect `handler.records`.
    """
    logger = logging.getLogger("sparrow_engine")
    prev_level = logger.level
    prev_propagate = logger.propagate

    handler = _RecordCollector()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        logger.setLevel(prev_level)
        logger.propagate = prev_propagate
