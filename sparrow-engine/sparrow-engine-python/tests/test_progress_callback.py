"""Tests for the per-file progress callback (S6).

These tests verify the `progress_callback: Callable[[int, int, str], None]`
semantics exposed on `detect`, `classify`, `detect_audio`, and `pipeline`.
Contract:

    The callback fires once per input file AFTER its inference attempt
    resolves (success or failure). Positional args: (index, total,
    filename). `index` is 0-based. If the callback raises, the batch
    aborts and the exception surfaces to Python.

The full inference loops need a loaded ONNX model, so these tests drive
the same `invoke_progress` helper via the `_invoke_test_progress_callback`
native hook that reproduces the GIL-dance and exception-propagation
semantics without touching the engine.
"""
from __future__ import annotations

import pytest

from sparrow_engine import _sparrow_engine_core


def test_callback_invoked_once_per_item() -> None:
    """N items → exactly N calls, in order 0..N-1."""
    calls: list[tuple[int, int, str]] = []

    def cb(idx: int, total: int, name: str) -> None:
        calls.append((idx, total, name))

    _sparrow_engine_core._invoke_test_progress_callback(cb, 4)

    assert calls == [
        (0, 4, "test_0.jpg"),
        (1, 4, "test_1.jpg"),
        (2, 4, "test_2.jpg"),
        (3, 4, "test_3.jpg"),
    ]


def test_callback_total_matches_batch_size() -> None:
    """Every call passes the same `total`, equal to the batch size."""
    totals: set[int] = set()

    def cb(idx: int, total: int, name: str) -> None:
        totals.add(total)

    _sparrow_engine_core._invoke_test_progress_callback(cb, 7)

    assert totals == {7}


def test_callback_zero_items_no_calls() -> None:
    """Empty batch → callback never invoked."""
    called = False

    def cb(idx: int, total: int, name: str) -> None:
        nonlocal called
        called = True

    _sparrow_engine_core._invoke_test_progress_callback(cb, 0)

    assert not called


def test_callback_exception_propagates_and_aborts() -> None:
    """If the callback raises, the helper must surface the exception
    through the Rust boundary — confirming that inference batches abort
    cleanly on a user-raised error (e.g. `KeyboardInterrupt` from a
    progress-bar UI).
    """
    calls: list[int] = []

    class CallbackBoom(RuntimeError):
        pass

    def cb(idx: int, total: int, name: str) -> None:
        calls.append(idx)
        if idx == 2:
            raise CallbackBoom("stop")

    with pytest.raises(CallbackBoom, match="stop"):
        _sparrow_engine_core._invoke_test_progress_callback(cb, 5)

    # Callback ran for 0, 1, 2 — then the exception on index 2 aborted
    # the loop before 3, 4 would have fired.
    assert calls == [0, 1, 2]


def test_callback_index_is_valid_into_paths() -> None:
    """Acceptance-criterion shape:

        `sparrow_engine.detect(files, progress_callback=fn)` invokes
        `fn(i, len(files), files[i])` per file.

    With 0-based `i` ranging over `0..N-1`, `files[i]` is always valid.
    Verify this property directly.
    """
    # In the real batch, `filename` is `files[i]`. The test helper
    # synthesizes `test_{i}.jpg`, but the shape-contract is the same:
    # `index` must always be a valid 0-based position within `[0, total)`
    # and the third arg must be the filename at that position.
    seen: list[tuple[int, int, str]] = []

    def cb(idx: int, total: int, name: str) -> None:
        assert 0 <= idx < total, f"index {idx} out of range [0, {total})"
        assert name == f"test_{idx}.jpg", (
            f"third arg {name!r} disagrees with index {idx}"
        )
        seen.append((idx, total, name))

    _sparrow_engine_core._invoke_test_progress_callback(cb, 3)
    assert len(seen) == 3


def test_callback_may_hold_gil_safely() -> None:
    """Python work inside the callback (dict update, print, etc.) must
    be safe — the helper reacquires the GIL via `Python::with_gil` before
    invoking. Exercise by doing non-trivial Python work in the callback.
    """
    state = {"count": 0, "last": None}

    def cb(idx: int, total: int, name: str) -> None:
        state["count"] += 1
        state["last"] = (idx, total, name)

    _sparrow_engine_core._invoke_test_progress_callback(cb, 10)

    assert state["count"] == 10
    assert state["last"] == (9, 10, "test_9.jpg")
