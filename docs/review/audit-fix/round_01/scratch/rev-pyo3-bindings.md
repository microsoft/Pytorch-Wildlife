# rev-pyo3-bindings

## Findings

No owned-file behavioral correctness findings in `sparrow-engine/sparrow-engine-python/src/lib.rs`.

- Evidence: `detect_audio` runs model loading and inference under `py.allow_threads` (`src/lib.rs:778-785`) and invokes the progress callback after each file while `invoke_progress` reacquires the GIL (`src/lib.rs:593-606`, `src/lib.rs:799`).
- Evidence: native engine errors are mapped to the package exception type by `to_pyerr` (`src/lib.rs:61-69`); all-file audio failures return `SparrowEngineError` (`src/lib.rs:804-805`).
- Evidence: `AudioClass` is a PyO3 class with `class_idx`, `label`, and `probability`; `AudioSegment` keeps backward-compatible `confidence` and adds `classes: Vec<AudioClass>` (`src/lib.rs:306-345`); module init registers all three audio result classes (`src/lib.rs:1411-1413`).
- Evidence: conversion clones owned labels/classes, so Python results do not borrow native temporaries (`src/lib.rs:449-462`). Empty native `classes` remains an empty Python-visible vector via the same collect path (`src/lib.rs:462`).
- Evidence: Perch 2 top-K data produced by the CPU raw-audio path (`sparrow-engine-cpu/src/detect_audio.rs:525-548`, `:643-662`) reaches Python through `AudioResult.segments = r.segments.iter().map(convert_audio_segment).collect()` (`src/lib.rs:786-792`).
- Confidence: HIGH — verified against the owned binding file and native CPU audio path.

## Test Gaps

### TG-PYO3-1 — Missing multi-class and empty-class conversion tests

- File: `sparrow-engine/sparrow-engine-python/src/lib.rs:1608-1627`
- Observed evidence: `convert_audio_segment_maps_classes` covers only one `AudioClass` with `Some(label)`. It does not cover multiple classes, `None` labels, empty `classes`, or the invariant that `AudioSegment.confidence` matches the top class probability for Perch 2-style segments.
- Proposed fix: add unit cases for (1) two or more classes preserving order and probabilities, (2) `label: None`, (3) `classes: vec![]`, and (4) `confidence == classes[0].probability` when classes are non-empty.
- Rationale: these are the behavioral compatibility edges for Perch 2 top-K and legacy/classless audio segments.
- Confidence: HIGH — the current test body is limited to a single labeled class.

### TG-PYO3-2 — No Python-level coverage for audio class object shape

- File: `sparrow-engine/sparrow-engine-python/tests/conftest.py:3-6`, `sparrow-engine/sparrow-engine-python/tests/test_progress_callback.py:1-15`
- Observed evidence: existing pytest coverage targets logging and progress-callback helper paths; grep over `sparrow-engine-python/tests/**/*.py` finds no `AudioClass`, `AudioSegment.classes`, Perch, or top-K assertions except the progress-callback docstring mentioning `detect_audio`.
- Proposed fix: add a small test-only native helper or fixture-backed Python integration test that returns an `AudioResult` containing multiple `AudioClass` entries, then assert `segment.classes` is a Python list of `AudioClass`, empty classes are exposed as `[]`, and top-K length/order/probabilities are preserved.
- Rationale: Rust unit tests prove conversion structs, but not Python-visible attribute/getter behavior or import ergonomics.
- Confidence: HIGH — verified by direct test grep and current test files.

## Cross-Scope Findings

### CS-PYO3-1 — Top-level package does not re-export `AudioClass`

- File: `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py:14-27`, `:53-60`; related stub: `python/sparrow_engine/_core.pyi:53-62`
- Observed evidence: `__init__.py` imports `AudioResult` and `AudioSegment` from `_sparrow_engine_core`, but not `AudioClass`; `__all__` lists `AudioSegment` and `AudioResult`, but not `AudioClass`. The core stub defines `AudioClass` and `AudioSegment.classes: list[AudioClass]`.
- Proposed fix: add `AudioClass` to the `_sparrow_engine_core` import list and `__all__`. If a top-level `__init__.pyi` is later added, include the same export there.
- Rationale: Python users can receive `AudioClass` objects in `segment.classes`, but cannot use the documented top-level package namespace for `isinstance` checks or annotations (`sparrow_engine.AudioClass`) even though sibling audio result types are re-exported.
- Confidence: HIGH — direct source evidence shows the missing import/export.

STATUS: DONE
