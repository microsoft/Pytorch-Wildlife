# Python binding review — round 02 Step 1

Status: findings identified.

## Scope reads
- Read `~/.copilot/skills/_shared/iterative-anti-drift.md`: editing-mode convergence requires round-level coverage; editor Step 1 appends audited coverage for planned files.
- Read `docs/review/audit-fix/SCOPE_LEDGER.json`: `sparrow-engine/sparrow-engine-python/src/lib.rs` and `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py` are in reviewer-owned ledger scope.
- Read `docs/review/audit-fix/COVERAGE_LOG.jsonl`: round 1 fixed `ITEM-REV-007` in `src/lib.rs`; no round-2 entries yet.
- Read `docs/review/audit-fix/round_01/inquisitor_review.md`: line 129 explicitly defers the `AudioClass` top-level import/`__all__` gap to round 2.
- Read `docs/review/audit-fix/round_02/file_ownership.md`: Python binding files are reviewer-owned in round 2.

## Finding
- `sparrow-engine-python/src/lib.rs` defines/registers `AudioClass`:
  - `#[pyclass(...)] pub struct AudioClass` at lib.rs:309-319.
  - native module registration `m.add_class::<AudioClass>()?` at lib.rs:1419.
  - conversion path populates `AudioSegment.classes: Vec<AudioClass>` at lib.rs:452-466.
- `sparrow-engine-python/python/sparrow_engine/__init__.py` imports `AudioResult` and `AudioSegment` from `_sparrow_engine_core` at lines 14-27 but omits `AudioClass`; `__all__` includes `AudioSegment` and `AudioResult` at lines 59-60 but omits `AudioClass`.
- `_core.pyi` already declares `class AudioClass` and `AudioSegment.classes: list[AudioClass]`, so the missing public name is the top-level Python package export, not the native module or stub for the native module.


## Automated export comparison
- A read-only parser over `m.add_class::<...>()`, top-level `_sparrow_engine_core` imports, and `__all__` found:
  - registered public result classes, excluding internal `PyEngine`: `BBox`, `Detection`, `DetectResult`, `Classification`, `ClassifyResult`, `PipelineDetection`, `PipelineResult`, `AudioClass`, `AudioSegment`, `AudioResult`, `ModelInfo`.
  - missing from top-level import: `AudioClass` only.
  - missing from `__all__`: `AudioClass` only.

## Exact reviewer plan item
- `ITEM-REV-010` (or next reviewer ID): Export Python `AudioClass` at the top-level package.
  - File: `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py`
  - Change: add `AudioClass` to the `from sparrow_engine._sparrow_engine_core import (...)` list adjacent to `AudioSegment`/`AudioResult`, and add `"AudioClass"` to `__all__` in the Types section adjacent to `"AudioSegment"`/`"AudioResult"`.
  - Acceptance: `import sparrow_engine; sparrow_engine.AudioClass` resolves to the same native class object as `sparrow_engine._sparrow_engine_core.AudioClass`, and `"AudioClass" in sparrow_engine.__all__`.

## Tests needed
- Add a Python package test, e.g. `sparrow-engine/sparrow-engine-python/tests/test_public_exports.py`, with:
  1. `def test_audio_class_is_top_level_exported(): import sparrow_engine; from sparrow_engine import _sparrow_engine_core; assert sparrow_engine.AudioClass is _sparrow_engine_core.AudioClass`
  2. `def test_audio_class_is_listed_in_all(): import sparrow_engine; assert "AudioClass" in sparrow_engine.__all__`
  3. Optional guard: verify no AudioClass-only omission remains by comparing the audio result type trio: `for name in ("AudioClass", "AudioSegment", "AudioResult"): assert hasattr(sparrow_engine, name); assert name in sparrow_engine.__all__`.
- Targeted validation after reviewer edits: build/install the Python extension in the usual project environment, then run the new test file with pytest. If a built editable extension is already present, `uv run --no-project --with pytest python -m pytest sparrow-engine/sparrow-engine-python/tests/test_public_exports.py -q` is sufficient; otherwise first run the existing maturin develop/build step used for Python binding tests, then the same pytest selector.

## Analogous missing names check
- Within the top-level `__init__.py` import/`__all__` surface, all pre-existing result types registered by `src/lib.rs` are already exported: `BBox`, `Detection`, `DetectResult`, `Classification`, `ClassifyResult`, `PipelineDetection`, `PipelineResult`, `AudioSegment`, `AudioResult`, `ModelInfo`, and `SparrowEngineError`.
- The only AudioClass-addition-related omission found in the owned Python binding surface is `AudioClass` itself.
