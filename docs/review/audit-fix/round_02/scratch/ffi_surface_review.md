# FFI surface review notes — round 2 Step 1

Scope: reviewer-owned FFI/bindings files and owned tests only. Source files were read-only; no source edits or commits performed.

## Required inputs read
- `~/.copilot/skills/_shared/iterative-anti-drift.md`
- `docs/review/audit-fix/SCOPE_LEDGER.json`
- `docs/review/audit-fix/COVERAGE_LOG.jsonl`
- `docs/review/audit-fix/round_01/inquisitor_review.md`
- `docs/review/audit-fix/round_02/file_ownership.md`
- Round-1 reviewer plan/report and inquisitor approvals for item context.

## Findings worth round-2 planning

### FFI-R2-001 — non-audio FFI arrays still expose dangling non-null pointers on empty result sets

Evidence:
- CPU `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:343`: `SparrowEngineDetections.data` is assigned `combined._owner.detections.as_ptr()` unconditionally.
- CPU `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:415`: `SparrowEngineClassifyResult.top_results` is assigned `combined._owner.top_results.as_ptr()` unconditionally.
- CPU `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:510`: `SparrowEnginePipelineResult.data` is assigned `combined._owner.data.as_ptr()` unconditionally.
- GPU has the same patterns at `sparrow-engine/sparrow-engine-gpu/src/ffi.rs:352`, `:424`, and `:519`.

Why this matters:
- Round 1 fixed the same `Vec::as_ptr()`-on-empty sentinel problem for audio V1/V2 top-level arrays.
- Empty detections and empty pipeline outputs are normal threshold outcomes; empty classification output is defensively represented by existing code.
- Leaving the non-audio arrays divergent keeps the same ABI edge case for C consumers that treat `(len == 0, ptr == NULL)` as the only empty-array sentinel.

Suggested fix shape:
- Apply the same null-on-empty convention to CPU/GPU detection, classification top_results, and pipeline result arrays.
- Add unit coverage in both CPU/GPU `ffi.rs` for empty `DetectResult`, empty `ClassifyResult`, and empty `PipelineResult` conversions.

### FFI-R2-002 — Python package still does not re-export `AudioClass`

Evidence:
- Native module registers `AudioClass` in `sparrow-engine/sparrow-engine-python/src/lib.rs:1419` and `AudioSegment.classes` returns `Vec<AudioClass>`.
- Top-level package import list in `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py:14-27` imports `AudioResult` and `AudioSegment`, but not `AudioClass`.
- `__all__` in `sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py:35-67` lists `AudioSegment` and `AudioResult`, but not `AudioClass`.

Why this matters:
- Python users can receive `AudioClass` objects from `detect_audio`, but cannot name `sparrow_engine.AudioClass` for annotations or `isinstance` checks.
- This was identified in round 1 and remains present.

Suggested fix shape:
- Import `AudioClass` from `_sparrow_engine_core` and add it to `__all__` alongside `AudioSegment`/`AudioResult`.

## Verified round-1 fixes without new finding
- CPU/GPU audio V1/V2 top-level result arrays now return `data = null` when `len = 0`.
- CPU/GPU V2 class arrays return `classes = null` when `classes_len = 0` and preserve non-empty arena pointers.
- CPU/GPU V2 label conversion removes embedded NUL bytes before creating C strings.
- CPU symbol test now pins V2 audio functions and checks expected/actual exported symbol set equality.
- CPU Perch2 ignored FFI integration exercises `detect_audio_v2`, top-5 classes, label UTF-8, confidence/top-1 parity, and the V2 free path.

## Not treated as FFI/bindings findings here
- `sparrow-engine-server/src/handlers/audio.rs:109` still uses `model_id` as every stored drift label; this is a carried-forward server/drift issue, not an FFI-surface issue.
- `sparrow-engine-server/src/response.rs` optional-label serialization coverage is present for mixed labeled/unlabeled multiclass entries.
