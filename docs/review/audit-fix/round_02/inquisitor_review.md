# Inquisitor review — round 2

## Approval decisions recap

Phase 1 approvals (see `round_02/inquisitor_approvals.md`):

| Item | Phase 1 verdict | Phase 2 outcome |
|------|------------------|------------------|
| ITEM-AUD-R2-001 (viz.rs re-indent ×7) | APPROVED | applied in commit `eaa1fd3` |
| ITEM-REV-R2-001 (tighten `derive_model_type`) | MODIFY (require engine.rs ledger append + test patch in same round) | both edits applied in commit `20aa643`; ledger appended `sparrow-engine-cpu/src/engine.rs` (`added_round=2`) |
| ITEM-REV-R2-002 (Python re-export `AudioClass`) | APPROVED | applied in commit `20aa643` |
| ITEM-REV-R2-003 (drift labels from segment top-1) | APPROVED (index-0 lookup, no re-sort) | applied in commit `20aa643` — uses `classes.first()` as required |
| ITEM-REV-R2-004 (null-on-empty FFI arrays) | APPROVED | applied in commit `20aa643` (cpu+gpu, detect/classify/pipeline) |

## Fix verification (per ledger file)
<a name="sparrow-engine-sparrow-engine-cli-src-main-rs"></a>
### sparrow-engine/sparrow-engine-cli/src/main.rs
VERIFIED — no edit this round. Reviewer plan: no regression in round-1 visualize threshold policy. Source unchanged since round 1; no new issue surfaced.

<a name="sparrow-engine-sparrow-engine-core-src-preprocess-audio-rs"></a>
### sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs
VERIFIED — no edit this round. Auditor plan rationale (helpers `compute_segment_offsets` @311 / `segment_time_range` @334 in place from round 1) confirmed by direct read. Cosmetic double-blank-line gap noted but not a defect.

<a name="sparrow-engine-sparrow-engine-core-src-viz-rs"></a>
### sparrow-engine/sparrow-engine-core/src/viz.rs
VERIFIED FIX — ITEM-AUD-R2-001 applied. Spot-checked lines 1057, 1102, 1115: `classes: Vec::new(),` now aligned at col 13 (col 17 for the nested literal at 1115), matching sibling struct fields. Pure whitespace delta confirmed by report. cargo check -p sparrow-engine-core --tests clean.

<a name="sparrow-engine-sparrow-engine-core-tests-audio-heatmap-e2e-rs"></a>
### sparrow-engine/sparrow-engine-core/tests/audio_heatmap_e2e.rs
VERIFIED — no edit this round. Auditor confirmed the three one-line `classes: Vec::new()` literals at 69-71 do not have the indent defect.

<a name="sparrow-engine-sparrow-engine-cpu-src-classify-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/classify.rs
VERIFIED — no edit this round. Reviewer: no behavioral issue. Unchanged.

<a name="sparrow-engine-sparrow-engine-cpu-src-detect-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/detect.rs
VERIFIED — no edit this round. Reviewer: no behavioral issue. Unchanged.

<a name="sparrow-engine-sparrow-engine-cpu-src-detect-audio-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs
VERIFIED — no edit this round. Auditor reviewed and explicitly justified non-edit: `outputs.len() == 0` idiom retained (`ort::SessionOutputs` lacks `.is_empty()`); three `pin_session` sites have distinct output-shape extraction and don't pass the 3+ dedup bar. Tripwire doc at L675-685 still planted re `try_softmax` reuse (deferred). Cross-scope behavioral findings (segment_duration_s override, top-K rank-2 validation, merge_segments dedup) carried to round 3.

<a name="sparrow-engine-sparrow-engine-cpu-src-ffi-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/ffi.rs
VERIFIED FIX — ITEM-REV-R2-004 applied. Confirmed: L343 `data` now `is_empty()? null : as_ptr()`; same conditional pattern at L419 (top_results), L518 (pipeline data). Audio paths from round 1 unchanged (L568, L658). New tests: `detect_result_to_c_uses_null_data_for_empty_detections` @2046, `classify_result_to_c_uses_null_top_results_for_empty_classifications` @2066, `pipeline_result_to_c_uses_null_data_for_empty_detections` @2086.

<a name="sparrow-engine-sparrow-engine-cpu-src-preprocess-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/preprocess.rs
VERIFIED — no edit this round. Reviewer: no new issue. Unchanged.

<a name="sparrow-engine-sparrow-engine-cpu-tests-integration-ffi-symbols-rs"></a>
### sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs
VERIFIED — no edit this round. Unchanged; reviewer noted no regression.

<a name="sparrow-engine-sparrow-engine-cpu-tests-integration-perch2-rs"></a>
### sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs
VERIFIED — no edit this round. Unchanged.

<a name="sparrow-engine-sparrow-engine-cpu-tests-integration-reexports-rs"></a>
### sparrow-engine/sparrow-engine-cpu/tests/integration_reexports.rs
VERIFIED — no edit this round. Unchanged.

<a name="sparrow-engine-sparrow-engine-gpu-src-classify-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/classify.rs
VERIFIED — no edit this round. Reviewer: no behavioral issue. Unchanged.

<a name="sparrow-engine-sparrow-engine-gpu-src-detect-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/detect.rs
VERIFIED — no edit this round. Reviewer: no behavioral issue. Unchanged.

<a name="sparrow-engine-sparrow-engine-gpu-src-detect-audio-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/detect_audio.rs
VERIFIED — no edit this round. Auditor reviewed: round-1 AUD-003 indent fix in place; merge_segments verbatim-copy cross-scope re-flagged to round 3 (still only 2 callers, below 3+ rule).

<a name="sparrow-engine-sparrow-engine-gpu-src-ffi-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/ffi.rs
VERIFIED FIX — ITEM-REV-R2-004 applied (gpu mirror). Confirmed conditional null-on-empty at L352 (detections), L428 (top_results), L527 (pipeline data); audio paths at L577/L667 unchanged from round 1. New tests at L2054/2074/2094 mirror CPU; audio V1/V2 tests at L2184/2204 retained.

<a name="sparrow-engine-sparrow-engine-gpu-src-models-audio-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/models/audio.rs
VERIFIED — no edit this round.

<a name="sparrow-engine-sparrow-engine-gpu-src-models-classifier-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/models/classifier.rs
VERIFIED — no edit this round. Only Perch 2 touch points are defense-in-depth audio-rejection arms at L467 & L780; same disposition as round 1.

<a name="sparrow-engine-sparrow-engine-python-src-lib-rs"></a>
### sparrow-engine/sparrow-engine-python/src/lib.rs
VERIFIED — no edit this round. AudioClass pyclass at L312 + module registration at L1419 already in place from prior work; ITEM-REV-R2-002 only needed Python-side re-export.

<a name="sparrow-engine-sparrow-engine-server-src-response-rs"></a>
### sparrow-engine/sparrow-engine-server/src/response.rs
VERIFIED — no edit this round. `AudioSegmentResponse::from` (L161-184) only populates `classes` when `len > 1`; that shape is exactly what the new `drift_label_for_audio_segment` helper consumes via `classes.as_ref().and_then(.first())`. Consistent.

<a name="sparrow-engine-sparrow-engine-types-src-manifest-rs"></a>
### sparrow-engine/sparrow-engine-types/src/manifest.rs
VERIFIED — no edit this round. Audio matrix enforcement at L899-910 confirmed; this is the authoritative source for ITEM-REV-R2-001's tightening rationale. Unchanged.

<a name="sparrow-engine-sparrow-engine-types-src-model-type-rs"></a>
### sparrow-engine/sparrow-engine-types/src/model_type.rs
VERIFIED FIX — ITEM-REV-R2-001 applied. Direct read confirms: (Mel,Sigmoid)→AudioDetector and (RawAudio,Softmax)→AudioClassifier are the ONLY audio match arms; the previous (Mel,Softmax) and (RawAudio,Sigmoid) arms are removed and fall through to (`_,Softmax)→Classifier` / wildcard→Detector. The Overhead-promotion guard was strengthened with an `is_audio_preprocess` flag so even an audio fallback that resolves to base==Detector cannot be promoted. In-file tests refreshed accordingly.

<a name="sparrow-engine-sparrow-engine-types-src-types-rs"></a>
### sparrow-engine/sparrow-engine-types/src/types.rs
VERIFIED — no edit this round. Shared audio type shape already matches round-1 fix surface. Unchanged.

<a name="sparrow-engine-sparrow-engine-python-python-sparrow-engine-init-py"></a>
### sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py
VERIFIED FIX — ITEM-REV-R2-002 applied. `AudioClass` is imported alongside `AudioSegment`/`AudioResult` at L15 and present in `__all__` (between Classification and AudioSegment). Pure additive.

<a name="sparrow-engine-sparrow-engine-server-src-handlers-audio-rs"></a>
### sparrow-engine/sparrow-engine-server/src/handlers/audio.rs
VERIFIED FIX — ITEM-REV-R2-003 applied. `drift_label_for_audio_segment` at L28-36 uses `classes.as_ref().and_then(|c| c.first()).and_then(|c| c.label.as_ref()).cloned().unwrap_or_else(|| model_id.to_string())` — index-0 lookup, no re-sort, falls back to model_id when classes are None/empty or top-1 label is None. Four new tests: index-0-no-resort, classes-missing-or-empty fallback, top-1-unlabeled fallback, empty-string-label preserved. Exactly matches Phase 1 acceptance refinement.

<a name="sparrow-engine-sparrow-engine-cpu-src-engine-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/engine.rs
VERIFIED FIX (newly appended this round, added_round=2). The in-crate `model_type_from_preprocess_postprocess` test at L1267-1272 now asserts `(mel, Softmax) ⇒ Classifier` (was AudioClassifier) with a descriptive failure message citing manifest rejection. This is the dependent test patch I required in Phase 1 to keep `cargo test -p sparrow-engine-cpu` green; verification.txt confirms 399/399 pass. Ledger anti-narrowing clause honored: appended, not narrowed.


## Missed issues

None. I re-read all 26 ledger files; the auditor's "no edit" rationales for the 6 unedited files in its own ownership and the reviewer's "no edit" rationales for its 14 unedited files (CLI, image classify/detect, preprocess, manifest, types, response, python lib, integration tests) are defensible. No latent bug pattern survived the round that I'm aware of.

Carry-forward cross-scope items (already flagged by editors; not regressions, deferred to round 3):
- CPU↔GPU `merge_segments`/`merge_segments_with_class` verbatim copy (2 sites, below 3+ dedup rule).
- `postprocess::try_softmax` reuse blocker (tripwire doc in CPU detect_audio.rs:675-685).
- `resolve_classifier_output` doc-vs-code drift at CPU detect_audio.rs:267.
- RawAudio behavioral edges: `segment_duration_s` opt override ignored; top-K validates flat length not rank-2 shape.
- `sparrow-engine-server/src/discover.rs` audio-classifier fixture comments still describe Mel+Softmax as AudioClassifier — stale post-REV-R2-001 (comment-only, not in ledger this round).

## Cross-impact

- REV-R2-001 tightening of `derive_model_type` is internally consistent with `manifest.rs:899-910` (manifest already enforces the same matrix). No other consumer of `derive_model_type` (grepped: pyo3 binding, server discovery, CLI) asserts the now-removed combos at runtime; the only place that did was the in-crate engine.rs test, which was updated in the same commit.
- REV-R2-003 `drift_label_for_audio_segment` reads `AudioSegmentResponse` shape produced by `response.rs::From<AudioSegment>`. That impl only populates `classes` when `len>1`, so the helper's `None`/empty fallback path is the binary-detector path. Behaviorally consistent with prior round-1 audio drift semantics for binary detectors (single-bucket PSI).
- REV-R2-004 null-on-empty changes apply to the C ABI surface; consumers must check `len==0` before dereferencing `data` regardless, so this tightens (does not relax) the safety contract. Audio V1/V2 sites from round 1 retain the same convention.
- ITEM-AUD-R2-001 is whitespace-only and cannot affect semantics.

No cross-impact regressions detected.

## Verification results

Per `round_02/verification.txt`:
- cpu workspace (types+core+cpu): 399 pass / 0 fail.
- sparrow-engine-python --lib --no-default-features --features cpu: 32 pass / 0 fail.
- Total: 431/431 in 22 suites.
- Clippy clean: workspace (excl python), python --features cpu, gpu --all-targets.
- Pre-existing not-in-scope failures documented (python without --features cpu compile_error gate; `cargo test --workspace --no-run` blocked by dual `[lib] name = "sparrow_engine"` + cpu-as-dev-dep-of-gpu). Bisected to pre-R2 baseline (`f92e2c5`); not a round-2 regression.

I additionally spot-verified the in-source state of the six modified files via direct read (viz.rs L1057/1102/1115, model_type.rs L24-46, engine.rs L1267-1272, __init__.py L14-28+35-67, cpu+gpu ffi.rs null-on-empty conditionals + new tests at L2046/2066/2086, handlers/audio.rs L28-36 + tests L160-194) — all match the reviewer/auditor reports.

## Coverage analysis

Ledger size grew from 25 → 26 with the appended `sparrow-engine/sparrow-engine-cpu/src/engine.rs` (anti-narrowing-compliant append). All 26 ledger files have a `verified` coverage_log entry from this inquisitor pass.

`scope_check.sh` returned `covered=26/26 uncovered=[]` → SCOPE_CHECK=PASS.

## Convergence judgment

Both editors made approved+applied edits this round (5 items total: AUD-R2-001 + REV-R2-001..004). Per the convergence rules, NOTHING-TO-DO at Phase 1 is required for CONVERGED; because there were 5 approved-applied items, NEW=5 and this round must be NEEDS-MORE. No new issues raised by me beyond the items already approved and applied. Verification was clean and scope coverage is complete, so a round-3 NOTHING-TO-DO pass is realistic.

STATUS: NEEDS-MORE SCOPE_CHECK=PASS  COVERED=26/26  UNCOVERED=[] NEW=5
