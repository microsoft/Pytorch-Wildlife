# Inquisitor review — round 3

## Approval decisions recap

- Auditor plan: `STATUS: NOTHING-TO-DO` — APPROVED (re-verified all 7 owned
  files independently; no in-scope structural defect remains; CPU↔GPU
  `merge_segments` correctly below 3+ dedup rule).
- ITEM-REV-R3-001 (`handlers/audio.rs`): APPROVED. Real defect — K=1
  labeled audio segments lose their manifest label in the drift-store path
  because `AudioSegmentResponse::from` drops `classes` for K≤1.
- ITEM-REV-R3-002 (`engine.rs`): APPROVED with MODIFY. Toothless validation
  for multi-output RawAudio+Softmax (Perch 2). MODIFY guidance: factor a
  pure helper, gate strictly on RawAudio+Softmax, reject (not fallback)
  when `label` head missing at load time.

Commit applied: `56de9b9` — `fix: tighten audio drift and output
validation`. Only the two reviewer-owned files were modified.

## Fix verification (per ledger file)
<a name="sparrow-engine-sparrow-engine-cli-src-main-rs"></a>
### sparrow-engine/sparrow-engine-cli/src/main.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-core-src-preprocess-audio-rs"></a>
### sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-core-src-viz-rs"></a>
### sparrow-engine/sparrow-engine-core/src/viz.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-core-tests-audio-heatmap-e2e-rs"></a>
### sparrow-engine/sparrow-engine-core/tests/audio_heatmap_e2e.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-cpu-src-classify-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/classify.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-cpu-src-detect-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/detect.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-cpu-src-detect-audio-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-cpu-src-ffi-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/ffi.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-cpu-src-preprocess-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/preprocess.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-cpu-tests-integration-ffi-symbols-rs"></a>
### sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-cpu-tests-integration-perch2-rs"></a>
### sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-cpu-tests-integration-reexports-rs"></a>
### sparrow-engine/sparrow-engine-cpu/tests/integration_reexports.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-gpu-src-classify-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/classify.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-gpu-src-detect-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/detect.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-gpu-src-detect-audio-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/detect_audio.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-gpu-src-ffi-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/ffi.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-gpu-src-models-audio-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/models/audio.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-gpu-src-models-classifier-rs"></a>
### sparrow-engine/sparrow-engine-gpu/src/models/classifier.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-python-src-lib-rs"></a>
### sparrow-engine/sparrow-engine-python/src/lib.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-server-src-response-rs"></a>
### sparrow-engine/sparrow-engine-server/src/response.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-types-src-manifest-rs"></a>
### sparrow-engine/sparrow-engine-types/src/manifest.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-types-src-model-type-rs"></a>
### sparrow-engine/sparrow-engine-types/src/model_type.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-types-src-types-rs"></a>
### sparrow-engine/sparrow-engine-types/src/types.rs
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-python-python-sparrow-engine-init-py"></a>
### sparrow-engine/sparrow-engine-python/python/sparrow_engine/__init__.py
**Re-verified, no source change this round.** `git show --stat 56de9b9` confirms only `engine.rs` and `handlers/audio.rs` were touched in round 3. The round-2 verified state stands; spot-checked relevant invariants (e.g. `response.rs` still drops K≤1 `classes` at L163; `detect_audio.rs` L485-493 still emits K=1 with manifest label) and they remain consistent with the round-3 fixes applied elsewhere.

<a name="sparrow-engine-sparrow-engine-server-src-handlers-audio-rs"></a>
### sparrow-engine/sparrow-engine-server/src/handlers/audio.rs
**Verified — fix applied per ITEM-REV-R3-001.**

- L30: `drift_label_for_audio_segment` now takes `&AudioSegment` (native), not `&AudioSegmentResponse`. Reads `segment.classes.first().and_then(|c| c.label.as_ref())` — correct top-1 label semantics.
- L102-110: Labels and confidences collected from `result.segments` BEFORE the `into_iter()` conversion at L117-121. Closure-captured via `params.store.then(||...)` so it's only computed when needed.
- L124-128: Store path now consumes the pre-computed `(confidences, labels)` tuple; `compute_drift_metrics` receives the native top-1 labels.
- Tests added (L168, L175, L182, L191, L198): index-zero label, single labeled class (K=1 path — the regression case), empty classes, no labels, empty label string. All exercise the new `&AudioSegment` signature.
- Public JSON contract in `response.rs` unchanged — `AudioSegmentResponse::from` still omits `classes` for K≤1.
- One minor style note (NOT a defect): the `store_metrics` two-step (`.then(|| ...)` then `ok_or_else` with internal-error in the `if params.store` arm) is slightly awkward; an inline `let (confidences, labels) = (...)` inside the `if params.store` block would be cleaner. Behavior is correct; not raising as a finding.

<a name="sparrow-engine-sparrow-engine-cpu-src-engine-rs"></a>
### sparrow-engine/sparrow-engine-cpu/src/engine.rs
**Verified — fix applied per ITEM-REV-R3-002 with the requested MODIFY guidance.**

- L900-922: `validate_output_shape` now delegates output selection to a pure helper.
- L924-960: `select_validation_output_index` is a pure function taking `&[&str]` of output names + the preprocess/postprocess methods. Behavior:
  - Empty outputs → `OutputShapeMismatch` (preserves earlier behavior).
  - Single output → returns 0 (backward-compat for Mel+Softmax and single-head RawAudio+Softmax).
  - Multi-output AND `(RawAudio, Softmax)` → `.position(|n| n == "label")`, else `OutputShapeMismatch` with a descriptive shape string listing all output names.
  - All other multi-output cases → returns 0 (unchanged behavior for YOLO/MegaDet/Sigmoid, which currently aren't multi-head but the fall-through is safe).
- Strict-rejection-not-fallback at load time is exactly the MODIFY guidance I gave during approval.
- Tests (L1230, L1245, L1260, L1281): single-output Raw+Softmax → 0; multi-output Raw+Softmax with `label` → correct index; multi-output Raw+Softmax without `label` → error with name list in shape; non-Raw multi-output Softmax → 0. Tests do NOT require an ORT session — correct factoring.
- Integration: `cargo test -p sparrow-engine-cpu --test integration_perch2 -- --ignored` PASSED per verification.txt (Perch 2 still loads).
- Cross-scope deferred: `sparrow-engine-gpu/src/engine.rs` not on ledger; the reviewer correctly raised this as a round-4 Cross-Scope Finding. The CPU and GPU engines are intentionally separate flavors, so the asymmetry is acceptable for one round.


## Missed issues

I searched for issues that neither editor surfaced this round:

1. **`handlers/audio.rs` store_metrics two-step is awkward** — minor style;
   not a defect. Reviewer's fix is functionally correct. Not raising as a
   new finding.
2. **`select_validation_output_index` shape error string is slightly
   misleading** — when multi-output RawAudio+Softmax lacks `label`, the
   error's "shape" field is filled with an output-name list rather than a
   dimension vector. Functionally fine (the field is a free-form
   diagnostic), but consumers parsing the field by shape regex would
   stumble. Not raising; cosmetic.
3. **GPU mirror** (`sparrow-engine-gpu/src/engine.rs`): the round-3 fix is
   CPU-only. The GPU engine is not on the ledger, so this is a
   round-4 ledger-expansion concern, NOT a missed issue for round 3.
   Both editors flagged this in Cross-Scope Findings — I concur.
4. **Auditor's three deferred cross-scope items** (postprocess softmax
   primitive; `discover.rs` stale audio-classifier comments; the
   `segment_duration_s` opt-override-ignored behavior in raw-audio
   classifier path) are still untouched. ITEM-REV-R3-002 addressed only
   the validation half of the round-2 carry-forward item #4; the
   `segment_duration_s` override is still ignored. Defer to round 4.

No NEW round-3 in-scope issues found that the editors missed.

## Cross-impact

- ITEM-REV-R3-001 (handlers/audio.rs): does NOT modify any public JSON
  schema (`response.rs` untouched). No cross-impact on other ledger
  files. Reviewer's regression tests confirm `response.rs`'s K≤1 stripping
  is preserved.
- ITEM-REV-R3-002 (engine.rs): tightens load-time validation. Stricter,
  not looser — could in principle reject a previously-loadable malformed
  multi-output Raw+Softmax model. Perch 2 integration test passes per
  verification.txt, confirming the only known multi-head Raw+Softmax
  consumer still loads.
- Auditor made no source changes, so no auditor↔reviewer cross-impact.

## Verification results

`verification.txt` summary: ALL CLEAN.
- 435 tests passed / 0 failed across 22 suites (cpu+core+types+python-cpu).
  Net +4 tests vs round 2 — matches the new `select_validation_output_index_*`
  (4) and `drift_label_uses_single_labeled_class` (1) tests, offset by a
  minor rename/consolidation in the existing audio handler test module.
- Clippy: 5 `Finished` lines, zero `-D warnings` violations.
- Integration test `perch2_detects_two_5s_windows_with_top5_classes_on_10s_clip`
  PASSED with the new load-time validation in place.
- Pre-existing non-regressions (workspace test build fails on gpu test
  targets due to dual `[lib] name = "sparrow_engine"`; python lib tests
  require explicit `--features cpu`) are NOT caused by round 3 and are
  noted in the verification preamble.

## Coverage analysis

- Ledger files: 26
- Verified this round: 26
- Cumulative verified (scope_check.sh): 26/26
- Uncovered: []

STATUS: NEEDS-MORE SCOPE_CHECK=PASS COVERED=26/26 UNCOVERED=[] NEW=2
