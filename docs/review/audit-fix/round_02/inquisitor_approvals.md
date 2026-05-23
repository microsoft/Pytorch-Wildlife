# Inquisitor approvals — round 2 (Phase 1)

I read each ledger target directly. Notes below cite the source state I observed.

---

## ITEM-AUD-R2-001 — viz.rs re-indent of `classes: Vec::new(),`

Decision: **APPROVED** (with minor wording nit)

Verification: viewed `sparrow-engine-core/src/viz.rs` lines 1050–1175 directly. All seven cited sites (1057, 1083, 1102, 1115, 1135, 1150, 1168) are indeed at column 9, while sibling struct-literal fields (`start_time_s`/`end_time_s`/`confidence`) are at column 13. Pure cosmetic fix; no behavior change. The pattern is identical to round-1 AUD-003 in `sparrow-engine-gpu/src/detect_audio.rs:222,228,234`, so the rationale is consistent with prior precedent.

Nit (does not block): plan body opens with "the six `classes: Vec::new(),` lines" but enumerates **seven** sites. Update the prose to "seven" before applying or after applying so the report stays auditable.

---

## ITEM-REV-R2-001 — Tighten `derive_model_type` audio matrix

Decision: **MODIFY** (substance APPROVED; scope must be expanded)

Verification:
- `sparrow-engine-types/src/model_type.rs:24-39` currently maps four audio combos: (Mel,Sigmoid)→AudioDetector, (Mel,Softmax)→AudioClassifier, (RawAudio,Sigmoid)→AudioDetector, (RawAudio,Softmax)→AudioClassifier.
- `sparrow-engine-types/src/manifest.rs:899-910` confirms the authoritative audio matrix: only `(Mel,Sigmoid)` and `(RawAudio,Softmax)` survive manifest validation; the other two return `InvalidManifest`. So tightening `derive_model_type` to match that matrix is legitimate.
- In-file tests at `model_type.rs:111-121` (`audio_classifier_when_mel_plus_softmax_either_subtype`) and `model_type.rs:285-291` (`audio_detector_when_raw_audio_plus_sigmoid`) currently assert the looser semantics. Plan acknowledges these need updating.

**Blocker for round-2 verification:** `sparrow-engine-cpu/src/engine.rs:1268` asserts `derive_model_type(&mel, &Softmax, std_sub) == AudioClassifier` (and the function is reachable as `pub fn` from the workspace re-export). If REV-R2-001 ships as planned, `cargo test -p sparrow-engine-cpu` fails — which violates the convergence rule "Verification clean." Reviewer flagged this as round-3 cross-scope, but per the anti-narrowing clause we may **append** new in-scope files, not defer broken builds.

**Required modification:**
1. Append `sparrow-engine/sparrow-engine-cpu/src/engine.rs` to `SCOPE_LEDGER.json` with `added_round: 2` (auditor's ownership boundary already includes the audio CPU stack, but engine.rs was not previously listed — anti-narrowing permits appending, not removing).
2. As part of ITEM-REV-R2-001, also update the engine.rs in-crate test at lines 1267-1282 so the `(mel, Softmax) ⇒ AudioClassifier` assertion (and any sibling RawAudio+Sigmoid assertion, if added later) reflects the new semantics.
3. Owner: since reviewer is making the source change to model_type.rs, reviewer should also patch the dependent test in engine.rs in the same round. Update `round_02/file_ownership.md` to reflect that, or have auditor pick it up — but it MUST be in round 2.

If the team prefers not to expand scope this round, the alternative is to **REJECT** REV-R2-001 and re-propose it in round 3 once engine.rs is added to the ledger. I lean toward MODIFY (do it now) because the looser pub semantics is a real consistency bug and waiting another round risks a third caller drifting onto the wrong matrix.

Cross-scope note from reviewer about `sparrow-engine-server/src/discover.rs` comments is acceptable to defer (comments only, no compile-time impact).

---

## ITEM-REV-R2-002 — Re-export `AudioClass` from Python `__init__.py`

Decision: **APPROVED**

Verification:
- `sparrow-engine-python/src/lib.rs:312` defines the `pyclass AudioClass` and `lib.rs:1419` registers it on the module (`m.add_class::<AudioClass>()?`), so the native symbol exists.
- `sparrow-engine-python/python/sparrow_engine/__init__.py:14-27` currently imports `AudioSegment` and `AudioResult` but NOT `AudioClass`, and `__all__` (lines 35-67) likewise omits it. Confirmed gap — users who receive `AudioClass` instances via `AudioSegment.classes` cannot reference `sparrow_engine.AudioClass` for isinstance checks or type annotations.

Pure additive change; no risk. Acceptance criteria (import + `__all__` + smoke check) are clear.

---

## ITEM-REV-R2-003 — Drift labels from segment top-1 class

Decision: **APPROVED** (with one acceptance refinement)

Verification:
- `sparrow-engine-server/src/handlers/audio.rs:108-115` currently hard-codes `labels = vec![model_id.clone(); response.segments.len()]`, losing Perch-2 class identity in drift logs.
- `sparrow-engine-server/src/response.rs:161-176`: `AudioSegmentResponse::from` only populates `classes: Some(...)` when `s.classes.len() > 1`. So when len ∈ {0,1} the response carries `classes: None`, and the handler MUST fall back to `model_id`. Plan acceptance already covers this: "Preserve model_id fallback for binary detector responses where response.classes is omitted."
- `AudioClassResponse.label: Option<String>` (response.rs:148) — None when the model has no label file. Plan correctly says "Do not substitute lower-ranked class labels when the top-1 label is None."

**Acceptance refinement (please honor):** The top-1 selection must NOT re-sort; it must read `classes.get(0)` (or equivalent index-0 lookup) because `AudioSegmentResponse::from` preserves the source-order semantics from `sparrow_engine::AudioSegment.classes`, which is already sorted top-K. If the new handler logic accidentally introduces a sort or `max_by`, that's a behavioral drift; spell out "index-0 lookup, no re-sort" in the implementation.

Also confirm: classes whose top-1 label is empty-string vs None — the plan says "label is None", so empty-string labels are passed through (consistent with manifest semantics; AudioClass.label is `Option<String>` so empty-string would be `Some("")`, not None). OK.

---

## ITEM-REV-R2-004 — Null-on-empty for non-audio FFI arrays

Decision: **APPROVED**

Verification:
- `sparrow-engine-cpu/src/ffi.rs:343` does `combined.header.data = combined._owner.detections.as_ptr();` unconditionally. When `result.detections.is_empty()`, `Vec::as_ptr()` returns the dangling sentinel `0x1` (or similar non-null) — exactly the bug pattern round 1 fixed for audio FFI.
- `sparrow-engine-gpu/src/ffi.rs:352` mirrors the same bug.
- The same pattern exists in `classify_result_to_c` (cpu/ffi.rs:~430 and gpu/ffi.rs:~440) for `top_results`, and in pipeline conversions — plan covers "detection data, classify top_results, and pipeline data" which is correct.

Approved. Acceptance criterion: unit tests must cover len=0 → ptr.is_null() for each of the three array kinds in both CPU and GPU crates. Make sure the GPU tests are runnable on CPU-only CI (no actual GPU init required for FFI struct-conversion tests).

---

## Summary

| Item | Decision |
|------|----------|
| ITEM-AUD-R2-001 | APPROVED (fix "six"→"seven" wording) |
| ITEM-REV-R2-001 | MODIFY (append `sparrow-engine-cpu/src/engine.rs` to ledger; patch engine.rs:1267-1282 test in same round) |
| ITEM-REV-R2-002 | APPROVED |
| ITEM-REV-R2-003 | APPROVED (no re-sort; index-0 lookup) |
| ITEM-REV-R2-004 | APPROVED |

The auditor's "no-edit" rationales for the other 6 owned files were spot-checked and look defensible (the merge_segments cross-scope deferral is consistent with round-1's defer rationale; the postprocess.rs softmax reuse blocker is unchanged).

STATUS: APPROVALS-DONE
