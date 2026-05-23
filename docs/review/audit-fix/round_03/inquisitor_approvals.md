# Inquisitor Approvals — round 3 (APPROVAL phase)

## Auditor plan

Auditor returned `STATUS: NOTHING-TO-DO`. I independently re-read each of the
seven owned files (`detect_audio.rs` CPU/GPU, `preprocess_audio.rs`, `viz.rs`,
`audio_heatmap_e2e.rs`, `gpu/models/audio.rs`, `gpu/models/classifier.rs`) and
confirm no in-scope structural defect remains:

- `viz.rs` audio test literals are now consistently indented (round-2 fix).
- `preprocess_audio.rs` shared helpers (`compute_segment_offsets`,
  `segment_time_range`) are correctly reused; no new 3+ dedup hit.
- `cpu/detect_audio.rs` keeps `outputs.len() <= logits_idx` style and the
  softmax/topk tripwire doc at L675-685 intact.
- `gpu/detect_audio.rs` `merge_segments` duplication remains at **2 call
  sites** (CPU+GPU) — correctly below the 3+ dedup rule. Auditor cited this
  correctly.
- `audio_heatmap_e2e.rs`, `gpu/models/{audio,classifier}.rs` — nothing new.

Auditor's "Cross-Scope Findings" (postprocess softmax primitive, RawAudio
opt override + top-K shape, `discover.rs` stale comments) are properly
deferred — items 2 and 3 there overlap with reviewer's ITEM-REV-R3-002 and
ITEM-REV-R3-001 in spirit but the auditor correctly did not edit them.

**Verdict on auditor plan: APPROVED — NOTHING-TO-DO confirmed.**

## Reviewer plan

Reviewer returned `STATUS: PLAN-READY` with two items. I evaluated each
against the ledger, the round-2 carry-forward list, and the actual source.

<a name="ITEM-REV-R3-001"></a>
### ITEM-REV-R3-001 — handlers/audio.rs drift label loss for labeled K=1

**Decision: APPROVED.**

Independent verification:
- `response.rs:161-176` — `AudioSegmentResponse::from` drops `classes` when
  `s.classes.len() <= 1`. Confirmed.
- `cpu/detect_audio.rs:485-493` — binary-detector path emits exactly one
  `AudioClass { class_idx: 0, label: detector_label.clone(), probability }`,
  so K=1 segments carry a real label from the manifest.
- `handlers/audio.rs:106-119` — store/drift path iterates `response.segments`
  AFTER the `into_iter().map(AudioSegmentResponse::from)` conversion has
  already dropped `classes` for K=1. Result: the manifest's binary-detector
  label is silently replaced by `model_id` in the drift labels vector.

This is a real behavioral defect, in the reviewer's owned file
(`handlers/audio.rs` is on the ledger). Not a re-litigation: round 2 did not
raise this — it surfaced through the reviewer's round-3 sub-agent notes.

Implementation guidance (not a blocker): the natural fix is to compute
`labels` from `result.segments` (the native `AudioDetectResult`) before
consuming it via `into_iter()`, e.g. capture
`let native_labels: Vec<String> = result.segments.iter().map(...).collect();`
under the `if params.store` guard prior to the response conversion. Keep
`drift_label_for_audio_segment`'s public JSON behavior unaffected, and add
the requested in-file test for the labeled-K=1 path.

<a name="ITEM-REV-R3-002"></a>
### ITEM-REV-R3-002 — engine.rs RawAudio+Softmax: validate by output name

**Decision: APPROVED with MODIFY.**

Independent verification:
- `engine.rs:901-916` — `validate_output_shape` only ever inspects
  `outputs[0]`. Confirmed.
- `engine.rs:999-1013` — Softmax check accepts rank 1 or rank 2. Perch 2's
  first output is `embedding` (rank 2, e.g. `[1, 1280]`), which trivially
  passes — i.e. the load-time check is effectively a no-op for multi-head
  RawAudio+Softmax models.
- `cpu/detect_audio.rs:267-320` — runtime resolves the logits head by name
  `"label"` with fallback to output 0; `integration_perch2.rs` exercises
  this. Tightening load-time validation to mirror this resolution closes a
  real gap.
- This is the rank-2-shape half of round-2 carry-forward item #4
  ("top-K validates flat length not rank-2 shape"). Not re-litigation — it
  was explicitly deferred.

**MODIFY (non-blocking guidance):**

1. Preserve current behavior for single-output classifiers (Mel+Softmax,
   single-head RawAudio+Softmax) — the plan already says this; keep that
   explicit in code via `if outputs.len() == 1 { validate(outputs[0]) }`.
2. When multi-output AND `manifest.preprocess_method == RawAudio` AND
   `postprocess_method == Softmax`: select the `label`-named output; if
   absent, return `OutputShapeMismatch` rather than silently falling back
   to `outputs[0]` (the runtime path's `.unwrap_or(0)` fallback is fine for
   single-output models but at load time a multi-output model with no
   `label` head is malformed and should be rejected).
3. The proposed gating on `RawAudio + Softmax` only is correct — do NOT
   extend the name-based selection to Mel+Softmax (`engine.rs:1270`
   already forbids that combo via manifest validation).
4. Tests: factor the selection into a pure helper
   `select_validation_output_index(outputs: &[Outlet], preprocess, postprocess) -> Result<usize>`
   that takes a slice of `&dyn`-like accessors (or names + shapes) so a
   unit test can drive it without an ORT session.
5. GPU engine.rs is NOT on the ledger — the reviewer correctly noted this
   as a Cross-Scope Finding for round 4 ledger expansion. Do not edit GPU
   engine.rs this round.

## Cross-impact assessment

- ITEM-REV-R3-001 only touches `handlers/audio.rs` (and possibly a tiny
  refactor to compute labels pre-conversion). It does NOT alter the public
  JSON schema (`response.rs` is untouched), so it cannot affect any other
  ledger file.
- ITEM-REV-R3-002 changes load-time behavior in `engine.rs`. New rejection
  cases (multi-output RawAudio+Softmax lacking a `label` head) are stricter
  than today — that could theoretically reject a previously-loadable
  malformed model. That is the desired outcome but warrants a clear error
  message and a verification run of `integration_perch2.rs` to confirm
  Perch 2 still loads.

## Convergence outlook

Both approvals are valid round-3 work. Round 3 therefore will NOT converge
(NEW≥2 expected). Round 4 should pick up the deferred Cross-Scope items
(`gpu/src/engine.rs` mirror; `postprocess::softmax_probs` primitive;
`discover.rs` stale comments), expanding the ledger as needed.

STATUS: APPROVALS-DONE
