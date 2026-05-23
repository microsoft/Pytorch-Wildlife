# Auditor report — round 2

Commit: `eaa1fd35d2431988c4f94b85452674be3dfb0533`

One in-scope item proposed and approved this round (ITEM-AUD-R2-001).
Other 6 owned files reviewed and intentionally not touched per the
round-2 plan; no `-V2` re-proposals (all round-1 auditor items were
APPROVED + applied, none REJECTED).

## Changes Applied

### <a name="ITEM-AUD-R2-001"></a>ITEM-AUD-R2-001 — viz.rs `classes` field re-indent

- `sparrow-engine/sparrow-engine-core/src/viz.rs` lines 1057, 1083,
  1102, 1115, 1135, 1150, 1168 (seven sites — plan body originally
  said "six" but enumerated seven; per inquisitor nit the count is
  corrected here in the report).
  - before: `classes: Vec::new(),` at column 9 (eight leading
    spaces), one indent step shallower than sibling struct-literal
    fields `start_time_s`/`end_time_s`/`confidence`.
  - after: re-indented to match the sibling indent. Six of the
    seven sites land at column 13 (twelve leading spaces). The one
    site nested inside `vec![AudioSegment { … }]` inside an outer
    struct literal (`audio_segments_to_annotations_normalizes_time`,
    line 1115) lands at column 17 (sixteen leading spaces) to match
    its deeper siblings.
- why: same bug pattern fixed by round-1 ITEM-AUD-003 in
  `sparrow-engine-gpu/src/detect_audio.rs:222,228,234`. Pure
  rustfmt-equivalent readability fix; struct literal is
  semantically identical regardless of leading whitespace, but the
  misaligned field visually breaks the block and invites future
  formatting churn.
- verification: `cargo check -p sparrow-engine-core --tests` clean
  (1.31 s). `git diff` is a 7-line indentation-only delta — no
  token-level change.

## Tests Added/Updated

None. ITEM-AUD-R2-001 is pure whitespace; the affected lines are
already test bodies and the field shape is unchanged. Existing viz
tests cover the heatmap render paths involved.

## Cross-Scope Findings

Carried forward to round 3 (or beyond) — outside auditor's owned-file
set or out of structural scope:

1. **CPU↔GPU `merge_segments` / `merge_segments_with_class` verbatim
   copy.** 2 sites in `sparrow-engine-{cpu,gpu}/src/detect_audio.rs`;
   GPU file's doc literally says "Verbatim from sparrow-engine-cpu
   (no GPU dependency)". Same drift-admission pattern that round-1
   AUD-001 caught for `compute_segment_offsets`. Below the 3+ dedup
   threshold; the natural home (`sparrow-engine-types` next to
   `AudioRange`, or a new `sparrow-engine-core::postprocess_audio`
   module) lives outside the auditor's editable surface this round.
   Re-flag for round 3 if a third caller appears or if the team
   judges 2-site verbatim copy worth a pre-emptive lift.
2. **`postprocess::try_softmax` reuse blocker** — carry-over from
   round 1 cross-scope. `postprocess.rs` outside owned set; tripwire
   doc planted at `sparrow-engine-cpu/src/detect_audio.rs:675-685`
   remains.
3. **`resolve_classifier_output` fallback vs docstring**
   (CPU detect_audio.rs:267). `unwrap_or(0)` actually fires for ANY
   multi-head classifier without a `"label"`-named output, not just
   single-output classifiers as the doc claims. Behavioural edge,
   out of structural scope.
4. **Reviewer's round-1 behavioural findings on detect_audio.rs**
   (carried forward): RawAudio `segment_duration_s` override is
   ignored (`window_samples` wins); RawAudio top-K batching
   validates only flattened output length, should require rank-2
   `[batch, num_classes]` shape.

## Skipped

- The 6 other owned files reviewed in the plan (`preprocess_audio.rs`,
  `audio_heatmap_e2e.rs`, CPU+GPU `detect_audio.rs`, GPU `audio.rs`,
  GPU `classifier.rs`) — no in-scope structural finding meeting the
  3+ dedup rule or other simplification/readability bar. Rationales
  enumerated in `round_02/auditor_plan.md` ("Files NOT modified"
  section).
- No REJECTED items to skip.

STATUS: DONE COMMIT=eaa1fd35d2431988c4f94b85452674be3dfb0533
