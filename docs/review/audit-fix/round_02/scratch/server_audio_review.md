# Server audio review notes

Status: findings.

## Scope

Read-only review of `sparrow-engine-server/src/handlers/audio.rs` and `sparrow-engine-server/src/response.rs` for the round-1 deferred finding: `store=true` drift labels use `model_id` for every audio segment and ignore `AudioSegment.classes[0].label`.

## Required context read

- `~/.copilot/skills/_shared/iterative-anti-drift.md`: ledger/coverage protocol; round needs append-only scope and full verification coverage.
- `docs/review/audit-fix/SCOPE_LEDGER.json`: confirms `sparrow-engine-server/src/handlers/audio.rs` and `sparrow-engine-server/src/response.rs` are in scope.
- `docs/review/audit-fix/COVERAGE_LOG.jsonl`: round 1 logged the known deferred server audio issue through inquisitor verification.
- `docs/review/audit-fix/round_01/inquisitor_review.md`: lines 131-134 identify the exact deferred bug.
- `docs/review/audit-fix/round_02/file_ownership.md`: lines 12-30 assign both server files to Reviewer.

## Source observations

1. `handlers/audio.rs` currently builds the public response by consuming `result.segments` into `AudioSegmentResponse` at lines 91-101, then computes drift at lines 103-115.
2. Drift confidences use per-segment confidence at line 108, but labels are `vec![model_id.clone(); response.segments.len()]` at line 109.
3. The line 104-107 comment is now stale for Phase 4.2+ audio: `AudioSegment` can carry top-K classes. `sparrow-engine-types/src/types.rs` documents `classes[0]` as top class and `confidence == classes[0].probability` when classes is non-empty.
4. `response.rs` intentionally hides `classes` from the public JSON when `s.classes.len() <= 1` (lines 161-176) and preserves full top-K only when there are multiple classes. This is fine for the API surface, but it means drift labels should be derived from the engine `AudioSegment` before response conversion if single-entry labeled segments should keep their class label internally.
5. `AudioClassResponse.label` is optional and skipped in JSON when `None` (response.rs lines 144-150), and existing tests already cover unlabeled non-top classes. There is no test for store/drift label selection.

## Correct label fallback semantics

Use this per segment for drift `class_labels`:

1. If `segment.classes.first().and_then(|c| c.label.as_deref())` is `Some(label)`, use that top-1 label.
2. If `classes` is empty, fall back to `model_id`.
3. If the top-1 class exists but `label` is `None`, fall back to `model_id`.
4. Do not use a lower-ranked class label when the top-1 label is `None`; drift should bucket the observed top class only. A lower-ranked label would mislabel the segment.
5. Do not invent class-index string buckets unless the manifest/schema establishes that convention. The existing documented fallback bucket is `model_id`, and `AudioClass.label == None` means there is no resolved label file.

Implementation implication for Reviewer: compute drift inputs from `result.segments` before `.into_iter().map(AudioSegmentResponse::from)` consumes them, or clone/save labels first. Computing from `response.segments` would miss single-entry labeled `classes` because `response.rs` deliberately omits them.

## Suggested tests within owned files

Best fit: add private helper + unit tests in `sparrow-engine-server/src/handlers/audio.rs`.

Suggested helper shape:

```rust
fn audio_segment_drift_label(segment: &AudioSegment, model_id: &str) -> String {
    segment
        .classes
        .first()
        .and_then(|c| c.label.as_deref())
        .unwrap_or(model_id)
        .to_string()
}
```

Suggested unit tests:

1. `audio_drift_label_uses_top1_label_for_multiclass_segment` — classes `[Some("sparrow"), Some("warbler")]` returns `"sparrow"`, not model_id.
2. `audio_drift_label_falls_back_to_model_id_for_empty_classes` — empty classes returns model_id.
3. `audio_drift_label_falls_back_to_model_id_when_top1_label_missing` — first class `label=None`, second class `Some("warbler")`; returns model_id, proving lower-ranked labels are not substituted.
4. Optional: `audio_drift_labels_are_computed_before_response_conversion` if Reviewer factors a `audio_drift_inputs(&[AudioSegment], model_id)` helper; single-class labeled segment returns its label even though `AudioSegmentResponse` omits `classes`.

`response.rs` already has serialization tests for empty/single/multiclass/unlabeled entries. Add there only if Reviewer moves label-selection helper into `response.rs`; otherwise handler tests are more direct and avoid changing public JSON semantics.

## Finding

FINDING SERVER-AUDIO-DRIFT-LABELS: `store=true` audio drift labels still collapse every segment to `model_id`, so Perch 2/RawAudio softmax outputs lose top-1 class identity before PSI. Fix in `handlers/audio.rs` by deriving drift labels from pre-response `AudioSegment.classes[0].label` with model_id fallback for empty classes or missing top-1 label.
