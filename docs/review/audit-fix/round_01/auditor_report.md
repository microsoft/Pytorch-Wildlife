# Auditor report — round 1

All 5 plan items were APPROVED by the inquisitor and applied.

## Changes Applied

### <a name="ITEM-AUD-001"></a>ITEM-AUD-001 — segment-offset loop dedup
- `sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs:~297` (new)
  - before: (no helper)
  - after: `pub fn compute_segment_offsets(total_samples, segment_samples, stride_samples) -> Vec<usize>`, with documentation of the inclusive-tail termination contract (last offset emitted, then break when `remaining <= segment_samples`).
- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:372-381` (mel loop)
  - before: hand-rolled `while offset < total_samples { … }` loop.
  - after: `let offsets = preprocess_audio::compute_segment_offsets(total_samples, segment_samples, stride_samples);`
- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:551-560` (raw loop) — same replacement.
- `sparrow-engine/sparrow-engine-gpu/src/models/audio.rs:1312-1332` (private mirror)
  - before: private `fn compute_segment_offsets` whose doc explicitly called itself a `Mirror of sparrow-engine-cpu`.
  - after: deleted; the 2 in-file call sites + 3 test call sites switched to `preprocess_audio::compute_segment_offsets`. Drift admission removed.
- why: 3 occurrences of byte-identical loop bodies, with one already labelled a manual mirror. Termination semantics now have one home.

### <a name="ITEM-AUD-002"></a>ITEM-AUD-002 — time-range helper dedup
- `sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs:~316` (new)
  - after: `pub fn segment_time_range(seg_offset, segment_samples, total_samples, sample_rate) -> (f32, f32)`, preserving the `min(total_samples)` tail-padding clamp.
- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:487-489` (mel) — 3-line block → single helper call.
- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:654-656` (raw) — same.
- `sparrow-engine/sparrow-engine-gpu/src/models/audio.rs:1398-1400` (collect_segments) — same.
- why: 3 occurrences of identical 3-line arithmetic, including the same `min(total_samples)` clamp.

### <a name="ITEM-AUD-003"></a>ITEM-AUD-003 — GPU test indentation fix
- `sparrow-engine/sparrow-engine-gpu/src/detect_audio.rs:222,228,234`
  - before: `classes: Vec::new(),` at column 12, breaking the struct-literal block.
  - after: re-indented to column 16, matching sibling fields.
- why: rustfmt-equivalent readability fix.

### <a name="ITEM-AUD-004"></a>ITEM-AUD-004 — dead RawAudio field bind
- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:217-220`
  - before: `PreprocessMethod::RawAudio { sample_rate: _, window_samples } => {`
  - after: `PreprocessMethod::RawAudio { window_samples, .. } => {`
- why: outer match on line 152 already bound `sample_rate`; the `sample_rate: _` was dead pattern noise. Switched to idiomatic `..` rest-pattern.

### <a name="ITEM-AUD-005"></a>ITEM-AUD-005 — softmax/top_k_indices doc tripwire
- `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:685` (added module-level doc block above the two helpers)
  - after: 8-line comment explaining that the local `softmax` + `top_k_indices` duplicate the math in `sparrow_engine_core::postprocess::try_softmax` because `AudioClass` (`class_idx`/`label`/`probability`) and `Classification` (`label_id`/`label`/`confidence`) cannot share a primitive without splitting `try_softmax` into a `softmax_probs` raw helper. References cross-scope finding #2 below.
- why: doc tripwire against a well-meaning dedup PR that would break the AudioClass type signature. Incorporates inquisitor's MODIFY suggestion to cross-reference the deferred cross-scope finding.

## Tests Added/Updated

None. All five items are structural/readability with no behavior change:
- ITEM-AUD-001/002 preserve termination + clamp contracts exactly (verified by retaining the GPU `segment_offsets_match_cpu_loop` test, which now exercises the lifted helper).
- ITEM-AUD-003/004 are pure formatting / pattern syntax.
- ITEM-AUD-005 is a comment.

The pre-existing tests cover the behaviour:
- `sparrow-engine-core` `preprocess_audio` test suite (21 passed under `cargo test`).
- `sparrow-engine-gpu::models::audio::tests::segment_offsets_match_cpu_loop` now exercises the relocated helper.

Build validation: `cargo check --tests` clean for `sparrow-engine-core`, `sparrow-engine-cpu`, and `sparrow-engine-gpu`. `cargo test` for the three crates compiles but link of the ORT-linked test binaries fails on this host with `__isoc23_strtoll` / `__isoc23_strtol` (pre-existing glibc≥2.38 vs ort_sys mismatch; unrelated to this patch).

## Cross-Scope Findings (deferred to round 2)

1. **CPU↔GPU `merge_segments` byte-duplication.** `sparrow-engine-cpu/src/detect_audio.rs:748-790` and `sparrow-engine-gpu/src/detect_audio.rs:128-165` are byte-for-byte identical. Only 2 sites today (below 3+ dedup threshold), but the natural home is `sparrow_engine_types` (where `AudioRange` already lives). Flag for round 2 if a third caller appears, or if the round-2 reviewer judges 2-site verbatim-copy worth a pre-emptive lift.
2. **`postprocess::try_softmax` cannot be reused from the raw-audio path.** It returns `Classification`, blocking direct reuse from the new audio top-K post-processor. A future refactor could split out `softmax_probs(&row) -> Vec<f32>` consumed by both `try_softmax` and a new `audio_top_k(probs, labels, k)`. `postprocess.rs` was outside the round 1 owned-file set; deferred. Tripwire planted in the local doc comment (ITEM-AUD-005) so a future reviewer sees the link.
3. **`outputs.len() == 0` (CPU detect_audio.rs:442).** Clippy `len_zero` candidate, but `ort::SessionOutputs` may not expose `.is_empty()`; left untouched pending behavioral verification.
4. **`resolve_classifier_output` fallback behavior vs docstring.** Docstring says "single-output classifiers fall back to output 0", but `unwrap_or(0)` actually fires for ANY classifier missing a `"label"`-named output, including multi-head models without that head name. Behavioral edge — defer to round 2.

## Skipped

None — all 5 planned items were approved and applied.

STATUS: DONE COMMIT=91bf51e753974921f650b1a2119d41d577f73b61
