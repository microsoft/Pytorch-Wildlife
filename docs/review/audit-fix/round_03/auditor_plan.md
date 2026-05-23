# Auditor plan — round 3

Scope: STRUCTURAL only (simplification, dead code, dedup with 3+ rule, naming, readability).
Owned files (7):
- sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs
- sparrow-engine/sparrow-engine-gpu/src/detect_audio.rs
- sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs
- sparrow-engine/sparrow-engine-core/src/viz.rs
- sparrow-engine/sparrow-engine-core/tests/audio_heatmap_e2e.rs
- sparrow-engine/sparrow-engine-gpu/src/models/audio.rs
- sparrow-engine/sparrow-engine-gpu/src/models/classifier.rs

## Re-read of round-2 carry-forward items vs structural scope

Round 2 inquisitor (`round_02/inquisitor_review.md`, "Missed issues" + "Carry-forward")
enumerated five deferred items. I re-classify them against my structural scope:

| Carry-forward item | Location | Disposition |
|---|---|---|
| CPU↔GPU `merge_segments` / `merge_segments_with_class` verbatim copy | cpu/detect_audio.rs L?? + gpu/detect_audio.rs L128-165 | **Below 3+ dedup rule** (exactly 2 call sites). Skill explicitly bars dedup under 3 occurrences. No action. |
| `postprocess::try_softmax` reuse blocker (tripwire doc) | cpu/detect_audio.rs L675-685 | Behavioral/API split (`AudioClass` vs `Classification` struct shapes). Tripwire doc already planted; cannot be unlocked without editing `sparrow-engine-core/src/postprocess.rs` which is OUTSIDE my owned set. → **Cross-Scope Findings**. |
| `resolve_classifier_output` doc-vs-code drift @ cpu/detect_audio.rs:267 | cpu/detect_audio.rs L260-281 | Doc says "pick label head by name; fall back to output 0" — code does exactly that. Re-reading L260-281 I see no actual drift: docstring matches `.position(|o| o.name() == "label").unwrap_or(0)`. The "drift" flagged in round 2 was behavioral (probe-class-count semantics vs. authoritative count) — not a structural defect. No structural action. |
| RawAudio `segment_duration_s` opt override ignored; top-K validates flat length not rank-2 shape | cpu/detect_audio.rs (raw-audio path) | **Behavioral** — outside structural scope. → **Cross-Scope Findings**. |
| `discover.rs` stale audio-classifier comments | sparrow-engine-server/src/discover.rs | NOT in my owned set. → **Cross-Scope Findings**. |

## Owned-file re-read

Spot-checked the seven owned files for new structural issues that surfaced
since round 2 (no source edits since `eaa1fd3` for any owned file):

- `viz.rs` — round-2 indent fix verified (`classes: Vec::new(),` aligned at the
  audio test struct literals). No further indent / dead-code / dup hits.
- `preprocess_audio.rs` — `compute_segment_offsets` (L311) and
  `segment_time_range` (L334) helpers (round-1 dedup) remain the only shared
  shape extractors. No new 3+ pattern.
- `audio_heatmap_e2e.rs` — three `classes: Vec::new()` literals at L69-71 are
  single-line; no indent defect.
- `cpu/detect_audio.rs` — `outputs.len() == 0` idiom retained
  (`ort::SessionOutputs` lacks `.is_empty()`); pin_session call sites have
  distinct downstream extraction; softmax/topk locals are deliberately
  scoped (tripwire doc L675-685 preserved).
- `gpu/detect_audio.rs` — file is the thin GPU mirror; `merge_segments`
  duplication is at 2 sites (CPU+GPU), below the 3+ rule.
- `gpu/models/audio.rs`, `gpu/models/classifier.rs` — re-scanned; no new
  structural defect (round-2 inquisitor already verified).

## Verdict

Zero in-scope structural items remain. Round-2 inquisitor explicitly
forecasted "a round-3 NOTHING-TO-DO pass is realistic" with "No new issues
raised by me beyond the items already approved and applied."

## Cross-Scope Findings (DEFERRED to round 4 reassignment, not fixed here)

1. `postprocess::try_softmax` does not expose a `softmax_probs(&[f32]) -> Vec<f32>`
   primitive that the raw-audio classifier path could share. Requires editing
   `sparrow-engine-core/src/postprocess.rs` (outside owned set).
2. RawAudio classifier path in `cpu/detect_audio.rs` ignores
   `AudioDetectOpts::segment_duration_s` override and validates top-K against
   flat length rather than rank-2 shape — behavioral, requires reviewer.
3. `sparrow-engine-server/src/discover.rs` comments still describe
   `(Mel, Softmax)` as `AudioClassifier` — stale after REV-R2-001 retightened
   `derive_model_type`. Comment-only; not in any current ledger entry.

STATUS: NOTHING-TO-DO
