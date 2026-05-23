# Auditor plan — round 4

Scope: STRUCTURAL only (simplification, dead code, dedup 3+, naming, readability).
Owned files (7):
- sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs
- sparrow-engine/sparrow-engine-gpu/src/detect_audio.rs
- sparrow-engine/sparrow-engine-core/src/preprocess_audio.rs
- sparrow-engine/sparrow-engine-core/src/viz.rs
- sparrow-engine/sparrow-engine-core/tests/audio_heatmap_e2e.rs
- sparrow-engine/sparrow-engine-gpu/src/models/audio.rs
- sparrow-engine/sparrow-engine-gpu/src/models/classifier.rs

## Re-audit

No owned-file source change since `eaa1fd3` (R2 viz re-indent). R3 commit
`56de9b9` (drift label + output validation) modified only reviewer-owned
files (`engine.rs`, `handlers/audio.rs`); R3 review confirmed STATUS:
NOTHING-TO-DO for the auditor side. I re-read each owned file and the
R3 inquisitor review:

- `viz.rs` — R2 indent fix intact; no further structural hits.
- `preprocess_audio.rs` — `compute_segment_offsets` / `segment_time_range`
  remain the only shared helpers; no new 3+ pattern.
- `audio_heatmap_e2e.rs` — three `classes: Vec::new()` literals are
  single-line and aligned; no defect.
- `cpu/detect_audio.rs` — `outputs.len() == 0` retained (`ort::SessionOutputs`
  lacks `.is_empty()`); softmax/topk locals deliberately scoped; tripwire doc
  preserved.
- `gpu/detect_audio.rs` — thin GPU mirror; `merge_segments` duplication is
  2 sites (CPU+GPU), below the 3+ rule.
- `gpu/models/audio.rs`, `gpu/models/classifier.rs` — re-scanned, no
  structural defect introduced.

## Carry-forward from R3 Cross-Scope (still deferred — not in my scope)

1. `postprocess::try_softmax` shared-primitive opportunity — requires editing
   `sparrow-engine-core/src/postprocess.rs` (NOT in ledger).
2. RawAudio `segment_duration_s` opt override ignored + top-K flat-length
   validation in cpu/detect_audio.rs raw-audio path — behavioral (reviewer
   scope); R3 inquisitor confirmed still untouched.
3. `sparrow-engine-server/src/discover.rs` stale audio-classifier comments —
   not in ledger.
4. GPU `engine.rs` mirror of R3 CPU `select_validation_output_index` —
   `sparrow-engine-gpu/src/engine.rs` not in ledger; R3 inquisitor flagged
   as round-4 ledger-expansion question, not a missed issue. Cross-scope.

No round-1..round-3 ledger entry has been removed or ignored.

## Verdict

Zero in-scope structural items remain. R4 is the convergence-confirmation
pass; no new structural concerns surfaced.

STATUS: NOTHING-TO-DO
