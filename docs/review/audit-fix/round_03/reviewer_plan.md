# Reviewer Plan — Round 3

## Inputs read

- `~/.copilot/skills/_shared/iterative-anti-drift.md`
- `docs/review/audit-fix/SCOPE_LEDGER.json`
- `docs/review/audit-fix/COVERAGE_LOG.jsonl`
- `docs/review/audit-fix/round_02/inquisitor_review.md`
- Round-3 read-only sub-agent notes:
  - `round_03/subagent_model_audio_notes.md` — `STATUS: FINDINGS`
  - `round_03/subagent_ffi_notes.md` — `STATUS: OK`
  - `round_03/subagent_rest_owned_notes.md` — `STATUS: OK`

## Planned behavioral fixes

<a name="ITEM-REV-R3-001"></a>
### ITEM-REV-R3-001 | `sparrow-engine/sparrow-engine-server/src/handlers/audio.rs:101` | Compute stored audio drift labels from native audio segments before response conversion

**Proposed fix**: Change the drift-label helper and store path to read top-1 labels from the native `AudioSegment` values in `AudioDetectResult` before converting them into `AudioSegmentResponse`. Keep the public JSON conversion unchanged, including omission of `classes` for 0/1 class segments in `response.rs`. Add/update in-file tests in `handlers/audio.rs` covering single labeled `AudioClass` so store/drift labels use that label even though the JSON response can omit `classes`.

**Rationale**: `AudioSegmentResponse::from` intentionally drops `classes` when `s.classes.len() <= 1` (`response.rs:161-176`). The current store path computes labels from `response.segments` (`handlers/audio.rs:113-119`), so labeled K=1 audio segments are stored under `model_id` instead of their native top-1 label. `types.rs:156-179` documents K=1 as valid for binary detectors; `detect_audio.rs:485-493` emits exactly that shape when a detector label exists.

<a name="ITEM-REV-R3-002"></a>
### ITEM-REV-R3-002 | `sparrow-engine/sparrow-engine-cpu/src/engine.rs:901` | Validate RawAudio+Softmax logits output by name at load time

**Proposed fix**: Make CPU output-shape validation preprocess-aware for `RawAudio + Softmax`: for single-output classifiers, validate output 0 as today; for multi-output RawAudio+Softmax models, require an output named `label` and validate that output's shape instead of always validating `outputs[0]`. Add helper-level tests around the selection/validation logic if this can be factored without constructing an ORT `Session` in tests.

**Rationale**: `validate_output_shape` always checks `outputs[0]` (`engine.rs:913-916`). Perch 2-style RawAudio+Softmax models are multi-head and the runtime path explicitly resolves the logits head named `label` (`detect_audio.rs:256-320`; `integration_perch2.rs:3-12`). Current load-time validation can accept an unrelated first output and only fail later during inference, or silently use output 0 for a malformed no-label multi-output classifier.

## Cross-Scope Findings

- `sparrow-engine/sparrow-engine-gpu/src/engine.rs` is not in my owned-file list. If ITEM-REV-R3-002 is approved for CPU, the next round should assign the GPU engine equivalent so RawAudio+Softmax validation remains flavor-consistent.

STATUS: PLAN-READY
