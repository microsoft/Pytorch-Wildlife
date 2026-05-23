# Reviewer Report — Round 3

## Approved items applied

<a name="ITEM-REV-R3-001"></a>
### ITEM-REV-R3-001 — stored audio drift labels use native K=1 class labels

**File**: `sparrow-engine/sparrow-engine-server/src/handlers/audio.rs`

Changed `audio_detect` so store/drift confidences and labels are collected from native `AudioSegment` values before conversion to `AudioDetectResponse`. This preserves labeled K=1 audio segments for `InferenceLogRecord` drift metrics while leaving the public JSON response conversion unchanged.

Added regression coverage for a single labeled `AudioClass` so the drift helper returns the label instead of falling back to `model_id`.

<a name="ITEM-REV-R3-002"></a>
### ITEM-REV-R3-002 — RawAudio+Softmax validates the named logits head

**File**: `sparrow-engine/sparrow-engine-cpu/src/engine.rs`

Changed CPU load-time output validation to select output 0 for single-output models, but require a `label` output for multi-output `RawAudio + Softmax` models. The selected output is then shape-validated by the existing softmax rank checks. Multi-output RawAudio+Softmax models without `label` now fail load-time validation with an `OutputShapeMismatch` instead of silently validating an unrelated first output.

Added pure helper tests for single-output fallback, Perch-style `label` selection, missing-label rejection, and non-RawAudio softmax preserving output-0 behavior.

## Verification

- `source scripts/ort-env.sh > /dev/null 2>&1 && cargo test -p sparrow-engine-server drift_label_ && cargo test -p sparrow-engine-cpu select_validation_output_index` — PASS
- `source scripts/ort-env.sh > /dev/null 2>&1 && cargo test -p sparrow-engine-cpu --test integration_perch2 -- --ignored --test-threads=1` — PASS (`perch2_detects_two_5s_windows_with_top5_classes_on_10s_clip`)
- `git diff --check -- sparrow-engine/sparrow-engine-cpu/src/engine.rs sparrow-engine/sparrow-engine-server/src/handlers/audio.rs docs/review/audit-fix/COVERAGE_LOG.jsonl` — PASS

## Cross-Scope Findings

- `sparrow-engine/sparrow-engine-gpu/src/engine.rs` is outside my owned files. Round 4 should evaluate whether the RawAudio+Softmax validation change needs a GPU flavor mirror.

STATUS: DONE COMMIT=56de9b920a1616d4e8cfd2007c10d0b6093072e1
