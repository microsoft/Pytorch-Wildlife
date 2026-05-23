# Round 3 reviewer sub-agent notes — remaining owned files

## Scope Checked

Focused read-only review for behavioral bugs, edge cases, error handling, correctness, and validation only. No source files edited.

Files checked:
- `sparrow-engine/sparrow-engine-cli/src/main.rs`
  - Command dispatch/global helpers: lines 485-760
  - Visualization/audio-viz helpers: lines 760-1190
  - `detect`/`classify` commands and writers: lines 1193-1518
  - `detect-audio` command and audio writers: lines 1520-1841
  - `pipeline`/`models`/`device`/`init`/utility commands: lines 1847-2275
  - In-file behavioral regression tests: spot-checked relevant sections under lines 2281-3182
- `sparrow-engine/sparrow-engine-cpu/src/classify.rs`
- `sparrow-engine/sparrow-engine-cpu/src/detect.rs`
- `sparrow-engine/sparrow-engine-cpu/src/preprocess.rs`
- `sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs`
- `sparrow-engine/sparrow-engine-gpu/src/classify.rs`
- `sparrow-engine/sparrow-engine-gpu/src/detect.rs`
- `sparrow-engine/sparrow-engine-types/src/types.rs`

Context checked to avoid duplicate findings:
- `docs/review/audit-fix/round_02/scratch/cli_surface_review.md`
- `docs/review/audit-fix/round_02/inquisitor_review.md`
- Direct import/validation context for manifest input/tile validation and output-shape validation.

## Findings

None.

The focused files still match the round-2 expectation:
- CLI audio visualization threshold handling preserves the round-1 policy (`main.rs:1513-1569`, `1608-1628`).
- CLI file collection follows symlinks via `std::fs::metadata` and surfaces unreadable entries as warnings (`main.rs:663-739`).
- CPU/GPU image classify/detect entry points reject wrong model families before inference and validate confidence thresholds before dispatch.
- CPU preprocess rejects audio preprocessing methods, guards zero-sized decoded images, and manifest parsing validates non-zero `input_size` before these paths are reachable.
- Perch 2 integration test remains ignored/gated on bundle presence and checks top-5 class invariants without introducing production behavior.
- `types.rs` is POD-only; no behavioral defect found in the reviewed public structs/enums.

## Cross-Scope Findings

None from this focused review.

## Suggested Reviewer Plan Items

None. No fresh reviewer-owned behavioral item is suggested for these remaining files.

STATUS: OK
