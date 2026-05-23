# Reviewer scratch — CLI/server audio classes

## Findings

### F1 — `spe detect-audio --visualize` filters Perch 2 segments differently from non-visualized output
- **File:line**: `sparrow-engine/sparrow-engine-cli/src/main.rs:1558-1565`, `sparrow-engine/sparrow-engine-cli/src/main.rs:1607-1614`; supporting core evidence: `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:525-527`, `sparrow-engine/sparrow-engine-cpu/src/detect_audio.rs:640-662`.
- **Observed code evidence**: `cmd_detect_audio` lowers inference threshold to `0.0` for visualization, then post-filters `output_result.segments` by `output_threshold`. For a model without a manifest audio threshold, `output_threshold` falls back to `0.5`. The raw-audio/Perch 2 path documents and implements unconditional window emission, with `confidence` set to top-1 probability and no threshold gate.
- **User-visible defect**: the same Perch 2 command can print different JSON/CSV segments solely because `--visualize` is present. The Phase C smoke output records a second Perch 2 window at `confidence = 0.24102566` (`docs/implement/perch2-phase-c/round_01/coder-cli_report.md:76-104`), which non-visualized raw JSON prints but visualized output would drop by the default `0.5` post-filter.
- **Proposed fix**: centralize the policy. If Perch 2 classifiers should honor `--threshold`, apply the threshold in `detect_audio_loop_raw` so CLI and server behave the same. If classifiers should emit every window, skip the CLI visualization post-filter for thresholdless softmax/raw-audio models instead of using the `0.5` fallback.
- **Rationale**: visualization should not change the machine-readable inference output except for writing PNG side effects. This also keeps CLI/server boundary behavior consistent.
- **Confidence**: HIGH (static trace across CLI threshold rewrite and raw-audio emission; no runtime command run).

## Test Gaps

- `sparrow-engine/sparrow-engine-cli/src/main.rs:1513-1705`: no unit test covers `detect-audio --print --raw-segments --format json --visualize` for a Perch 2-like `AudioDetectResult` with one top-1 probability below `0.5`. Add a regression test around the filtering policy chosen for F1.
- `sparrow-engine/sparrow-engine-server/src/response.rs:161-184`: response tests cover empty, single-class, and multi-class `Some(label)` serialization. Add a multi-class `label: None` case to pin `class_idx`/`probability` serialization when labels are unavailable.
- `sparrow-engine/sparrow-engine-cli/src/main.rs:1762-1775`, `sparrow-engine/sparrow-engine-cli/src/main.rs:3234-3259`: CSV raw-segment output intentionally preserves the old six-column schema and does not expose top-K classes. If top-K display is required for CSV, add a new additive CSV mode/columns plus tests; otherwise document JSON-only top-K display.

## Cross-Scope Findings

### XS1 — `store=true` audio drift labels ignore Perch 2 top-1 classes
- **File:line**: `sparrow-engine/sparrow-engine-server/src/handlers/audio.rs:103-116`; supporting spec evidence: `sparrow-engine/sparrow-engine-server/src/drift.rs:15-24`.
- **Observed code evidence**: the audio handler builds `labels` as `vec![model_id.clone(); response.segments.len()]` for every stored audio request. `compute_drift_metrics` expects observed class labels per output/segment for PSI.
- **User-visible defect**: Perch 2 responses now carry top-K classes, but stored drift metrics bucket all segments as `perch-v2`, so class-distribution PSI cannot reflect species/class changes.
- **Proposed fix**: derive audio drift labels from each segment's top-1 class when present (`label`, with `class_idx` string fallback), and keep `model_id` fallback for binary/no-class segments.
- **Rationale**: keeps `confidence` and class labels compatible with the new multi-class audio response without changing schema v1.0.
- **Confidence**: HIGH (direct handler and drift API evidence; no runtime command run).

STATUS: DONE
