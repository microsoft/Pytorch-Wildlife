# FFI array pointer/len review notes — round 3

## Scope Checked

Owned files inspected for FFI array pointer/len behavior and relevant tests:

- `sparrow-engine/sparrow-engine-cpu/src/ffi.rs`
  - Output conversion structs/functions: `detect_result_to_c`, `classify_result_to_c`, `pipeline_result_to_c`, `audio_result_to_c`, `audio_result_v2_to_c`.
  - FFI exports using those conversions: detect/classify/pipeline/batch/audio free paths.
  - In-file unit tests for null-on-empty output arrays.
- `sparrow-engine/sparrow-engine-gpu/src/ffi.rs`
  - Same conversion and free-function paths as CPU.
  - In-file unit tests for null-on-empty output arrays.
- `sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs`
  - Symbol reachability/export checks only.
- `sparrow-engine/sparrow-engine-cpu/tests/integration_reexports.rs`
  - Crate-root re-export reachability checks only.

Static review only; no cargo test/build was run because this sub-agent is restricted to writing notes only under this round directory.

## Findings

None.

Reviewed behavior:

- CPU `detect_result_to_c` sets `SparrowEngineDetections.data = null` when `len == 0`; non-empty results use the stable owner vector pointer.
- GPU `detect_result_to_c` matches CPU behavior.
- CPU `classify_result_to_c` sets `top_results = null` when `top_results_len == 0`; non-empty results use the stable owner vector pointer.
- GPU `classify_result_to_c` matches CPU behavior.
- CPU `pipeline_result_to_c` sets `SparrowEnginePipelineResult.data = null` when `len == 0`; non-empty results use the stable owner vector pointer.
- GPU `pipeline_result_to_c` matches CPU behavior.
- Existing audio null-on-empty behavior remains consistent: v1/v2 result `data` pointers are null for empty segment arrays, and v2 per-segment `classes` is null when `classes_len == 0`.
- Free functions recover the wrapper allocation by casting the returned first-field header pointer back to the matching `*WithOwner` wrapper; wrappers are `#[repr(C)]`, so the header-first address invariant holds.
- The CPU/GPU in-file unit tests cover empty detection/classification/pipeline output arrays. The integration symbol/re-export tests are not behavioral FFI conversion tests, but that is acceptable because the conversion helpers are private and already have colocated unit coverage.

## Cross-Scope Findings

None.

## Suggested Reviewer Plan Items

None.

STATUS: OK
