# Reviewer scratch: CPU FFI audio V2 / Perch 2 tests

## Findings

### F-CPU-FFI-1 — Empty V2 audio results expose a non-null dangling `data` pointer
- **Location:** `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:630-640`
- **Evidence:** `audio_result_v2_to_c` always assigns `combined.header.data = combined._owner.segments.as_ptr()` even when `owned.segments.len() == 0`. For an empty `Vec`, `as_ptr()` is a non-owning dangling sentinel; callers must use `len` correctly to avoid dereferencing it.
- **Proposed fix:** Set `header.data = ptr::null()` when `owned.segments.is_empty()`, mirroring the existing per-segment zero-class convention at `ffi.rs:607-620` (`classes_len == 0` => `classes = null`). Consider applying the same normalization to V1 audio/detection/classification result builders if the public C contract wants null-on-empty consistently.
- **Rationale:** The V2 ABI already uses null to represent an empty nested class array. Returning null for empty top-level arrays makes zero handling explicit for C/C#/Python consumers and avoids exposing a dangling-but-len-0 pointer.
- **Confidence:** MEDIUM — code evidence is direct; impact depends on whether the intended ABI permits non-null pointers with zero length.

## Test Gaps

### TG-CPU-FFI-1 — Perch 2 integration bypasses the V2 FFI surface
- **Location:** `sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs:96-131`; `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:1337-1354`
- **Evidence:** The Perch 2 integration test calls `sparrow_engine::detect_audio::detect_audio` directly and asserts Rust `AudioSegment.classes.len() == 5`. It never calls `sparrow_engine_detect_audio_v2`, never walks `SparrowEngineAudioResult_v2.data`, and never frees through `sparrow_engine_audio_result_v2_free`.
- **Proposed fix:** Add an ignored `--features ffi` integration test that loads the Perch 2 manifest through FFI, calls `sparrow_engine_detect_audio_v2`, verifies two segments with `classes_len == 5`, validates C labels/probabilities/class indices, then frees with `sparrow_engine_audio_result_v2_free`.
- **Rationale:** The Rust API test proves the model path emits top-K classes, but not that the public C ABI preserves top-K layout, label pointers, ownership, or free behavior.
- **Confidence:** HIGH — the test body and FFI entry point were both inspected.

### TG-CPU-FFI-2 — CPU V2 conversion unit test does not cover zero-class segments or nul-byte label sanitization
- **Location:** `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:2003-2121`; `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:607-620`
- **Evidence:** The CPU unit test covers non-empty `classes` arrays with both `Some(label)` and `None` labels, but every segment has three classes. The implementation has a distinct zero-class branch that sets `classes = null`; that branch is not asserted in CPU tests. The CPU test also does not cover label strings containing `\0`, though conversion strips nul bytes at `ffi.rs:581`.
- **Proposed fix:** Add a CPU fixture segment with `classes: Vec::new()` and assert `classes_len == 0` plus `classes.is_null()`. Add a label containing an interior nul and assert the C string is sanitized. Optionally add an all-zero-segment `AudioDetectResult` case to codify the desired top-level `data` null/len contract.
- **Rationale:** Zero classes are documented as possible for legacy binary detectors without labels; the branch is security/ABI-sensitive because callers gate pointer dereference on `classes_len` and `classes`.
- **Confidence:** HIGH — branch and test coverage gap are directly visible.

### TG-CPU-FFI-3 — Symbol stability test permits extra exported `sparrow_engine_*` symbols and can skip the V2 export check
- **Location:** `sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs:27-48`, `:76-84`, `:130-147`; `sparrow-engine/sparrow-engine-cpu/exports.map:1-6`
- **Evidence:** The compile-time smoke test references only five legacy symbols, not `sparrow_engine_detect_audio_v2` or `sparrow_engine_audio_result_v2_free`. The `nm` test skips when `target/release/libsparrow_engine.so` is absent. When it does run, it checks only that every `exports.def` symbol is present; it does not fail on extra `sparrow_engine_*` exports even though `exports.map` globally exports any matching prefix.
- **Proposed fix:** Add the V2 audio functions to the compile-time smoke set. In the `nm` test, assert equality between actual exported `sparrow_engine_*` names and `exports.def` (or at least fail on extras) and assert the actual count is 34.
- **Rationale:** The public ABI needs both missing-symbol and accidental-extra-symbol protection; extra exports can become accidental compatibility commitments.
- **Confidence:** HIGH — the assertions and export map behavior are direct code evidence.

### TG-CPU-FFI-4 — Null/error-path coverage is missing for audio FFI entry points and frees
- **Location:** `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:1297-1367`, `:1436-1470`; `sparrow-engine/sparrow-engine-cpu/tests/integration_perch2.rs:58-247`
- **Evidence:** `sparrow_engine_detect_audio` and `_v2` map null model and null/invalid path failures through `last_error`; `sparrow_engine_audio_result_free` and `_v2_free` are null no-ops. The listed integration tests do not call these FFI functions with null pointers or verify `sparrow_engine_last_error`.
- **Proposed fix:** Add `--features ffi` tests for null model, null `audio_path`, null `opts` defaulting, and null frees for both V1 and V2 audio result frees.
- **Rationale:** These paths are the consumer-visible error contract for C ABI callers and are not exercised by the current Rust API Perch 2 test.
- **Confidence:** HIGH — entry-point branches and test absence are visible.

## Cross-Scope Findings

### CS-CPU-GPU-1 — GPU has zero-class V2 FFI coverage that CPU lacks
- **Location:** `sparrow-engine/sparrow-engine-gpu/src/ffi.rs:2012-2074`; CPU counterpart `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:2003-2121`
- **Evidence:** The GPU unit test includes a second segment with `classes: Vec::new()` and asserts `classes_len == 0` plus `classes.is_null()`. The CPU unit test only covers non-empty class arrays.
- **Proposed fix:** Mirror the GPU zero-class assertion in the CPU unit test to keep FFI behavior parity across flavors.
- **Rationale:** CPU/GPU ship the same C ABI name and should test the same empty nested-array contract.
- **Confidence:** HIGH — both unit tests were inspected.

### CS-CPU-FFI-2 — V1 streaming audio callback is not nullable-safe
- **Location:** `sparrow-engine/sparrow-engine-cpu/src/ffi.rs:1370-1414`
- **Evidence:** `SparrowEngineAudioSegmentCallback` is a bare `unsafe extern "C" fn`, and `sparrow_engine_detect_audio_streaming` invokes `callback(...)` without an `Option`/null check. A C caller can still pass a null function pointer, which is invalid for Rust's non-null function-pointer type.
- **Proposed fix:** Change the FFI argument to `Option<SparrowEngineAudioSegmentCallback>` and return `-1`/null with `last_error` when the callback is null. This is representation-compatible for nullable C function pointers.
- **Rationale:** This is outside the V2 top-K path but inside the audio FFI null-handling surface; it prevents undefined behavior from a common C misuse case.
- **Confidence:** MEDIUM — the code evidence is direct; impact depends on whether callers ever pass null despite the documented non-null callback requirement.

STATUS: DONE
