# coder-ffi-cpu report — round 01

## ffi-v2-audio-cpu

Implemented CPU FFI V2 audio support for top-K classes.

### Changed files
- `sparrow-engine/sparrow-engine-cpu/src/ffi.rs`: added V2 audio class/segment/result repr(C) types, owner-backed conversion, `sparrow_engine_detect_audio_v2`, `sparrow_engine_audio_result_v2_free`, and unit coverage for class labels/null labels/free.
- `sparrow-engine/sparrow-engine-cpu/exports.def`: added the two V2 exports.
- `sparrow-engine/sparrow-engine-cpu/sparrow_engine.h`: regenerated cbindgen header.
- `sparrow-engine/include/sparrow_engine.h`: mirrored the regenerated header (forced add because include/ ignores generated files).
- `sparrow-engine/sparrow-engine-cpu/tests/integration_ffi_symbols.rs`: bumped symbol invariant 32→34 and fixed the smoke test to resolve symbols through `sparrow_engine::ffi`.

### Verification
- `cargo build -p sparrow-engine-cpu --features ffi` — PASS. The prompt's `--features cpu --features ffi` form fails because `sparrow-engine-cpu` has no `cpu` feature.
- `cargo build --release -p sparrow-engine-cpu --features ffi` — PASS.
- `cargo test -p sparrow-engine-cpu --features ffi --test integration_ffi_symbols` — PASS (2 passed).
- `cargo test -p sparrow-engine-cpu --features ffi --lib` — PASS (70 passed).
- `nm -D --defined-only target/release/libsparrow_engine.so | grep '^sparrow_engine_' | wc -l` — 34; includes `sparrow_engine_detect_audio_v2` and `sparrow_engine_audio_result_v2_free`.
- `diff -q sparrow-engine-cpu/sparrow_engine.h include/sparrow_engine.h` — PASS, headers identical.
- code-auditor pass — PASS.

### Commit
- `d54d5427c42e0758796b5f750bcb5502f9637580`

STATUS: DONE COMMIT=d54d5427c42e0758796b5f750bcb5502f9637580 SIGNATURE="sparrow_engine_detect_audio_v2" ITEM=ffi-v2-audio-cpu
