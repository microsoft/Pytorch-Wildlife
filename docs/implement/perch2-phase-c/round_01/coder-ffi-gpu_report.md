# coder-ffi-gpu report — round 01

<a name="ffi-v2-audio-gpu"></a>
## ffi-v2-audio-gpu

## Changes
- Mirrored CPU V2 audio FFI layouts into `sparrow-engine-gpu/src/ffi.rs`:
  - `SparrowEngineAudioClass`
  - `SparrowEngineAudioSegment_v2`
  - `SparrowEngineAudioResult_v2`
- Added `audio_result_v2_to_c` with label/class arenas so returned pointers stay valid until free.
- Added `sparrow_engine_detect_audio_v2`, routed through `crate::detect_audio::detect_audio`; GPU RawAudio models therefore return the existing clean GPU raw_audio error.
- Added `sparrow_engine_audio_result_v2_free`.
- Updated `exports.def` to 34 exports and regenerated `sparrow_engine.h` via the GPU build.
- Added unit test `audio_result_v2_to_c_preserves_top_k_classes`.

## Coordination
- CPU branch did not have a committed V2 layout yet; CPU worktree had the V2 changes. Mirrored those layouts/conversion semantics directly from `/home/miao/repos/PW_refactor/Pytorch-Wildlife/.copilot/worktrees/coder-ffi-cpu/sparrow-engine/sparrow-engine-cpu/src/ffi.rs` and confirmed header V2 declarations match.

## Verification
- `cargo test -p sparrow-engine-gpu --features ffi audio_result_v2_to_c_preserves_top_k_classes --lib` — PASS.
- `cargo build -p sparrow-engine-gpu --features ffi --release` — PASS.
- `nm -D --defined-only target/release/libsparrow_engine.so | grep -E 'sparrow_engine_detect_audio_v2|sparrow_engine_audio_result_v2_free'` — PASS:
  - `T sparrow_engine_audio_result_v2_free`
  - `T sparrow_engine_detect_audio_v2`
- `git diff --check -- sparrow-engine/sparrow-engine-gpu/src/ffi.rs sparrow-engine/sparrow-engine-gpu/exports.def sparrow-engine/sparrow-engine-gpu/sparrow_engine.h` — PASS.
- code-auditor review — PASS.

## Commit
- `9b73e03`

STATUS: DONE COMMIT=9b73e03 SIGNATURE="sparrow_engine_detect_audio_v2" ITEM=ffi-v2-audio-gpu
