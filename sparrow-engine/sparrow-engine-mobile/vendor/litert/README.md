# sparrow-engine-mobile — vendored Google AI Edge LiteRT C SDK

Pinned to **v2.1.5** (sha `9d26e89`) — matches the `libLiteRt.so`
sidecars used by `sparrow-engine-mobile` for host and aarch64 builds.

## Why vendor

`sparrow-engine-mobile` links dynamically against `libLiteRt.so` while using
`bindgen` against these headers at build time. Pinning both the header tree and
runtime sidecar to the same upstream release keeps the LiteRT C application
binary interface stable for the focused mobile orca cascade.

The crate's `build.rs` searches for `libLiteRt.so` in this order:

1. `LITERT_LIB_DIR` when set by the caller.
2. The PW workspace-local `sparrow-engine/artifacts/` directory, which is also
   visible inside the `cross` container used for aarch64 builds.

## What's in here

- `litert/c/` — public C API: model load, environment, compiled model, tensor buffers.
- `litert/c/internal/` — internal C API referenced transitively by public headers.
- `litert/c/options/` — per-accelerator option types, including CPU options.
- `litert/build_common/build_config.h` — CPU-only preprocessor build config.
- `litert/build_common/config/build_config_*.h` — alternative variants kept for diff reference.

## Why we removed everything else

The upstream `litert/c/` tree at v2.1.5 contains headers plus Bazel, `.bzl`,
CMake, and symlink manifest files. The Rust crate only needs the headers.
Removing build-system scaffolding keeps the vendor footprint to the API surface
that bindgen reads.

## Refresh procedure

When the staged `libLiteRt.so` is bumped to a newer LiteRT release:

1. Identify the new tag, for example `v2.2.0`. Download matching x86_64 and
   aarch64 wheels with the project Python environment.
2. Extract the aarch64 `ai_edge_litert/libLiteRt.so` to
   `sparrow-engine/artifacts/libLiteRt.so` and refresh
   `sparrow-engine/artifacts/SHA256SUMS` if present.
3. Fetch the matching LiteRT repository tarball into a project-local scratch
   directory such as `sparrow-engine/artifacts/litert-refresh/`.
4. Replace headers under `sparrow-engine-mobile/vendor/litert/litert/c/` and
   `sparrow-engine-mobile/vendor/litert/litert/build_common/` from the tarball.
5. Re-run `cargo build -p sparrow-engine-mobile --features ffi`. If bindgen
   breaks, the C API changed and the safe wrappers need reconciliation.

## v2.1.5 quirks worked around

Earlier diagnosis suggested `litert/c/litert_model.h` referenced
`LiteRtQuantizationBlockWise` missing from v2.1.5 `litert_model_types.h`, but
re-checking the v2.1.5 tarball directly showed the original v2.1.5 header does
not reference that function. The issue was a header-mix problem during
exploration. No patches are applied to the vendored tree.

## License

Apache-2.0, per the upstream header license blocks. Copyright Google LLC.
