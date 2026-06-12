# RP-38 TFLite half — vendored Google AI Edge LiteRT C SDK

Pinned to **v2.1.5** (sha `9d26e89`) — matches the `libLiteRt.so` aarch64
binary staged in `bench-binaries/artifacts/`.

## Why vendor

`bench-binaries/spe-bench-tflite` (Rust) links dynamically against
`libLiteRt.so` (the 5.5 MB aarch64 sidecar in `bench-binaries/artifacts/`).
The Rust binding uses `bindgen` against these headers at build time. Pinning
both header tree and `.so` to the same upstream release keeps ABI stable.

## What's in here

- `litert/c/`          — public C API (model load, env, compiled model, tensor buffers)
- `litert/c/internal/` — internal C API (scheduling info, accelerator def, etc. — referenced transitively by the public headers)
- `litert/c/options/`  — per-accelerator option types (CPU + GPU + NPU vendor-specific)
- `litert/build_common/build_config.h`         — preprocessor build config (we use the cpu-only variant; LITERT_DISABLE_GPU + LITERT_DISABLE_NPU)
- `litert/build_common/config/build_config_*.h` — alternative variants kept for diff reference

## Why we removed everything else

The upstream `litert/c/` tree at v2.1.5 contains 60 headers + Bazel `BUILD`
files + `.bzl` rules + CMakeLists + symlink_files manifests. We only need
the headers. Removing the Bazel scaffolding keeps the vendor footprint to
just the API surface.

## Refresh procedure

When the staged `libLiteRt.so` is bumped to a newer LiteRT release:

1. Identify the new tag: e.g. `v2.2.0`. Get the matching x86_64 + aarch64 wheels:
   ```
   python3 -m pip download --no-deps --only-binary=:all: \
     --platform manylinux_2_27_aarch64 --python-version 3.11 ai-edge-litert==<version>
   ```
2. Extract `ai_edge_litert/libLiteRt.so` to `bench-binaries/artifacts/libLiteRt.so` (aarch64).
   Refresh `bench-binaries/artifacts/SHA256SUMS`.
3. Fetch the matching repo tarball:
   ```
   curl -sL https://api.github.com/repos/google-ai-edge/litert/tarball/<tag> -o /tmp/litert.tar.gz
   tar xzf /tmp/litert.tar.gz -C /tmp google-ai-edge-LiteRT-<sha>/litert
   ```
4. Replace the headers under `bench-binaries/vendor/litert/litert/c/` and
   `bench-binaries/vendor/litert/litert/build_common/` from the tarball.
5. Re-run `cargo build -p spe-bench-tflite` — if bindgen breaks, the API surface
   changed; reconcile our safe wrappers.

## v2.1.5 quirks worked around

None at the moment. Earlier diagnosis suggested `litert/c/litert_model.h` referenced
`LiteRtQuantizationBlockWise` (missing from v2.1.5 `litert_model_types.h`), but
re-checking the v2.1.5 tarball directly shows the original v2.1.5 `litert_model.h`
doesn't reference that function — the issue was a header-mix problem during
exploration. No patches applied to the vendored tree.

## License

Apache-2.0 (per the upstream headers' license headers). Copyright Google LLC.
