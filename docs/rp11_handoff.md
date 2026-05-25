# RP-11 handoff — GPU wheel publish CI after Phase E

Phase E removes the build-time `libnvjpeg.so` link from `sparrow-engine-gpu`.
RP-11 owns the GitHub Actions wiring that turns the repaired local GPU wheel flow into publish CI.

## Scope

Applies to the Linux `sparrow-engine-gpu` wheel. CPU, macOS, and Windows jobs keep their existing RP-11 flow. Phase E ships local scripts; RP-11 wires them into `.github/workflows/release.yml`. Target wheel tag: `manylinux_2_28_x86_64`.

## Required pins

| Item | Pin | Notes |
|---|---|---|
| `auditwheel` | `auditwheel >= 6.0.0` | Phase E verified the `auditwheel show <wheel>` plus `auditwheel repair --plat --exclude` command shape against this interface. |
| manylinux policy | `manylinux_2_28_x86_64` | Must match the repaired wheel tag and the publish artifact assertion. |
| manylinux image | `quay.io/pypa/manylinux_2_28_x86_64` | Use this directly, through `CIBW_MANYLINUX_X86_64_IMAGE`, or through `PyO3/maturin-action@v1` with `manylinux: 2_28`. |
| current GPU image to replace | `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` | Verified in `.github/workflows/release.yml`; it is an Ubuntu 24.04 CUDA build container, not a manylinux publish base. |
| runtime nvjpeg source | system CUDA 12 or `nvidia-nvjpeg-cu12` | Do not bundle `libnvjpeg.so.12` into the `sparrow-engine-gpu` wheel. |

If RP-11 pins images by digest, resolve the current digest for `quay.io/pypa/manylinux_2_28_x86_64` when editing the workflow and pin that digest in YAML. Do not leave the GPU publish job on the Ubuntu 24.04 CUDA image.

## Environment variable contract

`SPARROW_ENGINE_NVJPEG_LIBRARY_PATH` is the single override env-var for the nvjpeg loader.

| Value | Meaning in CI |
|---|---|
| unset or empty | Normal loader search: sidecar preload, SONAME lookup, then known CUDA paths. |
| absolute path | `dlopen` exactly that path; all other search paths are disabled. |
| bad path | Expected to trigger `NvjpegInitError::LibraryNotFound` in negative tests. |

CI must preserve this variable if a caller sets it for a non-standard test scenario. Do not overwrite it with an empty value in publish jobs.

## Commands to wire into GPU publish validation

Install the auditwheel version used by the Phase E gate:

```bash
python3 -m pip install 'auditwheel>=6.0.0'
```

Run `auditwheel show` on the raw GPU wheel:

```bash
for WHEEL in sparrow-engine/target/wheels/sparrow_engine_gpu-*.whl; do
  python3 -m auditwheel show "$WHEEL"
done
```

Repair to the publishable manylinux tag while keeping ORT external:

```bash
for WHEEL in sparrow-engine/target/wheels/sparrow_engine_gpu-*.whl; do
  python3 -m auditwheel repair \
      --plat manylinux_2_28_x86_64 \
      --exclude libonnxruntime.so.1 \
      --wheel-dir sparrow-engine/target/wheels-repaired/ \
      "$WHEEL"
done
```

Validate the repaired artifact before upload:

```bash
for REPAIRED in sparrow-engine/target/wheels-repaired/sparrow_engine_gpu-*.whl; do
  python3 -m auditwheel show "$REPAIRED"
  case "$REPAIRED" in
    *manylinux_2_28_x86_64*) ;;
    *) echo "FAIL: repaired GPU wheel is not manylinux_2_28_x86_64"; exit 1 ;;
  esac
done
```

Publish jobs should upload only `sparrow-engine/target/wheels-repaired/sparrow_engine_gpu-*.whl` for the GPU package. The raw `linux_x86_64` wheel is a build intermediate.

## Undefined-symbol gate

The `nm -u` gate must run in CI in addition to auditwheel. It catches a future regression where `libnvjpeg.so.12` is no longer in `DT_NEEDED` but unresolved `nvjpeg*` symbols remain in the cdylib.

```bash
nm -u sparrow-engine/target-gpu/release/libsparrow_engine.so \
  | grep -E '^[[:space:]]*U[[:space:]]+nvjpeg' && {
      echo 'FAIL: undefined nvjpeg symbol(s) in cdylib (Phase E regression)' >&2
      exit 1
    } || true
```

Expected output: no matching `nvjpeg*` lines. The command should exit 0 without printing a failure message.

## Local script hook

Once coder-3's script lands, prefer calling it from CI rather than duplicating shell fragments:

```bash
bash sparrow-engine/scripts/audit_wheel_gate.sh
```

It owns T6 (`auditwheel show` + repair) and T9 (`nm -u` purity). Keep version pins and artifact paths visible in workflow YAML.

## Where to place the gates

1. `build-gpu-linux`: build inside the manylinux base, run repair, run `audit_wheel_gate.sh`, then upload the repaired wheel artifact.
2. `publish-testpypi-gpu`: download the repaired artifact, show it, validate the filename contains `manylinux_2_28_x86_64`, then publish to TestPyPI.
3. `publish-pypi-gpu`: repeat the same pre-upload validation and tag-version check used by CPU, then publish to PyPI.

The existing gated GPU publish jobs are the right insertion points. Do not enable `publish-testpypi-gpu` or `publish-pypi-gpu` until the repaired wheel is the uploaded artifact.

## Optional cibuildwheel sketch

The project currently builds wheels with maturin. If RP-11 moves Linux GPU builds to cibuildwheel, use the same pins and preserve `SPARROW_ENGINE_NVJPEG_LIBRARY_PATH` pass-through behavior:

```yaml
env:
  CIBW_BUILD: cp311-manylinux_x86_64
  CIBW_ARCHS_LINUX: x86_64
  CIBW_MANYLINUX_X86_64_IMAGE: quay.io/pypa/manylinux_2_28_x86_64
  CIBW_BEFORE_BUILD: python -m pip install 'auditwheel>=6.0.0' maturin
  CIBW_TEST_REQUIRES: nvidia-nvjpeg-cu12 nvidia-cuda-runtime-cu12
  CIBW_TEST_COMMAND: python -c "import sparrow_engine"
```
