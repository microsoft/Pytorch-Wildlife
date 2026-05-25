#!/usr/bin/env bash
# Phase E nvjpeg dlopen regression matrix.
#
# Runs T1-T5, T7, T10, and T11 from docs/design/phase-e-nvjpeg-dlopen/final.md.
# Every cell executes the native sparrow_engine import/engine path in a fresh
# Python subprocess so a failed OnceLock initialization in one cell cannot poison
# another cell. T2/T3 build tiny mock libnvjpeg.so.12 libraries at runtime.
#
# Operator inputs:
# - SPARROW_ENGINE_PYTHON: Python executable to run (default: python3).
# - SPARROW_ENGINE_DEVICE: CUDA device string (default: cuda:0).
# - SPARROW_ENGINE_MODEL_DIR: model directory used by detect-path cells.
# - SPARROW_ENGINE_TEST_MODEL: detector model id (default: megadetector-v6-yolov10e).
# - SPARROW_ENGINE_TEST_IMAGE: JPEG fixture for T5/T7/T10. If unset, a tiny
#   embedded JPEG is written under .phase-e-nvjpeg-dlopen/.
# - SPARROW_ENGINE_NVJPEG_TEST_TMP: runtime scratch root for mock libraries.
#   Defaults to /tmp as specified by the Phase E test contract.
#
# Exit codes per cell: 0 PASS, 1 FAIL, 2 SKIP.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARROW_ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SPARROW_ENGINE_DIR" || exit 1

PYTHON_BIN="${SPARROW_ENGINE_PYTHON:-python3}"
DEVICE="${SPARROW_ENGINE_DEVICE:-cuda:0}"
MODEL_ID="${SPARROW_ENGINE_TEST_MODEL:-megadetector-v6-yolov10e}"
RUNTIME_DIR="${SPARROW_ENGINE_NVJPEG_TEST_WORKDIR:-.phase-e-nvjpeg-dlopen}"
MOCK_ROOT="${SPARROW_ENGINE_NVJPEG_TEST_TMP:-/tmp}"
mkdir -p "$RUNTIME_DIR"

PASS=0
FAIL=0
SKIP=0

python_common=''
read -r -d '' python_common <<'PYCOMMON' || true
import os
from pathlib import Path

DEVICE = os.environ.get("SPARROW_ENGINE_DEVICE", "cuda:0")
MODEL_DIR = os.environ.get("SPARROW_ENGINE_MODEL_DIR", str(Path.home() / ".sparrow-engine" / "models"))
MODEL_ID = os.environ.get("SPARROW_ENGINE_TEST_MODEL", "megadetector-v6-yolov10e")


def message(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def expect_error(text: str, *needles: str) -> bool:
    lower = text.lower()
    return all(needle.lower() in lower for needle in needles)


def construct_engine():
    import sparrow_engine
    engine_cls = getattr(sparrow_engine, "Engine", None) or getattr(sparrow_engine, "PyEngine", None)
    if engine_cls is not None:
        return engine_cls(DEVICE, MODEL_DIR)
    sparrow_engine.init(device=DEVICE, model_dir=MODEL_DIR)
    return getattr(sparrow_engine, "_engine", None)


def detect_once(image_path: str):
    import sparrow_engine
    engine_cls = getattr(sparrow_engine, "Engine", None) or getattr(sparrow_engine, "PyEngine", None)
    if engine_cls is not None:
        engine = engine_cls(DEVICE, MODEL_DIR)
        return engine.detect([image_path], MODEL_ID)
    return sparrow_engine.detect(image_path, model=MODEL_ID)
PYCOMMON

write_fixture_jpeg() {
    local out="$RUNTIME_DIR/fixture.jpg"
    if [[ -n "${SPARROW_ENGINE_TEST_IMAGE:-}" ]]; then
        printf '%s\n' "$SPARROW_ENGINE_TEST_IMAGE"
        return 0
    fi
    "$PYTHON_BIN" - "$out" <<'PY'
import base64
import sys
from pathlib import Path
# 1x1 white JPEG, generated once and embedded to avoid a Pillow dependency.
jpeg_b64 = (
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////"
    "////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////////////"
    "//////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAX/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oADAMBAAIQAxAAAAH/"
    "xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAEFAqf/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/ASP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oA"
    "CAECAQE/ASP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAY/Al//xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/IV//2gAMAwEAAgADAAAAEP/"
    "xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEDAQE/ECP/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAECAQE/ECP/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/9oA"
    "CAEBAAE/EEP/2Q=="
)
path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_bytes(base64.b64decode(jpeg_b64))
print(path)
PY
}

compile_mock_major11() {
    local dir="$MOCK_ROOT/mock-nvjpeg-major11"
    mkdir -p "$dir"
    gcc -shared -fPIC -x c -o "$dir/libnvjpeg.so.12" - <<'C'
#include <stddef.h>
#include <stdint.h>
typedef void* nvjpegHandle_t;
typedef void* nvjpegJpegState_t;
typedef unsigned int nvjpegStatus_t;
typedef unsigned int libraryPropertyType_t;
typedef unsigned int nvjpegOutputFormat_t;
typedef int nvjpegChromaSubsampling_t;
typedef void* cudaStream_t;
typedef struct { unsigned char* channel[4]; size_t pitch[4]; } nvjpegImage_t;
nvjpegStatus_t nvjpegGetProperty(libraryPropertyType_t type, int* value) { if (value) *value = (type == 0 ? 11 : 0); return 0; }
nvjpegStatus_t nvjpegCreateSimple(nvjpegHandle_t* handle) { if (handle) *handle = (void*)0x1; return 6; }
nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t handle) { (void)handle; return 6; }
nvjpegStatus_t nvjpegJpegStateCreate(nvjpegHandle_t handle, nvjpegJpegState_t* state) { (void)handle; if (state) *state = (void*)0x2; return 6; }
nvjpegStatus_t nvjpegJpegStateDestroy(nvjpegJpegState_t state) { (void)state; return 6; }
nvjpegStatus_t nvjpegGetImageInfo(nvjpegHandle_t h, const unsigned char* data, size_t len, int* components, nvjpegChromaSubsampling_t* subsampling, int* widths, int* heights) { (void)h; (void)data; (void)len; (void)subsampling; if (components) *components = 3; if (widths) widths[0] = 1; if (heights) heights[0] = 1; return 6; }
nvjpegStatus_t nvjpegDecode(nvjpegHandle_t h, nvjpegJpegState_t s, const unsigned char* data, size_t len, nvjpegOutputFormat_t fmt, nvjpegImage_t* out, cudaStream_t stream) { (void)h; (void)s; (void)data; (void)len; (void)fmt; (void)out; (void)stream; return 6; }
C
    printf '%s\n' "$dir/libnvjpeg.so.12"
}

compile_mock_missing_decode() {
    local dir="$MOCK_ROOT/mock-nvjpeg-missing-decode"
    mkdir -p "$dir"
    gcc -shared -fPIC -x c -o "$dir/libnvjpeg.so.12" - <<'C'
#include <stddef.h>
#include <stdint.h>
typedef void* nvjpegHandle_t;
typedef void* nvjpegJpegState_t;
typedef unsigned int nvjpegStatus_t;
typedef unsigned int libraryPropertyType_t;
typedef int nvjpegChromaSubsampling_t;
nvjpegStatus_t nvjpegGetProperty(libraryPropertyType_t type, int* value) { if (value) *value = (type == 0 ? 12 : 0); return 0; }
nvjpegStatus_t nvjpegCreateSimple(nvjpegHandle_t* handle) { if (handle) *handle = (void*)0x1; return 6; }
nvjpegStatus_t nvjpegDestroy(nvjpegHandle_t handle) { (void)handle; return 6; }
nvjpegStatus_t nvjpegJpegStateCreate(nvjpegHandle_t handle, nvjpegJpegState_t* state) { (void)handle; if (state) *state = (void*)0x2; return 6; }
nvjpegStatus_t nvjpegJpegStateDestroy(nvjpegJpegState_t state) { (void)state; return 6; }
nvjpegStatus_t nvjpegGetImageInfo(nvjpegHandle_t h, const unsigned char* data, size_t len, int* components, nvjpegChromaSubsampling_t* subsampling, int* widths, int* heights) { (void)h; (void)data; (void)len; (void)subsampling; if (components) *components = 3; if (widths) widths[0] = 1; if (heights) heights[0] = 1; return 6; }
/* Intentionally omit nvjpegDecode for T3. */
C
    printf '%s\n' "$dir/libnvjpeg.so.12"
}

t1_absent() {
    SPARROW_ENGINE_NVJPEG_LIBRARY_PATH=/nonexistent \
    SPARROW_ENGINE_DEVICE="$DEVICE" \
    SPARROW_ENGINE_TEST_MODEL="$MODEL_ID" \
    "$PYTHON_BIN" - <<PY
$python_common
try:
    construct_engine()
except Exception as exc:
    text = message(exc)
    if expect_error(text, "LibraryNotFound") or expect_error(text, "libnvjpeg"):
        print("CELL T1 (nvjpeg absent — SPARROW_ENGINE_NVJPEG_LIBRARY_PATH=/nonexistent): PASS — NvjpegInitError::LibraryNotFound")
        raise SystemExit(0)
    print(f"CELL T1 (nvjpeg absent — SPARROW_ENGINE_NVJPEG_LIBRARY_PATH=/nonexistent): FAIL — unexpected error: {text}")
    raise SystemExit(1)
print("CELL T1 (nvjpeg absent — SPARROW_ENGINE_NVJPEG_LIBRARY_PATH=/nonexistent): FAIL — Engine initialized without nvjpeg")
raise SystemExit(1)
PY
}

t2_wrong_major() {
    local lib
    lib="$(compile_mock_major11)" || { echo "CELL T2 (wrong major — mock nvjpegGetProperty returns 11): FAIL — mock compile failed"; return 1; }
    SPARROW_ENGINE_NVJPEG_LIBRARY_PATH="$lib" SPARROW_ENGINE_DEVICE="$DEVICE" "$PYTHON_BIN" - <<PY
$python_common
try:
    construct_engine()
except Exception as exc:
    text = message(exc)
    if expect_error(text, "IncompatibleMajor") or (expect_error(text, "major") and "11" in text):
        print("CELL T2 (wrong major — mock nvjpegGetProperty returns 11): PASS — NvjpegInitError::IncompatibleMajor{found:11, expected:12}")
        raise SystemExit(0)
    print(f"CELL T2 (wrong major — mock nvjpegGetProperty returns 11): FAIL — unexpected error: {text}")
    raise SystemExit(1)
print("CELL T2 (wrong major — mock nvjpegGetProperty returns 11): FAIL — Engine initialized with wrong-major nvjpeg")
raise SystemExit(1)
PY
}

t3_missing_symbol() {
    local lib
    lib="$(compile_mock_missing_decode)" || { echo "CELL T3 (missing symbol — mock omits nvjpegDecode): FAIL — mock compile failed"; return 1; }
    SPARROW_ENGINE_NVJPEG_LIBRARY_PATH="$lib" SPARROW_ENGINE_DEVICE="$DEVICE" "$PYTHON_BIN" - <<PY
$python_common
try:
    construct_engine()
except Exception as exc:
    text = message(exc)
    if (expect_error(text, "SymbolMissing") and "nvjpegDecode" in text) or expect_error(text, "nvjpegDecode"):
        print('CELL T3 (missing symbol — mock omits nvjpegDecode): PASS — NvjpegInitError::SymbolMissing("nvjpegDecode")')
        raise SystemExit(0)
    print(f"CELL T3 (missing symbol — mock omits nvjpegDecode): FAIL — unexpected error: {text}")
    raise SystemExit(1)
print("CELL T3 (missing symbol — mock omits nvjpegDecode): FAIL — Engine initialized despite missing nvjpegDecode")
raise SystemExit(1)
PY
}

t4_concurrent() {
    SPARROW_ENGINE_NVJPEG_LIBRARY_PATH=/nonexistent SPARROW_ENGINE_DEVICE="$DEVICE" "$PYTHON_BIN" - <<PY
$python_common
import concurrent.futures

def call_engine(_):
    try:
        construct_engine()
        return "OK"
    except Exception as exc:
        return message(exc)

with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(call_engine, range(8)))
expected = [r for r in results if "LibraryNotFound" in r or "libnvjpeg" in r.lower()]
if len(expected) == 8:
    print("CELL T4 (concurrent first-call race — 8 threads × JpegDecoder::new): PASS — all 8 threads received the cached LibraryNotFound result")
    raise SystemExit(0)
print(f"CELL T4 (concurrent first-call race — 8 threads × JpegDecoder::new): FAIL — outcomes={results!r}")
raise SystemExit(1)
PY
}

t5_perf() {
    # Phase E (2026-05-25): SKIP cleanly when the test model is not installed
    # at SPARROW_ENGINE_MODEL_DIR. T5 measures the P3.8-Step1-Wave2 cached-
    # decode invariant (≤ 0.74 ms / call); it requires a real ONNX model to
    # run inference. The harness should not FAIL on a missing fixture.
    local model_dir="${SPARROW_ENGINE_MODEL_DIR:-$HOME/.sparrow-engine/models}"
    local manifest="$model_dir/$MODEL_ID/manifest.toml"
    if [[ ! -f "$manifest" ]]; then
        echo "CELL T5 (cached-decode regression — P3.8 invariant): SKIP — model manifest not found at $manifest (set SPARROW_ENGINE_MODEL_DIR or install $MODEL_ID)"
        return 2
    fi
    local image
    image="$(write_fixture_jpeg)" || { echo "CELL T5 (cached-decode regression — P3.8 invariant): FAIL — fixture setup failed"; return 1; }
    SPARROW_ENGINE_DEVICE="$DEVICE" SPARROW_ENGINE_TEST_MODEL="$MODEL_ID" "$PYTHON_BIN" - "$image" <<PY
$python_common
import statistics
import sys
import time
image = sys.argv[1]
try:
    import sparrow_engine
    engine_cls = getattr(sparrow_engine, "Engine", None) or getattr(sparrow_engine, "PyEngine", None)
    if engine_cls is not None:
        engine = engine_cls(DEVICE, MODEL_DIR)
        detector = lambda: engine.detect([image], MODEL_ID)
    else:
        detector = lambda: sparrow_engine.detect(image, model=MODEL_ID)
    detector()
    samples = []
    for _ in range(100):
        start = time.perf_counter_ns()
        detector()
        samples.append((time.perf_counter_ns() - start) / 1_000_000.0)
except Exception as exc:
    print(f"CELL T5 (cached-decode regression — P3.8 invariant): FAIL — detect loop failed: {message(exc)}")
    raise SystemExit(1)
median_ms = statistics.median(samples)
if median_ms <= 0.74:
    print(f"CELL T5 (cached-decode regression — P3.8 invariant): PASS — median {median_ms:.3f} ms / call (≤ 0.74 ms gate)")
    raise SystemExit(0)
print(f"CELL T5 (cached-decode regression — P3.8 invariant): FAIL — median {median_ms:.3f} ms / call (> 0.74 ms gate)")
raise SystemExit(1)
PY
}

t7_full_inference() {
    local image
    image="$(write_fixture_jpeg)" || { echo "CELL T7 (GPU host with hidden nvjpeg — full Engine.detect path): FAIL — fixture setup failed"; return 1; }
    SPARROW_ENGINE_NVJPEG_LIBRARY_PATH=/nonexistent \
    SPARROW_ENGINE_DEVICE="$DEVICE" \
    SPARROW_ENGINE_TEST_MODEL="$MODEL_ID" \
    "$PYTHON_BIN" - "$image" <<PY
$python_common
import sys
try:
    detect_once(sys.argv[1])
except Exception as exc:
    text = message(exc)
    if expect_error(text, "LibraryNotFound") or expect_error(text, "libnvjpeg"):
        print("CELL T7 (GPU host with hidden nvjpeg — full Engine.detect path): PASS — detect path returns NvjpegInitError::LibraryNotFound")
        raise SystemExit(0)
    print(f"CELL T7 (GPU host with hidden nvjpeg — full Engine.detect path): FAIL — unexpected error: {text}")
    raise SystemExit(1)
print("CELL T7 (GPU host with hidden nvjpeg — full Engine.detect path): FAIL — detect path succeeded with hidden nvjpeg")
raise SystemExit(1)
PY
}

t10_sidecar() {
    # Phase E (2026-05-25): two prerequisites — nvidia-nvjpeg-cu12 sidecar
    # PyPI wheel installed + a test model under SPARROW_ENGINE_MODEL_DIR.
    # SKIP cleanly if either is missing; this cell exercises the
    # ctypes.CDLL(RTLD_GLOBAL) preload via importlib.resources.files.
    local model_dir="${SPARROW_ENGINE_MODEL_DIR:-$HOME/.sparrow-engine/models}"
    local manifest="$model_dir/$MODEL_ID/manifest.toml"
    if [[ ! -f "$manifest" ]]; then
        echo "CELL T10 (sidecar shim end-to-end — nvidia-nvjpeg-cu12 in venv): SKIP — model manifest not found at $manifest"
        return 2
    fi
    local image
    image="$(write_fixture_jpeg)" || { echo "CELL T10 (sidecar shim end-to-end — nvidia-nvjpeg-cu12 in venv): FAIL — fixture setup failed"; return 1; }
    "$PYTHON_BIN" - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("nvidia.nvjpeg") is not None else 2)
PY
    local rc=$?
    if [[ $rc -eq 2 ]]; then
        echo "CELL T10 (sidecar shim end-to-end — nvidia-nvjpeg-cu12 in venv): SKIP — nvidia-nvjpeg-cu12 not in path"
        return 2
    fi
    unset SPARROW_ENGINE_NVJPEG_LIBRARY_PATH
    SPARROW_ENGINE_DEVICE="$DEVICE" SPARROW_ENGINE_TEST_MODEL="$MODEL_ID" "$PYTHON_BIN" - "$image" <<PY
$python_common
import sys
try:
    detect_once(sys.argv[1])
except Exception as exc:
    print(f"CELL T10 (sidecar shim end-to-end — nvidia-nvjpeg-cu12 in venv): FAIL — inference failed: {message(exc)}")
    raise SystemExit(1)
print("CELL T10 (sidecar shim end-to-end — nvidia-nvjpeg-cu12 in venv): PASS — preload via ctypes.CDLL RTLD_GLOBAL, inference round-trip OK")
raise SystemExit(0)
PY
}

t11_wrapper_overhead() {
    local bench_c="$RUNTIME_DIR/t11_wrapper_overhead.c"
    local bench_bin="$RUNTIME_DIR/t11_wrapper_overhead"
    cat > "$bench_c" <<'C'
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

volatile uint64_t sink = 0;
__attribute__((noinline, visibility("default"))) uint64_t nvjpeg_gate_t11_target(uint64_t x) { return x + 1; }

static uint64_t elapsed_ns(struct timespec a, struct timespec b) {
    return (uint64_t)(b.tv_sec - a.tv_sec) * 1000000000ull + (uint64_t)(b.tv_nsec - a.tv_nsec);
}

static int cmp_double(const void *a, const void *b) {
    double da = *(const double *)a;
    double db = *(const double *)b;
    return (da > db) - (da < db);
}

int main(void) {
    typedef uint64_t (*fn_t)(uint64_t);
    fn_t fp = (fn_t)dlsym(RTLD_DEFAULT, "nvjpeg_gate_t11_target");
    if (!fp) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        return 1;
    }
    const uint64_t iters = 10000000ull;
    double samples[9];
    for (int t = 0; t < 9; ++t) {
        struct timespec a, b;
        clock_gettime(CLOCK_MONOTONIC_RAW, &a);
        for (uint64_t i = 0; i < iters; ++i) sink += nvjpeg_gate_t11_target(i);
        clock_gettime(CLOCK_MONOTONIC_RAW, &b);
        double direct = (double)elapsed_ns(a, b) / (double)iters;
        clock_gettime(CLOCK_MONOTONIC_RAW, &a);
        for (uint64_t i = 0; i < iters; ++i) sink += fp(i);
        clock_gettime(CLOCK_MONOTONIC_RAW, &b);
        double indirect = (double)elapsed_ns(a, b) / (double)iters;
        double overhead = indirect - direct;
        samples[t] = overhead > 0.0 ? overhead : 0.0;
    }
    qsort(samples, 9, sizeof(samples[0]), cmp_double);
    printf("%.3f\n", samples[4]);
    return samples[4] <= 10.0 ? 0 : 1;
}
C
    gcc -O3 -rdynamic "$bench_c" -ldl -o "$bench_bin" || {
        echo "CELL T11 (wrapper overhead microbench — criterion): FAIL — gcc benchmark compile failed"
        return 1
    }
    local median
    median="$($bench_bin)"
    local rc=$?
    if [[ $rc -eq 0 ]]; then
        echo "CELL T11 (wrapper overhead microbench — criterion): PASS — median ${median} ns / call (≤ 10 ns gate)"
        return 0
    fi
    echo "CELL T11 (wrapper overhead microbench — criterion): FAIL — median ${median} ns / call (> 10 ns gate)"
    return 1
}

run_cell() {
    local cell="$1"
    local result rc status
    result="$($cell 2>&1)"
    rc=$?
    case "$rc" in
        0) status="PASS" ;;
        2) status="SKIP" ;;
        *) status="FAIL" ;;
    esac
    printf '%s\n' "$result" | tail -1
    case "$status" in
        PASS) PASS=$((PASS + 1)) ;;
        SKIP) SKIP=$((SKIP + 1)) ;;
        FAIL) FAIL=$((FAIL + 1)) ;;
    esac
}

for cell in \
    t1_absent \
    t2_wrong_major \
    t3_missing_symbol \
    t4_concurrent \
    t5_perf \
    t7_full_inference \
    t10_sidecar \
    t11_wrapper_overhead
    do
        run_cell "$cell"
    done

echo "Summary: PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
[[ "$FAIL" -eq 0 ]]
