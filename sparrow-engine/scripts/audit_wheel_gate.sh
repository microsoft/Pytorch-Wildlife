#!/usr/bin/env bash
# Phase E nvjpeg dlopen audit gates.
#
# Runs:
#   T9: `nm -u` purity check on the GPU cdylib. Undefined nvjpeg* symbols mean
#       a static extern binding leaked back into the build.
#   T6: `auditwheel show` + `auditwheel repair` for the GPU Python wheel.
#
# Edge cases:
# - No GPU host / no prior GPU cdylib build: the T9 preflight aborts cleanly with
#   the exact missing cdylib path(s). Build first with
#   `cargo build -p sparrow-engine-gpu --release --features ffi`.
# - No wheel under dist/ or target/wheels/: aborts cleanly; this script does not
#   build wheels itself.
# - `auditwheel` missing: aborts before running T6 with an operator action.
# - Runtime logs default to a project-local file to avoid relying on host temp dirs;
#   set AUDITWHEEL_SHOW_LOG to override.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARROW_ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SPARROW_ENGINE_DIR"

find_wheel() {
    local candidates=()
    while IFS= read -r path; do candidates+=("$path"); done < <(
        { compgen -G 'dist/sparrow_engine_gpu-*.whl' || true; \
          compgen -G 'target/wheels/sparrow_engine_gpu-*.whl' || true; \
          compgen -G 'sparrow-engine-python/dist/sparrow_engine_gpu-*.whl' || true; } | sort -r
    )
    if [[ ${#candidates[@]} -gt 0 ]]; then
        printf '%s\n' "${candidates[0]}"
    fi
}

WHEEL="${1:-$(find_wheel)}"
[[ -f "$WHEEL" ]] || { echo "FAIL: no GPU wheel found"; exit 1; }

command -v nm >/dev/null 2>&1 || { echo "FAIL: nm not found in PATH"; exit 1; }
command -v auditwheel >/dev/null 2>&1 || { echo "FAIL: auditwheel not found in PATH"; exit 1; }

CDYLIB="${SPARROW_ENGINE_GPU_CDYLIB:-}"
if [[ -z "$CDYLIB" ]]; then
    if [[ -f target/release/libsparrow_engine.so ]]; then
        CDYLIB="target/release/libsparrow_engine.so"
    elif [[ -f target-gpu/release/libsparrow_engine.so ]]; then
        CDYLIB="target-gpu/release/libsparrow_engine.so"
    fi
fi
[[ -n "$CDYLIB" && -f "$CDYLIB" ]] || {
    echo "FAIL: GPU cdylib not found (tried target/release/libsparrow_engine.so and target-gpu/release/libsparrow_engine.so)"
    exit 1
}

# T9 — nm -u purity gate: cdylib must NOT have undefined nvjpeg* symbols.
if nm -u "$CDYLIB" 2>/dev/null | grep -E '^\s*U\s+nvjpeg'; then
    echo "CELL T9 (nm -u purity): FAIL — undefined nvjpeg symbol(s) in cdylib"
    exit 1
else
    echo "CELL T9 (nm -u purity): PASS — no undefined nvjpeg symbols"
fi

# T6 — auditwheel show + repair. Keep a log because the manual-test plan asks
# operators to grep the auditwheel output when signing off Phase E.
AUDITWHEEL_SHOW_LOG="${AUDITWHEEL_SHOW_LOG:-.phase-e-auditwheel-show.txt}"
if ! auditwheel show "$WHEEL" | tee "$AUDITWHEEL_SHOW_LOG"; then
    echo "CELL T6 (auditwheel show): FAIL — auditwheel show failed"
    exit 1
fi

grep -q 'manylinux_2_28_x86_64' "$AUDITWHEEL_SHOW_LOG" || {
    echo "CELL T6 (auditwheel show): FAIL — not manylinux_2_28_x86_64"
    exit 1
}
if grep -q 'libnvjpeg' "$AUDITWHEEL_SHOW_LOG"; then
    echo "CELL T6 (auditwheel show): FAIL — libnvjpeg in DT_NEEDED"
    exit 1
fi
echo "CELL T6 (auditwheel show): PASS — manylinux_2_28_x86_64, no nvjpeg DT_NEEDED"

mkdir -p dist-repaired
if ! auditwheel repair --plat manylinux_2_28_x86_64 --exclude libonnxruntime.so.1 \
    --wheel-dir dist-repaired/ "$WHEEL"; then
    echo "CELL T6 (auditwheel repair): FAIL — auditwheel repair failed"
    exit 1
fi
echo "CELL T6 (auditwheel repair): PASS — wheel written to dist-repaired/"
