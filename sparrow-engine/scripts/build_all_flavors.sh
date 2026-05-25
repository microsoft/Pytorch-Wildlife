#!/usr/bin/env bash
# Phase 3.8 Phase C local build matrix runner, updated for Phase E nvjpeg dlopen gates.
#
# Builds the dual-flavor artifacts that ship out of sparrow-engine:
#   - sparrow-engine-cpu / sparrow-engine-gpu cdylibs (libsparrow_engine.so)
#   - spe / spe-gpu CLI binaries
#   - sparrow-engine / sparrow-engine-gpu Python wheels
#   - sparrow-engine CPU / GPU Docker images
#
# Usage:
#   scripts/build_all_flavors.sh                # full matrix
#   FLAVOR=cpu  scripts/build_all_flavors.sh    # CPU-only artifacts
#   FLAVOR=gpu  scripts/build_all_flavors.sh    # GPU-only artifacts
#   STAGE=lib   scripts/build_all_flavors.sh    # cdylibs only
#   STAGE=cli   scripts/build_all_flavors.sh    # CLI binaries only
#   STAGE=wheel scripts/build_all_flavors.sh    # Python wheels only
#   STAGE=docker scripts/build_all_flavors.sh   # Docker images only
#
# Phase E: when a GPU wheel is built, this script fail-fast runs:
#   1. scripts/audit_wheel_gate.sh     (T6 auditwheel + T9 nm -u)
#   2. scripts/test_nvjpeg_dlopen.sh   (T1-T5 + T7 + T10 + T11)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARROW_ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
REPO_ROOT="$(cd "$SPARROW_ENGINE_DIR/.." && pwd)"
ARTIFACTS="${ARTIFACTS:-$SPARROW_ENGINE_DIR/target/phase-c-artifacts}"
mkdir -p "$ARTIFACTS"

FLAVOR="${FLAVOR:-both}"
STAGE="${STAGE:-all}"

# Source ORT discovery so cargo/maturin picks up the pip-shipped ORT dylib on
# Ubuntu hosts where the prebuilt ORT static library cannot be used.
# shellcheck source=/dev/null
source "$SPARROW_ENGINE_DIR/scripts/ort-env.sh"

build_lib_cpu() {
    echo "[build_all_flavors] === CPU cdylib (libsparrow_engine.so) ==="
    cd "$SPARROW_ENGINE_DIR"
    cargo clean -p sparrow-engine-cpu -p sparrow-engine-gpu
    cargo build -p sparrow-engine-cpu --release --features ffi
    cp target/release/libsparrow_engine.so "$ARTIFACTS/libsparrow_engine_cpu.so"
    nm -D "$ARTIFACTS/libsparrow_engine_cpu.so" | grep -E ' T sparrow_engine_' | awk '{print $3}' | sort >"$ARTIFACTS/symbols_cpu.txt"
    echo "[build_all_flavors]   $(wc -l <"$ARTIFACTS/symbols_cpu.txt") FFI symbols"
}

build_lib_gpu() {
    echo "[build_all_flavors] === GPU cdylib (libsparrow_engine.so) ==="
    cd "$SPARROW_ENGINE_DIR"
    cargo clean -p sparrow-engine-cpu -p sparrow-engine-gpu
    cargo build -p sparrow-engine-gpu --release --features ffi
    cp target/release/libsparrow_engine.so "$ARTIFACTS/libsparrow_engine_gpu.so"
    nm -D "$ARTIFACTS/libsparrow_engine_gpu.so" | grep -E ' T sparrow_engine_' | awk '{print $3}' | sort >"$ARTIFACTS/symbols_gpu.txt"
    echo "[build_all_flavors]   $(wc -l <"$ARTIFACTS/symbols_gpu.txt") FFI symbols"
}

verify_g5_symbols() {
    echo "[build_all_flavors] === G5 symbol diff ==="
    if diff "$ARTIFACTS/symbols_cpu.txt" "$ARTIFACTS/symbols_gpu.txt" >/dev/null; then
        echo "[build_all_flavors]   G5 PASS: $(wc -l <"$ARTIFACTS/symbols_cpu.txt") symbols identical"
    else
        echo "[build_all_flavors]   G5 FAIL — symbol diff:" >&2
        diff "$ARTIFACTS/symbols_cpu.txt" "$ARTIFACTS/symbols_gpu.txt" >&2
        return 1
    fi
}

build_cli_cpu() {
    echo "[build_all_flavors] === CPU CLI (spe) ==="
    cd "$SPARROW_ENGINE_DIR"
    cargo build -p sparrow-engine-cli --release --no-default-features --features cpu --bin spe
    cp target/release/spe "$ARTIFACTS/spe"
    echo "[build_all_flavors]   spe size: $(stat -c '%s' "$ARTIFACTS/spe") bytes"
}

build_cli_gpu() {
    echo "[build_all_flavors] === GPU CLI (spe-gpu) ==="
    cd "$SPARROW_ENGINE_DIR"
    cargo build -p sparrow-engine-cli --release --no-default-features --features gpu --bin spe-gpu
    cp target/release/spe-gpu "$ARTIFACTS/spe-gpu"
    echo "[build_all_flavors]   spe-gpu size: $(stat -c '%s' "$ARTIFACTS/spe-gpu") bytes"
}

run_phase_e_gpu_gates() {
    echo "[build_all_flavors] === Phase E GPU wheel gates ==="
    cd "$SPARROW_ENGINE_DIR"
    # Rebuild the GPU cdylib immediately before T9 so target/release contains
    # the GPU flavor, not a stale CPU lib with the same filename.
    cargo clean -p sparrow-engine-cpu -p sparrow-engine-gpu
    cargo build -p sparrow-engine-gpu --release --features ffi
    bash scripts/audit_wheel_gate.sh
    bash scripts/test_nvjpeg_dlopen.sh
}

build_wheels() {
    echo "[build_all_flavors] === Python wheels ==="
    cd "$SPARROW_ENGINE_DIR/sparrow-engine-python"
    SPARROW_ENGINE_FLAVOR="$FLAVOR" ./build.sh
    ls -lh "$SPARROW_ENGINE_DIR/target/wheels/"
    if [[ "$FLAVOR" == "gpu" || "$FLAVOR" == "both" ]]; then
        run_phase_e_gpu_gates
    fi
}

build_docker() {
    echo "[build_all_flavors] === Docker images ==="
    cd "$REPO_ROOT"
    if [[ "$FLAVOR" == "cpu" || "$FLAVOR" == "both" ]]; then
        docker build -f sparrow-engine/docker/Dockerfile.cpu -t sparrow-engine:cpu-phase-c-w5 sparrow-engine
    fi
    if [[ "$FLAVOR" == "gpu" || "$FLAVOR" == "both" ]]; then
        docker build -f sparrow-engine/docker/Dockerfile.gpu -t sparrow-engine:gpu-phase-c-w5 sparrow-engine
    fi
    docker images sparrow-engine
}

case "$STAGE" in
    lib)
        if [[ "$FLAVOR" == "cpu" || "$FLAVOR" == "both" ]]; then build_lib_cpu; fi
        if [[ "$FLAVOR" == "gpu" || "$FLAVOR" == "both" ]]; then build_lib_gpu; fi
        if [[ "$FLAVOR" == "both" ]]; then verify_g5_symbols; fi
        ;;
    cli)
        if [[ "$FLAVOR" == "cpu" || "$FLAVOR" == "both" ]]; then build_cli_cpu; fi
        if [[ "$FLAVOR" == "gpu" || "$FLAVOR" == "both" ]]; then build_cli_gpu; fi
        ;;
    wheel)
        build_wheels
        ;;
    docker)
        build_docker
        ;;
    all)
        if [[ "$FLAVOR" == "cpu" || "$FLAVOR" == "both" ]]; then build_lib_cpu; fi
        if [[ "$FLAVOR" == "gpu" || "$FLAVOR" == "both" ]]; then build_lib_gpu; fi
        if [[ "$FLAVOR" == "both" ]]; then verify_g5_symbols; fi
        if [[ "$FLAVOR" == "cpu" || "$FLAVOR" == "both" ]]; then build_cli_cpu; fi
        if [[ "$FLAVOR" == "gpu" || "$FLAVOR" == "both" ]]; then build_cli_gpu; fi
        build_wheels
        build_docker
        ;;
    *)
        echo "[build_all_flavors] ERROR: STAGE must be lib / cli / wheel / docker / all (got '$STAGE')" >&2
        exit 1
        ;;
esac

echo
echo "[build_all_flavors] Artifacts: $ARTIFACTS"
ls -lh "$ARTIFACTS" 2>/dev/null || true
