#!/usr/bin/env bash
# Run cargo tests with ORT dynamic linking.
#
# GPU is the default. The script prefers onnxruntime-gpu over onnxruntime-cpu.
# The pre-built ORT static lib (ort-sys) requires glibc 2.38+.
# Ubuntu 22.04 has glibc 2.35, so we link against the pip onnxruntime
# shared library instead.
#
# Usage:
#   ./scripts/test.sh                    # run all tests, debug profile (GPU preferred)
#   ./scripts/test.sh --release          # run all tests, release profile
#   ./scripts/test.sh -p sparrow-engine-cpu --lib  # pass extra cargo test args
#   ./scripts/test.sh --release -p sparrow-engine-cpu --lib  # combine
#   ORT_DIR=/custom/path ./scripts/test.sh  # override ORT location

set -euo pipefail

# Shared ORT discovery: sets ORT_CAPI, ORT_LIB_LOCATION, ORT_PREFER_DYNAMIC_LINK, LD_LIBRARY_PATH.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/ort-env.sh"

# sparrow-engine-python tests (--no-default-features) link to libpython.
# Add python lib dir if available.
PYLIB_DIR=$(python3-config --ldflags 2>/dev/null | grep -oP '(?<=-L)\S+' | head -1 || true)
if [[ -n "$PYLIB_DIR" ]]; then
    export LD_LIBRARY_PATH="${ORT_CAPI}:${PYLIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Enforce: forbid eprintln!/println! in sparrow-engine-python/src/lib.rs (Phase 3.5 S6).
# Load-bearing — every former site is now tracing::warn!/tracing::info!; a new
# eprintln!/println! fails this script and the test pipeline. Set
# SPARROW_ENGINE_PY_GUARD_STRICT=0 to demote to warn for transient debugging.
"$SCRIPT_DIR/guard_no_print.sh"

# Extract --release from args so cargo sees it as a build flag, not a test-binary
# flag (cargo test routes anything after '--' to the test binary).
CARGO_FLAGS=()
CARGO_ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--release" ]]; then
        CARGO_FLAGS+=("--release")
    else
        CARGO_ARGS+=("$arg")
    fi
done

# If no non-flag args given, run workspace-wide but handle sparrow-engine-python specially:
# it has `default = ["extension-module"]`, which asks pyo3 NOT to link libpython
# (symbols come from the host Python at runtime). `cargo test` builds a standalone
# lib-test binary that then cannot link — rust-lld errors on Py_* symbols.
# Workaround: exclude sparrow-engine-python from --workspace, then run it separately with
# --lib --no-default-features so pyo3 links libpython statically.
# --test-threads=1 because the ORT engine is a process-global singleton.
if [[ ${#CARGO_ARGS[@]} -eq 0 ]]; then
    echo "Running: cargo test --workspace --exclude sparrow-engine-python ${CARGO_FLAGS[*]:-} -- --test-threads=1"
    echo "---"
    cargo test --workspace --exclude sparrow-engine-python ${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"} -- --test-threads=1
    echo "---"
    echo "Running: cargo test -p sparrow-engine-python --lib --no-default-features ${CARGO_FLAGS[*]:-} -- --test-threads=1"
    echo "---"
    exec cargo test -p sparrow-engine-python --lib --no-default-features ${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"} -- --test-threads=1
else
    echo "Running: cargo test ${CARGO_FLAGS[*]:-} ${CARGO_ARGS[*]}"
    echo "---"
    exec cargo test ${CARGO_FLAGS[@]+"${CARGO_FLAGS[@]}"} "${CARGO_ARGS[@]}"
fi
