#!/bin/bash
# Engine benchmark runner — all 4 configurations with comparison table.
#
# Run:
#   ORT_LIB_LOCATION=/tmp/ort-lib LD_LIBRARY_PATH=/tmp/ort-lib bash bench/run_engine_bench.sh
#
# Prerequisites:
#   - Rust toolchain (cargo)
#   - Python with uv
#   - ONNX model at test_files/onnx/
#   - Test images at test_files/test_cameratrap/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Auto-discover cuDNN for onnxruntime-gpu CUDA EP.
# System cuDNN is typically in PyTorch's lib directory.
CUDNN_DIR="/usr/lib/python3/dist-packages/torch/lib"
if [[ -f "$CUDNN_DIR/libcudnn.so" ]]; then
    export LD_LIBRARY_PATH="${CUDNN_DIR}:${LD_LIBRARY_PATH:-}"
fi

# Collect results: "name device total_ms per_image_ms detections"
declare -a RESULTS=()

# ---------------------------------------------------------------------------
# 1. Build Rust benchmark
# ---------------------------------------------------------------------------
echo "=== Building Rust engine_bench ==="
if ! cargo build --release --example engine_bench 2>&1; then
    echo "WARNING: Rust engine_bench build failed — skipping Rust benchmarks" >&2
    RUST_OK=0
else
    RUST_OK=1
fi

# ---------------------------------------------------------------------------
# 2. Run benchmarks
# ---------------------------------------------------------------------------

run_rust() {
    local device="$1"
    local label="rust_${device}"
    echo "=== Running Rust ($device) ==="
    local output
    if output=$(./target/release/examples/engine_bench --device "$device" 2>&1); then
        local result_line
        result_line=$(echo "$output" | grep "^RESULT " | tail -1)
        if [[ -n "$result_line" ]]; then
            RESULTS+=("$result_line")
            echo "$output" >&2
        else
            echo "WARNING: No RESULT line from Rust $device benchmark" >&2
            echo "$output" >&2
        fi
    else
        echo "WARNING: Rust $device benchmark failed" >&2
    fi
}

run_python() {
    local device="$1"
    echo "=== Running Python ($device) ==="
    local output
    if output=$(uv run --no-project --with onnxruntime-gpu --with numpy --with Pillow \
        python bench/python_ort_inference.py --device "$device" 2>&1); then
        local result_line
        result_line=$(echo "$output" | grep "^RESULT " | tail -1)
        if [[ -n "$result_line" ]]; then
            RESULTS+=("$result_line")
            echo "$output" >&2
        else
            echo "WARNING: No RESULT line from Python $device benchmark" >&2
            echo "$output" >&2
        fi
    else
        echo "WARNING: Python $device benchmark failed" >&2
        echo "$output" >&2
    fi
}

# Run all 4 configs
if [[ "$RUST_OK" -eq 1 ]]; then
    run_rust "auto"
    run_rust "cpu"
fi
run_python "gpu"
run_python "cpu"

# ---------------------------------------------------------------------------
# 3. Print comparison table
# ---------------------------------------------------------------------------
echo ""
echo "=============================================="
echo "         Engine Benchmark Comparison"
echo "=============================================="
printf "%-20s %-8s %10s %12s %10s\n" "Engine" "Device" "Total(ms)" "Per-img(ms)" "Detections"
printf "%-20s %-8s %10s %12s %10s\n" "------" "------" "---------" "-----------" "----------"

for r in "${RESULTS[@]}"; do
    # Format: RESULT <engine> <device> <total_ms> <per_image_ms> <detections>
    read -r _ engine device total_ms per_img_ms detections <<< "$r"
    printf "%-20s %-8s %10s %12s %10s\n" "$engine" "$device" "$total_ms" "$per_img_ms" "$detections"
done

echo "=============================================="
echo ""

# Also emit raw RESULT lines for downstream parsing
echo "--- Raw RESULT lines ---"
for r in "${RESULTS[@]}"; do
    echo "$r"
done
