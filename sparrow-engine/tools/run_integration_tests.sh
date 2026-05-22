#!/bin/bash
set -euo pipefail

# Shared ORT discovery: sets ORT_CAPI, ORT_LIB_LOCATION, ORT_PREFER_DYNAMIC_LINK, LD_LIBRARY_PATH.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "$SCRIPT_DIR/../scripts/ort-env.sh"

PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_DIR"

# Ensure output directory exists
mkdir -p test_outputs

echo "=== Phase B: Integration Tests ==="

# Step 1: Run detection tests
echo "[1/5] Running MDV6 + deepfaune detection tests..."
cargo test -p sparrow-engine-cpu --test integration_detect -- --ignored --test-threads=1 2>&1 | tee test_outputs/detect_test.log
echo "  Detection tests: DONE"

# Step 2: Run tiled detection tests
echo "[2/5] Running HerdNet tiled detection tests..."
cargo test -p sparrow-engine-cpu --test integration_tiled -- --ignored --test-threads=1 2>&1 | tee test_outputs/tiled_test.log
echo "  Tiled tests: DONE"

# Step 3: Run classification tests
echo "[3/5] Running SpeciesNet classification tests..."
cargo test -p sparrow-engine-cpu --test integration_classify -- --ignored --test-threads=1 2>&1 | tee test_outputs/classify_test.log
echo "  Classification tests: DONE"

# Step 4: Generate sparrow-engine-cpu visualizations
echo "[4/5] Generating sparrow-engine-cpu visualization overlays..."
uv run --no-project --with pillow tools/visualize_libsparrow_engine_outputs.py \
    --input test_outputs/libsparrow_engine \
    --images /home/miao/repos/PW_refactor/test_files/test_data \
    --output test_outputs/libsparrow_engine
echo "  Visualizations: DONE"

# Step 5: Run comparison
echo "[5/5] Comparing golden vs sparrow-engine-cpu outputs..."
uv run --no-project --with pillow tools/compare_outputs.py \
    --golden test_outputs/golden \
    --libsparrow-engine test_outputs/libsparrow_engine \
    --output test_outputs/comparison
echo "  Comparison: DONE"

echo ""
echo "=== Results ==="
echo "Golden overlays:    test_outputs/golden/"
echo "sparrow-engine-cpu overlays: test_outputs/libsparrow_engine/"
echo "Comparisons:        test_outputs/comparison/"
echo "Report:             test_outputs/comparison/report.json"

# Check report
if [ -f test_outputs/comparison/report.json ]; then
    python3 -c "
import json
with open('test_outputs/comparison/report.json') as f:
    r = json.load(f)
for model, info in r.get('models', {}).items():
    status = info.get('status', 'UNKNOWN')
    print(f'  {model}: {status}')
"
fi
