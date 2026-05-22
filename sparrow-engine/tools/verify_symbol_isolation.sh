#!/bin/bash
# Verify that only sparrow_engine_* symbols are exported from the cdylib.
# Usage: ./tools/verify_symbol_isolation.sh [path/to/libsparrow_engine.so]
set -euo pipefail

LIB="${1:-target/release/libsparrow_engine.so}"

if [ ! -f "$LIB" ]; then
    echo "FAIL: $LIB not found. Build with:"
    echo "  ORT_LIB_LOCATION=/tmp/ort-lib ORT_PREFER_DYNAMIC_LINK=1 \\"
    echo "    cargo rustc -p sparrow-engine-cpu --features ffi --crate-type cdylib --release"
    exit 1
fi

echo "=== Symbol isolation check: $LIB ==="
echo ""

# 1. Positive: sparrow_engine_* symbols must be present
SPARROW_ENGINE_SYMS=$(nm -D "$LIB" | grep " T sparrow_engine_" | wc -l)
echo "sparrow_engine_* exported symbols: $SPARROW_ENGINE_SYMS"
nm -D "$LIB" | grep " T sparrow_engine_" | awk '{print "  " $3}'

if [ "$SPARROW_ENGINE_SYMS" -eq 0 ]; then
    echo "FAIL: no sparrow_engine_* symbols found"
    exit 1
fi

echo ""

# 2. Negative: no non-sparrow_engine text symbols should be exported
LEAKED=$(nm -D "$LIB" | grep " T " | { grep -v "sparrow_engine_" || true; } | wc -l)
echo "Non-sparrow_engine exported T symbols: $LEAKED"

if [ "$LEAKED" -gt 0 ]; then
    echo "FAIL: leaked text symbols:"
    nm -D "$LIB" | grep " T " | grep -v "sparrow_engine_"
    exit 1
fi

echo ""

# 3. Weak symbols (W) — flag only ORT/protobuf leaks
ORT_PATTERN="onnxruntime\|Ort\|protobuf\|ort_\|OrtApi\|google::protobuf"
WEAK_ORT=$({ nm -D "$LIB" | grep " W " || true; } | { grep -v "sparrow_engine_" || true; } | { grep -i "$ORT_PATTERN" || true; } | wc -l)
echo "Weak ORT/protobuf symbols (W): $WEAK_ORT"

if [ "$WEAK_ORT" -gt 0 ]; then
    echo "WARNING: leaked weak ORT/protobuf symbols:"
    { nm -D "$LIB" | grep " W " || true; } | grep -v "sparrow_engine_" | grep -i "$ORT_PATTERN"
    # Weak symbols are a warning, not a hard fail (linker may resolve them)
fi

echo ""

# 4. Data symbols (D/B) — flag only ORT/protobuf leaks
DATA_ORT=$({ nm -D "$LIB" | grep " [DB] " || true; } | { grep -v "sparrow_engine_" || true; } | { grep -i "$ORT_PATTERN" || true; } | wc -l)
echo "Data ORT/protobuf symbols (D/B): $DATA_ORT"

if [ "$DATA_ORT" -gt 0 ]; then
    echo "WARNING: leaked data ORT/protobuf symbols:"
    { nm -D "$LIB" | grep " [DB] " || true; } | grep -v "sparrow_engine_" | grep -i "$ORT_PATTERN"
fi

echo ""

# Summary
if [ "$WEAK_ORT" -gt 0 ] || [ "$DATA_ORT" -gt 0 ]; then
    echo "PASS (with warnings): $SPARROW_ENGINE_SYMS sparrow_engine_* symbols, 0 leaked T, $WEAK_ORT weak ORT, $DATA_ORT data ORT"
else
    echo "PASS: $SPARROW_ENGINE_SYMS sparrow_engine_* symbols, 0 leaked"
fi
