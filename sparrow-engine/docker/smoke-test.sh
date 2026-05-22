#!/usr/bin/env bash
set -euo pipefail

# Smoke test for sparrow-engine CPU Docker image.
# Usage: SPARROW_ENGINE_MODEL_DIR=/path/to/models TEST_IMAGE=/path/to/image.jpg bash smoke-test.sh
# Requires: bash, curl, docker, docker compose, python3

SPARROW_ENGINE_MODEL_DIR="${SPARROW_ENGINE_MODEL_DIR:?Set SPARROW_ENGINE_MODEL_DIR to a directory containing model subdirs}"
TEST_IMAGE="${TEST_IMAGE:-}"
COMPOSE_FILE="$(cd "$(dirname "$0")" && pwd)/docker-compose.yml"
COMPOSE="docker compose -f $COMPOSE_FILE --profile cpu"
PORT=8080
PASS=0
FAIL=0

cleanup() {
    echo "--- Container logs ---"
    $COMPOSE logs 2>/dev/null || true
    echo "--- Cleaning up ---"
    $COMPOSE down --remove-orphans 2>/dev/null || true
}
trap cleanup EXIT

check() {
    local desc="$1" expected_code="$2"
    shift 2
    local status_code
    status_code=$(curl -s -o /dev/null -w "%{http_code}" "$@" 2>/dev/null) || status_code="000"
    if [ "$status_code" = "$expected_code" ]; then
        echo "  PASS: $desc (HTTP $status_code)"
        PASS=$((PASS + 1))
    else
        echo "  FAIL: $desc (expected $expected_code, got $status_code)"
        FAIL=$((FAIL + 1))
    fi
}

echo "=== Building CPU image ==="
SPARROW_ENGINE_MODEL_DIR="$SPARROW_ENGINE_MODEL_DIR" $COMPOSE build sparrow-engine-cpu

echo "=== Starting container ==="
SPARROW_ENGINE_MODEL_DIR="$SPARROW_ENGINE_MODEL_DIR" $COMPOSE up -d

echo "=== Waiting for health check ==="
attempts=0
max_attempts=30
until curl -sf "http://localhost:${PORT}/v1/health" > /dev/null 2>&1; do
    attempts=$((attempts + 1))
    if [ "$attempts" -ge "$max_attempts" ]; then
        echo "FAIL: Server did not become healthy after ${max_attempts}s"
        $COMPOSE logs
        exit 1
    fi
    sleep 1
done
echo "  Server healthy after ${attempts}s"

echo "=== Running endpoint tests ==="

check "GET /v1/health" "200" "http://localhost:${PORT}/v1/health"
check "GET /healthz" "200" "http://localhost:${PORT}/healthz"
check "GET /v1/models" "200" "http://localhost:${PORT}/v1/models"

# Verify /v1/models returns a non-empty array
models_body=$(curl -s "http://localhost:${PORT}/v1/models")
model_count=$(echo "$models_body" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))" 2>/dev/null || echo "0")
if [ "$model_count" -gt 0 ]; then
    echo "  PASS: /v1/models returned $model_count model(s)"
    PASS=$((PASS + 1))

    # Extract first model ID for inference tests
    first_model=$(echo "$models_body" | python3 -c "import sys,json; print(json.load(sys.stdin)['models'][0]['id'])" 2>/dev/null || echo "")
else
    echo "  FAIL: /v1/models returned 0 models (is SPARROW_ENGINE_MODEL_DIR correct?)"
    FAIL=$((FAIL + 1))
    first_model=""
fi

# Inference tests require a test image and at least one loaded model
if [ -n "$TEST_IMAGE" ] && [ -f "$TEST_IMAGE" ] && [ -n "$first_model" ]; then
    check "POST /v1/detect" "200" \
        -X POST -F "image=@${TEST_IMAGE}" \
        "http://localhost:${PORT}/v1/detect?model=${first_model}"

    check "POST /v1/classify" "200" \
        -X POST -F "image=@${TEST_IMAGE}" \
        "http://localhost:${PORT}/v1/classify?model=${first_model}"
else
    echo "  SKIP: inference tests (set TEST_IMAGE and ensure models are loaded)"
fi

echo ""
echo "=== Results: ${PASS} passed, ${FAIL} failed ==="
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
