#!/usr/bin/env bash
#
# Phase 3.5 Wave 5 — clean-room install test orchestrator.
#
# Runs the 2×3 CPU install matrix (Ubuntu 22.04 + 24.04 × CLI + Python wheel
# + pytorchwildlife shim wheel) inside throwaway Docker containers that have
# NOTHING sparrow-engine-adjacent pre-installed. Exits non-zero on any cell failure.
#
# What it is NOT:
# - A CI job. Run it locally when you need to mimic a fresh user's box.
# - A public-release gate. GitHub Actions, push-tests, PyPI install paths
#   are all deliberately out of scope (see
#   docs/design/phase3.5/final_design.md §S11 reshape).
#
# What it catches:
# - dev-env leakage: missing-at-fresh-install system libs, undeclared
#   runtime deps, broken wheel metadata, ort-env.sh / LD_LIBRARY_PATH
#   assumptions that silently work on the developer box.
#
# Usage:
#   ./scripts/clean_room_test.sh                # full matrix, build missing artifacts
#   ./scripts/clean_room_test.sh --no-build     # fail loudly if any artifact is missing
#   ./scripts/clean_room_test.sh --only cli     # CLI-only across both OS
#   ./scripts/clean_room_test.sh --only python  # Python wheel only
#   ./scripts/clean_room_test.sh --only shim    # pytorchwildlife-compat only
#   ./scripts/clean_room_test.sh --os 22        # Ubuntu 22.04 only
#   ./scripts/clean_room_test.sh --os 24        # Ubuntu 24.04 only
#
# Results append to docs/review/phase3.5-clean-room/results.md on each run.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths & config
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SPARROW_ENGINE_DIR="$REPO_ROOT/sparrow-engine"
PW_COMPAT_DIR="$REPO_ROOT/pytorchwildlife-compat"
RESULTS_MD="$REPO_ROOT/docs/review/phase3.5-clean-room/results.md"

CLI_BIN="$SPARROW_ENGINE_DIR/target/release/spe"
CLI_TARBALL_GLOB="$SPARROW_ENGINE_DIR/dist/sparrow-engine-cpu-*-linux-x86_64.tar.gz"
PY_WHEEL_GLOB="$SPARROW_ENGINE_DIR/sparrow-engine-python/dist/sparrow_engine-*.whl"
SHIM_WHEEL_GLOB="$PW_COMPAT_DIR/dist/pytorchwildlife-*.whl"

MATRIX_OS=(22 24)
MATRIX_ARTIFACT=(cli python shim)

BUILD_MISSING=1
declare -A CELL_RESULT=()

# ---------------------------------------------------------------------------
# Argparse (minimal)
# ---------------------------------------------------------------------------
ONLY_ARTIFACT=""
ONLY_OS=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-build) BUILD_MISSING=0; shift;;
    --only) ONLY_ARTIFACT="$2"; shift 2;;
    --os) ONLY_OS="$2"; shift 2;;
    -h|--help)
      # Skip line 1 (shebang) and stop at first non-comment line.
      # Strips leading "# " from comment lines for readability.
      awk 'NR==1 {next} /^[^#]/ {exit} {sub(/^# ?/,""); print}' "$0"
      exit 0;;
    *) echo "unknown arg: $1" >&2; exit 2;;
  esac
done

if [[ -n "$ONLY_OS" ]]; then
  case "$ONLY_OS" in
    22) MATRIX_OS=(22);;
    24) MATRIX_OS=(24);;
    *) echo "invalid --os: must be 22 or 24" >&2; exit 2;;
  esac
fi
if [[ -n "$ONLY_ARTIFACT" ]]; then
  case "$ONLY_ARTIFACT" in
    cli|python|shim) MATRIX_ARTIFACT=("$ONLY_ARTIFACT");;
    *) echo "invalid --only: must be cli, python, or shim" >&2; exit 2;;
  esac
fi

# ---------------------------------------------------------------------------
# Pretty logging
# ---------------------------------------------------------------------------
LOG() { printf '[clean-room] %s\n' "$*" >&2; }
FAIL() { printf '[clean-room] FAIL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Artifact discovery / build
# ---------------------------------------------------------------------------
find_first() {
  # Return the first match of a glob, or empty if nothing matches.
  local g="$1"
  compgen -G "$g" | head -n1 || true
}

ensure_cli_bin() {
  if [[ -x "$CLI_BIN" ]]; then
    LOG "CLI binary present: $CLI_BIN ($(du -h "$CLI_BIN" | cut -f1))"
  else
    (( BUILD_MISSING )) || FAIL "CLI binary missing at $CLI_BIN; --no-build forbids build"
    LOG "Building CLI binary (cargo build --release -p sparrow-engine-cli)..."
    ( cd "$SPARROW_ENGINE_DIR" && cargo build --release -p sparrow-engine-cli )
    [[ -x "$CLI_BIN" ]] || FAIL "CLI build finished but $CLI_BIN still missing"
  fi
  # Preflight: production CLI must NOT DT_NEEDED libonnxruntime.
  # The release tarballs (RP-4 / 2026-05-26) bundle `libonnxruntime.so.X.Y.Z`
  # under `lib/` and resolve it at runtime via the in-binary
  # `ort_resolver::init_ort_env()` shim (PW commit cdbdb39). The `ort`
  # crate's `load-dynamic` feature in sparrow-engine-{cpu,gpu}/Cargo.toml
  # means `libonnxruntime` is `dlopen`'d, never DT_NEEDED-linked. If
  # readelf shows DT_NEEDED libonnxruntime, the dev box is producing a
  # statically-broken binary that contradicts the documented load-dynamic
  # contract — fail the harness rather than mount ORT into the clean-room
  # container as a workaround (D2=HYBRID forbids that).
  if command -v readelf >/dev/null 2>&1; then
    if readelf -d "$CLI_BIN" 2>/dev/null | grep -q 'NEEDED.*libonnxruntime'; then
      FAIL "CLI binary has DT_NEEDED libonnxruntime.so.* — load-dynamic contract violated (RP-3 / RP-4). Check sparrow-engine-{cpu,gpu}/Cargo.toml has \`ort\` built with \`load-dynamic\`."
    fi
  else
    LOG "warning: readelf not available — skipping CLI load-dynamic preflight check"
  fi
}

ensure_cli_tarball() {
  local tarball
  tarball="$(find_first "$CLI_TARBALL_GLOB")"
  if [[ -n "$tarball" ]]; then
    LOG "CLI tarball present: $tarball"
    return
  fi
  (( BUILD_MISSING )) || FAIL "CLI tarball missing (glob $CLI_TARBALL_GLOB); --no-build forbids build"
  ensure_cli_bin

  local work_dir="$SPARROW_ENGINE_DIR/target/clean-room"
  local ort_venv="$work_dir/ort-venv"
  mkdir -p "$work_dir"
  rm -rf "$ort_venv"
  LOG "Staging ORT runtime for CLI tarball (onnxruntime>=1.25.1,<1.26)..."
  python3 -m venv "$ort_venv"
  "$ort_venv/bin/pip" install --quiet "onnxruntime>=1.25.1,<1.26"
  local ort_capi
  ort_capi=$("$ort_venv/bin/python" -c 'import onnxruntime, pathlib; print(pathlib.Path(onnxruntime.__file__).parent / "capi")')
  local version
  version=$(awk -F' = ' '/^version = / {gsub(/"/, "", $2); print $2; exit}' "$SPARROW_ENGINE_DIR/sparrow-engine-cli/Cargo.toml")
  [[ -n "$version" ]] || FAIL "could not determine CLI version from Cargo.toml"
  LOG "Packaging CLI tarball (version=$version, ORT_CAPI=$ort_capi)..."
  ( cd "$SPARROW_ENGINE_DIR" && ORT_STAGE_DIR="$ort_capi" FLAVOR=cpu PLATFORM=linux-x86_64 VERSION="$version" ./scripts/package_cli_tarball.sh )
  tarball="$(find_first "$CLI_TARBALL_GLOB")"
  [[ -n "$tarball" ]] || FAIL "CLI tarball build finished but nothing under $CLI_TARBALL_GLOB"
}

ensure_python_wheel() {
  local wheel
  wheel="$(find_first "$PY_WHEEL_GLOB")"
  if [[ -n "$wheel" ]]; then
    LOG "Python wheel present: $wheel"
    return
  fi
  (( BUILD_MISSING )) || FAIL "Python wheel missing (glob $PY_WHEEL_GLOB); --no-build forbids build"
  # Clear any stale wheels (e.g. cp310 from a prior Python-version build)
  # so find_first below picks the freshly-built wheel deterministically.
  # Without this, mixed cp3XX ABI tags can confuse the smoke containers
  # after a host-Python bump (R1 inquisitor F-11).
  rm -f "$SPARROW_ENGINE_DIR/sparrow-engine-python/dist/sparrow_engine-"*.whl
  LOG "Building Python wheel (maturin build --release --no-default-features for CPU)..."
  ( cd "$SPARROW_ENGINE_DIR/sparrow-engine-python" && uv run --with maturin maturin build --release --out dist )
  wheel="$(find_first "$PY_WHEEL_GLOB")"
  [[ -n "$wheel" ]] || FAIL "Python wheel build finished but nothing under $PY_WHEEL_GLOB"
}

ensure_shim_wheel() {
  local wheel
  wheel="$(find_first "$SHIM_WHEEL_GLOB")"
  if [[ -n "$wheel" ]]; then
    LOG "Shim wheel present: $wheel"
    return
  fi
  (( BUILD_MISSING )) || FAIL "Shim wheel missing (glob $SHIM_WHEEL_GLOB); --no-build forbids build"
  # Clear any stale wheels (e.g. a prior version bump) so find_first below
  # picks the freshly-built wheel deterministically. Mirrors F-11 in
  # ensure_python_wheel; without this, a stale shim wheel could be picked
  # up and tested against a newer sparrow-engine wheel, papering over a co-bump miss.
  # Scoped to *.whl only — uv build also writes *.tar.gz which we do NOT
  # need to clear (sdists are harmless for the clean-room smoke).
  rm -f "$PW_COMPAT_DIR/dist/pytorchwildlife-"*.whl
  LOG "Building shim wheel (uv build)..."
  ( cd "$PW_COMPAT_DIR" && uv build )
  wheel="$(find_first "$SHIM_WHEEL_GLOB")"
  [[ -n "$wheel" ]] || FAIL "Shim build finished but nothing under $SHIM_WHEEL_GLOB"
}

# ---------------------------------------------------------------------------
# Image build
# ---------------------------------------------------------------------------
ensure_image() {
  # $1 = os (22 or 24)
  local os="$1"
  local tag="sparrow-engine-clean-room-u${os}:latest"
  local df="$SCRIPT_DIR/clean_room/Dockerfile.ubuntu${os}"
  [[ -f "$df" ]] || FAIL "missing Dockerfile: $df"
  LOG "Building $tag from $df ..."
  docker build --quiet -t "$tag" -f "$df" "$SCRIPT_DIR/clean_room" >&2
  printf '%s' "$tag"
}

# ---------------------------------------------------------------------------
# Smoke tests per artifact
# ---------------------------------------------------------------------------
# Each returns 0 on PASS, 2 on preflight SKIP (smoke_python/smoke_shim only;
# wheel-ABI mismatch — see C1), non-zero (other) on FAIL.
#
# Smoke tests intentionally avoid model downloads and inference — we're
# verifying the INSTALL path, not the inference path. `spe models list`
# without models installed still prints an empty list + exit 0.

smoke_cli() {
  # $1 = image tag
  local tag="$1"
  local tarball
  tarball="$(find_first "$CLI_TARBALL_GLOB")"
  [[ -n "$tarball" ]] || FAIL "CLI tarball missing (glob $CLI_TARBALL_GLOB)"
  local tarball_name
  tarball_name="$(basename "$tarball")"
  docker run --rm \
    -e TARBALL_NAME="$tarball_name" \
    -v "$tarball:/clean_room/$tarball_name:ro" \
    "$tag" \
    bash -c '
set -e
mkdir -p /opt/sparrow-engine-cli
tar -xzf "/clean_room/$TARBALL_NAME" -C /opt/sparrow-engine-cli
bundle=$(find /opt/sparrow-engine-cli -mindepth 1 -maxdepth 1 -type d | head -n1)
"$bundle/bin/spe" --version
"$bundle/bin/spe" models list
'
}

smoke_python() {
  # $1 = image tag
  # Returns: 0 on PASS, 2 on SKIP (wheel-ABI mismatch), non-zero on FAIL.
  local tag="$1"
  local wheel
  wheel="$(find_first "$PY_WHEEL_GLOB")"
  local wheel_name
  wheel_name="$(basename "$wheel")"
  docker run --rm \
    -e WHEEL_NAME="$wheel_name" \
    -v "$wheel:/clean_room/$wheel_name:ro" \
    "$tag" \
    bash -c '
set -e
# Preflight: wheel ABI tag vs container Python (C1). Mismatch = SKIP, not FAIL.
# Note: this is an equality check on the python tag; once T-1 lands and the
# sparrow-engine wheel becomes abi3 (e.g. cp310-abi3), a Python >= floor compatibility
# check should replace the equality. For now, every dev build is cpXXX-cpXXX
# and equality is the right gate.
container_py=$(python3 -V 2>&1 | sed -nE "s/^Python ([0-9]+)\.([0-9]+).*/\1\2/p")
if [[ "$WHEEL_NAME" =~ -cp([0-9]+)- ]]; then
  wheel_py="${BASH_REMATCH[1]}"
else
  echo "[smoke] cannot parse cp tag from wheel name: $WHEEL_NAME" >&2
  exit 1
fi
if [[ "$wheel_py" != "$container_py" ]]; then
  echo "[smoke] wheel tag cp${wheel_py} vs container py${container_py} — rebuild sparrow-engine-python with abi3-py310 (T-1 ticket) to unify" >&2
  exit 2
fi
python3 -m venv /opt/venv
# ORT pin matches sparrow-engine-python/pyproject.toml Requires-Dist
# (>=1.25.1,<1.26). The CLI tarball ships its own bundled libonnxruntime
# (RP-4); this branch tests the *Python wheel* install path, where the
# RP-3 `_discover_ort_dylib()` shim auto-discovers `libonnxruntime.so.X.Y.Z`
# from the pip onnxruntime install at import time — no manual symlink
# dance required.
/opt/venv/bin/pip install "onnxruntime>=1.25.1,<1.26" "/clean_room/$WHEEL_NAME"
/opt/venv/bin/python -c "import sparrow_engine; print(\"sparrow-engine \" + sparrow_engine.__version__ if hasattr(sparrow_engine, \"__version__\") else \"sparrow-engine imported\")"
'
}

smoke_shim() {
  # $1 = image tag
  # Returns: 0 on PASS, 2 on SKIP (wheel-ABI mismatch), non-zero on FAIL.
  # Install sparrow-engine first (the shim's runtime dep), then the shim with
  # --no-deps so pip doesn't try to hit PyPI for sparrow-engine. The shim's
  # sparrow-engine==0.1.0 pin is bypassed here — see W3 in
  # docs/review/phase3.5-clean-room/manual_test_plan.md §4 + the shim README
  # for the co-bump-rule.
  local tag="$1"
  local wheel bongo_wheel
  wheel="$(find_first "$SHIM_WHEEL_GLOB")"
  bongo_wheel="$(find_first "$PY_WHEEL_GLOB")"
  local wheel_name bongo_wheel_name
  wheel_name="$(basename "$wheel")"
  bongo_wheel_name="$(basename "$bongo_wheel")"
  docker run --rm \
    -e WHEEL_NAME="$wheel_name" \
    -e SPARROW_ENGINE_WHEEL_NAME="$bongo_wheel_name" \
    -v "$wheel:/clean_room/$wheel_name:ro" \
    -v "$bongo_wheel:/clean_room/$bongo_wheel_name:ro" \
    "$tag" \
    bash -c '
set -e
# Preflight: wheel ABI tag vs container Python (C1). Mismatch = SKIP.
# We check the sparrow-engine wheel here (the shim wheel is py3-none-any so always
# compatible); if sparrow-engine is incompatible the shim cannot import sparrow_engine at runtime
# anyway. Equality check now; relax once T-1 lands abi3.
container_py=$(python3 -V 2>&1 | sed -nE "s/^Python ([0-9]+)\.([0-9]+).*/\1\2/p")
if [[ "$SPARROW_ENGINE_WHEEL_NAME" =~ -cp([0-9]+)- ]]; then
  wheel_py="${BASH_REMATCH[1]}"
else
  echo "[smoke] cannot parse cp tag from wheel name: $SPARROW_ENGINE_WHEEL_NAME" >&2
  exit 1
fi
if [[ "$wheel_py" != "$container_py" ]]; then
  echo "[smoke] wheel tag cp${wheel_py} vs container py${container_py} — rebuild sparrow-engine-python with abi3-py310 (T-1 ticket) to unify" >&2
  exit 2
fi
python3 -m venv /opt/venv
# ORT pin matches sparrow-engine-python/pyproject.toml Requires-Dist
# (>=1.25.1,<1.26). The sparrow-engine wheel import-time
# `_discover_ort_dylib()` shim sets ORT_DYLIB_PATH from the pip install,
# so no manual libonnxruntime symlink workaround is needed here.
/opt/venv/bin/pip install "onnxruntime>=1.25.1,<1.26" "/clean_room/$SPARROW_ENGINE_WHEEL_NAME"
# Keep --no-deps: this local smoke intentionally bypasses the shim package
# co-bump pin so the just-built sparrow-engine wheel is the runtime dependency.
/opt/venv/bin/pip install --no-deps "/clean_room/$WHEEL_NAME"
/opt/venv/bin/python -c "import pytorchwildlife; print(\"pytorchwildlife shim OK\")"
'
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
LOG "Repo root: $REPO_ROOT"
LOG "Matrix: os={${MATRIX_OS[*]}} artifact={${MATRIX_ARTIFACT[*]}}"
LOG "Build missing: $BUILD_MISSING"

# Prep artifacts once.
for a in "${MATRIX_ARTIFACT[@]}"; do
  case "$a" in
    cli)    ensure_cli_tarball;;
    python) ensure_python_wheel;;
    shim)   ensure_shim_wheel; ensure_python_wheel;;  # shim needs sparrow-engine
  esac
done

# Per-OS image build.
declare -A OS_TAG=()
for os in "${MATRIX_OS[@]}"; do
  OS_TAG[$os]="$(ensure_image "$os")"
done

# Run the matrix.
START_TS=$(date '+%Y-%m-%d %H:%M:%S %Z')
for os in "${MATRIX_OS[@]}"; do
  tag="${OS_TAG[$os]}"
  for a in "${MATRIX_ARTIFACT[@]}"; do
    cell="u${os}/${a}"
    LOG "--- running cell: $cell ---"
    set +e
    case "$a" in
      cli)    smoke_cli    "$tag" ;;
      python) smoke_python "$tag" ;;
      shim)   smoke_shim   "$tag" ;;
    esac
    rc=$?
    set -e
    if [[ $rc -eq 0 ]]; then
      CELL_RESULT[$cell]="PASS"
      LOG "$cell PASS"
    elif [[ $rc -eq 2 ]]; then
      # Smoke functions return rc=2 on preflight SKIP (e.g. C1 wheel-ABI
      # mismatch). Not a failure — the underlying artifact ticket is
      # tracked separately (see docs/review/phase3.5-clean-room/README.md
      # §Troubleshooting).
      CELL_RESULT[$cell]="SKIP"
      LOG "$cell SKIP (preflight guard fired; see preceding [smoke] message)"
    else
      CELL_RESULT[$cell]="FAIL"
      LOG "$cell FAIL (rc=$rc)"
    fi
  done
done
END_TS=$(date '+%Y-%m-%d %H:%M:%S %Z')

# ---------------------------------------------------------------------------
# Append results
# ---------------------------------------------------------------------------
mkdir -p "$(dirname "$RESULTS_MD")"
{
  printf '\n## Run — %s\n\n' "$START_TS"
  printf 'Host: %s\n' "$(uname -srm)"
  printf 'Docker: %s\n' "$(docker --version)"
  printf 'Started: %s\n' "$START_TS"
  printf 'Finished: %s\n\n' "$END_TS"
  printf '| OS / Artifact | Result |\n'
  printf '|---------------|--------|\n'
  for os in "${MATRIX_OS[@]}"; do
    for a in "${MATRIX_ARTIFACT[@]}"; do
      cell="u${os}/${a}"
      printf '| `%s` | %s |\n' "$cell" "${CELL_RESULT[$cell]:-SKIP}"
    done
  done
  printf '\n'
} >> "$RESULTS_MD"
LOG "Results appended to $RESULTS_MD"

# Exit non-zero if any cell failed. SKIPs (preflight-rejected) do NOT fail
# the run — the underlying artifact ticket carries that load (T-1 / T-2).
failures=0
skips=0
for v in "${CELL_RESULT[@]}"; do
  case "$v" in
    FAIL) ((failures++)) || true ;;
    SKIP) ((skips++)) || true ;;
  esac
done
if (( failures > 0 )); then
  LOG "summary: $failures cell(s) FAILED, $skips SKIPPED"
  exit 1
fi
if (( skips > 0 )); then
  LOG "summary: all non-skipped cells PASS ($skips SKIPPED — see preceding [smoke] messages and README §Troubleshooting)"
else
  LOG "summary: all cells PASS"
fi
