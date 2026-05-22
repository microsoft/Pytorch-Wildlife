#!/usr/bin/env bash
# Forbid `eprintln!` / `println!` in sparrow-engine-python/src/lib.rs.
#
# Background: PyO3 issue #2247 — stdio writes from Rust are invisible inside
# Jupyter kernels. The Python package routes all diagnostics through the
# `tracing` crate bridged to Python logging.
#
# Status: Phase 3.5 S6 landed; every former `eprintln!`/`println!` site in
# sparrow-engine-python/src/lib.rs is now `tracing::warn!` / `tracing::info!`. This
# guard is load-bearing — a regression (new `eprintln!`/`println!`) fails
# the script. The historical 11-site backlog is closed; see:
#   - docs/design/phase3.5/final_design.md §4 S2
#   - docs/design/phase3.5/final_design.md §4 S6
#   - docs/review/phase3.5/cfg-gated-tests.md
#
# Opt-out: set `SPARROW_ENGINE_PY_GUARD_STRICT=0` to demote a violation back to a
# warning (only useful for transient debugging — do not set in CI).
#
# Usage:
#   ./scripts/guard_no_print.sh                          # enforcing (default)
#   SPARROW_ENGINE_PY_GUARD_STRICT=0 ./scripts/guard_no_print.sh  # advisory (debug)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
TARGET="${1:-$SCRIPT_DIR/../sparrow-engine-python/src/lib.rs}"

if [[ ! -f "$TARGET" ]]; then
    echo "guard_no_print: target not found: $TARGET" >&2
    exit 0
fi

HITS=$(grep -nE 'eprintln!|println!' "$TARGET" || true)

if [[ -z "$HITS" ]]; then
    echo "guard_no_print: OK (zero eprintln!/println! in $TARGET)"
    exit 0
fi

echo "guard_no_print: found disallowed print calls in $TARGET:"
echo "$HITS"

if [[ "${SPARROW_ENGINE_PY_GUARD_STRICT:-1}" == "0" ]]; then
    echo "guard_no_print: ADVISORY — SPARROW_ENGINE_PY_GUARD_STRICT=0; route all diagnostics through tracing (see header)."
    exit 0
fi

echo "guard_no_print: FAIL — route all diagnostics through tracing (see header)."
exit 1
