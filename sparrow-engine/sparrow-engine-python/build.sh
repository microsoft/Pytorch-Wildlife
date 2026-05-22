#!/usr/bin/env bash
# Phase 3.8 Phase C Wave 4a (2026-05-06): build the sparrow-engine / sparrow-engine-gpu
# Python wheels from this single source tree.
#
# Usage:
#   SPARROW_ENGINE_FLAVOR=cpu  ./build.sh   # sparrow-engine wheel (default; pulls onnxruntime)
#   SPARROW_ENGINE_FLAVOR=gpu  ./build.sh   # sparrow-engine-gpu wheel (pulls onnxruntime-gpu)
#   SPARROW_ENGINE_FLAVOR=both ./build.sh   # both wheels (default if SPARROW_ENGINE_FLAVOR is unset)
#
# Output: ../target/wheels/sparrow_engine-*.whl  +/-  ../target/wheels/sparrow_engine_gpu-*.whl
#
# The CPU wheel uses this directory's `pyproject.toml` as-is. The GPU
# wheel sed-patches a temp copy (rename `sparrow-engine` -> `sparrow-engine-gpu`,
# `onnxruntime` -> `onnxruntime-gpu`, `extension-module,cpu` ->
# `extension-module,gpu`) and post-build patches `Provides-Dist: sparrow-engine`
# into the wheel's METADATA so pip refuses both wheels in the same env.
#
# References: `docs/design/phase3.8/phase_c/implementation_plan.md`
# §2.3 + §4 W4a + §9 item 4.

set -euo pipefail
cd "$(dirname "$0")"

: "${SPARROW_ENGINE_FLAVOR:=both}"

# Use uv-managed maturin if available; falls back to PATH lookup.
MATURIN="${MATURIN:-maturin}"

build_cpu() {
    echo "[build.sh] Building CPU wheel (sparrow-engine, onnxruntime)..."
    # `--auditwheel skip`: do NOT bundle libonnxruntime.so.1 into the
    # wheel — the runtime dep on `onnxruntime>=1.25.1` (pip) provides
    # it. Per `docs/design/phase3.8/phase_c/implementation_plan.md`
    # §2.3 + §4 W4a wheel-size discipline.
    "$MATURIN" build --release \
        --auditwheel skip \
        --no-default-features \
        --features extension-module \
        --features cpu
    echo "[build.sh] CPU wheel built."
}

build_gpu() {
    echo "[build.sh] Building GPU wheel (sparrow-engine-gpu, onnxruntime-gpu)..."

    # Single combined cleanup trap (Phase C audit-fix R1 I-6). Bash traps
    # don't stack — a second `trap ... EXIT` REPLACES the first — so the
    # earlier two-trap install→clear→install→clear pattern was fragile
    # against future code added between the install/clear pairs. One
    # trap with state-aware cleanup keeps both pyproject restore AND
    # tmpdir removal covered for the entire function lifetime regardless
    # of where a failure occurs.
    local backup="" tmpdir=""
    cleanup_gpu() {
        local rc=$?
        if [[ -n "$backup" && -f "$backup" ]]; then
            mv "$backup" pyproject.toml || true
        fi
        if [[ -n "$tmpdir" && -d "$tmpdir" ]]; then
            trash-put "$tmpdir" 2>/dev/null || rm -rf "$tmpdir"
        fi
        return $rc
    }
    trap cleanup_gpu EXIT

    # Snapshot pyproject.toml so the sed patch below does not corrupt the
    # source-tree CPU template on failure.
    backup="$(mktemp pyproject.toml.bak.XXXXXX)"
    cp pyproject.toml "$backup"

    # Phase C audit-fix R1 (A4 / 2026-05-06): the
    # `features = ["extension-module", "cpu"]` -> `[..., "gpu"]` sed line
    # was dropped. Maturin's `--no-default-features --features
    # extension-module --features gpu` (below) overrides the
    # `[tool.maturin] features = [...]` block, so the substitution was
    # functionally inert and a future-fragility hazard.
    sed -i \
        -e 's|^name = "sparrow-engine"$|name = "sparrow-engine-gpu"|' \
        -e 's|^description = "Camera-trap ML inference engine — Python bindings (sparrow-engine CPU pipeline)"$|description = "Camera-trap ML inference engine — Python bindings (sparrow-engine GPU pipeline)"|' \
        -e 's|"onnxruntime>=1.25.1,<1.26"|"onnxruntime-gpu>=1.25.1,<1.26"|' \
        pyproject.toml

    # Build the GPU wheel. `--auditwheel skip` keeps libonnxruntime +
    # libnvjpeg out of the wheel — they ship via the pip
    # `onnxruntime-gpu>=1.25.1` runtime dep + the system CUDA toolkit
    # (libnvjpeg.so.12 from `nvidia-cuda-runtime-cu12`).
    "$MATURIN" build --release \
        --auditwheel skip \
        --no-default-features \
        --features extension-module \
        --features gpu

    # Restore the source-tree pyproject.toml NOW (before the post-build
    # steps below). Clear the backup tracking so the cleanup_gpu trap
    # does not try to re-restore on EXIT.
    mv "$backup" pyproject.toml
    backup=""

    # Post-build: patch `Provides-Dist: sparrow-engine` into the GPU wheel's
    # METADATA so pip refuses to install both sparrow-engine and sparrow-engine-gpu into
    # the same env (Acceptance Gate G3 conflict-test).
    #
    # Phase C audit-fix R1 (A3 / 2026-05-06): pipe assignment uses
    # `|| true` so `set -o pipefail` does not abort the script when no
    # GPU wheel is found — the friendly `[[ -z "$wheel" ]]` diagnostic
    # below would otherwise be unreachable.
    local wheel
    wheel="$(ls -t ../target/wheels/sparrow_engine_gpu-*.whl 2>/dev/null | head -1 || true)"
    if [[ -z "$wheel" ]]; then
        echo "[build.sh] ERROR: could not locate sparrow_engine_gpu-*.whl in ../target/wheels/ (did maturin succeed?)" >&2
        exit 1
    fi

    tmpdir="$(mktemp -d)"
    # Use uv-managed temp env with the `wheel` package — the `wheel pack`
    # step regenerates the RECORD's SHA256 hashes after the METADATA
    # patch (a raw zip rewrite would leave a stale RECORD and cause pip
    # to refuse the wheel as corrupt).
    uv run --no-project --with wheel python -m wheel unpack "$wheel" -d "$tmpdir"

    # The unpacked layout is $tmpdir/sparrow_engine_gpu-<version>/...; the dist-info
    # dir is at $tmpdir/sparrow_engine_gpu-<version>/sparrow_engine_gpu-<version>.dist-info/
    local meta
    meta="$(find "$tmpdir" -name METADATA -path '*sparrow_engine_gpu-*.dist-info/METADATA' | head -1)"
    if [[ -z "$meta" ]]; then
        echo "[build.sh] ERROR: METADATA not found inside unpacked GPU wheel" >&2
        exit 1
    fi
    if grep -q '^Provides-Dist: sparrow-engine$' "$meta"; then
        echo "[build.sh] (Provides-Dist: sparrow-engine already present; skipping patch.)"
    else
        sed -i '/^Name: sparrow-engine-gpu$/a Provides-Dist: sparrow-engine' "$meta"
    fi

    # Repack and replace the original GPU wheel with the patched one.
    local unpacked
    unpacked="$(find "$tmpdir" -maxdepth 1 -mindepth 1 -type d -name 'sparrow_engine_gpu-*' | head -1)"
    if [[ -z "$unpacked" ]]; then
        echo "[build.sh] ERROR: unpacked sparrow_engine_gpu-* dir not found" >&2
        exit 1
    fi
    uv run --no-project --with wheel python -m wheel pack "$unpacked" -d ../target/wheels/

    # Clean tmpdir on the success path. The cleanup_gpu trap still runs
    # on EXIT but skips an already-clean tmpdir.
    trash-put "$tmpdir" 2>/dev/null || rm -rf "$tmpdir"
    tmpdir=""

    trap - EXIT
    cleanup_gpu
    echo "[build.sh] GPU wheel built and Provides-Dist patched."
}

case "$SPARROW_ENGINE_FLAVOR" in
    cpu)  build_cpu ;;
    gpu)  build_gpu ;;
    both) build_cpu; build_gpu ;;
    *)
        echo "[build.sh] ERROR: SPARROW_ENGINE_FLAVOR must be cpu / gpu / both (got '$SPARROW_ENGINE_FLAVOR')" >&2
        exit 1
        ;;
esac

ls -lh ../target/wheels/ 2>/dev/null || true
