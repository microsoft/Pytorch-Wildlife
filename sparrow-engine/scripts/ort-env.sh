#!/usr/bin/env bash
# DEVELOPMENT ONLY — not needed by end users.
#
# This script exists because our dev machine (Ubuntu 22.04, glibc 2.35) can't
# use ORT's static lib (needs glibc 2.38+). We link dynamically against the
# pip onnxruntime-gpu package instead. This script finds that package and sets
# the linker paths cargo needs.
#
# End users don't need this:
#   - Docker: ORT is bundled in the container image
#   - CLI release tarball (RP-4 2026-05-26): bundles `libonnxruntime.so.X.Y.Z`
#     under `lib/` and resolves it via the in-binary
#     `ort_resolver::init_ort_env()` shim; no `LD_LIBRARY_PATH` shell setup
#     required. See `installer/sparrow-engine-install.sh --cli`.
#   - Python wheel (RP-3 2026-05-23): the `_discover_ort_dylib()` shim in
#     `sparrow_engine.__init__` sets `ORT_DYLIB_PATH` from the pip
#     `onnxruntime[-gpu]` install at import time.
#
# GPU is the default. Prefers onnxruntime-gpu over onnxruntime-cpu.
# Sets: ORT_CAPI, ORT_LIB_LOCATION, ORT_PREFER_DYNAMIC_LINK, LD_LIBRARY_PATH.
#
# Usage:
#   source scripts/ort-env.sh                          # from sparrow-engine/
#   source "$(dirname "$0")/../scripts/ort-env.sh"     # from tools/
#   source "$(dirname "$0")/ort-env.sh"                # from scripts/

# NOTE: No `set -euo pipefail` here — this file is sourced into the caller's
# shell. Setting -u would break zsh plugins (e.g., zvm_update_cursor) that
# reference unset variables. Callers (test.sh, etc.) set their own options.

# Find ORT shared library from pip onnxruntime package.
# Prefers GPU (onnxruntime-gpu) over CPU (onnxruntime).
#
# Multiple onnxruntime archives can coexist in the uv cache (e.g. a project
# bumps from 1.24.4 → 1.25.0 → 1.25.1 over time). `find -print -quit` returns
# the first match in filesystem-traversal order, which is non-deterministic
# across calls. That makes builds and runs pick different ORT versions and
# produces "version `VERS_1.X.Y' not found" at runtime. Pick the newest
# version deterministically by reading the ELF symbol-version table.
#
# Drop `archive-v0` segment so this survives a future uv cache-format bump
# (uv has different vN per subdir already: archive-v0, environments-v2,
# interpreter-v4, sdists-v9, simple-v18 on this dev box). The `-path "$1"`
# filter (e.g. `*/onnxruntime/capi/libonnxruntime_providers_cuda.so`) is
# specific enough to identify ORT capi dirs regardless of which uv subdir
# layout houses them. Mirrors the analogous fix in pick_newest_cudnn_dir
# below + setup.sh wheel-cache cleanup. Without this fix, a uv cache
# format bump silently degrades us to the python3 fallback at lines 92-102,
# losing the version-deterministic ELF-symbol read this function exists for.
pick_newest_ort_dir() {
    # Soft-fail if `strings` (binutils) is missing — without it we cannot
    # extract VERS_ symbols, every archive becomes a silent skip, and the
    # caller falls through to python3 with no diagnostic. Warn once.
    if ! command -v strings >/dev/null 2>&1; then
        echo >&2 "warn: 'strings' (binutils) not found — ORT version detection disabled; install with 'apt-get install binutils' or equivalent"
        return
    fi
    # NOTE: `find … | while … done` runs the while body in a subshell under
    # bash (per Pipelines manual §3.2.3) but in the parent shell under zsh.
    # This loop body only `printf`s to stdout (consumed by the trailing
    # sort|head|awk), so the divergence is invisible. DO NOT add cross-
    # iteration state (counters, "last seen" vars) — those would silently
    # be zero/empty under bash. See bash manual + BashFAQ #024.
    find "$HOME/.cache/uv" -path "$1" -print 2>/dev/null |
        while IFS= read -r p; do
            d=$(dirname "$p")
            # Use `find` not shell glob to stay safe under zsh's NO_MATCH
            # default. `sort -V` defensive in case a future ORT release ships
            # multiple .so files in one archive (debug variant, etc.) — pick
            # the highest.
            real=$(find "$d" -maxdepth 1 -name 'libonnxruntime.so.*.*.*' -type f 2>/dev/null | sort -V | tail -1)
            [[ -z "$real" ]] && continue
            v=$(strings "$real" 2>/dev/null |
                grep -E '^VERS_1\.[0-9]+\.[0-9]+' |
                sort -V | tail -1)
            if [[ -n "$v" ]]; then
                printf '%s %s\n' "$v" "$d"
            else
                # Custom/community ORT build without --version-script. Microsoft
                # pip wheels always export VERS_, so this is rare — but warn
                # rather than silently exclude the archive from selection.
                echo >&2 "warn: $real has no VERS_ symbols (custom ORT build?); excluding from version selection"
            fi
        done |
        sort -V -r | head -1 | awk '{print $2}'
}

ort_dir_has_runtime_lib() {
    local dir="$1"
    local versioned
    if [[ -f "$dir/libonnxruntime.so" ]]; then
        return 0
    fi
    versioned=$(find "$dir" -maxdepth 1 -name 'libonnxruntime.so.*.*.*' -type f -print -quit 2>/dev/null)
    [[ -n "$versioned" ]]
}

find_ort_dir() {
    # Check explicit override first.
    if [[ -n "${ORT_DIR:-}" ]]; then
        if [[ ! -d "$ORT_DIR" ]]; then
            echo >&2 "error: ORT_DIR is not a directory: $ORT_DIR"
            echo >&2 "Check ORT_DIR points to an onnxruntime/capi directory."
            echo >&2 "Unset ORT_DIR to fall back to auto-discovery."
            return 1
        fi
        if ! ort_dir_has_runtime_lib "$ORT_DIR"; then
            echo >&2 "error: ORT_DIR does not contain libonnxruntime.so: $ORT_DIR"
            echo >&2 "Check ORT_DIR points to an onnxruntime/capi directory."
            echo >&2 "Unset ORT_DIR to fall back to auto-discovery."
            return 1
        fi
        echo "$ORT_DIR"
        return
    fi

    # Search uv cache for onnxruntime-gpu first (has CUDA provider .so).
    local gpu_candidate
    gpu_candidate=$(pick_newest_ort_dir "*/onnxruntime/capi/libonnxruntime_providers_cuda.so")

    if [[ -n "$gpu_candidate" ]]; then
        echo "$gpu_candidate"
        return
    fi

    # Fallback: any onnxruntime capi directory (CPU).
    local candidate
    candidate=$(pick_newest_ort_dir "*/onnxruntime/capi/libonnxruntime.so")

    if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return
    fi

    # Last resort: ask Python where onnxruntime lives.
    local pyort
    pyort=$(python3 -c "
import onnxruntime, pathlib
print(pathlib.Path(onnxruntime.__file__).parent / 'capi')
" 2>/dev/null || true)

    if [[ -n "$pyort" && -f "$pyort/libonnxruntime.so" ]]; then
        echo "$pyort"
        return
    fi

    # Caller signals total-failure via empty stdout. We do NOT call `exit 1`
    # here: this script is sourced (see header), and `exit` from a sourced
    # function terminates the caller's interactive shell — verified in bash
    # and zsh. agent-health rule 10 calls this out explicitly: a sourced
    # `exit` from the user's tmux pane can kill the entire tmux server when
    # that pane is the last one. Print the diagnostic, return empty, and let
    # the caller `return 1` from the sourced script.
    echo >&2 "error: cannot find ORT shared library."
    echo >&2 "Install onnxruntime-gpu: uv pip install onnxruntime-gpu"
    echo >&2 "Or set ORT_DIR=/path/to/onnxruntime/capi"
    return 1
}

ORT_CAPI=$(find_ort_dir)
# Empty ORT_CAPI means find_ort_dir hit the diagnostic path. Return from the
# sourced script — never `exit` from a sourced context.
[[ -z "$ORT_CAPI" ]] && return 1

# Ensure symlinks exist and point at the currently-selected versioned .so.
# `find` not shell glob to stay safe under zsh's NO_MATCH default when ORT_CAPI is empty.
# `sort -V | tail -1` for defensive coding (currently one .so per archive,
# but a future ORT debug variant would expose head -1's non-determinism).
#
# `[[ ! -e "$so1" ]] && ln -sf` would skip when the symlink already exists,
# even if it points to the WRONG (older) versioned .so — the same bug class
# d23861d closed on the picker side. Compare via `readlink` and refresh
# whenever the link target diverges. Self-healing on stale, idempotent on
# correct, no-op on dangling (then -z current → refresh).
if [[ -n "$ORT_CAPI" ]]; then
    real=$(find "$ORT_CAPI" -maxdepth 1 -name 'libonnxruntime.so.*.*.*' -type f 2>/dev/null | sort -V | tail -1)
    if [[ -n "$real" ]]; then
        expected=$(basename "$real")
        for alias_name in libonnxruntime.so.1 libonnxruntime.so; do
            current=$(readlink "$ORT_CAPI/$alias_name" 2>/dev/null || true)
            [[ "$current" != "$expected" ]] && ln -sf "$expected" "$ORT_CAPI/$alias_name"
        done
    fi
fi

export ORT_LIB_LOCATION="$ORT_CAPI"
export ORT_PREFER_DYNAMIC_LINK=1

# For GPU: ORT CUDA EP needs CUDA runtime (libcudart) and cuDNN (libcudnn) at runtime.
# Auto-discover common locations.
# Exported so aliases like `sparrow-engine-gpu` can read $EXTRA_LIB_PATHS directly.
export EXTRA_LIB_PATHS=""

# RP-24 dev support: ORT's TensorRT EP dlopens TensorRT 10 runtime libraries
# (libnvinfer, libnvinfer_plugin, libnvonnxparser). Production Docker images
# install those libs via apt; local GPU tests usually get them from the
# `tensorrt-cu12` pip package's sibling `tensorrt_libs` directory.
find_tensorrt_libs_dir() {
    if [[ -n "${TENSORRT_LIBS_DIR:-}" ]]; then
        if [[ -f "$TENSORRT_LIBS_DIR/libnvinfer.so.10" ]]; then
            echo "$TENSORRT_LIBS_DIR"
            return 0
        fi
        echo >&2 "warn: TENSORRT_LIBS_DIR does not contain libnvinfer.so.10: $TENSORRT_LIBS_DIR"
        return 1
    fi

    local candidate
    candidate=$(
        for search_root in \
            "${VIRTUAL_ENV:-}" \
            "$HOME/.local/lib" \
            "$HOME/.venvs" \
            "$HOME/.cache/uv"; do
            [[ -z "$search_root" || ! -d "$search_root" ]] && continue
            find "$search_root" -path '*/site-packages/tensorrt_libs/libnvinfer.so.10' -type f 2>/dev/null
        done |
            while IFS= read -r p; do
                lib_dir=$(dirname "$p")
                site_dir=$(dirname "$lib_dir")
                dist_info=$(find "$site_dir" -maxdepth 1 -name 'tensorrt_cu12_libs-*.dist-info' -type d 2>/dev/null | sort -V | tail -1)
                if [[ -n "$dist_info" ]]; then
                    version=$(basename "$dist_info" | sed -E 's/^tensorrt_cu12_libs-([0-9][^/]+)\.dist-info$/\1/')
                else
                    version=0
                fi
                printf '%s %s\n' "$version" "$lib_dir"
            done |
            sort -V -r | head -1 | awk '{print $2}'
    )
    if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi

    for search_root in \
        "${VIRTUAL_ENV:-}" \
        "$HOME/.local/lib" \
        "$HOME/.venvs" \
        "$HOME/.cache/uv"; do
        [[ -z "$search_root" || ! -d "$search_root" ]] && continue
        candidate=$(find "$search_root" -path '*/site-packages/tensorrt_libs/libnvinfer.so.10' -type f 2>/dev/null | sort -V | tail -1)
        if [[ -n "$candidate" ]]; then
            dirname "$candidate"
            return 0
        fi
    done

    candidate=$(python3 -c "
import pathlib
import site
import sys

roots = []
for getter in (getattr(site, 'getusersitepackages', None),):
    if getter is not None:
        try:
            roots.append(getter())
        except Exception:
            pass
try:
    roots.extend(site.getsitepackages())
except Exception:
    pass
roots.extend(sys.path)

seen = set()
for root in roots:
    if not root or root in seen:
        continue
    seen.add(root)
    p = pathlib.Path(root) / 'tensorrt_libs' / 'libnvinfer.so.10'
    if p.is_file():
        print(p.parent)
        raise SystemExit(0)
" 2>/dev/null || true)
    if [[ -n "$candidate" ]]; then
        echo "$candidate"
        return 0
    fi

    return 1
}

tensorrt_libs_dir=$(find_tensorrt_libs_dir || true)
if [[ -n "$tensorrt_libs_dir" ]]; then
    EXTRA_LIB_PATHS="${tensorrt_libs_dir}:${EXTRA_LIB_PATHS}"
    echo "TensorRT: $tensorrt_libs_dir"
else
    echo >&2 "warn: TensorRT libs not found; install tensorrt-cu12 or set TENSORRT_LIBS_DIR for TRT EP dev tests"
fi

# cuDNN: we require 9.10+ for SpeciesNet on sm_89 (cuDNN 9.8 has a Conv engine
# bug with asymmetric padding — "No valid engine configs for ConvFwd_").
# PyTorch/TF bundle 9.8, so we prefer a standalone nvidia-cudnn-cu12>=9.10 if
# installed to ~/.local/cudnn or ~/.cache/uv.
# Install: uv pip install --target ~/.local/cudnn 'nvidia-cudnn-cu12>=9.10'
#
# Pick the newest cuDNN dir from the uv cache deterministically. This mirrors
# `pick_newest_ort_dir` (commit d23861d) — `find -print -quit` returned the
# first FS-traversal hit, so build vs. runtime could resolve to different
# versions. Also enforces the documented 9.10+ floor: sub-9.10 candidates are
# filtered out before the version-sort. (Drop `archive-v0` segment so this
# survives a future uv cache-format bump — see analogous fix in setup.sh.)
pick_newest_cudnn_dir() {
    # NOTE: same bash-subshell-vs-zsh-parent-shell pattern as pick_newest_ort_dir.
    # Body only `printf`s; if you add state, hoist it out of the pipeline
    # (use `< <(find …)` process-substitution instead). See the NOTE above
    # pick_newest_ort_dir's find|while pipeline.
    find "$HOME/.cache/uv" -path '*/nvidia/cudnn/lib/libcudnn.so.9.*.*.*' -type f 2>/dev/null |
        while IFS= read -r p; do
            d=$(dirname "$p")
            # Filename is libcudnn.so.MAJOR.MINOR.PATCH.PATCH2
            v=$(basename "$p" | sed 's/^libcudnn\.so\.//')
            # Enforce 9.10+ floor: skip 9.0..9.9 (PyTorch/TF bundles 9.8 with
            # the buggy asymmetric-padding Conv engine).
            major=$(echo "$v" | cut -d. -f1)
            minor=$(echo "$v" | cut -d. -f2)
            [[ "$major" -lt 9 ]] && continue
            [[ "$major" -eq 9 && "$minor" -lt 10 ]] && continue
            printf '%s %s\n' "$v" "$d"
        done |
        sort -V -r | head -1 | awk '{print $2}'
}

cudnn_cache_dir=$(pick_newest_cudnn_dir)

for cudnn_dir in \
    "$HOME/.local/cudnn/nvidia/cudnn/lib" \
    "$cudnn_cache_dir" \
    /usr/lib/python3/dist-packages/torch/lib \
    /usr/local/cuda/lib64 \
    /usr/lib/x86_64-linux-gnu; do
    if [[ -n "$cudnn_dir" && ( -f "$cudnn_dir/libcudnn.so" || -f "$cudnn_dir/libcudnn.so.9" ) ]]; then
        EXTRA_LIB_PATHS="${cudnn_dir}:${EXTRA_LIB_PATHS}"
        echo "cuDNN: $cudnn_dir"
        break
    fi
done

# CUDA runtime: usually in system lib path or /usr/local/cuda.
for cuda_dir in \
    /usr/lib/x86_64-linux-gnu \
    /usr/local/cuda/lib64; do
    if [[ -f "$cuda_dir/libcudart.so" ]] || [[ -f "$cuda_dir/libcudart.so.12" ]]; then
        EXTRA_LIB_PATHS="${cuda_dir}:${EXTRA_LIB_PATHS}"
        break
    fi
done

# Build LD_LIBRARY_PATH idempotently — re-sourcing must not multiply entries.
# Empirically, the prior `${ORT_CAPI}:${EXTRA_LIB_PATHS}${LD_LIBRARY_PATH:-}`
# pattern grew the path linearly: 138 → 276 → 414 chars over 3 sources, with
# every component triplicated. The dynamic linker dedupes at lookup so this
# was never functional, but `echo $LD_LIBRARY_PATH` for diagnostics became
# unreadable. Concat the new prefix (ORT_CAPI + EXTRA_LIB_PATHS + prior
# LD_LIBRARY_PATH), then de-duplicate via awk's first-seen idiom — preserves
# left-most occurrence (so newly-resolved ORT/cuDNN paths win priority over
# stale entries left over from a prior source). Portable across bash + zsh.
_combined_libpath="${ORT_CAPI}:${EXTRA_LIB_PATHS%:}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
LD_LIBRARY_PATH=$(printf '%s' "$_combined_libpath" | tr ':' '\n' | awk 'NF && !seen[$0]++' | paste -sd:)
unset _combined_libpath
export LD_LIBRARY_PATH

echo "ORT: $ORT_CAPI"
