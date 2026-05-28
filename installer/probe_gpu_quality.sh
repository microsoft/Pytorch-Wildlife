#!/usr/bin/env bash
# installer/probe_gpu_quality.sh — sourceable layer-2 quality probe
#                                  (GPU runtime sidecar DLL surface +
#                                  cuDNN ≥9.10 floor + driver-version sanity).
#
# Purpose
#   Layer-2 of the Sparrow Engine install-time selector. Runs ONLY after
#   layer-1 (`probe.sh`) has returned `gpu`. The basic CUDA probe answers
#   "is CUDA reachable?" — this layer answers three follow-up questions:
#     1. Are all hard-required GPU runtime sidecar libraries reachable?
#        ORT 1.25.1's `libonnxruntime_providers_cuda.so` DT_NEEDED list
#        (verified via `readelf -d` on the published wheel, 2026-05-27)
#        names six CUDA libs: `libcudart.so.12`, `libcublas.so.12`,
#        `libcublasLt.so.12`, `libcurand.so.10`, `libcufft.so.11`,
#        `libcudnn.so.9`. The sparrow-engine GPU image-decode path additionally
#        loads `libnvjpeg.so.12` via dlopen — missing nvJPEG fails at first
#        inference with `SparrowEngineError::NvjpegUnavailable`
#        (sparrow-engine-gpu/src/models/{yolo,tiled,classifier}.rs).
#     2. Is cuDNN ≥9.10 reachable? (canonical project floor — cuDNN 9.8 has
#        the asymmetric-padding ConvFwd engine bug that breaks SpeciesNet on
#        sm_89; sources cited in `probe_cudnn_check` below).
#     3. Is the GPU compute-capability ≥sm_80? (FP16 production cells need
#        Ampere Tensor Cores; T4 and earlier silently fall back to FP32 at
#        2-3× the latency of advertised perf).
#
# Usage
#   Sourceable form (preferred — wrapper integration):
#       . installer/probe_gpu_quality.sh
#       probe_gpu_quality
#       case "$SPARROW_ENGINE_GPU_QUALITY" in
#           ok)          : ;;                                # silent install
#           sm_warn)     warn "FP16 perf will be degraded" ;;
#           cudnn_warn)  warn "SpeciesNet will fail until cuDNN ≥9.10" ;;
#           cudnn_err)   die 11 "GPU runtime sidecar missing — block install" ;;
#       esac
#
#   The `cudnn_err` verdict is the hard-fail bucket for ANY missing GPU
#   runtime prerequisite (cuDNN, cuBLAS, cuFFT, nvJPEG, etc.); the verdict
#   name is preserved for back-compat with `sparrow-engine-install.sh`'s
#   case-handling. The verdict REASON string carries which specific library
#   is the actual culprit.
#
#   Direct invocation:
#       bash installer/probe_gpu_quality.sh    # stdout = quality verdict
#
# Env vars set
#   SPARROW_ENGINE_GPU_QUALITY         ok | sm_warn | cudnn_warn | cudnn_err
#   SPARROW_ENGINE_GPU_QUALITY_REASON  short string explaining the verdict
#
# Exit codes
#   This script always returns 0. The wrapper translates `cudnn_err` into
#   exit 11 per `final_design.md § 2.10`.
#
# Design source
#   docs/design/phase4.1-install-selector/final_design.md § 2.4
#   docs/design/phase4.1-install-selector/round_02/scripts-architect_proposal.md § 1.2.1
#   docs/design/phase4.1-install-selector/round_01/scripts-architect_proposal.md § 1.2.1 (canonical pseudocode)
#
# DLL surface ground truth (2026-05-27, RP-20):
#   ORT 1.25.1 CUDA provider DT_NEEDED CUDA libs (readelf -d
#   libonnxruntime_providers_cuda.so):
#     libcublasLt.so.12, libcublas.so.12, libcurand.so.10, libcufft.so.11,
#     libcudart.so.12, libcudnn.so.9
#   Engine-side nvJPEG (dlopen at first JpegDecoder::new):
#     libnvjpeg.so.12
#
# cuDNN ≥9.10 floor citation (verified 2026-05-08):
#   - sparrow-engine/scripts/ort-env.sh:167-168 — "cuDNN: we require 9.10+ for SpeciesNet
#     on sm_89 (cuDNN 9.8 has a Conv engine bug with asymmetric padding —
#     'No valid engine configs for ConvFwd_'). PyTorch/TF bundle 9.8."
#   - docs/lessons.md:29 — same lesson recorded against Phase 3.5 manual test.
#   - docs/tech_report/06_gotchas_and_constraints.md:17-25 — public technical
#     report entry on the bug.
#
# This script must NEVER `exit` from a sourced context — caller may have
# sourced it. Use `return` from inside the function.

# ---------------------------------------------------------------------------
# Helper: search for a CUDA runtime sidecar library by exact basename.
#
# Looks in the canonical Linux locations sparrow-engine-gpu's wrapper
# script (installer/homebrew/sparrow-engine-gpu.rb caveats) advertises,
# plus the user-set $LD_LIBRARY_PATH. Returns the first matching absolute
# path on stdout, or empty string if not found.
#
# Search order (mirrors the brew GPU wrapper's auto-discovery so the
# probe's "yes / no" verdict agrees with what `spe-gpu` will actually
# resolve at runtime):
#   1. Each entry of $LD_LIBRARY_PATH (colon-separated).
#   2. ~/.sparrow-engine/cuda-sidecars/lib/python*/site-packages/nvidia/<pkg>/lib
#      (Python sidecar venv pattern from user-manual.md §2.5 Option B).
#   3. /usr/lib/python3/dist-packages/torch/lib (Lambda Stack / PyTorch-bundled).
#   4. /usr/lib/python3/dist-packages/tensorflow (Lambda Stack / TF-bundled).
#   5. /usr/local/cuda/lib64 (NVIDIA CUDA toolkit / system install).
#   6. /usr/lib/x86_64-linux-gnu (Ubuntu apt nvidia-cudnn / system).
#   7. /usr/lib64 + /usr/lib (RHEL / fallback).
# ---------------------------------------------------------------------------
_search_required_lib() {
    _basename="$1"
    # 1. LD_LIBRARY_PATH (colon-split). Quoting `$LD_LIBRARY_PATH:` and using
    #    parameter expansion avoids `set -u` blowups when the var is unset.
    _OLDIFS="$IFS"
    IFS=':'
    for _d in ${LD_LIBRARY_PATH:-}; do
        IFS="$_OLDIFS"
        if [ -n "$_d" ] && [ -e "$_d/$_basename" ]; then
            printf '%s\n' "$_d/$_basename"
            return 0
        fi
        IFS=':'
    done
    IFS="$_OLDIFS"
    # 2-7. Deterministic dir list. Glob expansion handles the Python sidecar
    #      venv variants. `find -maxdepth 4` bounds the cost.
    for _glob in \
        "$HOME/.sparrow-engine/cuda-sidecars"/lib/python*/site-packages/nvidia/*/lib \
        /usr/lib/python3/dist-packages/torch/lib \
        /usr/lib/python3/dist-packages/tensorflow \
        /usr/local/cuda/lib64 \
        /usr/lib/x86_64-linux-gnu \
        /usr/lib64 \
        /usr/lib; do
        # Word-splitting on the glob is intentional — wildcard expansion
        # produces multiple dirs.
        for _d in $_glob; do
            [ -d "$_d" ] || continue
            if [ -e "$_d/$_basename" ]; then
                printf '%s\n' "$_d/$_basename"
                return 0
            fi
        done
    done
    return 1
}

probe_gpu_quality() {
    SPARROW_ENGINE_GPU_QUALITY=""
    SPARROW_ENGINE_GPU_QUALITY_REASON=""

    # 0. Hard-required GPU runtime sidecar libraries (RP-20). Each missing
    #    library prevents `spe-gpu` from running. Collect ALL missing names
    #    into a list so the user fixes them in one cycle rather than
    #    one-per-rerun.
    _required_libs="
        libcudart.so.12
        libcublas.so.12
        libcublasLt.so.12
        libcurand.so.10
        libcufft.so.11
        libnvjpeg.so.12
    "
    _missing_libs=""
    for _lib in $_required_libs; do
        if ! _search_required_lib "$_lib" >/dev/null; then
            if [ -z "$_missing_libs" ]; then
                _missing_libs="$_lib"
            else
                _missing_libs="$_missing_libs $_lib"
            fi
        fi
    done

    if [ -n "$_missing_libs" ]; then
        # Build install-hint with the matching pip-sidecar package names.
        # Mapping (basename → wheel) is determined from the canonical
        # NVIDIA cu12 wheel naming convention.
        _hint=""
        for _miss in $_missing_libs; do
            case "$_miss" in
                libcudart.so.12)   _pkg="nvidia-cuda-runtime-cu12" ;;
                libcublas.so.12)   _pkg="nvidia-cublas-cu12" ;;
                libcublasLt.so.12) _pkg="nvidia-cublas-cu12" ;;
                libcurand.so.10)   _pkg="nvidia-curand-cu12" ;;
                libcufft.so.11)    _pkg="nvidia-cufft-cu12" ;;
                libnvjpeg.so.12)   _pkg="nvidia-nvjpeg-cu12" ;;
                *)                 _pkg="(unknown)" ;;
            esac
            _hint="${_hint}    - ${_miss}  (pip pkg: ${_pkg})
"
        done
        SPARROW_ENGINE_GPU_QUALITY="cudnn_err"
        SPARROW_ENGINE_GPU_QUALITY_REASON="GPU runtime sidecar(s) missing — \`spe-gpu\` will fail at first inference with a libdl 'cannot open shared object file' error.
$_hint
Install via ONE of (see docs/user-manual.md §2.5):
  Option A — system CUDA: sudo apt install nvidia-cuda-toolkit nvidia-cudnn libnvjpeg
  Option B — Python sidecar wheels (no root): uv pip install --target ~/.sparrow-engine/cuda-sidecars nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-curand-cu12 nvidia-cufft-cu12 nvidia-nvjpeg-cu12 nvidia-cuda-runtime-cu12"
        export SPARROW_ENGINE_GPU_QUALITY SPARROW_ENGINE_GPU_QUALITY_REASON
        printf '%s\n' "fail"
        return 0
    fi

    # 1. cuDNN check — search the engine-canonical paths in priority order.
    #    Mirrors `sparrow-engine/scripts/ort-env.sh::pick_newest_cudnn_dir` (lines
    #    179-198). Two filename patterns can carry the version:
    #      (a) `libcudnn.so.9.X.Y.Z` — version-stamped sidecar (rare; ships
    #          with the standalone NVIDIA cuDNN tarball).
    #      (b) `nvidia_cudnn_cu12-X.Y.Z.W.dist-info/` adjacent to `lib/` —
    #          pip-wheel install (canonical engine path: `uv pip install
    #          --target ~/.local/cudnn 'nvidia-cudnn-cu12>=9.10'`). The
    #          versionless `libcudnn.so.9` lives in `lib/` and the dist-info
    #          dir lives one directory above (the wheel root).
    #
    #    Detection order: (a) first, then (b), then bare `libcudnn.so.9`
    #    presence with degraded `cudnn_warn` if version cannot be derived.
    _cudnn_ver=""
    _cudnn_path=""
    _cudnn_search_dirs="
        $HOME/.local/cudnn/nvidia/cudnn/lib
        /usr/lib/x86_64-linux-gnu
        /usr/local/cuda/lib64
        /usr/lib64
        /usr/lib
    "
    for _dir in $_cudnn_search_dirs; do
        [ -d "$_dir" ] || continue
        # (a) Version-stamped filename — pick newest by version sort.
        _f=$(find "$_dir" -maxdepth 1 -name 'libcudnn.so.9.*.*.*' -type f 2>/dev/null | sort -V | tail -n 1)
        if [ -n "$_f" ]; then
            _cudnn_ver=$(basename "$_f" | sed 's/^libcudnn\.so\.//')
            _cudnn_path="$_f"
            break
        fi
        # (b) Bare `libcudnn.so.9` + dist-info sidecar (pip/uv wheel install).
        if [ -e "$_dir/libcudnn.so.9" ]; then
            # The wheel root is the parent's parent ($_dir → cudnn/lib → cudnn → wheel-root).
            # nvidia_cudnn_cu12-X.Y.Z.W.dist-info usually lives at the wheel root.
            for _root in "$_dir/../.." "$_dir/.." "$_dir"; do
                # Pick highest version in case of side-by-side wheel installs.
                _di=$(find "$_root" -maxdepth 2 -name 'nvidia_cudnn_cu12-*.dist-info' -type d 2>/dev/null | sort -V | tail -n 1)
                if [ -n "$_di" ]; then
                    _cudnn_ver=$(basename "$_di" | sed -e 's/^nvidia_cudnn_cu12-//' -e 's/\.dist-info$//')
                    _cudnn_path="$_dir/libcudnn.so.9 (wheel: $_di)"
                    break
                fi
            done
            [ -n "$_cudnn_ver" ] && break
            # Bare `libcudnn.so.9` with no parseable dist-info: degrade to
            # cudnn_warn rather than reporting a fake version.
            _cudnn_path="$_dir/libcudnn.so.9 (version metadata missing)"
            break
        fi
    done

    # Fallback: also check `~/.cache/uv` (uv-managed wheels with version-stamped
    # archive dirs — `archive-v0/<hash>/nvidia_cudnn_cu12-X.Y.Z.W.dist-info`).
    if [ -z "$_cudnn_ver" ] && [ -d "$HOME/.cache/uv" ]; then
        # First try version-stamped filenames inside the cache.
        _f=$(find "$HOME/.cache/uv" -path '*/nvidia/cudnn/lib/libcudnn.so.9.*.*.*' -type f 2>/dev/null | sort -V | tail -n 1)
        if [ -n "$_f" ]; then
            _cudnn_ver=$(basename "$_f" | sed 's/^libcudnn\.so\.//')
            _cudnn_path="$_f"
        else
            # Try wheel dist-info parsing — `_di` is the full absolute path
            # to the dist-info directory; sort -V picks the highest version.
            _di=$(find "$HOME/.cache/uv" -name 'nvidia_cudnn_cu12-*.dist-info' -type d 2>/dev/null | sort -V | tail -n 1)
            if [ -n "$_di" ]; then
                _cudnn_ver=$(basename "$_di" | sed -e 's/^nvidia_cudnn_cu12-//' -e 's/\.dist-info$//')
                _cudnn_path="$_di (uv wheel cache)"
            fi
        fi
    fi

    # Diagnostic-string quoting (3 sites below): the install hint
    #   uv pip install --target ~/.local/cudnn 'nvidia-cudnn-cu12>=9.10'
    # is built via `printf '...%s...%s' "'" "'"` so the literal single
    # quotes survive in the output WITHOUT triggering shellcheck SC2089
    # (which fires on inline single-quoted-inside-double-quoted assignment
    # constructs). The user copy-pastes the printed line as-is into their
    # shell. Mirrors inquisitor F-6 R1 finding.
    _q="'"
    _pip_cmd=$(printf 'uv pip install --target ~/.local/cudnn %snvidia-cudnn-cu12>=9.10%s' "$_q" "$_q")
    if [ -z "$_cudnn_ver" ] && [ -z "$_cudnn_path" ]; then
        SPARROW_ENGINE_GPU_QUALITY="cudnn_err"
        SPARROW_ENGINE_GPU_QUALITY_REASON="cuDNN 9.x not found in expected paths (\$HOME/.local/cudnn, /usr/lib/x86_64-linux-gnu, /usr/local/cuda/lib64, ~/.cache/uv); the GPU flavor will fail at first inference. Install with: $_pip_cmd"
    elif [ -z "$_cudnn_ver" ]; then
        # Bare `libcudnn.so.9` present but no version metadata — degraded
        # warn (cannot verify floor, but the library is reachable).
        SPARROW_ENGINE_GPU_QUALITY="cudnn_warn"
        SPARROW_ENGINE_GPU_QUALITY_REASON="cuDNN found at $_cudnn_path but version metadata missing; cannot verify the 9.10 floor. Reinstall with: $_pip_cmd"
    else
        # Compare against 9.10.0 floor with portable version-sort. `sort -V`
        # is ascending: smaller versions come first. So if "9.10.0" is the
        # first element of `sort -V (ver, 9.10.0)`, then 9.10.0 <= ver, i.e.
        # ver >= 9.10.0 ⇒ ok. Conversely, if ver sorts first, then ver is
        # below the floor. Mirrors R1 § 1.2.1 + ort-env.sh:189-194.
        _floor_first=$(printf '%s\n9.10.0\n' "$_cudnn_ver" | sort -V | head -n 1)
        if [ "$_floor_first" = "9.10.0" ]; then
            SPARROW_ENGINE_GPU_QUALITY="ok"
            SPARROW_ENGINE_GPU_QUALITY_REASON="cuDNN $_cudnn_ver found at $_cudnn_path (at or above the 9.10.0 floor)"
        else
            SPARROW_ENGINE_GPU_QUALITY="cudnn_warn"
            SPARROW_ENGINE_GPU_QUALITY_REASON="cuDNN $_cudnn_ver found at $_cudnn_path, below the 9.10.0 floor; SpeciesNet on sm_89 will fail (known 9.8 ConvFwd asymmetric-padding bug). Install fix: $_pip_cmd"
        fi
    fi

    # cudnn_err is hard-fail by policy — promote to error exit at wrapper layer.
    # Don't escalate to error here in the function (`return 0` is invariant).
    if [ "$SPARROW_ENGINE_GPU_QUALITY" = "cudnn_err" ]; then
        export SPARROW_ENGINE_GPU_QUALITY SPARROW_ENGINE_GPU_QUALITY_REASON
        printf '%s\n' "fail"
        return 0
    fi

    # 2. Compute-capability check — only fires when cuDNN was at least found.
    #    nvidia-smi --query-gpu=compute_cap reports e.g. "8.9" → strip the dot
    #    → 89. < 80 means pre-Ampere (Volta/Turing/older); FP16 falls back to
    #    FP32 at 2-3× the production-cell latency.
    _nvsmi_path=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        _nvsmi_path="$(command -v nvidia-smi)"
    elif [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
        _nvsmi_path="/usr/lib/wsl/lib/nvidia-smi"
    fi

    _cc=""
    if [ -n "$_nvsmi_path" ]; then
        _cc=$("$_nvsmi_path" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' .')
    fi

    if [ -n "$_cc" ] && [ "$_cc" -lt 80 ] 2>/dev/null; then
        if [ "$SPARROW_ENGINE_GPU_QUALITY" = "ok" ]; then
            SPARROW_ENGINE_GPU_QUALITY="sm_warn"
            SPARROW_ENGINE_GPU_QUALITY_REASON="$SPARROW_ENGINE_GPU_QUALITY_REASON; compute_cap=$_cc (< sm_80) — FP16 production cells fall back to FP32, ~2-3× slower"
        else
            SPARROW_ENGINE_GPU_QUALITY_REASON="$SPARROW_ENGINE_GPU_QUALITY_REASON; compute_cap=$_cc (< sm_80)"
        fi
    fi

    export SPARROW_ENGINE_GPU_QUALITY SPARROW_ENGINE_GPU_QUALITY_REASON

    # Map the 4-state quality verdict onto a 3-state stdout signal for the
    # wrapper: pass | warn | fail.
    case "$SPARROW_ENGINE_GPU_QUALITY" in
        ok)                     printf '%s\n' "pass" ;;
        sm_warn|cudnn_warn)     printf '%s\n' "warn" ;;
        cudnn_err)              printf '%s\n' "fail" ;;
        *)                      printf '%s\n' "warn" ;;
    esac
    return 0
}

# Direct-invocation block — fires only when this file is executed (not sourced).
# Portable bash + zsh detection (mirrors `probe.sh`).
_probe_sourced=0
if [ -n "${ZSH_EVAL_CONTEXT-}" ]; then
    case "$ZSH_EVAL_CONTEXT" in *:file*) _probe_sourced=1 ;; esac
elif [ -n "${BASH_SOURCE-}" ]; then
    [ "${BASH_SOURCE[0]}" != "$0" ] && _probe_sourced=1
fi
if [ "$_probe_sourced" -eq 0 ]; then
    probe_gpu_quality
fi
unset _probe_sourced
