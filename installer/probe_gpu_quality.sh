#!/usr/bin/env bash
# installer/probe_gpu_quality.sh — sourceable layer-2 quality probe
#                                  (cuDNN ≥9.10 floor + driver-version sanity).
#
# Purpose
#   Layer-2 of the Sparrow Engine install-time selector. Runs ONLY after layer-1
#   (`probe.sh`) has returned `gpu`. The basic CUDA probe answers "is CUDA
#   reachable?" — this layer answers two follow-up questions:
#     1. Is cuDNN ≥9.10 reachable? (canonical project floor — cuDNN 9.8 has
#        the asymmetric-padding ConvFwd engine bug that breaks SpeciesNet on
#        sm_89; sources cited in `probe_cudnn_check` below).
#     2. Is the GPU compute-capability ≥sm_80? (FP16 production cells need
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
#           cudnn_err)   die 11 "cuDNN <9.10 — block install" ;;
#       esac
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

probe_gpu_quality() {
    SPARROW_ENGINE_GPU_QUALITY=""
    SPARROW_ENGINE_GPU_QUALITY_REASON=""

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
