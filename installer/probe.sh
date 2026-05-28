#!/usr/bin/env bash
# installer/probe.sh — sourceable CUDA-detection probe (Linux/macOS/WSL2).
#
# Purpose
#   Layer-1 of the Sparrow Engine install-time selector: decide whether the host
#   should prefer the CPU CLI binary (`spe`) or the GPU CLI binary (`spe-gpu`).
#   Sets two POSIX env vars that the wrapper (`installer/sparrow-engine-install.sh`) reads, and
#   echoes the verdict to stdout for callers using the function form.
#
# Usage
#   Sourceable form (preferred — wrapper integration):
#       . installer/probe.sh
#       probe_cuda
#       echo "$SPARROW_ENGINE_DETECTED_FLAVOR"          # cpu | gpu
#       echo "$SPARROW_ENGINE_DETECTED_PROBE_REASON"    # diagnostic string
#
#   Direct invocation (one-shot, e.g. `--probe-only` or smoke test):
#       bash installer/probe.sh                # writes verdict to stdout
#       SPARROW_ENGINE_INSTALL_FLAVOR=gpu bash installer/probe.sh    # honors override
#
#   Cache-source form (used by the install.sh wrapper after the Phase E B-01 fix):
#       # install.sh fetches probe.sh once into a cache dir, then sources it:
#       probe_path=$(locate_helper probe.sh)        # returns disk path
#       . "$probe_path"
#       probe_cuda                                  # caller invokes explicitly
#       echo "$SPARROW_ENGINE_DETECTED_FLAVOR"
#
#   In all source/cache-source modes the standalone-invocation block at the
#   bottom of this file detects ${BASH_SOURCE[0]} != $0 and SKIPS auto-running
#   probe_cuda. Auto-run fires only when bash is invoked with probe.sh as the
#   entry-point argv[0] (e.g. `bash installer/probe.sh`).
#
# Env vars set
#   SPARROW_ENGINE_DETECTED_FLAVOR        cpu | gpu
#   SPARROW_ENGINE_DETECTED_PROBE_REASON  short string explaining the decision
#
# Exit codes
#   This script always returns 0 (probe never blocks). The wrapper consumes
#   SPARROW_ENGINE_DETECTED_FLAVOR + SPARROW_ENGINE_DETECTED_PROBE_REASON and emits its own
#   exit codes per `final_design.md § 2.10`.
#
# Design source
#   docs/design/phase4.1-install-selector/final_design.md § 2.3
#   docs/design/phase4.1-install-selector/round_04/scripts-architect_proposal.md
#   docs/design/phase4.1-install-selector/round_02/scripts-architect_proposal.md § 1.1.WSL2 + § 1.4
#   docs/design/phase4.1-install-selector/round_01/scripts-architect_proposal.md § 2.2 (canonical pseudocode)
#
# NEVER `exit` from this file — caller may have sourced it. Use `return` from
# inside the function; the standalone-invocation block at the bottom prints
# the verdict to stdout and exits there. (Mirrors `sparrow-engine/scripts/ort-env.sh`
# discipline; ort-env.sh comments warn that `exit` from a sourced context
# kills the caller's interactive shell.)

probe_cuda() {
    # 1. Override path — highest priority (wrapper resolves CLI flags first).
    if [ -n "${SPARROW_ENGINE_INSTALL_FLAVOR:-}" ]; then
        case "$SPARROW_ENGINE_INSTALL_FLAVOR" in
            cpu|gpu)
                SPARROW_ENGINE_DETECTED_FLAVOR="$SPARROW_ENGINE_INSTALL_FLAVOR"
                SPARROW_ENGINE_DETECTED_PROBE_REASON="SPARROW_ENGINE_INSTALL_FLAVOR=$SPARROW_ENGINE_INSTALL_FLAVOR (env override)"
                export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
                printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
                return 0
                ;;
            *)
                # Unknown value: warn and fall through to probe. NEVER exit.
                echo >&2 "warn: SPARROW_ENGINE_INSTALL_FLAVOR='$SPARROW_ENGINE_INSTALL_FLAVOR' not in {cpu, gpu}; ignoring."
                ;;
        esac
    fi

    # 1.5. aarch64 Linux subpath — Jetson + Grace Hopper. Always CPU today
    #      (we do not ship aarch64 GPU artifacts). Fires BEFORE nvidia-smi
    #      probe so Jetson configs lacking nvidia-smi are not misclassified.
    #      Per R2 § 1.4 (M-6).
    if [ "$(uname -m 2>/dev/null)" = "aarch64" ] && [ "$(uname -s 2>/dev/null)" = "Linux" ]; then
        if [ -e /etc/nv_tegra_release ] || [ -d /proc/device-tree/tegra-pmc ]; then
            SPARROW_ENGINE_DETECTED_FLAVOR="cpu"
            SPARROW_ENGINE_DETECTED_PROBE_REASON="Jetson detected (aarch64 + Tegra signature); spe-gpu does not ship for L4T. Override with SPARROW_ENGINE_INSTALL_FLAVOR=gpu if building from source."
            export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
            printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
            return 0
        fi
        if command -v nvidia-smi >/dev/null 2>&1 && timeout 5 nvidia-smi -L >/dev/null 2>&1; then
            SPARROW_ENGINE_DETECTED_FLAVOR="cpu"
            SPARROW_ENGINE_DETECTED_PROBE_REASON="aarch64 Linux + NVIDIA GPU detected (likely Grace Hopper / GH200). spe-gpu does not ship aarch64 artifacts yet."
            export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
            printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
            return 0
        fi
        # aarch64 Linux without NVIDIA: fall through to default CPU path below.
    fi

    # 2. macOS short-circuit — final.
    case "$(uname -s 2>/dev/null)" in
        Darwin)
            SPARROW_ENGINE_DETECTED_FLAVOR="cpu"
            SPARROW_ENGINE_DETECTED_PROBE_REASON="macOS detected (uname=Darwin); spe-gpu unsupported on macOS"
            export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
            printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
            return 0
            ;;
    esac

    # 3. WSL2 detection — distinct probe path because /dev/nvidia* is absent
    #    on WSL2; CUDA-on-WSL2 ships through /dev/dxg + /usr/lib/wsl/lib/.
    _is_wsl=0
    if [ -r /proc/sys/kernel/osrelease ]; then
        case "$(cat /proc/sys/kernel/osrelease 2>/dev/null)" in
            *microsoft*|*Microsoft*|*WSL*|*wsl*) _is_wsl=1 ;;
        esac
    fi

    # 4. nvidia-smi primary probe (with 5s timeout to bound a wedged driver).
    _nvsmi_path=""
    if command -v nvidia-smi >/dev/null 2>&1; then
        _nvsmi_path="$(command -v nvidia-smi)"
    elif [ "$_is_wsl" -eq 1 ] && [ -x /usr/lib/wsl/lib/nvidia-smi ]; then
        # WSL2 SSH sessions may not have /usr/lib/wsl/lib on PATH (microsoft/WSL #9185).
        _nvsmi_path="/usr/lib/wsl/lib/nvidia-smi"
    fi

    _nvsmi_diag=""
    _nvsmi_rc=1
    _nvsmi_out=""
    if [ -n "$_nvsmi_path" ]; then
        if command -v timeout >/dev/null 2>&1; then
            _nvsmi_out="$(timeout 5 "$_nvsmi_path" -L 2>&1)"
            _nvsmi_rc=$?
        else
            _nvsmi_out="$("$_nvsmi_path" -L 2>&1)"
            _nvsmi_rc=$?
        fi

        if [ "$_nvsmi_rc" -eq 0 ] && [ -n "$_nvsmi_out" ] && [ "$_is_wsl" -eq 0 ]; then
            # Bare-metal / container Linux + nvidia-smi healthy ⇒ GPU.
            _gpu_line="$(printf '%s\n' "$_nvsmi_out" | head -n1)"
            SPARROW_ENGINE_DETECTED_FLAVOR="gpu"
            SPARROW_ENGINE_DETECTED_PROBE_REASON="nvidia-smi reports: $_gpu_line"
            export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
            printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
            return 0
        elif [ "$_nvsmi_rc" -eq 124 ]; then
            _nvsmi_diag="nvidia-smi timed out (>5s); falling back to filesystem probe"
        elif [ "$_nvsmi_rc" -ne 0 ]; then
            _nvsmi_diag="nvidia-smi exit $_nvsmi_rc: $(printf '%s' "$_nvsmi_out" | head -n1)"
        fi
    fi

    # 5. WSL2 three-way gate — DXG + nvidia-smi + libcuda.so.1 all reachable.
    #    Per R2 § 1.1.WSL2 (F-8 resolution). Conservative gate: missing any
    #    one component yields CPU with a precise diagnostic. User override
    #    (--flavor gpu) is the documented escape hatch for partial-state.
    if [ "$_is_wsl" -eq 1 ]; then
        _has_dxg=0; _has_nvsmi=0; _has_libcuda=0
        [ -e /dev/dxg ] && _has_dxg=1
        [ "$_nvsmi_rc" -eq 0 ] && [ -n "$_nvsmi_out" ] && _has_nvsmi=1
        if [ -e /usr/lib/wsl/lib/libcuda.so.1 ]; then
            _has_libcuda=1
        elif [ -e /usr/lib/wsl/lib/libcuda.so ]; then
            # Stub ships as libcuda.so without .1 suffix on some Windows-driver
            # versions (NVIDIA WSL forum issue #236301). Loaders looking for
            # libcuda.so.1 will fail; warn but treat as present.
            _has_libcuda=1
        elif command -v ldconfig >/dev/null 2>&1 && ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; then
            _has_libcuda=1
        fi

        if [ "$_has_dxg" -eq 1 ] && [ "$_has_nvsmi" -eq 1 ] && [ "$_has_libcuda" -eq 1 ]; then
            SPARROW_ENGINE_DETECTED_FLAVOR="gpu"
            SPARROW_ENGINE_DETECTED_PROBE_REASON="WSL2: /dev/dxg + nvidia-smi + libcuda.so.1 all reachable; CUDA-on-WSL2 active"
            export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
            printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
            return 0
        fi

        SPARROW_ENGINE_DETECTED_FLAVOR="cpu"
        _wsl_missing=""
        [ "$_has_dxg" -eq 0 ] && _wsl_missing="$_wsl_missing /dev/dxg"
        [ "$_has_nvsmi" -eq 0 ] && _wsl_missing="$_wsl_missing nvidia-smi"
        [ "$_has_libcuda" -eq 0 ] && _wsl_missing="$_wsl_missing libcuda.so.1"
        SPARROW_ENGINE_DETECTED_PROBE_REASON="WSL2 detected, missing:$_wsl_missing — install CUDA-on-WSL2 per https://docs.nvidia.com/cuda/wsl-user-guide/"
        export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
        printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
        return 0
    fi

    # 6. Filesystem fallback — for the case where nvidia-smi is missing or
    #    failed but /dev/nvidia0 + /dev/nvidiactl are present (rare but real:
    #    minimal containers, broken PATH).
    if [ -e /dev/nvidia0 ] && [ -e /dev/nvidiactl ]; then
        _libcuda_found=0
        if command -v ldconfig >/dev/null 2>&1; then
            if ldconfig -p 2>/dev/null | grep -q 'libcuda\.so\.1'; then
                _libcuda_found=1
            fi
        fi
        # Glob fallback for musl/Alpine (no `ldconfig -p`).
        if [ "$_libcuda_found" -eq 0 ]; then
            for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib /lib/x86_64-linux-gnu; do
                if [ -e "$d/libcuda.so.1" ]; then
                    _libcuda_found=1
                    break
                fi
            done
        fi

        if [ "$_libcuda_found" -eq 1 ]; then
            SPARROW_ENGINE_DETECTED_FLAVOR="gpu"
            SPARROW_ENGINE_DETECTED_PROBE_REASON="device nodes /dev/nvidia0 + /dev/nvidiactl present; libcuda.so.1 reachable"
            export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
            printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
            return 0
        fi
    fi

    # 7. Partial-state diagnostic — kernel module loaded but device nodes
    #    missing (vfio-pci passthrough host, broken nvidia-modprobe, etc.).
    if [ -e /proc/driver/nvidia/version ]; then
        SPARROW_ENGINE_DETECTED_FLAVOR="cpu"
        SPARROW_ENGINE_DETECTED_PROBE_REASON="NVIDIA kernel module loaded but /dev/nvidia0 missing — partial driver state; reinstalling driver may help, or override with --flavor gpu if PCIe passthrough is intended"
        export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
        printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
        return 0
    fi

    # 8. Default CPU — no NVIDIA driver detected anywhere.
    SPARROW_ENGINE_DETECTED_FLAVOR="cpu"
    if [ -n "$_nvsmi_diag" ]; then
        SPARROW_ENGINE_DETECTED_PROBE_REASON="$_nvsmi_diag"
    else
        SPARROW_ENGINE_DETECTED_PROBE_REASON="no NVIDIA driver detected (no nvidia-smi, no /dev/nvidia0, no /proc/driver/nvidia/version)"
    fi
    export SPARROW_ENGINE_DETECTED_FLAVOR SPARROW_ENGINE_DETECTED_PROBE_REASON
    printf '%s\n' "$SPARROW_ENGINE_DETECTED_FLAVOR"
    return 0
}

# Direct-invocation block — fires only when this file is executed (not sourced).
# Portable bash + zsh detection:
#   bash sourced: ${BASH_SOURCE[0]} is the script path, $0 is the parent shell name.
#   bash exec:    ${BASH_SOURCE[0]} == $0.
#   zsh sourced:  $ZSH_EVAL_CONTEXT contains ":file"; $0 is the script path too.
#   zsh exec:     $ZSH_EVAL_CONTEXT is "toplevel" (or empty in `zsh -c`).
_probe_sourced=0
if [ -n "${ZSH_EVAL_CONTEXT-}" ]; then
    case "$ZSH_EVAL_CONTEXT" in *:file*) _probe_sourced=1 ;; esac
elif [ -n "${BASH_SOURCE-}" ]; then
    [ "${BASH_SOURCE[0]}" != "$0" ] && _probe_sourced=1
fi
if [ "$_probe_sourced" -eq 0 ]; then
    probe_cuda
fi
unset _probe_sourced
