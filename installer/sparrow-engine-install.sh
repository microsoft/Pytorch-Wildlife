#!/usr/bin/env bash
# installer/sparrow-engine-install.sh — Sparrow Engine install-time selector wrapper (Linux/macOS)
#
# Defense-in-depth wrapping for `curl ... | sh` truncation safety:
#   1. Wrap entire body in `main() { ... }` and call as last line: `main "$@" || exit 1`.
#      A truncated download leaves an incomplete `main()` definition → bash parse-fail
#      → no partial execution.
#   2. `set -euo pipefail` at top of main().
#      pipefail closes silent-failure paths in `state_read_flavor` (grep|head|sed)
#      and `verify_sha256` (sha256sum|awk): when an upstream pipe element fails,
#      the substitution captures empty and the caller sees the failure surface.
#      Per inquisitor R1 LOW finding (review/phase4.1-install-selector-impl-audit-fix/
#      round_01/inquisitor_pre_emptive_findings.md § 4).
#   3. `ensure()` wrapper around every fallible command (rustup-style).
#
# Reference: docs/design/phase4.1-install-selector/final_design.md §2.2 (canonical).
# Probe contract: sources installer/probe.sh; calls probe_cuda; reads SPARROW_ENGINE_DETECTED_FLAVOR.
set -euo pipefail

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# SPARROW_ENGINE_VERSION is the pinned default release tag. Bumped per
# Phase F release-CI step on every published GH Release (`vX.Y.Z`). The
# `SPARROW_ENGINE_VERSION` env var overrides this for advanced users
# wanting to install an older or newer release.
SPARROW_ENGINE_VERSION="${SPARROW_ENGINE_VERSION:-0.1.17}"
SPARROW_ENGINE_PREFIX="${SPARROW_ENGINE_PREFIX:-$HOME/.sparrow-engine}"
SPARROW_ENGINE_STATE_FILE="$SPARROW_ENGINE_PREFIX/installed.json"
# Default release base = public GH Releases asset URL. Phase E B-02 fix
# (was: file:///tmp/sparrow-engine-release/v${ver}). Operator override via
# `SPARROW_ENGINE_RELEASE_BASE=<url>` (staging mirror / internal proxy).
SPARROW_ENGINE_RELEASE_BASE_DEFAULT="https://github.com/microsoft/Pytorch-Wildlife/releases/download/v${SPARROW_ENGINE_VERSION}"
# Helper-script base = immutable raw-tag path. Helper scripts (probe.sh,
# probe_gpu_quality.sh) are NOT published as release assets — they only
# live in the tagged source tree. Phase E round-2 fix for E-R2-1 (the
# release_base()/probe.sh URL returned 404). Operator override via
# `SPARROW_ENGINE_HELPER_BASE=<url>` for testing against a local mirror.
SPARROW_ENGINE_HELPER_BASE_DEFAULT="https://raw.githubusercontent.com/microsoft/Pytorch-Wildlife/refs/tags/v${SPARROW_ENGINE_VERSION}/installer"
# Helper-script cache dir for piped install (B-01). Used when the wrapper
# is invoked via `curl ... | sh` / `bash <(curl ...)` and no probe.sh /
# probe_gpu_quality.sh exists on disk next to the wrapper.
SPARROW_ENGINE_HELPER_CACHE="${XDG_CACHE_HOME:-$HOME/.cache}/sparrow-engine/v${SPARROW_ENGINE_VERSION}"
SENTINEL_BEGIN="# >>> sparrow-engine >>>"
SENTINEL_END="# <<< sparrow-engine <<<"

# Globals populated by parse_args.
mode_arg=""
flavor=""
reinstall=0
reprobe=0
uninstall=0
force_rc_overwrite=0
dry_run=0
probe_only=0
retries=3
yes=0

# -----------------------------------------------------------------------------
# Logging + error helpers
# -----------------------------------------------------------------------------
say() {
    printf 'sparrow-engine-install: %s\n' "$*"
}

warn() {
    printf 'sparrow-engine-install: warn: %s\n' "$*" >&2
}

err() {
    printf 'sparrow-engine-install: error: %s\n' "$*" >&2
}

die() {
    # die EXITCODE MESSAGE
    local code=$1
    shift
    err "$*"
    exit "$code"
}

# rustup-style ensure: run the command; abort with the same exit code on failure.
# Logs a one-line failure annotation that points at the failing command.
ensure() {
    if ! "$@"; then
        die 1 "command failed: $*"
    fi
}

usage() {
    cat <<EOF
sparrow-engine-install ${SPARROW_ENGINE_VERSION} — install-time selector for Sparrow Engine

Usage:
  sparrow-engine-install.sh [MODE] [FLAVOR] [BYPASS] [OTHER]

Modes (auto-detected if omitted):
  --pip               Install Python wheel (sparrow-engine CPU / sparrow-engine-gpu GPU) via pip / uv pip
  --cli, --cli-only   Download + extract CLI tarball into \$HOME/.sparrow-engine
  --docker            Pull docker image zhongqimiao/sparrow-engine-server:latest (CPU) or zhongqimiao/sparrow-engine-server-gpu:latest (GPU)

Flavor:
  --flavor cpu|gpu|auto    auto = run probe (default).
                           Honors SPARROW_ENGINE_INSTALL_FLAVOR env unless --flavor=auto.

Bypass:
  --reinstall              Same-flavor force-overwrite (skip "already installed" gate)
  --reprobe                Cross-flavor switch: re-run probe; uninstall existing; install new
  --uninstall              Wrapper-owned uninstall

Other:
  --force-rc-overwrite     Replace sparrow-engine block in rc file even if manually edited
  --probe-only             Run probe(s); print resolved flavor + diagnostic; do not install
  --retries=N              HTTP retry attempts (N total attempts; 0 = never try; default 3)
  --dry-run                Print actions; no network / install / state-write
  -y, --yes                Suppress [y/N] prompts (non-interactive)
  --help                   This message
  --version                Print installer version

Environment:
  SPARROW_ENGINE_INSTALL_FLAVOR=cpu|gpu       Override probe (subordinate to --flavor)
  SPARROW_ENGINE_RELEASE_BASE=<url>           Override release URL prefix
                                     (default: ${SPARROW_ENGINE_RELEASE_BASE_DEFAULT})
  SPARROW_ENGINE_VERSION=X.Y.Z                Override target version
  SPARROW_ENGINE_NO_MODIFY_PATH=1              Skip rc-file edits

Exit codes (canonical: docs/design/phase4.1-install-selector/final_design.md §2.10):
  0  Success
  1  Generic error
  2  User aborted (Ctrl-C)
  3  Probe disagreement (override conflicts with hardware)
  4  Network failure (after retries)
  5  Python too old (<3.11)
  6  sha256 verification failed
  7  Disk space insufficient
  8  Required tool missing (curl/tar/docker/pip)
  9  Platform/flavor combination not supported
  10 OS not supported
  11 cuDNN <9.10 (driver layer-2 probe failure)
  12 Cross-flavor install attempted without --reprobe
  13 Manual rc-file edit detected without --force-rc-overwrite

See docs/install.md for the user guide.
EOF
}

# -----------------------------------------------------------------------------
# Args
# -----------------------------------------------------------------------------
parse_args() {
    flavor="auto"
    while [ $# -gt 0 ]; do
        case "$1" in
            --pip)                 mode_arg="pip" ;;
            --cli|--cli-only)      mode_arg="cli" ;;
            --docker)              mode_arg="docker" ;;
            --flavor=*)            flavor="${1#--flavor=}" ;;
            --flavor)              shift; flavor="${1:-}" ;;
            --reinstall)           reinstall=1 ;;
            --reprobe)             reprobe=1 ;;
            --uninstall)           uninstall=1 ;;
            --force-rc-overwrite)  force_rc_overwrite=1 ;;
            --probe-only)          probe_only=1 ;;
            --retries=*)           retries="${1#--retries=}" ;;
            --retries)             shift; retries="${1:-}" ;;
            --dry-run)             dry_run=1 ;;
            -y|--yes)              yes=1 ;;
            --help|-h)             usage; exit 0 ;;
            --version|-V)          printf '%s\n' "$SPARROW_ENGINE_VERSION"; exit 0 ;;
            *)                     die 1 "unknown argument: $1 (run with --help)" ;;
        esac
        shift
    done

    case "$flavor" in
        cpu|gpu|auto) : ;;
        *) die 1 "invalid --flavor: $flavor (must be cpu, gpu, or auto)" ;;
    esac

    case "$retries" in
        ''|*[!0-9]*) die 1 "invalid --retries value: $retries (must be a non-negative integer)" ;;
    esac
}

# -----------------------------------------------------------------------------
# OS / arch detection
# -----------------------------------------------------------------------------
detect_os_arch() {
    local uname_s uname_m
    uname_s=$(uname -s)
    uname_m=$(uname -m)

    case "$uname_s" in
        Linux)  OS="linux" ;;
        Darwin) OS="macos" ;;
        *) die 10 "unsupported OS: $uname_s (Linux + macOS only on the .sh wrapper; Windows uses sparrow-engine-install.ps1)" ;;
    esac

    case "$uname_m" in
        x86_64|amd64) ARCH="x86_64" ;;
        arm64|aarch64) ARCH="aarch64" ;;
        *) die 10 "unsupported arch: $uname_m" ;;
    esac
}

# -----------------------------------------------------------------------------
# Mode auto-detection
# -----------------------------------------------------------------------------
auto_detect_mode() {
    # Heuristics in priority order (final_design §2.2, packaging R4 §5):
    #   (1) Running inside an active virtualenv? → pip
    #   (2) `docker` on PATH and DOCKER_HOST set / docker.sock present? → docker
    #   (3) Default: cli
    if [ -n "${VIRTUAL_ENV:-}" ] || [ -n "${CONDA_PREFIX:-}" ]; then
        printf 'pip\n'
        return 0
    fi
    if command -v docker >/dev/null 2>&1 && [ -n "${DOCKER_HOST:-}" ]; then
        printf 'docker\n'
        return 0
    fi
    printf 'cli\n'
}

# -----------------------------------------------------------------------------
# Flavor resolution (sources probe.sh)
# -----------------------------------------------------------------------------
resolve_flavor() {
    if [ "$flavor" = "auto" ]; then
        # Honor --flavor auto: ignore SPARROW_ENGINE_INSTALL_FLAVOR env so probe runs.
        unset SPARROW_ENGINE_INSTALL_FLAVOR

        # probe.sh is owned by coder-probe; we source it. Resolved via
        # locate_helper to support both disk install (adjacent on disk) and
        # piped install (`curl ... | sh`) — see locate_helper definition.
        local probe_path
        probe_path=$(locate_helper probe.sh)

        # Source the probe; it defines probe_cuda which echoes "cpu" or "gpu"
        # AND exports SPARROW_ENGINE_DETECTED_FLAVOR + SPARROW_ENGINE_DETECTED_PROBE_REASON.
        # shellcheck source=installer/probe.sh
        # shellcheck disable=SC1091
        . "$probe_path"

        if ! command -v probe_cuda >/dev/null 2>&1; then
            die 1 "probe.sh did not define probe_cuda function"
        fi

        # Call probe_cuda DIRECTLY (not via $()) so its `export` calls reach
        # this shell. probe_cuda also writes the verdict to stdout — redirect
        # to a tempfile so we don't pollute our own stdout and can recover
        # the verdict for callers that don't read $SPARROW_ENGINE_DETECTED_FLAVOR.
        # The exports persist into SPARROW_ENGINE_DETECTED_PROBE_REASON for --probe-only.
        local _probe_out
        _probe_out=$(mktemp)
        probe_cuda > "$_probe_out"
        flavor="${SPARROW_ENGINE_DETECTED_FLAVOR:-$(cat "$_probe_out")}"
        rm -f "$_probe_out"
        case "$flavor" in
            cpu|gpu) : ;;
            *) die 1 "probe_cuda returned invalid flavor: $flavor" ;;
        esac
    elif [ -n "${SPARROW_ENGINE_INSTALL_FLAVOR:-}" ] && [ "$flavor" != "${SPARROW_ENGINE_INSTALL_FLAVOR}" ]; then
        # User passed both --flavor and SPARROW_ENGINE_INSTALL_FLAVOR; --flavor wins (per §5.1)
        # but warn so the env-var setter notices.
        warn "SPARROW_ENGINE_INSTALL_FLAVOR=$SPARROW_ENGINE_INSTALL_FLAVOR ignored; --flavor=$flavor takes precedence"
    fi

    say "selected flavor: $flavor"
}

# -----------------------------------------------------------------------------
# Layer-2 GPU quality probe (cuDNN ≥9.10 floor + compute-cap warn).
# Only invoked when flavor=gpu. Sources installer/probe_gpu_quality.sh,
# calls probe_gpu_quality (which exports SPARROW_ENGINE_GPU_QUALITY +
# SPARROW_ENGINE_GPU_QUALITY_REASON), then dispatches on the 4-state verdict per
# `final_design.md § 2.4 + § 2.10`:
#   - ok         → silent continue
#   - sm_warn    → log warn + continue (FP16 falls back to FP32; not blocking)
#   - cudnn_warn → log warn + continue (cuDNN reachable but version unknown
#                  or below floor; SpeciesNet may fail at first inference)
#   - cudnn_err  → die 11 (cuDNN <9.10 or absent; install would fail)
# Unknown verdict → log warn + continue (defensive fallback).
#
# Citation chain for the cuDNN ≥9.10 floor (Conv-engine bug on sm_89 for
# cuDNN 9.8 that breaks SpeciesNet inference):
#   - sparrow-engine/scripts/ort-env.sh:167-168
#   - docs/lessons.md:29
#   - docs/tech_report/06_gotchas_and_constraints.md:17-25
# -----------------------------------------------------------------------------
gpu_quality_check() {
    if [ "$flavor" != "gpu" ]; then
        return 0
    fi
    local probeq_path
    probeq_path=$(locate_helper probe_gpu_quality.sh)
    # shellcheck source=installer/probe_gpu_quality.sh
    # shellcheck disable=SC1091
    . "$probeq_path"
    if ! command -v probe_gpu_quality >/dev/null 2>&1; then
        die 1 "probe_gpu_quality.sh did not define probe_gpu_quality function"
    fi
    # Run the probe. probe_gpu_quality writes pass|warn|fail to stdout but the
    # authoritative verdict is in the SPARROW_ENGINE_GPU_QUALITY env var (4-state).
    # Discard stdout so it doesn't pollute our own output.
    probe_gpu_quality > /dev/null
    case "${SPARROW_ENGINE_GPU_QUALITY:-}" in
        ok)
            say "GPU quality: ${SPARROW_ENGINE_GPU_QUALITY_REASON:-ok}"
            ;;
        sm_warn|cudnn_warn)
            warn "GPU quality: ${SPARROW_ENGINE_GPU_QUALITY_REASON:-degraded}"
            ;;
        cudnn_err)
            die 11 "${SPARROW_ENGINE_GPU_QUALITY_REASON:-cuDNN <9.10 floor not met}"
            ;;
        *)
            warn "GPU quality: unknown verdict (SPARROW_ENGINE_GPU_QUALITY=${SPARROW_ENGINE_GPU_QUALITY:-empty})"
            ;;
    esac
}

# -----------------------------------------------------------------------------
# State file
# -----------------------------------------------------------------------------
state_read_flavor() {
    if [ ! -f "$SPARROW_ENGINE_STATE_FILE" ]; then
        return 1
    fi
    # Tolerant of whitespace + jq absence: grep + sed.
    grep -Eo '"flavor"[[:space:]]*:[[:space:]]*"[^"]+"' "$SPARROW_ENGINE_STATE_FILE" \
        | head -1 | sed -E 's/.*"flavor"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/'
}

record_state() {
    local rec_flavor=$1
    local rec_mode=$2
    local ts
    ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if [ "$dry_run" -eq 1 ]; then
        say "[dry-run] would write $SPARROW_ENGINE_STATE_FILE (flavor=$rec_flavor mode=$rec_mode ts=$ts)"
        return 0
    fi
    mkdir -p "$SPARROW_ENGINE_PREFIX"
    cat > "$SPARROW_ENGINE_STATE_FILE" <<EOF
{
  "flavor": "$rec_flavor",
  "mode": "$rec_mode",
  "version": "$SPARROW_ENGINE_VERSION",
  "installed_at": "$ts",
  "os": "${OS:-unknown}",
  "arch": "${ARCH:-unknown}"
}
EOF
}

# -----------------------------------------------------------------------------
# Cross-flavor / reinstall gating
# -----------------------------------------------------------------------------
check_existing_install() {
    local existing
    existing=$(state_read_flavor 2>/dev/null || true)
    if [ -z "$existing" ]; then
        return 0   # fresh install
    fi

    if [ "$existing" = "$flavor" ]; then
        if [ "$reinstall" -eq 1 ] || [ "$reprobe" -eq 1 ]; then
            say "existing $existing install detected; reinstalling"
            return 0
        fi
        say "$existing flavor already installed at $SPARROW_ENGINE_PREFIX (pass --reinstall to overwrite)"
        # Same-flavor re-invocation is a soft no-op.
        exit 0
    fi

    # Different flavor.
    if [ "$reprobe" -ne 1 ]; then
        die 12 "cannot install $flavor: $existing flavor already installed at $SPARROW_ENGINE_PREFIX. Use --reprobe to switch flavors, or --uninstall first."
    fi

    # Cross-flavor with --reprobe — ask for confirmation unless -y.
    if [ "$yes" -ne 1 ]; then
        printf 'switch from %s to %s? [y/N] ' "$existing" "$flavor"
        local answer=""
        if ! IFS= read -r answer; then
            answer=""
        fi
        case "$answer" in
            [yY]|[yY][eE][sS]) : ;;
            *) die 2 "user aborted flavor switch" ;;
        esac
    fi
    say "switching $existing → $flavor (reprobe)"
    do_uninstall_silent
}

# -----------------------------------------------------------------------------
# pip mode
# -----------------------------------------------------------------------------
install_python_wheel() {
    local wheel_flavor=$1
    local pkg
    case "$wheel_flavor" in
        cpu) pkg="sparrow-engine" ;;
        gpu) pkg="sparrow-engine-gpu" ;;
        *)   die 1 "internal: unknown flavor $wheel_flavor in install_python_wheel" ;;
    esac

    # Python ≥3.11 floor (CLAUDE.md PyO3 0.25 invariant).
    if command -v python3 >/dev/null 2>&1; then
        local py_major py_minor
        py_major=$(python3 -c 'import sys; print(sys.version_info.major)' 2>/dev/null || echo 0)
        py_minor=$(python3 -c 'import sys; print(sys.version_info.minor)' 2>/dev/null || echo 0)
        if [ "$py_major" -lt 3 ] || { [ "$py_major" -eq 3 ] && [ "$py_minor" -lt 11 ]; }; then
            die 5 "python ${py_major}.${py_minor} is too old; Sparrow Engine requires Python >=3.11"
        fi
    else
        die 8 "python3 not found on PATH (required for --pip mode)"
    fi

    local installer
    if command -v uv >/dev/null 2>&1; then
        installer="uv pip install"
    elif command -v pip >/dev/null 2>&1; then
        installer="pip install"
    elif command -v pip3 >/dev/null 2>&1; then
        installer="pip3 install"
    else
        die 8 "no Python package manager found (need uv, pip, or pip3)"
    fi

    if [ "$reinstall" -eq 1 ] || [ "$reprobe" -eq 1 ]; then
        installer="$installer --force-reinstall"
    fi

    if [ "$dry_run" -eq 1 ]; then
        say "[dry-run] would run: $installer $pkg"
        return 0
    fi

    say "running: $installer $pkg"
    # shellcheck disable=SC2086
    ensure $installer "$pkg"
}

# -----------------------------------------------------------------------------
# cli mode (tarball)
# -----------------------------------------------------------------------------
release_base() {
    local rb="${SPARROW_ENGINE_RELEASE_BASE:-}"
    if [ -z "$rb" ]; then
        # Default = public GH Releases asset URL (Phase E B-02 fix; was
        # file:///tmp/... dev placeholder). Honors operator override via
        # SPARROW_ENGINE_RELEASE_BASE for staging mirrors / internal proxies.
        rb="$SPARROW_ENGINE_RELEASE_BASE_DEFAULT"
    fi
    # Strip trailing slashes for consistent join.
    printf '%s' "${rb%/}"
}

# Helper-script base URL. Distinct from release_base() because helper
# scripts live in the tagged source tree (raw.githubusercontent.com), not
# as release assets. E-R2-1 fix.
helper_base() {
    local hb="${SPARROW_ENGINE_HELPER_BASE:-}"
    if [ -z "$hb" ]; then
        hb="$SPARROW_ENGINE_HELPER_BASE_DEFAULT"
    fi
    printf '%s' "${hb%/}"
}

# Resolve a helper script (probe.sh, probe_gpu_quality.sh) — either
# co-located on disk next to this wrapper (disk install) OR fetched once
# from the release URL into $SPARROW_ENGINE_HELPER_CACHE (piped install via
# `curl ... | sh` / `bash <(curl ...)`). Phase E B-01 fix: piped invocation
# previously failed with "probe.sh not found at /dev/fd/probe.sh" because
# dirname "$0" pointed at the process-substitution fd, not the source repo.
# The fetch path uses release_base() which honors SPARROW_ENGINE_RELEASE_BASE
# — no manual env-var setup required for the default GH Releases URL.
# Prints the absolute resolved path to stdout.
locate_helper() {
    local name=$1
    local local_path cache_path hb url ua self self_base
    # E-R2-2 fix: only consult the on-disk adjacent file when $0 points at a
    # real file path. Under piped invocation (`curl ... | sh`,
    # `bash <(curl ...)`) $0 is the shell name (`bash`, `/bin/sh`, `-bash`)
    # and dirname "$0" is `.` — sourcing `./probe.sh` from the user's cwd
    # would execute an unrelated / stale / hostile script. Skip the disk
    # leg entirely in that case and go straight to cache/fetch.
    self=$0
    if [ -f "$self" ]; then
        case "$(basename -- "$self")" in
            bash|sh|-bash|-sh|dash|zsh)
                # Defensive: a sourced wrapper can leave $0 = bash but still
                # be a real file. Treat as piped.
                self_base=""
                ;;
            *)
                self_base="$(dirname -- "$self")"
                ;;
        esac
    else
        self_base=""
    fi
    if [ -n "$self_base" ]; then
        local_path="$self_base/$name"
        if [ -f "$local_path" ]; then
            printf '%s' "$local_path"
            return 0
        fi
    fi
    cache_path="$SPARROW_ENGINE_HELPER_CACHE/$name"
    if [ -f "$cache_path" ]; then
        printf '%s' "$cache_path"
        return 0
    fi
    if ! command -v curl >/dev/null 2>&1; then
        die 8 "$name not found on disk and curl unavailable for fallback fetch"
    fi
    hb=$(helper_base)
    url="$hb/$name"
    ua="sparrow-engine-install/${SPARROW_ENGINE_VERSION} (${OS:-unknown}/${ARCH:-unknown})"
    mkdir -p "$SPARROW_ENGINE_HELPER_CACHE"
    say "fetching $name from $url" >&2
    if ! curl -fsSL -A "$ua" --connect-timeout 10 --max-time 60 \
              -o "$cache_path.tmp" "$url"; then
        rm -f "$cache_path.tmp"
        die 4 "failed to fetch $name from $url (piped install fallback; download install.sh + probe.sh + probe_gpu_quality.sh from the same tag and run from disk if your network blocks raw.githubusercontent.com)"
    fi
    mv "$cache_path.tmp" "$cache_path"
    printf '%s' "$cache_path"
}

# Atomic-ish HTTP fetch with retry. Permanent-4xx aborts; transient retries.
download_with_retry() {
    local url=$1
    local out=$2
    local attempt=1
    local delay=1
    local ua
    ua="sparrow-engine-install/${SPARROW_ENGINE_VERSION} (${OS}/${ARCH})"

    if ! command -v curl >/dev/null 2>&1; then
        die 8 "curl not found on PATH (required for --cli mode)"
    fi

    while [ "$attempt" -le "$retries" ]; do
        local code rc
        # `-fL` follow redirects, fail on 4xx/5xx; `-w` capture status code.
        # `--retry 0` to keep retry policy in our hands.
        code=$(curl -fL -A "$ua" --connect-timeout 10 --max-time 600 \
                    --retry 0 \
                    -o "$out" -w '%{http_code}' \
                    "$url" 2>/dev/null) || rc=$?
        rc=${rc:-0}
        if [ "$rc" -eq 0 ]; then
            return 0
        fi

        case "$rc" in
            22)
                # HTTP 4xx/5xx via -f. Permanent-4xx → no retry.
                case "$code" in
                    401|403|404|410|451)
                        die 4 "fetch $url returned HTTP $code (permanent; no retry)"
                        ;;
                    408|429|5*) : ;; # transient — retry
                    *) die 4 "fetch $url returned HTTP $code" ;;
                esac
                ;;
            6|7|28|35|52|55|56) : ;; # DNS/connect/timeout/SSL — retry
            *) die 4 "curl exit $rc on $url" ;;
        esac

        if [ "$attempt" -lt "$retries" ]; then
            warn "retry ${attempt}/${retries} in ${delay}s ($url)"
            sleep "$delay"
            delay=$((delay * 2))
        fi
        attempt=$((attempt + 1))
    done
    die 4 "exhausted retries fetching $url"
}

verify_sha256() {
    local file=$1
    local sidecar=$2
    if ! command -v sha256sum >/dev/null 2>&1 && ! command -v shasum >/dev/null 2>&1; then
        die 8 "no sha256 tool found (need sha256sum or shasum)"
    fi
    local expected actual
    expected=$(awk '{print $1}' "$sidecar")
    if command -v sha256sum >/dev/null 2>&1; then
        actual=$(sha256sum -b "$file" | awk '{print $1}')
    else
        actual=$(shasum -a 256 -b "$file" | awk '{print $1}')
    fi
    if [ "$expected" != "$actual" ]; then
        die 6 "sha256 mismatch on $file: expected $expected, got $actual (artifact corrupt or tampered; not retrying)"
    fi
}

install_cli_binary() {
    local cli_flavor=$1
    local rb tarball sha_file url sha_url stage
    rb=$(release_base)

    # Flavor / OS gating: macOS-GPU is unsupported (no CUDA on macOS).
    if [ "$cli_flavor" = "gpu" ] && [ "$OS" = "macos" ]; then
        die 9 "GPU flavor not supported on macOS (no CUDA on macOS)"
    fi

    tarball="sparrow-engine-${cli_flavor}-${SPARROW_ENGINE_VERSION}-${OS}-${ARCH}.tar.gz"
    sha_file="${tarball}.sha256"
    url="${rb}/${tarball}"
    sha_url="${rb}/${sha_file}"

    if [ "$dry_run" -eq 1 ]; then
        say "[dry-run] would fetch $url"
        say "[dry-run] would verify $sha_url"
        say "[dry-run] would extract into $SPARROW_ENGINE_PREFIX"
        return 0
    fi

    if ! command -v tar >/dev/null 2>&1; then
        die 8 "tar not found on PATH (required for --cli mode)"
    fi

    mkdir -p "$SPARROW_ENGINE_PREFIX/.staging"
    stage=$(mktemp -d -p "$SPARROW_ENGINE_PREFIX/.staging" "sparrow-engine-${cli_flavor}.XXXXXX")

    say "downloading $url"
    download_with_retry "$url" "$stage/$tarball"
    say "downloading $sha_url"
    download_with_retry "$sha_url" "$stage/$sha_file"

    say "verifying sha256"
    verify_sha256 "$stage/$tarball" "$stage/$sha_file"

    say "extracting"
    ensure tar -xzf "$stage/$tarball" -C "$stage"

    # The tarball roots at sparrow-engine-{flavor}-{ver}-{os}-{arch}/ per §2.6.
    local root
    root="$stage/sparrow-engine-${cli_flavor}-${SPARROW_ENGINE_VERSION}-${OS}-${ARCH}"
    if [ ! -d "$root" ]; then
        die 1 "extracted tarball missing expected root: $root"
    fi

    # Atomic-ish swap: backup existing dirs to .bak-<prev>, then mv from staging.
    local prev
    prev=$(state_read_flavor 2>/dev/null || true)
    if [ -d "$SPARROW_ENGINE_PREFIX/bin" ] || [ -d "$SPARROW_ENGINE_PREFIX/lib" ]; then
        local backup_dir
        backup_dir="$SPARROW_ENGINE_PREFIX/.bak-${prev:-prev}"
        rm -rf "$backup_dir"
        mkdir -p "$backup_dir"
        for sub in bin lib include share; do
            if [ -d "$SPARROW_ENGINE_PREFIX/$sub" ]; then
                mv "$SPARROW_ENGINE_PREFIX/$sub" "$backup_dir/$sub"
            fi
        done
    fi
    for sub in bin lib include share; do
        if [ -d "$root/$sub" ]; then
            mv "$root/$sub" "$SPARROW_ENGINE_PREFIX/$sub"
        fi
    done

    # Cleanup staging.
    rm -rf "$stage"
    rmdir "$SPARROW_ENGINE_PREFIX/.staging" 2>/dev/null || true
}

# -----------------------------------------------------------------------------
# docker mode
# -----------------------------------------------------------------------------
install_docker_image() {
    local docker_flavor=$1
    if ! command -v docker >/dev/null 2>&1; then
        die 8 "docker not found on PATH (required for --docker mode)"
    fi
    local image
    if [ "$docker_flavor" = "cpu" ]; then
        image="zhongqimiao/sparrow-engine-server:latest"
    else
        image="zhongqimiao/sparrow-engine-server-gpu:latest"
    fi
    if [ "$dry_run" -eq 1 ]; then
        say "[dry-run] would run: docker pull $image"
        return 0
    fi
    say "pulling $image"
    ensure docker pull "$image"
}

# -----------------------------------------------------------------------------
# rc-file edits (idempotent; conda-style sentinel)
# -----------------------------------------------------------------------------
update_rc_files() {
    local rc_mode=${1:-}
    # F-R8: only --cli mode adds binaries to PATH; pip + docker modes do
    # NOT need a sentinel-block rc-file edit. Mirrors sparrow-engine-install.ps1::
    # Update-ProfileRc which gates on `$ResolvedMode -eq 'cli'`.
    if [ "$rc_mode" != "cli" ]; then
        return 0
    fi
    if [ "${SPARROW_ENGINE_NO_MODIFY_PATH:-0}" = "1" ]; then
        say "SPARROW_ENGINE_NO_MODIFY_PATH=1; skipping rc-file edit"
        return 0
    fi
    if [ "$dry_run" -eq 1 ]; then
        say "[dry-run] would edit ~/.bashrc, ~/.zshrc, ~/.bash_profile, ~/.profile (only existing) with sentinel block"
        return 0
    fi

    # Asymmetric semantic (D-5 / Inq §4 LOW-3):
    #   - ~/.bashrc       : ALWAYS create-if-missing. Bash on Linux is the
    #                        default shell on most distros; if the user has
    #                        no ~/.bashrc yet, create one so PATH is updated.
    #   - other rc files  : only edit if pre-existing. Don't materialize a
    #                        zsh/login-shell config for users who don't use
    #                        those shells. Idempotent if the user later
    #                        creates one (next install round picks it up).
    edit_rc "$HOME/.bashrc" 1
    if [ -f "$HOME/.zshrc" ]; then
        edit_rc "$HOME/.zshrc" 0
    fi
    # macOS Homebrew users + `bash --login` shells edit ~/.bash_profile rather
    # than ~/.bashrc. POSIX login shells (sh, dash) read ~/.profile.
    # Both are only-if-exists per the asymmetric semantic above.
    if [ -f "$HOME/.bash_profile" ]; then
        edit_rc "$HOME/.bash_profile" 0
    fi
    if [ -f "$HOME/.profile" ]; then
        edit_rc "$HOME/.profile" 0
    fi
}

edit_rc() {
    local rc=$1
    local create_if_missing=$2
    if [ ! -f "$rc" ]; then
        if [ "$create_if_missing" -ne 1 ]; then
            return 0
        fi
        : > "$rc"
    fi

    local block
    block=$(printf '%s\nexport PATH="%s/bin:$PATH"\n%s\n' "$SENTINEL_BEGIN" "$SPARROW_ENGINE_PREFIX" "$SENTINEL_END")

    if grep -qF "$SENTINEL_BEGIN" "$rc"; then
        # Existing block — check for manual edits.
        local existing
        existing=$(awk -v b="$SENTINEL_BEGIN" -v e="$SENTINEL_END" '
            $0==b {p=1}
            p {print}
            $0==e {p=0}
        ' "$rc")
        if [ "$existing" != "$block" ] && [ "$force_rc_overwrite" -ne 1 ]; then
            die 13 "manual edits detected inside sparrow-engine block in $rc (pass --force-rc-overwrite to replace)"
        fi
        # Replace block in-place via temp file.
        local tmp
        tmp=$(mktemp "${rc}.tmp.XXXXXX")
        awk -v b="$SENTINEL_BEGIN" -v e="$SENTINEL_END" -v new="$block" '
            $0==b {print new; p=1; next}
            p && $0==e {p=0; next}
            p {next}
            {print}
        ' "$rc" > "$tmp"
        mv "$tmp" "$rc"
        say "updated sparrow-engine block in $rc"
    else
        # Append a fresh block (with leading blank line for readability).
        printf '\n%s\n' "$block" >> "$rc"
        say "appended sparrow-engine block to $rc"
    fi
}

remove_rc_block() {
    local rc=$1
    if [ ! -f "$rc" ]; then
        return 0
    fi
    if ! grep -qF "$SENTINEL_BEGIN" "$rc"; then
        return 0
    fi
    local tmp
    tmp=$(mktemp "${rc}.tmp.XXXXXX")
    awk -v b="$SENTINEL_BEGIN" -v e="$SENTINEL_END" '
        $0==b {p=1; next}
        p && $0==e {p=0; next}
        p {next}
        {print}
    ' "$rc" > "$tmp"
    mv "$tmp" "$rc"
    say "removed sparrow-engine block from $rc"
}

# -----------------------------------------------------------------------------
# Uninstall
# -----------------------------------------------------------------------------
do_uninstall_silent() {
    # Used by --reprobe cross-flavor switch; suppress chatter where possible.
    do_uninstall 1
}

do_uninstall() {
    local quiet=${1:-0}
    if [ ! -d "$SPARROW_ENGINE_PREFIX" ]; then
        if [ "$quiet" -ne 1 ]; then
            say "no install detected at $SPARROW_ENGINE_PREFIX (nothing to uninstall)"
        fi
        return 0
    fi
    if [ "$dry_run" -eq 1 ]; then
        say "[dry-run] would remove $SPARROW_ENGINE_PREFIX and rc-file blocks"
        return 0
    fi
    say "uninstalling $SPARROW_ENGINE_PREFIX"
    # Use trash-put when available (per project rule); fall back to rm with a
    # narrowed scope (only our prefix dirs) — never a free-form rm -rf $HOME.
    if command -v trash-put >/dev/null 2>&1; then
        ensure trash-put "$SPARROW_ENGINE_PREFIX"
    else
        # Defensive: refuse to remove a non-default $SPARROW_ENGINE_PREFIX without explicit
        # confirmation. The default ($HOME/.sparrow-engine) is our well-known location.
        if [ "$SPARROW_ENGINE_PREFIX" != "$HOME/.sparrow-engine" ] && [ "$yes" -ne 1 ]; then
            die 1 "non-default SPARROW_ENGINE_PREFIX ($SPARROW_ENGINE_PREFIX); pass -y to confirm rm -rf"
        fi
        rm -rf "$SPARROW_ENGINE_PREFIX"
    fi
    remove_rc_block "$HOME/.bashrc"
    remove_rc_block "$HOME/.zshrc"
    remove_rc_block "$HOME/.bash_profile"
    remove_rc_block "$HOME/.profile"
}

# -----------------------------------------------------------------------------
# Signal handling
# -----------------------------------------------------------------------------
on_interrupt() {
    err "interrupted (SIGINT/SIGTERM); aborting"
    # Cleanup staging if present.
    if [ -d "$SPARROW_ENGINE_PREFIX/.staging" ] 2>/dev/null; then
        rm -rf "$SPARROW_ENGINE_PREFIX/.staging" 2>/dev/null || true
    fi
    exit 2
}

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------
main() {
    trap on_interrupt INT TERM

    parse_args "$@"
    detect_os_arch

    # Uninstall short-circuits before any flavor work.
    if [ "$uninstall" -eq 1 ]; then
        do_uninstall 0
        exit 0
    fi

    local mode
    case "$mode_arg" in
        ""|"auto") mode=$(auto_detect_mode) ;;
        pip|cli|docker) mode="$mode_arg" ;;
        *) die 1 "unknown mode: $mode_arg" ;;
    esac

    resolve_flavor

    # --probe-only short-circuits AFTER flavor resolution but BEFORE the
    # cross-flavor refusal gate. Prints the verdict + diagnostic env vars
    # so the user can debug probe results without touching state.
    if [ "$probe_only" -eq 1 ]; then
        printf '%s\n' "$flavor"
        if [ -n "${SPARROW_ENGINE_DETECTED_PROBE_REASON:-}" ]; then
            say "probe reason: $SPARROW_ENGINE_DETECTED_PROBE_REASON"
        fi
        exit 0
    fi

    # Layer-2 quality probe FIRST — only fires when flavor=gpu. Dispatches
    # SPARROW_ENGINE_GPU_QUALITY: ok → silent; sm_warn / cudnn_warn → log + continue;
    # cudnn_err → die 11. Must run BEFORE check_existing_install: that
    # function calls do_uninstall_silent on cross-flavor `--reprobe` (line 383),
    # destroying the existing install. If the GPU quality gate fails AFTER
    # the destructive uninstall, the user is left with NO install.
    # (impl-af R3 Inq F-R3-1 fix.)
    gpu_quality_check

    check_existing_install

    say "mode=$mode flavor=$flavor os=$OS arch=$ARCH dry_run=$dry_run"

    case "$mode" in
        pip)    install_python_wheel "$flavor" ;;
        cli)    install_cli_binary "$flavor" ;;
        docker) install_docker_image "$flavor" ;;
        *)      die 1 "internal: unknown mode $mode" ;;
    esac

    record_state "$flavor" "$mode"
    update_rc_files "$mode"
    say "done. flavor=$flavor mode=$mode prefix=$SPARROW_ENGINE_PREFIX"
}

main "$@" || exit 1
