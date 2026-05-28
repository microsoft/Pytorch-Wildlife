#!/usr/bin/env bash
# Package the `spe` / `spe-gpu` CLI into a self-contained tarball
# (RP-4 Path B, 2026-05-26).
#
# Usage:
#   FLAVOR=cpu|gpu TARBALL_PLATFORM=linux-x86_64|macos-aarch64|windows-x86_64 \
#   VERSION=X.Y.Z [OUT_DIR=dist] [BIN_DIR=target/release] \
#   ./scripts/package_cli_tarball.sh
#
# The env-var name is `TARBALL_PLATFORM` (not `PLATFORM`) because the
# `ilammy/msvc-dev-cmd@v1` GHA action injects `Platform=x64` into the runner
# environment. Windows env vars are case-insensitive, so a YAML step-level
# `env: PLATFORM: windows-x86_64` is silently coerced by GHA to
# `Platform=windows-x86_64` (lowercase-cased to match the pre-existing entry).
# MSYS Git Bash is POSIX case-sensitive, so it then sees `$Platform` but NOT
# `$PLATFORM`, and the original `${PLATFORM:?...}` validation fired. Renaming
# to `TARBALL_PLATFORM` sidesteps the collision entirely (PW RP-21).
#
# Outputs:
#   <OUT_DIR>/sparrow-engine-<FLAVOR>-<VERSION>-<TARBALL_PLATFORM>.tar.gz
#   <OUT_DIR>/sparrow-engine-<FLAVOR>-<VERSION>-<TARBALL_PLATFORM>.tar.gz.sha256
#
# Archive name matches `installer/sparrow-engine-install.sh:531` so the
# wrapper script downloads these assets unmodified from GH Releases.
#
# Expected to be called from sparrow-engine/ (workspace root) on every build.
# Idempotent: re-runs cleanly overwrite a previous output.
#
# The bundle layout is the one `ort_resolver::init_ort_env()` expects
# (RP-4 step 1 / commit cdbdb39) AND the one `installer/sparrow-engine-install.sh`
# pre-existing tarball flow expects:
#
#   sparrow-engine-<FLAVOR>-<VERSION>-<TARBALL_PLATFORM>/
#   ├── bin/spe[-gpu](.exe)
#   ├── lib/libonnxruntime.{so.X.Y.Z,dylib,dll}
#   │   (GPU adds libonnxruntime_providers_cuda.so + _providers_shared.so)
#   ├── README.md
#   └── VERSION

set -euo pipefail

# ---------------------------------------------------------------------------
# Inputs + validation
# ---------------------------------------------------------------------------

# Back-compat: accept the legacy PLATFORM env var if TARBALL_PLATFORM is unset
# (operators may have local scripts that still set PLATFORM=...). The Windows
# GHA collision documented in the header only bites the YAML env: layer; a
# direct `PLATFORM=windows-x86_64 ./scripts/package_cli_tarball.sh` shell
# invocation works fine in any POSIX shell.
: "${TARBALL_PLATFORM:=${PLATFORM:-}}"

: "${FLAVOR:?FLAVOR=cpu|gpu required}"
: "${TARBALL_PLATFORM:?TARBALL_PLATFORM=linux-x86_64|macos-aarch64|windows-x86_64 required}"

# Default VERSION from sparrow-engine-cli's Cargo.toml when the caller did
# not supply one. This consolidates the version source with the binary's
# `CARGO_PKG_VERSION` (Phase D B-03 — `spe --version` reads the same value),
# so the tarball name and the binary it ships agree by construction on
# local invocations. CI explicitly passes a tag-derived VERSION and is
# unaffected; the reviewer-added `check-version-consistency` job on
# release.yml enforces `cli_version == py_version == tag_version` on
# tag-push, so this default lines up with the gated CI value too.
#
# Documented cwd is sparrow-engine/ (workspace root); the manifest probe
# is relative to that. Skip the default silently when cargo / jq are
# unavailable (non-cargo callers fall through to the existing hard-error).
if [[ -z "${VERSION:-}" ]] \
   && command -v cargo >/dev/null 2>&1 \
   && command -v jq >/dev/null 2>&1 \
   && [[ -f sparrow-engine-cli/Cargo.toml ]]; then
  VERSION="$(cargo metadata --no-deps --format-version 1 \
               --manifest-path sparrow-engine-cli/Cargo.toml \
             | jq -r '.packages[] | select(.name=="sparrow-engine-cli") | .version')"
  if [[ -n "$VERSION" ]]; then
    echo "[package_cli_tarball.sh] VERSION defaulted from sparrow-engine-cli Cargo.toml: $VERSION"
  fi
fi

: "${VERSION:?VERSION=X.Y.Z required}"
OUT_DIR="${OUT_DIR:-dist}"
BIN_DIR="${BIN_DIR:-target/release}"

case "$FLAVOR" in
  cpu|gpu) ;;
  *) echo "FLAVOR must be cpu or gpu (got: $FLAVOR)" >&2; exit 2 ;;
esac
case "$TARBALL_PLATFORM" in
  linux-x86_64|macos-aarch64|windows-x86_64) ;;
  *) echo "TARBALL_PLATFORM must be linux-x86_64 | macos-aarch64 | windows-x86_64 (got: $TARBALL_PLATFORM)" >&2; exit 2 ;;
esac
if [[ "$FLAVOR" = "gpu" && "$TARBALL_PLATFORM" != "linux-x86_64" ]]; then
  echo "GPU flavor is linux-x86_64 only (got TARBALL_PLATFORM=$TARBALL_PLATFORM)" >&2
  exit 2
fi

# Binary name (with .exe on Windows; flavor suffix for gpu).
case "$FLAVOR" in
  cpu) bin_basename="spe" ;;
  gpu) bin_basename="spe-gpu" ;;
esac
if [[ "$TARBALL_PLATFORM" = "windows-x86_64" ]]; then
  bin_filename="${bin_basename}.exe"
  archive_ext="zip"
else
  bin_filename="$bin_basename"
  archive_ext="tar.gz"
fi

# Archive + bundle naming follows the convention `installer/sparrow-engine-install.sh`
# already expects (line 531: `sparrow-engine-${cli_flavor}-${SPARROW_ENGINE_VERSION}-${OS}-${ARCH}.tar.gz`).
# Keeping in sync means the installer can grab these GH Release assets unmodified.
bundle_name="sparrow-engine-${FLAVOR}-${VERSION}-${TARBALL_PLATFORM}"
out_archive="${OUT_DIR}/${bundle_name}.${archive_ext}"

# ---------------------------------------------------------------------------
# Locate inputs
# ---------------------------------------------------------------------------

src_bin="${BIN_DIR}/${bin_filename}"
if [[ ! -f "$src_bin" ]]; then
  echo "ERROR: binary not found at $src_bin" >&2
  echo "  Build first: cargo build --release -p sparrow-engine-cli --no-default-features --features ${FLAVOR}" >&2
  exit 3
fi

# Stage ORT runtime libs from the user's / CI's onnxruntime install.
# Source order: ORT_STAGE_DIR override, then a pip-installed onnxruntime in
# the active venv. `uv run --no-project --with onnxruntime[-gpu]==1.25.1`
# is the CI pattern.
case "$FLAVOR" in
  cpu) ort_pkg="onnxruntime" ;;
  gpu) ort_pkg="onnxruntime-gpu" ;;
esac

if [[ -n "${ORT_STAGE_DIR:-}" ]]; then
  ort_capi="$ORT_STAGE_DIR"
else
  ort_capi="$(uv run --no-project --with "${ort_pkg}==1.25.1" python -c \
    "import importlib.util, pathlib; \
spec = importlib.util.find_spec('onnxruntime'); \
print(pathlib.Path(spec.origin).parent / 'capi')")"
fi
if [[ ! -d "$ort_capi" ]]; then
  echo "ERROR: ORT capi dir not found at $ort_capi" >&2
  echo "  Set ORT_STAGE_DIR=/path/to/onnxruntime/capi to override." >&2
  exit 4
fi

# Pick the highest-versioned libonnxruntime + (GPU) provider sidecars.
case "$TARBALL_PLATFORM" in
  linux-x86_64)
    ort_dylib="$(ls -1 "$ort_capi"/libonnxruntime.so.* 2>/dev/null \
                 | grep -v '\.symlink$' | sort -V | tail -1)"
    gpu_sidecars=()
    if [[ "$FLAVOR" = "gpu" ]]; then
      for s in libonnxruntime_providers_cuda.so libonnxruntime_providers_shared.so; do
        if [[ -f "$ort_capi/$s" ]]; then
          gpu_sidecars+=("$ort_capi/$s")
        else
          echo "WARN: GPU sidecar $s missing in $ort_capi (provider load may fail at runtime)" >&2
        fi
      done
    fi
    ;;
  macos-aarch64)
    ort_dylib="$(ls -1 "$ort_capi"/libonnxruntime.*.dylib 2>/dev/null \
                 | sort -V | tail -1)"
    gpu_sidecars=()
    ;;
  windows-x86_64)
    ort_dylib="$ort_capi/onnxruntime.dll"
    [[ -f "$ort_dylib" ]] || ort_dylib=""
    gpu_sidecars=()
    ;;
esac

if [[ -z "$ort_dylib" || ! -f "$ort_dylib" ]]; then
  echo "ERROR: could not locate libonnxruntime in $ort_capi" >&2
  ls -la "$ort_capi" >&2 || true
  exit 5
fi

# ---------------------------------------------------------------------------
# Stage bundle
# ---------------------------------------------------------------------------

work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT
bundle="$work/$bundle_name"
mkdir -p "$bundle/bin" "$bundle/lib"

cp "$src_bin" "$bundle/bin/"
chmod +x "$bundle/bin/$bin_filename"
cp "$ort_dylib" "$bundle/lib/"
if ((${#gpu_sidecars[@]})); then
  cp "${gpu_sidecars[@]}" "$bundle/lib/"
fi

echo "$VERSION" > "$bundle/VERSION"

cat > "$bundle/README.md" <<EOF
# Sparrow Engine CLI — \`${bin_basename}\` v${VERSION} (${TARBALL_PLATFORM})

Self-contained tarball release per RP-4. The binary loads its bundled
\`libonnxruntime\` from \`lib/\` automatically — no extra setup required.

## Quickstart

\`\`\`
./bin/${bin_filename} --version
./bin/${bin_filename} detect --image <path-to-image>
\`\`\`

## Layout

\`\`\`
${bundle_name}/
├── bin/${bin_filename}
├── lib/
$(ls "$bundle/lib" | sed 's/^/│   ├── /')
├── VERSION
└── README.md
\`\`\`

EOF

if [[ "$FLAVOR" = "gpu" ]]; then
  cat >> "$bundle/README.md" <<'EOF'
## GPU runtime requirements

The GPU bundle does NOT include cuDNN, cuBLAS, cuRAND, or cuFFT (combined
~500MB). The host must provide them via one of:

- NVIDIA driver + system CUDA toolkit (apt: `nvidia-cuda-toolkit` + `nvidia-cudnn`)
- The matching `nvidia-cudnn-cu12`, `nvidia-cublas-cu12`, `nvidia-curand-cu12`,
  `nvidia-cufft-cu12` pip packages installed in any environment whose `lib/`
  is on `LD_LIBRARY_PATH`

See `docs/user-manual.md §2.5` for the full GPU install path.
EOF
fi

# ---------------------------------------------------------------------------
# Archive + checksum
# ---------------------------------------------------------------------------

mkdir -p "$OUT_DIR"
# Pre-compute an absolute path so both archive arms (and the subshell-cd'd
# zip arm in particular) refer to the same target without relying on $OLDPWD
# (caller-controlled, not set inside this script) or GNU-only
# `realpath --relative-to=`.
out_archive_abs="$(cd "$OUT_DIR" && pwd)/$(basename "$out_archive")"
rm -f "$out_archive_abs" "${out_archive_abs}.sha256"

case "$archive_ext" in
  tar.gz)
    tar -C "$work" -czf "$out_archive_abs" "$bundle_name"
    ;;
  zip)
    # Prefer bsdtar (built into Windows 10+ as `tar.exe`; cross-platform on
    # macOS/Linux when available) — it auto-detects the archive format from
    # the .zip extension and is more portable than MSYS `zip`. Fall back to
    # the standalone `zip` binary if bsdtar isn't present.
    if tar --version 2>/dev/null | grep -qi bsdtar; then
      tar -a -cf "$out_archive_abs" -C "$work" "$bundle_name"
    elif command -v zip >/dev/null 2>&1; then
      ( cd "$work" && zip -qr "$out_archive_abs" "$bundle_name" )
    else
      echo "ERROR: neither bsdtar nor zip available to create $out_archive_abs" >&2
      exit 6
    fi
    ;;
esac

sha256sum "$out_archive_abs" | awk '{print $1}' > "${out_archive_abs}.sha256"

echo "OK: $out_archive_abs ($(du -h "$out_archive_abs" | cut -f1))"
echo "    sha256 $(cat "${out_archive_abs}.sha256")"
