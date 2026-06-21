#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPARROW_ENGINE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    cat <<'EOF'
Usage: scripts/check_ffi_symbols.sh [--build]

Checks that every built sparrow-engine flavor cdylib exports the same number of
sparrow_engine_* FFI symbols as the CPU baseline. Counts are parsed at runtime;
there is no hardcoded symbol count.

Options:
  --build    Build CPU and GPU cdylibs first in isolated target dirs:
             target/cpu/ and target/gpu/
  -h, --help Show this help.
EOF
}

build_requested=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --build)
            build_requested=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "error: unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

target_dir_for() {
    local flavor="$1"
    printf '%s/target/%s' "$SPARROW_ENGINE_DIR" "$flavor"
}

cdylib_for() {
    local flavor="$1"
    printf '%s/release/libsparrow_engine.so' "$(target_dir_for "$flavor")"
}

build_flavor() {
    local flavor="$1"
    local package="$2"

    echo "[check_ffi_symbols] building $flavor cdylib in target/$flavor/"
    (
        cd "$SPARROW_ENGINE_DIR"
        CARGO_TARGET_DIR="$(target_dir_for "$flavor")" \
            cargo build -p "$package" --release --features ffi
    )
}

symbol_count() {
    local cdylib="$1"
    local nm_output
    if ! command -v nm >/dev/null 2>&1; then
        echo "error: nm not found; install binutils to inspect cdylib exports" >&2
        return 1
    fi
    if ! nm_output=$(nm -D --defined-only "$cdylib"); then
        echo "error: nm failed while reading exports from $cdylib" >&2
        return 1
    fi
    printf '%s\n' "$nm_output" | grep -c ' T sparrow_engine_' || true
}

print_build_command() {
    local flavor="$1"
    local package="$2"
    cat <<EOF
[check_ffi_symbols] $flavor cdylib not found: $(cdylib_for "$flavor")
[check_ffi_symbols] build it with:
  cd "$SPARROW_ENGINE_DIR" && CARGO_TARGET_DIR=target/$flavor cargo build -p $package --release --features ffi
EOF
}

if [[ "$build_requested" -eq 1 ]]; then
    build_flavor cpu sparrow-engine-cpu
    build_flavor gpu sparrow-engine-gpu
fi

cpu_cdylib="$(cdylib_for cpu)"
if [[ ! -f "$cpu_cdylib" ]]; then
    print_build_command cpu sparrow-engine-cpu >&2
    echo "[check_ffi_symbols] cannot establish CPU baseline N; aborting." >&2
    exit 2
fi

baseline_n="$(symbol_count "$cpu_cdylib")"
if [[ "$baseline_n" -eq 0 ]]; then
    echo "[check_ffi_symbols] CPU baseline N=0; refusing to pass an empty FFI export set." >&2
    exit 1
fi
echo "[check_ffi_symbols] CPU baseline N=$baseline_n ($cpu_cdylib)"

mismatch=0
found_flavors=("cpu")

check_optional_flavor() {
    local flavor="$1"
    local package="$2"
    local cdylib
    local count

    cdylib="$(cdylib_for "$flavor")"
    if [[ ! -f "$cdylib" ]]; then
        print_build_command "$flavor" "$package"
        echo "[check_ffi_symbols] skipping $flavor because it is not built yet."
        return 0
    fi

    found_flavors+=("$flavor")
    count="$(symbol_count "$cdylib")"
    echo "[check_ffi_symbols] $flavor count=$count ($cdylib)"
    if [[ "$count" != "$baseline_n" ]]; then
        echo "[check_ffi_symbols] mismatch: $flavor count $count != CPU baseline $baseline_n" >&2
        mismatch=1
    fi
}

check_optional_flavor gpu sparrow-engine-gpu
check_optional_flavor mobile sparrow-engine-mobile

printf '[check_ffi_symbols] found flavor cdylibs:'
printf ' %s' "${found_flavors[@]}"
printf '\n'

if [[ "$mismatch" -ne 0 ]]; then
    echo "[check_ffi_symbols] FAIL: FFI symbol counts differ." >&2
    exit 1
fi

echo "[check_ffi_symbols] PASS: all present flavor counts equal N=$baseline_n."
