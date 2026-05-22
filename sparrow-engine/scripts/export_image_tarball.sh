#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <tag> [<output-dir>]" >&2
    exit 1
}

die() {
    echo "error: $*" >&2
    exit 1
}

export_one() {
    local flavor="$1"
    local image tarball sha_file sha256

    case "$flavor" in
        cpu) image="sparrow-engine:cpu-$tag" ;;
        gpu) image="sparrow-engine:gpu-$tag" ;;
        *) die "unsupported flavor: $flavor" ;;
    esac

    tarball="$out_dir/sparrow-engine-$flavor-$tag.tar.zst"
    sha_file="$tarball.sha256"

    docker save "$image" | zstd -19 > "$tarball"
    sha256sum "$tarball" > "$sha_file"
    sha256="$(awk '{print $1}' "$sha_file")"

    printf -v "${flavor}_tarball" '%s' "$tarball"
    printf -v "${flavor}_sha_file" '%s' "$sha_file"
    printf -v "${flavor}_sha256" '%s' "$sha256"
}

tag="${1:-}"
[[ -n "$tag" ]] || usage
[[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || die "tag must match [A-Za-z0-9._-]+"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/../.." rev-parse --show-toplevel 2>/dev/null)" || die "script must run inside the sparrow-engine-dev repository"
cd "$repo_root"

out_dir="${2:-$repo_root/sparrow-engine/dist/$tag}"
if [[ "$out_dir" != /* ]]; then
    out_dir="$repo_root/$out_dir"
fi
mkdir -p "$out_dir"

export_one cpu
export_one gpu

printf 'CPU tarball: %s (%s)\n' "$cpu_tarball" "$cpu_sha256"
printf 'CPU sidecar: %s (%s)\n' "$cpu_sha_file" "$cpu_sha256"
printf 'GPU tarball: %s (%s)\n' "$gpu_tarball" "$gpu_sha256"
printf 'GPU sidecar: %s (%s)\n' "$gpu_sha_file" "$gpu_sha256"
