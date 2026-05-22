#!/usr/bin/env bash
set -euo pipefail

usage() {
    echo "Usage: $0 <tag>" >&2
    exit 1
}

die() {
    echo "error: $*" >&2
    exit 1
}

extract_ort_crate_version() {
    awk '
        $0 == "[[package]]" { in_pkg = 0 }
        $0 == "name = \"ort\"" { in_pkg = 1; next }
        in_pkg && $1 == "version" {
            gsub(/"/, "", $3)
            print $3
            exit
        }
    ' sparrow-engine/Cargo.lock
}

extract_ort_runtime_version() {
    local dep_line
    dep_line=$(grep -E '^dependencies = \["onnxruntime>=' sparrow-engine/sparrow-engine-python/pyproject.toml | head -1 || true)
    [[ -n "$dep_line" ]] || die "failed to read onnxruntime dependency from sparrow-engine/sparrow-engine-python/pyproject.toml"
    sed -E 's/.*onnxruntime>=([^,"]+).*/\1/' <<<"$dep_line"
}

tag="${1:-}"
[[ -n "$tag" ]] || usage
[[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || die "tag must match [A-Za-z0-9._-]+"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(git -C "$script_dir/../.." rev-parse --show-toplevel 2>/dev/null)" || die "script must run inside the bongo_dev repository"
cd "$repo_root"

ort_crate_version="$(extract_ort_crate_version)"
[[ -n "$ort_crate_version" ]] || die "failed to parse ort crate version from sparrow-engine/Cargo.lock"
ort_runtime_version="$(extract_ort_runtime_version)"
[[ -n "$ort_runtime_version" ]] || die "failed to parse ORT runtime version from sparrow-engine/sparrow-engine-python/pyproject.toml"
source_commit="$(git rev-parse --short HEAD)"
cpu_image="sparrow-engine:cpu-$tag"
gpu_image="sparrow-engine:gpu-$tag"

docker build -t "$cpu_image" -f sparrow-engine/docker/Dockerfile.cpu sparrow-engine/
docker build -t "$gpu_image" -f sparrow-engine/docker/Dockerfile.gpu sparrow-engine/

cpu_digest="$(docker image inspect --format='{{.Id}}' "$cpu_image")"
gpu_digest="$(docker image inspect --format='{{.Id}}' "$gpu_image")"

printf 'CPU  image: %-24s @ %s\n' "$cpu_image" "$cpu_digest"
printf 'GPU  image: %-24s @ %s\n' "$gpu_image" "$gpu_digest"
printf 'sparrow_engine_source_commit: %s\n' "$source_commit"
printf 'ort_crate_version:          %s\n' "$ort_crate_version"
printf 'ort_runtime_version:        %s\n' "$ort_runtime_version"
