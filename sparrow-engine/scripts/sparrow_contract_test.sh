#!/usr/bin/env bash
# sparrow_contract_test_v1_r2
# Reverse-contract gate for Sparrow's HTTP client surface.
# This script prefers Sparrow's migrated models/<id>/manifest.toml tree, falls
# back to Sparrow's legacy top-level *_manifest.toml files, then finally falls
# back to repo-local test_files fixtures. The staged /models tree uses hard-link
# copies instead of host-path symlinks so the container can resolve the ONNX
# bytes through its single /models bind mount. If no compatible detector asset
# is available, the gate fails.
set -euo pipefail

usage() {
    echo "Usage: $0 <tag>" >&2
    exit 1
}

die() {
    echo "error: $*" >&2
    exit 1
}

require_cmd() {
    command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

manifest_value() {
    local manifest="$1"
    local section="$2"
    local key="$3"
    python3 - "$manifest" "$section" "$key" <<'PY'
import sys
import tomllib

manifest_path, section, key = sys.argv[1:4]
with open(manifest_path, 'rb') as fh:
    data = tomllib.load(fh)
value = data.get(section, {}).get(key, '')
if isinstance(value, str):
    print(value)
PY
}

manifest_assets_complete() {
    local manifest="$1"
    local manifest_dir="$(dirname "$manifest")"
    local model_rel labels_rel model_fp16_rel

    model_rel="$(manifest_value "$manifest" model file 2>/dev/null || true)"
    labels_rel="$(manifest_value "$manifest" labels file 2>/dev/null || true)"
    model_fp16_rel="$(manifest_value "$manifest" model file_fp16 2>/dev/null || true)"

    [[ -n "$model_rel" && -f "$manifest_dir/$model_rel" ]] || return 1
    [[ -z "$labels_rel" || -f "$manifest_dir/$labels_rel" ]] || return 1
    [[ -z "$model_fp16_rel" || -f "$manifest_dir/$model_fp16_rel" ]] || return 1
}

manifest_supported_for_kind() {
    local manifest="$1"
    local kind="$2"
    local method layout

    method="$(manifest_value "$manifest" postprocessing method 2>/dev/null || true)"
    layout="$(manifest_value "$manifest" preprocessing layout 2>/dev/null || true)"

    if [[ -n "$layout" && "$layout" != "nchw" ]]; then
        return 1
    fi

    case "$kind" in
        detector)
            [[ "$method" == "yolo" || "$method" == "yolo_e2e" || "$method" == "yolo_v6" || "$method" == "owl_t" ]]
            ;;
        classifier)
            [[ "$method" == "softmax" ]]
            ;;
        *)
            return 1
            ;;
    esac
}

mirror_dir_fast() {
    local src_dir="$1"
    local dst_dir="$2"

    rm -rf "$dst_dir"
    mkdir -p "$dst_dir"
    if ! cp -al "$src_dir/." "$dst_dir/" 2>/dev/null; then
        rm -rf "$dst_dir"
        mkdir -p "$dst_dir"
        cp -a "$src_dir/." "$dst_dir/"
    fi
}

rewrite_manifest_paths() {
    local manifest_path="$1"
    local old_model_rel="$2"
    local new_model_rel="$3"
    local old_labels_rel="$4"
    local new_labels_rel="$5"
    local old_fp16_rel="${6:-}"
    local new_fp16_rel="${7:-}"

    python3 - "$manifest_path" "$old_model_rel" "$new_model_rel" "$old_labels_rel" "$new_labels_rel" "$old_fp16_rel" "$new_fp16_rel" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
text = path.read_text()
replacements = [
    (sys.argv[2], sys.argv[3]),
    (sys.argv[4], sys.argv[5]),
]
if sys.argv[6]:
    replacements.append((sys.argv[6], sys.argv[7]))
for old, new in replacements:
    text = text.replace(f'"{old}"', f'"{new}"')
path.write_text(text)
PY
}

stage_flat_sparrow_manifest() {
    local manifest_src="$1"
    local model_root="$(dirname "$manifest_src")"
    local model_id model_rel labels_rel model_fp16_rel source_dir dst_dir

    manifest_supported_for_kind "$manifest_src" "$2" || return 1

    model_id="$(manifest_value "$manifest_src" model id 2>/dev/null || true)"
    model_rel="$(manifest_value "$manifest_src" model file 2>/dev/null || true)"
    labels_rel="$(manifest_value "$manifest_src" labels file 2>/dev/null || true)"
    model_fp16_rel="$(manifest_value "$manifest_src" model file_fp16 2>/dev/null || true)"

    [[ -n "$model_id" && -n "$model_rel" && -n "$labels_rel" ]] || return 1
    [[ "$model_rel" == */* && "$labels_rel" == */* ]] || return 1
    [[ -f "$model_root/$model_rel" && -f "$model_root/$labels_rel" ]] || return 1
    [[ -z "$model_fp16_rel" || -f "$model_root/$model_fp16_rel" ]] || return 1

    source_dir="$model_root/${model_rel%%/*}"
    [[ -d "$source_dir" ]] || return 1

    dst_dir="$contract_models/$model_id"
    mirror_dir_fast "$source_dir" "$dst_dir"
    cp "$manifest_src" "$dst_dir/manifest.toml"
    rewrite_manifest_paths \
        "$dst_dir/manifest.toml" \
        "$model_rel" "${model_rel#*/}" \
        "$labels_rel" "${labels_rel#*/}" \
        "$model_fp16_rel" "${model_fp16_rel#*/}"

    printf '%s\n' "$model_id"
}

stage_manifest_dir() {
    local source_dir="$1"
    local manifest="$source_dir/manifest.toml"
    local model_id

    [[ -f "$manifest" ]] || return 1
    manifest_assets_complete "$manifest" || return 1
    manifest_supported_for_kind "$manifest" "$2" || return 1

    model_id="$(manifest_value "$manifest" model id 2>/dev/null || true)"
    [[ -n "$model_id" ]] || return 1

    mirror_dir_fast "$source_dir" "$contract_models/$model_id"
    printf '%s\n' "$model_id"
}

stage_from_sparrow_root() {
    local root="$1"
    local kind="$2"
    local preferred_id="${3:-}"
    local manifest candidate_dir model_id

    [[ -d "$root" ]] || return 1

    if [[ -n "$preferred_id" ]]; then
        candidate_dir="$root/$preferred_id"
        if [[ -d "$candidate_dir" ]]; then
            model_id="$(stage_manifest_dir "$candidate_dir" "$kind" || true)"
            if [[ -n "$model_id" ]]; then
                printf '%s\t%s\n' "$model_id" "$candidate_dir/manifest.toml"
                return 0
            fi
        fi

        manifest="$root/${preferred_id}_manifest.toml"
        if [[ -f "$manifest" ]]; then
            model_id="$(stage_flat_sparrow_manifest "$manifest" "$kind" || true)"
            if [[ -n "$model_id" ]]; then
                printf '%s\t%s\n' "$model_id" "$manifest"
                return 0
            fi
        fi
    fi

    while IFS= read -r manifest; do
        candidate_dir="$(dirname "$manifest")"
        if [[ -n "$preferred_id" && "$candidate_dir" == "$root/$preferred_id" ]]; then
            continue
        fi
        model_id="$(stage_manifest_dir "$candidate_dir" "$kind" || true)"
        if [[ -n "$model_id" ]]; then
            printf '%s\t%s\n' "$model_id" "$manifest"
            return 0
        fi
    done < <(find "$root" -mindepth 2 -maxdepth 2 -type f -name manifest.toml | sort)

    while IFS= read -r manifest; do
        if [[ -n "$preferred_id" && "$manifest" == "$root/${preferred_id}_manifest.toml" ]]; then
            continue
        fi
        model_id="$(stage_flat_sparrow_manifest "$manifest" "$kind" || true)"
        if [[ -n "$model_id" ]]; then
            printf '%s\t%s\n' "$model_id" "$manifest"
            return 0
        fi
    done < <(find "$root" -maxdepth 1 -type f -name '*_manifest.toml' | sort)

    return 1
}

find_repo_manifest_dir() {
    local kind="$1"
    local root manifest

    for root in "$repo_root/test_files" "$repo_root/sparrow-engine/test_files"; do
        [[ -d "$root" ]] || continue
        while IFS= read -r manifest; do
            if manifest_supported_for_kind "$manifest" "$kind" && manifest_assets_complete "$manifest"; then
                dirname "$manifest"
                return 0
            fi
        done < <(find "$root" -type f -name manifest.toml | sort)
    done

    return 1
}

pick_port() {
    local port
    for port in $(seq 18095 18120); do
        if python3 - "$port" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    sock.bind(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    sock.close()
PY
        then
            printf '%s\n' "$port"
            return 0
        fi
    done

    return 1
}

generate_test_jpeg() {
    local output_path="$1"

    if python3 - "$output_path" <<'PY'
from pathlib import Path
import sys

try:
    from PIL import Image
except ModuleNotFoundError:
    raise SystemExit(1)

path = Path(sys.argv[1])
path.parent.mkdir(parents=True, exist_ok=True)
Image.new('RGB', (40, 40), color=(128, 128, 128)).save(path, format='JPEG')
PY
    then
        return 0
    fi

    if command -v convert >/dev/null 2>&1; then
        convert -size 40x40 xc:gray "$output_path"
        return 0
    fi

    die "neither Pillow nor ImageMagick 'convert' is available to generate the test JPEG"
}

dump_container_logs() {
    docker logs "$container_name" >&2 || true
}

wait_for_health() {
    local deadline=$((SECONDS + 60))
    while (( SECONDS < deadline )); do
        if curl -fsS "http://127.0.0.1:$port/v1/health" >/dev/null; then
            return 0
        fi
        sleep 2
    done

    dump_container_logs
    die "sparrow-engine-server health endpoint did not become ready within 60 seconds"
}

validate_detect_response() {
    local json_path="$1"
    jq -e '
        (.detections | type == "array") and
        ((.detections | length) == 0 or ((.detections[0] | has("bbox")) and (.detections[0] | has("label")) and ((.detections[0] | has("confidence")) or (.detections[0] | has("score")))))
    ' "$json_path" >/dev/null
}

validate_classify_response() {
    local json_path="$1"
    jq -e '
        if has("top") then
            (.top | type == "array") and
            ((.top | length) == 0 or ((.top[0] | has("label")) and ((.top[0] | has("confidence")) or (.top[0] | has("score")))))
        elif has("classifications") then
            (.classifications | type == "array") and
            ((.classifications | length) == 0 or ((.classifications[0] | has("label")) and ((.classifications[0] | has("confidence")) or (.classifications[0] | has("score")))))
        else
            false
        end
    ' "$json_path" >/dev/null
}

main() {
    tag="${1:-}"
    [[ -n "$tag" ]] || usage
    [[ "$tag" =~ ^[A-Za-z0-9._-]+$ ]] || die "tag must match [A-Za-z0-9._-]+"

    require_cmd docker
    require_cmd curl
    require_cmd jq
    require_cmd python3

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    repo_root="$(git -C "$script_dir/../.." rev-parse --show-toplevel 2>/dev/null)" || die "script must run inside the sparrow-engine-dev repository"
    cd "$repo_root"

    scratch_root="$repo_root/scratch"
    contract_models="$scratch_root/contract-models"
    jpeg_path="$scratch_root/contract-test.jpg"
    detect_json="$scratch_root/detect-response.json"
    classify_json="$scratch_root/classify-response.json"
    container_name="sparrow-engine-contract-test"
    sparrow_models_root="${SPARROW_MODELS_ROOT:-/home/miao/repos/PW_refactor/sparrow/models}"
    mkdir -p "$scratch_root"
    rm -rf "$contract_models"
    mkdir -p "$contract_models"

    port="$(pick_port)" || die "failed to find a free host port in the 18095-18120 range"

    detect_model_id=""
    classify_model_id=""
    detector_source=""
    classifier_source=""

    if [[ -d "$sparrow_models_root" ]]; then
        IFS=$'\t' read -r detect_model_id detector_source < <(stage_from_sparrow_root "$sparrow_models_root" detector MDV6-yolov10-c || true) || true
    fi

    if [[ -z "$detect_model_id" ]]; then
        for candidate in "$repo_root/test_files/MDV6-yolov10-c" "$repo_root/sparrow-engine/test_files/MDV6-yolov10-c"; do
            if [[ -d "$candidate" ]]; then
                detect_model_id="$(stage_manifest_dir "$candidate" detector || true)"
                if [[ -n "$detect_model_id" ]]; then
                    detector_source="$candidate/manifest.toml"
                    break
                fi
            fi
        done
    fi

    if [[ -z "$detect_model_id" ]]; then
        alt_detector_dir="$(find_repo_manifest_dir detector || true)"
        if [[ -n "$alt_detector_dir" ]]; then
            detect_model_id="$(stage_manifest_dir "$alt_detector_dir" detector || true)"
            if [[ -n "$detect_model_id" ]]; then
                detector_source="$alt_detector_dir/manifest.toml"
            fi
        fi
    fi

    [[ -n "$detect_model_id" ]] || die "no compatible detector model asset is available under $sparrow_models_root or repo-local test_files; the contract test cannot run"

    if [[ -d "$sparrow_models_root" ]]; then
        IFS=$'\t' read -r classify_model_id classifier_source < <(stage_from_sparrow_root "$sparrow_models_root" classifier || true) || true
    fi

    if [[ -z "$classify_model_id" ]]; then
        alt_classifier_dir="$(find_repo_manifest_dir classifier || true)"
        if [[ -n "$alt_classifier_dir" ]]; then
            classify_model_id="$(stage_manifest_dir "$alt_classifier_dir" classifier || true)"
            if [[ -n "$classify_model_id" ]]; then
                classifier_source="$alt_classifier_dir/manifest.toml"
            fi
        fi
    fi

    preload_csv="$detect_model_id"
    if [[ -n "$classify_model_id" && "$classify_model_id" != "$detect_model_id" ]]; then
        preload_csv+=",$classify_model_id"
    fi

    cleanup() {
        docker stop "$container_name" >/dev/null 2>&1 || true
    }
    trap cleanup EXIT

    docker rm -f "$container_name" >/dev/null 2>&1 || true
    generate_test_jpeg "$jpeg_path"

    docker run -d --rm \
        -p "$port:8080" \
        -e SPARROW_ENGINE_PRELOAD="$preload_csv" \
        -v "$contract_models:/models:ro" \
        --name "$container_name" \
        "sparrow-engine:cpu-$tag" >/dev/null

    wait_for_health

    curl -fsS \
        -F "image=@$jpeg_path;type=image/jpeg" \
        "http://127.0.0.1:$port/v1/detect?model=$detect_model_id&threshold=0.1" \
        > "$detect_json" || {
        dump_container_logs
        die "detect endpoint request failed for model '$detect_model_id'"
    }

    validate_detect_response "$detect_json" || {
        cat "$detect_json" >&2 || true
        dump_container_logs
        die "detect endpoint response did not match the expected JSON contract"
    }

    printf 'detect endpoint OK: model=%s source=%s port=%s\n' "$detect_model_id" "$detector_source" "$port"

    if [[ -n "$classify_model_id" ]]; then
        curl -fsS \
            -F "image=@$jpeg_path;type=image/jpeg" \
            "http://127.0.0.1:$port/v1/classify?model=$classify_model_id&top_k=1" \
            > "$classify_json" || {
            dump_container_logs
            die "classify endpoint request failed for model '$classify_model_id'"
        }

        validate_classify_response "$classify_json" || {
            cat "$classify_json" >&2 || true
            dump_container_logs
            die "classify endpoint response did not match the expected JSON contract"
        }

        printf 'classify endpoint OK: model=%s source=%s\n' "$classify_model_id" "$classifier_source"
    else
        echo 'WARN: classify endpoint not tested (no classifier in scope)'
    fi
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    main "$@"
fi
