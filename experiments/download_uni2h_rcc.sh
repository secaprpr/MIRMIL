#!/usr/bin/env bash
set -euo pipefail

local_dir="${1:-/mnt/d/datasets/UNI2-h_features}"
wait_pid="${2:-}"
max_retries="${HF_DOWNLOAD_MAX_RETRIES:-20}"
token="${HF_TOKEN:-}"
if [[ -z "$token" ]]; then
    token="$(cat "${HF_TOKEN_PATH:-$HOME/.cache/huggingface/token}")"
fi
auth_config="$(mktemp)"
chmod 600 "$auth_config"
printf 'header = "Authorization: Bearer %s"\n' "$token" > "$auth_config"
unset token
trap 'rm -f "$auth_config"' EXIT
shift $(( $# >= 2 ? 2 : $# ))
projects=("$@")
if [[ ${#projects[@]} -eq 0 ]]; then
    projects=(TCGA-KIRC TCGA-KIRP TCGA-KICH)
fi

if [[ "$wait_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "Waiting for process $wait_pid to finish."
    while kill -0 "$wait_pid" 2>/dev/null; do
        sleep 60
    done
fi

mkdir -p "$local_dir"
mkdir -p "$local_dir/TCGA"
for project in "${projects[@]}"; do
    echo "[$(date -Is)] START $project"
    destination="$local_dir/TCGA/${project}.tar.gz"
    partial="${destination}.part"
    if [[ -s "$destination" ]]; then
        echo "[$(date -Is)] SKIP existing $destination"
        continue
    fi
    attempt=1
    until curl \
            --config "$auth_config" \
            --fail \
            --location \
            --continue-at - \
            --connect-timeout 30 \
            --retry 5 \
            --retry-all-errors \
            --retry-delay 10 \
            --output "$partial" \
            "https://huggingface.co/datasets/MahmoodLab/UNI2-h-features/resolve/main/TCGA/${project}.tar.gz"; do
        if (( attempt >= max_retries )); then
            echo "[$(date -Is)] FAILED $project after $attempt attempts"
            exit 1
        fi
        delay=$(( attempt * 30 ))
        echo "[$(date -Is)] RETRY $project attempt $((attempt + 1)) in ${delay}s"
        sleep "$delay"
        attempt=$(( attempt + 1 ))
    done
    mv "$partial" "$destination"
    echo "[$(date -Is)] DONE $project"
done
