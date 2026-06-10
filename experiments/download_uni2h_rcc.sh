#!/usr/bin/env bash
set -euo pipefail

local_dir="${1:-/mnt/d/datasets/UNI2-h_features}"
wait_pid="${2:-}"
hf_bin="${HF_BIN:-/home/sigirika/miniforge3/envs/pathowm/bin/hf}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
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
for project in "${projects[@]}"; do
    echo "[$(date -Is)] START $project"
    "$hf_bin" download \
        MahmoodLab/UNI2-h-features \
        "TCGA/${project}.tar.gz" \
        --repo-type dataset \
        --local-dir "$local_dir"
    echo "[$(date -Is)] DONE $project"
done
