#!/usr/bin/env bash
set -euo pipefail

local_dir="${1:-/mnt/d/datasets/UNI2-h_features}"
shift $(( $# >= 1 ? 1 : $# ))
projects=("$@")
if [[ ${#projects[@]} -eq 0 ]]; then
    projects=(TCGA-KIRC TCGA-KIRP TCGA-KICH)
fi

for project in "${projects[@]}"; do
    archive="$local_dir/TCGA/${project}.tar.gz"
    destination="$local_dir/$project"
    temporary="$local_dir/.${project}.extracting"
    if [[ -d "$destination" ]] &&
            find "$destination" -maxdepth 1 -type f -name '*.h5' -print -quit |
                grep -q .; then
        echo "[$(date -Is)] SKIP existing $destination"
        continue
    fi
    if [[ ! -s "$archive" ]]; then
        echo "Missing archive: $archive" >&2
        exit 1
    fi
    if [[ -e "$destination" || -e "$temporary" ]]; then
        echo "Incomplete extraction path already exists for $project" >&2
        exit 1
    fi

    echo "[$(date -Is)] EXTRACT $project"
    mkdir -p "$temporary"
    tar --extract --gzip --file "$archive" --directory "$temporary" \
        --no-same-owner
    count="$(
        find "$temporary" -maxdepth 1 -type f -name '*.h5' -printf '.' |
            wc -c
    )"
    if (( count == 0 )); then
        echo "No H5 files extracted from $archive" >&2
        exit 1
    fi
    mv "$temporary" "$destination"
    echo "[$(date -Is)] DONE $project ($count H5 files)"
done
