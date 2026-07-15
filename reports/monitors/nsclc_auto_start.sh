#!/usr/bin/env bash
set -u

PROJECT=/data15/data15_5/fanhao/projects/MIRMIL
PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
DATA_ROOT=/data15/data15_5/fanhao/datasets/TCGA-NSCLC
DEST="$DATA_ROOT/CPathPatchFeature"
NSCLC="$DEST/nsclc"
LOG_ROOT=/data15/data15_5/fanhao/experiments/MIRMIL_NSCLC
CTRL="$LOG_ROOT/controller_logs"
RUN_LOG="$PROJECT/reports/monitors/server_codex_runs.md"
AUTO_LOG="$PROJECT/reports/monitors/nsclc_auto_start.log"
SOURCE_LOG="$DEST/logs/download_nsclc_r50_uni_patches_20260716_002213.log"
LOCKDIR="$PROJECT/reports/monitors/nsclc_auto_start.lock"
START_MARKER="$PROJECT/reports/monitors/nsclc_benchmark_started.marker"

mkdir -p "$CTRL" "$PROJECT/reports/monitors"

log() {
  echo "[$(date '+%F %T %Z')] $*" | tee -a "$AUTO_LOG"
}

append_run_log() {
  {
    echo
    echo "## $(date '+%Y-%m-%d %H:%M %Z')"
    echo
    printf '%s\n' "$@"
  } >> "$RUN_LOG"
}

sync_run_log() {
  if git diff --quiet -- "$RUN_LOG"; then
    return 0
  fi
  git add "$RUN_LOG"
  git commit -m "Log NSCLC auto-start progress" >> "$AUTO_LOG" 2>&1 || return 0
  git push >> "$AUTO_LOG" 2>&1 || true
}

if ! mkdir "$LOCKDIR" 2>/dev/null; then
  log "another nsclc_auto_start instance is running; exiting"
  exit 0
fi
trap 'rmdir "$LOCKDIR" 2>/dev/null || true' EXIT

cd "$PROJECT" || exit 1

download_done() {
  grep -R "\] done" "$DEST/logs"/download_nsclc*.log >/dev/null 2>&1
}

download_alive() {
  pgrep -f 'snapshot_download.*Dearcat/CPathPatchFeature.*nsclc' >/dev/null 2>&1
}

resume_download() {
  local ts resume_log
  ts=$(date +%Y%m%d_%H%M%S)
  resume_log="$DEST/logs/download_nsclc_resume_${ts}.log"
  log "download process exited without done marker; resuming to $resume_log"
  append_run_log \
    "- Task: NSCLC HuggingFace download resume" \
    "- Reason: download process exited without a detected done marker." \
    "- Command: \`snapshot_download(repo_id='Dearcat/CPathPatchFeature', allow_patterns=['nsclc/r50/**','nsclc/uni/**','nsclc/patches/**'])\`" \
    "- Log: \`$resume_log\`"
  sync_run_log
  DEST="$DEST" nohup "$PY" -u - <<'PY' > "$resume_log" 2>&1 &
import datetime
import os
from huggingface_hub import snapshot_download

repo = "Dearcat/CPathPatchFeature"
dest = os.environ["DEST"]
allow = ["nsclc/r50/**", "nsclc/uni/**", "nsclc/patches/**"]
print(f"[{datetime.datetime.now().isoformat()}] resume repo={repo} dest={dest} allow={allow}", flush=True)
snapshot_download(repo_id=repo, repo_type="dataset", allow_patterns=allow, local_dir=dest, max_workers=6)
print(f"[{datetime.datetime.now().isoformat()}] done", flush=True)
PY
  log "resume pid=$!"
}

validate_split_paths() {
  local split="$1"
  "$PY" - "$split" <<'PY'
import csv
import os
import sys

split = sys.argv[1]
missing = []
with open(split, newline="") as f:
    reader = csv.DictReader(f)
    path_cols = [c for c in reader.fieldnames or [] if c.endswith("_slide_path") or c.endswith("_feature_path")]
    if not path_cols:
        raise SystemExit(f"no path columns in {split}")
    for row_idx, row in enumerate(reader, start=2):
        for col in path_cols:
            path = row.get(col)
            if path and not os.path.exists(path):
                missing.append((row_idx, col, path))
                if len(missing) >= 20:
                    break
        if len(missing) >= 20:
            break
if missing:
    for item in missing:
        print("missing", *item, sep="\t")
    raise SystemExit(1)
print(f"validated {split}")
PY
}

wait_gpu_free() {
  local gpu="$1" threshold="${2:-500}"
  while true; do
    local used
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')
    if [ "${used:-999999}" -lt "$threshold" ]; then
      log "GPU $gpu free enough: ${used} MiB < ${threshold} MiB"
      return 0
    fi
    log "waiting for GPU $gpu: ${used} MiB used"
    sleep 300
  done
}

run_dry_run() {
  local feature="$1" split="$2" dataset="$3" device="$4" split_id="$5" log="$CTRL/nsclc_${feature}_dry_run_$(date +%Y%m%d_%H%M%S).log"
  log "dry-run $feature -> $log"
  "$PY" experiments/run_benchmark.py \
    --split "$split" \
    --dataset-name "$dataset" \
    --num-classes 2 \
    --log-root "$LOG_ROOT" \
    --models AB_MIL CLAM_SB_MIL CLAM_MB_MIL DS_MIL TRANS_MIL RRT_MIL WIKG_MIL AC_MIL MO_MIL MAMBA2D_MIL MIR_MIL MIR_MIL_MT_V1 \
    --seeds 2024 2025 2026 \
    --epochs 30 \
    --device "$device" \
    --num-workers 4 \
    --in-dim 1024 \
    --max-instances 4096 \
    --feature "$feature" \
    --protocol nsclc_luad_lusc \
    --split-id "$split_id" \
    --dry-run > "$log" 2>&1
}

launch_benchmark() {
  local feature="$1" split="$2" dataset="$3" gpu="$4" split_id="$5"
  local ts log status
  ts=$(date +%Y%m%d_%H%M%S)
  log="$CTRL/nsclc_${feature}_gpu${gpu}_${ts}.log"
  status="$CTRL/nsclc_${feature}_gpu${gpu}_${ts}.status"
  wait_gpu_free "$gpu" 500
  setsid bash -lc "cd '$PROJECT'; CUDA_VISIBLE_DEVICES=$gpu '$PY' -u experiments/run_benchmark.py \
    --split '$split' \
    --dataset-name '$dataset' \
    --num-classes 2 \
    --log-root '$LOG_ROOT' \
    --models AB_MIL CLAM_SB_MIL CLAM_MB_MIL DS_MIL TRANS_MIL RRT_MIL WIKG_MIL AC_MIL MO_MIL MAMBA2D_MIL MIR_MIL MIR_MIL_MT_V1 \
    --seeds 2024 2025 2026 \
    --epochs 30 \
    --device 0 \
    --num-workers 4 \
    --in-dim 1024 \
    --max-instances 4096 \
    --feature '$feature' \
    --protocol nsclc_luad_lusc \
    --split-id '$split_id'; code=\$?; echo exit_code=\$code > '$status'; exit \$code" > "$log" 2>&1 < /dev/null &
  local pid=$!
  log "launched NSCLC $feature benchmark pid=$pid gpu=$gpu log=$log status=$status"
  append_run_log \
    "- Task: NSCLC LUAD vs LUSC benchmark, $feature features" \
    "- PID/session leader: \`$pid\`" \
    "- GPU: \`$gpu\` via \`CUDA_VISIBLE_DEVICES=$gpu\`" \
    "- Command: \`experiments/run_benchmark.py --models AB_MIL CLAM_SB_MIL CLAM_MB_MIL DS_MIL TRANS_MIL RRT_MIL WIKG_MIL AC_MIL MO_MIL MAMBA2D_MIL MIR_MIL MIR_MIL_MT_V1 --seeds 2024 2025 2026 --epochs 30 --feature $feature --protocol nsclc_luad_lusc\`" \
    "- Log: \`$log\`" \
    "- Status file: \`$status\`" \
    "- Split: \`$split\`"
}

log "nsclc auto-start watcher active"

while true; do
  if [ -f "$START_MARKER" ]; then
    log "start marker exists: $START_MARKER; exiting"
    exit 0
  fi

  if download_done; then
    log "download done marker detected"
    break
  fi

  if ! download_alive; then
    resume_download
  fi

  for sub in patches r50 uni; do
    if [ -d "$NSCLC/$sub" ]; then
      files=$(find "$NSCLC/$sub" -type f | wc -l)
      size=$(du -sh "$NSCLC/$sub" 2>/dev/null | awk '{print $1}')
      log "waiting: $sub files=$files size=$size"
    else
      log "waiting: $sub missing"
    fi
  done
  sleep 300
done

R50_SPLIT="$DATA_ROOT/metadata/TCGA_NSCLC_LUAD_LUSC_R50_split.csv"
UNI_SPLIT="$DATA_ROOT/metadata/TCGA_NSCLC_LUAD_LUSC_UNI_split.csv"

validate_split_paths "$R50_SPLIT" | tee -a "$AUTO_LOG"
validate_split_paths "$UNI_SPLIT" | tee -a "$AUTO_LOG"

run_dry_run r50 "$R50_SPLIT" TCGA_NSCLC_LUAD_LUSC_R50 0 r50_v1
run_dry_run uni "$UNI_SPLIT" TCGA_NSCLC_LUAD_LUSC_UNI 0 uni_v1

date > "$START_MARKER"
append_run_log \
  "- Task: NSCLC download completed and benchmark dry-runs passed" \
  "- Marker: \`$START_MARKER\`" \
  "- R50 split validation: passed" \
  "- UNI split validation: passed" \
  "- Next: launching R50 on GPU0 and UNI on GPU1 when each GPU memory is below 500 MiB."
sync_run_log

launch_benchmark r50 "$R50_SPLIT" TCGA_NSCLC_LUAD_LUSC_R50 0 r50_v1
launch_benchmark uni "$UNI_SPLIT" TCGA_NSCLC_LUAD_LUSC_UNI 1 uni_v1
sync_run_log

log "nsclc auto-start watcher finished launching benchmarks"
