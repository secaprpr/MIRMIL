#!/usr/bin/env bash
set -euo pipefail

cd /data15/data15_5/fanhao/projects/MIRMIL

PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
SPLIT=/data15/data15_5/fanhao/projects/MIRMIL/artifacts/bracs3_mir_long/noearlystop200/r50/seed2024/BRACS3/MIR_MIL/time_2026-07-10-18-28_BRACS3_MIR_MIL_seed_2024/BRACS3_r50_split_official_train_val.csv
ROOT=artifacts/bracs3_v4_architecture_ablation/r50

run_candidate() {
  local gpu="$1"
  local variant="$2"
  local seed="$3"
  local output="$ROOT/$variant/seed$seed"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled "$PY" \
    experiments/run_benchmark.py \
    --split "$SPLIT" \
    --dataset-name BRACS3 \
    --num-classes 3 \
    --log-root "$output" \
    --models "$variant" \
    --seeds "$seed" \
    --epochs 30 \
    --patience 8 \
    --scheduler-t-max 28 \
    --clamp-cosine \
    --max-instances 4096 \
    --in-dim 1024 \
    --feature r50 \
    --protocol bracs3-v4-architecture-ablation-exploratory \
    --split-id official-train-val-3class \
    --comparison-id "bracs3-r50-$variant" \
    --device 0 \
    --num-workers 2 \
    --wandb-mode disabled \
    --balanced \
    > "$output/controller.log" 2>&1
}

# GPU 1 is reserved by an unrelated job. Two workers run a second seed after
# their first completes; all other candidates are independent and concurrent.
(
  run_candidate 0 MIR_MIL_V3_REACHABLE_ORDINAL 2024
  run_candidate 0 MIR_MIL_V4_RISK_BOUNDARY 2026
) &
(
  run_candidate 2 MIR_MIL_V3_REACHABLE_ORDINAL 2025
  run_candidate 2 MIR_MIL_V4_MEAN_BOUNDARY 2026
) &
run_candidate 3 MIR_MIL_V3_REACHABLE_ORDINAL 2026 &
run_candidate 4 MIR_MIL_V4_MEAN_BOUNDARY 2024 &
run_candidate 5 MIR_MIL_V4_MEAN_BOUNDARY 2025 &
run_candidate 6 MIR_MIL_V4_RISK_BOUNDARY 2024 &
run_candidate 7 MIR_MIL_V4_RISK_BOUNDARY 2025 &

wait
