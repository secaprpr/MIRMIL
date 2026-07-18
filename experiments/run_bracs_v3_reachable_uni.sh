#!/usr/bin/env bash
set -euo pipefail

cd /data15/data15_5/fanhao/projects/MIRMIL

PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
SPLIT=/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_train_val.csv
FULL=/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_full.csv
ROOT=artifacts/bracs3_v4_architecture_ablation/uni/MIR_MIL_V3_REACHABLE_ORDINAL

run_seed() {
  local gpu="$1"
  local seed="$2"
  local output="$ROOT/seed$seed"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled "$PY" \
    experiments/run_benchmark.py \
    --split "$SPLIT" \
    --dataset-name BRACS3 \
    --num-classes 3 \
    --log-root "$output" \
    --models MIR_MIL_V3_REACHABLE_ORDINAL \
    --seeds "$seed" \
    --epochs 30 \
    --patience 8 \
    --scheduler-t-max 28 \
    --clamp-cosine \
    --max-instances 4096 \
    --in-dim 1024 \
    --feature uni \
    --protocol bracs3-v3-reachable-uni-exploratory \
    --split-id official-train-val-3class \
    --comparison-id bracs3-uni-MIR_MIL_V3_REACHABLE_ORDINAL \
    --device 0 \
    --num-workers 2 \
    --wandb-mode disabled \
    --balanced \
    > "$output/controller.log" 2>&1
}

run_seed 0 2024 &
run_seed 2 2025 &
run_seed 3 2026 &
wait

mkdir -p "$ROOT/official_test"
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=disabled "$PY" \
  experiments/evaluate_checkpoints.py \
  --run-root "$ROOT" \
  --output-dir "$ROOT/official_test" \
  --models MIR_MIL \
  --budgets 4096 \
  --device 0 \
  --num-workers 2 \
  --group test \
  --checkpoint-kind best \
  --split-override "$FULL" \
  --wandb-mode disabled \
  > "$ROOT/official_test/controller.log" 2>&1
