#!/usr/bin/env bash
set -euo pipefail

cd /data15/data15_5/fanhao/projects/MIRMIL

PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
SPLIT=/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_train_val.csv
FULL=/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_full.csv
ROOT=artifacts/bracs3_v3_targeted_hpo/uni/final_narrow
SELECT=macro_auc_hmean_auc_class_1

run_seed2026() {
  local gpu="$1"
  local variant="$2"
  local output="$ROOT/$variant/seed2026"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled "$PY" \
    experiments/run_benchmark.py \
    --split "$SPLIT" \
    --dataset-name BRACS3 \
    --num-classes 3 \
    --log-root "$output" \
    --models "$variant" \
    --seeds 2026 \
    --epochs 30 \
    --patience 8 \
    --best-model-metric "$SELECT" \
    --earlystop-metric "$SELECT" \
    --scheduler-t-max 28 \
    --clamp-cosine \
    --max-instances 4096 \
    --in-dim 1024 \
    --feature uni \
    --protocol bracs3-v3-uni-finalist-val-only \
    --split-id official-train-val-3class \
    --comparison-id "bracs3-uni-$variant-finalist" \
    --device 0 \
    --num-workers 2 \
    --wandb-mode disabled \
    --balanced \
    > "$output/controller.log" 2>&1
}

run_seed2026 0 MIR_MIL_V3_FINAL_GAP135 &
run_seed2026 2 MIR_MIL_V3_FINAL_GAP175 &
wait

evaluate_variant() {
  local gpu="$1"
  local variant="$2"
  local output="$ROOT/$variant/official_test"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled "$PY" \
    experiments/evaluate_checkpoints.py \
    --run-root "$ROOT/$variant" \
    --output-dir "$output" \
    --models MIR_MIL \
    --budgets 4096 \
    --device 0 \
    --num-workers 2 \
    --group test \
    --checkpoint-kind best \
    --split-override "$FULL" \
    --wandb-mode disabled \
    > "$output/controller.log" 2>&1
}

evaluate_variant 0 MIR_MIL_V3_FINAL_GAP135 &
evaluate_variant 2 MIR_MIL_V3_FINAL_GAP175 &
wait
