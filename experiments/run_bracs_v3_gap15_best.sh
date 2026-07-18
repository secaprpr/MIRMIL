#!/usr/bin/env bash
set -euo pipefail

cd /data15/data15_5/fanhao/projects/MIRMIL

PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
TRAIN_VAL=/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_train_val.csv
FULL=/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_full.csv
ROOT=artifacts/bracs3_v3_gap15_frozen/uni
SELECT=macro_auc_hmean_auc_class_1

# Frozen best BRACS3 configuration:
# entropic risk temperature=0.5, ordinal decision temperature=0.5,
# centered initial threshold gap=1.5, UNI features, 4096 instances.
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" WANDB_MODE=disabled "$PY" \
  experiments/run_benchmark.py \
  --split "$TRAIN_VAL" \
  --dataset-name BRACS3 \
  --num-classes 3 \
  --log-root "$ROOT" \
  --models MIR_MIL_V3_HPO_GAP15 \
  --seeds 2024 2025 2026 \
  --epochs 30 \
  --patience 8 \
  --best-model-metric "$SELECT" \
  --earlystop-metric "$SELECT" \
  --scheduler-t-max 28 \
  --clamp-cosine \
  --max-instances 4096 \
  --in-dim 1024 \
  --feature uni \
  --protocol bracs3-v3-gap15-frozen \
  --split-id official-train-val-3class \
  --comparison-id bracs3-uni-MIR_MIL_V3_GAP15-frozen \
  --device 0 \
  --num-workers 2 \
  --wandb-mode disabled \
  --balanced

mkdir -p "$ROOT/official_test"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" WANDB_MODE=disabled "$PY" \
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
  --wandb-mode disabled
