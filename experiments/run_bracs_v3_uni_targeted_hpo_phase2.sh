#!/usr/bin/env bash
set -euo pipefail

cd /data15/data15_5/fanhao/projects/MIRMIL

PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
SPLIT=/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_train_val.csv
ROOT=artifacts/bracs3_v3_targeted_hpo/uni/phase1
SELECT=macro_auc_hmean_auc_class_1

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
    --best-model-metric "$SELECT" \
    --earlystop-metric "$SELECT" \
    --scheduler-t-max 28 \
    --clamp-cosine \
    --max-instances 4096 \
    --in-dim 1024 \
    --feature uni \
    --protocol bracs3-v3-uni-targeted-hpo-phase2-val-only \
    --split-id official-train-val-3class \
    --comparison-id "bracs3-uni-$variant-phase2" \
    --device 0 \
    --num-workers 2 \
    --wandb-mode disabled \
    --balanced \
    > "$output/controller.log" 2>&1
}

run_candidate 0 MIR_MIL_V3_HPO_GAP15 2026 &
run_candidate 2 MIR_MIL_V3_HPO_RT025 2026 &
run_candidate 3 MIR_MIL_V3_HPO_GAP15_RT025 2024 &
run_candidate 4 MIR_MIL_V3_HPO_GAP15_RT025 2025 &

wait
