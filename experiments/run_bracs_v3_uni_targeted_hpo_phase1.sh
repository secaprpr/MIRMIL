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
    --protocol bracs3-v3-uni-targeted-hpo-phase1-val-only \
    --split-id official-train-val-3class \
    --comparison-id "bracs3-uni-$variant-phase1" \
    --device 0 \
    --num-workers 2 \
    --wandb-mode disabled \
    --balanced \
    > "$output/controller.log" 2>&1
}

run_pair() {
  local gpu="$1"
  local variant="$2"
  run_candidate "$gpu" "$variant" 2024
  run_candidate "$gpu" "$variant" 2025
}

run_pair 0 MIR_MIL_V3_HPO_BASE_HMEAN &
run_pair 2 MIR_MIL_V3_HPO_GAP15 &
run_pair 3 MIR_MIL_V3_HPO_GAP20 &
run_pair 4 MIR_MIL_V3_HPO_DT035 &
run_pair 5 MIR_MIL_V3_HPO_RT025 &
run_candidate 6 MIR_MIL_V3_HPO_RT100 2024 &
run_candidate 7 MIR_MIL_V3_HPO_RT100 2025 &

wait
