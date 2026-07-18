#!/usr/bin/env bash
set -euo pipefail

cd /data15/data15_5/fanhao/projects/MIRMIL

PY=/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python
MODEL=MIR_MIL_V3_HPO_GAP15
PANDA_META=/data15/data15_5/fanhao/datasets/PANDA/MIRMIL_FEATURES/metadata
NSCLC_META=/data15/data15_5/fanhao/datasets/TCGA-NSCLC/metadata
PANDA_ROOT=artifacts/v3_gap15_generalization/panda
NSCLC_ROOT=artifacts/v3_gap15_generalization/nsclc

run_panda_seed() {
  local gpu="$1"
  local feature="$2"
  local seed="$3"
  local split="$PANDA_META/PANDA_${feature}_split_v1_train_val.csv"
  local output="$PANDA_ROOT/$feature/seed$seed"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled "$PY" \
    experiments/run_benchmark.py \
    --split "$split" \
    --dataset-name PANDA \
    --num-classes 6 \
    --log-root "$output" \
    --models "$MODEL" \
    --seeds "$seed" \
    --epochs 30 \
    --patience 8 \
    --best-model-metric macro_auc \
    --earlystop-metric macro_auc \
    --scheduler-t-max 28 \
    --clamp-cosine \
    --max-instances 4096 \
    --in-dim 1024 \
    --feature "$feature" \
    --protocol panda6-v3-gap15-frozen \
    --split-id panda-v1-train-val \
    --comparison-id "panda6-$feature-v3-gap15" \
    --device 0 \
    --num-workers 2 \
    --wandb-mode disabled \
    --balanced \
    > "$output/controller.log" 2>&1
}

run_nsclc_feature() {
  local gpu="$1"
  local feature="$2"
  local feature_upper
  feature_upper=$(printf '%s' "$feature" | tr '[:lower:]' '[:upper:]')
  local split="$NSCLC_META/TCGA_NSCLC_LUAD_LUSC_${feature_upper}_split.csv"
  local output="$NSCLC_ROOT/$feature"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled "$PY" \
    experiments/run_benchmark.py \
    --split "$split" \
    --dataset-name "TCGA_NSCLC_LUAD_LUSC_${feature_upper}" \
    --num-classes 2 \
    --log-root "$output" \
    --models "$MODEL" \
    --seeds 2024 2025 2026 \
    --epochs 30 \
    --patience 8 \
    --best-model-metric macro_auc \
    --earlystop-metric macro_auc \
    --scheduler-t-max 28 \
    --clamp-cosine \
    --max-instances 4096 \
    --in-dim 1024 \
    --feature "$feature" \
    --protocol nsclc-luad-lusc-v3-gap15-frozen \
    --split-id "nsclc-luad-lusc-$feature-v1" \
    --comparison-id "nsclc-$feature-v3-gap15" \
    --device 0 \
    --num-workers 2 \
    --wandb-mode disabled \
    --balanced \
    > "$output/controller.log" 2>&1
}

# PANDA is the long-running six-class experiment: one seed per GPU.
run_panda_seed 0 r50 2024 &
run_panda_seed 1 r50 2025 &
run_panda_seed 2 r50 2026 &
run_panda_seed 3 uni 2024 &
run_panda_seed 4 uni 2025 &
run_panda_seed 5 uni 2026 &

# NSCLC is smaller: each feature runs its three seeds sequentially.
run_nsclc_feature 6 r50 &
run_nsclc_feature 7 uni &
wait

evaluate_group() {
  local gpu="$1"
  local run_root="$2"
  local full_split="$3"
  local output="$run_root/official_test"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" WANDB_MODE=disabled "$PY" \
    experiments/evaluate_checkpoints.py \
    --run-root "$run_root" \
    --output-dir "$output" \
    --models MIR_MIL \
    --budgets 4096 \
    --device 0 \
    --num-workers 2 \
    --group test \
    --checkpoint-kind best \
    --split-override "$full_split" \
    --wandb-mode disabled \
    > "$output/controller.log" 2>&1
}

evaluate_group 0 "$PANDA_ROOT/r50" \
  "$PANDA_META/PANDA_r50_split_v1_full_qc.csv" &
evaluate_group 1 "$PANDA_ROOT/uni" \
  "$PANDA_META/PANDA_uni_split_v1_full_qc.csv" &
evaluate_group 2 "$NSCLC_ROOT/r50" \
  "$NSCLC_META/TCGA_NSCLC_LUAD_LUSC_R50_split.csv" &
evaluate_group 3 "$NSCLC_ROOT/uni" \
  "$NSCLC_META/TCGA_NSCLC_LUAD_LUSC_UNI_split.csv" &
wait
