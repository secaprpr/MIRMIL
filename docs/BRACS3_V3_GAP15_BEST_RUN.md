# BRACS3 V3 GAP1.5 frozen best run

This record freezes the strongest overall BRACS3 MIR-MIL result found during
the V3 architecture study. It uses UNI features and independent seeds 2024,
2025, and 2026. Seed probabilities are not ensembled.

## Frozen configuration

- Model variant: `MIR_MIL_V3_HPO_GAP15`
- Prediction path: scalar patch severity, normalized entropic bag risk,
  cumulative ordinal probabilities
- Feature: UNI, input dimension 1024
- Maximum instances: 4096
- Risk temperature: 0.5
- Decision temperature: 0.5
- Initial ordered-threshold gap: 1.5, centered at zero
- Balanced training sampler: enabled
- Epoch limit: 30
- Early-stopping patience: 8
- Checkpoint-selection metric: `macro_auc_hmean_auc_class_1`
- Seeds: 2024, 2025, 2026

The model parameters above are encoded by the
`MIR_MIL_V3_HPO_GAP15` entry in `experiments/run_benchmark.py`.

## Canonical command

On the server:

```bash
ssh -p 6000 fanhao@175.178.238.223
cd /data15/data15_5/fanhao/projects/MIRMIL
mamba activate mirmil
bash experiments/run_bracs_v3_gap15_best.sh
```

The script runs training first and then evaluates the three independently
selected best checkpoints on the official test split. Its canonical output
directory is:

```text
artifacts/bracs3_v3_gap15_frozen/uni
```

## Explicit training command

```bash
/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python \
  experiments/run_benchmark.py \
  --split /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_train_val.csv \
  --dataset-name BRACS3 \
  --num-classes 3 \
  --log-root artifacts/bracs3_v3_gap15_frozen/uni \
  --models MIR_MIL_V3_HPO_GAP15 \
  --seeds 2024 2025 2026 \
  --epochs 30 \
  --patience 8 \
  --best-model-metric macro_auc_hmean_auc_class_1 \
  --earlystop-metric macro_auc_hmean_auc_class_1 \
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
```

## Explicit official-test command

```bash
/data15/data15_5/fanhao/miniforge3/envs/mirmil/bin/python \
  experiments/evaluate_checkpoints.py \
  --run-root artifacts/bracs3_v3_gap15_frozen/uni \
  --output-dir artifacts/bracs3_v3_gap15_frozen/uni/official_test \
  --models MIR_MIL \
  --budgets 4096 \
  --device 0 \
  --num-workers 2 \
  --group test \
  --checkpoint-kind best \
  --split-override /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_full.csv \
  --wandb-mode disabled
```

## Recorded result

Official test, mean and sample standard deviation over three independent
seeds:

| Model | Accuracy | Balanced accuracy | Macro-AUC | Macro-F1 |
|---|---:|---:|---:|---:|
| MIR-MIL V3 GAP1.5 | 0.6935 +/- 0.0133 | 0.6760 +/- 0.0110 | 0.8485 +/- 0.0019 | 0.6774 +/- 0.0132 |
| UNI AC-MIL | 0.6820 +/- 0.0332 | 0.6670 +/- 0.0471 | 0.8529 +/- 0.0097 | 0.6620 +/- 0.0476 |

The development-run checkpoints and predictions that produced the recorded
MIR-MIL result are retained on the server under:

```text
artifacts/bracs3_v3_targeted_hpo/uni/phase1/MIR_MIL_V3_HPO_GAP15
```
