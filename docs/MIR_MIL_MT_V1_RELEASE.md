# MIR-MIL-MT V1 Release

Frozen on: 2026-07-14

Status: frozen architecture for cross-dataset evaluation

## Purpose

MIR-MIL-MT V1 is the final architecture selected after the PANDA and BRACS
experiments. It preserves the original MIR-MIL measure-state encoder and adds a
moment-token residual readout. The architecture is frozen before evaluation on
a third dataset so that cross-dataset evidence is not influenced by further
BRACS-specific tuning.

## Canonical configuration

- Config: `configs/releases/MIR_MIL_MT_V1.yaml`
- Benchmark model name: `MIR_MIL_MT_V1`
- Config SHA256: `8b83192de8be1835f382a283973277f9913c3914e812273f88d47cd4c6091d34`
- Recommended feature encoder: UNI
- Seeds: 2024, 2025, 2026
- Primary selection metric: validation macro-AUC
- Primary multi-seed prediction: equal probability averaging
- Maximum instances per slide: 4096

The release YAML is a complete configuration snapshot. It does not inherit from
`configs/MIR_MIL.yaml`, so later experimental defaults cannot silently alter the
frozen model.

## Frozen architecture

The release enables only the accepted moment-token extension:

```yaml
Model:
  potential_type: adaptive_multiscale
  num_local_routes: 12
  prototype_regularization_weight: 0.01
  moment_token_weight: 0.1
  moment_token_count: 4
  moment_token_dim: 64
  moment_token_readout_dim: 128
  moment_token_temperature: 1.0
  moment_token_dropout: 0.0
```

Evidence, multi-token, class-token, cosine, sparse-class, spatial, focus-class,
focus-sparse, ranking-memory, and distillation experiment branches are disabled
in the frozen configuration.

## Locked BRACS result

UNI features and the BRACS3 protocol produced:

| Result | Test macro-AUC |
| --- | ---: |
| MIR-MIL-MT V1, three-seed mean | `0.842568 +/- 0.009488` |
| MIR-MIL-MT V1, equal-probability seed ensemble | `0.858910` |
| MIR-MIL-MT V1, validation-selected coarse calibration | `0.859247` |
| AC-MIL comparison result | `0.852852 +/- 0.009653` |

The uncalibrated ensemble (`0.858910`) is the primary architecture result. The
calibrated result (`0.859247`) must be reported separately as a validation-selected
post-hoc operating transform, not as an architecture improvement.

No further architecture or hyperparameter decisions should be made using the
BRACS test set after this release.

## PANDA evidence

The archived original MIR-MIL comparison matrix reports:

| Feature | MIR-MIL macro-AUC | Best recorded competitor |
| --- | ---: | ---: |
| R50 | `0.8992 +/- 0.0007` | RRTMIL `0.8954 +/- 0.0022` |
| UNI | `0.9504 +/- 0.0011` | CLAM-MB `0.9428 +/- 0.0034` |

For MIR-MIL-MT V1, the completed UNI seed-2024 validation result is `0.958328`,
compared with `0.951178` for the original MIR-MIL under the same split and seed.
The gain is `+0.007150` macro-AUC.

The MIR-MIL-MT V1 PANDA three-seed run is still required before this value is
treated as a formal multi-seed result.

## Training command

Use the registered release name and change only dataset-specific fields:

```bash
mamba run -n mirmil python experiments/run_benchmark.py \
  --split <split.csv> \
  --dataset-name <dataset_name> \
  --num-classes <num_classes> \
  --log-root artifacts/<dataset_name>/mir_mil_mt_v1 \
  --models MIR_MIL_MT_V1 \
  --seeds 2024 2025 2026 \
  --epochs 30 \
  --patience 8 \
  --best-model-metric macro_auc \
  --earlystop-metric macro_auc \
  --scheduler-t-max 28 \
  --max-instances 4096 \
  --in-dim 1024 \
  --feature uni \
  --protocol frozen-cross-dataset-evaluation \
  --split-id <locked_split_id> \
  --device 0 \
  --num-workers 2 \
  --balanced \
  --wandb-mode disabled
```

Do not add `--model-option` architecture overrides for confirmatory evaluation.
Only dataset paths, dataset name, class count, input dimension, feature label,
device, and worker count may change.

## Claim boundary

A third-dataset result supports a general state-of-the-art claim only when the
same frozen architecture is compared with strong baselines under matched splits,
features, seeds, instance budgets, and metrics. PANDA has no public hidden test,
so PANDA claims must explicitly state that results use the fixed internal split
shared by all compared methods.

