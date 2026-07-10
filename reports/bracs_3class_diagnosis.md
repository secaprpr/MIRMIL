# BRACS 3-class diagnostic

Date: 2026-07-10

Purpose: test whether MIR-MIL underperformance on BRACS is mainly caused by the original 7-class fine-grained label space rather than feature extraction or model architecture.

This is a diagnostic task, not a 7-class SOTA experiment.

## Label mapping

The mapping follows the BRACS group labels in `bracs_source_manifest.csv`:

| 7-class label | Name | 3-class label | Group |
|---:|---|---:|---|
| 0 | N | 0 | BT |
| 1 | PB | 0 | BT |
| 2 | UDH | 0 | BT |
| 3 | FEA | 1 | AT |
| 4 | ADH | 1 | AT |
| 5 | DCIS | 2 | MT |
| 6 | IC | 2 | MT |

Generated CSVs:

- `/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_r50_split_official_train_val.csv`
- `/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_r50_split_official_full.csv`
- `/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_train_val.csv`
- `/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_full.csv`

## Class distribution

Official train/val:

| Split | BT | AT | MT |
|---|---:|---:|---:|
| train | 203 | 52 | 140 |
| val | 30 | 14 | 21 |

Official test, not opened during training:

| Split | BT | AT | MT |
|---|---:|---:|---:|
| test | 32 | 23 | 32 |

Compared with 7-class BRACS, the validation labels are less sparse: the smallest val class has 14 slides instead of 6.

## Controlled run

Config:

```text
configs/experiments/MIR_MIL_BRACS3_FROZEN_DEFAULT_UNI.yaml
```

Command:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python train_mil.py \
  --yaml_path configs/experiments/MIR_MIL_BRACS3_FROZEN_DEFAULT_UNI.yaml
```

Feature extractor: unchanged UNI pre-extracted features.

Model architecture: unchanged MIR-MIL backbone/aggregation/head except `num_classes: 3`, required by the task definition.

Official test: not opened.

## Result

Best validation result, seed2024:

- epoch: `14`
- val macro AUC: `0.866961`
- val acc: `0.800000`
- val balanced acc: `0.726984`
- val macro F1: `0.728893`

Best confusion matrix:

```text
[[29  0  1]
 [ 5  5  4]
 [ 1  2 18]]
```

## Interpretation

The 3-class result is much stronger than the corresponding 7-class UNI current-protocol run (`0.798629` val macro AUC). This supports the diagnosis that BRACS 7-class underperformance is substantially driven by fine-grained histological categories and label ambiguity.

The remaining bottleneck is the AT group: at the best epoch, AT recall is `5/14`, while BT and MT are much cleaner. This is consistent with the biological/pathological ambiguity between atypical and adjacent benign/malignant categories.

## Reviewer caveat

This result cannot be used to claim improvement on the 7-class BRACS SOTA target. It is a different label protocol. It is useful because it isolates one failure factor: the model and fixed UNI features can separate coarse BRACS groups much better than the original seven categories.

Next diagnostic step, if needed: run seeds 2025/2026 for BRACS3 to check whether the diagnosis is stable.
