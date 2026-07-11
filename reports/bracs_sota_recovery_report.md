# BRACS3 SOTA Recovery Report

Date: 2026-07-11

## Status

BRACS3 SOTA has not been reached.

The best current MIR-MIL result under the valid single-feature UNI protocol is:

- `UNI + MIR-MIL moment-token w01`
- BRACS3 official test macro-AUC: `0.842568 ± 0.009488`
- Current internal target/SOTA reference: `UNI + AC_MIL = 0.852852 ± 0.009653`
- Remaining gap: `0.010284` macro-AUC

This result is the best accepted MIR-MIL recovery result so far, but it is not SOTA.

## Protocol constraints respected

- BRACS official split was kept unchanged.
- R50 and UNI pre-extracted features were kept fixed.
- No feature extractor was re-trained, fine-tuned, replaced, or re-run.
- No raw WSI training was introduced.
- BRACS official test was not used for iterative tuning.
- Official test was opened only for frozen validation-selected candidates that passed the required gates.
- R50 and UNI were treated as separate feature settings; no R50+UNI fusion is claimed.

## Original BRACS baseline

Archived valid MIR-MIL BRACS3 official-test results:

| Feature | Protocol | Official test macro-AUC |
| --- | --- | ---: |
| UNI | original early-stop protocol | `0.827973 ± 0.027678` |
| UNI | noES / best-val protocol | `0.8403 ± 0.0184` |
| R50 | original early-stop protocol | `0.7570 ± 0.0120` |
| R50 | noES / best-val protocol | `0.7743 ± 0.0077` |

## Best previous BRACS result before architecture recovery

The strongest pre-recovery MIR-MIL reference was:

- `UNI + MIR-MIL`, noES / best-val: about `0.8403 ± 0.0184`

The stronger non-MIR target in the local comparison matrix was:

- `UNI + AC_MIL`: `0.852852 ± 0.009653`

## Best new BRACS result

The best new accepted MIR-MIL architecture extension is:

| Method | Feature | Validation gate | PANDA sanity | BRACS3 official test |
| --- | --- | ---: | ---: | ---: |
| MIR-MIL moment-token w01 | UNI | `0.913452 ± 0.015874` | `0.958328` | `0.842568 ± 0.009488` |

Official BRACS3 test details at budget4096:

- seed2024: `0.848053`
- seed2025: `0.848040`
- seed2026: `0.831612`
- mean/std: `0.842568 ± 0.009488`
- acc: `0.662835`
- bacc: `0.619716`
- macro-F1: `0.596231`

This improves over:

- original UNI MIR-MIL by `0.014595` macro-AUC;
- noES/best-val UNI MIR-MIL by about `0.002268` macro-AUC.

It remains below AC_MIL by `0.010284` macro-AUC.

## Whether BRACS SOTA was reached

No.

The current best MIR-MIL result is close but still below the target:

```text
0.852852 - 0.842568 = 0.010284
```

Because SOTA was not reached and the gap is similar in scale to seed variance, no SOTA claim should be made.

## Whether improvement came from R50, UNI, or both

The accepted improvement came from UNI only.

- R50 remains substantially weaker for MIR-MIL on BRACS3.
- UNI remains the primary feature setting for the current recovery path.
- No R50+UNI fusion is included or claimed.

## Which changes helped most

Accepted:

- Moment multi-token attention readout:
  - adds per-token weighted mean and variance statistics;
  - preserves the original MIR-MIL measure-potential path;
  - is generic for arbitrary class counts;
  - improves PANDA seed2024 validation from original MIR-MIL `0.951178` to `0.958328`;
  - improves BRACS3 official test to `0.842568 ± 0.009488`.

Partially useful but not accepted:

- Fixed multi-token readout:
  - BRACS3 validation: `0.909829 ± 0.004094`;
  - PANDA sanity: `0.953990`;
  - BRACS3 official test: `0.836596 ± 0.013349`;
  - helpful, but weaker than moment-token.

## Which changes did not help

Rejected candidates:

| Candidate | Primary result | Decision |
| --- | ---: | --- |
| class-aware evidence w005 | BRACS3 official test `0.808322 ± 0.024521` | rejected |
| subset consistency | BRACS3 val `0.885514 ± 0.008701` | rejected |
| gated multi-token | PANDA sanity `0.946393` | rejected |
| low-rank class-token | BRACS3 val `0.908484 ± 0.013847` | rejected |
| latent evidence transformer | BRACS3 val `0.892086 ± 0.013869` | rejected |
| ordinal calibration residual | BRACS3 val `0.898148 ± 0.031633` | rejected |
| cosine state residual | BRACS3 val `0.926786 ± 0.003129`, PANDA `0.941824` | rejected due PANDA regression |
| mean+moment token split | BRACS3 val `0.909230 ± 0.009978` | rejected |
| class-conditioned moment-token | BRACS3 val `0.914885 ± 0.009920`, PANDA `0.956593` | rejected; weaker than moment-token on PANDA and unstable |
| tail-aware token | BRACS3 val `0.908273 ± 0.011531` | rejected |
| logit-margin objective | BRACS3 val `0.906968 ± 0.027732` | rejected; seed sensitivity |

## Feature extractor status

Feature extractors remained unchanged:

- no R50 re-extraction;
- no UNI re-extraction;
- no R50/UNI fine-tuning;
- no feature replacement;
- no raw WSI training.

All accepted and rejected experiments used existing pre-extracted features.

## Model architecture status

The accepted MIR-MIL extension is an architecture/module-level change:

- `MomentMultiTokenAttentionReadout`
- enabled by `Model.moment_token_weight=0.1`
- default-disabled in the base config

Several additional modules now exist as default-disabled ablation utilities:

- class-aware evidence head;
- fixed multi-token readout;
- gated multi-token readout;
- low-rank class-token readout;
- latent evidence transformer readout;
- ordinal residual head;
- cosine residual head;
- class-conditioned moment-token readout;
- tail-aware token readout;
- logit-margin auxiliary objective.

Only moment-token w01 is accepted as a current improvement.

## Exact reproduction commands

### BRACS3 moment-token validation gate

```bash
mamba run -n mirmil python experiments/run_benchmark.py \
  --split /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_train_val.csv \
  --dataset-name BRACS3 \
  --num-classes 3 \
  --log-root artifacts/bracs3_arch_ablation/uni/moment_token_w01 \
  --models MIR_MIL \
  --seeds 2024 2025 2026 \
  --epochs 30 \
  --patience 8 \
  --best-model-metric macro_auc \
  --earlystop-metric macro_auc \
  --scheduler-t-max 28 \
  --model-option Model.evidence_weight=0.0 \
  --model-option Model.multi_token_weight=0.0 \
  --model-option Model.class_token_weight=0.0 \
  --model-option Model.latent_readout_weight=0.0 \
  --model-option Model.ordinal_head_weight=0.0 \
  --model-option Model.cosine_head_weight=0.0 \
  --model-option Model.moment_token_weight=0.1 \
  --model-option Model.moment_token_count=4 \
  --model-option Model.moment_token_dim=64 \
  --model-option Model.moment_token_readout_dim=128 \
  --model-option Model.moment_token_temperature=1.0 \
  --model-option Model.moment_token_dropout=0.0 \
  --model-option Model.subset_consistency_weight=0.0 \
  --model-option Model.subset_consistency_supervised_weight=0.0 \
  --model-option Model.potential_type=adaptive_multiscale \
  --model-option Model.num_local_routes=12 \
  --model-option Model.multiscale_gate_initial_bias=-0.5 \
  --model-option Model.multiscale_local_initial_scale=0.5 \
  --model-option Model.prototype_regularization_weight=0.01 \
  --max-instances 4096 \
  --in-dim 1024 \
  --feature uni \
  --protocol bracs3-arch-ablation-val-only \
  --split-id official-train-val-3class \
  --comparison-id bracs3-uni-mir-moment-token-w01 \
  --device 0 \
  --num-workers 2 \
  --wandb-mode disabled
```

### PANDA sanity for moment-token

```bash
mamba run -n mirmil python experiments/run_benchmark.py \
  --split /data15/data15_5/fanhao/datasets/PANDA/MIRMIL_FEATURES/metadata/PANDA_uni_split_v1_train_val_qc.csv \
  --dataset-name PANDA \
  --num-classes 6 \
  --log-root artifacts/panda_arch_ablation/uni/moment_token_w01 \
  --models MIR_MIL \
  --seeds 2024 \
  --epochs 30 \
  --patience 8 \
  --best-model-metric macro_auc \
  --earlystop-metric macro_auc \
  --scheduler-t-max 28 \
  --model-option Model.evidence_weight=0.0 \
  --model-option Model.multi_token_weight=0.0 \
  --model-option Model.class_token_weight=0.0 \
  --model-option Model.latent_readout_weight=0.0 \
  --model-option Model.ordinal_head_weight=0.0 \
  --model-option Model.cosine_head_weight=0.0 \
  --model-option Model.moment_token_weight=0.1 \
  --model-option Model.moment_token_count=4 \
  --model-option Model.moment_token_dim=64 \
  --model-option Model.moment_token_readout_dim=128 \
  --model-option Model.moment_token_temperature=1.0 \
  --model-option Model.moment_token_dropout=0.0 \
  --model-option Model.subset_consistency_weight=0.0 \
  --model-option Model.subset_consistency_supervised_weight=0.0 \
  --model-option Model.potential_type=adaptive_multiscale \
  --model-option Model.num_local_routes=12 \
  --model-option Model.multiscale_gate_initial_bias=-0.5 \
  --model-option Model.multiscale_local_initial_scale=0.5 \
  --model-option Model.prototype_regularization_weight=0.01 \
  --max-instances 4096 \
  --in-dim 1024 \
  --feature uni \
  --protocol panda-arch-ablation-val-only \
  --split-id split-v1-qc \
  --comparison-id panda-uni-mir-moment-token-w01 \
  --device 0 \
  --num-workers 2 \
  --wandb-mode disabled
```

### BRACS3 official test for frozen moment-token checkpoints

```bash
mamba run -n mirmil python experiments/evaluate_checkpoints.py \
  --run-root artifacts/bracs3_arch_ablation/uni/moment_token_w01 \
  --output-dir artifacts/bracs3_arch_ablation/uni/moment_token_w01_official_test_budget4096 \
  --models MIR_MIL \
  --budgets 4096 \
  --device 0 \
  --num-workers 2 \
  --group test \
  --checkpoint-kind best \
  --split-override /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_full.csv \
  --wandb-mode disabled
```

## Reviewer challenge

Was the improvement selected using validation performance?

- Yes. Moment-token was selected from BRACS3 official train/val, not from test.

Was the test set used for tuning?

- No. The moment-token official test was opened once after BRACS validation and PANDA sanity gates.

Is the improvement stable across seeds?

- Partially. The official test mean is improved, but seed2026 is weaker than seeds 2024/2025. This should be reported as a limitation.

Is the comparison fair?

- Yes for the reported setting: same official split, same fixed UNI features, same 4096-instance budget, same multi-seed protocol.

Did hidden config changes occur?

- The changed factors are explicit in the command and config logs. Moment-token is the accepted architecture change.

Can this result be reproduced with one command?

- Yes. Commands above reproduce validation, PANDA sanity, and official test evaluation from saved checkpoints.

Would this convince a paper reviewer as SOTA?

- No. It supports a credible MIR-MIL improvement, but not a SOTA claim because the result remains below AC_MIL by `0.010284` macro-AUC.

## Remaining limitations

1. BRACS3 validation is small and unstable; several strong validation candidates failed PANDA sanity or official test.
2. Moment-token improves both PANDA and BRACS, but does not close the full BRACS gap.
3. Later residual readout and loss candidates either failed macro-AUC or amplified seed variance.
4. BRACS3 operating-point metrics and macro-AUC can diverge; bacc/F1 improvements alone are not sufficient.
5. Further progress likely requires a larger generic MIL architecture change rather than another residual readout head.

## Honest assessment

The current MIR-MIL family has likely reached a local ceiling on BRACS3 under:

- fixed UNI/R50 features;
- official BRACS split;
- validation-driven model selection;
- no BRACS test tuning;
- conservative PANDA sanity gating.

Moment-token is a valid and reproducible improvement, but not SOTA. The remaining gap to `0.852852` is small numerically but difficult scientifically because most simple, generic follow-up changes either overfit BRACS validation, reduce PANDA, or increase seed sensitivity.

The next credible step should be a larger, still generic WSI-MIL architecture redesign that changes the state/readout interface more fundamentally while preserving the strengths that make MIR-MIL effective on PANDA.
