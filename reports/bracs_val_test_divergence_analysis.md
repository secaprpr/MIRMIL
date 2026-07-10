# BRACS3 validation-test divergence analysis

## Scope

This report analyzes the first generic architecture ablation, `MIR-MIL + ClassAwareEvidenceHead` with `Model.evidence_weight=0.05`.

The analysis uses only existing artifacts:

- BRACS3 validation logs from `artifacts/bracs3_arch_ablation/uni/evidence_w005`
- BRACS3 official-test predictions from `artifacts/bracs3_arch_ablation/uni/evidence_w005_official_test_budget4096`
- archived BRACS3 official-test baseline predictions from `artifacts/bracs3_evaluation_nomamba`
- PANDA sanity logs from `artifacts/panda_arch_ablation/uni/evidence_w005`

No new training, feature extraction, split change, or test-set tuning is performed here.

## Protocol correction

The archived BRACS3 official-test baseline matrix used `--max-instances 4096`, as recorded in `artifacts/bracs3_evaluation_nomamba/evaluation_manifest.json`.

An initial evidence evaluation used the default `evaluate_checkpoints.py` budgets `128/256/512`. Those numbers are useful for budget sensitivity, but they are not the fair comparison against the archived matrix.

The fair comparison is the corrected `4096`-budget evaluation:

```text
mamba run -n mirmil python experiments/evaluate_checkpoints.py \
  --run-root artifacts/bracs3_arch_ablation/uni/evidence_w005 \
  --output-dir artifacts/bracs3_arch_ablation/uni/evidence_w005_official_test_budget4096 \
  --models MIR_MIL \
  --budgets 4096 \
  --device 0 \
  --num-workers 2 \
  --group test \
  --checkpoint-kind best \
  --split-override /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS3_uni_split_official_full.csv \
  --wandb-mode disabled
```

## Summary result

BRACS3 validation improved strongly:

| setting | seeds | validation macro-AUC |
|---|---:|---:|
| UNI + MIR-MIL evidence_w005 | 2024/2025/2026 | `0.901497 ± 0.002426` |

PANDA sanity did not regress:

| setting | seed | validation macro-AUC |
|---|---:|---:|
| archived/default PANDA UNI + MIR-MIL | 2024 | `0.951178` |
| PANDA UNI + MIR-MIL evidence_w005 | 2024 | `0.951225` |

BRACS3 official test did regress under the fair 4096-budget protocol:

| setting | seeds | official-test macro-AUC |
|---|---:|---:|
| UNI + MIR-MIL baseline | 2024/2025/2026 | `0.827973 ± 0.027678` |
| UNI + MIR-MIL evidence_w005 | 2024/2025/2026 | `0.808322 ± 0.024521` |
| UNI + AC_MIL | 2024/2025/2026 | `0.852852 ± 0.009653` |

Conclusion: `evidence_w005` is not an accepted improvement. It passes PANDA validation sanity but fails BRACS3 official-test transfer.

## Per-class behavior on BRACS3 test

Macro-AUC decomposition:

| setting | macro-AUC | class 0 AUC | class 1 AUC | class 2 AUC |
|---|---:|---:|---:|---:|
| UNI + MIR-MIL baseline | `0.8280 ± 0.0277` | `0.8316 ± 0.0292` | `0.7663 ± 0.0607` | `0.8860 ± 0.0230` |
| UNI + MIR-MIL evidence_w005 | `0.8083 ± 0.0245` | `0.8413 ± 0.0225` | `0.7070 ± 0.0714` | `0.8767 ± 0.0405` |
| UNI + AC_MIL | `0.8529 ± 0.0097` | `0.8519 ± 0.0290` | `0.7930 ± 0.0172` | `0.9136 ± 0.0174` |

The dominant failure is class 1. Evidence_w005 slightly improves/maintains class 0 ranking but substantially worsens class 1 ranking.

Summed confusion matrices over three seeds, rows=true class and columns=predicted class:

Baseline UNI + MIR-MIL:

```text
[[74, 12, 10],
 [33, 18, 18],
 [20,  4, 72]]
```

Evidence_w005:

```text
[[67, 17, 12],
 [27, 16, 26],
 [12,  8, 76]]
```

AC_MIL:

```text
[[64, 19, 13],
 [12, 36, 21],
 [ 8, 10, 78]]
```

Class recall:

| setting | class 0 recall | class 1 recall | class 2 recall |
|---|---:|---:|---:|
| UNI + MIR-MIL baseline | `0.7708` | `0.2609` | `0.7500` |
| UNI + MIR-MIL evidence_w005 | `0.6979` | `0.2319` | `0.7917` |
| UNI + AC_MIL | `0.6667` | `0.5217` | `0.8125` |

The target gap is mostly class 1 handling. AC_MIL is not just better overall; it doubles class 1 recall relative to MIR-MIL. Evidence_w005 does not solve this and slightly worsens class 1 recall.

## Confidence behavior

Mean confidence on wrong predictions:

| setting | mean confidence | wrong-prediction confidence | true-class probability |
|---|---:|---:|---:|
| UNI + MIR-MIL baseline | `0.7967 ± 0.0897` | `0.6928 ± 0.0954` | `0.6042 ± 0.0302` |
| UNI + MIR-MIL evidence_w005 | `0.7991 ± 0.0689` | `0.7046 ± 0.0846` | `0.5929 ± 0.0389` |
| UNI + AC_MIL | `0.7785 ± 0.0806` | `0.6765 ± 0.0916` | `0.6227 ± 0.0380` |

Evidence_w005 is slightly more confident on wrong predictions than baseline MIR-MIL and assigns lower probability to the true class on average. This supports the hypothesis that the evidence branch amplifies validation-specific or spurious evidence rather than improving robust class separation.

## Split and bag-size evidence

Class counts:

| split | total | class 0 | class 1 | class 2 |
|---|---:|---:|---:|---:|
| train | 395 | 203 | 52 | 140 |
| val | 65 | 30 | 14 | 21 |
| test | 87 | 32 | 23 | 32 |

Bag-size statistics:

| split | mean | median | 75% | 90% | max |
|---|---:|---:|---:|---:|---:|
| train | 35,629 | 32,831 | 50,278 | 68,897 | 116,727 |
| val | 29,377 | 25,892 | 47,190 | 55,829 | 92,100 |
| test | 30,639 | 25,949 | 40,478 | 58,857 | 91,300 |

The val/test bag-size distributions are not radically different at the global level. However, all BRACS splits are heavily truncated by a 4096-instance budget; most slides expose only a small subset of available patches. This makes the validation estimate sensitive to patch subset and class-1 evidence sparsity.

Class-1 mean bag sizes:

| split | class-1 mean bag size | class-1 median bag size |
|---|---:|---:|
| train | 42,245 | 39,971 |
| val | 27,959 | 28,880 |
| test | 35,417 | 27,532 |

Class 1 is small in validation (`14` slides) and test (`23` slides). The validation-selected evidence branch can improve ranking on those 14 validation slides while failing to generalize to the 23 test slides.

## Interpretation

The first architecture hypothesis was only partially correct.

Supported:

- A class-aware patch evidence branch is generic.
- It does not harm PANDA validation.
- It can strongly improve BRACS validation.

Rejected:

- The BRACS bottleneck is not solved by simply adding class-aware evidence capacity.
- High BRACS validation macro-AUC is not sufficient evidence of official-test improvement.
- The evidence branch should not be tuned further on BRACS test.

Most likely failure mechanism:

1. BRACS class 1 is the unstable boundary.
2. The evidence branch increases class-specific confidence but does not learn a robust class-1 boundary.
3. Validation has too few class-1 slides to select this branch reliably.
4. The model remains sensitive to patch subset and slide-domain differences under the 4096 budget.

## Next direction

Do not continue tuning `evidence_weight`.

The next generic improvement should address validation/test robustness and class-boundary stability without using BRACS-specific labels or test feedback. Reasonable candidates:

1. Calibration/stability regularization on logits, generic for any class count.
2. Inference-time multi-crop/Monte-Carlo patch subset averaging using the existing architecture, selected on validation only.
3. A generic consistency loss between random patch subsets and uniform patch subsets.
4. Class-balanced validation reporting and seed robustness gates before any test evaluation.

Any next candidate must first be evaluated on BRACS validation and PANDA validation. Official BRACS test should be used only once after the candidate is frozen.
