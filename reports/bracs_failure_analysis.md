# BRACS failure analysis

## Scope

This analysis uses the current frozen-feature BRACS3 evidence in the repository. It does not propose changing the official BRACS split, re-extracting features, changing R50/UNI encoders, or tuning on the BRACS test set.

Relevant artifacts:

- `reports/bracs3_baseline_official_test.md`
- `reports/bracs3_mir_noearlystop200_diagnostic.md`
- `reports/bracs3_mir_hpo_round1_status.md`
- `artifacts/bracs3_evaluation_nomamba/aggregate_results.csv`
- `artifacts/bracs3_mir_long/noearlystop200_evaluation/aggregate_results.csv`

## Observed BRACS3 results

The best official-test non-MAMBA baseline is `UNI + AC_MIL`:

- macro-AUC: `0.8529 ± 0.0097`
- macro-F1: `0.6620 ± 0.0476`

The archived MIR-MIL official-test results are:

- `UNI + MIR_MIL`, original early-stop protocol: `0.8280 ± 0.0277` macro-AUC
- `UNI + MIR_MIL`, 30/200 no-early-stop best-val protocol: `0.8403 ± 0.0184` macro-AUC
- `R50 + MIR_MIL`, original early-stop protocol: `0.7570 ± 0.0120` macro-AUC
- `R50 + MIR_MIL`, 30/200 no-early-stop best-val protocol: `0.7743 ± 0.0077` macro-AUC

The best current MIR-MIL gap to the internal BRACS3 target is therefore:

- `0.8529 - 0.8403 = 0.0126` macro-AUC

The gap to the protocol-dependent ACMIL literature reference around `0.888 ± 0.010` is larger:

- `0.888 - 0.8403 = 0.0477` macro-AUC

## What failed

The first training-only HPO round improved validation but not official test.

Top validation variants:

- `mild_reg`: `0.9041 ± 0.0224` validation macro-AUC
- `unbalanced_mild`: `0.9027 ± 0.0282`
- `loss_focal1`: `0.9007 ± 0.0157`
- `smooth005`: `0.8981 ± 0.0141`

Official test for those validation-selected variants:

- `smooth005`: `0.8245 ± 0.0092`
- `unbalanced_mild`: `0.8220 ± 0.0319`
- `loss_focal1`: `0.8214 ± 0.0178`
- `mild_reg`: `0.8115 ± 0.0059`

This is strong evidence that additional naive HPO is selecting BRACS validation artifacts instead of improving generalization.

## Failure mode

The failure is not primarily a feature quality failure. UNI features improve most BRACS3 baselines, and `UNI + AC_MIL` is the current strongest internal baseline. The failure is more specific:

1. MIR-MIL benefits from stronger feature encoders but does not fully exploit UNI on BRACS3.
2. MIR-MIL validation peaks can be high, but selected checkpoints do not transfer reliably to the official test set.
3. BRACS3 seems to reward localized, class-specific evidence selection more than global bag statistics alone.
4. AC_MIL, AB_MIL, CLAM, and TRANS/DTFD baselines with UNI are competitive, suggesting that direct attention-style readout is a useful inductive bias for BRACS3.

## Why current MIR-MIL underperforms on BRACS3

Current MIR-MIL compresses each WSI bag into a measure state:

- response-basis mean, optionally variance;
- tail log-sum-exp scores;
- local route states;
- optional anchor route state;
- an MLP/multiscale/prototype potential over this state.

This is strong for distributional WSI tasks because it summarizes the slide as a learned measure. However, BRACS3 has relatively fine-grained diagnostic boundaries. A small number of atypical or malignant regions can be decisive. If those regions are diluted during state construction or routed through shared global state, the final head may not receive enough class-specific evidence.

The key weakness is therefore not that MIR-MIL lacks capacity, but that its default aggregation is not explicitly class-evidence preserving. It asks one compressed state to support all class decisions. A class-aware evidence readout can preserve the existing measure-potential path while adding a generic mechanism for each class to ask: "which patches support this class?"

## Why not continue simple HPO

The HPO evidence is negative:

- validation increased above `0.90`;
- official test decreased below the no-early-stop baseline;
- top validation variants were not reliable.

Further tuning learning rate, weight decay, label smoothing, focal loss, or sampler settings is likely to increase validation over-selection unless a more general inductive bias is added and evaluated on both PANDA and BRACS.

## Architecture-ablation update: validation gains are not enough

Several generic residual modules have now been tested under the same discipline: BRACS3 official train/val first, PANDA sanity second, and BRACS3 official test only if both gates pass.

The strongest BRACS3 validation-only candidate was the normalized cosine state residual:

- BRACS3 UNI train/val, seeds 2024/2025/2026: `0.926786 ± 0.003129` macro-AUC.
- This exceeds the fixed multi-token candidate validation result (`0.909829 ± 0.004094`) by about `0.016957`.
- However, PANDA UNI seed2024 sanity reached only `0.941824` macro-AUC.
- Original PANDA UNI MIR-MIL seed2024 was `0.951178`, and fixed multi-token was `0.953990`.

This rejects the normalized cosine residual as an accepted general improvement. It also provides an important diagnostic: BRACS validation can strongly favor decision-geometry or calibration changes that do not preserve PANDA performance. Therefore, future architecture work should not optimize only for BRACS train/val macro-AUC. A credible general module must improve or preserve PANDA while improving BRACS, otherwise it is likely fitting the BRACS validation split rather than fixing the underlying MIL weakness.

The subsequent moment multi-token evidence readout is a stronger positive control:

- BRACS3 validation: `0.913452 ± 0.015874` macro-AUC.
- PANDA seed2024 sanity: `0.958328` macro-AUC, higher than both original MIR-MIL (`0.951178`) and fixed multi-token (`0.953990`) on the same PANDA split/seed.
- BRACS3 official test at 4096 instances: `0.842568 ± 0.009488` macro-AUC.

This is the best current MIR-MIL BRACS3 official-test result and narrows the gap to AC_MIL target `0.852852` to `0.010284`. It supports the hypothesis that BRACS needs a more evidence-preserving readout, but it also shows the remaining gap is not solved by adding token-level second moments alone. The likely remaining weakness is class/boundary-specific evidence selection under a small, ambiguous BRACS split, especially because seed2026 transfers worse than seeds 2024/2025.

## Later architecture/objective ablations: what is now rejected

After moment-token became the best accepted MIR-MIL extension, three additional generic candidates were tested under validation-first gating:

1. Class-conditioned moment-token readout.
2. Tail-aware token readout.
3. Multiclass logit-margin auxiliary objective.

None of these opened BRACS official test except moment-token, because none passed the validation/PANDA gate strongly enough.

Class-conditioned moment-token:

- BRACS3 validation: `0.914885 ± 0.009920` macro-AUC.
- This is only `+0.001433` over moment-token validation (`0.913452 ± 0.015874`).
- PANDA seed2024 sanity: `0.956593`, which is above original MIR-MIL (`0.951178`) but below moment-token (`0.958328`).
- Decision metrics were unstable: seed2024 had high AUC but weak bacc/F1.
- Rejected because it was not a stronger general module than moment-token and its BRACS gain was too small relative to variance.

Tail-aware token readout:

- BRACS3 validation: `0.908273 ± 0.011531` macro-AUC.
- Mean acc/bacc/macro-F1: `0.779487 ± 0.023500`, `0.714815 ± 0.061483`, `0.699343 ± 0.088432`.
- It improved some operating-point metrics, but failed the primary macro-AUC ranking gate versus moment-token and fixed multi-token.
- Rejected because hard top-response evidence alone improves threshold behavior more than ranking quality.

Multiclass logit-margin auxiliary objective:

- BRACS3 validation with moment-token + margin weight `0.05`: `0.906968 ± 0.027732` macro-AUC.
- Seed results were highly unstable: seed2024 `0.875500`, seed2025 `0.927838`, seed2026 `0.917566`.
- Rejected because it created one strong seed while damaging another and increasing seed sensitivity.

These negative results narrow the failure mode:

- The problem is not solved by simply making the readout more class-conditioned.
- The problem is not solved by adding hard tail/top-k evidence.
- The problem is not solved by forcing larger per-sample logit margins.
- BRACS3 macro-AUC appears sensitive to seed and validation split effects; methods that improve bacc/F1 or one seed can still fail mean AUC and robustness.

The strongest defensible conclusion is that the current MIR-MIL family has likely reached a local ceiling on BRACS3 under fixed features, official split, and conservative validation-gated testing. Moment-token is a real general improvement because it improves PANDA and BRACS official test, but the remaining `~0.0103` macro-AUC gap to the internal AC_MIL target probably requires a more substantial architecture change than residual readout heads or simple auxiliary objectives.

## Requirements for the next method

A valid next method should:

- be generic for any number of classes;
- not encode BRACS classes, BRACS split, BRACS class counts, or BRACS-specific thresholds;
- preserve the existing strong MIR-MIL measure-potential path;
- add a direct class-aware or multi-token evidence path that is easy to ablate;
- be evaluated on BRACS3 and PANDA before any SOTA claim.
