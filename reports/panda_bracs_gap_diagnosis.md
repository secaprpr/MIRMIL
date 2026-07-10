# PANDA-vs-BRACS Gap Diagnosis

Date: 2026-07-10

## Key answer

MIR-MIL works well on PANDA mainly because PANDA is large, stratified, less severely truncated by the 4,096-patch budget, and has stable validation/test estimates. It underperforms on BRACS mainly because BRACS is small, seven-class, imbalanced, fine-grained, high-bag-size, and validation selection is noisy. The evidence points more strongly to dataset/statistical instability, bag sampling mismatch, and hyperparameter/protocol mismatch than to an encoder failure alone.

No evidence supports changing the feature extractor or model architecture at this stage.

## Observed PANDA performance

From `artifacts/panda_evaluation/aggregate_results.csv`:

| Feature | MIR-MIL test macro AUC | Best competing macro AUC | MIR rank |
| --- | ---: | ---: | --- |
| R50 | `0.8992 ± 0.0007` | RRTMIL `0.8954 ± 0.0022` / WiKG `0.8930 ± 0.0036` | 1st |
| UNI | `0.9504 ± 0.0011` | CLAM-MB `0.9428 ± 0.0034` / WiKG `0.9417 ± 0.0017` | 1st |

MIR-MIL is first on accuracy, balanced accuracy, macro AUC, and macro F1 for both R50 and UNI in the PANDA benchmark.

## Observed BRACS performance

From `artifacts/bracs_evaluation/aggregate_results.csv`:

| Feature | MIR-MIL test macro AUC | Best competing macro AUC | Gap |
| --- | ---: | ---: | ---: |
| R50 | `0.7087 ± 0.0073` | ACMIL `0.7178 ± 0.0147` | `-0.0091` |
| UNI | `0.7694 ± 0.0356` | CLAM-SB `0.8039 ± 0.0060` | `-0.0345` |

Best previous single-feature optimized MIR-MIL:

- R50 `capacity_potential256`: official test mean about `0.7137`, still below R50 ACMIL.
- UNI `capacity_h384_s128`: official test mean about `0.7862`, still below UNI CLAM-SB.

Exploratory fusion:

- R50+UNI fusion: official test `0.8151 ± 0.0093`.
- Excluded from primary comparison because it combines features.

## Current BRACS SOTA target

Within the local baseline matrix:

- R50 target: ACMIL macro AUC `0.7178 ± 0.0147`.
- UNI target: CLAM-SB macro AUC `0.8039 ± 0.0060`.
- Overall single-feature target: UNI CLAM-SB `0.8039 ± 0.0060`.

A convincing MIR-MIL claim should exceed the corresponding feature-specific baseline with validation-selected, multi-seed, official-test evidence.

## Dataset statistics evidence

### Size and split

| Dataset | Train | Val | Test | Classes |
| --- | ---: | ---: | ---: | ---: |
| PANDA QC | 6,369 | 2,123 | 2,123 | 6 |
| BRACS official | 395 | 65 | 87 | 7 |

BRACS has about 16× fewer training slides and about 24× fewer test slides than PANDA. Its validation set has only 65 slides across 7 classes.

### Label distribution

BRACS train labels:

- class 0: 27
- class 1: 120
- class 2: 56
- class 3: 24
- class 4: 28
- class 5: 40
- class 6: 100

The BRACS train max/min class ratio is `120/24 = 5.0`. The validation set has only 6-12 slides per class. This is enough to make macro AUC and early stopping high variance.

PANDA train labels:

- class 0: 1,735
- class 1: 1,599
- class 2: 806
- class 3: 745
- class 4: 749
- class 5: 735

PANDA is still imbalanced but each class has hundreds to thousands of slides, so validation macro AUC is much more stable.

### Bag size / patch count distribution

BRACS full H5 patch-count audit:

- 547 slides.
- median patch count: `31,183`.
- mean: `34,092`.
- p90: `63,777`.
- max: `116,727`.
- `504/547` slides exceed 4,096 patches.
- `415/547` slides exceed 16,384 patches.

PANDA sampled patch-count audit:

- sampled 240 slides.
- median around `496-497`.
- mean around `485-509`.
- max in sample `2,082`.
- none exceeded 4,096.

This is the largest mechanical difference between datasets. On PANDA, a 4,096-instance budget effectively sees the whole bag. On BRACS, the same budget sees only a small random subset, often around 6-13% of patches. This makes BRACS more sensitive to sampling seed, patch selection, and validation noise.

### Feature distribution evidence

Small sampled feature-norm check:

| Dataset/feature | Mean patch-feature norm | Approx feature std |
| --- | ---: | ---: |
| PANDA R50 sample | `~1.999` | `~0.046` |
| BRACS R50 small sample | `~1.972` | `~0.045` |
| PANDA UNI sample | `~38.53` | `~1.204` |
| BRACS UNI small sample | `~38.43` | `~1.201` |

The coarse feature norm statistics do not show a simple scale mismatch. Therefore, feature normalization may still help at the training level, but there is no obvious gross norm bug explaining BRACS failure.

## R50 vs UNI behavior on BRACS

BRACS baselines show UNI helps many methods:

- R50 best macro AUC: `0.7178`.
- UNI best macro AUC: `0.8039`.

For MIR-MIL:

- R50 baseline: `0.7087`.
- UNI baseline: `0.7694`.

UNI helps MIR-MIL by about `+0.0607` macro AUC on BRACS, but less effectively than it helps the best competing baselines. On PANDA, UNI helps MIR-MIL much more strongly:

- PANDA MIR R50: `0.8992`.
- PANDA MIR UNI: `0.9504`.
- gain `+0.0512`, but on a much higher and more stable baseline.

The issue is not that UNI is useless on BRACS. The issue is that MIR-MIL does not exploit UNI as well as CLAM-SB/ACMIL/CLAM-MB on BRACS.

## Validation/test behavior

Several BRACS candidates improve validation but not test:

- UNI `capacity_h384_s128`:
  - validation macro AUC about `0.8470`, `0.8267`, `0.8503`;
  - official test mean about `0.7862`, below CLAM-SB.
- R50 `capacity_potential256`:
  - validation macro AUC about `0.8040`, `0.7745`, `0.7543`;
  - official test mean about `0.7137`, below ACMIL.
- `moment_order=2`:
  - seed2024 validation looked promising;
  - cross-seed validation/test were weak.

This pattern is consistent with unreliable validation selection on BRACS rather than a simple underfit fix.

## Does BRACS underperformance appear in both features?

Yes, but differently:

- R50: MIR-MIL is close to best baseline, gap about `0.009` macro AUC. The problem is smaller and may be statistical.
- UNI: MIR-MIL is clearly below best baseline, gap about `0.0345` macro AUC. This suggests a feature-method interaction: UNI contains stronger signal, but MIR-MIL’s current training/sampling protocol does not reliably convert it into BRACS test performance.

## Underfitting, overfitting, seed variance, class imbalance, domain shift, or metric mismatch?

Ranked diagnosis:

1. **Validation instability / small validation set**  
   Strong evidence. Only 65 validation slides across 7 classes. High validation candidates do not transfer to test.

2. **Bag sampling mismatch**  
   Strong evidence. BRACS has huge bags; 4,096 patches heavily subsample tissue. PANDA nearly does not truncate. MIR-MIL may depend more on representative empirical measures than attention baselines, so random subsampling can hurt.

3. **Fine-grained label ambiguity**  
   Strongly plausible. BRACS categories such as PB/UDH/FEA/ADH/DCIS/IC are finer histological categories than PANDA ISUP grades. Slide-level labels may be harder to infer from sparse patch subsets.

4. **Class imbalance**  
   Moderate evidence. BRACS train class ratio is 5:1 and val/test per-class counts are tiny. Weighted/focal screens did not help, but they were incomplete and not necessarily the right balancing mechanism.

5. **Hyperparameter mismatch from PANDA**  
   Moderate evidence. PANDA-tuned defaults work with small bags and large data. BRACS likely needs BRACS-specific sampling/early stopping/regularization, but many simple changes already failed.

6. **Feature distribution shift**  
   Plausible but not proven by norm statistics. Gross scale mismatch is not evident. Domain/histology shift may still be semantic, not norm-based.

7. **Metric computation mismatch**  
   Less likely. The same evaluation utilities and aggregate checks were used, prediction files were audited, and rows/probabilities were validated.

8. **Split handling bug**  
   Less likely but still worth a specific audit. Official BRACS split is preserved, including known Patient 67 overlap. Split comparability should be rechecked before any new claim.

## Why MIR-MIL likely works well on PANDA

1. PANDA has enough slides for MIR-MIL’s higher-capacity measure-response modeling.
2. Validation/test estimates are stable; seed std is small for MIR-MIL:
   - R50 macro AUC std `0.0007`;
   - UNI macro AUC std `0.0011`.
3. Patch budget is effectively full-bag for PANDA.
4. UNI features provide a clean and strong representation for PANDA’s grading task.
5. PANDA labels are ordinal; even though the baseline matrix used macro AUC, the disease grading structure may align with MIR-MIL’s global distribution modeling.

## Why the same method underperforms on BRACS

1. BRACS has far fewer slides and much noisier validation/test estimates.
2. BRACS bags are huge, so the empirical measure is poorly approximated by 4,096 random patches.
3. Seven fine-grained categories require discriminating subtle histology; sparse random sampling may miss rare diagnostic regions.
4. MIR-MIL’s validation-selected capacity variants may overfit the tiny validation set.
5. UNI gives strong signal to the dataset overall, but MIR-MIL’s current training protocol does not extract the same BRACS benefit as CLAM-SB/ACMIL.

## Hypotheses rejected by existing evidence

- **“BRACS failure is because UNI is bad.”** Rejected. UNI baselines are strong; UNI CLAM-SB reaches `0.8039`.
- **“Simply increasing MIR capacity solves BRACS.”** Rejected. UNI validation improves, but test does not reach SOTA.
- **“Moment order 2 solves BRACS.”** Rejected by cross-seed/test results.
- **“EMA/SWA will stabilize BRACS.”** Current evidence is negative.
- **“Feature fusion proves single-feature SOTA.”** Rejected by protocol; fusion is not a single-feature result.
- **“Metric computation is obviously broken.”** Not supported; aggregate and prediction audits passed.

## Remaining uncertain hypotheses

1. Whether deterministic or stratified patch sampling can reduce BRACS variance without changing features or architecture.
2. Whether validation reliability can be improved by model selection rules using smoother validation windows, not single best epoch.
3. Whether class balancing should operate at sampler level, loss level, or both.
4. Whether feature normalization/filtering using existing features can reduce BRACS bag noise.
5. Whether larger fixed inference-time averaging over multiple random patch subsets improves test in a validation-driven way without changing architecture.

## Prioritized optimization plan

Architecture-frozen and feature-frozen:

1. Reproduce the best previous BRACS single-feature settings exactly:
   - UNI `capacity_h384_s128`;
   - R50 `capacity_potential256`.
2. Run seed robustness only if a setting is selected by validation, not by test.
3. Audit split/label/metric equivalence again before new official test claims.
4. Compare R50 and UNI under identical default and best-known training settings.
5. Investigate validation selection reliability:
   - best epoch distribution;
   - validation curve smoothness;
   - whether last-k checkpoint averaging at inference helps validation.
6. Test high-value sampling hypotheses:
   - deterministic uniform training sampling;
   - fixed cached patch subsets;
   - larger existing-feature candidate pools with 4,096 sampled per epoch;
   - inference-time averaging over random 4,096 subsets.
7. Test class balancing narrowly:
   - sampler on/off;
   - weighted loss only if a completed validation screen supports it.
8. Test feature normalization/filtering using existing features only:
   - per-slide feature standardization/L2 normalization as preprocessing or data-loader transform;
   - remove near-zero/low-norm patches if justified by validation.

The next experiments should be few, hypothesis-driven, and logged before any official test is opened.

