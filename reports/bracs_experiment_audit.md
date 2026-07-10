# BRACS Experiment Audit

Date: 2026-07-10

Scope: existing local artifacts, result tables, configs, split files, logs, and git history in `/data15/data15_5/fanhao/projects/MIRMIL`. No new experiment was launched for this audit.

## Executive conclusion

The current valid BRACS primary comparison is still the single-feature protocol:

- R50 and UNI are reported separately.
- R50/UNI fusion is excluded from the primary comparison.
- Feature extractors and pre-extracted features are fixed.
- Checkpoints are selected by validation macro AUC and evaluated on the official test split.

Under this valid protocol, MIR-MIL has not reached BRACS SOTA:

| Feature | Best baseline by official test macro AUC | MIR-MIL baseline | Best MIR-MIL optimized single-feature result found | Status |
| --- | ---: | ---: | ---: | --- |
| R50 | ACMIL `0.7178 ± 0.0147` | `0.7087 ± 0.0073` | `capacity_potential256` test mean about `0.7137` | below R50 SOTA |
| UNI | CLAM-SB `0.8039 ± 0.0060` | `0.7694 ± 0.0356` | `capacity_h384_s128` test mean about `0.7862` | below UNI SOTA |

The exploratory R50+UNI fusion reached `0.8151 ± 0.0093`, but it is not a valid primary result because it changes the feature protocol by combining encoders. It also did not establish significance over UNI CLAM-SB by the recorded bootstrap interval.

## 1. What BRACS experiments have already been tried

The main experiment families are listed in [bracs_experiment_matrix.tsv](bracs_experiment_matrix.tsv). In summary:

1. Complete baseline matrix:
   - 12 MIL models.
   - R50 and UNI features.
   - seeds 2024, 2025, 2026.
   - official BRACS train/val/test split.
   - 4,096-instance bag budget.
2. MIR-MIL HPO:
   - learning rate and weight decay.
   - dropout, patch dropout, feature noise.
   - label smoothing.
   - balanced versus unbalanced sampling.
   - EMA.
   - ordinal loss.
   - `moment_order=2`.
   - potential/prototype/multiscale variants.
   - capacity variants (`hidden_dim`, `sketch_dim`, `potential_hidden_dim`, routes).
   - inference budget checks.
   - train+val refit.
3. Exploratory excluded experiments:
   - R50+UNI fusion.
   - temporary instance top-k loss.
   - uncommitted weighted/focal CE screen.
   - uncommitted SWA screen.
   - pt split mirrors for faster screening.

## 2. Which experiments used R50 features

R50 was used in:

- full baseline matrix: `artifacts/bracs_evaluation/aggregate_results.csv`;
- MIR-MIL baseline: official test macro AUC `0.7087 ± 0.0073`;
- R50 capacity search:
  - `artifacts/bracs_mir_r50_capacity_hpo/hpo_results.csv`;
  - `artifacts/bracs_mir_r50_potential_reval_seed2025/hpo_results.csv`;
  - `artifacts/bracs_mir_r50_potential_reval_seed2026/hpo_results.csv`;
  - `artifacts/bracs_mir_r50_potential_test/`;
- July 10 pt-mirror screen:
  - `artifacts/bracs_mir_r50_h384_reg_hpo_seed2024/hpo_results.csv`;
  - `artifacts/bracs_mir_r50_swa_hpo_seed2024/hpo_results.csv`.

The best R50 MIR-MIL variant remains `capacity_potential256` by validation, but its official test mean is still below R50 ACMIL.

## 3. Which experiments used UNI features

UNI was used in:

- full baseline matrix: `artifacts/bracs_evaluation/aggregate_results.csv`;
- MIR-MIL baseline: official test macro AUC `0.7694 ± 0.0356`;
- MIR-MIL HPO stages:
  - `artifacts/bracs_mir_hpo/`;
  - `artifacts/bracs_mir_hpo_ordinal/`;
  - `artifacts/bracs_mir_hpo_stage2/`;
  - `artifacts/bracs_mir_hpo_stage3/`;
  - `artifacts/bracs_mir_hpo_stage4*`;
  - `artifacts/bracs_mir_uni_capacity_hpo/`;
  - revalidation directories for seeds 2025 and 2026;
  - several test prediction directories under `artifacts/bracs_mir_test_predictions/`.

The best UNI validation candidate is `capacity_h384_s128`, with validation macro AUC about `0.8470`, `0.8267`, and `0.8503` across seeds 2024-2026. Its official test mean, however, is about `0.7862`, below UNI CLAM-SB.

## 4. Whether any experiments combined R50 and UNI

Yes. R50+UNI fusion experiments exist under:

- `experiments/prepare_bracs_fused_features.py`;
- `/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/r50_uni_fusion/`;
- `artifacts/bracs_mir_fusion_*`;
- `artifacts/bracs_mir_fusion_test/`.

These concatenate aligned R50 and UNI patch features and optionally apply parameter-free group L2 normalization. They are explicitly excluded from the primary comparison because the user’s reporting protocol requires R50 and UNI to be reported separately using their original features.

## 5. Which factors were changed

Already tried factors include:

- training hyperparameters:
  - learning rate;
  - weight decay;
  - dropout;
  - label smoothing;
  - patch dropout;
  - feature noise;
  - EMA;
  - SWA screen, later interrupted;
- sampler:
  - balanced sampler;
  - unbalanced sampler;
- objective:
  - CE;
  - ordinal auxiliary loss;
  - temporary instance top-k loss;
  - weighted/focal CE screen, interrupted;
- model capacity and existing MIR parameters:
  - hidden dimension;
  - sketch dimension;
  - potential hidden dimension;
  - number of local routes;
  - moment order;
  - potential type/prototype variants;
- inference/evaluation:
  - larger budget evaluation;
  - train+val refit;
  - pt mirror for faster screening;
- excluded feature protocol:
  - R50+UNI fusion.

Several of these model-capacity/potential changes are not allowed under the current architecture-frozen request. They are retained in the audit because they explain previous results, but they should not be part of new architecture-frozen optimization.

## 6. Which changes improved BRACS validation performance

Validation improvements were mostly concentrated in capacity changes:

- UNI `capacity_h384_s128`:
  - validation macro AUC about `0.8470`, `0.8267`, `0.8503` across seeds;
  - improved validation over baseline MIR-MIL.
- R50 `capacity_potential256`:
  - validation macro AUC about `0.8040`, `0.7745`, `0.7543`;
  - improved validation over default R50 MIR-MIL.
- R50+UNI fusion:
  - validation around `0.8400`, `0.8337`, `0.8227`;
  - excluded from primary comparison.

Smaller regularization, ordinal, moment, prototype, EMA, SWA, weighted/focal, and instance-loss attempts did not consistently beat the above.

## 7. Which changes improved BRACS test performance but may be unreliable

- R50 `capacity_potential256` slightly improved MIR-MIL R50 baseline test mean (`~0.7137` versus `0.7087`) but did not beat R50 ACMIL (`0.7178`). The gain is too small and not SOTA.
- UNI `capacity_h384_s128` improved over default MIR-MIL UNI in some seeds but not enough; mean remained below UNI CLAM-SB.
- R50+UNI fusion improved official test mean to `0.8151`, but it is excluded because it changes the feature protocol.
- Some individual seeds look strong, e.g. UNI MIR seed 2026 near `0.8103`, but single-seed observations are not valid SOTA evidence.

## 8. Which changes were harmful

The following were harmful or weak:

- EMA in stage-1 HPO: validation macro AUC around `0.74-0.75`, worse than non-EMA.
- `moment_order=2`: seed2024 validation looked promising, but cross-seed validation and official test were poor.
- Strong regularization around h384: did not improve validation.
- temporary instance top-k loss: best validation only about `0.805`, later removed.
- weighted/focal CE screen: partial validation was below best known candidates.
- SWA screen: post-SWA validation did not improve over non-SWA checkpoints.
- several prototype/potential variants: mostly around `0.77-0.80`, below capacity candidate.

## 9. Inconclusive experiments

The following are inconclusive or non-primary:

- Interrupted weighted/focal and SWA screens: useful as negative trend evidence, but not complete.
- pt mirror experiments: useful for fast screening, but primary conclusions should use the original H5/full official protocol unless pt/H5 equivalence is explicitly audited.
- Train+val refit: lacks validation selection and therefore cannot be compared cleanly.
- R50+UNI fusion: valid exploratory result, invalid primary result.
- Any single-seed result: cannot support SOTA or robust claims.
- Temporary instance loss: code removed and outside the current architecture-frozen setting.

## 10. Validation versus test versus possible test-set tuning

The valid baseline matrix follows a clean selection protocol: validation macro AUC selects checkpoints, then official test evaluates once.

The later MIR optimization phase repeatedly inspected official test results for candidate variants. This creates test-set tuning risk. Therefore:

- future optimization must select candidates only by validation;
- official test should be opened only after freezing a configuration;
- any candidate already chosen because of observed test behavior should be treated skeptically.

The strongest warning sign is the UNI `capacity_h384_s128` result: validation is consistently high, but official test mean is below CLAM-SB. This suggests BRACS validation may be too small/noisy to reliably predict test ranking.

## 11. Reproducibility status

Reproducible:

- baseline matrix: complete aggregate and seed files exist;
- PANDA matrix: complete aggregate and seed files exist;
- BRACS single-feature capacity HPO: hpo CSVs and manifests exist;
- fusion exploratory test: logs and aggregate exist.

Partially reproducible:

- interrupted July 10 weighted/focal/SWA screens: hpo CSVs record interrupted return codes but not complete results.
- pt mirror screens: CSVs exist, but they should be treated as screening unless pt/H5 equivalence is documented.

Not accepted:

- temporary instance-loss experiments: code was removed and should not be used for claims.

## 12. PANDA and BRACS protocol comparability

Comparable:

- both use pre-extracted R50/UNI features;
- both use fixed split files;
- both use three seeds for baseline matrices;
- both use validation selection and sealed/official test evaluation;
- both report acc, balanced acc, macro AUC, macro F1.

Not comparable:

- PANDA has 10,615 QC slides, while BRACS has 547 slides.
- PANDA has 6 classes; BRACS has 7.
- PANDA test has 2,123 slides; BRACS test has only 87.
- PANDA bags are small; sampled PANDA median patch count is about 500, so a 4,096 budget rarely truncates.
- BRACS bags are huge; full BRACS median patch count is about 31,183 and 504/547 slides exceed 4,096 patches.
- BRACS categories are finer histological entities with likely greater label ambiguity.
- Current MIR hyperparameters were largely inherited from PANDA-scale experiments and then searched on a much smaller BRACS validation set.

## 13. Reviewer-style conclusion

A skeptical reviewer would not accept a BRACS SOTA claim yet because:

- single-feature MIR-MIL does not beat the best single-feature baselines;
- the only apparent SOTA-like result uses excluded R50+UNI fusion;
- validation improvements often fail to transfer to test;
- several later candidates were explored after looking at official test performance;
- seed stability is weak on BRACS;
- current evidence suggests dataset/split/statistical instability rather than a simple missing hyperparameter.

