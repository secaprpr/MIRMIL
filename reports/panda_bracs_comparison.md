# PANDA vs BRACS comparison

## Current official-test performance

From `artifacts/panda_evaluation/aggregate_results.csv`, PANDA results show MIR-MIL is strong:

- `UNI + MIR_MIL`: `0.9504 ± 0.0011` macro-AUC
- `R50 + MIR_MIL`: `0.8992 ± 0.0007` macro-AUC

From `reports/bracs3_baseline_official_test.md` and `reports/bracs3_mir_noearlystop200_diagnostic.md`, BRACS3 results show a weaker MIR-MIL position:

- `UNI + MIR_MIL`, best archived setting: `0.8403 ± 0.0184` macro-AUC
- `UNI + AC_MIL`: `0.8529 ± 0.0097` macro-AUC

Thus MIR-MIL is strong on PANDA but not dominant on BRACS3.

## Dataset and task differences

PANDA and BRACS differ in ways that affect MIL aggregation.

PANDA:

- larger and more stable training signal;
- prostate grading task has strong slide-level distributional patterns;
- tumor grade often correlates with the abundance and distribution of morphologic patterns across the slide;
- MIR-MIL's measure-state representation is well matched to this kind of distributional signal.

BRACS3:

- much smaller WSI dataset;
- official validation/test splits are small;
- the atypical category is small and histologically ambiguous;
- classification depends more on localized, class-specific diagnostic evidence;
- benign/atypical/malignant boundaries may be driven by relatively sparse regions.

## Feature behavior

UNI improves both PANDA and BRACS, but the model response differs.

On PANDA:

- `UNI + MIR_MIL` is the strongest listed result in the available matrix.
- The gap between MIR-MIL and other baselines favors MIR-MIL.

On BRACS3:

- UNI improves the overall baseline matrix.
- `UNI + AC_MIL` is stronger than `UNI + MIR_MIL`.
- `UNI + MIR_MIL` shows high validation but weaker official test.

This means the feature extractor is not the main bottleneck. The aggregation/readout inductive bias is the likely bottleneck.

## Why MIR-MIL is strong on PANDA

MIR-MIL represents a WSI as a neural measure:

- global composition captures overall tissue-pattern distribution;
- tail scores capture high-response extremes;
- local routes capture a limited set of salient modes;
- the potential head maps this state to class logits.

This is a good fit for PANDA because the slide label is strongly related to global and distributional tissue patterns. A compact measure state can preserve the signal needed for grading, and UNI features make that state highly separable.

## Why the same method is weaker on BRACS3

BRACS3 likely requires more explicit patch-level evidence selection:

- a small atypical focus can change the label;
- different classes may require different evidence regions;
- a single shared measure state can blur class-specific evidence;
- validation selection is unstable because the validation split is small.

The BRACS3 result therefore suggests an evidence-preservation gap, not a generic capacity gap.

## Scientific interpretation

The next architecture should not replace MIR-MIL's measure state, because that state explains PANDA strength. Instead, it should add a generic evidence readout that complements the measure state:

- preserve global distribution modeling for PANDA;
- add class-aware patch evidence for BRACS-like fine-grained tasks;
- expose a simple ablation switch to verify whether the evidence path helps generally.

This is a general WSI-MIL improvement, not a BRACS-only trick.

## Update from architecture ablations

The fixed multi-token residual is the best accepted generic extension so far:

- BRACS3 validation: `0.909829 ± 0.004094` macro-AUC.
- PANDA seed2024 sanity: `0.953990` macro-AUC, above original MIR-MIL seed2024 `0.951178`.
- BRACS3 official test at 4096 instances: `0.836596 ± 0.013349`, improving the archived original MIR-MIL official test (`0.827973 ± 0.027678`) but still below the no-early-stop/best-val MIR setting (`0.8403 ± 0.0184`) and AC_MIL target (`0.852852 ± 0.009653`).

The normalized cosine state residual is a useful negative control:

- BRACS3 validation: `0.926786 ± 0.003129` macro-AUC, the strongest validation-only signal so far.
- PANDA seed2024 sanity: only `0.941824` macro-AUC.
- This is a drop of `0.009354` versus original MIR-MIL and `0.012166` versus fixed multi-token.

Interpretation:

- Improving BRACS validation alone is not a reliable criterion.
- A module that only regularizes final-state decision geometry can fit BRACS train/val boundaries while damaging PANDA's stronger distributional signal.
- The more plausible path remains a general evidence-preserving aggregation/readout improvement that does not suppress the original measure-state behavior.

The moment multi-token readout is the strongest positive update so far:

- It extends fixed multi-token readout by adding per-token weighted variance statistics, not by changing final logit geometry.
- PANDA seed2024 reaches `0.958328` macro-AUC, exceeding original MIR-MIL and fixed multi-token.
- BRACS3 official test reaches `0.842568 ± 0.009488`, the best current MIR-MIL result.
- The remaining gap to the AC_MIL target `0.852852 ± 0.009653` is `0.010284` macro-AUC.

This suggests the PANDA/BRACS gap is not caused by feature quality alone. A more distribution-aware evidence readout can help both datasets, but BRACS still likely requires more robust class-boundary evidence selection than the current generic moment-token branch provides.

## Update: why later candidates did not close the BRACS gap

The later candidates clarify the PANDA/BRACS difference further.

Class-conditioned moment-token did not damage PANDA, but it also did not beat moment-token on PANDA:

- PANDA seed2024: `0.956593`.
- Moment-token PANDA seed2024: `0.958328`.
- BRACS3 validation gain over moment-token was only about `0.001433` macro-AUC and decision metrics were unstable.

Tail-aware token readout improved some BRACS3 decision metrics but not BRACS3 macro-AUC:

- BRACS3 validation macro-AUC: `0.908273 ± 0.011531`.
- This is below moment-token and fixed multi-token.
- Interpretation: BRACS3 does contain sparse or local evidence, but hard top-k pooling alone does not produce better ranking.

The logit-margin auxiliary objective directly targeted class separation, but increased seed sensitivity:

- BRACS3 validation macro-AUC: `0.906968 ± 0.027732`.
- One seed improved strongly, while seed2024 collapsed.
- Interpretation: sharper margins can overfit small BRACS validation boundaries and are not a stable general fix.

Current interpretation:

- PANDA benefits from distribution-aware token statistics; moment-token improves PANDA to `0.958328`.
- BRACS also benefits somewhat, but its remaining error is not fixed by residual readouts, hard tails, or simple margin regularization.
- The most likely limitation is now the MIR-MIL state/readout interface itself: the model preserves strong global distributional information, but does not reliably learn class-boundary evidence under small, ambiguous BRACS splits.
- Further progress to the AC_MIL target may require a larger, still generic architecture change rather than another small residual head.
