# BRACS3 Goal Completion Audit

Date: 2026-07-11

Repository state audited at commit: `985c49f`

## Current conclusion

The current MIR-MIL architecture path has not reached the BRACS3 SOTA target.

The best accepted result remains:

- Method: `UNI + MIR-MIL moment-token w01`
- BRACS3 official test macro-AUC: `0.842568 ± 0.009488`
- Internal BRACS3 SOTA target: `UNI + AC_MIL = 0.852852 ± 0.009653`
- Remaining gap: `0.010284` macro-AUC

This result is a valid MIR-MIL improvement, but not a SOTA result.

## Constraint audit

Feature extractors remained fixed.

- No R50 or UNI feature re-extraction was used during the BRACS3 recovery stage.
- No encoder fine-tuning was introduced.
- No raw WSI training was introduced.
- Experiments used existing pre-extracted UNI/R50 feature paths and fixed feature dimensions.

Splits remained fixed.

- BRACS3 experiments used the official train/validation split metadata.
- The official split was not changed to improve performance.

Test-set access was controlled.

- Rejected candidates were stopped at validation-only or PANDA-sanity stages.
- Official BRACS3 test was opened only for frozen candidates that passed the stated validation/PANDA gates.
- Later rejected candidates after moment-token did not open BRACS3 official test.

Architecture changes were generic rather than BRACS-specific.

- Added modules are default-disabled config options.
- No BRACS label-count hardcoding, split hardcoding, or test-set-dependent logic was introduced.
- Candidate modules were evaluated as generic WSI-MIL improvements, including PANDA sanity where required.

## Evidence files

- `reports/bracs_sota_recovery_report.md`
  - Current best accepted result.
  - SOTA target and gap.
  - Reproduction commands.
  - Reviewer-style challenge.
  - Explicit statement that SOTA was not reached.

- `reports/bracs_deep_optimization_log.tsv`
  - Controlled log of BRACS3 experiments.
  - Includes accepted and rejected candidates with commands, seeds, metrics, and interpretation.

- `reports/general_mil_improvement_plan.md`
  - General architecture-improvement hypotheses.
  - Records validation-first decisions for each module.
  - Documents why later candidates were rejected.

- `reports/bracs_failure_analysis.md`
  - Explains why BRACS3 remains difficult under fixed features and official split.
  - States the current local-ceiling interpretation.

- `reports/panda_bracs_comparison.md`
  - Compares why MIR-MIL works better on PANDA than BRACS.
  - Records that moment-token helps both datasets but does not close the BRACS3 gap.

## Accepted positive result

The only accepted architecture extension is `moment-token w01`.

Reason:

- It improves BRACS3 official test over the previous MIR-MIL variants.
- It preserves/improves PANDA sanity performance.
- It was selected through validation-first gating.
- It was reproduced across three BRACS3 seeds.

Limit:

- It remains below the AC_MIL target by `0.010284` macro-AUC.
- Seed `2026` is weaker than seeds `2024/2025`, so the result is useful but not enough for a SOTA claim.

## Rejected follow-up candidates

The following later candidates did not justify opening BRACS3 official test or replacing moment-token:

- Class-conditioned moment-token readout
  - Slight BRACS3 validation AUC gain over moment-token.
  - Weaker than moment-token on PANDA sanity.
  - Decision metrics unstable.

- Tail-aware token readout
  - Improved some operating-point metrics.
  - Failed the primary validation macro-AUC gate.

- Multiclass logit-margin auxiliary objective
  - Produced one strong seed.
  - Increased seed variance and reduced mean validation macro-AUC.

These results argue against stacking small residual heads or simple loss terms without stronger evidence.

## Stop-condition assessment

The SOTA target was not reached.

However, the controlled search has satisfied the practical stop condition for the current small-module/architecture-frozen route:

1. The best accepted MIR-MIL result is documented and reproducible.
2. Multiple high-value hypotheses were tested under the same official split and validation-first protocol.
3. Later candidates failed to improve robust validation macro-AUC, PANDA sanity, or both.
4. The remaining gap is small numerically but not closed by the tested generic residual readouts or objective changes.
5. Claiming SOTA from the current evidence would not be defensible.

## Recommended next stage

Do not continue stacking minor heads, token variants, or BRACS-only tuning under the current protocol.

If the project needs to exceed the BRACS3 target, the next stage should be a deliberate architecture redesign with a new method identity, while preserving the fixed-feature experimental setting for fair comparison. The redesign should directly target:

- class-boundary evidence selection;
- validation/test transfer under small official splits;
- seed robustness;
- localized evidence preservation without hard BRACS-specific rules;
- comparable PANDA sanity performance.

The current commit/tag should be treated as the recoverable baseline for the existing MIR-MIL architecture family.
