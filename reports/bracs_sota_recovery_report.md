# BRACS SOTA Recovery Report

Date: 2026-07-10

## Status

BRACS SOTA has not been reached under the valid single-feature, architecture-frozen protocol.

This recovery pass completed:

- experiment archaeology;
- PANDA-vs-BRACS diagnosis;
- prioritized next-experiment plan;
- one controlled current-code frozen-default reproduction run for UNI seed2024;
- two architecture-frozen UNI uniform-sampling seed-robustness runs;
- one R50 frozen-default current-protocol comparison run.

No new official test result was opened during this recovery pass.

## Original BRACS baseline

From `artifacts/bracs_evaluation/aggregate_results.csv`:

| Feature | MIR-MIL official test macro AUC |
| --- | ---: |
| R50 | `0.7087 ± 0.0073` |
| UNI | `0.7694 ± 0.0356` |

## Best previous BRACS result before this optimization

Valid single-feature previous best MIR-MIL:

- R50 `capacity_potential256`: official test mean about `0.7137`.
- UNI `capacity_h384_s128`: official test mean about `0.7862`.

However, these capacity changes are not allowed for new optimization under the current architecture-frozen constraint. They remain historical evidence only.

Excluded exploratory result:

- R50+UNI fusion: `0.8151 ± 0.0093`.
- Excluded because it combines features and violates separate R50/UNI reporting.

## Current BRACS SOTA target

Local baseline matrix target:

- R50: ACMIL `0.7178 ± 0.0147`.
- UNI: CLAM-SB `0.8039 ± 0.0060`.
- Overall single-feature target: UNI CLAM-SB `0.8039 ± 0.0060`.

## Best new BRACS result in this recovery pass

No accepted official-test improvement was found.

Completed controlled runs:

| Experiment | Feature | Seed | Split | Best val macro AUC | Best epoch | Test |
| --- | --- | ---: | --- | ---: | ---: | --- |
| `BRACS_REPRO_DEFAULT_UNI_SEED2024_PT` | UNI | 2024 | official train/val PT split | `0.798629` | 9 | not opened |
| `BRACS_UNI_SAMPLING_UNIFORM_SEED2024` | UNI | 2024 | official train/val PT split | `0.810399` | 10 | not opened |
| `BRACS_UNI_SAMPLING_UNIFORM_SEED2025` | UNI | 2025 | official train/val PT split | `0.794504` | 7 | not opened |
| `BRACS_REPRO_DEFAULT_R50_SEED2024_PT` | R50 | 2024 | official train/val PT split | `0.753742` | 28 | not opened |

Interpretation:

- Uniform training subsampling produced a small validation improvement over the current-code random reproduction.
- The seed2025 robustness run did not reproduce this gain.
- The two-seed validation mean is `0.802451` with sample std `0.011239`.
- Uniform sampling is therefore not an accepted BRACS improvement.
- R50 under the same current frozen default protocol is much weaker than UNI and does not explain BRACS recovery.

## Whether BRACS SOTA was reached

No.

No new official test evaluation was run. The uniform-sampling result improved validation on seed2024 but failed the seed2025 robustness check and cannot support a SOTA claim.

## Whether improvement came from R50, UNI, or both

No improvement was established.

Historical observations:

- R50 MIR-MIL is closer to R50 SOTA but still below ACMIL.
- UNI MIR-MIL has a larger gap to UNI CLAM-SB.
- UNI helps BRACS overall, but MIR-MIL does not exploit UNI as well as competing baselines on BRACS.
- Under the current frozen protocol, R50 seed2024 validation macro AUC is only `0.753742`, while UNI seed2024 is `0.798629`.

## Which changes helped most

In historical artifacts:

- UNI capacity `hidden_dim=384, sketch_dim=128, potential_hidden_dim=192` improved validation but not enough on test.
- R50 `potential_hidden_dim=256` slightly improved MIR-MIL R50 test but not enough to beat ACMIL.
- R50+UNI fusion helped test but is excluded from primary comparison.

In the architecture-frozen recovery pass:

- UNI `sampling: uniform` improved validation from `0.798629` to `0.810399` on seed2024, but seed2025 dropped to `0.794504`; it is not accepted as robust.

## Which changes did not help

From existing artifacts and current recovery:

- EMA;
- SWA screen;
- weighted/focal CE screen;
- ordinal loss;
- `moment_order=2`;
- strong regularization around h384;
- temporary instance-loss screen;
- several potential/prototype variants;
- current-code default reproduction did not improve validation.
- uniform sampling failed initial seed robustness and is not counted as accepted.
- R50 frozen default under the same current protocol is not competitive with UNI.

## Feature extractor status

Feature extractors remained unchanged:

- no R50 re-extraction;
- no UNI re-extraction;
- no fine-tuning;
- no raw WSI training.

The aborted H5 reproduction used a mirror of the same features but was discarded because the original baseline protocol used PT split paths. The completed reproduction used the original BRACS UNI PT split.

## Model architecture status

No accepted new BRACS improvement changes the model architecture.

The working tree currently contains default-off utilities from prior exploration:

- class weighting/focal CE support;
- SWA support;
- HPO variant definitions;
- PT split construction script.

These are not claimed as architecture improvements or accepted BRACS gains. Future accepted experiments must use new config files and keep architecture parameters unchanged.

## Exact reproduction command for the completed recovery run

Frozen default:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python train_mil.py \
  --yaml_path configs/experiments/MIR_MIL_BRACS_FROZEN_DEFAULT_UNI.yaml
```

Uniform-sampling candidate:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python train_mil.py \
  --yaml_path configs/experiments/MIR_MIL_BRACS_FROZEN_UNI_SAMPLING_UNIFORM.yaml
```

Uniform-sampling seed2025 robustness:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python train_mil.py \
  --yaml_path configs/experiments/MIR_MIL_BRACS_FROZEN_UNI_SAMPLING_UNIFORM_SEED2025.yaml
```

R50 frozen default comparison:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python train_mil.py \
  --yaml_path configs/experiments/MIR_MIL_BRACS_FROZEN_DEFAULT_R50.yaml
```

Config:

```text
configs/experiments/MIR_MIL_BRACS_FROZEN_DEFAULT_UNI.yaml
```

Output:

```text
artifacts/bracs_deep_opt/repro_default_uni_seed2024_pt/BRACS/MIR_MIL/time_2026-07-10-06-52_BRACS_MIR_MIL_seed_2024/
```

Best log:

```text
Best_Log_seed2024_BRACS_MIR_MIL.csv
```

## Reviewer challenge

Was the improvement selected using validation performance?

- Uniform sampling was evaluated by validation only. It has not been tested on the official test set.

Was the test set used?

- No new test evaluation was opened in this pass.

Is the result stable across seeds?

- No. Uniform sampling improved seed2024 but failed to reproduce on seed2025.

Is the comparison fair?

- The corrected completed run used the original official UNI train/val PT split. The aborted H5 attempt is discarded.

Did hidden config changes occur?

- A new explicit config was created. It disables optional weighted/focal/SWA/fusion settings and keeps default architecture dimensions.

Can this result be reproduced with one command?

- Yes, commands above.

Would this convince a paper reviewer?

- It would not support a SOTA claim. It is useful only as negative reproducibility evidence and a warning to audit config drift before tuning.

## Remaining limitations

1. Current-code default reproduction is lower than expected historical default validation. Need exact diff against the original BRACS baseline `MIR_MIL.yaml` saved under artifacts.
2. Split/label/baseline metric audit passed; see `reports/bracs_protocol_audit.json`.
3. Config drift audit found the historical `default_refit` run used 460 train slides and a different schedule; see `reports/bracs_config_drift_audit.md`.
4. One architecture-frozen sampling experiment was run: `random -> uniform` train subsampling. It improved validation on seed2024 but failed seed2025 robustness.
5. Official test should remain closed until a validation-selected, multi-seed architecture-frozen setting improves.

## Next action

Do not run more random HPO.

Next high-value action:

1. Do not open official test for uniform sampling.
2. Next highest-value step is to test lower-risk schedule/early-stop alignment on UNI before trying more sampling variants.
3. Do not use the historical `default_refit` high validation result as an official baseline because it used a different training pool.
4. Keep the official test closed until a validation-selected setting survives seed robustness.
