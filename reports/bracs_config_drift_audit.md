# BRACS config drift audit

Date: 2026-07-10

Scope: read-only audit after the first controlled BRACS recovery run. No feature extraction, no model architecture change, and no official test evaluation were performed.

## Compared runs

Historical run:

```text
artifacts/bracs_mir_refit/seed2024/default_refit/BRACS/MIR_MIL/time_2026-07-05-08-24_BRACS_MIR_MIL_seed_2024/MIR_MIL.yaml
```

Current recovery reproduction:

```text
configs/experiments/MIR_MIL_BRACS_FROZEN_DEFAULT_UNI.yaml
```

## Key differences

| Factor | Historical `default_refit` | Current recovery reproduction | Interpretation |
|---|---:|---:|---|
| Split CSV | `artifacts/bracs_mir_refit/BRACS_uni_refit_train460_monitor65.csv` | official `BRACS_uni_split_official_train_val.csv` | Not comparable as a direct reproduction. |
| Train slides | 460 | 395 | Historical refit trains on 65 additional slides. |
| Validation slides | 65 | 65 | Same validation count and label distribution. |
| Test slides in CSV | 0 | 0 | Neither run opened official test during training. |
| Epochs | 10 | 30 | Different training horizon. |
| Early-stop patience | 100 | 8 | Different selection/stopping behavior. |
| Cosine `T_max` | 8 | 28 | Different LR schedule. |
| Workers | 4 | 2 | Operational difference only. |

Both configs keep the same MIR-MIL architectural hyperparameters relevant to the frozen-method constraint:

- `in_dim: 1024`
- `hidden_dim: 256`
- `sketch_dim: 128`
- `moment_order: 1`
- `num_local_routes: 12`
- `potential_type: adaptive_multiscale`
- `max_instances: 4096`
- `sampling: random`

## Split-count evidence

`BRACS_uni_refit_train460_monitor65.csv`:

- train: 460 slides, label counts `{0: 37, 1: 131, 2: 65, 3: 30, 4: 36, 5: 49, 6: 112}`
- val: 65 slides, label counts `{0: 10, 1: 11, 2: 9, 3: 6, 4: 8, 5: 9, 6: 12}`
- test: 0 slides

Official `BRACS_uni_split_official_train_val.csv`:

- train: 395 slides, label counts `{0: 27, 1: 120, 2: 56, 3: 24, 4: 28, 5: 40, 6: 100}`
- val: 65 slides, label counts `{0: 10, 1: 11, 2: 9, 3: 6, 4: 8, 5: 9, 6: 12}`
- test: 0 slides

Official `BRACS_uni_split_official_full.csv`:

- train: 395 slides
- val: 65 slides
- test: 87 slides

## Conclusion

The current recovery run with val macro AUC `0.798629` is not a strict reproduction of the historical `default_refit` seed2024 run with val macro AUC `0.997531`, because the historical refit used a different training pool and different optimization schedule.

The historical `default_refit` result should not be treated as an official BRACS test baseline. It is a refit/monitor result useful for understanding whether adding more training slides makes validation easier, but it is not directly comparable to official train/val/test baseline results.

For disciplined BRACS optimization, use the official 395/65 train/val protocol for validation-driven selection and reserve the 87-slide official test set for final evaluation only.

## Next action

The next controlled experiment should use the official train/val split and test one architecture-frozen hypothesis at a time. The highest-priority hypothesis remains BRACS sampling instability caused by very large bags under `max_instances: 4096`.
