# RRT-MIL reference vs MIRMIL survival common baselines

Date: 2026-07-17

This report compares shared survival baselines between the external RRT-MIL reference implementation and the MIRMIL integrated survival module.

## Common baseline mapping

| MIRMIL backbone | RRT-MIL reference model |
|---|---|
| `MEAN_MIL` | `MeanMIL` |
| `MAX_MIL` | `MaxMIL` |
| `AB_MIL` | `AttMIL` |

Not compared as common baselines: `MIR_MIL` (our method), MIRMIL `RRT_MIL` wrapper, and RRT reference `TransMIL`/`RRTMIL` because the full-bag official run OOMed on the available 11GB GPUs.

## Protocols

- RRT official: original fixed random 5-fold CV, full bags, best epoch by fold validation C-index.
- RRT matched split: RRT reference model/loss, forced onto the MIRMIL train/val/test split, `max_instances=4096`, validation/test deterministic sampling, train deterministic sampled cache, best epoch by validation C-index.
- MIRMIL integrated: existing MIRMIL survival runs, `max_instances=4096`, train/val/test split, best run selected by validation C-index within the same common baseline.

Machine-readable tables:

- `reports/rrtmil_official_5fold_survival_results.tsv`
- `reports/rrtmil_matched_split_survival_results.tsv`
- `reports/rrtmil_vs_mirmil_common_survival_baselines.tsv`

## Matched-split result summary

| Dataset | Feature | Endpoint | Baseline | MIRMIL val | MIRMIL test | RRT matched val | RRT matched test | Δ val | Δ test | RRT official CV |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| BLCA | R50 | OS | AttMIL | 0.5366 | 0.5282 | 0.5054 | 0.5711 | -0.0312 | 0.0429 | 0.5782 |
| BLCA | R50 | OS | MaxMIL | 0.5177 | 0.5789 | 0.5810 | 0.4433 | 0.0633 | -0.1356 | 0.5565 |
| BLCA | R50 | OS | MeanMIL | NA | NA | 0.5343 | 0.5836 | NA | NA | 0.6002 |
| BLCA | UNI | OS | AttMIL | 0.5372 | 0.6322 | 0.5022 | 0.4815 | -0.0350 | -0.1507 | 0.6281 |
| BLCA | UNI | OS | MaxMIL | 0.5504 | 0.6579 | 0.5684 | 0.6218 | 0.0179 | -0.0361 | 0.5897 |
| BLCA | UNI | OS | MeanMIL | 0.5145 | 0.6322 | 0.5501 | 0.5805 | 0.0356 | -0.0517 | 0.6129 |
| KIRC | R50 | OS | AttMIL | 0.7019 | 0.7604 | 0.6087 | 0.6693 | -0.0932 | -0.0912 | 0.6263 |
| KIRC | R50 | OS | MaxMIL | 0.6626 | 0.5844 | 0.6445 | 0.5781 | -0.0180 | -0.0063 | 0.6399 |
| KIRC | R50 | OS | MeanMIL | 0.5411 | 0.6043 | 0.5555 | 0.6188 | 0.0144 | 0.0145 | 0.6303 |
| KIRC | R50 | PFS | AttMIL | 0.6821 | 0.6252 | 0.6929 | 0.6461 | 0.0108 | 0.0209 | 0.6652 |
| KIRC | R50 | PFS | MaxMIL | 0.5335 | 0.5391 | 0.7136 | 0.6129 | 0.1801 | 0.0738 | 0.6548 |
| KIRC | R50 | PFS | MeanMIL | 0.6988 | 0.6204 | 0.6998 | 0.6256 | 0.0010 | 0.0053 | 0.6634 |
| KIRC | UNI | OS | AttMIL | 0.7063 | 0.7759 | 0.7153 | 0.7692 | 0.0090 | -0.0068 | 0.7416 |
| KIRC | UNI | OS | MaxMIL | 0.7264 | 0.7187 | 0.6646 | 0.6877 | -0.0618 | -0.0310 | 0.6850 |
| KIRC | UNI | OS | MeanMIL | 0.6722 | 0.7478 | 0.6847 | 0.7546 | 0.0125 | 0.0068 | 0.7275 |
| KIRC | UNI | PFS | AttMIL | 0.7411 | 0.7387 | 0.7421 | 0.7537 | 0.0010 | 0.0149 | 0.7746 |
| KIRC | UNI | PFS | MaxMIL | 0.7431 | 0.7290 | 0.7028 | 0.6618 | -0.0404 | -0.0672 | 0.7177 |
| KIRC | UNI | PFS | MeanMIL | 0.7530 | 0.7554 | 0.7274 | 0.7370 | -0.0256 | -0.0183 | 0.7679 |

## Interpretation

- Matched-split comparison has 17 rows with both MIRMIL and RRT matched values.
- RRT matched validation is higher in 10/17 shared-baseline rows.
- RRT matched test-at-best-val is higher in 7/17 shared-baseline rows.
- Therefore the earlier observation that the external RRT-MIL project is generally stronger is true for its official 5-fold/full-bag protocol, but under the MIRMIL split and 4096-instance protocol the gap is mixed rather than uniformly positive.
- Large BLCA disagreement remains unstable because BLCA has fewer events per split and the held-out test C-index moves strongly with validation epoch selection.
- KIRC UNI remains the clearest setting where the RRT reference baselines are strong and consistent.

## OOM note

RRT reference `TransMIL` and `RRTMIL` official full-bag runs failed with CUDA OOM on the available 11GB GPUs. They should be rerun only under sampled matched-split protocol or on larger-memory GPUs if full-bag official comparability is required.
