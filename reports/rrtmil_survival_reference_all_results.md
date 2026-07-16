# RRT-MIL survival reference experiment completion report

Date: 2026-07-17

Scope: external RRT-MIL survival code run in the isolated `rrtmil-surv` environment, using the already-prepared MIRMIL R50/UNI feature files. No MIR project environment, feature extractor, or raw WSI pipeline was modified.

## Completed experiment sets

- Official RRT-MIL 5-fold full-bag common baselines: `MeanMIL`, `MaxMIL`, `AttMIL` completed for BLCA OS and KIRC OS/PFS with R50/UNI.
- Official RRT-MIL 5-fold full-bag `TransMIL`/`RRTMIL`: attempted but failed with CUDA OOM on the available 11GB GPUs.
- Matched-split sampled RRT-MIL reference runs: all 30 combinations completed (`MeanMIL`, `MaxMIL`, `AttMIL`, `TransMIL`, `RRTMIL` × BLCA/KIRC endpoint-feature combinations), `max_instances=4096`, seed=1.

Machine-readable outputs:

- `reports/rrtmil_official_5fold_survival_results.tsv`
- `reports/rrtmil_official_5fold_survival_failures.tsv`
- `reports/rrtmil_matched_split_survival_results.tsv`
- `reports/rrtmil_vs_mirmil_common_survival_baselines.tsv`

## Matched-split sampled results

| Dataset | Feature | Endpoint | Model | Best val C-index | Test C-index at best val | Best epoch |
|---|---|---|---|---:|---:|---:|
| BLCA | R50 | OS | AttMIL | 0.5054 | 0.5711 | 18 |
| BLCA | R50 | OS | MaxMIL | 0.5810 | 0.4433 | 9 |
| BLCA | R50 | OS | MeanMIL | 0.5343 | 0.5836 | 9 |
| BLCA | R50 | OS | RRTMIL | 0.6484 | 0.5078 | 1 |
| BLCA | R50 | OS | TransMIL | 0.5236 | 0.5949 | 18 |
| BLCA | UNI | OS | AttMIL | 0.5022 | 0.4815 | 18 |
| BLCA | UNI | OS | MaxMIL | 0.5684 | 0.6218 | 11 |
| BLCA | UNI | OS | MeanMIL | 0.5501 | 0.5805 | 1 |
| BLCA | UNI | OS | RRTMIL | 0.5394 | 0.6274 | 0 |
| BLCA | UNI | OS | TransMIL | 0.5703 | 0.5479 | 1 |
| KIRC | R50 | OS | AttMIL | 0.6087 | 0.6693 | 19 |
| KIRC | R50 | OS | MaxMIL | 0.6445 | 0.5781 | 11 |
| KIRC | R50 | OS | MeanMIL | 0.5555 | 0.6188 | 19 |
| KIRC | R50 | OS | RRTMIL | 0.6472 | 0.6169 | 13 |
| KIRC | R50 | OS | TransMIL | 0.6218 | 0.5567 | 15 |
| KIRC | R50 | PFS | AttMIL | 0.6929 | 0.6461 | 5 |
| KIRC | R50 | PFS | MaxMIL | 0.7136 | 0.6129 | 6 |
| KIRC | R50 | PFS | MeanMIL | 0.6998 | 0.6256 | 4 |
| KIRC | R50 | PFS | RRTMIL | 0.7165 | 0.7067 | 15 |
| KIRC | R50 | PFS | TransMIL | 0.6624 | 0.7292 | 3 |
| KIRC | UNI | OS | AttMIL | 0.7153 | 0.7692 | 6 |
| KIRC | UNI | OS | MaxMIL | 0.6646 | 0.6877 | 16 |
| KIRC | UNI | OS | MeanMIL | 0.6847 | 0.7546 | 17 |
| KIRC | UNI | OS | RRTMIL | 0.6873 | 0.7798 | 2 |
| KIRC | UNI | OS | TransMIL | 0.6943 | 0.7032 | 0 |
| KIRC | UNI | PFS | AttMIL | 0.7421 | 0.7537 | 1 |
| KIRC | UNI | PFS | MaxMIL | 0.7028 | 0.6618 | 3 |
| KIRC | UNI | PFS | MeanMIL | 0.7274 | 0.7370 | 4 |
| KIRC | UNI | PFS | RRTMIL | 0.7589 | 0.6569 | 8 |
| KIRC | UNI | PFS | TransMIL | 0.7490 | 0.6725 | 7 |

## Best matched-split result per endpoint-feature

| Dataset | Feature | Endpoint | Best by val | Val | Test at best val | Best by test (diagnostic only) | Test |
|---|---|---|---|---:|---:|---|---:|
| BLCA | R50 | OS | RRTMIL | 0.6484 | 0.5078 | TransMIL | 0.5949 |
| BLCA | UNI | OS | TransMIL | 0.5703 | 0.5479 | RRTMIL | 0.6274 |
| KIRC | R50 | OS | RRTMIL | 0.6472 | 0.6169 | AttMIL | 0.6693 |
| KIRC | R50 | PFS | RRTMIL | 0.7165 | 0.7067 | TransMIL | 0.7292 |
| KIRC | UNI | OS | AttMIL | 0.7153 | 0.7692 | RRTMIL | 0.7798 |
| KIRC | UNI | PFS | RRTMIL | 0.7589 | 0.6569 | AttMIL | 0.7537 |

## Official full-bag common baseline summary

| Dataset | Feature | Endpoint | Model | CV mean C-index | CV std |
|---|---|---|---|---:|---:|
| BLCA | R50 | OS | AttMIL | 0.5782 | 0.0320 |
| BLCA | R50 | OS | MaxMIL | 0.5565 | 0.0742 |
| BLCA | R50 | OS | MeanMIL | 0.6002 | 0.0395 |
| BLCA | UNI | OS | AttMIL | 0.6281 | 0.0535 |
| BLCA | UNI | OS | MaxMIL | 0.5897 | 0.0590 |
| BLCA | UNI | OS | MeanMIL | 0.6129 | 0.0568 |
| KIRC | R50 | OS | AttMIL | 0.6263 | 0.0495 |
| KIRC | R50 | OS | MaxMIL | 0.6399 | 0.0343 |
| KIRC | R50 | OS | MeanMIL | 0.6303 | 0.0452 |
| KIRC | R50 | PFS | AttMIL | 0.6652 | 0.0598 |
| KIRC | R50 | PFS | MaxMIL | 0.6548 | 0.0463 |
| KIRC | R50 | PFS | MeanMIL | 0.6634 | 0.0481 |
| KIRC | UNI | OS | AttMIL | 0.7416 | 0.0159 |
| KIRC | UNI | OS | MaxMIL | 0.6850 | 0.0437 |
| KIRC | UNI | OS | MeanMIL | 0.7275 | 0.0198 |
| KIRC | UNI | PFS | AttMIL | 0.7746 | 0.0633 |
| KIRC | UNI | PFS | MaxMIL | 0.7177 | 0.0488 |
| KIRC | UNI | PFS | MeanMIL | 0.7679 | 0.0655 |

## Official full-bag failures

- Failed official full-bag tasks: 12.
- Failure mode: CUDA OOM for `TransMIL`/`RRTMIL` on full bags with the available 11GB GPUs.


## Interpretation

- The external RRT-MIL project does show higher survival performance under its own official full-bag 5-fold protocol, especially on KIRC UNI.
- Under the MIRMIL matched split and 4096-instance sampling, the gap is mixed: RRT reference common baselines are not uniformly better than the MIRMIL integrated common baselines.
- This points to protocol/data-handling differences as a major source of the apparent performance gap, not only model definition.
- For paper-quality survival claims, use validation-selected results and avoid selecting by held-out test. The `Best by test` column above is diagnostic only.
