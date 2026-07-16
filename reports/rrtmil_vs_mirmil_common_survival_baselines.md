# RRT-MIL reference vs MIRMIL survival common baselines

Date: 2026-07-16

This report compares only common survival baselines between the external RRT-MIL reference implementation and the MIRMIL integrated survival module.

## Common baseline mapping

| MIRMIL backbone | RRT-MIL reference model |
|---|---|
| `MEAN_MIL` | `MeanMIL` |
| `MAX_MIL` | `MaxMIL` |
| `AB_MIL` | `AttMIL` |

Not compared here:

- `MIR_MIL`: our method, not a common baseline.
- `RRT_MIL`: implementation names overlap, but the MIRMIL wrapper and the RRT-MIL reference implementation are not yet protocol-matched.
- RRT-MIL `TransMIL` / `RRTMIL`: currently OOM on KIRC full bags with 11GB GPUs.

## Protocol caveat

The current comparison is diagnostic, not a final fair paper comparison.

RRT-MIL reference:

- Uses its original fixed random 5-fold CV.
- Uses full patient bags from the `.pt` features.
- Selects best epoch per fold by validation C-index.

MIRMIL integrated survival:

- Uses our prepared train/val/test split.
- Uses `max_instances=4096` random sampling in the saved configs.
- Reports validation-selected test C-index on the held-out test split.

Therefore, the correct interpretation is:

- RRT reference higher than MIRMIL suggests our integrated survival protocol/code may be weaker.
- It does not yet prove a bug until the RRT reference models are run under the exact MIRMIL split and sampling protocol, or MIRMIL baselines are run under RRT's official 5-fold protocol.

## Current partial/full comparison

Machine-readable table:

`reports/rrtmil_vs_mirmil_common_survival_baselines.tsv`

Available RRT reference folds are parsed from completed `model_best_*.pth.tar` checkpoints. A row is complete when `rrt_folds_done=5`.

| Dataset | Feature | Endpoint | Baseline | MIRMIL val | MIRMIL test | RRT folds | RRT CV mean | RRT - MIRMIL val | RRT - MIRMIL test |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| KIRC | R50 | OS | MeanMIL | 0.5411 | 0.6043 | 5 | 0.6286 | +0.0875 | +0.0243 |
| KIRC | R50 | PFS | MeanMIL | 0.6988 | 0.6204 | 5 | 0.6625 | -0.0363 | +0.0421 |
| KIRC | UNI | OS | AttMIL | 0.7063 | 0.7759 | 5 | 0.7416 | +0.0353 | -0.0344 |
| KIRC | UNI | OS | MaxMIL | 0.7264 | 0.7187 | 5 | 0.6850 | -0.0414 | -0.0337 |
| KIRC | UNI | OS | MeanMIL | 0.6722 | 0.7478 | 5 | 0.7275 | +0.0553 | -0.0203 |
| KIRC | UNI | PFS | MeanMIL | 0.7530 | 0.7554 | 5 | 0.7679 | +0.0149 | +0.0125 |

BLCA and several KIRC Max/Att rows are still pending in the running RRT baseline queue.

## Early interpretation

The user's observation is partly correct but should be stated carefully.

Against MIRMIL validation C-index, RRT reference is often higher for shared baselines:

- KIRC R50 OS MeanMIL: +0.0875
- KIRC UNI OS MeanMIL: +0.0553
- KIRC UNI OS AttMIL: +0.0353
- KIRC UNI PFS MeanMIL: +0.0149

Against MIRMIL held-out test C-index, the picture is mixed:

- RRT is slightly higher for KIRC R50 OS MeanMIL and KIRC UNI PFS MeanMIL.
- RRT is lower than MIRMIL for KIRC UNI OS MeanMIL / AttMIL / MaxMIL.

This mixed result is expected because the protocols differ. RRT's value is 5-fold validation CV mean; MIRMIL's value is a single official held-out test split.

## Main technical gap found so far

The biggest protocol difference is not the model head alone:

1. RRT reference uses full bags.
2. MIRMIL integrated survival uses `max_instances=4096`.
3. RRT reference uses 5-fold CV over all patients.
4. MIRMIL uses one fixed train/val/test split.
5. RRT reference internally discretizes survival bins from all rows in the input CSV.
6. MIRMIL fits cutpoints on the train split and reuses them for val/test.

These differences are large enough to explain substantial metric movement.

## Immediate next step

For a defensible conclusion, run one matched-split diagnostic first:

- Dataset: KIRC UNI OS
- Baselines: `MeanMIL`, `MaxMIL`, `AttMIL`
- Protocol: RRT reference models, but forced to use MIRMIL train/val/test split
- Feature handling: unchanged
- Bag handling: first full bag if memory allows; then `max_instances=4096` if we want exact MIRMIL protocol parity

If RRT matched-split baselines remain consistently higher than the MIRMIL integrated baselines, then we should treat the RRT-MIL survival implementation as the reference and audit the MIRMIL survival module for:

- risk sign convention;
- censorship/event encoding;
- discrete label cutpoints;
- validation/test metric computation;
- random sampling at train/val/test;
- survival head representation extraction;
- scheduler/optimizer differences.

