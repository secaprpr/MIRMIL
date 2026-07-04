# BRACS Baseline Benchmark

Status: complete, July 5, 2026.

## Protocol

The benchmark preserves the official BRACS WSI split from `BRACS.xlsx`:
395 train, 65 validation, and 87 test slides. The seven-class label order is
`N, PB, UDH, FEA, ADH, DCIS, IC`. The known Patient 67 overlap between the
official train and validation sets is retained because this experiment
targets direct comparability with the official split.

Each result is the mean and sample standard deviation over seeds 2024, 2025,
and 2026. Validation macro AUC selects the checkpoint. Test is evaluated only
from the selected checkpoint. Both R50 and UNI use the same 18,648,610
level-0 patch coordinates.

All models use a 4,096-instance bag budget. BRACS 2DMamba uses a physical
coordinate grid with `coord_scale=1024` and `d_model=256`; this prevents the
sparse 40x WSI coordinate span from expanding beyond 11 GiB GPU memory.
The MIR-MIL architecture was not changed.

## Official Test Results

| Feature | Model | ACC | BACC | Macro AUC | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| R50 | ABMIL | .2797 ± .0404 | .2581 ± .0335 | .6737 ± .0229 | .2029 ± .0143 |
| R50 | ACMIL | .3180 ± .0782 | .3165 ± .0363 | **.7178 ± .0147** | **.2748 ± .0568** |
| R50 | C2Aug | .2720 ± .0765 | .1946 ± .0451 | .6312 ± .0446 | .1075 ± .0547 |
| R50 | CLAM-MB | .3180 ± .0133 | .2996 ± .0070 | .6989 ± .0177 | .2581 ± .0153 |
| R50 | CLAM-SB | .2720 ± .0567 | .2373 ± .0149 | .6710 ± .0103 | .2064 ± .0195 |
| R50 | DSMIL | .2835 ± .0351 | .2600 ± .0346 | .7069 ± .0103 | .2119 ± .0097 |
| R50 | DTFD-MIL | .2759 ± .0414 | .2787 ± .0298 | .7151 ± .0088 | .2161 ± .0449 |
| R50 | 2DMamba | .2490 ± .0978 | .2397 ± .0728 | .6498 ± .0281 | .1864 ± .0912 |
| R50 | MIR-MIL | .2950 ± .0465 | .2964 ± .0479 | .7087 ± .0073 | .2723 ± .0356 |
| R50 | RRTMIL | .2069 ± .0414 | .1708 ± .0382 | .6656 ± .0138 | .1416 ± .0176 |
| R50 | TransMIL | .2720 ± .0807 | .2284 ± .0657 | .6881 ± .0443 | .2054 ± .0596 |
| R50 | WiKG | .3103 ± .0304 | .3083 ± .0015 | .7103 ± .0214 | .2709 ± .0391 |
| UNI | ABMIL | .4176 ± .0531 | .3879 ± .0554 | .7935 ± .0138 | **.3593 ± .0737** |
| UNI | ACMIL | **.4406 ± .0239** | **.4034 ± .0404** | .8024 ± .0083 | .3441 ± .0343 |
| UNI | C2Aug | .4291 ± .0289 | .3488 ± .0098 | .7707 ± .0086 | .3032 ± .0248 |
| UNI | CLAM-MB | .4291 ± .0265 | .3909 ± .0227 | .7993 ± .0104 | .3532 ± .0304 |
| UNI | CLAM-SB | .4291 ± .0239 | .3863 ± .0612 | **.8039 ± .0060** | .3525 ± .0610 |
| UNI | DSMIL | .4176 ± .0066 | .3511 ± .0266 | .7819 ± .0123 | .3354 ± .0246 |
| UNI | DTFD-MIL | .4061 ± .0066 | .3714 ± .0143 | .7876 ± .0128 | .3571 ± .0182 |
| UNI | 2DMamba | .2950 ± .0066 | .2506 ± .0051 | .6632 ± .0073 | .2188 ± .0198 |
| UNI | MIR-MIL | .3908 ± .0398 | .3352 ± .0304 | .7694 ± .0356 | .3263 ± .0266 |
| UNI | RRTMIL | .3908 ± .0699 | .3499 ± .0673 | .7933 ± .0244 | .2818 ± .0746 |
| UNI | TransMIL | .3716 ± .0543 | .3274 ± .0454 | .7530 ± .0146 | .2824 ± .0487 |
| UNI | WiKG | .4215 ± .0579 | .3809 ± .0539 | .7981 ± .0223 | .3448 ± .0636 |

UNI materially improves most baselines. MIR-MIL ranks fourth by macro AUC
with R50 and tenth with UNI; unlike the PANDA benchmark, it is not the
best-performing BRACS model.

## Audit and Outputs

The completed matrix contains 72 training runs, 72 test result files, and 72
prediction files. Every prediction file contains the same 87 official test
slides, seven finite class probabilities per slide, and probabilities that
sum to one. The aggregate has 24 rows and exactly three seeds per row.

Machine-readable outputs are ignored by Git and stored under:

```text
artifacts/bracs_baselines/
artifacts/bracs_evaluation/seed_results.csv
artifacts/bracs_evaluation/aggregate_results.csv
artifacts/bracs_evaluation/<feature>/<model>/seed<seed>/
```

Training, test evaluation, prediction artifacts, and aggregate summaries are
recorded in the `MIR-MIL` W&B project. Model weights remain local.
