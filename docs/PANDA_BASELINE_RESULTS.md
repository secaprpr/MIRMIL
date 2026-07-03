# PANDA Baseline Benchmark

Status: complete, July 3, 2026.

## Protocol

The benchmark uses the fixed QC split of 10,615 PANDA slides: 6,369 train,
2,123 validation, and a sealed 2,123-slide test set. Models were selected on
validation data and evaluated on test only after all training runs completed.
Each entry is the mean and sample standard deviation over seeds 2024, 2025,
and 2026. All models use a 4,096-instance uniform sampling budget.

The matrix contains 72 training and 72 test evaluations:

- features: ImageNet R50 and UNI;
- models: ABMIL, ACMIL, C2Aug, CLAM-MB, CLAM-SB, DSMIL, DTFD-MIL, 2DMamba,
  MIR-MIL, RRTMIL, TransMIL, and WiKG;
- metrics: accuracy, balanced accuracy, multiclass macro one-vs-rest AUC, and
  macro F1.

## Sealed Test Results

| Feature | Model | ACC | BACC | Macro AUC | Macro F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| R50 | ABMIL | .6112 ± .0019 | .5523 ± .0058 | .8845 ± .0024 | .5447 ± .0075 |
| R50 | ACMIL | .6150 ± .0083 | .5652 ± .0103 | .8878 ± .0042 | .5588 ± .0111 |
| R50 | C2Aug | .5346 ± .0078 | .4504 ± .0138 | .8321 ± .0025 | .4398 ± .0282 |
| R50 | CLAM-MB | .6326 ± .0082 | .5794 ± .0052 | .8912 ± .0021 | .5715 ± .0071 |
| R50 | CLAM-SB | .6100 ± .0059 | .5491 ± .0143 | .8846 ± .0013 | .5506 ± .0124 |
| R50 | DSMIL | .5853 ± .0073 | .5290 ± .0086 | .8730 ± .0013 | .5227 ± .0134 |
| R50 | DTFD-MIL | .6109 ± .0098 | .5475 ± .0096 | .8840 ± .0014 | .5482 ± .0069 |
| R50 | 2DMamba | .5698 ± .0052 | .5157 ± .0124 | .8610 ± .0064 | .5108 ± .0140 |
| R50 | MIR-MIL | **.6477 ± .0078** | **.6043 ± .0070** | **.8992 ± .0007** | **.6019 ± .0074** |
| R50 | RRTMIL | .6350 ± .0031 | .5744 ± .0051 | .8954 ± .0022 | .5720 ± .0054 |
| R50 | TransMIL | .6131 ± .0118 | .5442 ± .0172 | .8812 ± .0032 | .5364 ± .0220 |
| R50 | WiKG | .6221 ± .0110 | .5636 ± .0041 | .8930 ± .0036 | .5648 ± .0073 |
| UNI | ABMIL | .7332 ± .0182 | .6920 ± .0129 | .9366 ± .0031 | .6925 ± .0144 |
| UNI | ACMIL | .7196 ± .0267 | .6869 ± .0229 | .9294 ± .0027 | .6820 ± .0235 |
| UNI | C2Aug | .6718 ± .0205 | .5901 ± .0219 | .9017 ± .0050 | .5893 ± .0220 |
| UNI | CLAM-MB | .7513 ± .0154 | .7058 ± .0104 | .9428 ± .0034 | .7059 ± .0159 |
| UNI | CLAM-SB | .7467 ± .0097 | .6971 ± .0152 | .9369 ± .0011 | .6974 ± .0142 |
| UNI | DSMIL | .7084 ± .0143 | .6691 ± .0036 | .9317 ± .0013 | .6654 ± .0047 |
| UNI | DTFD-MIL | .6865 ± .0122 | .6269 ± .0179 | .9142 ± .0039 | .6251 ± .0165 |
| UNI | 2DMamba | .6935 ± .0191 | .6384 ± .0066 | .9138 ± .0034 | .6341 ± .0079 |
| UNI | MIR-MIL | **.7905 ± .0077** | **.7460 ± .0074** | **.9504 ± .0011** | **.7495 ± .0083** |
| UNI | RRTMIL | .7318 ± .0075 | .6892 ± .0072 | .9365 ± .0006 | .6880 ± .0040 |
| UNI | TransMIL | .6437 ± .0257 | .5871 ± .0040 | .9089 ± .0026 | .5725 ± .0285 |
| UNI | WiKG | .7672 ± .0068 | .7276 ± .0044 | .9417 ± .0017 | .7286 ± .0061 |

MIR-MIL is first on all four reported metrics with both feature encoders.
UNI improves MIR-MIL macro AUC from .8992 to .9504 and accuracy from .6477
to .7905.

## Reproducibility and Audit

Local machine-readable outputs are under the ignored
`artifacts/panda_evaluation/` directory:

```text
seed_results.csv
aggregate_results.csv
<feature>/<model>/seed<seed>/predictions.csv
<feature>/<model>/seed<seed>/result.json
```

The final audit found 72 result files and 72 prediction files. Every
prediction file has exactly 2,123 rows, finite probabilities, and probability
rows summing to one. The aggregate contains 24 rows with exactly three seeds
per row. Result records include the SHA-256 hashes of their split manifest
and selected checkpoint.

Training, sealed evaluation, prediction artifacts, and the 24 aggregate runs
are recorded in the `MIR-MIL` W&B project. Model checkpoints remain local and
were not uploaded.
