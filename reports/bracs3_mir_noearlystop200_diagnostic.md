# BRACS3 MIR-MIL no-early-stop 200 epoch diagnostic

This diagnostic keeps MIR-MIL architecture and R50/UNI features unchanged. Early stopping is disabled, training runs for 200 epochs, and the best validation macro-AUC checkpoint is evaluated once on the official test split.

## Validation peak during 200 epochs

| feature | seed | best_epoch | best_val_auc | last_epoch_val_auc |
|---|---:|---:|---:|---:|
| r50 | 2024 | 20 | 0.8914 | 0.8545 |
| r50 | 2025 | 12 | 0.9090 | 0.8576 |
| r50 | 2026 | 25 | 0.8953 | 0.8538 |
| uni | 2024 | 10 | 0.8869 | 0.8425 |
| uni | 2025 | 10 | 0.9096 | 0.8535 |
| uni | 2026 | 8 | 0.8840 | 0.8162 |

## Official test comparison

| setting | feature | test macro-AUC | test macro-F1 | acc | bacc |
|---|---|---:|---:|---:|---:|
| earlystop30_patience8 | r50 | 0.7570±0.0120 | 0.4951±0.0765 | 0.5287±0.0345 | 0.5226±0.0545 |
| noearlystop200_bestval | r50 | 0.7743±0.0077 | 0.5850±0.0275 | 0.5939±0.0351 | 0.5966±0.0254 |
| earlystop30_patience8 | uni | 0.8280±0.0277 | 0.5727±0.0799 | 0.6284±0.0351 | 0.5939±0.0520 |
| noearlystop200_bestval | uni | 0.8403±0.0184 | 0.5991±0.0423 | 0.6322±0.0230 | 0.6055±0.0353 |

Interpretation: disabling early stopping and allowing longer training improves MIR-MIL official test performance when selecting the best validation checkpoint, especially for UNI. The validation peak still occurs early (roughly epoch 8-25), and the final 200th epoch is worse, so the gain is from a later/better checkpoint search window, not from using the final epoch.
