# BRACS3 baseline official test matrix

Evaluation uses the official BRACS full split with held-out test slides. Checkpoints were selected by validation macro-AUC, then evaluated once on test. Feature extractors and pre-extracted features were unchanged.

Output artifacts:

- `artifacts/bracs3_evaluation_nomamba/seed_results.csv`
- `artifacts/bracs3_evaluation_nomamba/aggregate_results.csv`
- `reports/bracs3_baseline_matrix.tsv`
- `reports/bracs3_baseline_seed_results.tsv`

Note: MAMBA2D is excluded from this main matrix because `r50/MAMBA2D_MIL/seed2024` was an incomplete long-runtime run. The main matrix contains 10 non-MAMBA baselines × 2 features × 3 seeds = 60 completed evaluations.

## Aggregate official test results

| feature | model | seeds | acc | bacc | macro_auc | macro_f1 |
|---|---|---:|---:|---:|---:|---:|
| uni | AC_MIL | 3 | 0.6820±0.0332 | 0.6670±0.0471 | 0.8529±0.0097 | 0.6620±0.0476 |
| uni | AB_MIL | 3 | 0.6398±0.0289 | 0.6383±0.0218 | 0.8434±0.0140 | 0.6356±0.0257 |
| uni | TRANS_MIL | 3 | 0.6705±0.0863 | 0.6416±0.1198 | 0.8430±0.0437 | 0.6071±0.1497 |
| uni | DTFD_MIL | 3 | 0.6169±0.0066 | 0.6256±0.0123 | 0.8408±0.0229 | 0.6171±0.0055 |
| uni | CLAM_MB_MIL | 3 | 0.6207±0.0640 | 0.6223±0.0449 | 0.8366±0.0329 | 0.6030±0.0636 |
| uni | CLAM_SB_MIL | 3 | 0.5977±0.0527 | 0.5960±0.0267 | 0.8335±0.0127 | 0.5771±0.0425 |
| uni | MIR_MIL | 3 | 0.6284±0.0351 | 0.5939±0.0520 | 0.8280±0.0277 | 0.5727±0.0799 |
| uni | DS_MIL | 3 | 0.6552±0.0460 | 0.6386±0.0364 | 0.8210±0.0200 | 0.6330±0.0411 |
| uni | WIKG_MIL | 3 | 0.6513±0.0133 | 0.6419±0.0127 | 0.8209±0.0376 | 0.6430±0.0104 |
| uni | RRT_MIL | 3 | 0.6322±0.0501 | 0.6259±0.0442 | 0.8178±0.0219 | 0.6159±0.0421 |
| r50 | CLAM_MB_MIL | 3 | 0.6092±0.0199 | 0.6119±0.0139 | 0.7971±0.0098 | 0.6090±0.0145 |
| r50 | DTFD_MIL | 3 | 0.6169±0.0351 | 0.6283±0.0278 | 0.7966±0.0098 | 0.6041±0.0441 |
| r50 | TRANS_MIL | 3 | 0.5364±0.0518 | 0.5500±0.0368 | 0.7893±0.0301 | 0.4963±0.0673 |
| r50 | CLAM_SB_MIL | 3 | 0.5862±0.0115 | 0.5802±0.0446 | 0.7827±0.0203 | 0.5475±0.0473 |
| r50 | AB_MIL | 3 | 0.5939±0.0239 | 0.6048±0.0137 | 0.7737±0.0128 | 0.5753±0.0335 |
| r50 | AC_MIL | 3 | 0.5862±0.0414 | 0.5938±0.0350 | 0.7722±0.0107 | 0.5799±0.0350 |
| r50 | WIKG_MIL | 3 | 0.5517±0.0527 | 0.5163±0.0270 | 0.7627±0.0230 | 0.4787±0.0351 |
| r50 | RRT_MIL | 3 | 0.5594±0.0332 | 0.5382±0.0410 | 0.7596±0.0096 | 0.5161±0.0635 |
| r50 | MIR_MIL | 3 | 0.5287±0.0345 | 0.5226±0.0545 | 0.7570±0.0120 | 0.4951±0.0765 |
| r50 | DS_MIL | 3 | 0.5287±0.0345 | 0.5104±0.0537 | 0.7284±0.0229 | 0.4945±0.0584 |

## MIR-MIL position

- r50 + MIR_MIL: test macro-AUC 0.7570±0.0120, macro-F1 0.4951±0.0765.
- uni + MIR_MIL: test macro-AUC 0.8280±0.0277, macro-F1 0.5727±0.0799.

Best current main-matrix result by test macro-AUC: `uni + AC_MIL` with 0.8529±0.0097.
