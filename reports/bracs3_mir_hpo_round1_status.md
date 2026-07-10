# BRACS3 MIR-MIL HPO round1 status

Search used UNI features, 30 epochs, no early stopping, validation macro-AUC checkpoint selection. Test was evaluated only for the validation-selected top four variants.

## Validation top variants

| variant | seeds | val macro-AUC | val macro-F1 |
|---|---:|---:|---:|
| mild_reg | 3 | 0.9041±0.0224 | 0.6099±0.0799 |
| unbalanced_mild | 3 | 0.9027±0.0282 | 0.5628±0.0715 |
| loss_focal1 | 3 | 0.9007±0.0157 | 0.6770±0.0634 |
| smooth005 | 3 | 0.8981±0.0141 | 0.6511±0.0342 |
| smooth010 | 3 | 0.8957±0.0198 | 0.6903±0.0748 |
| mild_smooth005 | 3 | 0.8940±0.0060 | 0.6456±0.0804 |
| baseline_noes30 | 3 | 0.8881±0.0189 | 0.6580±0.0694 |

## Official test for validation-selected top4

| variant | test macro-AUC | test macro-F1 | acc | bacc |
|---|---:|---:|---:|---:|
| smooth005 | 0.8245±0.0092 | 0.5697±0.0431 | 0.6360±0.0239 | 0.5941±0.0302 |
| unbalanced_mild | 0.8220±0.0319 | 0.4982±0.0400 | 0.6245±0.0332 | 0.5687±0.0324 |
| loss_focal1 | 0.8214±0.0178 | 0.5452±0.0500 | 0.6245±0.0239 | 0.5796±0.0302 |
| mild_reg | 0.8115±0.0059 | 0.5368±0.0325 | 0.6245±0.0176 | 0.5782±0.0125 |

Current best MIR-MIL official-test result remains the previous UNI no-early-stop 200/30 best-val setting: 0.8403±0.0184 macro-AUC. Current internal best baseline remains UNI+AC_MIL: 0.8529±0.0097 macro-AUC. Literature ACMIL reference is approximately 0.888±0.010 macro-AUC, protocol-dependent.
