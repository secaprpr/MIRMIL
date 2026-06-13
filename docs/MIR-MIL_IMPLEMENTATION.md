# MIR-MIL Implementation

Status: reproducible PANDA prototype, June 14, 2026.

MIR-MIL implements the revised Measure Influence Response method as a model
separate from OT-MIL. It does not use Sinkhorn transport.

## Method Mapping

| Method component | Implementation |
| --- | --- |
| Patch-state encoder \(E_\theta\) | `MIR_MIL.encoder` |
| Composition basis \(\Phi_\theta\) | `MIR_MIL.response_basis` |
| Tail scores \(s_{\theta,m}\) | `MIR_MIL.tail_scorer` |
| Neural potential \(V_\theta\) | `MIR_MIL.potential` |
| Class-margin score | `MIR_MIL.explained_score` |
| Closed-form response \(R_\theta\) | `measure_influence_response` |
| Functional derivative \(\phi_\theta\) | `functional_derivative` |
| Complete path attribution | `integrated_functional_attribution` |
| Finite-difference audit | `finite_difference_response` |
| Measure perturbation stability | `augment_bag` and `compute_loss` |

The repository's binary classifiers produce two logits. MIR-MIL therefore
explains the two-class margin \(f_c-f_{\bar c}\), which is the binary special
case of the multiclass class-margin definition.

Coordinates are optional. With `Model.coordinate_dim=0`, MIR-MIL consumes the
existing feature caches. With `Model.coordinate_dim=2`, source H5/PT files
must contain real coordinates; the dataset loader samples features and
coordinates jointly and normalizes each coordinate axis.

The optional Lipschitz penalty uses randomized directional derivatives of the
response basis. It is a practical smoothness regularizer, not an exact
spectral Jacobian norm.

## Verification

Automated tests verify:

- permutation invariance;
- zero mean of the centered response under the current empirical measure;
- agreement between the closed-form response and finite differences;
- completeness of integrated functional attribution;
- multiclass class-margin semantics;
- stability-loss backpropagation;
- coordinate-aware sampling;
- YAML and benchmark integration.

The initial real-feature smoke run used the sealed STAD train/validation split,
128 patches, seed 2024, and two epochs. Validation macro-AUC increased from
`0.7142` to `0.8042`. This is only an execution check, not a benchmark result.

On its best smoke checkpoint:

- mean centered response: `-3.35e-8`;
- finite-difference MAE over 16 sampled patches: `0.00116` at
  \(\epsilon=10^{-4}\);
- 65-point path completeness absolute error: `5.90e-5`.

The standalone evaluator was also run on three validation slides and 48
sampled patches. It produced Pearson `0.9999998`, Spearman `1.0`, mean absolute
error `0.00198`, and top-5 overlap `1.0`.

The complete automated suite contains 103 passing tests.

## PANDA Experiment

The formal experiment uses the repository's fixed six-class PANDA split and
ResNet-50 patch-feature cache:

- split SHA-256:
  `49a305255880bc9b93a0a1be82214d657aef2e99b7a7b5c083dda7ec68dcdf6a`;
- 5,763 training, 1,921 validation, and 1,920 sealed test slides;
- seeds 2024, 2025, and 2026;
- 512 patches per slide, balanced sampling, 30 epochs, patience 8;
- identical split, feature dimension, patch budget, and seed set to the
  historical MO-MIL and OT-MIL runs.

Training used a derived split with empty test columns. Frozen checkpoints were
evaluated on the sealed test split once after all three runs completed.

| Model | Macro-AUC | Accuracy | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| MO-MIL | 0.8996 +/- 0.0038 | 0.6363 | 0.5868 | 0.5887 |
| OT-MIL | **0.9051 +/- 0.0037** | **0.6462** | **0.6043** | **0.6032** |
| MIR-MIL | 0.8984 +/- 0.0008 | 0.6398 | 0.5969 | 0.5963 |

For paired stratified bootstrap with 10,000 iterations:

- MIR-MIL minus MO-MIL macro-AUC was `-0.00123`, with 95% CI
  `[-0.00450, 0.00205]`;
- MIR-MIL minus MO-MIL balanced accuracy was `+0.01014`, with 95% CI
  `[-0.00191, 0.02251]`;
- MIR-MIL minus OT-MIL macro-AUC was `-0.00678`, with 95% CI
  `[-0.00998, -0.00366]`.

Thus MIR-MIL is statistically indistinguishable from MO-MIL in macro-AUC but
is significantly worse than OT-MIL in macro-AUC under this protocol.

Faithfulness was audited independently for each seed on 100 sealed test slides
and 64 sampled patches per slide at epsilon `1e-4`:

| Seed | Pearson | Spearman | FD MAE | Top-10 overlap | Centered mean |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | 0.999997 | 0.999980 | 0.00928 | 0.999 | 8.03e-8 |
| 2025 | 0.999996 | 0.999979 | 0.00855 | 1.000 | 7.82e-8 |
| 2026 | 0.999992 | 0.999977 | 0.00996 | 0.999 | 6.80e-8 |

These results support the local contamination-response formula and numerical
centering. They do not show that high-response patches are clinically correct
or that MIR-MIL improves classification.

### Convergence Follow-up

The initial 30-epoch runs reached their best validation macro-AUC at epochs
26, 28, and 26. Because these optima were close to the training cap, a
validation-only convergence follow-up used a 60-epoch cap, patience 10, and a
cosine schedule with `T_max=58`. The sealed test split remained hidden.

| Seed | Best epoch | Stop epoch | Best validation AUC | BAcc | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | 35 | 45 | 0.90124 | 0.59310 | 0.59803 |
| 2025 | 32 | 42 | 0.90075 | 0.61004 | 0.61321 |
| 2026 | 43 | 53 | 0.90412 | 0.61166 | 0.61141 |

The mean best epoch moved from 26.7 to 36.7, and mean validation macro-AUC
increased from `0.89862` to `0.90204`. Therefore the original 30-epoch budget
was too short for MIR-MIL. A 60-epoch cap with patience 10 is sufficient to
observe natural stopping in the current setup; all seeds stopped between
epochs 42 and 53.

These longer-run checkpoints have not been evaluated on the sealed test set.
The convergence result should be treated as validation evidence for selecting
the future training budget, not as a revised test benchmark.

## Training

```bash
python train_mil.py --yaml_path configs/MIR_MIL.yaml --options \
  Dataset.dataset_csv_path=/path/to/train_val.csv \
  Dataset.DATASET_NAME=DATASET \
  Logs.log_root_dir=/path/to/logs \
  General.num_classes=4 \
  Model.in_dim=1536 \
  Model.max_instances=4096
```

Fair comparison with repository baselines:

```bash
python experiments/run_benchmark.py \
  --split /path/to/train_val.csv \
  --dataset-name DATASET --num-classes 4 \
  --log-root /path/to/logs \
  --models MIR_MIL AB_MIL MO_MIL \
  --seeds 2024 2025 2026 \
  --epochs 25 --patience 6 \
  --max-instances 4096 --in-dim 1536
```

The PANDA command was:

```bash
python experiments/run_benchmark.py \
  --split /home/sigirika/experiment_splits/mir_mil_v1/PANDA_R50_CACHE_train_val.csv \
  --dataset-name PANDA_R50_CACHE --num-classes 6 \
  --log-root /home/sigirika/experiment_logs/mir_mil_v1/panda_512 \
  --models MIR_MIL --seeds 2024 2025 2026 \
  --epochs 30 --patience 8 --max-instances 512 \
  --in-dim 1024 --device 0 --num-workers 4
```

## Faithfulness Evaluation

```bash
python experiments/evaluate_mir_faithfulness.py \
  --run-dir /path/to/MIR_MIL/time_RUN \
  --split /path/to/split.csv \
  --group val \
  --output-dir /path/to/faithfulness \
  --budget 512 \
  --max-slides 20 \
  --patches-per-slide 64 \
  --epsilon 1e-4 \
  --target predicted
```

The evaluator writes slide-level metrics, patch-level responses and finite
differences, plus checkpoint/split provenance hashes.

## Current Boundary

The implementation establishes mathematical and numerical reproducibility. It
does not yet establish:

- superior classification performance;
- better localization than attention, IG, occlusion, or pathology-specific
  explanation methods;
- rare-lesion robustness;
- clinical or causal interpretation.

The PANDA experiment is fresh and sealed, but does not support a
classification-superiority claim. Localization superiority, rare-lesion
robustness, and clinical interpretation still require annotation-based
evaluation and targeted ablations.
