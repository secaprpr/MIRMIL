# MIR-MIL Implementation

Status: initial reproducible prototype, June 13, 2026.

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

Those claims require fresh, sealed experiments and annotation-based
faithfulness evaluation.
