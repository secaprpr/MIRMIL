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

### Ordinal Geometry Follow-up

PANDA ISUP grades are ordered. MIR-MIL therefore supports an optional
one-dimensional squared Wasserstein loss, implemented as the squared distance
between predicted and target cumulative class distributions. The default
weight is zero, so unordered classification behavior is unchanged.

A validation-only seed-2024 sweep compared ordinal weights `0.1`, `0.3`, and
`1.0`. Weights `0.1` and `0.3` reached `66.53%` and `66.48%` validation
accuracy, while weight `1.0` fell back to `64.60%`. Weight `0.3` was selected
before opening the sealed test because it retained the higher macro-AUC.

The frozen confirmation protocol used weight `0.3`, an 80-epoch cap,
patience 12, accuracy checkpoint selection, and the original three seeds:

| Seed | Best epoch | Validation accuracy | Macro-AUC | BAcc | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2024 | 77 | 0.6804 | 0.9046 | 0.6456 | 0.6447 |
| 2025 | 36 | 0.6580 | 0.8989 | 0.6148 | 0.6142 |
| 2026 | 74 | 0.6757 | 0.9072 | 0.6427 | 0.6410 |
| Mean | - | **0.6714** | 0.9036 | 0.6344 | 0.6333 |

Two seeds reached their best accuracy close to the 80-epoch cap, so this
protocol did not establish convergence. A follow-up decoupled the optimization
horizon from the safety cap: cosine `T_max=138`, an 180-epoch cap, patience 20,
accuracy `min_delta=0.001`, and a scheduler that remains at `eta_min` after
`T_max`. The first 140 epochs exactly reproduce the corresponding native
cosine trajectory.

| Seed | Best epoch | Stop epoch | Validation accuracy | Macro-AUC | BAcc | Macro-F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2024 | 127 | 147 | 0.7022 | 0.9085 | 0.6682 | 0.6682 |
| 2025 | 48 | 68 | 0.6658 | 0.9013 | 0.6282 | 0.6265 |
| 2026 | 68 | 88 | 0.6830 | 0.9072 | 0.6460 | 0.6482 |
| Mean | - | - | **0.6837** | 0.9057 | 0.6475 | 0.6476 |

All three seeds now stop naturally. The practical recommendation is therefore
an 180-epoch safety cap rather than a fixed training length; observed stopping
occurred between epochs 68 and 147. Relative to the 80-epoch validation
protocol, mean accuracy increased by `1.23` percentage points.

An initial implementation of clamped cosine directly subclassed PyTorch's
scheduler. It interacted incorrectly with the existing warmup scheduler and
changed the first-epoch trajectory; its epoch-88 result was discarded. The
replacement wraps the native cosine scheduler and only suppresses steps after
`T_max`. Unit tests verify both the warmup trajectory and the absence of
post-horizon reheating.

After freezing this protocol, the same 1,920-slide sealed test split was
opened once with deterministic uniform 512-patch sampling. The split SHA256
was `49a305255880bc9b93a0a1be82214d657aef2e99b7a7b5c083dda7ec68dcdf6a`.

The historical 80-epoch ordinal checkpoints produced:

| Model | Macro-AUC | Accuracy | BAcc | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| MO-MIL | 0.8996 | 0.6363 | 0.5868 | 0.5887 |
| OT-MIL | 0.9051 | 0.6462 | 0.6043 | 0.6032 |
| MIR-MIL + ordinal geometry | **0.9055** | **0.6568** | **0.6203** | **0.6176** |

The converged MIR-MIL checkpoints produced:

| Seed | Macro-AUC | Accuracy | BAcc | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| 2024 | 0.9138 | 0.6885 | 0.6567 | 0.6575 |
| 2025 | 0.9059 | 0.6500 | 0.6187 | 0.6145 |
| 2026 | 0.9100 | 0.6635 | 0.6293 | 0.6319 |
| Mean | **0.9099** | **0.6674** | **0.6349** | **0.6346** |

Relative to the 80-epoch checkpoints, convergence improved mean macro-AUC by
`0.00438`, accuracy by `1.06` percentage points, BAcc by `1.47` points, and
macro-F1 by `1.70` points.

Paired stratified bootstrap with 10,000 iterations against the existing
frozen baseline predictions found:

- versus MO-MIL, macro-AUC improved by `0.01028`
  (`95% CI [0.00632, 0.01425]`) and accuracy by `0.03108`
  (`95% CI [0.01962, 0.04288]`);
- versus MO-MIL, BAcc improved by `0.04815`
  (`95% CI [0.03471, 0.06177]`) and macro-F1 by `0.04595`
  (`95% CI [0.03249, 0.05980]`);
- versus OT-MIL, macro-AUC improved by `0.00474`
  (`95% CI [0.00086, 0.00854]`) and accuracy by `0.02118`
  (`95% CI [0.00990, 0.03247]`);
- versus OT-MIL, BAcc improved by `0.03059`
  (`95% CI [0.01726, 0.04365]`) and macro-F1 by `0.03137`
  (`95% CI [0.01798, 0.04465]`).

These intervals establish improvement over the evaluated frozen baseline
checkpoints. They are not yet an equal-convergence-budget comparison: MO-MIL
and OT-MIL have not been rerun under the longer stopping protocol.

The 80-epoch ordinal checkpoints retained finite-difference faithfulness on
100 sealed test slides per seed: Pearson was `0.999987-0.999996`, Spearman was
about `0.99998`, top-10 overlap was at least `0.999`, and the centered response
mean remained on the order of `1e-7`. The converged checkpoints still require
the same audit.

This establishes the interim PANDA target of mean accuracy above 65% and
significant BAcc/F1 gains over the repository's strongest baseline. It does
not establish general SOTA or ICLR-level novelty. The ordinal term is a
task-appropriate auxiliary geometry for ordered labels and must be disabled
or replaced by another label geometry on unordered tasks.

### Local Routed Measure Follow-up

STAD molecular subtype classification exposed a limitation of the original
state: global composition and smooth maxima can indicate that an unusual
region exists, but do not retain the representation of that region. Two
generic alternatives did not solve this:

- replacing the potential with class-wise prototype mixtures reached at most
  `0.8297` validation macro-AUC;
- adding a global second central moment reached `0.8165`;
- the original MLP potential reached `0.8396` under the same seed-2024
  protocol.

MIR-MIL now optionally adds local routed measures. For route \(j\), a learned
score defines a density ratio \(q_j(x)\), and the state retains the conditional
local mean

\[
m_j(\mu)=\frac{\int \exp(s_j(x)/\tau)\ell(x)\,d\mu(x)}
{\int \exp(s_j(x)/\tau)\,d\mu(x)}.
\]

Its contamination response is exactly

\[
q_j(x)\left(\ell(x)-m_j(\mu)\right),
\]

so the local contribution remains centered and differentiable. Unit tests
verify finite-difference agreement and integrated-path completeness. Setting
`num_local_routes=0` recovers the previous model.

On the fixed STAD validation split, four routes of dimension 32 improved all
three seeds:

| Model | Macro-AUC | Accuracy | BAcc | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| MIR global state | 0.8248 +/- 0.0217 | 0.6418 | 0.5724 | 0.5579 |
| MIR + 4 local routes | **0.8696 +/- 0.0071** | **0.7662** | **0.7047** | **0.6978** |

The route count was selected before opening the sealed test. Frozen
checkpoints then produced:

| Model | Macro-AUC | Accuracy | BAcc | Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Historical AB-MIL | **0.8803** | 0.7150 | 0.6180 | 0.6328 |
| MIR + 4 local routes | 0.8622 | **0.7343** | 0.6127 | 0.6244 |

The local routes therefore improve MIR substantially and exceed AB-MIL in
accuracy, but do not establish STAD SOTA because macro-AUC remains lower.
Route diagnostics do not indicate collapse: mean pairwise route-weight cosine
overlap was `0.069`, normalized entropy was `0.643`, and each route covered
about 300 effective patches on average.

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

- consistent SOTA classification performance across datasets;
- better localization than attention, IG, occlusion, or pathology-specific
  explanation methods;
- rare-lesion robustness;
- clinical or causal interpretation.

PANDA and STAD now provide positive cross-task evidence, but neither alone
supports a general SOTA claim. Localization superiority, rare-lesion
robustness, and clinical interpretation still require annotation-based
evaluation and targeted ablations.

## Unified Adaptive Multiscale Evaluation

To keep the architecture identical across datasets, MIR now always uses four
local measure routes. A global-state gate controls a local residual potential
for each sample and class. Dataset-specific choices are limited to training
hyperparameters such as learning rate, regularization, patch budget, and early
stopping. The adaptive potential preserves the closed-form MIR response; the
finite-difference and attribution tests pass.

The first fixed-structure evaluation used UNI2-h features and patient-grouped
splits. Model selection used only macro-AUC on the visible validation set.
Sealed test columns were opened once after three-seed confirmation.

| Task | Split | Macro AUC | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| COAD CMS | validation | 0.8220 +/- 0.0058 | - | 0.5956 | 0.6046 |
| COAD CMS | sealed test | 0.7566 +/- 0.0077 | 0.5311 | 0.5209 | 0.5208 |
| RCC | validation | 0.9930 +/- 0.0003 | - | 0.9569 | 0.9314 |
| RCC | sealed test | 0.9948 +/- 0.0012 | 0.9646 | 0.9719 | 0.9606 |

BRCA-PAM50 screening did not qualify for sealed evaluation. The initial
4,096-patch run reached validation macro-AUC 0.8551, while a stronger
regularization run with 1,024 patches reached only 0.8431. Historical AB-MIL
validation runs on the same split reached 0.8681-0.8738.

The COAD result is an important negative control: its apparent validation gain
over historical OT-MIL did not transfer to the 59-patient sealed test set.
RCC is stable but saturated. Relative to the historical MO-MIL test result,
unified MIR improves macro-AUC by about 0.44 points, accuracy by 2.05 points,
balanced accuracy by 1.20 points, and macro-F1 by 2.28 points. It therefore
does not satisfy the desired three-point margin.

Representative commands:

```bash
python experiments/run_benchmark.py \
  --split /home/sigirika/experiment_splits/otmil_multiclass_v3/RCC_train_val.csv \
  --dataset-name MIR_UNIFIED_RCC --num-classes 3 \
  --log-root /home/sigirika/experiment_logs/mir_mil_v6/rcc_confirm \
  --models MIR_MIL --seeds 2025 2026 --epochs 45 --patience 12 \
  --earlystop-min-delta 0.001 --scheduler-t-max 43 --clamp-cosine \
  --max-instances 4096 --in-dim 1536 --device 0 --num-workers 2 \
  --model-option Model.optimizer.adamw_config.lr=0.0001 \
  --model-option Model.optimizer.adamw_config.weight_decay=0.0001 \
  --model-option Model.dropout=0.2

python experiments/evaluate_checkpoints.py \
  --run-root /home/sigirika/experiment_logs/mir_mil_v6/rcc_confirm \
  --output-dir /home/sigirika/experiment_logs/mir_mil_v6/rcc_frozen_test_seed2025_2026 \
  --models MIR_MIL --budgets 4096 --device 0 --num-workers 2 \
  --split-override /home/sigirika/datasets/tcga_rcc_uni2h/TCGA_RCC_UNI2H_CACHE4096_split.csv
```

## Active Local Route Follow-up

The four-route adaptive model initialized the local residual with
`sigmoid(-2) * 0.1`, an effective scale of about `0.012`. On the best BRCA
checkpoint, the learned effective class scales remained only `0.016-0.023`.
The local representation was therefore present in the architecture but nearly
disabled during classification.

A visible-validation ablation increased the shared route count to 12, the
gate bias to `-0.5`, and the local scale to `0.5`. The model equations and
closed-form response are unchanged. These settings are now the default for
unordered multiclass experiments; PANDA retains its separately recorded
ordinal protocol.

COAD robustness was assessed without reopening its sealed test set. The
original train and validation examples were pooled and repartitioned with
three-fold stratified group cross-validation. TCGA case IDs define groups, so
the two slides from the one duplicated patient never cross a fold boundary.

| COAD visible CV | Macro-AUC | BAcc | Macro-F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| 4 routes, weak local initialization | 0.8501 +/- 0.0206 | 0.6310 | 0.6358 | 0.6693 |
| Label smoothing 0.2 | 0.8537 +/- 0.0175 | 0.6161 | 0.6149 | 0.6529 |
| 12 active local routes | **0.8665 +/- 0.0273** | 0.6176 | 0.6252 | 0.6445 |

The active routes improve COAD ranking by 1.64 AUC points across folds, but
do not improve thresholded metrics. On the original visible COAD split, its
three-seed macro-AUC is `0.8349 +/- 0.0110`, compared with `0.8220 +/- 0.0058`
for the previous unified model. The sealed COAD test result remains unchanged
and was not used for this follow-up.

BRCA-PAM50 uses the same model structure and active-route settings. Increasing
the route count from 4 to 8 and 12 raised seed-2024 macro-AUC from `0.8607` to
`0.8694` and `0.8697`, respectively. Twelve routes were retained because
their balanced accuracy and macro-F1 were also higher.

| BRCA visible validation | Macro-AUC | BAcc | Macro-F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| MIR, 12 active routes | 0.8646 +/- 0.0047 | 0.6574 | 0.6683 | 0.7058 |
| MO-MIL | 0.8652 +/- 0.0017 | 0.6252 | 0.6237 | 0.6735 |
| AB-MIL | **0.8707 +/- 0.0029** | **0.6815** | **0.6834** | **0.7313** |

Thus the active-route change is a reproducible cross-dataset improvement over
the weakly initialized MIR model, and it improves BRCA BAcc/F1 over MO-MIL.
It does not establish a three-point AUC advantage: BRCA AUC is statistically
close to MO-MIL and remains about 0.61 points below AB-MIL. BRCA has no sealed
test evaluation in this phase.

Representative confirmation command:

```bash
python experiments/run_benchmark.py \
  --split /home/sigirika/datasets/brca_pam50_uni2h/BRCA_PAM50_UNI2H_CACHE4096_train_val.csv \
  --dataset-name MIR_BRCA_LOCAL12_CONFIRM --num-classes 4 \
  --log-root /home/sigirika/experiment_logs/mir_mil_v9/brca_local12_confirm \
  --models MIR_MIL --seeds 2025 2026 --epochs 45 --patience 10 \
  --earlystop-min-delta 0.001 --scheduler-t-max 43 --clamp-cosine \
  --max-instances 4096 --in-dim 1536 --device 0 --num-workers 2 \
  --model-option Model.num_local_routes=12 \
  --model-option Model.multiscale_gate_initial_bias=-0.5 \
  --model-option Model.multiscale_local_initial_scale=0.5 \
  --model-option Model.label_smoothing=0.0
```

## BRCA and COAD Optimization Follow-up

This phase continued to use visible validation data only. The COAD sealed test
was not reopened, and BRCA still has no sealed-test evaluation.

Several generic extensions were implemented with closed-form response tests:

- class-owned local route groups;
- learned shared/class-conditional mixtures and class residuals;
- optional exponential moving average validation;
- a full-rank anchor measure route;
- an anchor-primary potential with global and low-rank MIR residuals.

All variants preserve permutation invariance and the centered contamination
response. Finite-difference and path-completeness tests cover the new measure
states. The complete repository test suite contains 136 passing tests.

For COAD CMS, the corrected hybrid potential used 12 routes of dimension 32
and initialized the class-conditional mixture at 0.1. It improved both the
three-fold visible-pool result and the original visible split:

| COAD visible protocol | Macro-AUC | BAcc | Macro-F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| MO-MIL, original split | 0.8231 +/- 0.0073 | 0.5299 | 0.5182 | 0.6183 |
| OT-MIL, original split | 0.8340 +/- 0.0150 | **0.6365** | **0.6379** | 0.6613 |
| MIR shared routes | 0.8349 +/- 0.0110 | 0.6035 | 0.6126 | 0.6559 |
| MIR hybrid routes | **0.8368 +/- 0.0204** | 0.6243 | 0.6376 | **0.6720** |

The AUC gain over OT-MIL is 0.28 percentage points, so COAD now exceeds the
evaluated baseline mean but does not support a practically large superiority
claim. In patient-grouped three-fold cross-validation, hybrid MIR reached
`0.86685` AUC versus `0.86648` for shared-route MIR, while improving BAcc and
macro-F1 by about 2.3 and 2.5 points.

BRCA-PAM50 exposed a different bottleneck. Increasing each route state from
32 to 128 dimensions improved the previously weakest seed, but did not exceed
AB-MIL after three-seed confirmation:

| BRCA visible validation | Macro-AUC | BAcc | Macro-F1 | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| MO-MIL | 0.8652 +/- 0.0017 | 0.6252 | 0.6237 | 0.6735 |
| MIR, 12 x 32 routes | 0.8646 +/- 0.0047 | 0.6574 | **0.6683** | 0.7058 |
| MIR, 12 x 128 routes | **0.8669 +/- 0.0045** | **0.6644** | 0.6612 | **0.7058** |
| AB-MIL | 0.8707 +/- 0.0029 | 0.6815 | 0.6834 | 0.7313 |

Thus MIR exceeds MO-MIL by 3.2-4.5 points on BRCA thresholded metrics and by
0.18 AUC points with the wider route state, but remains 0.38 AUC points below
AB-MIL. Higher learning rate, weight decay, route temperature, route count,
EMA, class residuals, and full-rank anchor variants did not survive
multi-seed or worst-seed screening. A single-seed peak is not treated as an
improvement.

Representative COAD confirmation command:

```bash
python experiments/run_benchmark.py \
  --split /home/sigirika/experiment_splits/otmil_v2_tuning/COAD_CMS_train_val.csv \
  --dataset-name MIR_COAD_HYBRID01_CONFIRM --num-classes 4 \
  --log-root /home/sigirika/experiment_logs/mir_mil_v14/coad_hybrid01_confirm \
  --models MIR_MIL --seeds 2024 2025 2026 --epochs 35 --patience 10 \
  --max-instances 4096 --in-dim 1536 --device 0 --num-workers 2 \
  --model-option Model.potential_type=hybrid_multiscale \
  --model-option Model.multiscale_class_mix_initial=0.1
```
