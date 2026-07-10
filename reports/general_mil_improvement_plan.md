# General MIL improvement plan

## Goal

Improve MIR-MIL as a general WSI-MIL method, not as a BRACS-specific trick. The method must support arbitrary class counts and datasets, keep frozen pre-extracted features, and remain evaluable on both PANDA and BRACS.

## Rejected path: more naive HPO

The first HPO round showed validation gains but official-test drops. This indicates validation over-selection. More of the same HPO is not scientifically justified.

## Candidate general improvements

### 1. Class-aware evidence aggregation

Add a generic class-query evidence head over encoded patch tokens. Each class owns a learnable query. The query attends over patch embeddings and produces a class-specific evidence vector/logit.

Why it is general:

- works for any `num_classes`;
- no dataset-specific class names or counts;
- applicable to PANDA, BRACS, and other WSI-MIL tasks;
- complements existing measure-state potential.

Why it addresses the failure:

- preserves sparse class-specific evidence;
- gives each class a direct patch readout;
- reduces reliance on a single compressed state for all classes.

Why it is easy to ablate:

- controlled by `Model.evidence_weight`;
- `evidence_weight=0.0` recovers the current archived architecture;
- can compare `0.0` vs small positive values under identical training.

Risk:

- adds parameters and may overfit small datasets;
- can harm PANDA if it over-emphasizes local evidence at the expense of global distribution;
- should start with small residual scale.

### 2. Multi-token attention readout

Add multiple learnable evidence tokens that attend to patches, then pool token outputs for classification.

Benefit:

- captures multiple tissue patterns per slide.

Risk:

- less directly interpretable per class than class-aware queries;
- extra pooling design decisions.

### 3. Prototype/query-based evidence aggregation

Use learnable prototypes/queries to retrieve patch evidence and compare class evidence.

Benefit:

- explicit evidence prototypes.

Risk:

- higher complexity;
- harder to isolate whether improvements come from prototypes, query attention, or regularization.

### 4. Stability-focused prediction head

Use EMA/SWA/checkpoint averaging or confidence-stabilized heads.

Benefit:

- may reduce validation/test gap.

Risk:

- does not address missing class-specific evidence;
- previous training-only search suggests this is unlikely to be sufficient.

## Selected first improvement

Implement class-aware evidence aggregation as a residual branch inside MIR-MIL.

Design:

1. Use the existing encoded patch features from `self.encoder`.
2. Add learnable class queries with shape `[num_classes, evidence_query_dim]`.
3. Project encoded patch tokens to key/value vectors.
4. For each class, attend to patches and compute a class-specific evidence vector.
5. Convert each class evidence vector to one logit.
6. Add the evidence logits to the existing measure-potential logits with a scalar residual weight.

Default behavior:

- `evidence_weight = 0.0`
- archived behavior is preserved unless explicitly enabled.

Initial experiment:

- BRACS3 UNI, three seeds, 30 epochs, no early stopping, best validation macro-AUC checkpoint.
- PANDA UNI, at least a small matched run or reuse full matrix when compute allows.
- No SOTA claim until both datasets are evaluated with multi-seed official protocol.

Success criteria:

- BRACS3 improves over `UNI + MIR_MIL noES/best-val = 0.8403 ± 0.0184`;
- PANDA does not materially drop from `UNI + MIR_MIL = 0.9504 ± 0.0011`;
- improvement is stable across seeds and not selected by BRACS test.

## Implementation status

Implemented as `ClassAwareEvidenceHead` in `modules/MIR_MIL/mir_mil.py`.

Config switches:

- `Model.evidence_weight`: residual scale; default `0.0`, disabled.
- `Model.evidence_query_dim`: class-query/key dimension; default `64`.
- `Model.evidence_value_dim`: value/evidence dimension; default `128`.
- `Model.evidence_temperature`: attention temperature; default `1.0`.
- `Model.evidence_dropout`: evidence value dropout; default `0.0`.

The branch uses only the existing MIR-MIL encoded patch representations and frozen pre-extracted features. It is generic in `num_classes` and contains no BRACS/PANDA-specific constants. With `evidence_weight=0.0`, the evidence head is not instantiated.

Smoke checks completed on 2026-07-11:

- Python compile check for `modules/MIR_MIL/mir_mil.py` and `utils/model_utils.py`.
- Forward smoke test for 2, 3, and 6 classes.
- One-epoch BRACS3 UNI training smoke on the official train/val split with `Model.evidence_weight=0.1`.

The smoke result was `val_macro_auc=0.779278` at epoch 1 and is only a pipeline check, not a performance claim.

## First ablation outcome

`evidence_weight=0.05` was evaluated as the first controlled ablation.

BRACS3 UNI validation:

- seed2024: `0.902793` macro-AUC at epoch 20
- seed2025: `0.902999` macro-AUC at epoch 15
- seed2026: `0.898698` macro-AUC at epoch 3
- mean ± std: `0.901497 ± 0.002426`

PANDA UNI validation sanity:

- seed2024 evidence_w005: `0.951225` macro-AUC at epoch 29
- seed2024 archived/default MIR-MIL: `0.951178` macro-AUC

Thus the module passed the PANDA sanity gate and did not behave like a BRACS-only trick on validation.

However, the single frozen BRACS3 official-test evaluation failed.

Protocol correction: the archived BRACS3 official-test matrix used `--max-instances 4096`. An initial evidence evaluation used the default diagnostic budgets `128/256/512`; those values are budget-sensitivity evidence, not the primary fair comparison.

Fair 4096-budget official test:

- budget4096: `0.808322 ± 0.024521` macro-AUC; `0.609195` accuracy; `0.573822` balanced accuracy; `0.556995` macro-F1

Supplemental low-budget diagnostic official test:

- budget128: `0.796290 ± 0.016543` macro-AUC
- budget256: `0.789243 ± 0.015511` macro-AUC
- budget512: `0.809526 ± 0.014712` macro-AUC

The fair 4096 result is worse than the archived `UNI + MIR_MIL` official-test result (`0.827973 ± 0.027678`) and much worse than `UNI + AC_MIL` (`0.852852 ± 0.009653`). Therefore `evidence_weight=0.05` must be discarded as an accepted improvement.

Scientific interpretation:

- The idea is generic and does not harm PANDA validation.
- BRACS validation improves strongly and consistently.
- The improvement does not transfer to BRACS official test.
- The current bottleneck is therefore not simply missing class-aware evidence capacity. BRACS validation selection remains unreliable, and the evidence branch may amplify split-specific evidence or calibration artifacts.

Next generic architecture work should not tune `evidence_weight` against BRACS test. The next step should first analyze validation/test divergence, including class confusion, calibration, bag-size dependence, and whether evidence logits become over-confident on BRACS test.

## Second candidate: subset prediction consistency

Motivation:

- BRACS validation/test divergence suggests that the model is sensitive to which patches are observed under the 4096-instance budget.
- BRACS class 1 appears especially sensitive to sparse or unstable evidence.
- PANDA has stronger coarse labels and more stable global evidence, so a good generic improvement should not damage PANDA validation.

The implemented candidate is a dataset-agnostic patch-subset consistency objective. For each training WSI, the model computes the usual full sampled-bag logits. It then samples one or more random patch subsets from the same bag and penalizes KL divergence between the full-bag predictive distribution and each subset predictive distribution. An optional supervised CE term on the subset views can be added.

This is not BRACS-specific:

- no dataset name, split, class label, class count, or BRACS category is hard-coded;
- it uses only existing pre-extracted features;
- it does not change the feature extractor;
- it does not use BRACS test feedback;
- it works for any `num_classes >= 2`.

Implementation knobs, all disabled by default:

- `Model.subset_consistency_weight`: KL consistency weight; default `0.0`.
- `Model.subset_consistency_supervised_weight`: supervised CE weight on subset views; default `0.0`.
- `Model.subset_consistency_fraction`: fraction of the sampled bag used per subset view; default `0.75`.
- `Model.subset_consistency_views`: number of subset views; default `1`.
- `Model.subset_consistency_temperature`: distillation temperature; default `1.0`.

First smoke test:

- command: `mamba run -n mirmil python experiments/run_benchmark.py ... --max-instances 512 --epochs 1 --model-option Model.subset_consistency_weight=0.2 --model-option Model.subset_consistency_supervised_weight=0.2 --model-option Model.subset_consistency_fraction=0.5 --model-option Model.subset_consistency_views=2`
- result: completed one epoch on BRACS3 UNI official train/val split.
- logged components: `subset_consistency_loss=0.036721`, `subset_supervised_loss=0.934004`.

Interpretation: the implementation path trains and logs correctly. The smoke result is not a performance claim. The next valid step is a BRACS3 validation-only ablation, followed by PANDA validation sanity only if BRACS validation improves.

Validation-only ablation:

- command root: `artifacts/bracs3_arch_ablation/uni/subset_consistency_w02`
- seeds: `2024, 2025, 2026`
- split: official BRACS3 train/val only
- budget: `4096`
- official BRACS test: not opened

Results:

| seed | best epoch | val macro-AUC | val acc | val bacc | val macro-F1 |
|---:|---:|---:|---:|---:|---:|
| 2024 | 3 | `0.875570` | `0.646154` | `0.665079` | `0.647560` |
| 2025 | 3 | `0.889244` | `0.723077` | `0.688889` | `0.685290` |
| 2026 | 2 | `0.891728` | `0.753846` | `0.676190` | `0.675432` |
| mean ± std | - | `0.885514 ± 0.008701` | `0.707692 ± 0.055470` | `0.676720 ± 0.011914` | `0.669428 ± 0.019569` |

Interpretation:

- The hypothesis that simple full-bag to subset KL consistency is sufficient is not supported at this setting.
- The validation result is below the rejected evidence branch validation result (`0.901497 ± 0.002426`), and it does not provide a strong candidate signal.
- Because the validation gate failed, PANDA sanity and BRACS official-test evaluation are not justified for this candidate.
- The implementation remains useful as a disabled generic regularizer, but it should not be claimed as an improvement.

## Third candidate: multi-token attention readout

Motivation:

- Evidence_w005 showed that class-owned queries can improve BRACS validation but fail official-test transfer, likely by over-amplifying split-specific class evidence.
- Subset consistency showed that simple training regularization is not enough.
- A more plausible generic direction is to improve readout capacity while avoiding hard class-specific evidence queries.

The implemented candidate adds a dataset-agnostic multi-token attention residual head over the existing encoded patch features. It uses `K` learnable readout tokens shared across classes. Each token attends to patches, collects one evidence vector, concatenates all token evidence vectors, and maps them through a shared prediction layer to logits. The head is residual and disabled by default.

Why this is generic:

- no class label, class semantics, dataset name, or split information is hard-coded;
- the same module works for any `num_classes >= 2`;
- it uses only existing frozen R50/UNI features and MIR-MIL encoded patch representations;
- it does not change the feature extractor or use raw WSI images;
- it should help PANDA by preserving multiple high-level evidence modes and help BRACS by giving the readout more than one patch-evidence slot without BRACS-specific class queries.

Implementation knobs, all disabled by default:

- `Model.multi_token_weight`: residual logit weight; default `0.0`.
- `Model.multi_token_count`: number of shared readout tokens; default `4`.
- `Model.multi_token_dim`: token/key dimension; default `64`.
- `Model.multi_token_readout_dim`: value dimension per token; default `128`.
- `Model.multi_token_temperature`: attention temperature; default `1.0`.
- `Model.multi_token_dropout`: classifier dropout; default `0.0`.

Smoke tests:

- synthetic forward/backward passed for `2`, `3`, and `6` classes.
- one-epoch BRACS3 UNI training smoke with `Model.multi_token_weight=0.1`, `max_instances=512` completed.
- smoke validation macro-AUC was `0.791066`; this is not a performance claim.

Validation gate:

- first controlled ablation will use `multi_token_weight=0.1`, `multi_token_count=4`, `multi_token_dim=64`, `multi_token_readout_dim=128`.
- selection remains validation-only; BRACS official test is not opened unless the candidate passes BRACS validation and PANDA sanity.
