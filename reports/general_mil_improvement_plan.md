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
