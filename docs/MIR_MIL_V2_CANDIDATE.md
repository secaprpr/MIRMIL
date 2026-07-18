# MIR-MIL V2 Candidate

Status: implementation candidate; not a frozen release and not yet evaluated on
BRACS official test.

## Motivation

MIR-MIL-MT V1 has competitive BRACS3 macro-AUC but relatively weak macro-F1.
Existing diagnostics identify the ambiguous middle class as the main failure
mode. V2 therefore preserves the accepted measure-state and moment-token paths
and adds an explicitly contrastive class-boundary path.

## Architecture

For every unordered class pair `(a, b)`, V2 forms an antisymmetric query
direction from learned class embeddings:

```text
q_ab = normalize(q_b - q_a)
```

The direction and its negative retrieve evidence from opposite sides of the
same boundary. Each side is summarized by its attention-weighted mean and
variance. A shared low-rank projector produces pair evidence, while differences
of learned class factors produce a signed pair margin `m_ab`.

All pair margins are mapped to zero-sum class residual logits with the incidence
matrix of the complete class graph:

```text
z_pair = (1 / (C - 1)) * sum_(a,b) m_ab * (e_b - e_a)
```

The final prediction is:

```text
z = z_measure + 0.1 * z_moment + 0.1 * z_pair
```

Training additionally applies binary supervision to the `C - 1` pairs that
contain the ground-truth class. This loss is generic for ordered and unordered
tasks and does not encode BRACS class names or adjacency assumptions.

## Default candidate settings

```yaml
Model:
  moment_token_weight: 0.1
  pairwise_boundary_weight: 0.1
  pairwise_boundary_loss_weight: 0.1
  pairwise_boundary_query_dim: 64
  pairwise_boundary_value_dim: 128
  pairwise_boundary_rank_dim: 64
  pairwise_boundary_temperature: 1.0
  pairwise_boundary_dropout: 0.0
```

The benchmark registry applies these V2 overrides to the frozen MT V1 base
snapshot. V1 itself is unchanged.

## Validation protocol

V2 must first be evaluated on the official BRACS train/validation split using
seeds 2024, 2025, and 2026. It should pass all of the following gates before
the official BRACS test is opened:

1. Mean validation macro-AUC is not below MT V1.
2. Validation macro-F1 and balanced accuracy improve without a large increase
   in seed variance.
3. PANDA UNI seed-2024 macro-AUC does not regress materially from MT V1.
4. Per-class BRACS validation results show improvement in the ambiguous class,
   rather than only a class-prior shift.

Example BRACS validation command:

```bash
python experiments/run_benchmark.py \
  --split <BRACS3_train_val.csv> \
  --dataset-name BRACS3 \
  --num-classes 3 \
  --log-root artifacts/bracs3_v2_validation \
  --models MIR_MIL_V2 \
  --seeds 2024 2025 2026 \
  --epochs 30 \
  --patience 8 \
  --best-model-metric macro_auc \
  --earlystop-metric macro_auc \
  --scheduler-t-max 28 \
  --max-instances 4096 \
  --in-dim 1024 \
  --feature uni \
  --protocol bracs3-v2-validation-only \
  --split-id official-train-val-3class \
  --device 0 \
  --num-workers 2 \
  --balanced \
  --wandb-mode disabled
```

## Interpretation boundary

The original closed-form MIR response still explains only the explicit
measure-potential path. Moment-token and pairwise-boundary residual logits are
part of the final classifier but are not included in that closed-form
decomposition. V2 must not claim complete final-logit attribution without a new
derivation or a full-model numerical attribution method.
