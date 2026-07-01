# Weights & Biases Experiment Tracking Design

Status: proposed schema, July 1, 2026.

Implementation status: the schema is implemented for MIR-MIL training,
checkpoint evaluation and aggregation, faithfulness evaluation, and paired
bootstrap.

## Goals

The tracking system must support:

- compact comparison tables;
- exact experiment reproduction;
- multi-seed aggregation;
- checkpoint and prediction provenance;
- sealed-test discipline;
- faithfulness and paired-bootstrap audits;
- both R50 and UNI feature experiments.

Markdown remains suitable for conclusions and protocol decisions. W&B should
replace handwritten per-run metric tables, not the research narrative.

## Project

Use one stable project:

```text
MIR-MIL
```

Do not create a separate project for each dataset or encoder. Those are
queryable configuration dimensions.

## Run Unit

One W&B run represents one atomic job:

- `train`: one model, split, feature encoder, variant, and seed;
- `eval`: one frozen checkpoint evaluated under one explicit protocol;
- `aggregate`: one aggregation over a declared set of child runs;
- `faithfulness`: one explanation audit over one frozen checkpoint;
- `bootstrap`: one paired statistical comparison over frozen predictions.

Training and sealed-test evaluation should be separate runs. This prevents a
training run from silently acquiring test results before its checkpoint and
evaluation protocol are frozen.

## Naming

Recommended human-readable run name:

```text
{dataset}_{feature}_{model}_{variant}_seed{seed}_{job_type}
```

For jobs without a single seed, replace the seed component with the aggregate
identifier:

```text
{dataset}_{feature}_{comparison}_{protocol}_{job_type}
```

W&B run names are not guaranteed to be unique. The immutable W&B run ID is
the authoritative identifier. Store a deterministic `experiment_key` in the
config for duplicate detection.

## Grouping

For seed aggregation:

```text
{dataset}_{feature}_{model}_{variant}_{protocol}_{split_id}
```

The group deliberately excludes the seed and job type. All seeds for the same
experimental condition belong to one group.

Cross-model comparison is a separate concept. Store a shared
`comparison_id` in the config for models participating in the same benchmark.
Do not overload `group` to mean both a seed group and a benchmark cohort.

## Job Type

Allowed values:

```text
train
eval
aggregate
faithfulness
bootstrap
```

Use these controlled values rather than free-form variants.

## Tags

Tags are for low-cardinality, human-facing filters. Recommended examples:

```text
dataset:panda
feature:r50
feature:uni
model:mir_mil
stage:pilot
stage:formal
test:sealed
test:visible
status:valid
status:invalid
```

Do not encode exact seeds, hashes, learning rates, paths, or full model
settings as tags. Those values belong in the config. Excessive tags are hard
to validate and produce an unbounded vocabulary.

## Config

Use a nested, immutable-at-start configuration. Suggested structure:

```yaml
schema_version: 1
experiment_key: panda_uni_mir_base_protocol-v1_split-v1_seed2024_train
comparison_id: panda_encoder_comparison_v1

dataset:
  name: PANDA
  task: isup_grade
  num_classes: 6
  source_manifest_sha256: ...

split:
  id: panda_patient_split_v1
  path: ...
  sha256: ...
  unit: patient
  train_n: ...
  val_n: ...
  test_n: ...
  test_sealed: true

features:
  encoder: uni
  encoder_version: uni_v1_vit_l16
  checkpoint_sha256: ...
  manifest_sha256: ...
  feature_dim: 1024
  patch_level: 0
  patch_size: 256
  step_size: 256
  target_patch_size: 224
  coordinate_manifest_sha256: ...

model:
  name: MIR_MIL
  variant: base
  settings: {}

training:
  seed: 2024
  epochs_cap: ...
  patience: ...
  optimizer: {}
  scheduler: {}
  sampling: random
  max_instances: ...
  criterion: cross_entropy

evaluation:
  protocol: protocol_v1
  checkpoint_metric: val_macro_auc
  sampling: uniform
  patch_budget: ...
  deterministic: true
```

Avoid uploading machine-specific absolute paths as the only identifiers.
Store stable logical IDs and hashes; paths may be included as optional local
provenance.

## Metric History

Time-series metrics belong in run history:

```text
train/loss
train/accuracy
val/loss
val/accuracy
val/balanced_accuracy
val/macro_f1
val/macro_auc_ovr
system/epoch
```

Declare `epoch` as the step metric. Log exactly once per completed epoch.

Metric names must state averaging and multiclass semantics. Avoid ambiguous
names such as `auc` and `f1` when the actual definitions are macro one-vs-rest
AUC and macro-F1.

## Summary

Only final or selected values should populate summary columns:

```text
val/best_macro_auc_ovr
val/best_accuracy
val/best_balanced_accuracy
val/best_macro_f1
train/best_epoch
train/stop_epoch

test/macro_auc_ovr
test/accuracy
test/balanced_accuracy
test/macro_f1
```

Faithfulness summaries should be explicit:

```text
faithfulness/pearson
faithfulness/spearman
faithfulness/fd_mae
faithfulness/topk_overlap
faithfulness/centered_mean_abs
faithfulness/completeness_error
```

Bootstrap summaries must identify the comparison and metric:

```text
bootstrap/baseline_run_id
bootstrap/metric
bootstrap/delta
bootstrap/ci95_low
bootstrap/ci95_high
bootstrap/iterations
bootstrap/seed
```

Do not put several unrelated confidence intervals into one generic `CI`
field.

The visible W&B workspace can be configured to show only the key summary
columns. This is a UI choice; it is not a reason to move reproducibility
fields from config into tags.

## Artifacts and Lineage

Use W&B artifacts, or an equivalent local artifact manifest, for:

- split CSV;
- coordinate manifest;
- feature manifest;
- best checkpoint;
- per-slide predictions and probabilities;
- faithfulness patch responses;
- bootstrap inputs and outputs.

An evaluation run consumes a checkpoint artifact and a split artifact. An
aggregate run records the exact child run IDs. A bootstrap run consumes two
prediction artifacts evaluated on the same ordered sample IDs.

Large WSI files and patch-feature caches should not be uploaded by default.
Upload compact manifests containing paths, sizes, hashes, dimensions, and
generation parameters. Whether checkpoints and predictions may leave the
server must be decided explicitly.

## Sealed-Test Rules

For a sealed test:

1. training runs log only train and validation metrics;
2. the checkpoint rule and evaluation protocol are frozen;
3. an `eval` run consumes the frozen checkpoint;
4. the eval config records `test_sealed: true`;
5. predictions are stored as an immutable artifact;
6. aggregate and bootstrap jobs reference the eval run IDs;
7. invalid or accidental test openings are retained and tagged
   `status:invalid`, not deleted from the research record.

## Validation

The integration should reject or flag:

- missing split or feature hashes;
- duplicate `experiment_key` unless explicitly resumed;
- non-finite metrics;
- an eval run without a checkpoint parent;
- sample-order mismatch in paired bootstrap;
- different coordinate manifests in an encoder-only comparison;
- test metrics logged by a `train` job;
- ambiguous metric definitions;
- aggregate runs with missing child run IDs.

## Enabling Tracking

MIR-MIL keeps tracking disabled by default so local tests and legacy commands
remain offline. Enable it through YAML overrides:

```bash
python train_mil.py --yaml_path configs/MIR_MIL.yaml --options \
  Tracking.wandb.enabled=true \
  Tracking.wandb.project=MIR-MIL \
  Tracking.wandb.mode=online \
  Tracking.wandb.feature=uni \
  Tracking.wandb.variant=base \
  Tracking.wandb.protocol=panda_protocol_v1 \
  Tracking.wandb.split_id=panda_patient_split_v1 \
  Tracking.wandb.comparison_id=panda_encoder_comparison_v1 \
  Dataset.dataset_csv_path=/path/to/split.csv \
  Logs.log_root_dir=/path/to/logs
```

The following provenance fields should also be supplied for a formal run:

```text
Tracking.wandb.source_manifest_sha256
Tracking.wandb.feature_manifest_sha256
Tracking.wandb.coordinate_manifest_sha256
Tracking.wandb.encoder_checkpoint_sha256
```

Independent jobs use command-line flags:

```text
--wandb
--wandb-project MIR-MIL
--wandb-mode online
--wandb-tag stage:formal
--wandb-group OPTIONAL_GROUP_OVERRIDE
--wandb-comparison-id OPTIONAL_COMPARISON_ID
```

These flags are supported by:

```text
experiments/evaluate_checkpoints.py
experiments/evaluate_mir_faithfulness.py
experiments/paired_bootstrap.py
```

The implementation is centralized in `utils/wandb_utils.py`. If tracking is
enabled but the `wandb` package is unavailable, execution fails explicitly
instead of silently dropping the experiment record.

Checkpoint upload is disabled by default with
`Tracking.wandb.upload_checkpoints=false`. The local checkpoint path and
SHA-256 are recorded instead. The tracker also skips any individual artifact
file larger than `Tracking.wandb.max_artifact_mb` (50 MiB by default). WSI
files, feature caches, and pretrained encoder weights must never be passed to
the artifact logger.
