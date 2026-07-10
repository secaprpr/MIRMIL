# BRACS Next Experiments

Date: 2026-07-10

Constraint: model architecture and pre-extracted R50/UNI features remain unchanged. Therefore the next experiments must not change hidden dimension, sketch dimension, potential type, routes, attention/aggregation, heads, or branches. The historical capacity results are useful for diagnosis but are not valid next-step changes under the current frozen-architecture request.

All new experiments must write to `reports/bracs_deep_optimization_log.tsv` before continuing to the next experiment.

## Priority 1: Reproduce frozen-architecture default MIR-MIL

Experiment name: `BRACS_REPRO_DEFAULT_FROZEN`

Feature type: both R50 and UNI.

Hypothesis: The default MIR-MIL BRACS result is reproducible under the current code and the official split; any later optimization must be compared against this frozen-architecture baseline.

Why not fully answered: Baseline results exist, but the worktree now contains later default-off utilities. Before new work, we need a current-code reproduction that confirms defaults still match old architecture behavior.

Exact config changes:

- Create new config files copied from `configs/MIR_MIL.yaml` with all new optional settings disabled:
  - `class_weighting: none`
  - `focal_gamma: 0.0`
  - `swa_start_epoch: 0`
  - no feature normalization
  - no fusion
  - `hidden_dim=256`
  - `sketch_dim=128`
  - `potential_hidden_dim=128`
  - `potential_type=adaptive_multiscale`

Exact command:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python experiments/run_benchmark.py \
  --split /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS_uni_split_official_train_val.csv \
  --dataset-name BRACS --num-classes 7 \
  --log-root artifacts/bracs_deep_opt/repro_default_uni \
  --models MIR_MIL --seeds 2024 \
  --epochs 30 --patience 8 --max-instances 4096 --in-dim 1024 \
  --feature uni --protocol bracs-deep-opt-repro-default \
  --split-id official-train-val --comparison-id bracs-repro-default-uni-seed2024 \
  --device 0 --num-workers 2 --balanced --wandb --wandb-project MIR-MIL
```

Then repeat for R50 by changing split, log root, feature, and comparison id.

Expected improvement: none. This is a reproducibility check.

Risk: consumes compute without improving performance.

Success: validation curve and selected epoch are in the same range as previous default MIR-MIL.

Failure: indicates code/config drift; stop optimization and debug.

Keep or discard: keep as the frozen baseline for this phase.

## Priority 2: Split, label, metric, and loader audit

Experiment name: `BRACS_SPLIT_METRIC_AUDIT`

Feature type: both.

Hypothesis: BRACS underperformance is not caused by split/label/metric/data-loader bugs.

Why not fully answered: Previous audits checked prediction rows and probabilities, but the current phase needs an explicit frozen-architecture audit before further optimization.

Exact config changes: none.

Exact command:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python experiments/audit_bracs_protocol.py \
  --r50-split /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS_r50_split_official_full.csv \
  --uni-split /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS_uni_split_official_full.csv \
  --baseline-seed-results artifacts/bracs_evaluation/seed_results.csv \
  --output reports/bracs_protocol_audit.json
```

If `experiments/audit_bracs_protocol.py` does not exist, create it as a read-only audit script.

Expected improvement: none.

Risk: none except time.

Success: exact same slide ids/labels across R50/UNI, expected counts, no NaNs, same metric computation as aggregate.

Failure: fix data/metric bug before training.

Keep or discard: keep if audit passes; fix if it fails.

## Priority 3: Validation-selection reliability check

Experiment name: `BRACS_VAL_SELECTION_AUDIT`

Feature type: both, starting with UNI.

Hypothesis: The main BRACS failure mode is unreliable validation selection, not insufficient raw model capacity.

Why not fully answered: We have evidence that high validation settings fail on test, but no systematic curve audit of best epoch, last epoch, and smoothed validation behavior for frozen-architecture default MIR-MIL.

Exact config changes: none.

Exact command:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python experiments/audit_validation_selection.py \
  --logs artifacts/bracs_baselines/uni/MIR_MIL \
  --output reports/bracs_uni_default_validation_selection.json
```

Expected improvement: none directly. It should choose whether future experiments use:

- default best validation epoch;
- smoothed validation macro AUC;
- later checkpoint selection;
- early stopping with larger patience.

Risk: if this uses test metrics to choose a rule, it becomes test tuning. Do not use test in rule selection.

Success: identifies a validation-only selection rule that is more stable across seeds.

Failure: confirms validation set is too small to support more tuning.

Keep or discard: keep if rule is validation-only and reproducible.

## Priority 4: R50 vs UNI under identical frozen training settings

Experiment name: `BRACS_R50_UNI_IDENTICAL_DEFAULT`

Feature type: both.

Hypothesis: UNI and R50 need different training regularization or sampling even with the same architecture.

Why not fully answered: Baseline matrix uses same high-level protocol, but later HPO focused more on UNI and capacity variants. We need a frozen-architecture paired run with identical settings and the same current code.

Exact config changes:

- frozen default architecture;
- no new loss;
- no normalization;
- same seed;
- same sampler;
- same budget.

Exact command: same as Priority 1, run R50 and UNI with seed 2024. If reproduction passes, extend to seeds 2025 and 2026.

Expected improvement: none directly.

Risk: compute cost.

Success: determines whether UNI instability is larger than R50 under the same training settings.

Failure: if results drift from baseline, debug reproducibility first.

Keep or discard: keep as current-code paired reference.

## Priority 5: Training sampling strategy

Experiment name: `BRACS_UNI_SAMPLING_STRATEGY`

Feature type: UNI first, then R50 only if UNI improves.

Hypothesis: BRACS huge bags make random 4,096-patch training too noisy; deterministic or less variable sampling improves validation stability.

Why not fully answered: Previous experiments mostly kept `Model.sampling=random` during training. BRACS patch-count audit shows the 4,096 budget truncates most slides, unlike PANDA.

Exact config changes:

- architecture unchanged;
- compare only:
  - `Model.sampling=random` baseline;
  - `Model.sampling=uniform`;
  - optionally `Model.sampling=head` only as a negative control.

Exact command:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python experiments/run_benchmark.py \
  --split /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS_uni_split_official_train_val.csv \
  --dataset-name BRACS --num-classes 7 \
  --log-root artifacts/bracs_deep_opt/uni_sampling_uniform_seed2024 \
  --models MIR_MIL --seeds 2024 \
  --epochs 30 --patience 8 --max-instances 4096 --in-dim 1024 \
  --feature uni --protocol bracs-deep-opt-sampling \
  --split-id official-train-val --comparison-id bracs-uni-sampling-uniform-seed2024 \
  --device 0 --num-workers 2 --balanced --wandb --wandb-project MIR-MIL \
  --model-option Model.sampling=uniform
```

Expected improvement: modest validation stability gain; possible macro AUC improvement if random sampling is too noisy.

Risk: uniform sampling may reduce stochastic augmentation and overfit to a fixed subset.

Success: validation macro AUC improves over default seed2024 and curve is smoother without changing architecture.

Failure: rejects sampling noise as the dominant fix at 4,096 budget.

Keep or discard: keep only if validation improves and multi-seed confirms.

## Priority 6: Inference-time patch-subset averaging

Experiment name: `BRACS_UNI_INFERENCE_AVG`

Feature type: UNI first; R50 if UNI improves.

Hypothesis: BRACS predictions are unstable because each evaluation sees only one 4,096-patch subset from a huge slide. Averaging multiple existing-feature subsets improves validation without changing architecture or features.

Why not fully answered: Previous larger-budget checks are not the same as averaging predictions from multiple 4,096 subsets. This tests inference variance directly.

Exact config changes:

- no model change;
- no feature change;
- same checkpoint;
- validation-only selection of number of subsets, e.g. `K=4` and `K=8`.

Exact command:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python experiments/evaluate_patch_subset_averaging.py \
  --config <selected_default_MIR_config.yaml> \
  --checkpoint <validation_selected_checkpoint.pth> \
  --split /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS_uni_split_official_full.csv \
  --group val \
  --feature uni --model MIR_MIL --seed 2024 \
  --max-instances 4096 --num-subsets 4 \
  --output reports/bracs_uni_subsetavg_val_seed2024_k4.csv
```

Expected improvement: improved validation macro AUC if patch subset variance is a major failure mode.

Risk: increased inference cost; if K is selected using test, invalid.

Success: validation improves consistently across seeds using fixed K selected on validation.

Failure: suggests subsampling variance alone is insufficient.

Keep or discard: keep only after validation-selected K and multi-seed confirmation.

## Priority 7: Class balancing narrowly revisited

Experiment name: `BRACS_CLASS_BALANCE_MINIMAL`

Feature type: UNI first.

Hypothesis: BRACS imbalance hurts macro AUC/F1, but previous class weighting was interrupted and not cleanly completed.

Why not fully answered: Weighted/focal CE screen was interrupted and used pt mirror; it is negative trend evidence but not a complete H5 protocol result.

Exact config changes:

- compare sampler on/off with default CE;
- optionally use inverse-sqrt class weighting if sampler-only fails.

Exact command:

```bash
PYTHONPATH=$PWD mamba run -n mirmil python experiments/run_benchmark.py \
  --split /data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/metadata/BRACS_uni_split_official_train_val.csv \
  --dataset-name BRACS --num-classes 7 \
  --log-root artifacts/bracs_deep_opt/uni_unbalanced_default_seed2024 \
  --models MIR_MIL --seeds 2024 \
  --epochs 30 --patience 8 --max-instances 4096 --in-dim 1024 \
  --feature uni --protocol bracs-deep-opt-class-balance \
  --split-id official-train-val --comparison-id bracs-uni-unbalanced-default-seed2024 \
  --device 0 --num-workers 2 --no-balanced --wandb --wandb-project MIR-MIL
```

Expected improvement: possible macro AUC/BACC improvement if balanced sampler over-distorts BRACS train distribution.

Risk: minority classes may worsen.

Success: validation macro AUC and macro F1 improve together.

Failure: class balancing is not the first-order problem.

Keep or discard: keep only with validation and multi-seed support.

## Priority 8: Existing-feature normalization/filtering

Experiment name: `BRACS_FEATURE_NORM_FILTER`

Feature type: UNI first.

Hypothesis: existing feature bags include low-information or outlier patch embeddings; per-slide normalization or low-norm filtering improves robustness.

Why not fully answered: Coarse norm stats do not reveal gross scale mismatch, but bag-level noise remains plausible.

Exact config changes:

- no architecture change;
- apply data-loader transform only:
  - per-slide z-score using existing features;
  - or per-patch L2 normalization;
  - or low-norm patch filtering with validation-selected threshold.

Exact command: not run until an explicit transform script/config exists and is reviewed.

Expected improvement: uncertain.

Risk: this can silently change the feature protocol. It must be documented as preprocessing using fixed existing features, not new features.

Success: validation improves without reducing per-class recall.

Failure: discard; do not stack with other changes.

## Initial execution recommendation

Run Priority 1 first: current-code frozen default reproduction for UNI seed2024. It is not intended to improve SOTA, but it prevents building new claims on top of unverified code drift.
