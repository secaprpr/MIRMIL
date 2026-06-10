# OT-MIL Reproducibility Report

Date: 2026-06-10

## Scope

This report records the transition from the OT-MIL prototype to a tested,
reproducible implementation and its comparison with the repository's MO-MIL
baseline on PANDA, Camelyon16, and TCGA-NSCLC.

The shared protocol uses:

- seeds 2024, 2025, and 2026;
- validation macro-AUC for checkpoint selection;
- one final test evaluation per selected checkpoint;
- the same split, patch budget, deterministic validation/test sampling, and
  balanced training sampler for both methods;
- 30 epochs and early stopping patience 8;
- 512 sampled patches for the original PANDA/Camelyon16 study and 4,096
  patches for the PANDA capacity follow-up and TCGA-NSCLC study.

MO-MIL uses its method-specific optimizer and scheduler from the repository
configuration. Its sequence layer is the repository's dependency-free PyTorch
fallback, not the optional Mamba2 implementation.

## Data Provenance

### PANDA

- Original features: `/mnt/e/datasets/CPathPatchFeature/panda/r50/pt_files`
- Local ext4 cache: `/home/sigirika/datasets/panda_r50_pt`
- Slides: 9,999
- Split: 5,763 train / 1,921 validation / 1,920 test
- Assignment SHA-256:
  `e29b7c4...d8e0c3`
- Cached split SHA-256:
  `49a305255880bc9b93a0a1be82214d657aef2e99b7a7b5c083dda7ec68dcdf6a`

The cached and original normalized assignments were compared exactly.

### Camelyon16

- Original features:
  `/srv/storage1/hdd/wzx/Camelyon16/h5_files`
- Fixed 4,096-candidate cache:
  `/srv/storage1/hdd/cff/wqy/otmil/camelyon16_cache_4096`
- Slides: 270
- Split: 162 train / 54 validation / 54 test
- Source split SHA-256:
  `9cc5afac20e610991e44dc7892de75d02ae2053d2bff326a93e8d008e343ffeb`
- Cached split SHA-256:
  `2d4379939d3c58a45d3ac33b9015d539677ccc0b4aba87d32ee7d92af94f9ddd`
- Cached candidates: 1,079,387 instances, 4,421,403,179 bytes

The cache uses deterministic uniform candidate selection. Both methods sample
the same 512-patch budget from the same candidate cache.

### TCGA-NSCLC

- LUAD PLIP-QC features: 507 slides / 444 patients
- LUSC PLIP-QC features: 484 slides / 452 patients
- Feature dimension: 512
- Patient-level split: 572 train / 218 validation / 201 test slides
- Patients: 896
- Assignment SHA-256:
  `1c065c12dada02c957e97557895c3998d3fc33be3c0d5f32bd607607a2640b63`

All slides from one TCGA patient are constrained to one split. The reproducible
preparation script creates labels, patient IDs, and a unified symlink feature
directory without copying the source features.

## Main Results

Mean and sample standard deviation over three seeds:

| Dataset | Method | Macro-AUC | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| PANDA | MO-MIL | 0.8996 +/- 0.0038 | 0.6363 | 0.5868 | 0.5887 |
| PANDA | OT-MIL | **0.9051 +/- 0.0037** | **0.6462** | **0.6043** | **0.6032** |
| Camelyon16 | MO-MIL | **0.8116 +/- 0.0272** | **0.7099** | **0.6629** | **0.6318** |
| Camelyon16 | OT-MIL | 0.7173 +/- 0.0766 | 0.6235 | 0.6397 | 0.6201 |
| Camelyon16 | UOT-only | 0.7576 +/- 0.0342 | 0.6420 | 0.6506 | 0.6196 |
| TCGA-NSCLC PLIP-QC, 4,096 patches | MO-MIL | 0.9160 +/- 0.0110 | 0.8308 | 0.8310 | 0.8299 |
| TCGA-NSCLC PLIP-QC, 4,096 patches | OT-MIL | **0.9235 +/- 0.0025** | **0.8607** | **0.8605** | **0.8602** |

Paired stratified bootstrap, 5,000 iterations unless noted:

| Dataset/configuration | Mean AUC difference vs MO-MIL | 95% CI | P(OT > MO) |
|---|---:|---:|---:|
| PANDA OT-MIL, 512 patches | +0.00555 | [+0.00208, +0.00896] | 0.9988 |
| Camelyon16 OT-MIL, 512 patches | -0.09422 | [-0.16903, -0.01799] | 0.0072 |
| Camelyon16 UOT-only, 512 patches | -0.05398 | [-0.12358, +0.01278] | 0.0614 |
| TCGA-NSCLC OT-MIL, 4,096 patches (10,000) | +0.00752 | [-0.00515, +0.02076] | 0.8720 |

## PANDA Capacity Follow-up

A validation-only search with empty test columns selected `hidden_dim=512`,
`num_prototypes=16`. Its validation AUC exceeded the default configuration for
all three seeds. The frozen checkpoints were then evaluated once using the
full split through the evaluator's audited `--split-override`.

| Method, 4,096 patches | Macro-AUC | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|
| MO-MIL | 0.9005 +/- 0.0033 | 0.6384 | 0.5897 | 0.5914 |
| Default OT-MIL | 0.9063 +/- 0.0038 | 0.6493 | 0.6082 | 0.6076 |
| OT-MIL H512 | **0.9075 +/- 0.0025** | **0.6495** | **0.6109** | **0.6097** |

H512 exceeded MO-MIL in every seed. The paired AUC difference was `+0.00696`,
95% CI `[+0.00355, +0.01030]`, with `P(OT > MO) = 1.0`. H512 also raised the
mean AUC over default OT-MIL by `+0.00120`, but that comparison was not
significant: 95% CI `[-0.00118, +0.00365]`.

## PANDA Robustness

| Patch budget | MO-MIL AUC | OT-MIL AUC | Difference |
|---:|---:|---:|---:|
| 128 | 0.8744 | **0.8820** | +0.0077 |
| 256 | 0.8902 | **0.8977** | +0.0075 |
| 512 | 0.8996 | **0.9051** | +0.0055 |

At 512 patches, OT-MIL selected `0.3755 +/- 0.0250` of the soft mass.
Selected evidence exceeded equal-mass random evidence by `0.0165 +/- 0.0046`
true-class confidence. Complement AUC was `0.8800`, below selected AUC
`0.9051`, and the mean necessity confidence drop was `0.0576`.

OT-MIL has 1,461,815 trainable parameters; MO-MIL has 14,076,936 under the
evaluated configurations.

## Camelyon16 Ablations

The following variants were selected and compared using a validation-only
split with all test columns removed.

| Variant, seed 2024 | Best validation AUC |
|---|---:|
| UOT-only, H256, temperature 0.5 | **0.9091** |
| UOT-only, H512, temperature 0.5 | 0.8040 |
| UOT-only, H256, temperature 0.1 | 0.8494 |
| UOT-only, H512, temperature 0.1 | 0.8210 |
| Quantile gate, 5% | 0.8352 |
| Quantile gate, 10% | 0.8665 |
| Quantile gate, 20% | 0.8977 |
| Instance evidence weight 0.25 | 0.8906 |
| Instance evidence weight 0.5 | 0.9048 |
| Instance evidence weight 1.0 | 0.8835 |
| 20% gate + instance weight 0.5 | 0.8636 |

None exceeded the pre-existing UOT-only validation result, so none was
promoted to a new formal test comparison.

Inference-budget robustness did not close the gap:

| Budget | UOT-only AUC | MO-MIL AUC |
|---:|---:|---:|
| 512 | 0.7576 | 0.8116 |
| 1,024 | 0.7330 | 0.8177 |
| 2,048 | 0.7590 | 0.8319 |
| 4,096 | 0.7656 | 0.8404 |

### Rare-lesion branch

A binary rare-lesion branch was added after the first Camelyon16 analysis. It
uses a positive-class patch score and weighted top-k log-mean-exp pooling. The
branch is disabled by default, preserves legacy checkpoints, and rejects
active use for multiclass tasks.

The validation-only screen disabled all auxiliary regularizers and used all
4,096 cached candidates. Directly injecting the rare score into the OT gate
was unstable. The frozen configuration used `rare_instance_topk=16`,
`rare_instance_weight=0.25`, and `rare_gate_weight=0`.

| Seed | Validation AUC | Balanced accuracy | Macro-F1 |
|---:|---:|---:|---:|
| 2024 | 0.8920 | 0.8068 | 0.7945 |
| 2025 | 0.9332 | 0.8395 | 0.8442 |
| 2026 | 0.8935 | 0.7926 | 0.7905 |
| Mean +/- SD | 0.9063 +/- 0.0234 | 0.8130 +/- 0.0240 | 0.8097 +/- 0.0299 |

The configuration was then compared fairly with MO-MIL using 4,096 patches,
the same full split, seeds, balanced sampler, epoch budget, and early stopping.

| Method, 4,096 patches | Test AUC | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| MO-MIL | **0.8414 +/- 0.0349** | **0.7405** | **0.7340** |
| OT-MIL rare branch | 0.7798 +/- 0.0867 | 0.7188 | 0.7187 |

The rare branch therefore did not transfer its validation improvement to the
test set. Its seed AUCs were `0.7088`, `0.8764`, and `0.7543`, indicating
substantial initialization sensitivity. Equal-weight three-seed ensembles
reached 0.8253 for OT-MIL and 0.8480 for MO-MIL. An exploratory equal-weight
cross-model ensemble reached 0.8707, showing complementary errors, but this
was inspected after test evaluation and is not a registered formal result.

The candidate cache is a major remaining data limitation. Original positive
slides contain a median of 47,358 train, 47,199 validation, and 33,742 test
patches. The 4,096 cache retains median fractions of only 8.6%, 8.7%, and
12.1%, respectively. A 16,384-candidate cache was started to test whether
greater tissue coverage resolves the rare-lesion failure, but server
authentication became unavailable before completion could be verified.

## Commands

Main PANDA benchmark:

```bash
python experiments/run_benchmark.py \
  --split experiment_artifacts/splits/PANDA_R50_CACHE_split.csv \
  --dataset-name PANDA_R50_CACHE --num-classes 6 \
  --log-root experiment_artifacts/logs/panda_full \
  --models OT_MIL MO_MIL --seeds 2024 2025 2026 \
  --epochs 30 --patience 8 --max-instances 512 \
  --in-dim 1024 --device 0 --num-workers 4
```

PANDA robustness and statistics:

```bash
python experiments/evaluate_checkpoints.py \
  --run-root experiment_artifacts/logs/panda_full \
  --output-dir experiment_artifacts/logs/panda_robustness \
  --budgets 128 256 512 --device 0 --num-workers 4

python experiments/paired_bootstrap.py \
  --input-dir experiment_artifacts/logs/panda_robustness \
  --budget 512 --iterations 5000
```

Camelyon16 cache:

```bash
python experiments/cache_feature_subset.py \
  --split experiment_artifacts/splits/CAMELYON16_R50_split.csv \
  --output-dir /srv/storage1/hdd/cff/wqy/otmil/camelyon16_cache_4096 \
  --output-split experiment_artifacts/splits/CAMELYON16_R50_CACHE4096_split.csv \
  --max-candidates 4096
```

Camelyon16 formal runs used `train_mil.py` with seeds 2024-2026, 30 epochs,
patience 8, balanced sampling, and `Model.max_instances=512`. UOT-only set
necessity, minimality, diversity, full-classification, and consistency weights
to zero. Validation-only searches used a split with zero test rows.

Camelyon16 rare-lesion formal comparison:

```bash
python train_mil.py --yaml_path configs/OT_MIL.yaml --options \
  General.seed=2024 General.num_classes=2 General.num_epochs=30 \
  General.device=0 General.num_workers=2 \
  General.earlystop.use=true General.earlystop.patience=8 \
  Dataset.DATASET_NAME=CAM16_OT_RARE4096 \
  Dataset.dataset_csv_path=experiment_artifacts/splits/CAMELYON16_R50_CACHE4096_split.csv \
  Dataset.balanced_sampler.use=true \
  Logs.log_root_dir=/srv/storage1/hdd/cff/wqy/otmil/experiment_logs/camelyon16_rare_formal4096 \
  Model.max_instances=4096 Model.sampling=random \
  Model.rare_instance_weight=0.25 Model.rare_instance_topk=16 \
  Model.rare_gate_weight=0 Model.necessity_weight=0 \
  Model.minimality_weight=0 Model.diversity_weight=0 \
  Model.full_classification_weight=0 Model.consistency_weight=0 \
  Model.scheduler.cosine_config.T_max=28

python train_mil.py --yaml_path configs/MO_MIL.yaml --options \
  General.seed=2024 General.num_classes=2 General.num_epochs=30 \
  General.device=0 General.num_workers=2 \
  General.earlystop.use=true General.earlystop.patience=8 \
  Dataset.DATASET_NAME=CAM16_MO4096 \
  Dataset.dataset_csv_path=experiment_artifacts/splits/CAMELYON16_R50_CACHE4096_split.csv \
  Dataset.balanced_sampler.use=true \
  Logs.log_root_dir=/srv/storage1/hdd/cff/wqy/otmil/experiment_logs/camelyon16_rare_formal4096 \
  Model.in_dim=1024 Model.max_instances=4096 Model.sampling=random
```

The two commands were repeated for seeds 2025 and 2026.

TCGA-NSCLC preparation and patient-level split:

```bash
python experiments/prepare_tcga_nsclc.py \
  --luad-dir /srv/storage/hdd/ychong/chong/Datasets/TCGA-LUAD/features/plip_qc/pt_files \
  --lusc-dir /srv/storage/hdd/ychong/chong/Datasets/TCGA-LUSC/features/plip_qc/pt_files \
  --output-dir /srv/storage1/hdd/cff/wqy/otmil/nsclc_plip_qc_repro

python experiments/prepare_split.py \
  --labels /srv/storage1/hdd/cff/wqy/otmil/nsclc_plip_qc_repro/NSCLC_labels.csv \
  --feature-dir /srv/storage1/hdd/cff/wqy/otmil/nsclc_plip_qc_repro/pt_files \
  --output-dir /srv/storage1/hdd/cff/wqy/otmil/nsclc_plip_qc_repro \
  --dataset-name NSCLC_PLIP_QC --group-column patient_id --seed 2024
```

TCGA-NSCLC formal comparison:

```bash
python experiments/run_benchmark.py \
  --split /srv/storage1/hdd/cff/wqy/otmil/nsclc_plip_qc/NSCLC_PLIP_QC_split.csv \
  --dataset-name NSCLC_PLIP_QC --num-classes 2 \
  --log-root /srv/storage1/hdd/cff/wqy/otmil/experiment_logs/nsclc_plip_qc_formal \
  --models OT_MIL MO_MIL --seeds 2024 2025 2026 \
  --epochs 30 --patience 8 --max-instances 4096 \
  --in-dim 512 --num-workers 2 --balanced
```

PANDA H512 frozen-checkpoint evaluation:

```bash
python experiments/evaluate_checkpoints.py \
  --run-root experiment_artifacts/logs/panda_h512_tune \
  --output-dir experiment_artifacts/logs/panda_h512_formal_eval \
  --models OT_MIL --budgets 4096 --device 0 --num-workers 4 \
  --split-override experiment_artifacts/splits/PANDA_R50_CACHE_split.csv
```

Final verification:

```bash
python -m pytest -q
# 29 passed
```

## Git History

Research-stage commits:

- `4979409cf93b11ef23ad58212a1e7cadd35b31de` initial OT-MIL prototype
- `e4c0f90a064cc9f39922fa87a15bef42ad4ba83f` reproducible evaluation,
  log-domain UOT, diagnostics
- `3137f4d55c3ca09af2b858ab7a40bce3669f9749` dataset-level sampling
- `dc9c94e4c36ff094a92919d7eddee1fa154e3471` safe YAML overrides
- `cdedfc6727f4f07c14db5e13ae19fd256e4cd817` lower evidence regularization
  weights
- `76cc66607d7285a7bc1794b91531f6b910225675` pytest collection fix
- `e9f0ad39c61f903aa0cefe06282afe92c24f40cd` global mean/std submeasure
  representation
- `8159d1a08b3ce5c25bb6c5e4a976d57905df935f` reproducible benchmark runner
- `7ef04c3f65ec1368e3b4c02cf236a3289afa6d9a` split and Git provenance
  manifests
- `599bbba20e6b7258a379ac3bb5a2e5a60561b5f2` equal-mass random control
- `618028d4faae31c0b3a952ddb8a564e3f55bd114` budget robustness and paired
  bootstrap
- `5398b8905bff3f495e7dfa06a617c1cdb3c84c03` reproducible candidate cache
- `6410c31344937fba7c5590281be3af806ee49e65` interrupted-cache recovery
- `2727eaad207831b284edfa2586c96675db3b6dfe` quantile sparse gate
- `58265544a5177703833f0cea69a116a9ce75352c` OT-gated instance evidence
  branch and legacy compatibility
- `5cd547e72838e686e25be8ad479787f558d9075d` binary paired-bootstrap fix
- `706a6cb75a3839b44411c5715fb4b3033eace9e9` patient-aware deterministic
  splits
- `8bf2a22f61b6bbd9161e2527aa4ccaaa84cd603b` frozen-checkpoint split
  override with provenance
- `597c16dd79a1db15c0173b7ac4e3c43c58d9596f` reproducible TCGA-NSCLC
  preparation
- `8fde720` binary rare-lesion instance evidence with legacy-compatible
  defaults

## Failures And Resolutions

- WSL intermittently returned `E_UNEXPECTED`; runs were resumed after the
  subsystem recovered without changing results.
- Shared A100s stayed at 100% utilization. The experiments used roughly
  0.5 GB additional GPU memory and lower CPU priority.
- Direct H5 training was I/O-bound. A deterministic 4,096-candidate cache
  reduced the first epoch from more than three minutes to about 36 seconds.
- The `wqy` home directory had an effective quota near 500 MB despite free
  filesystem capacity. Caches and logs were moved to the writable `cff` disk,
  with a repository symlink preserving log paths.
- An interrupted cache produced a corrupt PT file. The cache builder now
  validates and atomically rebuilds incomplete files.
- The bootstrap tool originally handled only multiclass probabilities. Binary
  AUC support and a regression test were added.
- Sampled CONCH files included corrupt serialized tensors, so TCGA-NSCLC used
  the independently checked PLIP-QC feature set.
- The server environment did not include pytest. Repository tests were run
  locally, while server smoke tests used Python's standard `unittest`.
- A 10,000-iteration multiclass PANDA bootstrap exceeded the initial local
  timeout; the registered 5,000-iteration protocol completed successfully.
- The first 4,096-patch MO-MIL command omitted `Model.in_dim=1024` and failed
  before its first training batch with a matrix-shape error. It was restarted
  with the correct feature dimension; no result from the failed run was used.
- During construction of the 16,384-candidate Camelyon16 cache, the server
  stopped accepting the previously working `wqy` public keys. Password
  authentication also rejected the supplied `wqy` and `qfh` credentials, so
  cache completion and follow-up experiments remain unverified.

## Research Assessment

The PANDA result supports a narrow claim: OT-induced submeasure selection is
competitive and consistently improves over the evaluated MO-MIL baseline on a
large multiclass cohort, including reduced patch budgets.

The follow-up evidence now supports improvement over MO-MIL on PANDA and
TCGA-NSCLC. PANDA H512 is significant under paired bootstrap. TCGA-NSCLC has a
higher three-seed mean, much lower variance, and better accuracy-based metrics,
but its AUC confidence interval still crosses zero. Camelyon16 remains a clear
negative result.

These results support continuing toward an AAAI submission, but not yet a
broad superiority claim. The defensible claim is that OT-induced submeasure
selection improves a large multiclass cohort and transfers positively to a
patient-split TCGA subtype task, while rare-lesion sensitivity remains a known
limitation. Submission readiness still requires official Mamba2 MO-MIL
validation, at least one more independent cohort or cross-center test, and a
principled mechanism that addresses the Camelyon16 failure.
