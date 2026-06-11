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

### TCGA-RCC UNI2-h

- Official feature source: `MahmoodLab/UNI2-h-features`
- Projects: TCGA-KIRC, TCGA-KIRP, and TCGA-KICH
- Feature dimension: 1,536
- Slides: 925; patients: 895
- Class slides: 519 KIRC / 297 KIRP / 109 KICH
- Patient-level split: 561 train / 185 validation / 179 test slides
- Label CSV SHA-256:
  `506b041e7032c70d9f1b148e94378019239ee2e29c03d41dcab54d8dc0485f6d`
- Assignment SHA-256:
  `a054aba0d4b1e5bddbab0986d53e0bbf0883fa89a4cb4db5955427d826a1abb1`
- Cached split SHA-256:
  `e34b60c9decf75c25558eba3640481ceef3ad27e3c77a18910b755311ff90868`
- Fixed cache: 3,602,153 instances and 22,133,947,932 bytes

The official archives were downloaded directly from Hugging Face without
downloading TCGA whole-slide images. Project membership defines the RCC
subtype label. H5 structure and feature dimension were validated before
patient-level splitting. The same deterministic 4,096-candidate cache was used
by both methods.

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
| TCGA-RCC UNI2-h, 4,096 patches | MO-MIL | **0.9904 +/- 0.0025** | 0.9441 | 0.9599 | 0.9378 |
| TCGA-RCC UNI2-h, 4,096 patches | OT-MIL | 0.9892 +/- 0.0029 | **0.9516** | **0.9642** | **0.9486** |

Paired stratified bootstrap, 5,000 iterations unless noted:

| Dataset/configuration | Mean AUC difference vs MO-MIL | 95% CI | P(OT > MO) |
|---|---:|---:|---:|
| PANDA OT-MIL, 512 patches | +0.00555 | [+0.00208, +0.00896] | 0.9988 |
| Camelyon16 OT-MIL, 512 patches | -0.09422 | [-0.16903, -0.01799] | 0.0072 |
| Camelyon16 UOT-only, 512 patches | -0.05398 | [-0.12358, +0.01278] | 0.0614 |
| TCGA-NSCLC OT-MIL, 4,096 patches (10,000) | +0.00752 | [-0.00515, +0.02076] | 0.8720 |
| TCGA-RCC macro-AUC, 4,096 patches (10,000) | -0.00122 | [-0.00457, +0.00192] | 0.2192 |
| TCGA-RCC balanced accuracy (10,000) | +0.00436 | [-0.00606, +0.01493] | 0.8046 |
| TCGA-RCC macro-F1 (10,000) | +0.01081 | [-0.00251, +0.02523] | 0.9401 |

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

TCGA-RCC UNI2-h preparation and formal comparison:

```bash
experiments/download_uni2h_rcc.sh \
  /mnt/d/datasets/UNI2-h_features none \
  TCGA-KIRC TCGA-KIRP TCGA-KICH

experiments/extract_uni2h_projects.sh \
  /mnt/d/datasets/UNI2-h_features \
  TCGA-KIRC TCGA-KIRP TCGA-KICH

python experiments/prepare_tcga_projects.py \
  --project KIRC=/mnt/d/datasets/UNI2-h_features/TCGA-KIRC \
  --project KIRP=/mnt/d/datasets/UNI2-h_features/TCGA-KIRP \
  --project KICH=/mnt/d/datasets/UNI2-h_features/TCGA-KICH \
  --output-dir /home/sigirika/datasets/tcga_rcc_uni2h

python experiments/prepare_split.py \
  --labels /home/sigirika/datasets/tcga_rcc_uni2h/TCGA_projects_labels.csv \
  --feature-dir /home/sigirika/datasets/tcga_rcc_uni2h/features \
  --output-dir /home/sigirika/datasets/tcga_rcc_uni2h \
  --dataset-name TCGA_RCC_UNI2H --group-column patient_id \
  --feature-extension .h5 --seed 2024

python experiments/run_benchmark.py \
  --split /home/sigirika/datasets/tcga_rcc_uni2h/TCGA_RCC_UNI2H_CACHE4096_split.csv \
  --dataset-name TCGA_RCC_UNI2H_CACHE4096 --num-classes 3 \
  --log-root /home/sigirika/experiment_logs/tcga_rcc_uni2h_formal \
  --models OT_MIL MO_MIL --seeds 2024 2025 2026 \
  --epochs 30 --patience 8 --max-instances 4096 \
  --in-dim 1536 --device 0 --num-workers 2 --balanced
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
# 52 passed
```

## TCGA-COAD Controlled Task Suite

Official UNI2-h features were matched exactly to 442 label rows and 434
patients. One metastatic slide marked `recommended_use_primary=No` was
excluded, leaving 441 primary-tumor slides. A single patient-level
260/87/87 train/validation/test assignment was optimized jointly across all
task classes and reused for every task.

The tasks are useful as a controlled task-complexity suite, but they are not
independent datasets. Of the 323 patients with both MSI and HMCINGS labels,
56/57 MSI patients were HM, while 217/217 CIN patients were MSS. CMS1 was
also strongly enriched for MSI and HM. Therefore, these results measure model
preference under different label granularities on one cohort rather than
three external validations.

Patient counts:

| Task | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| MSI: MSS/MSI | 207/48 | 73/14 | 69/14 |
| CMS: 1/2/3/4 | 29/74/25/48 | 9/25/9/19 | 11/21/9/18 |
| HMCINGS: CIN/GS/HM | 128/27/38 | 45/8/11 | 44/10/12 |

Formal 30-epoch, patience-8, balanced-sampling results:

| Task | Model | Macro AUC | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| MSI | OT-MIL | 0.9184 +/- 0.0454 | 0.8095 | 0.7905 | 0.7382 |
| MSI | MO-MIL | **0.9473 +/- 0.0050** | **0.8690** | **0.8548** | **0.8017** |
| CMS | OT-MIL | **0.7777 +/- 0.0053** | **0.5989** | **0.5863** | **0.5958** |
| CMS | MO-MIL | 0.7579 +/- 0.0185 | 0.5876 | 0.5525 | 0.5526 |
| HMCINGS | OT-MIL | 0.8463 +/- 0.0525 | 0.7562 | 0.6642 | 0.6446 |
| HMCINGS | MO-MIL | **0.8828 +/- 0.0283** | **0.7711** | **0.6716** | **0.6701** |

OT-MIL exceeded MO-MIL on CMS in all three seeds: macro AUC +0.0198,
balanced accuracy +0.0337, and macro-F1 +0.0432. The 10,000-iteration paired
bootstrap probabilities that OT-MIL was better were 0.903, 0.872, and 0.919,
respectively, although all 95% confidence intervals crossed zero because the
CMS test set contains only 59 slides/patients.

The three-seed probability ensemble preserved the CMS preference: macro AUC
0.7896 versus 0.7740 and macro-F1 0.5922 versus 0.5489. On MSI, the OT-MIL
ensemble improved macro-F1 over MO-MIL (0.8070 versus 0.7926) but not macro
AUC (0.9082 versus 0.9520). HMCINGS continued to favor MO-MIL.

OT-MIL selected roughly 49% of instances in all three tasks. The selected
submeasure reduced true-class confidence when removed, especially on
HMCINGS, but selected and equal-mass random subsets had similar AUC. This
means the current evidence supports a task-dependent predictive benefit on
CMS, but not yet a strong minimal-submeasure explanation claim.

Preparation and formal comparison:

```bash
python experiments/prepare_coad_multitask.py \
  --labels /mnt/d/datasets/UNI2-h_features/TCGA-COAD/label/TCGA_COAD_all_slide_internal_labels.csv \
  --feature-dir /mnt/d/datasets/UNI2-h_features/TCGA-COAD \
  --output-dir /home/sigirika/datasets/coad_multitask_v1 \
  --seed 2024 --search-iterations 20000

python experiments/cache_feature_subset.py \
  --split /home/sigirika/datasets/coad_multitask_v1/COAD_cms_split.csv \
  --output-dir /home/sigirika/datasets/coad_multitask_v1/cache4096 \
  --output-split /home/sigirika/datasets/coad_multitask_v1/cached_splits/COAD_cms_split.csv \
  --max-candidates 4096

python experiments/run_benchmark.py \
  --split /home/sigirika/datasets/coad_multitask_v1/cached_splits/COAD_cms_split.csv \
  --dataset-name COAD_CMS --num-classes 4 \
  --log-root /home/sigirika/experiment_logs/coad_formal_v1/cms \
  --models OT_MIL MO_MIL --seeds 2024 2025 2026 \
  --epochs 30 --patience 8 --max-instances 4096 \
  --in-dim 1536 --device 0 --num-workers 2 --balanced
```

The benchmark command was also run for MSI (`num-classes=2`) and HMCINGS
(`num-classes=3`) using their corresponding shared-split CSV files.

## Cross-Cohort Binary Versus Multiclass Validation

To test whether the COAD task preference replicated outside one cohort, a
second controlled task suite was built from official TCGA-UCEC UNI2-h
features. Both cohorts use the same 1,536-dimensional encoder features, a
shared patient split within each cohort, the same 4,096-instance cache,
balanced sampling, 30 epochs, patience 8, and seeds 2024-2026.

The pre-specified 2-by-2 matrix was:

| Cohort | Binary molecular task | Four-class molecular task |
| --- | --- | --- |
| TCGA-COAD | MANTIS MSI/MSS | CMS1/CMS2/CMS3/CMS4 |
| TCGA-UCEC | MANTIS MSI/MSS | POLE/MSI/CN-low/CN-high |

UCEC labels came from the cBioPortal PanCancer Atlas study
`ucec_tcga_pan_can_atlas_2018`. MANTIS scores below 0.4 were MSS, scores above
0.6 were MSI, and the indeterminate interval was excluded. Of 566 feature
files, 565 were primary-tumor slides. The final union contained 542 labelled
slides from 484 patients.

UCEC patient counts:

| Task | Train | Validation | Test |
| --- | ---: | ---: | ---: |
| MSI: MSS/MSI | 198/65 | 66/20 | 68/21 |
| Subtype: POLE/MSI/CN-low/CN-high | 29/84/83/88 | 9/26/28/28 | 9/27/27/30 |

Some UCEC patients had multiple primary slides, so frozen-checkpoint
predictions were also averaged within each patient. The following table uses
patient-level metrics as the primary analysis:

| Cohort/task | Model | Macro AUC | Accuracy | Balanced accuracy | Macro-F1 |
| --- | --- | ---: | ---: | ---: | ---: |
| COAD MSI | MO-MIL | **0.9465** | **0.8675** | **0.8539** | **0.8011** |
| COAD MSI | OT-MIL | 0.9179 | 0.8112 | 0.7916 | 0.7404 |
| COAD CMS | MO-MIL | 0.7579 | 0.5876 | 0.5525 | 0.5526 |
| COAD CMS | OT-MIL | **0.7777** | **0.5989** | **0.5863** | **0.5958** |
| UCEC MSI | MO-MIL | **0.7318** | **0.7790** | **0.6689** | **0.6758** |
| UCEC MSI | OT-MIL | 0.6909 | 0.7640 | 0.6481 | 0.6491 |
| UCEC subtype | MO-MIL | 0.7519 | **0.5591** | **0.4750** | **0.4728** |
| UCEC subtype | OT-MIL | **0.7536** | 0.5305 | 0.4685 | 0.4523 |

Patient-level macro-AUC differences, OT-MIL minus MO-MIL:

| Task type | COAD | UCEC |
| --- | ---: | ---: |
| Binary | -0.0286 | -0.0409 |
| Four-class | +0.0198 | +0.0017 |

For every cohort and training seed, the paired task-type interaction

```text
(multiclass OT-MIL - multiclass MO-MIL)
-
(binary OT-MIL - binary MO-MIL)
```

was positive. The mean macro-AUC interaction was `+0.0455`. A hierarchical
bootstrap that first resampled the two cohorts and then resampled seeds within
each cohort gave a 95% interval of `[+0.0157, +0.0806]`. The corresponding
balanced-accuracy interaction was `+0.0552`, with interval
`[+0.0014, +0.1313]`. Macro-F1 was directionally positive at `+0.0551`, but
its interval `[-0.0078, +0.1493]` crossed zero.

These results support a relative task-type preference: the current OT-MIL is
consistently less disadvantaged, and sometimes superior, on four-class
molecular tasks than on binary molecular-axis tasks. They do not establish
absolute multiclass superiority. In particular, UCEC subtype macro-AUC was
essentially tied and its accuracy and macro-F1 favored MO-MIL. With only two
independent cohorts, the conservative cohort-level one-sided sign-test
p-value is 0.25. More independent cohorts are required before treating the
interaction as a general law.

UCEC preparation and benchmark:

```bash
experiments/download_uni2h_rcc.sh \
  /mnt/d/datasets/UNI2-h_features none TCGA-UCEC

experiments/extract_uni2h_projects.sh \
  /mnt/d/datasets/UNI2-h_features TCGA-UCEC

python experiments/prepare_ucec_multitask.py \
  --feature-dir /mnt/d/datasets/UNI2-h_features/TCGA-UCEC \
  --output-dir /home/sigirika/datasets/ucec_multitask_v1 \
  --seed 2024 --search-iterations 20000

python experiments/run_benchmark.py \
  --split /home/sigirika/datasets/ucec_multitask_v1/cached_splits/UCEC_subtype_split.csv \
  --dataset-name UCEC_SUBTYPE --num-classes 4 \
  --log-root /home/sigirika/experiment_logs/ucec_formal_v1/subtype \
  --models OT_MIL MO_MIL --seeds 2024 2025 2026 \
  --epochs 30 --patience 8 --max-instances 4096 \
  --in-dim 1536 --device 0 --num-workers 2 --balanced

python experiments/aggregate_group_predictions.py \
  --input-dir /home/sigirika/experiment_logs/ucec_formal_v1/eval_subtype \
  --assignments /home/sigirika/datasets/ucec_multitask_v1/UCEC_subtype_assignments.csv \
  --cache-manifest /home/sigirika/datasets/ucec_multitask_v1/cached_splits/UCEC_subtype_split.csv.manifest.json \
  --output-dir /home/sigirika/experiment_logs/ucec_formal_v1/grouped_subtype

python experiments/analyze_task_type_preference.py \
  --cohort COAD=/home/sigirika/experiment_logs/coad_formal_v1/grouped_msi/grouped_results.csv,/home/sigirika/experiment_logs/coad_formal_v1/grouped_cms/grouped_results.csv \
  --cohort UCEC=/home/sigirika/experiment_logs/ucec_formal_v1/grouped_msi/grouped_results.csv,/home/sigirika/experiment_logs/ucec_formal_v1/grouped_subtype/grouped_results.csv \
  --metric macro_auc --iterations 10000 --seed 2024
```

## Theory-Driven Binary Revision

The original implementation normalized every UOT row by its transported
mass before building the slide representation. UOT therefore acted mainly as
prototype routing, while a standardized within-bag mass rank produced an
approximately 50% gate. On frozen test sets, selected and equal-mass random
subsets had nearly identical AUC.

The revised binary model adds four mechanisms:

1. mass-faithful submeasures built directly from `T * gate`;
2. class-conditional evidence potentials added to the UOT retention prior;
3. a zero-initialized residual evidence head anchored to full-bag logits;
4. a shared 64-dimensional projection of each prototype barycenter, reducing
   parameters from 1,601,623 to 819,479.

All configuration selection used train/validation CSVs with empty test
columns. The frozen configuration is `configs/OT_MIL_BINARY.yaml`.

Validation-only ablations, seed 2024:

| Variant | COAD MSI AUC | COAD CMS AUC |
| --- | ---: | ---: |
| Legacy OT-MIL | 0.9755 | 0.8400 |
| Mass-faithful scalar gate | 0.9653 | 0.8271 |
| Class-conditional submeasures | 0.9521 | 0.8228 |
| Full-bag residual, rank 256 | 0.9569 | 0.8402 |
| Full-bag residual, rank 64 | 0.9706 | **0.8436** |

The rank-64 binary model also reduced three-seed COAD-MSI macro-F1
variability from `0.7957 +/- 0.1441` to `0.8632 +/- 0.0008`.

Frozen patient-level MSI results:

| Cohort/model | Macro AUC | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| COAD legacy OT-MIL | 0.9179 | 0.7916 | 0.7404 |
| COAD MO-MIL | **0.9465** | 0.8539 | 0.8011 |
| COAD revised OT-MIL | 0.9286 | 0.7876 | 0.7887 |
| UCEC legacy OT-MIL | 0.6909 | 0.6481 | 0.6491 |
| UCEC MO-MIL | 0.7318 | 0.6689 | **0.6758** |
| UCEC revised OT-MIL | **0.7491** | **0.6968** | 0.6733 |

For UCEC, the paired three-seed macro-AUC difference versus MO-MIL was
`+0.0173`, with a 95% patient bootstrap interval of
`[-0.0348, +0.0693]`. COAD remained lower by `-0.0179`, with interval
`[-0.0487, +0.0114]`. The revision therefore changes the binary result from
two losses to one win and one loss, but does not establish universal
superiority.

Frozen slide-level selection diagnostics:

| Cohort | Selected AUC | Full AUC | Random-gate AUC |
| --- | ---: | ---: | ---: |
| COAD MSI | **0.9293** | 0.9054 | 0.9048 |
| UCEC MSI | **0.7402** | 0.7325 | 0.7329 |

Unlike the legacy model, the learned submeasure now consistently improves
over both the full representation and a permuted equal-mass gate.

## Binary Likelihood-Ratio Follow-up

The next revision tested whether binary molecular prediction should be
parameterized directly as an antisymmetric log-likelihood ratio. The learned
gate potential was decomposed as

`phi_0(x) = q(x) - s(x), phi_1(x) = q(x) + s(x)`,

where `s` is class-contrast evidence and `q` is class-shared diagnostic
quality. All mechanism selection used the same train/validation-only CSVs
with empty test columns.

Three-seed validation results:

| Variant | COAD MSI AUC | UCEC MSI AUC |
| --- | ---: | ---: |
| Pure log-odds, `q=0` | **0.9742 +/- 0.0150** | 0.6958 +/- 0.0240 |
| Shared quality, `q=1` | 0.9556 +/- 0.0152 | **0.7223 +/- 0.0167** |
| Learnable task coupling | 0.9690 | 0.7014 |
| Dual gate + endpoint supervision | 0.9677 +/- 0.0026 | 0.7188 +/- 0.0455 |

The two endpoint models were complementary. A validation-only probability
ensemble with shared-endpoint weight `0.60` reached a cross-cohort mean AUC
of `0.8487`, versus `0.8350` for pure log-odds and `0.8390` for shared
quality. A single shared-backbone dual-gate model did not reproduce that
gain. Endpoint supervision recovered seed-2024 UCEC validation AUC from
`0.7038` to `0.7451`, but one UCEC seed collapsed to `0.6663`; its contrast,
shared, and full-bag AUCs all declined, indicating shared-representation
optimization instability rather than a mixture-only failure.

After freezing `binary_dual_gate_mix=0.6` and
`binary_dual_endpoint_weight=1.0`, patient-level formal evaluation gave:

| Cohort/model | Macro AUC | Balanced accuracy | Macro-F1 |
| --- | ---: | ---: | ---: |
| COAD MO-MIL | **0.9465** | **0.8539** | 0.8011 |
| COAD dual-gate OT-MIL | 0.9231 +/- 0.0006 | 0.8328 | **0.8209** |
| UCEC MO-MIL | **0.7318** | **0.6689** | **0.6758** |
| UCEC dual-gate OT-MIL | 0.6937 +/- 0.0389 | 0.6340 | 0.6313 |

The dual-gate hypothesis therefore failed frozen-test generalization. The
default `OT_MIL_BINARY.yaml` was restored to the previously validated v2
class-conditional residual model. The likelihood-ratio, shared-quality,
balancing, task-coupling, and dual-gate mechanisms remain available as
explicit ablations, all disabled by default.

One local dual-gate run failed after epoch 10 with `cudaErrorUnknown` while
the Windows-side GPU had about 8 GB of unreported activity. A diagnostic
rerun with `CUDA_LAUNCH_BLOCKING=1` was terminated by the command timeout
after epoch 17. Both runs were excluded; a clean rerun completed normally.

Representative validation command:

```bash
python train_mil.py --yaml_path configs/OT_MIL_BINARY.yaml --options \
  General.seed=2024 General.num_classes=2 General.num_epochs=25 \
  Dataset.dataset_csv_path=/home/sigirika/experiment_splits/otmil_v2_tuning/UCEC_MSI_train_val.csv \
  Dataset.DATASET_NAME=UCEC_MSI_DUAL_SUP_S2024 \
  Logs.log_root_dir=/home/sigirika/experiment_logs/otmil_v3_tuning/dual_gate_supervised \
  Model.in_dim=1536 Model.max_instances=4096 \
  Model.binary_likelihood_ratio=True \
  Model.binary_common_gate_weight=1.0 \
  Model.binary_dual_gate_mix=0.6 \
  Model.binary_dual_endpoint_weight=1.0
```

Representative commands:

```bash
python train_mil.py --yaml_path configs/OT_MIL_BINARY.yaml --options \
  General.seed=2024 General.num_classes=2 General.num_epochs=25 \
  Dataset.dataset_csv_path=/home/sigirika/experiment_splits/otmil_v2_tuning/UCEC_MSI_train_val.csv \
  Dataset.DATASET_NAME=UCEC_MSI_LR \
  Logs.log_root_dir=/home/sigirika/experiment_logs/otmil_v2_tuning/ucec_transfer \
  Model.in_dim=1536 Model.max_instances=4096

python experiments/evaluate_checkpoints.py \
  --run-root /home/sigirika/experiment_logs/otmil_v2_tuning/ucec_transfer/UCEC_MSI_LR \
  --output-dir /home/sigirika/experiment_logs/otmil_v2_frozen/ucec_msi \
  --models OT_MIL --budgets 4096 --device 0 --num-workers 2 \
  --split-override /home/sigirika/datasets/ucec_multitask_v1/cached_splits/UCEC_msi_split.csv
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
- `f283619` generic TCGA multi-project labels, H5 validation, and links
- `8de7572`, `ff4ff5c`, `2dc97ce`, `a4d3f98` reproducible, resumable UNI2-h
  download workflow
- `1f05278` atomic UNI2-h archive extraction
- `01ebb10` efficient batched UNI2-h H5 loading and caching
- `fa01328` paired bootstrap for classification metrics
- `badb471` shared patient-level TCGA-COAD multi-task preparation
- `7b2aa93` concurrency-safe atomic feature-cache writes
- `b2aa36e` reproducible TCGA-UCEC multi-task preparation
- `1de3c5f` patient-level probability aggregation
- `4cd5f7e` patient-level paired-bootstrap support
- `8582069` hierarchical binary-versus-multiclass preference analysis
- `df05025`, `8d2631d` antisymmetric binary likelihood-ratio evidence and
  symmetry-breaking initialization
- `708bdc8`, `af26222` shared diagnostic-quality decomposition and minimality
  regularization
- `8d5f925`, `9bf6ef4` contrast-balanced and task-level shared-evidence
  coupling
- `a1b29d3`, `3aa9a0f` dual-submeasure probability fusion and endpoint
  supervision
- `d957a98`, `e1d3508` frozen dual-gate evaluation configuration and
  restoration of the validated v2 default

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
- Parallel COAD cache builders initially raced on a shared `.tmp` filename.
  Temporary files are now uniquely named and atomically replaced; a
  concurrency regression test was added. No source feature was modified.
- The first UCEC Hugging Face Xet download repeatedly encountered TLS
  truncation and fell below 0.2 MB/s. The authenticated curl downloader
  resumed the 41 GB archive across four interrupted transfers and completed
  without restarting from zero.

## Research Assessment

The PANDA result supports a narrow claim: OT-induced submeasure selection is
competitive and consistently improves over the evaluated MO-MIL baseline on a
large multiclass cohort, including reduced patch budgets.

The follow-up evidence supports improvement over MO-MIL on PANDA and
TCGA-NSCLC. PANDA H512 is significant under paired bootstrap. TCGA-NSCLC has a
higher three-seed mean, much lower variance, and better accuracy-based metrics,
but its AUC confidence interval still crosses zero. On official TCGA-RCC
UNI2-h features, AUC is statistically tied while OT-MIL improves accuracy,
balanced accuracy, and macro-F1; macro-F1 improves in all three seeds, although
its bootstrap confidence interval still crosses zero. Camelyon16 remains a
clear negative result.

TCGA-COAD adds a useful controlled preference result: OT-MIL consistently
improves the four-class CMS task but loses on the strongly correlated binary
MSI and three-class genomic-instability tasks. This sharpens the method claim:
the current implementation appears better suited to heterogeneous multiclass
morphology than to near-binary molecular axes. Because all three tasks share
one cohort and their labels are biologically correlated, COAD contributes one
dataset-level validation plus a task-profile analysis, not three independent
validations.

The independent UCEC suite replicates the relative binary-versus-four-class
macro-AUC interaction, although it does not replicate a clear absolute
multiclass win. Across COAD and UCEC, both binary task means favor MO-MIL,
while both four-class macro-AUC means favor OT-MIL. The task-type interaction
is positive under hierarchical bootstrap, but only two independent cohorts
are available. The defensible conclusion is therefore a reproducible relative
preference signal, not proof that class count or latent morphological
heterogeneity is the causal mechanism.

These results support continuing toward an AAAI submission, but not yet a
broad superiority claim. The defensible claim is that OT-induced submeasure
selection improves a large multiclass cohort and transfers positively to a
patient-split TCGA subtype task, while rare-lesion sensitivity remains a known
limitation. Submission readiness still requires official Mamba2 MO-MIL
validation and preferably an external CPTAC evaluation. Camelyon16 can be
retained as a limitation rather than a main benchmark.
