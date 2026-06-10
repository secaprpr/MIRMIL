# OT-MIL Reproducibility Report

Date: 2026-06-10

## Scope

This report records the transition from the OT-MIL prototype to a tested,
reproducible implementation and its comparison with the repository's MO-MIL
baseline on PANDA and Camelyon16.

The shared protocol uses:

- seeds 2024, 2025, and 2026;
- validation macro-AUC for checkpoint selection;
- one final test evaluation per selected checkpoint;
- the same split, patch budget, deterministic validation/test sampling, and
  balanced training sampler for both methods;
- 30 epochs, early stopping patience 8, and 512 sampled patches per slide.

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

## Main Results

Mean and sample standard deviation over three seeds:

| Dataset | Method | Macro-AUC | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| PANDA | MO-MIL | 0.8996 +/- 0.0038 | 0.6363 | 0.5868 | 0.5887 |
| PANDA | OT-MIL | **0.9051 +/- 0.0037** | **0.6462** | **0.6043** | **0.6032** |
| Camelyon16 | MO-MIL | **0.8116 +/- 0.0272** | **0.7099** | **0.6629** | **0.6318** |
| Camelyon16 | OT-MIL | 0.7173 +/- 0.0766 | 0.6235 | 0.6397 | 0.6201 |
| Camelyon16 | UOT-only | 0.7576 +/- 0.0342 | 0.6420 | 0.6506 | 0.6196 |

Paired stratified bootstrap, 5,000 iterations:

| Dataset/configuration | Mean AUC difference vs MO-MIL | 95% CI | P(OT > MO) |
|---|---:|---:|---:|
| PANDA OT-MIL, 512 patches | +0.00555 | [+0.00208, +0.00896] | 0.9988 |
| Camelyon16 OT-MIL, 512 patches | -0.09422 | [-0.16903, -0.01799] | 0.0072 |
| Camelyon16 UOT-only, 512 patches | -0.05398 | [-0.12358, +0.01278] | 0.0614 |

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

Final verification:

```bash
python -m pytest -q
# 22 passed
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

## Research Assessment

The PANDA result supports a narrow claim: OT-induced submeasure selection is
competitive and consistently improves over the evaluated MO-MIL baseline on a
large multiclass cohort, including reduced patch budgets.

The current evidence does not support a dataset-general superiority claim.
On Camelyon16, the full objective is significantly worse than MO-MIL, and
UOT-only does not close the gap. Validation-only capacity, temperature,
sparsity, and instance-evidence searches failed to improve on the original
UOT-only validation AUC. This is a reproducible technical limitation rather
than an unresolved execution failure.

The current package is not ready for an AAAI main-paper submission claiming a
general superior method. A credible next paper iteration needs a principled
class-conditional or rare-lesion transport mechanism, additional datasets and
domain labels or augmentation invariance, and validation on the official
Mamba2 MO-MIL implementation rather than only the PyTorch fallback.
