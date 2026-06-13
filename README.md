# OT-MIL

Research implementation of optimal-transport-based diagnostic submeasure
learning for whole-slide image multiple-instance learning.

## Project Status

This project is research-complete and archived as of June 13, 2026.

The implementation is functional and reproducible, but the experiments do not
support a broad claim that OT-MIL consistently outperforms strong MIL
baselines. The strongest result is a significant improvement over the
same-code OT baseline on TCGA-RCC. PANDA is positive, BRCA-PAM50 is competitive,
and Camelyon16 and STAD are clear negative results.

The authoritative conclusion is in [PROJECT_CLOSEOUT.md](PROJECT_CLOSEOUT.md).
Detailed commands, hashes, failures, and ablations are retained in
[docs/EXPERIMENT_REPORT_2026-06-10.md](docs/EXPERIMENT_REPORT_2026-06-10.md).

## Repository Map

- `modules/OT_MIL/`: OT-MIL model and objectives
- `process/OT_MIL/`: training and validation workflow
- `configs/OT_MIL.yaml`: general configuration
- `configs/OT_MIL_BINARY.yaml`: retained binary configuration
- `configs/OT_MIL_MULTICLASS.yaml`: retained diagnostic-mass configuration
- `experiments/`: preparation, sealed evaluation, aggregation, and statistics
- `tests/`: model and experiment regression tests
- `docs/OT-MIL.md`: original research idea
- `docs/EXPERIMENT_REPORT_2026-06-10.md`: complete research log

## Minimal Usage

The project was run with the existing `pathowm` mamba environment:

```bash
mamba activate pathowm
python -m pytest -q
python train_mil.py --yaml_path configs/OT_MIL_MULTICLASS.yaml --options \
  Dataset.dataset_csv_path=/path/to/split.csv \
  Dataset.DATASET_NAME=YOUR_DATASET \
  Logs.log_root_dir=/path/to/logs \
  General.num_classes=4 \
  Model.in_dim=1536
```

For controlled baseline comparisons:

```bash
python experiments/run_benchmark.py \
  --split /path/to/train_val.csv \
  --dataset-name YOUR_DATASET \
  --num-classes 4 \
  --log-root /path/to/logs \
  --models AB_MIL MO_MIL OT_MIL_ORIGINAL OT_MIL_CLASS_MASS \
  --seeds 2024 2025 2026 \
  --epochs 25 --patience 6 \
  --max-instances 4096 --in-dim 1536
```

Dataset features, checkpoints, and large logs are intentionally not committed.
Their local paths and provenance hashes are recorded in the experiment report.

## Final Recommendation

Treat this repository as a completed research artifact and a source of reusable
OT-MIL components, not as a finished state-of-the-art method. Any future paper
should begin with a new hypothesis and fresh external test sets rather than
continuing to tune the current benchmark suite.

## MIR-MIL Successor Prototype

A separate successor prototype based on Measure Influence Response is
implemented in `modules/MIR_MIL/`. Unlike OT-MIL, it models a neural functional
of the empirical tissue measure and derives patch attribution from its
functional derivative. See
`docs/MIR-MIL_IMPLEMENTATION.md` for the method mapping, mathematical tests,
smoke-run evidence, and faithfulness evaluation command.
