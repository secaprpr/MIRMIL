# TCGA-RCC R50/UNI Feature Preparation

Date started: 2026-07-13

## Goal

Prepare TCGA-RCC for the same fixed-feature MIL workflow used for PANDA and BRACS:

- validate the local GDC WSI download;
- build slide labels and patient-grouped splits;
- generate CLAM-style patch coordinate H5 files;
- extract ResNet-50 and UNI features;
- write split CSVs that point to `pt_files` and `h5_files`.

## Dataset location

```text
/data15/data15_5/fanhao/datasets/TCGA-RCC/
```

Raw GDC download:

```text
/data15/data15_5/fanhao/datasets/TCGA-RCC/raw_gdc/
```

GDC audit manifest:

```text
/data15/data15_5/fanhao/datasets/TCGA-RCC/manifests/tcga_rcc_primary_tumor_diagnostic_slides_audit.tsv
```

## Label definition

The task is three-class RCC subtype classification using GDC `project_id`.

| class_name | label | project_id |
| --- | ---: | --- |
| KICH | 0 | TCGA-KICH |
| KIRC | 1 | TCGA-KIRC |
| KIRP | 2 | TCGA-KIRP |

Only primary tumor diagnostic SVS slides are included.

## Source integrity audit

Prepared with:

```bash
mamba run -n mirmil python experiments/prepare_tcga_rcc_pipeline.py \
  --manifest /data15/data15_5/fanhao/datasets/TCGA-RCC/manifests/tcga_rcc_primary_tumor_diagnostic_slides_audit.tsv \
  --raw-dir /data15/data15_5/fanhao/datasets/TCGA-RCC/raw_gdc \
  --wsi-root /data15/data15_5/fanhao/datasets/TCGA-RCC/WSI \
  --feature-root /data15/data15_5/fanhao/datasets/TCGA-RCC/features \
  --output-dir /data15/data15_5/fanhao/datasets/TCGA-RCC/metadata \
  --seed 2024
```

Result:

- slides: `926`
- patients: `897`
- manifest filename match: passed
- byte-size match: passed
- full MD5: not run by default because the raw slides total about `860G`

Class counts:

| class_name | slides | patients |
| --- | ---: | ---: |
| KICH | 109 | 109 |
| KIRC | 518 | 513 |
| KIRP | 299 | 275 |

Patient-grouped split, seed `2024`, ratios `60/20/20`:

| split | KICH slides | KIRC slides | KIRP slides |
| --- | ---: | ---: | ---: |
| train | 65 | 310 | 171 |
| val | 22 | 103 | 62 |
| test | 22 | 105 | 66 |

Generated metadata:

```text
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/tcga_rcc_pipeline_manifest.json
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/tcga_rcc_source_manifest.csv
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/tcga_rcc_wsi_paths.csv
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/tcga_rcc_slide_labels.csv
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/tcga_rcc_split_v1_assignment.csv
```

Expected split CSVs for features:

```text
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA_RCC_r50_split_v1_full.csv
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA_RCC_r50_split_v1_train_val.csv
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA_RCC_uni_split_v1_full.csv
/data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/TCGA_RCC_uni_split_v1_train_val.csv
```

## Patch coordinate extraction

Initial attempt used `--multiprocess_slide 8` with stitching enabled. That produced HDF5 file-locking errors and was stopped. The recovery run uses single-slide processing and disables stitching, because R50/UNI feature extraction only requires the patch coordinate H5 files.

Recovery command:

```bash
cd /data15/data15_5/fanhao/projects/MIRMIL/feature_extractor

HDF5_USE_FILE_LOCKING=FALSE mamba run -n mirmil python create_h5_patches.py \
  --source /data15/data15_5/fanhao/datasets/TCGA-RCC/raw_gdc \
  --source_csv /data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/tcga_rcc_wsi_paths.csv \
  --save_dir /data15/data15_5/fanhao/datasets/TCGA-RCC/patches_level0_256 \
  --preset tcga.csv \
  --level_or_magnification_control level \
  --patch_level 0 \
  --patch_size 256 \
  --step_size 256 \
  --seg \
  --patch \
  --no-stitch \
  --save_mask \
  --multiprocess_slide 1
```

Final patch-coordinate audit on 2026-07-14:

- process-list rows: `926`
- standard processed/already-existing slides: `923`
- sparse fallback slides: `3`
- valid patch H5 files: `926 / 926`
- corrupted patch H5 files: `0`
- total patch coordinates: `51,359,422`
- maximum patch coordinates in one slide: `390,656`

Three very large single-level SVS files failed standard CLAM segmentation because
the segmentation path refused to build a level-0 mask for slides whose
dimensions exceed the safety threshold. They were recovered with the repository
fallback coordinate script in sparse tile mode:

```bash
mamba run -n mirmil python experiments/fallback_wsi_thumbnail_patches.py \
  --method sparse \
  --sample-stride 4096 \
  --sample-size 256 \
  --wsi <failed_single_level_svs> \
  --output-h5 /data15/data15_5/fanhao/datasets/TCGA-RCC/patches_level0_256/patches/<slide_id>.h5 \
  --mask-output /data15/data15_5/fanhao/datasets/TCGA-RCC/patches_level0_256/masks/<slide_id>_sparse_fallback_mask.png \
  --patch-size 256 \
  --step-size 256 \
  --overwrite
```

Recovered fallback slides:

| slide_id | coords | fallback_method |
| --- | ---: | --- |
| TCGA-5P-A9KA-01Z-00-DX1.6F4914E0-AB5D-4D5F-8BF6-FB862AA63A87 | 355,072 | sparse_tile_tissue_mask |
| TCGA-5P-A9KC-01Z-00-DX1.F3D67C35-111C-4EE6-A5F7-05CF8D01E783 | 390,656 | sparse_tile_tissue_mask |
| TCGA-UZ-A9PQ-01Z-00-DX1.C2CB0E94-2548-4399-BCAB-E4D556D533EF | 57,088 | sparse_tile_tissue_mask |

The fallback only writes CLAM-compatible patch coordinate H5 files from existing
WSIs. It does not extract features, fine-tune encoders, alter R50/UNI, or change
the downstream MIL architecture.

## Feature extraction plan

After all patch coordinate H5 files are valid, extract R50 and UNI with the same extractor used for PANDA/BRACS.

R50:

```bash
mamba run -n mirmil python experiments/extract_panda_features.py \
  --encoder r50 \
  --source-csv /data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/tcga_rcc_wsi_paths.csv \
  --patch-dir /data15/data15_5/fanhao/datasets/TCGA-RCC/patches_level0_256 \
  --output-dir /data15/data15_5/fanhao/datasets/TCGA-RCC/features/r50 \
  --weights-dir /data15/data15_5/fanhao/projects/MIRMIL/artifacts/pretrained/resnet50_imagenet \
  --gpus 0,1,2,3,4,5,6,7 \
  --batch-size 128 \
  --num-workers 4
```

UNI:

```bash
mamba run -n mirmil python experiments/extract_panda_features.py \
  --encoder uni \
  --source-csv /data15/data15_5/fanhao/datasets/TCGA-RCC/metadata/tcga_rcc_wsi_paths.csv \
  --patch-dir /data15/data15_5/fanhao/datasets/TCGA-RCC/patches_level0_256 \
  --output-dir /data15/data15_5/fanhao/datasets/TCGA-RCC/features/uni \
  --weights-dir /data15/data15_5/fanhao/projects/MIRMIL/artifacts/pretrained/uni \
  --gpus 0,1,2,3,4,5,6,7 \
  --batch-size 64 \
  --num-workers 4
```

After extraction, rerun:

```bash
mamba run -n mirmil python experiments/prepare_tcga_rcc_pipeline.py \
  --manifest /data15/data15_5/fanhao/datasets/TCGA-RCC/manifests/tcga_rcc_primary_tumor_diagnostic_slides_audit.tsv \
  --raw-dir /data15/data15_5/fanhao/datasets/TCGA-RCC/raw_gdc \
  --wsi-root /data15/data15_5/fanhao/datasets/TCGA-RCC/WSI \
  --feature-root /data15/data15_5/fanhao/datasets/TCGA-RCC/features \
  --output-dir /data15/data15_5/fanhao/datasets/TCGA-RCC/metadata \
  --seed 2024 \
  --require-features
```
