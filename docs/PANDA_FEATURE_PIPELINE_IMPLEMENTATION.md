# PANDA Feature Pipeline Implementation

Status: full dataset extraction and audit complete, July 3, 2026.

## Purpose

This document records the feature-generation pipeline intended for the new
PANDA experiments. The source WSI directory is treated as read-only:

```text
/data15/data15_5/fanhao/datasets/PANDA/WSI/train_images/
```

Trial outputs are stored under the repository's `artifacts/` directory. Full
outputs are stored outside the repository under:

```text
/data15/data15_5/fanhao/datasets/PANDA/MIRMIL_FEATURES/
```

The source WSI directory was not modified.

## Full-Scale Completion

The frozen cohort contains 10,615 slides: 6,369 train, 2,123 validation, and
2,123 sealed test. One all-white source slide,
`3790f55cad63053e956fb73027179707`, was excluded by QC before splitting.

The completed extraction contains:

- 10,615 patch-coordinate H5 files and 5,323,426 patch coordinates;
- 10,615 R50 H5 files and 10,615 matching PT files;
- 10,615 UNI H5 files and 10,615 matching PT files.

Every feature file passed count, dimensionality, finiteness, H5/PT equality,
and coordinate-alignment checks. Both encoders use the identical coordinate
cohort. The final split manifests are in the feature root's `metadata/`
directory; `_h5.csv` variants contain the equivalent H5 paths needed by the
spatial model.

## Environment

The pipeline uses the Mamba environment `mirmil`.

Validated GPU stack:

- PyTorch `2.5.1+cu118`;
- torchvision `0.20.1+cu118`;
- NVIDIA driver `535.146.02`;
- NVIDIA GeForce RTX 2080 Ti;
- timm `1.0.27`.

The host has eight RTX 2080 Ti GPUs. Feature extraction was validated on GPU
0. GPU access may be unavailable inside an isolated execution sandbox even
when the host GPUs are healthy, so both `nvidia-smi` and
`torch.cuda.is_available()` must be checked in the actual execution context.

## Tissue Segmentation

Implementation:

```text
feature_extractor/create_h5_patches.py
```

Validated preset:

```text
feature_extractor/presets/panda_trial.csv
```

The preset contains:

| Parameter | Value |
| --- | ---: |
| `seg_level` | -1 |
| `sthresh` | 15 |
| `mthresh` | 11 |
| `close` | 2 |
| `use_otsu` | false |
| `a_t` | 1 |
| `a_h` | 1 |
| `max_n_holes` | 2 |
| `vis_level` | -1 |
| `line_thickness` | 50 |
| `white_thresh` | 5 |
| `black_thresh` | 50 |
| `use_padding` | true |
| `contour_fn` | `four_pt` |
| `keep_ids` | `none` |
| `exclude_ids` | `none` |

On the trial TIFF files, automatic level selection resolved both
`seg_level=-1` and `vis_level=-1` to level 2. All 20 masks completed
successfully and passed manual inspection.

The current patching implementation preserves `white_thresh` and
`black_thresh` in the preset CSV but does not consume them. They must not be
claimed as active filters until the patching code explicitly wires them into
coordinate screening.

Trial records:

```text
feature_extractor/panda_trial_20.csv
artifacts/panda_preset_trial_20/
```

## Patch Coordinates

The validated patch configuration is:

- patch level: 0;
- patch size: 256 x 256 pixels;
- step size: 256 pixels;
- no overlap;
- contour rule: `four_pt`;
- padding enabled;
- individual patch JPEG export disabled.

The 20-slide trial produced:

- 20 coordinate H5 files;
- 20 stitch images;
- 9,396 total patches;
- 67 minimum patches per slide;
- 455.5 median patches per slide;
- 771 maximum patches per slide.

Each coordinate file contains an `N x 2` `coords` dataset. Trial outputs:

```text
artifacts/panda_patch_trial_20/patches/
artifacts/panda_patch_trial_20/stitches/
artifacts/panda_patch_trial_20/process_list_autogen.csv
```

`adjust_coords_order()` sorts coordinates in place without changing their
values or discarding H5 attributes. The earlier unexplained one-pixel offset
was removed before the full extraction.

## ResNet-50 Features

Backbone identifier:

```text
resnet50_imagenet
```

The extractor uses the CLAM truncated ResNet-50 through `layer3`, followed by
global average pooling. Its output dimension is 1,024.

Input processing:

- WSI patch: 256 x 256 at level 0;
- resized model input: 224 x 224;
- ImageNet mean: `(0.485, 0.456, 0.406)`;
- ImageNet standard deviation: `(0.229, 0.224, 0.225)`.

Official checkpoint:

```text
artifacts/pretrained/resnet50_imagenet/resnet50-19c8e357.pth
SHA256 19c8e3572231adff6824a2da93fd67b5986919a2e65f8b6007eab4edee220097
```

The checkpoint uses legacy PyTorch serialization. It must first be loaded on
CPU with `weights_only=False`. The custom encoder omits `layer4` and `fc`, so
loading uses `strict=False` and the model is moved to the target device after
deserialization.

Validated GPU outputs:

```text
artifacts/panda_r50_features_gpu_trial_20/
```

All 20 H5 and PT pairs passed the following checks:

- shape `N x 1024`;
- finite feature values;
- exact H5/PT equality;
- exact coordinate equality with the patch H5 files.

The CPU/GPU maximum absolute feature difference was approximately
`1.92e-6`.

## UNI Features

Backbone identifier:

```text
uni
```

This identifier denotes the first-generation MahmoodLab UNI encoder,
ViT-L/16, not UNI2-h. It produces 1,024-dimensional class-token features.

Input processing is matched to the R50 experiment:

- the same WSI patches and coordinates;
- 224 x 224 model input;
- the same ImageNet normalization.

Official checkpoint:

```text
artifacts/pretrained/uni/pytorch_model.bin
SHA256 56ef09b44a25dc5c7eedc55551b3d47bcd17659a7a33837cf9abc9ec4e2ffb40
```

The gated model was downloaded from `MahmoodLab/UNI` after Hugging Face
authentication. The license and access conditions of the upstream model
apply.

Validated extraction used GPU 0 and batch size 64. Outputs:

```text
artifacts/panda_uni_features_gpu_trial_20/
```

All 20 H5 and PT pairs passed:

- shape `N x 1024`;
- finite feature values;
- exact H5/PT equality;
- exact coordinate equality with both the source patch files and R50
  feature files.

## Required Full-Scale Safeguards

Before processing the complete dataset:

1. wait for WSI extraction to finish and freeze a source manifest;
2. retain the coordinate-value and boundary regression tests;
3. make `white_thresh` and `black_thresh` effective or remove them from the
   recorded active configuration;
4. record failed slides and exception text instead of silently skipping;
5. record WSI, coordinate, checkpoint, and output hashes;
6. verify `features.shape[0] == coords.shape[0]` for every slide;
7. write outputs atomically so interrupted H5 files cannot appear complete;
8. use PT existence only after H5 validation as the resume criterion;
9. retain masks and stitches for a representative QC sample;
10. use the identical coordinate manifest for every compared encoder.
