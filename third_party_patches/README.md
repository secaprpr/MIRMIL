# Third-party baseline patches

`c2aug_panda.patch` applies to the ignored checkout at `baselines/c2aug`,
upstream commit `14cb53bc71f5161105b0dbef45351243d7330561`.

It adds PANDA's explicit train/validation/test split, 1,024-dimensional R50
and UNI inputs, six-class outputs, a shared 4,096-instance budget, configurable
runtime paths, and W&B logging without checkpoint upload. The C2Aug
augmentation, attention fusion, DINO group loss, teacher EMA, and TransMIL
backbone remain upstream implementations.
Validation macro-AUC controls checkpoint selection and early stopping.

Apply to a clean checkout with:

```bash
git -C baselines/c2aug apply --unidiff-zero \
  ../../third_party_patches/c2aug_panda.patch
```
