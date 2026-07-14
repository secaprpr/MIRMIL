"""Generate CLAM-compatible patch coordinates from a downsampled thumbnail.

This is a conservative fallback for very large single-level WSIs where the
standard CLAM tissue segmentation path refuses to build a level-0 mask. It
does not extract image features. It only writes an HDF5 file containing a
`coords` dataset, compatible with `feature_extractor/create_pt_features.py`.
"""

import argparse
from pathlib import Path

import h5py
import numpy as np
import openslide
from PIL import Image


def thumbnail_tissue_mask(slide, max_dim, saturation_threshold, value_threshold):
    width, height = slide.dimensions
    scale = min(float(max_dim) / max(width, height), 1.0)
    thumb_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    thumbnail = slide.get_thumbnail(thumb_size).convert("RGB")
    rgb = np.asarray(thumbnail).astype(np.float32) / 255.0
    max_channel = rgb.max(axis=2)
    min_channel = rgb.min(axis=2)
    saturation = (max_channel - min_channel) / np.clip(max_channel, 1e-6, None)
    mask = (saturation >= saturation_threshold) & (max_channel <= value_threshold)
    return mask, thumbnail, scale


def block_fraction(mask, x, y, radius):
    y0 = np.maximum(y - radius, 0)
    y1 = np.minimum(y + radius + 1, mask.shape[0])
    x0 = np.maximum(x - radius, 0)
    x1 = np.minimum(x + radius + 1, mask.shape[1])
    fractions = np.empty(x.shape[0], dtype=np.float32)
    for index in range(x.shape[0]):
        block = mask[y0[index] : y1[index], x0[index] : x1[index]]
        fractions[index] = float(block.mean()) if block.size else 0.0
    return fractions


def build_coords(
    slide,
    patch_size,
    step_size,
    max_dim,
    saturation_threshold,
    value_threshold,
    min_tissue_fraction,
    block_radius,
):
    width, height = slide.dimensions
    mask, thumbnail, scale = thumbnail_tissue_mask(
        slide, max_dim, saturation_threshold, value_threshold
    )
    xs = np.arange(0, max(width - patch_size + 1, 1), step_size, dtype=np.int64)
    ys = np.arange(0, max(height - patch_size + 1, 1), step_size, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    centers_x = np.minimum(grid_x + patch_size // 2, width - 1)
    centers_y = np.minimum(grid_y + patch_size // 2, height - 1)
    thumb_x = np.clip((centers_x * scale).astype(np.int64), 0, mask.shape[1] - 1)
    thumb_y = np.clip((centers_y * scale).astype(np.int64), 0, mask.shape[0] - 1)
    flat_x = thumb_x.reshape(-1)
    flat_y = thumb_y.reshape(-1)
    center_keep = mask[flat_y, flat_x]
    candidate_indices = np.flatnonzero(center_keep)
    if candidate_indices.size == 0:
        return np.empty((0, 2), dtype=np.int64), thumbnail, mask, scale
    fractions = block_fraction(
        mask,
        flat_x[candidate_indices],
        flat_y[candidate_indices],
        radius=block_radius,
    )
    keep_indices = candidate_indices[fractions >= min_tissue_fraction]
    coords = np.stack(
        (grid_x.reshape(-1)[keep_indices], grid_y.reshape(-1)[keep_indices]),
        axis=1,
    ).astype(np.int64)
    order = np.lexsort((coords[:, 1], coords[:, 0])) if len(coords) else []
    return coords[order], thumbnail, mask, scale


def sparse_tissue_mask(
    slide,
    sample_stride,
    sample_size,
    saturation_threshold,
    value_threshold,
    min_tissue_fraction,
):
    width, height = slide.dimensions
    xs = np.arange(sample_stride // 2, width, sample_stride, dtype=np.int64)
    ys = np.arange(sample_stride // 2, height, sample_stride, dtype=np.int64)
    if xs.size == 0:
        xs = np.array([width // 2], dtype=np.int64)
    if ys.size == 0:
        ys = np.array([height // 2], dtype=np.int64)

    mask = np.zeros((ys.size, xs.size), dtype=bool)
    half = sample_size // 2
    for row, y in enumerate(ys):
        for col, x in enumerate(xs):
            x0 = int(np.clip(x - half, 0, max(width - sample_size, 0)))
            y0 = int(np.clip(y - half, 0, max(height - sample_size, 0)))
            region = slide.read_region((x0, y0), 0, (sample_size, sample_size)).convert("RGB")
            rgb = np.asarray(region).astype(np.float32) / 255.0
            max_channel = rgb.max(axis=2)
            min_channel = rgb.min(axis=2)
            saturation = (max_channel - min_channel) / np.clip(max_channel, 1e-6, None)
            tissue = (saturation >= saturation_threshold) & (max_channel <= value_threshold)
            mask[row, col] = float(tissue.mean()) >= min_tissue_fraction
    return mask, xs, ys


def build_coords_sparse(
    slide,
    patch_size,
    step_size,
    sample_stride,
    sample_size,
    saturation_threshold,
    value_threshold,
    min_tissue_fraction,
):
    width, height = slide.dimensions
    mask, sample_xs, sample_ys = sparse_tissue_mask(
        slide,
        sample_stride=sample_stride,
        sample_size=sample_size,
        saturation_threshold=saturation_threshold,
        value_threshold=value_threshold,
        min_tissue_fraction=min_tissue_fraction,
    )
    xs = np.arange(0, max(width - patch_size + 1, 1), step_size, dtype=np.int64)
    ys = np.arange(0, max(height - patch_size + 1, 1), step_size, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    centers_x = np.minimum(grid_x + patch_size // 2, width - 1)
    centers_y = np.minimum(grid_y + patch_size // 2, height - 1)

    col = np.clip(
        np.searchsorted(sample_xs, centers_x.reshape(-1), side="left"),
        0,
        sample_xs.size - 1,
    )
    row = np.clip(
        np.searchsorted(sample_ys, centers_y.reshape(-1), side="left"),
        0,
        sample_ys.size - 1,
    )
    if sample_xs.size > 1:
        left_col = np.maximum(col - 1, 0)
        use_left = np.abs(centers_x.reshape(-1) - sample_xs[left_col]) < np.abs(
            centers_x.reshape(-1) - sample_xs[col]
        )
        col = np.where(use_left, left_col, col)
    if sample_ys.size > 1:
        upper_row = np.maximum(row - 1, 0)
        use_upper = np.abs(centers_y.reshape(-1) - sample_ys[upper_row]) < np.abs(
            centers_y.reshape(-1) - sample_ys[row]
        )
        row = np.where(use_upper, upper_row, row)

    keep = mask[row, col]
    coords = np.stack((grid_x.reshape(-1)[keep], grid_y.reshape(-1)[keep]), axis=1).astype(
        np.int64
    )
    order = np.lexsort((coords[:, 1], coords[:, 0])) if len(coords) else []
    return coords[order], mask, sample_xs, sample_ys


def write_h5(path, coords, patch_size, step_size, source_wsi, scale, fallback_method, extra_attrs=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    with h5py.File(tmp_path, "w") as handle:
        coords_dataset = handle.create_dataset("coords", data=coords, dtype="int64")
        coords_dataset.attrs["patch_level"] = int(0)
        coords_dataset.attrs["patch_size"] = int(patch_size)
        coords_dataset.attrs["real_patch_size"] = int(patch_size)
        coords_dataset.attrs["step_size"] = int(step_size)
        handle.attrs["patch_size"] = int(patch_size)
        handle.attrs["step_size"] = int(step_size)
        handle.attrs["patch_level"] = 0
        handle.attrs["source_wsi"] = str(source_wsi)
        handle.attrs["thumbnail_scale"] = float(scale)
        handle.attrs["fallback_method"] = fallback_method
        if extra_attrs:
            for key, value in extra_attrs.items():
                handle.attrs[key] = value
    tmp_path.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wsi", type=Path, required=True)
    parser.add_argument("--output-h5", type=Path, required=True)
    parser.add_argument("--mask-output", type=Path, default=None)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--step-size", type=int, default=256)
    parser.add_argument("--thumbnail-max-dim", type=int, default=4096)
    parser.add_argument("--method", choices=["thumbnail", "sparse"], default="thumbnail")
    parser.add_argument("--sample-stride", type=int, default=2048)
    parser.add_argument("--sample-size", type=int, default=256)
    parser.add_argument("--saturation-threshold", type=float, default=0.05)
    parser.add_argument("--value-threshold", type=float, default=0.92)
    parser.add_argument("--min-tissue-fraction", type=float, default=0.20)
    parser.add_argument("--block-radius", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_h5.exists() and not args.overwrite:
        raise FileExistsError(args.output_h5)
    slide = openslide.open_slide(str(args.wsi))
    extra_attrs = {}
    if args.method == "thumbnail":
        coords, thumbnail, mask, scale = build_coords(
            slide,
            patch_size=args.patch_size,
            step_size=args.step_size,
            max_dim=args.thumbnail_max_dim,
            saturation_threshold=args.saturation_threshold,
            value_threshold=args.value_threshold,
            min_tissue_fraction=args.min_tissue_fraction,
            block_radius=args.block_radius,
        )
        fallback_method = "thumbnail_tissue_mask"
    else:
        coords, mask, sample_xs, sample_ys = build_coords_sparse(
            slide,
            patch_size=args.patch_size,
            step_size=args.step_size,
            sample_stride=args.sample_stride,
            sample_size=args.sample_size,
            saturation_threshold=args.saturation_threshold,
            value_threshold=args.value_threshold,
            min_tissue_fraction=args.min_tissue_fraction,
        )
        thumbnail = None
        scale = 1.0 / float(args.sample_stride)
        fallback_method = "sparse_tile_tissue_mask"
        extra_attrs = {
            "sample_stride": int(args.sample_stride),
            "sample_size": int(args.sample_size),
            "sample_grid_width": int(sample_xs.size),
            "sample_grid_height": int(sample_ys.size),
        }
    if coords.shape[0] == 0:
        raise RuntimeError(f"No tissue coordinates generated for {args.wsi}")
    write_h5(
        args.output_h5,
        coords,
        patch_size=args.patch_size,
        step_size=args.step_size,
        source_wsi=args.wsi.resolve(),
        scale=scale,
        fallback_method=fallback_method,
        extra_attrs=extra_attrs,
    )
    if args.mask_output is not None:
        args.mask_output.parent.mkdir(parents=True, exist_ok=True)
        mask_image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        mask_image.save(args.mask_output)
    print(
        f"wsi={args.wsi} coords={coords.shape[0]} output={args.output_h5} "
        f"fallback_method={fallback_method} thumbnail_scale={scale:.8f}"
    )


if __name__ == "__main__":
    main()
