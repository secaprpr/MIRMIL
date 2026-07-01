"""Validate PANDA patch-coordinate H5 files and write a provenance manifest."""

import argparse
import hashlib
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_coordinate_file(path):
    result = {
        "status": "ok",
        "detail": "",
        "num_patches": None,
        "patch_size": None,
        "patch_level": None,
        "sha256": None,
    }
    try:
        with h5py.File(path, "r") as handle:
            if "coords" not in handle:
                raise ValueError("missing coords dataset")
            dataset = handle["coords"]
            coords = dataset[:]
            if coords.ndim != 2 or coords.shape[1] != 2:
                raise ValueError(f"invalid coordinate shape {coords.shape}")
            if coords.shape[0] == 0:
                raise ValueError("empty coordinate dataset")
            if not np.issubdtype(coords.dtype, np.integer):
                raise ValueError(f"non-integer coordinate dtype {coords.dtype}")
            if np.any(coords < 0):
                raise ValueError("negative coordinate")
            if len(np.unique(coords, axis=0)) != len(coords):
                raise ValueError("duplicate coordinate")
            order = np.lexsort((coords[:, 1], coords[:, 0]))
            if not np.array_equal(order, np.arange(len(coords))):
                raise ValueError("coordinates are not deterministically sorted")
            result["num_patches"] = int(coords.shape[0])
            result["patch_size"] = int(dataset.attrs.get("patch_size", -1))
            result["patch_level"] = int(dataset.attrs.get("patch_level", -1))
            if result["patch_size"] != 256 or result["patch_level"] != 0:
                raise ValueError(
                    f"unexpected patch metadata size={result['patch_size']} "
                    f"level={result['patch_level']}"
                )
        result["sha256"] = file_sha256(path)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        result["status"] = "failed"
        result["detail"] = str(exc)
    return result


def audit_coordinates(source_csv, patch_dir):
    source = pd.read_csv(source_csv)
    if "wsi_path" not in source:
        raise ValueError("source CSV must contain wsi_path")
    records = []
    for wsi_path in source["wsi_path"]:
        slide_id = Path(wsi_path).stem
        coordinate_path = patch_dir / "patches" / f"{slide_id}.h5"
        result = inspect_coordinate_file(coordinate_path)
        records.append(
            {
                "slide_id": slide_id,
                "wsi_path": str(Path(wsi_path).resolve()),
                "coordinate_path": str(coordinate_path.resolve()),
                **result,
            }
        )
    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    audit = audit_coordinates(
        args.source_csv.resolve(), args.patch_dir.resolve()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.output, index=False)
    failed = audit[audit["status"] != "ok"]
    print(
        f"audited={len(audit)} ok={len(audit) - len(failed)} "
        f"failed={len(failed)} patches={audit['num_patches'].sum(skipna=True):.0f}"
    )
    print(f"manifest={args.output.resolve()}")
    if len(failed):
        print(failed[["slide_id", "detail"]].head(20).to_string(index=False))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
