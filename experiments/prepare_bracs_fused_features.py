"""Create lightweight R50+UNI HDF5 virtual feature datasets for BRACS."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import h5py
import numpy as np
import pandas as pd


SPLITS = ("train", "val", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def paired_records(r50_split, uni_split):
    r50 = pd.read_csv(r50_split)
    uni = pd.read_csv(uni_split)
    records = []
    for split in SPLITS:
        r50_paths = r50[f"{split}_slide_path"].dropna().tolist()
        uni_paths = uni[f"{split}_slide_path"].dropna().tolist()
        r50_labels = r50[f"{split}_label"].dropna().astype(int).tolist()
        uni_labels = uni[f"{split}_label"].dropna().astype(int).tolist()
        if r50_labels != uni_labels:
            raise ValueError(f"{split}: R50/UNI labels do not match")
        if len(r50_paths) != len(uni_paths):
            raise ValueError(f"{split}: R50/UNI path counts do not match")
        for r50_path, uni_path, label in zip(
            r50_paths, uni_paths, r50_labels
        ):
            r50_path = Path(r50_path).resolve()
            uni_path = Path(uni_path).resolve()
            if r50_path.stem != uni_path.stem:
                raise ValueError(
                    f"{split}: slide mismatch {r50_path} != {uni_path}"
                )
            records.append(
                {
                    "split": split,
                    "slide_id": r50_path.stem,
                    "label": label,
                    "r50_path": r50_path,
                    "uni_path": uni_path,
                }
            )
    return records


def create_virtual_feature(record, output_path):
    with h5py.File(record["r50_path"], "r") as r50, h5py.File(
        record["uni_path"], "r"
    ) as uni:
        r50_features = r50["features"]
        uni_features = uni["features"]
        if r50_features.shape[0] != uni_features.shape[0]:
            raise ValueError(
                f"{record['slide_id']}: patch count mismatch "
                f"{r50_features.shape[0]} != {uni_features.shape[0]}"
            )
        if not np.array_equal(r50["coords"][:], uni["coords"][:]):
            raise ValueError(
                f"{record['slide_id']}: R50/UNI coordinates differ"
            )
        rows = r50_features.shape[0]
        r50_dim = r50_features.shape[1]
        uni_dim = uni_features.shape[1]
        dtype = np.result_type(r50_features.dtype, uni_features.dtype)

    layout = h5py.VirtualLayout(
        shape=(rows, r50_dim + uni_dim), dtype=dtype
    )
    layout[:, :r50_dim] = h5py.VirtualSource(
        str(record["r50_path"]), "features", shape=(rows, r50_dim)
    )
    layout[:, r50_dim:] = h5py.VirtualSource(
        str(record["uni_path"]), "features", shape=(rows, uni_dim)
    )
    coordinate_layout = h5py.VirtualLayout(shape=(rows, 2), dtype=np.int64)
    coordinate_layout[:] = h5py.VirtualSource(
        str(record["r50_path"]), "coords", shape=(rows, 2)
    )
    with h5py.File(output_path, "w", libver="latest") as fused:
        fused.create_virtual_dataset("features", layout)
        fused.create_virtual_dataset("coords", coordinate_layout)
        fused.attrs["r50_source"] = str(record["r50_path"])
        fused.attrs["uni_source"] = str(record["uni_path"])

    return {
        "slide_id": record["slide_id"],
        "split": record["split"],
        "label": record["label"],
        "num_patches": rows,
        "r50_dim": r50_dim,
        "uni_dim": uni_dim,
        "fused_dim": r50_dim + uni_dim,
        "r50_path": str(record["r50_path"]),
        "uni_path": str(record["uni_path"]),
        "fused_path": str(output_path.resolve()),
    }


def create_record(arguments):
    record, feature_dir = arguments
    return create_virtual_feature(
        record, feature_dir / f"{record['slide_id']}.h5"
    )


def wide_split(manifest, include_test):
    columns = {}
    for split in SPLITS:
        frame = manifest[manifest["split"] == split]
        if split == "test" and not include_test:
            frame = frame.iloc[:0]
        columns[f"{split}_slide_path"] = pd.Series(
            frame["fused_path"].tolist(), dtype="object"
        )
        columns[f"{split}_label"] = pd.Series(
            frame["label"].tolist(), dtype="Int64"
        )
    return pd.DataFrame(columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r50-split", type=Path, required=True)
    parser.add_argument("--uni-split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = args.output_dir / "h5_files"
    feature_dir.mkdir(exist_ok=True)

    records = paired_records(args.r50_split, args.uni_split)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        rows = list(
            executor.map(
                create_record,
                ((record, feature_dir) for record in records),
            )
        )
    manifest = pd.DataFrame(rows)
    manifest_path = args.output_dir / "bracs_r50_uni_fusion_manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    split_paths = {}
    for include_test, suffix in ((True, "full"), (False, "train_val")):
        path = args.output_dir / (
            f"BRACS_r50_uni_split_official_{suffix}_h5.csv"
        )
        wide_split(manifest, include_test).to_csv(path, index=False)
        split_paths[suffix] = {
            "path": str(path.resolve()),
            "sha256": file_sha256(path),
        }
    metadata = {
        "schema_version": 1,
        "dataset": "BRACS",
        "feature": "r50_uni_concat",
        "in_dim": int(manifest["fused_dim"].unique().item()),
        "counts": {
            split: int((manifest["split"] == split).sum())
            for split in SPLITS
        },
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "splits": split_paths,
    }
    metadata_path = args.output_dir / "bracs_r50_uni_fusion.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
