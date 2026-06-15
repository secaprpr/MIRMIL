import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


TCGA_CASE_PATTERN = re.compile(r"(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def infer_group_id(slide_path):
    match = TCGA_CASE_PATTERN.search(Path(slide_path).name.upper())
    if match:
        return match.group(1)
    return Path(slide_path).stem


def load_visible_pool(split_path):
    frame = pd.read_csv(split_path)
    required = {
        "train_slide_path",
        "train_label",
        "val_slide_path",
        "val_label",
        "test_slide_path",
        "test_label",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing split columns: {sorted(missing)}")
    if frame[["test_slide_path", "test_label"]].notna().any().any():
        raise ValueError("Visible CV input must not contain test examples")

    parts = []
    for split in ("train", "val"):
        part = frame[
            [f"{split}_slide_path", f"{split}_label"]
        ].dropna()
        part.columns = ["slide_path", "label"]
        parts.append(part)
    pool = pd.concat(parts, ignore_index=True)
    pool["label"] = pool["label"].astype(int)
    pool["group_id"] = pool["slide_path"].map(infer_group_id)

    if pool["slide_path"].duplicated().any():
        duplicate = pool.loc[pool["slide_path"].duplicated(), "slide_path"].iloc[0]
        raise ValueError(f"Duplicate slide path: {duplicate}")
    labels_per_group = pool.groupby("group_id")["label"].nunique()
    if (labels_per_group > 1).any():
        group = labels_per_group[labels_per_group > 1].index[0]
        raise ValueError(f"Group occurs in multiple classes: {group}")
    return pool


def build_wide_split(train, val):
    columns = {}
    for split, part in (("train", train), ("val", val)):
        part = part.reset_index(drop=True)
        columns[f"{split}_slide_path"] = part["slide_path"]
        columns[f"{split}_label"] = part["label"]
    columns["test_slide_path"] = pd.Series(dtype="object")
    columns["test_label"] = pd.Series(dtype="float64")
    return pd.DataFrame(columns)


def class_counts(frame):
    return {
        str(int(label)): int(count)
        for label, count in frame["label"].value_counts().sort_index().items()
    }


def create_visible_folds(pool, n_splits, seed):
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    folds = []
    for fold, (train_indices, val_indices) in enumerate(
        splitter.split(pool, pool["label"], pool["group_id"])
    ):
        train = pool.iloc[train_indices].copy()
        val = pool.iloc[val_indices].copy()
        if set(train["group_id"]).intersection(val["group_id"]):
            raise RuntimeError(f"Group leakage detected in fold {fold}")
        folds.append((train, val))
    return folds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    if args.folds < 2:
        raise ValueError("--folds must be at least 2")

    pool = load_visible_pool(args.split)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for fold, (train, val) in enumerate(
        create_visible_folds(pool, args.folds, args.seed)
    ):
        path = output_dir / f"{args.dataset_name}_fold{fold}.csv"
        build_wide_split(train, val).to_csv(path, index=False)
        records.append(
            {
                "fold": fold,
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
                "train_slides": len(train),
                "train_patients": int(train["group_id"].nunique()),
                "train_class_counts": class_counts(train),
                "val_slides": len(val),
                "val_patients": int(val["group_id"].nunique()),
                "val_class_counts": class_counts(val),
            }
        )

    manifest = {
        "dataset": args.dataset_name,
        "source_split": str(Path(args.split).resolve()),
        "source_split_sha256": file_sha256(args.split),
        "seed": args.seed,
        "folds": args.folds,
        "pool_slides": len(pool),
        "pool_patients": int(pool["group_id"].nunique()),
        "pool_class_counts": class_counts(pool),
        "records": records,
    }
    manifest_path = output_dir / f"{args.dataset_name}_cv_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
