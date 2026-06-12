import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.cache_feature_subset import cache_feature
from experiments.prepare_split import (
    build_training_csv,
    deterministic_group_stratified_split,
)
from experiments.prepare_tcga_projects import collect_projects, parse_project


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_cached_projects(labels_path, cache_manifest_path):
    labels = pd.read_csv(labels_path)
    required = {"slide_id", "patient_id", "label", "class_name"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")

    with open(cache_manifest_path, encoding="utf-8") as file:
        manifest = json.load(file)
    path_by_slide = {
        Path(record["source_path"]).stem: record["cached_path"]
        for record in manifest["records"]
    }
    labels["slide_path"] = labels["slide_id"].map(path_by_slide)
    unmapped = labels.loc[labels["slide_path"].isna(), "slide_id"]
    if len(unmapped):
        raise ValueError(
            f"{len(unmapped)} cached paths are missing; first: "
            f"{unmapped.iloc[0]}"
        )
    missing_files = labels.loc[
        ~labels["slide_path"].map(lambda value: Path(value).is_file()),
        "slide_path",
    ]
    if len(missing_files):
        raise FileNotFoundError(
            f"{len(missing_files)} cached files are missing; first: "
            f"{missing_files.iloc[0]}"
        )
    return labels[
        ["slide_id", "patient_id", "label", "class_name", "slide_path"]
    ].copy()


def cache_additional_projects(
    projects,
    cache_dir,
    label_offset,
    max_candidates,
):
    frame = collect_projects(projects)
    frame["label"] += label_offset
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_paths = []
    cache_records = []
    for row in frame.itertuples(index=False):
        destination, instances, dimensions, created = cache_feature(
            row.source_path,
            str(cache_dir),
            max_candidates,
        )
        cached_paths.append(str(Path(destination).resolve()))
        cache_records.append(
            {
                "slide_id": row.slide_id,
                "source_path": row.source_path,
                "cached_path": str(Path(destination).resolve()),
                "instances": instances,
                "dimensions": dimensions,
                "created": created,
            }
        )
    frame["slide_path"] = cached_paths
    return (
        frame[
            ["slide_id", "patient_id", "label", "class_name", "slide_path"]
        ].copy(),
        cache_records,
    )


def write_visible_split(full_split_path, visible_split_path):
    visible = pd.read_csv(full_split_path)
    visible["test_slide_path"] = pd.NA
    visible["test_label"] = pd.NA
    visible.to_csv(visible_split_path, index=False)


def validate_combined_frame(frame):
    if frame["slide_id"].duplicated().any():
        duplicate = frame.loc[frame["slide_id"].duplicated(), "slide_id"].iloc[0]
        raise ValueError(f"Duplicate slide ID: {duplicate}")
    labels_per_patient = frame.groupby("patient_id")["label"].nunique()
    if (labels_per_patient > 1).any():
        patient = labels_per_patient[labels_per_patient > 1].index[0]
        raise ValueError(f"Patient occurs in multiple classes: {patient}")
    class_labels = frame.groupby("class_name")["label"].nunique()
    if (class_labels > 1).any():
        class_name = class_labels[class_labels > 1].index[0]
        raise ValueError(f"Class name has multiple labels: {class_name}")
    if sorted(frame["label"].unique()) != list(range(frame["label"].nunique())):
        raise ValueError("Labels must be contiguous and start at zero")


def split_counts(assignments):
    counts = {}
    for (split, class_name), group in assignments.groupby(
        ["split", "class_name"], sort=True
    ):
        counts.setdefault(split, {})[class_name] = {
            "slides": len(group),
            "patients": int(group["patient_id"].nunique()),
        }
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-labels", required=True)
    parser.add_argument("--base-cache-manifest", required=True)
    parser.add_argument(
        "--project",
        action="append",
        type=parse_project,
        required=True,
        help="Repeat CLASS_NAME=/path/to/features in desired label order",
    )
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--max-candidates", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    args = parser.parse_args()

    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train and validation ratios must be positive")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train and validation ratios must sum to less than 1")

    base = load_cached_projects(
        args.base_labels,
        args.base_cache_manifest,
    )
    label_offset = int(base["label"].max()) + 1
    additional, cache_records = cache_additional_projects(
        args.project,
        Path(args.cache_dir),
        label_offset,
        args.max_candidates,
    )
    combined = pd.concat([base, additional], ignore_index=True)
    combined["label"] = combined["label"].astype(int)
    validate_combined_frame(combined)

    assignments = deterministic_group_stratified_split(
        combined.rename(columns={"patient_id": "group_id"}),
        args.seed,
        args.train_ratio,
        args.val_ratio,
    ).rename(columns={"group_id": "patient_id"})
    leakage = assignments.groupby("patient_id")["split"].nunique()
    if (leakage > 1).any():
        raise RuntimeError("Patient leakage detected after split assignment")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = output_dir / args.dataset_name
    assignments_path = Path(f"{prefix}_assignments.csv")
    full_split_path = Path(f"{prefix}_sealed_split.csv")
    visible_split_path = Path(f"{prefix}_train_val.csv")
    manifest_path = Path(f"{prefix}_manifest.json")

    assignments.to_csv(assignments_path, index=False)
    build_training_csv(assignments, full_split_path)
    write_visible_split(full_split_path, visible_split_path)

    class_mapping = {
        row.class_name: int(row.label)
        for row in combined[["class_name", "label"]]
        .drop_duplicates()
        .sort_values("label")
        .itertuples(index=False)
    }
    manifest = {
        "dataset": args.dataset_name,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "base_labels": str(Path(args.base_labels).resolve()),
        "base_labels_sha256": file_sha256(args.base_labels),
        "base_cache_manifest": str(
            Path(args.base_cache_manifest).resolve()
        ),
        "base_cache_manifest_sha256": file_sha256(
            args.base_cache_manifest
        ),
        "class_mapping": class_mapping,
        "num_slides": len(assignments),
        "num_patients": int(assignments["patient_id"].nunique()),
        "counts": split_counts(assignments),
        "assignments": str(assignments_path.resolve()),
        "assignments_sha256": file_sha256(assignments_path),
        "sealed_split": str(full_split_path.resolve()),
        "sealed_split_sha256": file_sha256(full_split_path),
        "visible_train_val_split": str(visible_split_path.resolve()),
        "visible_train_val_split_sha256": file_sha256(visible_split_path),
        "new_cache_records": cache_records,
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    summary = {
        key: value
        for key, value in manifest.items()
        if key != "new_cache_records"
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
