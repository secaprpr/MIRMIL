import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_sha256(frame):
    columns = [
        "image_id",
        "data_provider",
        "isup_grade",
        "wsi_size",
    ]
    payload = frame[columns].sort_values("image_id").to_csv(
        index=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_manifest(label_csv, wsi_dir):
    labels = pd.read_csv(label_csv)
    required = {"image_id", "data_provider", "isup_grade"}
    missing_columns = required - set(labels.columns)
    if missing_columns:
        raise ValueError(
            f"Missing PANDA label columns: {sorted(missing_columns)}"
        )
    if labels["image_id"].duplicated().any():
        raise ValueError("PANDA image_id values must be unique")

    wsi_dir = Path(wsi_dir).resolve()
    records = []
    missing = []
    for row in labels.itertuples(index=False):
        path = wsi_dir / f"{row.image_id}.tiff"
        if not path.is_file():
            missing.append(row.image_id)
            continue
        records.append(
            {
                "image_id": str(row.image_id),
                "data_provider": str(row.data_provider),
                "isup_grade": int(row.isup_grade),
                "wsi_path": str(path),
                "wsi_size": path.stat().st_size,
            }
        )
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} labeled WSIs are missing; first: {missing[:5]}"
        )

    manifest = pd.DataFrame(records).sort_values("image_id").reset_index(
        drop=True
    )
    discovered = {path.stem for path in wsi_dir.glob("*.tiff")}
    unexpected = discovered - set(manifest["image_id"])
    if unexpected:
        raise ValueError(
            f"{len(unexpected)} WSI files have no label; "
            f"first: {sorted(unexpected)[:5]}"
        )
    return manifest


def stratified_assignment(
    manifest,
    seed=2024,
    train_ratio=0.6,
    val_ratio=0.2,
    test_ratio=0.2,
):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("split ratios must sum to one")
    strata = (
        manifest["data_provider"].astype(str)
        + "__"
        + manifest["isup_grade"].astype(str)
    )
    train, remainder = train_test_split(
        manifest,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=strata,
    )
    remainder_strata = (
        remainder["data_provider"].astype(str)
        + "__"
        + remainder["isup_grade"].astype(str)
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val, test = train_test_split(
        remainder,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
        stratify=remainder_strata,
    )
    pieces = []
    for split, frame in (("train", train), ("val", val), ("test", test)):
        part = frame.copy()
        part["split"] = split
        pieces.append(part)
    assignment = pd.concat(pieces, ignore_index=True)
    return assignment.sort_values(["split", "image_id"]).reset_index(
        drop=True
    )


def wide_feature_split(
    assignment, feature_dir, include_test=True, storage="pt"
):
    if storage not in {"pt", "h5"}:
        raise ValueError("storage must be pt or h5")
    feature_dir = Path(feature_dir).resolve()
    series = {}
    for split in ("train", "val", "test"):
        frame = assignment[assignment["split"] == split].copy()
        if split == "test" and not include_test:
            frame = frame.iloc[0:0]
        paths = [
            str(feature_dir / f"{storage}_files" / f"{image_id}.{storage}")
            for image_id in frame["image_id"]
        ]
        series[f"{split}_slide_path"] = pd.Series(paths, dtype="object")
        series[f"{split}_label"] = pd.Series(
            frame["isup_grade"].astype(int).tolist(),
            dtype="Int64",
        )
    return pd.DataFrame(series)


def parse_feature(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "features must use ENCODER=/path/to/feature_dir"
        )
    encoder, path = value.split("=", 1)
    if not encoder or not path:
        raise argparse.ArgumentTypeError(
            "features must use ENCODER=/path/to/feature_dir"
        )
    return encoder, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--wsi-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument(
        "--feature",
        action="append",
        default=[],
        type=parse_feature,
        help="Repeat ENCODER=/path/to/feature_dir",
    )
    parser.add_argument(
        "--require-features",
        action="store_true",
        help="Fail unless every materialized PT feature exists",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.labels, args.wsi_dir)
    assignment = stratified_assignment(
        manifest,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    manifest_path = output_dir / "panda_source_manifest.csv"
    source_csv_path = output_dir / "panda_wsi_paths.csv"
    assignment_path = output_dir / "panda_split_v1_assignment.csv"
    manifest.to_csv(manifest_path, index=False)
    manifest[["wsi_path"]].to_csv(source_csv_path, index=False)
    assignment.to_csv(assignment_path, index=False)

    feature_outputs = {}
    for encoder, feature_dir in args.feature:
        full = wide_feature_split(assignment, feature_dir, include_test=True)
        visible = wide_feature_split(assignment, feature_dir, include_test=False)
        full_h5 = wide_feature_split(
            assignment, feature_dir, include_test=True, storage="h5"
        )
        visible_h5 = wide_feature_split(
            assignment, feature_dir, include_test=False, storage="h5"
        )
        if args.require_features:
            feature_paths = [
                path
                for column in full.columns
                if column.endswith("_slide_path")
                for path in full[column].dropna()
            ]
            missing = [path for path in feature_paths if not os.path.isfile(path)]
            if missing:
                raise FileNotFoundError(
                    f"{len(missing)} {encoder} features are missing; "
                    f"first: {missing[:5]}"
                )
        full_path = output_dir / f"PANDA_{encoder}_split_v1_full.csv"
        visible_path = (
            output_dir / f"PANDA_{encoder}_split_v1_train_val.csv"
        )
        full.to_csv(full_path, index=False)
        visible.to_csv(visible_path, index=False)
        full_h5_path = output_dir / f"PANDA_{encoder}_split_v1_full_h5.csv"
        visible_h5_path = (
            output_dir / f"PANDA_{encoder}_split_v1_train_val_h5.csv"
        )
        full_h5.to_csv(full_h5_path, index=False)
        visible_h5.to_csv(visible_h5_path, index=False)
        feature_outputs[encoder] = {
            "feature_dir": str(Path(feature_dir).resolve()),
            "full_split": str(full_path),
            "full_split_sha256": file_sha256(full_path),
            "visible_split": str(visible_path),
            "visible_split_sha256": file_sha256(visible_path),
            "full_h5_split": str(full_h5_path),
            "full_h5_split_sha256": file_sha256(full_h5_path),
            "visible_h5_split": str(visible_h5_path),
            "visible_h5_split_sha256": file_sha256(visible_h5_path),
        }

    counts = assignment.groupby(
        ["split", "data_provider", "isup_grade"]
    ).size()
    metadata = {
        "schema_version": 1,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "split_unit": "image_case",
        "patient_id_available": False,
        "labels": str(Path(args.labels).resolve()),
        "labels_sha256": file_sha256(args.labels),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": manifest_sha256(manifest),
        "assignment": str(assignment_path),
        "assignment_sha256": file_sha256(assignment_path),
        "counts": {
            split: int((assignment["split"] == split).sum())
            for split in ("train", "val", "test")
        },
        "stratum_counts": {
            "__".join(map(str, key)): int(value)
            for key, value in counts.items()
        },
        "features": feature_outputs,
    }
    metadata_path = output_dir / "panda_pipeline_manifest.json"
    with open(metadata_path, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
