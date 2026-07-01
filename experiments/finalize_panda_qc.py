"""Apply image-quality exclusions without reshuffling a sealed PANDA split."""

import argparse
import json
from pathlib import Path

import pandas as pd

try:
    from experiments.prepare_panda_pipeline import (
        file_sha256,
        parse_feature,
        wide_feature_split,
    )
except ModuleNotFoundError:
    from prepare_panda_pipeline import (
        file_sha256,
        parse_feature,
        wide_feature_split,
    )


def finalize_assignment(assignment, exclusions):
    required = {"image_id", "reason"}
    if not required.issubset(exclusions.columns):
        raise ValueError("exclusions must contain image_id and reason")
    if exclusions["image_id"].duplicated().any():
        raise ValueError("exclusion image_id values must be unique")
    unknown = set(exclusions["image_id"]) - set(assignment["image_id"])
    if unknown:
        raise ValueError(f"unknown exclusion image IDs: {sorted(unknown)}")
    retained = assignment[
        ~assignment["image_id"].isin(exclusions["image_id"])
    ].copy()
    return retained.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--exclusions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--feature", action="append", default=[], type=parse_feature
    )
    args = parser.parse_args()

    assignment = pd.read_csv(args.assignment)
    source = pd.read_csv(args.source_manifest)
    exclusions = pd.read_csv(args.exclusions)
    retained = finalize_assignment(assignment, exclusions)
    retained_ids = set(retained["image_id"])
    source_qc = source[source["image_id"].isin(retained_ids)].copy()
    if len(source_qc) != len(retained):
        raise ValueError("retained assignment and source manifest differ")

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    assignment_path = output / "panda_split_v1_assignment_qc.csv"
    source_path = output / "panda_source_manifest_qc.csv"
    paths_path = output / "panda_wsi_paths_qc.csv"
    retained.to_csv(assignment_path, index=False)
    source_qc.to_csv(source_path, index=False)
    source_qc[["wsi_path"]].to_csv(paths_path, index=False)

    features = {}
    for encoder, feature_dir in args.feature:
        outputs = {}
        for storage in ("pt", "h5"):
            for visible in (False, True):
                frame = wide_feature_split(
                    retained,
                    feature_dir,
                    include_test=not visible,
                    storage=storage,
                )
                suffix = "train_val" if visible else "full"
                storage_suffix = "" if storage == "pt" else "_h5"
                path = output / (
                    f"PANDA_{encoder}_split_v1_{suffix}_qc"
                    f"{storage_suffix}.csv"
                )
                frame.to_csv(path, index=False)
                outputs[f"{suffix}_{storage}"] = {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
        features[encoder] = outputs

    metadata = {
        "schema_version": 1,
        "policy": "post_split_image_quality_exclusion_no_reshuffle",
        "input_assignment": str(args.assignment.resolve()),
        "input_assignment_sha256": file_sha256(args.assignment),
        "exclusions": str(args.exclusions.resolve()),
        "exclusions_sha256": file_sha256(args.exclusions),
        "retained_assignment": str(assignment_path),
        "retained_assignment_sha256": file_sha256(assignment_path),
        "retained_source_manifest": str(source_path),
        "retained_source_manifest_sha256": file_sha256(source_path),
        "source_paths": str(paths_path),
        "source_paths_sha256": file_sha256(paths_path),
        "excluded_count": len(exclusions),
        "retained_count": len(retained),
        "counts": {
            split: int((retained["split"] == split).sum())
            for split in ("train", "val", "test")
        },
        "features": features,
    }
    manifest_path = output / "panda_qc_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
