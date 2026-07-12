"""Prepare TCGA-RCC WSI metadata and feature split CSVs.

This script mirrors the PANDA/BRACS preparation style for a local GDC
download. It does not copy WSI files; it validates the GDC manifest against
the raw download, creates class-organized symlinks, writes source CSVs for
the CLAM feature extractor, and materializes patient-grouped split CSVs for
R50/UNI feature directories.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


LABEL_ORDER = ("KICH", "KIRC", "KIRP")
PROJECT_TO_CLASS = {
    "TCGA-KICH": "KICH",
    "TCGA-KIRC": "KIRC",
    "TCGA-KIRP": "KIRP",
}
TCGA_PATIENT_PATTERN = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024, ), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataframe_sha256(frame, columns):
    payload = frame[columns].sort_values(columns).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def patient_id_from_slide(slide_id):
    match = TCGA_PATIENT_PATTERN.match(str(slide_id).upper())
    if not match:
        raise ValueError(f"Cannot parse TCGA patient ID from {slide_id}")
    return match.group(1)


def read_manifest(path):
    frame = pd.read_csv(path, sep="\t", dtype=str)
    required = {
        "id",
        "filename",
        "md5",
        "size",
        "state",
        "project_id",
        "case_submitter_id",
        "data_format",
        "experimental_strategy",
        "sample_type",
        "slide_submitter_id",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["size"] = frame["size"].astype(int)
    if frame["id"].duplicated().any():
        raise ValueError("GDC file IDs must be unique")
    if frame["filename"].duplicated().any():
        raise ValueError("GDC filenames must be unique")
    unknown_projects = set(frame["project_id"]) - set(PROJECT_TO_CLASS)
    if unknown_projects:
        raise ValueError(f"Unexpected RCC projects: {sorted(unknown_projects)}")
    bad_format = frame[frame["data_format"].str.upper() != "SVS"]
    if len(bad_format):
        raise ValueError(f"Non-SVS rows found: {len(bad_format)}")
    bad_strategy = frame[
        frame["experimental_strategy"].str.lower() != "diagnostic slide"
    ]
    if len(bad_strategy):
        raise ValueError(f"Non-diagnostic slide rows found: {len(bad_strategy)}")
    bad_sample = frame[frame["sample_type"].str.lower() != "primary tumor"]
    if len(bad_sample):
        raise ValueError(f"Non-primary-tumor rows found: {len(bad_sample)}")
    frame["class_name"] = frame["project_id"].map(PROJECT_TO_CLASS)
    frame["label"] = frame["class_name"].map(
        {name: index for index, name in enumerate(LABEL_ORDER)}
    ).astype(int)
    frame["slide_id"] = frame["filename"].str.replace(
        r"\.svs$", "", case=False, regex=True
    )
    frame["patient_id"] = frame["case_submitter_id"].fillna("")
    missing_patient = frame["patient_id"].eq("")
    if missing_patient.any():
        frame.loc[missing_patient, "patient_id"] = frame.loc[
            missing_patient, "slide_id"
        ].map(patient_id_from_slide)
    return frame


def discover_raw_slides(raw_dir):
    raw_dir = Path(raw_dir).resolve()
    rows = []
    for path in sorted(raw_dir.rglob("*.svs")):
        rows.append(
            {
                "filename": path.name,
                "wsi_path": str(path.resolve()),
                "actual_size": path.stat().st_size,
                "gdc_dir": path.parent.name,
            }
        )
    if not rows:
        raise FileNotFoundError(f"No .svs files found under {raw_dir}")
    discovered = pd.DataFrame(rows)
    if discovered["filename"].duplicated().any():
        duplicated = discovered[
            discovered["filename"].duplicated(keep=False)
        ]["filename"].tolist()
        raise ValueError(f"Duplicate WSI filenames: {duplicated[:5]}")
    return discovered


def build_source_manifest(manifest, raw_dir):
    discovered = discover_raw_slides(raw_dir)
    merged = manifest.merge(discovered, on="filename", how="left")
    missing = merged[merged["wsi_path"].isna()]
    if len(missing):
        raise FileNotFoundError(
            f"{len(missing)} manifest slides missing from raw_gdc; "
            f"first={missing['filename'].head().tolist()}"
        )
    extra = sorted(set(discovered["filename"]) - set(manifest["filename"]))
    if extra:
        raise ValueError(
            f"{len(extra)} raw_gdc SVS files are not in manifest; first={extra[:5]}"
        )
    merged["size_match"] = merged["size"].astype(int).eq(
        merged["actual_size"].astype(int)
    )
    bad_size = merged[~merged["size_match"]]
    if len(bad_size):
        raise ValueError(
            f"{len(bad_size)} slides have size mismatch; "
            f"first={bad_size[['filename', 'size', 'actual_size']].head().to_dict('records')}"
        )
    if merged.groupby("patient_id")["label"].nunique().max() > 1:
        raise ValueError("At least one patient appears in multiple RCC classes")
    columns = [
        "id",
        "filename",
        "slide_id",
        "slide_submitter_id",
        "patient_id",
        "project_id",
        "class_name",
        "label",
        "md5",
        "size",
        "actual_size",
        "size_match",
        "wsi_path",
        "gdc_dir",
        "state",
        "data_format",
        "experimental_strategy",
        "sample_type",
    ]
    return merged[columns].sort_values("slide_id").reset_index(drop=True)


def materialize_symlinks(source_manifest, wsi_root):
    wsi_root = Path(wsi_root).resolve()
    records = []
    for row in source_manifest.itertuples(index=False):
        class_dir = wsi_root / row.class_name
        class_dir.mkdir(parents=True, exist_ok=True)
        destination = class_dir / row.filename
        source = Path(row.wsi_path).resolve()
        if destination.is_symlink():
            if destination.resolve() != source:
                raise ValueError(f"Conflicting symlink: {destination}")
        elif destination.exists():
            raise FileExistsError(destination)
        else:
            destination.symlink_to(source)
        records.append(
            {
                "slide_id": row.slide_id,
                "filename": row.filename,
                "class_name": row.class_name,
                "wsi_path": str(destination.resolve()),
                "wsi_link": str(destination),
            }
        )
    return pd.DataFrame(records).sort_values("slide_id").reset_index(drop=True)


def grouped_stratified_split(source_manifest, seed, train_ratio, val_ratio, test_ratio):
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to one")
    patient_frame = (
        source_manifest[["patient_id", "class_name", "label"]]
        .drop_duplicates()
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    train_patients, remainder = train_test_split(
        patient_frame,
        train_size=train_ratio,
        random_state=seed,
        shuffle=True,
        stratify=patient_frame["label"],
    )
    relative_test = test_ratio / (val_ratio + test_ratio)
    val_patients, test_patients = train_test_split(
        remainder,
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
        stratify=remainder["label"],
    )
    split_rows = []
    for split, frame in (
        ("train", train_patients),
        ("val", val_patients),
        ("test", test_patients),
    ):
        part = frame.copy()
        part["split"] = split
        split_rows.append(part)
    patient_split = pd.concat(split_rows, ignore_index=True)
    assignment = source_manifest.merge(
        patient_split[["patient_id", "split"]],
        on="patient_id",
        how="left",
        validate="many_to_one",
    )
    if assignment["split"].isna().any():
        raise ValueError("Some slides were not assigned to a split")
    return assignment.sort_values(["split", "slide_id"]).reset_index(drop=True)


def wide_feature_split(assignment, feature_dir, storage, include_test):
    feature_dir = Path(feature_dir).resolve()
    columns = {}
    for split in ("train", "val", "test"):
        frame = assignment[assignment["split"] == split]
        if split == "test" and not include_test:
            frame = frame.iloc[:0]
        columns[f"{split}_slide_path"] = pd.Series(
            [
                str(feature_dir / f"{storage}_files" / f"{row.slide_id}.{storage}")
                for row in frame.itertuples(index=False)
            ],
            dtype="object",
        )
        columns[f"{split}_label"] = pd.Series(
            frame["label"].astype(int).tolist(), dtype="Int64"
        )
    return pd.DataFrame(columns)


def write_feature_splits(assignment, feature_root, output_dir, require_features):
    feature_root = Path(feature_root).resolve()
    output_dir = Path(output_dir).resolve()
    metadata = {}
    for feature in ("r50", "uni"):
        metadata[feature] = {}
        feature_dir = feature_root / feature
        for storage in ("pt", "h5"):
            for include_test, suffix in (
                (True, "full"),
                (False, "train_val"),
            ):
                frame = wide_feature_split(
                    assignment, feature_dir, storage, include_test
                )
                storage_suffix = "" if storage == "pt" else "_h5"
                path = output_dir / (
                    f"TCGA_RCC_{feature}_split_v1_{suffix}{storage_suffix}.csv"
                )
                if require_features:
                    feature_paths = [
                        value
                        for column in frame.columns
                        if column.endswith("_slide_path")
                        for value in frame[column].dropna()
                    ]
                    missing = [
                        value for value in feature_paths if not os.path.isfile(value)
                    ]
                    if missing:
                        raise FileNotFoundError(
                            f"{len(missing)} missing {feature}/{storage} features; "
                            f"first={missing[:5]}"
                        )
                frame.to_csv(path, index=False)
                metadata[feature][f"{storage}_{suffix}"] = {
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--wsi-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--require-features", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.wsi_root.mkdir(parents=True, exist_ok=True)
    source_manifest = build_source_manifest(
        read_manifest(args.manifest), args.raw_dir
    )
    link_manifest = materialize_symlinks(source_manifest, args.wsi_root)
    assignment = grouped_stratified_split(
        source_manifest,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    source_manifest_path = args.output_dir / "tcga_rcc_source_manifest.csv"
    link_manifest_path = args.output_dir / "tcga_rcc_wsi_symlinks.csv"
    source_csv_path = args.output_dir / "tcga_rcc_wsi_paths.csv"
    assignment_path = args.output_dir / "tcga_rcc_split_v1_assignment.csv"
    labels_path = args.output_dir / "tcga_rcc_slide_labels.csv"

    source_manifest.to_csv(source_manifest_path, index=False)
    link_manifest.to_csv(link_manifest_path, index=False)
    source_manifest[["wsi_path"]].to_csv(source_csv_path, index=False)
    assignment.to_csv(assignment_path, index=False)
    source_manifest[
        ["slide_id", "patient_id", "project_id", "class_name", "label", "wsi_path"]
    ].to_csv(labels_path, index=False)
    feature_metadata = write_feature_splits(
        assignment,
        args.feature_root,
        args.output_dir,
        require_features=args.require_features,
    )

    split_counts = (
        assignment.groupby(["split", "class_name"]).size()
        .rename("slides")
        .reset_index()
    )
    patient_counts = (
        assignment[["split", "patient_id", "class_name"]]
        .drop_duplicates()
        .groupby(["split", "class_name"])
        .size()
        .rename("patients")
        .reset_index()
    )
    manifest_json = {
        "schema_version": 1,
        "dataset": "TCGA-RCC",
        "task": "three-class renal cell carcinoma subtype classification",
        "label_order": list(LABEL_ORDER),
        "label_mapping": {
            name: index for index, name in enumerate(LABEL_ORDER)
        },
        "source_manifest": str(source_manifest_path.resolve()),
        "source_manifest_sha256": file_sha256(source_manifest_path),
        "source_manifest_content_sha256": dataframe_sha256(
            source_manifest,
            ["slide_id", "patient_id", "project_id", "class_name", "label", "size"],
        ),
        "raw_gdc_manifest": str(args.manifest.resolve()),
        "raw_gdc_manifest_sha256": file_sha256(args.manifest),
        "wsi_source_csv": str(source_csv_path.resolve()),
        "wsi_source_csv_sha256": file_sha256(source_csv_path),
        "wsi_symlink_manifest": str(link_manifest_path.resolve()),
        "wsi_symlink_manifest_sha256": file_sha256(link_manifest_path),
        "labels": str(labels_path.resolve()),
        "labels_sha256": file_sha256(labels_path),
        "assignment": str(assignment_path.resolve()),
        "assignment_sha256": file_sha256(assignment_path),
        "split_seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "split_unit": "patient_id",
        "num_slides": int(len(source_manifest)),
        "num_patients": int(source_manifest["patient_id"].nunique()),
        "class_counts": {
            class_name: {
                "label": int(label),
                "slides": int((source_manifest["class_name"] == class_name).sum()),
                "patients": int(
                    source_manifest.loc[
                        source_manifest["class_name"] == class_name, "patient_id"
                    ].nunique()
                ),
            }
            for class_name, label in zip(LABEL_ORDER, range(len(LABEL_ORDER)))
        },
        "split_slide_counts": {
            f"{row.split}__{row.class_name}": int(row.slides)
            for row in split_counts.itertuples(index=False)
        },
        "split_patient_counts": {
            f"{row.split}__{row.class_name}": int(row.patients)
            for row in patient_counts.itertuples(index=False)
        },
        "features": feature_metadata,
        "integrity": {
            "filename_manifest_match": True,
            "size_match": True,
            "md5_checked": False,
            "md5_note": (
                "The script validates manifest membership and byte size. "
                "Full MD5 over 860G raw slides is intentionally not run by default."
            ),
        },
    }
    manifest_path = args.output_dir / "tcga_rcc_pipeline_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest_json, handle, indent=2)
    print(json.dumps(manifest_json, indent=2))


if __name__ == "__main__":
    main()
