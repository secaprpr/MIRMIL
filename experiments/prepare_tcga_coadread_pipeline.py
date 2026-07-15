"""Prepare TCGA-COADREAD WSI metadata and prognosis feature split CSVs.

This script consumes the patient-level prognosis assignments already prepared
under ``metadata/TCGA-COADREAD-PROGNOSIS`` and joins them to diagnostic slide
files from a GDC TSV manifest. It mirrors the BLCA prognosis split layout used
by ``SurvivalWSIDataset``:

``train_slide_id, train_patient_id, train_feature_path, train_time_months, ...``

The script can be run before feature extraction to materialize deterministic
expected feature paths. Use ``--require-wsi`` after the GDC download finishes
and ``--require-features`` after R50/UNI extraction finishes.
"""

import argparse
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd


TCGA_PATIENT_PATTERN = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")
PROJECT_TO_COHORT = {
    "TCGA-COAD": "COAD",
    "TCGA-READ": "READ",
}
ENDPOINTS = ("OS", "PFS", "DSS", "DFS")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def patient_id_from_slide(slide_id):
    match = TCGA_PATIENT_PATTERN.match(str(slide_id).upper())
    if not match:
        raise ValueError(f"Cannot parse TCGA patient ID from {slide_id}")
    return match.group(1)


def dataframe_sha256(frame, columns):
    payload = frame[columns].sort_values(columns).to_csv(index=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def read_gdc_manifest(path):
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
        raise ValueError(f"GDC manifest missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["size"] = frame["size"].astype(int)
    if frame["id"].duplicated().any():
        raise ValueError("GDC file IDs must be unique")
    if frame["filename"].duplicated().any():
        raise ValueError("GDC filenames must be unique")
    unknown_projects = set(frame["project_id"]) - set(PROJECT_TO_COHORT)
    if unknown_projects:
        raise ValueError(f"Unexpected projects: {sorted(unknown_projects)}")
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
    frame["cohort"] = frame["project_id"].map(PROJECT_TO_COHORT)
    frame["slide_id"] = frame["filename"].str.replace(
        r"\.svs$", "", case=False, regex=True
    )
    frame["patient_id"] = frame["case_submitter_id"].fillna("")
    missing_patient = frame["patient_id"].eq("")
    if missing_patient.any():
        frame.loc[missing_patient, "patient_id"] = frame.loc[
            missing_patient, "slide_id"
        ].map(patient_id_from_slide)
    return frame.sort_values("slide_id").reset_index(drop=True)


def attach_raw_paths(manifest, raw_dir, require_wsi):
    raw_dir = Path(raw_dir).resolve()
    rows = []
    for row in manifest.itertuples(index=False):
        path = raw_dir / row.id / row.filename
        exists = path.is_file()
        actual_size = path.stat().st_size if exists else None
        size_match = bool(exists and actual_size == int(row.size))
        if require_wsi and not size_match:
            raise FileNotFoundError(
                f"Missing/incomplete WSI for {row.filename}: "
                f"path={path}, actual_size={actual_size}, expected={row.size}"
            )
        rows.append(
            {
                **row._asdict(),
                "wsi_path": str(path),
                "wsi_exists": exists,
                "actual_size": actual_size,
                "size_match": size_match,
            }
        )
    source = pd.DataFrame(rows)
    return source


def write_wsi_symlinks(source_manifest, wsi_root, require_wsi):
    wsi_root = Path(wsi_root).resolve()
    records = []
    for row in source_manifest.itertuples(index=False):
        cohort_dir = wsi_root / row.cohort
        cohort_dir.mkdir(parents=True, exist_ok=True)
        destination = cohort_dir / row.filename
        source = Path(row.wsi_path)
        if row.size_match:
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise ValueError(f"Conflicting symlink: {destination}")
            elif destination.exists():
                raise FileExistsError(destination)
            else:
                destination.symlink_to(source.resolve())
            link = str(destination)
        else:
            if require_wsi:
                raise FileNotFoundError(source)
            link = ""
        records.append(
            {
                "slide_id": row.slide_id,
                "filename": row.filename,
                "patient_id": row.patient_id,
                "project_id": row.project_id,
                "cohort": row.cohort,
                "wsi_path": str(source),
                "wsi_link": link,
                "size_match": row.size_match,
            }
        )
    return pd.DataFrame(records).sort_values("slide_id").reset_index(drop=True)


def feature_path(feature_root, feature_type, storage, slide_id):
    return str(
        Path(feature_root).resolve()
        / feature_type
        / f"{storage}_files"
        / f"{slide_id}.{storage}"
    )


def build_long_endpoint(assignments, source_manifest, feature_root, feature_type, storage):
    rows = []
    slides_by_patient = {
        patient: group.sort_values("slide_id")
        for patient, group in source_manifest.groupby("patient_id")
    }
    unmatched = []
    for assignment in assignments.itertuples(index=False):
        patient_id = str(assignment.patient_id)
        slides = slides_by_patient.get(patient_id)
        if slides is None or slides.empty:
            unmatched.append(patient_id)
            continue
        for slide in slides.itertuples(index=False):
            rows.append(
                {
                    "split": assignment.split,
                    "slide_id": slide.slide_id,
                    "patient_id": patient_id,
                    "feature_path": feature_path(
                        feature_root, feature_type, storage, slide.slide_id
                    ),
                    "time_months": float(assignment.time_months),
                    "event": int(assignment.event),
                    "status": assignment.status,
                }
            )
    long_frame = pd.DataFrame(rows)
    if not long_frame.empty:
        long_frame = long_frame.sort_values(
            ["split", "patient_id", "slide_id"]
        ).reset_index(drop=True)
    return long_frame, sorted(set(unmatched))


def wide_from_long(long_frame, include_test):
    columns = {}
    for split in ("train", "val", "test"):
        frame = long_frame[long_frame["split"] == split]
        if split == "test" and not include_test:
            frame = frame.iloc[:0]
        columns[f"{split}_slide_id"] = pd.Series(
            frame["slide_id"].tolist(), dtype="object"
        )
        columns[f"{split}_patient_id"] = pd.Series(
            frame["patient_id"].tolist(), dtype="object"
        )
        columns[f"{split}_feature_path"] = pd.Series(
            frame["feature_path"].tolist(), dtype="object"
        )
        columns[f"{split}_time_months"] = pd.Series(
            frame["time_months"].tolist(), dtype="float64"
        )
        columns[f"{split}_event"] = pd.Series(
            frame["event"].astype(int).tolist(), dtype="Int64"
        )
        columns[f"{split}_status"] = pd.Series(
            frame["status"].tolist(), dtype="object"
        )
    return pd.DataFrame(columns)


def write_endpoint_splits(
    prognosis_dir,
    source_manifest,
    feature_root,
    output_dir,
    require_features,
):
    outputs = {}
    unmatched = {}
    for endpoint in ENDPOINTS:
        assignment_path = (
            Path(prognosis_dir) / f"TCGA_COADREAD_PROGNOSIS_{endpoint}_assignments.csv"
        )
        if not assignment_path.is_file():
            continue
        assignments = pd.read_csv(assignment_path)
        outputs[endpoint] = {}
        unmatched[endpoint] = {}
        for feature_type in ("r50", "uni"):
            outputs[endpoint][feature_type] = {}
            long_frame, missing_patients = build_long_endpoint(
                assignments,
                source_manifest,
                feature_root,
                feature_type,
                "pt",
            )
            unmatched[endpoint][feature_type] = missing_patients
            long_path = (
                Path(output_dir)
                / f"TCGA_COADREAD_PROGNOSIS_{feature_type.upper()}_{endpoint}_long.csv"
            )
            long_frame.to_csv(long_path, index=False)
            outputs[endpoint][feature_type]["long"] = {
                "path": str(long_path),
                "sha256": file_sha256(long_path),
                "rows": int(len(long_frame)),
                "patients": int(long_frame["patient_id"].nunique())
                if len(long_frame)
                else 0,
                "unmatched_assignment_patients": len(missing_patients),
            }
            for include_test, suffix in ((True, "split"), (False, "train_val")):
                wide = wide_from_long(long_frame, include_test=include_test)
                path = (
                    Path(output_dir)
                    / f"TCGA_COADREAD_PROGNOSIS_{feature_type.upper()}_{endpoint}_{suffix}.csv"
                )
                if require_features:
                    paths = [
                        value
                        for column in wide.columns
                        if column.endswith("_feature_path")
                        for value in wide[column].dropna()
                    ]
                    missing = [value for value in paths if not os.path.isfile(value)]
                    if missing:
                        raise FileNotFoundError(
                            f"{len(missing)} missing {feature_type} features; "
                            f"first={missing[:5]}"
                        )
                wide.to_csv(path, index=False)
                outputs[endpoint][feature_type][suffix] = {
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "rows": int(len(wide)),
                }
    return outputs, unmatched


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--wsi-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--prognosis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-wsi", action="store_true")
    parser.add_argument("--require-features", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.wsi_root.mkdir(parents=True, exist_ok=True)

    gdc_manifest = read_gdc_manifest(args.manifest)
    source_manifest = attach_raw_paths(
        gdc_manifest, args.raw_dir, require_wsi=args.require_wsi
    )
    symlink_manifest = write_wsi_symlinks(
        source_manifest, args.wsi_root, require_wsi=args.require_wsi
    )

    source_path = args.output_dir / "tcga_coadread_source_manifest.csv"
    wsi_paths_path = args.output_dir / "tcga_coadread_wsi_paths.csv"
    symlink_path = args.output_dir / "tcga_coadread_wsi_symlinks.csv"
    labels_path = args.output_dir / "tcga_coadread_slide_labels.csv"

    source_manifest.to_csv(source_path, index=False)
    source_manifest[["wsi_path"]].to_csv(wsi_paths_path, index=False)
    symlink_manifest.to_csv(symlink_path, index=False)
    source_manifest[
        ["slide_id", "patient_id", "project_id", "cohort", "wsi_path"]
    ].to_csv(labels_path, index=False)

    split_outputs, unmatched = write_endpoint_splits(
        prognosis_dir=args.prognosis_dir,
        source_manifest=source_manifest,
        feature_root=args.feature_root,
        output_dir=args.output_dir,
        require_features=args.require_features,
    )

    project_counts = (
        source_manifest.groupby("project_id")
        .agg(slides=("slide_id", "count"), patients=("patient_id", "nunique"))
        .reset_index()
    )
    manifest_json = {
        "schema_version": 1,
        "dataset": "TCGA-COADREAD",
        "task": "prognosis",
        "endpoints": list(split_outputs),
        "raw_gdc_manifest": str(args.manifest.resolve()),
        "raw_gdc_manifest_sha256": file_sha256(args.manifest),
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": file_sha256(source_path),
        "source_manifest_content_sha256": dataframe_sha256(
            source_manifest,
            ["slide_id", "patient_id", "project_id", "cohort", "size"],
        ),
        "wsi_source_csv": str(wsi_paths_path.resolve()),
        "wsi_source_csv_sha256": file_sha256(wsi_paths_path),
        "wsi_symlink_manifest": str(symlink_path.resolve()),
        "wsi_symlink_manifest_sha256": file_sha256(symlink_path),
        "labels": str(labels_path.resolve()),
        "labels_sha256": file_sha256(labels_path),
        "feature_root": str(args.feature_root.resolve()),
        "prognosis_dir": str(args.prognosis_dir.resolve()),
        "num_slides": int(len(source_manifest)),
        "num_patients": int(source_manifest["patient_id"].nunique()),
        "wsi_size_matched": int(source_manifest["size_match"].sum()),
        "wsi_missing_or_partial": int((~source_manifest["size_match"]).sum()),
        "project_counts": {
            row.project_id: {
                "slides": int(row.slides),
                "patients": int(row.patients),
            }
            for row in project_counts.itertuples(index=False)
        },
        "splits": split_outputs,
        "unmatched_assignment_patients": unmatched,
        "integrity": {
            "require_wsi": bool(args.require_wsi),
            "require_features": bool(args.require_features),
            "byte_size_checked": True,
            "md5_checked": False,
        },
    }
    manifest_path = args.output_dir / "tcga_coadread_pipeline_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest_json, handle, indent=2)
    print(json.dumps(manifest_json, indent=2))


if __name__ == "__main__":
    main()
