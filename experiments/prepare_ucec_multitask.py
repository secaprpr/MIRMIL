import argparse
import hashlib
import json
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd


STUDY_ID = "ucec_tcga_pan_can_atlas_2018"
API_ROOT = f"https://www.cbioportal.org/api/studies/{STUDY_ID}"
PATIENT_PATTERN = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")
SAMPLE_PATTERN = re.compile(
    r"^TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}-(\d{2})[A-Z]-"
)
SPLITS = ("train", "val", "test")
SUBTYPE_MAPPING = {
    "UCEC_POLE": 0,
    "UCEC_MSI": 1,
    "UCEC_CN_LOW": 2,
    "UCEC_CN_HIGH": 3,
}
TASKS = {
    "msi": {
        "label_column": "msi_label",
        "code_column": "msi_code",
        "classes": {0: "MSS", 1: "MSI"},
    },
    "subtype": {
        "label_column": "subtype_label",
        "code_column": "subtype_code",
        "classes": {
            0: "POLE",
            1: "MSI",
            2: "CN_LOW",
            3: "CN_HIGH",
        },
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_clinical_data(clinical_data_type):
    query = urllib.parse.urlencode(
        {"clinicalDataType": clinical_data_type}
    )
    request = urllib.request.Request(
        f"{API_ROOT}/clinical-data?{query}",
        headers={"Accept": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def parse_patient_subtypes(records):
    subtypes = {}
    for record in records:
        if record.get("clinicalAttributeId") != "SUBTYPE":
            continue
        value = record.get("value")
        if value not in SUBTYPE_MAPPING:
            raise ValueError(f"Unexpected UCEC subtype: {value}")
        patient_id = record["patientId"]
        if patient_id in subtypes and subtypes[patient_id] != value:
            raise ValueError(f"Conflicting subtype for {patient_id}")
        subtypes[patient_id] = value
    return subtypes


def parse_mantis_scores(records):
    scores = {}
    for record in records:
        if record.get("clinicalAttributeId") != "MSI_SCORE_MANTIS":
            continue
        patient_id = record["patientId"]
        value = float(record["value"])
        scores.setdefault(patient_id, []).append(value)
    return {patient_id: max(values) for patient_id, values in scores.items()}


def patient_id_from_filename(filename):
    match = PATIENT_PATTERN.match(filename.upper())
    if not match:
        raise ValueError(f"Cannot extract TCGA patient ID from: {filename}")
    return match.group(1)


def sample_type_from_filename(filename):
    match = SAMPLE_PATTERN.match(filename.upper())
    if not match:
        raise ValueError(f"Cannot extract TCGA sample type from: {filename}")
    return match.group(1)


def collect_slides(feature_dir, primary_only=True):
    rows = []
    for path in sorted(Path(feature_dir).glob("*.h5")):
        sample_type = sample_type_from_filename(path.name)
        if primary_only and sample_type != "01":
            continue
        rows.append(
            {
                "filename": path.name,
                "slide_id": path.stem,
                "case_id": patient_id_from_filename(path.name),
                "sample_type_code": sample_type,
                "feature_path": str(path.resolve()),
            }
        )
    if not rows:
        raise FileNotFoundError(f"No eligible H5 files found in {feature_dir}")
    frame = pd.DataFrame(rows)
    if frame["filename"].duplicated().any():
        raise ValueError("Duplicate feature filenames found")
    return frame


def attach_labels(slide_frame, subtypes, mantis_scores):
    frame = slide_frame.copy()
    frame["mantis_score"] = frame["case_id"].map(mantis_scores)
    frame["msi_label"] = np.select(
        [frame["mantis_score"] < 0.4, frame["mantis_score"] > 0.6],
        ["MSS", "MSI"],
        default=None,
    )
    frame["msi_code"] = frame["msi_label"].map({"MSS": 0, "MSI": 1})
    source_subtype = frame["case_id"].map(subtypes)
    frame["subtype_label"] = source_subtype.map(
        {
            "UCEC_POLE": "POLE",
            "UCEC_MSI": "MSI",
            "UCEC_CN_LOW": "CN_LOW",
            "UCEC_CN_HIGH": "CN_HIGH",
        }
    )
    frame["subtype_code"] = source_subtype.map(SUBTYPE_MAPPING)
    return frame


def build_case_frame(slide_frame):
    columns = ["case_id", "msi_code", "subtype_code"]
    return (
        slide_frame[columns]
        .drop_duplicates("case_id")
        .sort_values("case_id")
        .reset_index(drop=True)
    )


def indicator_matrix(case_frame):
    columns = []
    for task in TASKS.values():
        values = case_frame[task["code_column"]].to_numpy()
        for code in task["classes"]:
            columns.append(values == code)
    return np.column_stack(columns).astype(np.int16)


def assignment_score(counts, targets):
    relative_error = np.abs(counts - targets) / np.maximum(targets, 1.0)
    weighted_error = np.square(counts - targets) / np.maximum(targets, 1.0)
    return float(relative_error.max()), float(weighted_error.sum())


def shared_multitask_split(
    case_frame,
    seed=2024,
    train_ratio=0.6,
    val_ratio=0.2,
    search_iterations=20000,
):
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    if search_iterations <= 0:
        raise ValueError("search_iterations must be positive")

    indicators = indicator_matrix(case_frame)
    count = len(case_frame)
    sizes = np.array(
        [
            round(count * train_ratio),
            round(count * val_ratio),
            count - round(count * train_ratio) - round(count * val_ratio),
        ]
    )
    targets = sizes[:, None] / count * indicators.sum(axis=0)[None, :]
    rng = np.random.default_rng(seed)
    best_score = None
    best_assignment = None
    for _ in range(search_iterations):
        permutation = rng.permutation(count)
        assignment = np.empty(count, dtype=np.int8)
        counts = []
        start = 0
        for split_index, size in enumerate(sizes):
            indices = permutation[start : start + size]
            assignment[indices] = split_index
            counts.append(indicators[indices].sum(axis=0))
            start += size
        counts = np.stack(counts)
        if (counts == 0).any():
            continue
        score = assignment_score(counts, targets)
        if best_score is None or score < best_score:
            best_score = score
            best_assignment = assignment.copy()
    if best_assignment is None:
        raise RuntimeError("Could not find a split containing every task class")
    result = case_frame.copy()
    result["split"] = [SPLITS[index] for index in best_assignment]
    return result


def build_training_csv(assignments, output_path):
    columns = {}
    for split in SPLITS:
        subset = assignments[assignments["split"] == split].reset_index(drop=True)
        columns[f"{split}_slide_path"] = subset["feature_path"]
        columns[f"{split}_label"] = subset["label"]
    pd.DataFrame(columns).to_csv(output_path, index=False)


def count_task(assignments):
    case_frame = assignments.drop_duplicates("case_id")
    return {
        "slides": (
            assignments.groupby(["split", "class_name"])
            .size()
            .unstack(fill_value=0)
            .reindex(SPLITS)
            .to_dict(orient="index")
        ),
        "cases": (
            case_frame.groupby(["split", "class_name"])
            .size()
            .unstack(fill_value=0)
            .reindex(SPLITS)
            .to_dict(orient="index")
        ),
    }


def prepare_outputs(slide_frame, case_assignments, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / "UCEC_multitask_labels.csv"
    shared_path = output_dir / "UCEC_shared_case_assignments.csv"
    slide_frame.to_csv(labels_path, index=False)
    case_assignments.to_csv(shared_path, index=False)

    task_manifests = {}
    for task_name, task in TASKS.items():
        code_column = task["code_column"]
        label_column = task["label_column"]
        assignments = slide_frame[slide_frame[code_column].notna()].copy()
        assignments["label"] = assignments[code_column].astype(int)
        assignments["class_name"] = assignments[label_column]
        assignments = assignments.merge(
            case_assignments[["case_id", "split"]],
            on="case_id",
            how="left",
            validate="many_to_one",
        )
        if assignments["split"].isna().any():
            raise RuntimeError(f"Unassigned patients in {task_name}")
        if assignments.groupby("case_id")["split"].nunique().max() != 1:
            raise RuntimeError(f"Patient leakage detected in {task_name}")
        assignments = assignments.sort_values(
            ["split", "case_id", "filename"]
        ).reset_index(drop=True)
        assignment_path = output_dir / f"UCEC_{task_name}_assignments.csv"
        split_path = output_dir / f"UCEC_{task_name}_split.csv"
        assignments.to_csv(assignment_path, index=False)
        build_training_csv(assignments, split_path)
        task_manifests[task_name] = {
            "num_classes": len(task["classes"]),
            "class_mapping": task["classes"],
            "counts": count_task(assignments),
            "assignment_path": str(assignment_path.resolve()),
            "assignment_sha256": file_sha256(assignment_path),
            "split_path": str(split_path.resolve()),
            "split_sha256": file_sha256(split_path),
        }
    return labels_path, shared_path, task_manifests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--search-iterations", type=int, default=20000)
    parser.add_argument(
        "--primary-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    patient_records = fetch_clinical_data("PATIENT")
    sample_records = fetch_clinical_data("SAMPLE")
    subtypes = parse_patient_subtypes(patient_records)
    mantis_scores = parse_mantis_scores(sample_records)
    source_slides = collect_slides(
        args.feature_dir, primary_only=args.primary_only
    )
    slide_frame = attach_labels(source_slides, subtypes, mantis_scores)
    labelled = slide_frame[
        slide_frame[["msi_code", "subtype_code"]].notna().any(axis=1)
    ].copy()
    case_frame = build_case_frame(labelled)
    case_assignments = shared_multitask_split(
        case_frame,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        search_iterations=args.search_iterations,
    )
    labels_path, shared_path, tasks = prepare_outputs(
        labelled, case_assignments, args.output_dir
    )
    manifest = {
        "dataset": "TCGA-UCEC",
        "feature_dir": str(Path(args.feature_dir).resolve()),
        "label_source": {
            "study_id": STUDY_ID,
            "api_root": API_ROOT,
            "subtype_attribute": "SUBTYPE",
            "msi_attribute": "MSI_SCORE_MANTIS",
            "msi_thresholds": {"MSS": "<0.4", "MSI": ">0.6"},
        },
        "primary_only": args.primary_only,
        "num_source_slides": len(source_slides),
        "num_labelled_slides": len(labelled),
        "num_labelled_cases": len(case_frame),
        "num_subtype_records": len(subtypes),
        "num_mantis_records": len(mantis_scores),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "search_iterations": args.search_iterations,
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": file_sha256(labels_path),
        "shared_assignment_path": str(shared_path.resolve()),
        "shared_assignment_sha256": file_sha256(shared_path),
        "tasks": tasks,
    }
    manifest_path = Path(args.output_dir) / "UCEC_multitask_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
