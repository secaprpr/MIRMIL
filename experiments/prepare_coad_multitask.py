import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


TASKS = {
    "msi": {
        "label_column": "msi_label",
        "code_column": "msi_code",
        "classes": {0: "MSS", 1: "MSI"},
    },
    "cms": {
        "label_column": "cms_label",
        "code_column": "cms_code",
        "classes": {0: "CMS1", 1: "CMS2", 2: "CMS3", 3: "CMS4"},
    },
    "hmcings": {
        "label_column": "hmcings_label",
        "code_column": "hmcings_code",
        "classes": {0: "CIN", 1: "GS", 2: "HM"},
    },
}

SPLITS = ("train", "val", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_and_validate_labels(labels_path, feature_dir):
    frame = pd.read_csv(labels_path)
    required = {
        "filename",
        "slide_id",
        "case_id",
        "recommended_use_primary",
    }
    for task in TASKS.values():
        required.update((task["label_column"], task["code_column"]))
    missing_columns = required - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing label columns: {sorted(missing_columns)}")
    if frame["filename"].duplicated().any():
        raise ValueError("Duplicate feature filenames found")
    if frame["slide_id"].duplicated().any():
        raise ValueError("Duplicate slide IDs found")

    frame["feature_path"] = frame["filename"].map(
        lambda name: str((Path(feature_dir) / name).resolve())
    )
    missing_features = frame.loc[
        ~frame["feature_path"].map(lambda value: Path(value).is_file()),
        "feature_path",
    ]
    if len(missing_features):
        raise FileNotFoundError(
            f"{len(missing_features)} feature files are missing; "
            f"first: {missing_features.iloc[0]}"
        )

    for task_name, task in TASKS.items():
        label_column = task["label_column"]
        code_column = task["code_column"]
        labelled = frame[frame[label_column].notna()]
        inconsistent = labelled.groupby("case_id")[code_column].nunique()
        if (inconsistent > 1).any():
            case_id = inconsistent[inconsistent > 1].index[0]
            raise ValueError(
                f"Patient {case_id} has conflicting {task_name} labels"
            )
        observed = (
            labelled[[code_column, label_column]]
            .drop_duplicates()
            .set_index(code_column)[label_column]
            .to_dict()
        )
        expected = task["classes"]
        if observed != expected:
            raise ValueError(
                f"Unexpected {task_name} label mapping: "
                f"observed={observed}, expected={expected}"
            )
    return frame


def select_recommended_slides(slide_frame, recommended_only=True):
    if not recommended_only:
        return slide_frame.copy()
    selected = slide_frame[
        slide_frame["recommended_use_primary"].eq("Yes")
    ].copy()
    if selected.empty:
        raise ValueError("No recommended primary slides were found")
    return selected


def build_case_frame(slide_frame):
    columns = ["case_id"]
    columns.extend(task["code_column"] for task in TASKS.values())
    case_frame = slide_frame[columns].drop_duplicates("case_id").copy()
    for task in TASKS.values():
        code_column = task["code_column"]
        case_frame[code_column] = case_frame[code_column].astype(int)
    return case_frame.sort_values("case_id").reset_index(drop=True)


def build_indicator_matrix(case_frame):
    names = []
    columns = []
    for task_name, task in TASKS.items():
        values = case_frame[task["code_column"]].to_numpy()
        for code, class_name in task["classes"].items():
            names.append(f"{task_name}:{class_name}")
            columns.append(values == code)
    return names, np.column_stack(columns).astype(np.int16)


def split_sizes(count, train_ratio, val_ratio):
    train_count = round(count * train_ratio)
    val_count = round(count * val_ratio)
    return np.array(
        [train_count, val_count, count - train_count - val_count],
        dtype=int,
    )


def assignment_score(counts, targets):
    relative_error = np.abs(counts - targets) / np.maximum(targets, 1.0)
    weighted_squared_error = np.square(counts - targets) / np.maximum(
        targets, 1.0
    )
    return float(relative_error.max()), float(weighted_squared_error.sum())


def shared_multitask_split(
    case_frame,
    seed=2024,
    train_ratio=0.6,
    val_ratio=0.2,
    search_iterations=10000,
):
    if train_ratio <= 0 or val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")
    if search_iterations <= 0:
        raise ValueError("search_iterations must be positive")

    _, indicators = build_indicator_matrix(case_frame)
    sizes = split_sizes(len(case_frame), train_ratio, val_ratio)
    ratios = sizes / len(case_frame)
    targets = ratios[:, None] * indicators.sum(axis=0)[None, :]
    rng = np.random.default_rng(seed)
    best_score = None
    best_assignment = None

    for _ in range(search_iterations):
        permutation = rng.permutation(len(case_frame))
        assignment = np.empty(len(case_frame), dtype=np.int8)
        start = 0
        counts = []
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


def task_counts(assignments):
    slide_counts = (
        assignments.groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(SPLITS)
    )
    case_counts = (
        assignments.drop_duplicates("case_id")
        .groupby(["split", "class_name"])
        .size()
        .unstack(fill_value=0)
        .reindex(SPLITS)
    )
    return {
        "slides": slide_counts.to_dict(orient="index"),
        "cases": case_counts.to_dict(orient="index"),
    }


def prepare_outputs(slide_frame, case_assignments, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_path = output_dir / "COAD_shared_case_assignments.csv"
    case_assignments.to_csv(master_path, index=False)

    task_manifests = {}
    for task_name, task in TASKS.items():
        label_column = task["label_column"]
        code_column = task["code_column"]
        assignments = slide_frame[slide_frame[label_column].notna()].copy()
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

        assignment_path = output_dir / f"COAD_{task_name}_assignments.csv"
        split_path = output_dir / f"COAD_{task_name}_split.csv"
        assignments.to_csv(assignment_path, index=False)
        build_training_csv(assignments, split_path)
        task_manifests[task_name] = {
            "num_classes": len(task["classes"]),
            "class_mapping": task["classes"],
            "counts": task_counts(assignments),
            "assignment_path": str(assignment_path.resolve()),
            "assignment_sha256": file_sha256(assignment_path),
            "split_path": str(split_path.resolve()),
            "split_sha256": file_sha256(split_path),
        }
    return master_path, task_manifests


def task_overlap(case_frame):
    available = {}
    for task_name, task in TASKS.items():
        available[task_name] = case_frame[task["code_column"]] >= 0
    overlap = {}
    task_names = list(TASKS)
    for index, first in enumerate(task_names):
        for second in task_names[index + 1 :]:
            overlap[f"{first}+{second}"] = int(
                (available[first] & available[second]).sum()
            )
    overlap["all"] = int(
        np.logical_and.reduce([available[name] for name in task_names]).sum()
    )
    return overlap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--search-iterations", type=int, default=10000)
    parser.add_argument(
        "--recommended-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude slides not recommended as primary-tumor inputs",
    )
    args = parser.parse_args()

    source_frame = load_and_validate_labels(args.labels, args.feature_dir)
    slide_frame = select_recommended_slides(
        source_frame, recommended_only=args.recommended_only
    )
    case_frame = build_case_frame(slide_frame)
    case_assignments = shared_multitask_split(
        case_frame,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        search_iterations=args.search_iterations,
    )
    master_path, task_manifests = prepare_outputs(
        slide_frame, case_assignments, args.output_dir
    )
    manifest = {
        "dataset": "TCGA-COAD",
        "labels_path": str(Path(args.labels).resolve()),
        "labels_sha256": file_sha256(args.labels),
        "feature_dir": str(Path(args.feature_dir).resolve()),
        "num_source_slides": len(source_frame),
        "num_slides": len(slide_frame),
        "excluded_nonrecommended_slides": len(source_frame) - len(slide_frame),
        "num_cases": len(case_frame),
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "search_iterations": args.search_iterations,
        "task_overlap_cases": task_overlap(case_frame),
        "shared_assignment_path": str(master_path.resolve()),
        "shared_assignment_sha256": file_sha256(master_path),
        "tasks": task_manifests,
    }
    manifest_path = Path(args.output_dir) / "COAD_multitask_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
