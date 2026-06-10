import argparse
import hashlib
import json
import re
from pathlib import Path

import h5py
import pandas as pd


TCGA_PATIENT_PATTERN = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def patient_id_from_slide(slide_id):
    match = TCGA_PATIENT_PATTERN.match(slide_id.upper())
    if not match:
        raise ValueError(f"Cannot extract TCGA patient ID from: {slide_id}")
    return match.group(1)


def parse_project(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Projects must use CLASS_NAME=/path/to/features"
        )
    class_name, feature_dir = value.split("=", 1)
    if not class_name or not feature_dir:
        raise argparse.ArgumentTypeError(
            "Projects must use CLASS_NAME=/path/to/features"
        )
    return class_name, Path(feature_dir)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_h5(path):
    with h5py.File(path, "r") as file:
        if "features" not in file:
            raise ValueError(f"Missing features dataset: {path}")
        shape = tuple(file["features"].shape)
        if len(shape) == 3 and shape[0] == 1:
            num_instances, feature_dim = shape[1:]
        elif len(shape) == 2:
            num_instances, feature_dim = shape
        else:
            raise ValueError(f"Unexpected feature shape {shape}: {path}")
        if num_instances <= 0 or feature_dim <= 0:
            raise ValueError(f"Empty feature dataset: {path}")
    return int(num_instances), int(feature_dim)


def collect_projects(projects, extension=".h5", validate_h5=True):
    rows = []
    dimensions = set()
    seen_slides = {}
    for label, (class_name, feature_dir) in enumerate(projects):
        paths = sorted(feature_dir.glob(f"*{extension}"))
        if not paths:
            raise FileNotFoundError(
                f"No {extension} files found in {feature_dir}"
            )
        for path in paths:
            slide_id = path.name[: -len(extension)]
            if slide_id in seen_slides:
                raise ValueError(
                    f"Duplicate slide ID in {path} and "
                    f"{seen_slides[slide_id]}"
                )
            seen_slides[slide_id] = path
            num_instances = None
            feature_dim = None
            if validate_h5 and extension == ".h5":
                num_instances, feature_dim = inspect_h5(path)
                dimensions.add(feature_dim)
            rows.append(
                {
                    "slide_id": slide_id,
                    "patient_id": patient_id_from_slide(slide_id),
                    "label": label,
                    "class_name": class_name,
                    "source_path": str(path.resolve()),
                    "num_instances": num_instances,
                    "feature_dim": feature_dim,
                }
            )
    if len(dimensions) > 1:
        raise ValueError(f"Inconsistent feature dimensions: {dimensions}")
    frame = pd.DataFrame(rows)
    if frame.groupby("patient_id")["label"].nunique().max() > 1:
        raise ValueError("At least one patient occurs in multiple classes")
    return frame


def build_link_directory(frame, output_dir, extension):
    link_dir = Path(output_dir) / "features"
    link_dir.mkdir(parents=True, exist_ok=True)
    for row in frame.itertuples(index=False):
        destination = link_dir / f"{row.slide_id}{extension}"
        source = Path(row.source_path)
        if destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise ValueError(f"Conflicting symlink: {destination}")
        elif destination.exists():
            raise FileExistsError(destination)
        else:
            destination.symlink_to(source)
    return link_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--project",
        action="append",
        type=parse_project,
        required=True,
        help="Repeat CLASS_NAME=/path/to/features in desired label order",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extension", default=".h5")
    parser.add_argument(
        "--validate-h5",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = collect_projects(
        args.project,
        extension=args.extension,
        validate_h5=args.validate_h5,
    )
    labels_path = output_dir / "TCGA_projects_labels.csv"
    frame.drop(columns="source_path").to_csv(labels_path, index=False)
    link_dir = build_link_directory(frame, output_dir, args.extension)

    class_counts = {}
    for class_name, group in frame.groupby("class_name", sort=False):
        class_counts[class_name] = {
            "label": int(group["label"].iloc[0]),
            "slides": len(group),
            "patients": int(group["patient_id"].nunique()),
        }
    dimensions = frame["feature_dim"].dropna().unique().tolist()
    manifest = {
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": file_sha256(labels_path),
        "feature_dir": str(link_dir.resolve()),
        "extension": args.extension,
        "validated_h5": args.validate_h5,
        "feature_dim": int(dimensions[0]) if dimensions else None,
        "num_slides": len(frame),
        "num_patients": int(frame["patient_id"].nunique()),
        "classes": class_counts,
    }
    manifest_path = output_dir / "TCGA_projects_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
