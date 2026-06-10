import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd


TCGA_PATIENT_PATTERN = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def patient_id_from_slide(slide_id):
    match = TCGA_PATIENT_PATTERN.match(slide_id.upper())
    if not match:
        raise ValueError(f"Cannot extract TCGA patient ID from: {slide_id}")
    return match.group(1)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_features(class_dirs, extension):
    rows = []
    seen = {}
    for label, (class_name, feature_dir) in enumerate(class_dirs):
        paths = sorted(Path(feature_dir).glob(f"*{extension}"))
        if not paths:
            raise FileNotFoundError(
                f"No {extension} files found in {feature_dir}"
            )
        for path in paths:
            slide_id = path.name[: -len(extension)]
            if slide_id in seen:
                raise ValueError(
                    f"Duplicate slide ID in {path} and {seen[slide_id]}"
                )
            seen[slide_id] = path
            rows.append(
                {
                    "slide_id": slide_id,
                    "patient_id": patient_id_from_slide(slide_id),
                    "label": label,
                    "class_name": class_name,
                    "source_path": str(path.resolve()),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.groupby("patient_id")["label"].nunique().max() > 1:
        raise ValueError("At least one patient occurs in multiple classes")
    return frame


def build_link_directory(frame, output_dir, extension):
    link_dir = Path(output_dir) / "pt_files"
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
    parser.add_argument("--luad-dir", required=True)
    parser.add_argument("--lusc-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--extension", default=".pt")
    parser.add_argument(
        "--link-features",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = collect_features(
        [("LUAD", args.luad_dir), ("LUSC", args.lusc_dir)],
        args.extension,
    )
    labels_path = output_dir / "NSCLC_labels.csv"
    frame.drop(columns="source_path").to_csv(labels_path, index=False)
    link_dir = (
        build_link_directory(frame, output_dir, args.extension)
        if args.link_features
        else None
    )

    class_counts = {}
    for class_name, group in frame.groupby("class_name", sort=True):
        class_counts[class_name] = {
            "slides": len(group),
            "patients": int(group["patient_id"].nunique()),
        }
    manifest = {
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": file_sha256(labels_path),
        "feature_dir": str(link_dir.resolve()) if link_dir else None,
        "extension": args.extension,
        "num_slides": len(frame),
        "num_patients": int(frame["patient_id"].nunique()),
        "classes": class_counts,
    }
    manifest_path = output_dir / "NSCLC_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
