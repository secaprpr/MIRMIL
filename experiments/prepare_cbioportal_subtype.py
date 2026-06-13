import argparse
import hashlib
import json
import re
from pathlib import Path

import pandas as pd
import requests


TCGA_PATIENT_PATTERN = re.compile(r"^(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4})")


def parse_label_mapping(value):
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "Label mappings must use SOURCE_VALUE=CLASS_NAME"
        )
    source_value, class_name = value.split("=", 1)
    if not source_value or not class_name:
        raise argparse.ArgumentTypeError(
            "Label mappings must use SOURCE_VALUE=CLASS_NAME"
        )
    return source_value, class_name


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


def fetch_clinical_data(
    study_id,
    attribute_id,
    clinical_data_type="PATIENT",
    base_url="https://www.cbioportal.org/api",
):
    url = f"{base_url}/studies/{study_id}/clinical-data"
    response = requests.get(
        url,
        params={
            "clinicalDataType": clinical_data_type,
            "attributeId": attribute_id,
            "projection": "SUMMARY",
            "pageSize": 10000000,
        },
        timeout=120,
    )
    response.raise_for_status()
    records = response.json()
    if not records:
        raise ValueError(
            f"No clinical data returned for {study_id}/{attribute_id}"
        )
    return records, response.url


def collect_feature_slides(feature_dir, extension=".h5"):
    rows = []
    for path in sorted(Path(feature_dir).glob(f"*{extension}")):
        slide_id = path.name[: -len(extension)]
        rows.append(
            {
                "slide_id": slide_id,
                "patient_id": patient_id_from_slide(slide_id),
                "feature_path": str(path.resolve()),
            }
        )
    if not rows:
        raise FileNotFoundError(
            f"No {extension} files found in {feature_dir}"
        )
    frame = pd.DataFrame(rows)
    if frame["slide_id"].duplicated().any():
        raise ValueError("Duplicate slide IDs found")
    return frame


def build_labeled_slides(features, clinical_records, label_mappings):
    source_to_class = dict(label_mappings)
    if len(source_to_class) != len(label_mappings):
        raise ValueError("Duplicate source label mapping")
    class_names = list(dict.fromkeys(source_to_class.values()))
    if len(class_names) != len(source_to_class):
        raise ValueError("Each source value must map to a distinct class")
    class_to_label = {
        class_name: label for label, class_name in enumerate(class_names)
    }

    clinical = pd.DataFrame(clinical_records)
    required = {"patientId", "value"}
    missing = required.difference(clinical.columns)
    if missing:
        raise ValueError(f"Missing clinical fields: {sorted(missing)}")
    clinical = clinical[["patientId", "value"]].rename(
        columns={"patientId": "patient_id", "value": "source_label"}
    )
    clinical = clinical[clinical["source_label"].isin(source_to_class)].copy()
    clinical["class_name"] = clinical["source_label"].map(source_to_class)
    clinical["label"] = clinical["class_name"].map(class_to_label)
    if clinical["patient_id"].duplicated().any():
        raise ValueError("Duplicate clinical labels for a patient")

    labeled = features.merge(
        clinical,
        on="patient_id",
        how="left",
        validate="many_to_one",
    )
    retained = labeled[labeled["label"].notna()].copy()
    retained["label"] = retained["label"].astype(int)
    return retained, labeled[labeled["label"].isna()].copy(), class_to_label


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--attribute-id", default="SUBTYPE")
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument(
        "--label",
        action="append",
        type=parse_label_mapping,
        required=True,
        help="Repeat SOURCE_VALUE=CLASS_NAME in desired numeric label order",
    )
    parser.add_argument("--feature-extension", default=".h5")
    parser.add_argument(
        "--base-url",
        default="https://www.cbioportal.org/api",
    )
    args = parser.parse_args()

    features = collect_feature_slides(
        args.feature_dir,
        extension=args.feature_extension,
    )
    clinical_records, source_url = fetch_clinical_data(
        args.study_id,
        args.attribute_id,
        base_url=args.base_url,
    )
    labeled, unmatched, class_to_label = build_labeled_slides(
        features,
        clinical_records,
        args.label,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    labels_path = output_dir / f"{args.dataset_name}_labels.csv"
    unmatched_path = output_dir / f"{args.dataset_name}_unmatched.csv"
    manifest_path = output_dir / f"{args.dataset_name}_labels_manifest.json"
    labeled.to_csv(labels_path, index=False)
    unmatched[
        ["slide_id", "patient_id", "feature_path"]
    ].to_csv(unmatched_path, index=False)

    counts = {}
    for class_name, group in labeled.groupby("class_name", sort=False):
        counts[class_name] = {
            "label": int(group["label"].iloc[0]),
            "patients": int(group["patient_id"].nunique()),
            "slides": len(group),
        }
    manifest = {
        "dataset": args.dataset_name,
        "study_id": args.study_id,
        "attribute_id": args.attribute_id,
        "source_url": source_url,
        "feature_dir": str(Path(args.feature_dir).resolve()),
        "num_feature_slides": len(features),
        "num_feature_patients": int(features["patient_id"].nunique()),
        "num_labeled_slides": len(labeled),
        "num_labeled_patients": int(labeled["patient_id"].nunique()),
        "num_unmatched_slides": len(unmatched),
        "class_to_label": class_to_label,
        "counts": counts,
        "labels_path": str(labels_path.resolve()),
        "labels_sha256": file_sha256(labels_path),
        "unmatched_path": str(unmatched_path.resolve()),
        "unmatched_sha256": file_sha256(unmatched_path),
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
