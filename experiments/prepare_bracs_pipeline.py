"""Materialize the official BRACS split for extracted MIL features."""

import argparse
import hashlib
import json
import os
from pathlib import Path
from xml.etree import ElementTree
from zipfile import ZipFile

import pandas as pd


LABEL_ORDER = ("N", "PB", "UDH", "FEA", "ADH", "DCIS", "IC")
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABEL_ORDER)}
SET_TO_SPLIT = {"Training": "train", "Validation": "val", "Testing": "test"}
XML_NAMESPACE = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_label_rows(workbook):
    """Read WSI_Information without adding an openpyxl dependency."""
    with ZipFile(workbook) as archive:
        strings_root = ElementTree.fromstring(
            archive.read("xl/sharedStrings.xml")
        )
        strings = [
            "".join(node.text or "" for node in item.iter(XML_NAMESPACE + "t"))
            for item in strings_root.findall(XML_NAMESPACE + "si")
        ]
        sheet = ElementTree.fromstring(
            archive.read("xl/worksheets/sheet1.xml")
        )
        records = []
        for row in sheet.iter(XML_NAMESPACE + "row"):
            values = {}
            for cell in row.findall(XML_NAMESPACE + "c"):
                column = "".join(
                    character
                    for character in cell.attrib["r"]
                    if character.isalpha()
                )
                value_node = cell.find(XML_NAMESPACE + "v")
                value = None if value_node is None else value_node.text
                if cell.attrib.get("t") == "s" and value is not None:
                    value = strings[int(value)]
                values[column] = value
            if values.get("A") and values["A"] != "WSI Filename":
                records.append(
                    {
                        "slide_id": values["A"].strip(),
                        "patient_id": str(values["B"]).strip(),
                        "label_name": values["D"].strip(),
                        "official_set": values["E"].strip(),
                    }
                )
    return pd.DataFrame(records)


def build_assignment(workbook, wsi_root):
    labels = read_label_rows(workbook)
    if len(labels) != 547 or labels["slide_id"].duplicated().any():
        raise ValueError("BRACS.xlsx must contain 547 unique WSI records")
    unknown_labels = set(labels["label_name"]) - set(LABEL_TO_INDEX)
    unknown_sets = set(labels["official_set"]) - set(SET_TO_SPLIT)
    if unknown_labels or unknown_sets:
        raise ValueError(
            f"unknown labels={sorted(unknown_labels)}, sets={sorted(unknown_sets)}"
        )
    discovered = {path.stem: path.resolve() for path in wsi_root.rglob("*.svs")}
    if set(discovered) != set(labels["slide_id"]):
        raise ValueError(
            f"label/WSI mismatch: missing={sorted(set(labels.slide_id)-set(discovered))}, "
            f"extra={sorted(set(discovered)-set(labels.slide_id))}"
        )
    labels["split"] = labels["official_set"].map(SET_TO_SPLIT)
    labels["label"] = labels["label_name"].map(LABEL_TO_INDEX).astype(int)
    labels["wsi_path"] = labels["slide_id"].map(
        lambda slide_id: str(discovered[slide_id])
    )
    labels["wsi_size"] = labels["slide_id"].map(
        lambda slide_id: discovered[slide_id].stat().st_size
    )
    return labels.sort_values(["split", "slide_id"]).reset_index(drop=True)


def wide_split(assignment, feature_dir, storage, include_test):
    columns = {}
    for split in ("train", "val", "test"):
        frame = assignment[assignment["split"] == split]
        if split == "test" and not include_test:
            frame = frame.iloc[:0]
        columns[f"{split}_slide_path"] = pd.Series(
            [
                str(feature_dir / f"{storage}_files" / f"{slide_id}.{storage}")
                for slide_id in frame["slide_id"]
            ],
            dtype="object",
        )
        columns[f"{split}_label"] = pd.Series(
            frame["label"].tolist(), dtype="Int64"
        )
    return pd.DataFrame(columns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workbook", type=Path, required=True)
    parser.add_argument("--wsi-root", type=Path, required=True)
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--require-features", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignment = build_assignment(args.workbook, args.wsi_root)
    source_path = args.output_dir / "bracs_source_manifest.csv"
    assignment_path = args.output_dir / "bracs_official_split_assignment.csv"
    patch_manifest_path = args.output_dir / "bracs_patch_manifest.csv"
    assignment.to_csv(source_path, index=False)
    assignment.to_csv(assignment_path, index=False)

    patch_records = []
    for row in assignment.itertuples(index=False):
        path = (
            args.feature_root / "patches_level0_256" / "patches"
            / f"{row.slide_id}.h5"
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        patch_records.append(
            {"slide_id": row.slide_id, "coord_h5": str(path.resolve())}
        )
    pd.DataFrame(patch_records).to_csv(patch_manifest_path, index=False)

    feature_metadata = {}
    for feature in ("r50", "uni"):
        feature_metadata[feature] = {}
        feature_dir = (args.feature_root / feature).resolve()
        for storage in ("pt", "h5"):
            for include_test, suffix in (
                (True, "full"), (False, "train_val")
            ):
                frame = wide_split(
                    assignment, feature_dir, storage, include_test
                )
                storage_suffix = "" if storage == "pt" else "_h5"
                path = args.output_dir / (
                    f"BRACS_{feature}_split_official_{suffix}{storage_suffix}.csv"
                )
                if args.require_features:
                    paths = [
                        value
                        for column in frame.columns
                        if column.endswith("_slide_path")
                        for value in frame[column].dropna()
                    ]
                    missing = [value for value in paths if not os.path.isfile(value)]
                    if missing:
                        raise FileNotFoundError(
                            f"{len(missing)} missing {feature}/{storage} features"
                        )
                frame.to_csv(path, index=False)
                feature_metadata[feature][f"{storage}_{suffix}"] = {
                    "path": str(path.resolve()),
                    "sha256": file_sha256(path),
                }

    counts = (
        assignment.groupby(["split", "label_name"]).size()
        .rename("count").reset_index()
    )
    manifest = {
        "schema_version": 1,
        "dataset": "BRACS",
        "split_source": "official BRACS.xlsx WSI_Information",
        "label_order": list(LABEL_ORDER),
        "workbook": str(args.workbook.resolve()),
        "workbook_sha256": file_sha256(args.workbook),
        "source_manifest": str(source_path.resolve()),
        "source_manifest_sha256": file_sha256(source_path),
        "patch_manifest": str(patch_manifest_path.resolve()),
        "patch_manifest_sha256": file_sha256(patch_manifest_path),
        "counts": {
            split: int((assignment["split"] == split).sum())
            for split in ("train", "val", "test")
        },
        "class_counts": {
            f"{row.split}__{row.label_name}": int(row.count)
            for row in counts.itertuples(index=False)
        },
        "features": feature_metadata,
        "known_official_patient_overlap": {
            "patient_id": "67",
            "train": ["BRACS_1631", "BRACS_1633", "BRACS_1634"],
            "val": ["BRACS_1614", "BRACS_1632"],
        },
    }
    manifest_path = args.output_dir / "bracs_pipeline_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
