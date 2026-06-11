from pathlib import Path

import pandas as pd
import pytest

from experiments.prepare_coad_multitask import (
    TASKS,
    build_case_frame,
    load_and_validate_labels,
    select_recommended_slides,
    shared_multitask_split,
)


def make_slide_frame(tmp_path):
    rows = []
    for index in range(40):
        filename = f"TCGA-AA-{index:04d}-01Z-00-DX1.example.h5"
        (tmp_path / filename).touch()
        rows.append(
            {
                "filename": filename,
                "slide_id": filename.removesuffix(".example.h5"),
                "case_id": f"TCGA-AA-{index:04d}",
                "recommended_use_primary": "Yes",
                "msi_label": "MSS" if index % 2 == 0 else "MSI",
                "msi_code": index % 2,
                "cms_label": f"CMS{index % 4 + 1}",
                "cms_code": index % 4,
                "hmcings_label": ("CIN", "GS", "HM")[index % 3],
                "hmcings_code": index % 3,
            }
        )
    return pd.DataFrame(rows)


def test_shared_split_is_deterministic_and_covers_every_class(tmp_path):
    slide_frame = make_slide_frame(tmp_path)
    case_frame = build_case_frame(slide_frame)
    first = shared_multitask_split(
        case_frame, seed=17, search_iterations=500
    )
    second = shared_multitask_split(
        case_frame, seed=17, search_iterations=500
    )

    assert first.equals(second)
    assert first["split"].value_counts().to_dict() == {
        "train": 24,
        "val": 8,
        "test": 8,
    }
    for task in TASKS.values():
        table = pd.crosstab(first["split"], first[task["code_column"]])
        assert (table > 0).all().all()


def test_validation_rejects_missing_feature(tmp_path):
    frame = make_slide_frame(tmp_path)
    missing = tmp_path / frame.loc[0, "filename"]
    missing.unlink()
    labels_path = tmp_path / "labels.csv"
    frame.to_csv(labels_path, index=False)

    with pytest.raises(FileNotFoundError, match="feature files are missing"):
        load_and_validate_labels(labels_path, tmp_path)


def test_validation_rejects_conflicting_patient_labels(tmp_path):
    frame = make_slide_frame(tmp_path)
    duplicate = frame.iloc[0].copy()
    duplicate["filename"] = duplicate["filename"].replace(
        "DX1", "DX2"
    )
    duplicate["slide_id"] = duplicate["slide_id"].replace("DX1", "DX2")
    duplicate["msi_label"] = "MSI"
    duplicate["msi_code"] = 1
    (tmp_path / duplicate["filename"]).touch()
    frame = pd.concat([frame, duplicate.to_frame().T], ignore_index=True)
    labels_path = tmp_path / "labels.csv"
    frame.to_csv(labels_path, index=False)

    with pytest.raises(ValueError, match="conflicting msi labels"):
        load_and_validate_labels(labels_path, tmp_path)


def test_nonrecommended_slides_are_excluded():
    frame = pd.DataFrame(
        {"recommended_use_primary": ["Yes", "No", "Yes"]}
    )
    selected = select_recommended_slides(frame)
    assert selected.index.tolist() == [0, 2]
    assert len(select_recommended_slides(frame, recommended_only=False)) == 3
