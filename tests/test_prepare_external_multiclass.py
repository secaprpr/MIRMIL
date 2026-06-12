import json

import pandas as pd
import torch

from experiments.prepare_external_multiclass import (
    load_cached_projects,
    validate_combined_frame,
    write_visible_split,
)


def test_load_cached_projects_maps_manifest_records(tmp_path):
    cache = tmp_path / "slide.pt"
    torch.save(torch.zeros(4, 8), cache)
    labels = tmp_path / "labels.csv"
    pd.DataFrame(
        [
            {
                "slide_id": "TCGA-AA-0001-01Z-00-DX1.example",
                "patient_id": "TCGA-AA-0001",
                "label": 0,
                "class_name": "A",
            }
        ]
    ).to_csv(labels, index=False)
    manifest = tmp_path / "cache.json"
    manifest.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_path": (
                            "/source/TCGA-AA-0001-01Z-00-DX1.example.h5"
                        ),
                        "cached_path": str(cache),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    frame = load_cached_projects(labels, manifest)

    assert frame["slide_path"].tolist() == [str(cache)]


def test_write_visible_split_removes_test_data(tmp_path):
    full = tmp_path / "full.csv"
    visible = tmp_path / "visible.csv"
    pd.DataFrame(
        {
            "train_slide_path": ["train.pt"],
            "train_label": [0],
            "val_slide_path": ["val.pt"],
            "val_label": [0],
            "test_slide_path": ["test.pt"],
            "test_label": [0],
        }
    ).to_csv(full, index=False)

    write_visible_split(full, visible)
    result = pd.read_csv(visible)

    assert result["test_slide_path"].isna().all()
    assert result["test_label"].isna().all()


def test_validate_combined_frame_rejects_patient_label_leakage():
    frame = pd.DataFrame(
        {
            "slide_id": ["slide-a", "slide-b"],
            "patient_id": ["patient", "patient"],
            "label": [0, 1],
            "class_name": ["A", "B"],
            "slide_path": ["a.pt", "b.pt"],
        }
    )

    try:
        validate_combined_frame(frame)
    except ValueError as error:
        assert "multiple classes" in str(error)
    else:
        raise AssertionError("Expected patient label leakage to be rejected")
