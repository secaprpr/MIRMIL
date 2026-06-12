import json

import pandas as pd
import pytest

from experiments.aggregate_group_predictions import (
    aggregate_prediction_frame,
    build_group_mapping,
    evaluate_grouped_frame,
)


def test_group_predictions_are_mean_aggregated(tmp_path):
    source_a = tmp_path / "a.h5"
    source_b = tmp_path / "b.h5"
    source_c = tmp_path / "c.h5"
    cached_a = tmp_path / "a.pt"
    cached_b = tmp_path / "b.pt"
    cached_c = tmp_path / "c.pt"
    assignments = pd.DataFrame(
        {
            "feature_path": [source_a, source_b, source_c],
            "case_id": ["case_1", "case_1", "case_2"],
            "label": [1, 1, 0],
            "split": ["test", "test", "test"],
        }
    )
    assignments_path = tmp_path / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_path": str(source),
                        "cached_path": str(cached),
                    }
                    for source, cached in (
                        (source_a, cached_a),
                        (source_b, cached_b),
                        (source_c, cached_c),
                    )
                ]
            }
        ),
        encoding="utf-8",
    )
    predictions = pd.DataFrame(
        {
            "slide_path": [cached_a, cached_b, cached_c],
            "label": [1, 1, 0],
            "prob_0": [0.2, 0.4, 0.8],
            "prob_1": [0.8, 0.6, 0.2],
        }
    )

    mapping = build_group_mapping(assignments_path, manifest_path)
    grouped = aggregate_prediction_frame(predictions, mapping)
    metrics = evaluate_grouped_frame(grouped)

    assert grouped["case_id"].tolist() == ["case_1", "case_2"]
    assert grouped["num_slides"].tolist() == [2, 1]
    assert grouped["prob_1"].tolist() == pytest.approx([0.7, 0.2])
    assert metrics["acc"] == 1.0


def test_conflicting_group_labels_are_rejected(tmp_path):
    assignments = pd.DataFrame(
        {
            "feature_path": [tmp_path / "a.h5", tmp_path / "b.h5"],
            "case_id": ["case", "case"],
            "label": [0, 1],
            "split": ["test", "test"],
        }
    )
    assignments_path = tmp_path / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "source_path": str(tmp_path / "a.h5"),
                        "cached_path": str(tmp_path / "a.pt"),
                    },
                    {
                        "source_path": str(tmp_path / "b.h5"),
                        "cached_path": str(tmp_path / "b.pt"),
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="conflicting labels"):
        build_group_mapping(assignments_path, manifest_path)


def test_direct_slide_patient_assignments_need_no_manifest(tmp_path):
    assignments = pd.DataFrame(
        {
            "slide_path": [tmp_path / "a.pt", tmp_path / "b.pt"],
            "patient_id": ["patient_1", "patient_2"],
            "label": [0, 1],
            "split": ["test", "train"],
        }
    )
    assignments_path = tmp_path / "assignments.csv"
    assignments.to_csv(assignments_path, index=False)

    mapping = build_group_mapping(assignments_path)

    assert mapping["slide_path"].tolist() == [
        str((tmp_path / "a.pt").resolve())
    ]
    assert mapping["case_id"].tolist() == ["patient_1"]
