import pandas as pd
import pytest

from experiments.prepare_cbioportal_subtype import (
    build_labeled_slides,
    parse_label_mapping,
    patient_id_from_slide,
)


def test_parse_label_mapping():
    assert parse_label_mapping("STAD_EBV=EBV") == ("STAD_EBV", "EBV")


def test_patient_id_from_slide():
    assert (
        patient_id_from_slide("TCGA-AA-0001-01Z-00-DX1.example")
        == "TCGA-AA-0001"
    )


def test_build_labeled_slides_filters_unmapped_labels():
    features = pd.DataFrame(
        {
            "slide_id": ["slide-a", "slide-b", "slide-c"],
            "patient_id": ["patient-a", "patient-b", "patient-c"],
            "feature_path": ["a.h5", "b.h5", "c.h5"],
        }
    )
    clinical = [
        {"patientId": "patient-a", "value": "SOURCE_A"},
        {"patientId": "patient-b", "value": "SOURCE_B"},
        {"patientId": "patient-c", "value": "EXCLUDED"},
    ]

    labeled, unmatched, class_to_label = build_labeled_slides(
        features,
        clinical,
        [("SOURCE_A", "A"), ("SOURCE_B", "B")],
    )

    assert labeled["label"].tolist() == [0, 1]
    assert unmatched["patient_id"].tolist() == ["patient-c"]
    assert class_to_label == {"A": 0, "B": 1}


def test_build_labeled_slides_rejects_duplicate_patient_labels():
    features = pd.DataFrame(
        {
            "slide_id": ["slide-a"],
            "patient_id": ["patient-a"],
            "feature_path": ["a.h5"],
        }
    )
    clinical = [
        {"patientId": "patient-a", "value": "SOURCE_A"},
        {"patientId": "patient-a", "value": "SOURCE_A"},
    ]

    with pytest.raises(ValueError, match="Duplicate clinical labels"):
        build_labeled_slides(
            features,
            clinical,
            [("SOURCE_A", "A")],
        )
