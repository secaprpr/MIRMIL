from pathlib import Path

import pandas as pd
import pytest

from experiments.prepare_ucec_multitask import (
    attach_labels,
    build_case_frame,
    collect_slides,
    parse_mantis_scores,
    parse_patient_subtypes,
    patient_id_from_filename,
    sample_type_from_filename,
    shared_multitask_split,
)


def test_tcga_identifiers_are_parsed():
    filename = "TCGA-A5-A0G1-01Z-00-DX1.example.h5"
    assert patient_id_from_filename(filename) == "TCGA-A5-A0G1"
    assert sample_type_from_filename(filename) == "01"


def test_clinical_records_are_parsed():
    patient_records = [
        {
            "patientId": "TCGA-AA-0001",
            "clinicalAttributeId": "SUBTYPE",
            "value": "UCEC_POLE",
        }
    ]
    sample_records = [
        {
            "patientId": "TCGA-AA-0001",
            "clinicalAttributeId": "MSI_SCORE_MANTIS",
            "value": "0.72",
        },
        {
            "patientId": "TCGA-AA-0001",
            "clinicalAttributeId": "MSI_SCORE_MANTIS",
            "value": "0.70",
        },
    ]
    assert parse_patient_subtypes(patient_records) == {
        "TCGA-AA-0001": "UCEC_POLE"
    }
    assert parse_mantis_scores(sample_records) == {"TCGA-AA-0001": 0.72}


def test_collect_slides_excludes_nonprimary_samples(tmp_path):
    primary = tmp_path / "TCGA-AA-0001-01Z-00-DX1.primary.h5"
    metastatic = tmp_path / "TCGA-AA-0001-06Z-00-DX1.metastatic.h5"
    primary.touch()
    metastatic.touch()

    frame = collect_slides(tmp_path)
    assert frame["filename"].tolist() == [primary.name]
    assert len(collect_slides(tmp_path, primary_only=False)) == 2


def make_labelled_frame(tmp_path):
    rows = []
    subtypes = ("UCEC_POLE", "UCEC_MSI", "UCEC_CN_LOW", "UCEC_CN_HIGH")
    subtype_map = {}
    scores = {}
    for index in range(60):
        patient_id = f"TCGA-AA-{index:04d}"
        filename = f"{patient_id}-01Z-00-DX1.example.h5"
        path = tmp_path / filename
        path.touch()
        rows.append(
            {
                "filename": filename,
                "slide_id": path.stem,
                "case_id": patient_id,
                "sample_type_code": "01",
                "feature_path": str(path),
            }
        )
        subtype_map[patient_id] = subtypes[index % 4]
        scores[patient_id] = 0.7 if index % 3 == 0 else 0.3
    return attach_labels(pd.DataFrame(rows), subtype_map, scores)


def test_shared_split_is_deterministic_and_covers_all_classes(tmp_path):
    case_frame = build_case_frame(make_labelled_frame(tmp_path))
    first = shared_multitask_split(
        case_frame, seed=11, search_iterations=500
    )
    second = shared_multitask_split(
        case_frame, seed=11, search_iterations=500
    )

    assert first.equals(second)
    assert first["split"].value_counts().to_dict() == {
        "train": 36,
        "val": 12,
        "test": 12,
    }
    for column in ("msi_code", "subtype_code"):
        table = pd.crosstab(first["split"], first[column])
        assert (table > 0).all().all()


def test_unknown_subtype_is_rejected():
    records = [
        {
            "patientId": "TCGA-AA-0001",
            "clinicalAttributeId": "SUBTYPE",
            "value": "UNKNOWN",
        }
    ]
    with pytest.raises(ValueError, match="Unexpected UCEC subtype"):
        parse_patient_subtypes(records)
