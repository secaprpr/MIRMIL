from pathlib import Path

import pytest

from experiments.prepare_tcga_nsclc import (
    build_link_directory,
    collect_features,
    patient_id_from_slide,
)


def test_patient_id_from_slide():
    slide_id = "TCGA-55-6984-01Z-00-DX1.example"
    assert patient_id_from_slide(slide_id) == "TCGA-55-6984"


def test_patient_id_rejects_non_tcga_name():
    with pytest.raises(ValueError, match="Cannot extract"):
        patient_id_from_slide("slide_001")


def test_collect_and_link_features(tmp_path):
    luad_dir = tmp_path / "luad"
    lusc_dir = tmp_path / "lusc"
    luad_dir.mkdir()
    lusc_dir.mkdir()
    luad_feature = luad_dir / "TCGA-AA-0001-01Z-00-DX1.a.pt"
    lusc_feature = lusc_dir / "TCGA-BB-0002-01Z-00-DX1.b.pt"
    luad_feature.write_bytes(b"luad")
    lusc_feature.write_bytes(b"lusc")

    frame = collect_features(
        [("LUAD", luad_dir), ("LUSC", lusc_dir)], ".pt"
    )
    link_dir = build_link_directory(frame, tmp_path / "output", ".pt")

    assert frame["label"].tolist() == [0, 1]
    assert frame["patient_id"].tolist() == ["TCGA-AA-0001", "TCGA-BB-0002"]
    assert (link_dir / luad_feature.name).resolve() == luad_feature.resolve()
    assert (link_dir / lusc_feature.name).resolve() == lusc_feature.resolve()
