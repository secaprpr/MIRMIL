import h5py
import numpy as np
import pytest

from experiments.prepare_tcga_projects import (
    build_link_directory,
    collect_projects,
    inspect_h5,
    parse_project,
    patient_id_from_slide,
)


def write_h5(path, shape=(1, 4, 1536)):
    with h5py.File(path, "w") as file:
        file.create_dataset("features", data=np.zeros(shape, dtype=np.float32))


def test_parse_project():
    name, path = parse_project("KIRC=/data/kirc")
    assert name == "KIRC"
    assert str(path) == "/data/kirc"


def test_patient_id_from_slide():
    slide = "TCGA-AA-0001-01Z-00-DX1.example"
    assert patient_id_from_slide(slide) == "TCGA-AA-0001"


def test_inspect_uni_h5(tmp_path):
    feature = tmp_path / "feature.h5"
    write_h5(feature)
    assert inspect_h5(feature) == (4, 1536)


def test_collect_projects_and_links(tmp_path):
    kirc = tmp_path / "kirc"
    kirp = tmp_path / "kirp"
    kirc.mkdir()
    kirp.mkdir()
    kirc_feature = kirc / "TCGA-AA-0001-01Z-00-DX1.a.h5"
    kirp_feature = kirp / "TCGA-BB-0002-01Z-00-DX1.b.h5"
    write_h5(kirc_feature)
    write_h5(kirp_feature)

    frame = collect_projects([("KIRC", kirc), ("KIRP", kirp)])
    link_dir = build_link_directory(frame, tmp_path / "output", ".h5")

    assert frame["label"].tolist() == [0, 1]
    assert frame["feature_dim"].tolist() == [1536, 1536]
    assert (link_dir / kirc_feature.name).resolve() == kirc_feature.resolve()
    assert (link_dir / kirp_feature.name).resolve() == kirp_feature.resolve()


def test_collect_projects_rejects_dimension_mismatch(tmp_path):
    kirc = tmp_path / "kirc"
    kirp = tmp_path / "kirp"
    kirc.mkdir()
    kirp.mkdir()
    write_h5(kirc / "TCGA-AA-0001-01Z-00-DX1.a.h5")
    write_h5(
        kirp / "TCGA-BB-0002-01Z-00-DX1.b.h5",
        shape=(1, 4, 1024),
    )

    with pytest.raises(ValueError, match="Inconsistent feature dimensions"):
        collect_projects([("KIRC", kirc), ("KIRP", kirp)])
