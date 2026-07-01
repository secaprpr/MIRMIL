from pathlib import Path

import pandas as pd

from experiments.prepare_panda_pipeline import (
    build_manifest,
    stratified_assignment,
    wide_feature_split,
)


def synthetic_panda(tmp_path):
    rows = []
    wsi_dir = tmp_path / "wsi"
    wsi_dir.mkdir()
    for provider in ("a", "b"):
        for grade in range(2):
            for index in range(10):
                image_id = f"{provider}_{grade}_{index}"
                (wsi_dir / f"{image_id}.tiff").write_bytes(b"tiff")
                rows.append(
                    {
                        "image_id": image_id,
                        "data_provider": provider,
                        "isup_grade": grade,
                    }
                )
    labels = tmp_path / "labels.csv"
    pd.DataFrame(rows).to_csv(labels, index=False)
    return labels, wsi_dir


def test_manifest_and_split_are_complete_and_deterministic(tmp_path):
    labels, wsi_dir = synthetic_panda(tmp_path)
    manifest = build_manifest(labels, wsi_dir)
    first = stratified_assignment(manifest, seed=2024)
    second = stratified_assignment(manifest, seed=2024)
    assert len(manifest) == 40
    assert first.equals(second)
    assert first["image_id"].nunique() == 40
    assert first["split"].value_counts().to_dict() == {
        "train": 24,
        "val": 8,
        "test": 8,
    }
    assert (
        first.groupby(["split", "data_provider", "isup_grade"])
        .size()
        .min()
        > 0
    )


def test_wide_feature_split_uses_one_encoder_root(tmp_path):
    labels, wsi_dir = synthetic_panda(tmp_path)
    assignment = stratified_assignment(
        build_manifest(labels, wsi_dir), seed=2024
    )
    feature_dir = tmp_path / "uni"
    wide = wide_feature_split(
        assignment, feature_dir, include_test=False
    )
    assert len(wide["train_slide_path"].dropna()) == 24
    assert len(wide["val_slide_path"].dropna()) == 8
    assert wide["test_slide_path"].dropna().empty
    assert all(
        str(feature_dir.resolve()) in path
        for path in wide["train_slide_path"].dropna()
    )
