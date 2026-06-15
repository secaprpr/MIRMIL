import pandas as pd
import pytest

from experiments.prepare_visible_cv import (
    build_wide_split,
    create_visible_folds,
    infer_group_id,
    load_visible_pool,
)


def test_infer_group_id_uses_tcga_case():
    path = "/features/TCGA-A6-2674-01Z-00-DX1.some-id.pt"
    assert infer_group_id(path) == "TCGA-A6-2674"


def test_visible_folds_keep_patients_disjoint(tmp_path):
    rows = []
    for label in range(2):
        for patient in range(4):
            case = f"TCGA-AA-{label}{patient:03d}"
            rows.append((f"/features/{case}-01Z-slide.pt", label))
            if patient == 0:
                rows.append((f"/features/{case}-02Z-slide.pt", label))
    pool = pd.DataFrame(rows, columns=["slide_path", "label"])
    pool["group_id"] = pool["slide_path"].map(infer_group_id)

    folds = create_visible_folds(pool, n_splits=2, seed=2024)

    assert len(folds) == 2
    for train, val in folds:
        assert set(train["group_id"]).isdisjoint(val["group_id"])
        wide = build_wide_split(train, val)
        assert wide["test_slide_path"].isna().all()
        assert wide["test_label"].isna().all()


def test_load_visible_pool_rejects_test_examples(tmp_path):
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        {
            "train_slide_path": ["/features/train.pt"],
            "train_label": [0],
            "val_slide_path": ["/features/val.pt"],
            "val_label": [1],
            "test_slide_path": ["/features/test.pt"],
            "test_label": [0],
        }
    ).to_csv(split_path, index=False)

    with pytest.raises(ValueError, match="must not contain test"):
        load_visible_pool(split_path)
