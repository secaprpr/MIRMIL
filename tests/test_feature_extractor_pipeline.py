import sys
from pathlib import Path

import h5py
import numpy as np
import torch


FEATURE_EXTRACTOR = (
    Path(__file__).resolve().parents[1] / "feature_extractor"
)
if str(FEATURE_EXTRACTOR) not in sys.path:
    sys.path.insert(0, str(FEATURE_EXTRACTOR))

from create_h5_patches import adjust_coords_order, valid_coord_h5
from create_pt_features import valid_feature_outputs

sys.path.remove(str(FEATURE_EXTRACTOR))


def test_adjust_coords_order_sorts_without_shifting_or_losing_attrs(
    tmp_path,
):
    path = tmp_path / "coords.h5"
    original = np.array([[256, 512], [0, 256], [0, 0]], dtype=np.int64)
    with h5py.File(path, "w") as file:
        dataset = file.create_dataset("coords", data=original)
        dataset.attrs["patch_size"] = 256
        file.attrs["pipeline"] = "panda"

    adjust_coords_order(path)

    with h5py.File(path, "r") as file:
        assert np.array_equal(
            file["coords"][:],
            np.array([[0, 0], [0, 256], [256, 512]]),
        )
        assert file["coords"].attrs["patch_size"] == 256
        assert file.attrs["pipeline"] == "panda"


def test_valid_coord_h5_rejects_empty_and_malformed_files(tmp_path):
    valid = tmp_path / "valid.h5"
    empty = tmp_path / "empty.h5"
    malformed = tmp_path / "malformed.h5"
    with h5py.File(valid, "w") as file:
        file.create_dataset("coords", data=np.array([[0, 0]]))
    with h5py.File(empty, "w") as file:
        file.create_dataset("coords", data=np.empty((0, 2)))
    with h5py.File(malformed, "w") as file:
        file.create_dataset("features", data=np.ones((1, 2)))

    assert valid_coord_h5(valid)
    assert not valid_coord_h5(empty)
    assert not valid_coord_h5(malformed)
    assert not valid_coord_h5(tmp_path / "missing.h5")


def test_valid_feature_outputs_requires_matching_features_and_coords(tmp_path):
    h5_path = tmp_path / "features.h5"
    pt_path = tmp_path / "features.pt"
    features = np.ones((3, 8), dtype=np.float32)
    with h5py.File(h5_path, "w") as file:
        file.create_dataset("features", data=features)
        file.create_dataset("coords", data=np.ones((3, 2)))
    torch.save(torch.from_numpy(features), pt_path)
    assert valid_feature_outputs(h5_path, pt_path)

    torch.save(torch.ones(2, 8), pt_path)
    assert not valid_feature_outputs(h5_path, pt_path)
