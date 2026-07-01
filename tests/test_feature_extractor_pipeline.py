import sys
from pathlib import Path

import h5py
import numpy as np


FEATURE_EXTRACTOR = (
    Path(__file__).resolve().parents[1] / "feature_extractor"
)
if str(FEATURE_EXTRACTOR) not in sys.path:
    sys.path.insert(0, str(FEATURE_EXTRACTOR))

from create_h5_patches import adjust_coords_order

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
