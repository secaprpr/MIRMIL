import h5py
import numpy as np
import pandas as pd

from experiments.audit_panda_patches import (
    audit_coordinates,
    inspect_coordinate_file,
)


def write_coords(path, coords):
    with h5py.File(path, "w") as handle:
        dataset = handle.create_dataset("coords", data=np.asarray(coords))
        dataset.attrs["patch_size"] = 256
        dataset.attrs["patch_level"] = 0


def test_coordinate_audit_accepts_sorted_unique_level_zero_file(tmp_path):
    path = tmp_path / "case.h5"
    write_coords(path, [[0, 0], [0, 256], [256, 0]])
    result = inspect_coordinate_file(path)
    assert result["status"] == "ok"
    assert result["num_patches"] == 3
    assert len(result["sha256"]) == 64


def test_coordinate_audit_rejects_duplicates_and_unsorted_files(tmp_path):
    duplicate = tmp_path / "duplicate.h5"
    unsorted = tmp_path / "unsorted.h5"
    write_coords(duplicate, [[0, 0], [0, 0]])
    write_coords(unsorted, [[256, 0], [0, 0]])
    assert inspect_coordinate_file(duplicate)["status"] == "failed"
    assert inspect_coordinate_file(unsorted)["status"] == "failed"


def test_coordinate_audit_includes_missing_manifest_slide(tmp_path):
    patch_dir = tmp_path / "patches"
    (patch_dir / "patches").mkdir(parents=True)
    source_csv = tmp_path / "source.csv"
    pd.DataFrame({"wsi_path": ["/source/missing.tiff"]}).to_csv(
        source_csv, index=False
    )
    audit = audit_coordinates(source_csv, patch_dir)
    assert audit.iloc[0]["slide_id"] == "missing"
    assert audit.iloc[0]["status"] == "failed"
