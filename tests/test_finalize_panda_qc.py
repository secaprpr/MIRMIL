import pandas as pd
import pytest

from experiments.finalize_panda_qc import finalize_assignment


def test_qc_exclusion_preserves_all_other_split_assignments():
    assignment = pd.DataFrame(
        {
            "image_id": ["a", "b", "c"],
            "split": ["train", "val", "test"],
        }
    )
    exclusions = pd.DataFrame(
        {"image_id": ["c"], "reason": ["blank_wsi"]}
    )
    retained = finalize_assignment(assignment, exclusions)
    assert retained.to_dict("records") == [
        {"image_id": "a", "split": "train"},
        {"image_id": "b", "split": "val"},
    ]


def test_qc_exclusion_rejects_unknown_image():
    assignment = pd.DataFrame({"image_id": ["a"], "split": ["train"]})
    exclusions = pd.DataFrame(
        {"image_id": ["missing"], "reason": ["blank_wsi"]}
    )
    with pytest.raises(ValueError, match="unknown"):
        finalize_assignment(assignment, exclusions)
