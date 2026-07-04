from pathlib import Path

import pandas as pd

from experiments.prepare_bracs_pipeline import (
    LABEL_TO_INDEX,
    SET_TO_SPLIT,
    wide_split,
)


def test_bracs_label_order_and_official_split_paths(tmp_path):
    assignment = pd.DataFrame(
        {
            "slide_id": ["BRACS_1", "BRACS_2", "BRACS_3"],
            "split": ["train", "val", "test"],
            "label": [LABEL_TO_INDEX["N"], LABEL_TO_INDEX["ADH"], LABEL_TO_INDEX["IC"]],
        }
    )
    result = wide_split(assignment, Path("/features"), "pt", True)
    assert result.loc[0, "train_slide_path"] == "/features/pt_files/BRACS_1.pt"
    assert result.loc[0, "val_label"] == LABEL_TO_INDEX["ADH"]
    assert result.loc[0, "test_label"] == LABEL_TO_INDEX["IC"]
    assert SET_TO_SPLIT == {
        "Training": "train",
        "Validation": "val",
        "Testing": "test",
    }
