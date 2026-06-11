import numpy as np
import pandas as pd

from experiments.evaluate_checkpoints import json_safe
from experiments.paired_bootstrap import (
    macro_auc,
    metric_score,
    paired_frames,
    stratified_indices,
)


def test_stratified_indices_preserve_class_counts():
    labels = np.array([0, 0, 1, 1, 1, 2])
    indices = stratified_indices(labels, np.random.default_rng(7))
    sampled_labels = labels[indices]

    assert len(indices) == len(labels)
    assert pd.Series(sampled_labels).value_counts().sort_index().to_dict() == {
        0: 2,
        1: 3,
        2: 1,
    }


def test_json_safe_converts_numpy_values():
    value = {
        "scalar": np.float64(0.5),
        "matrix": np.array([[1, 2], [3, 4]]),
    }

    assert json_safe(value) == {
        "scalar": 0.5,
        "matrix": [[1, 2], [3, 4]],
    }


def test_macro_auc_supports_binary_probabilities():
    labels = np.array([0, 0, 1, 1])
    probabilities = np.array(
        [[0.9, 0.1], [0.8, 0.2], [0.2, 0.8], [0.1, 0.9]]
    )

    assert macro_auc(labels, probabilities) == 1.0


def test_metric_score_supports_classification_metrics():
    labels = np.array([0, 0, 1, 1, 2, 2])
    probabilities = np.array(
        [
            [0.9, 0.1, 0.0],
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.7, 0.2],
            [0.1, 0.2, 0.7],
            [0.1, 0.1, 0.8],
        ]
    )

    assert metric_score("accuracy", labels, probabilities) == 1.0
    assert metric_score("balanced_accuracy", labels, probabilities) == 1.0
    assert metric_score("macro_f1", labels, probabilities) == 1.0


def test_paired_frames_supports_group_identifier(tmp_path):
    common = {
        "case_id": ["case_a", "case_b"],
        "label": [0, 1],
        "seed": [2024, 2024],
        "budget": [4096, 4096],
    }
    for model, probabilities in (
        ("OT_MIL", ([0.8, 0.2], [0.1, 0.9])),
        ("MO_MIL", ([0.7, 0.3], [0.2, 0.8])),
    ):
        frame = pd.DataFrame(common)
        frame["prob_0"] = probabilities[0]
        frame["prob_1"] = probabilities[1]
        frame.to_csv(
            tmp_path / f"{model}_seed2024_budget4096.csv", index=False
        )

    pairs = paired_frames(tmp_path, 4096, id_column="case_id")
    assert len(pairs) == 1
    assert pairs[0][1]["case_id"].tolist() == ["case_a", "case_b"]
