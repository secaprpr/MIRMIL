import numpy as np
import pandas as pd

from experiments.evaluate_checkpoints import json_safe
from experiments.paired_bootstrap import macro_auc, stratified_indices


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
