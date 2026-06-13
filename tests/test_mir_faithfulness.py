import numpy as np
import pytest

from experiments.evaluate_mir_faithfulness import (
    correlation,
    faithfulness_metrics,
)


def test_faithfulness_metrics_identical_scores():
    values = np.array([-2.0, -0.5, 0.2, 3.0])

    metrics = faithfulness_metrics(values, values, topk=2)

    assert metrics["pearson"] == pytest.approx(1.0)
    assert metrics["spearman"] == pytest.approx(1.0)
    assert metrics["mse"] == 0.0
    assert metrics["mae"] == 0.0
    assert metrics["topk_overlap"] == 1.0


def test_correlation_returns_nan_for_constant_input():
    assert np.isnan(correlation([1, 1], [1, 2]))
