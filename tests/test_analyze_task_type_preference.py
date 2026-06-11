import pandas as pd
import pytest

from experiments.analyze_task_type_preference import (
    binomial_sign_pvalue,
    cohort_preference,
    hierarchical_bootstrap,
)


def write_results(path, ot_values, mo_values):
    rows = []
    for seed, (ot_value, mo_value) in enumerate(
        zip(ot_values, mo_values), start=2024
    ):
        rows.extend(
            [
                {"seed": seed, "model": "OT_MIL", "macro_auc": ot_value},
                {"seed": seed, "model": "MO_MIL", "macro_auc": mo_value},
            ]
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def test_cohort_preference_pairs_training_seeds(tmp_path):
    binary = tmp_path / "binary.csv"
    multiclass = tmp_path / "multiclass.csv"
    write_results(binary, [0.7, 0.8], [0.8, 0.8])
    write_results(multiclass, [0.9, 0.8], [0.8, 0.75])

    result = cohort_preference(
        "cohort", binary, multiclass, "macro_auc"
    )

    assert result["binary_differences"] == pytest.approx([-0.1, 0.0])
    assert result["multiclass_differences"] == pytest.approx([0.1, 0.05])
    assert result["task_type_preferences"] == pytest.approx([0.2, 0.05])


def test_hierarchical_bootstrap_and_sign_test_are_deterministic():
    cohorts = [
        {"task_type_preferences": [0.1, 0.2, 0.3]},
        {"task_type_preferences": [0.05, 0.1, 0.15]},
    ]
    first = hierarchical_bootstrap(cohorts, 100, 7)
    second = hierarchical_bootstrap(cohorts, 100, 7)

    assert (first == second).all()
    assert (first > 0).all()
    assert binomial_sign_pvalue(2, 2) == 0.25
