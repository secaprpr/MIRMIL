import pandas as pd

from experiments.paired_bootstrap import paired_frames


def write_predictions(path, model, seed, probabilities):
    frame = pd.DataFrame(
        {
            "slide_path": ["a.pt", "b.pt"],
            "label": [0, 1],
            "model": [model, model],
            "variant": [model, model],
            "seed": [seed, seed],
            "budget": [512, 512],
            "prob_0": [row[0] for row in probabilities],
            "prob_1": [row[1] for row in probabilities],
        }
    )
    frame.to_csv(path / f"{model}_seed{seed}_budget512.csv", index=False)


def test_paired_frames_supports_custom_models(tmp_path):
    write_predictions(tmp_path, "MIR_MIL", 2024, [(0.8, 0.2), (0.1, 0.9)])
    write_predictions(tmp_path, "MO_MIL", 2024, [(0.7, 0.3), (0.2, 0.8)])

    pairs = paired_frames(
        tmp_path,
        512,
        model_a="MIR_MIL",
        model_b="MO_MIL",
    )

    seed, frame, probability_columns = pairs[0]
    assert seed == 2024
    assert probability_columns == ["prob_0", "prob_1"]
    assert list(frame["prob_0_a"]) == [0.8, 0.1]
    assert list(frame["prob_0_b"]) == [0.7, 0.2]


def test_paired_frames_keeps_ot_mo_defaults(tmp_path):
    write_predictions(tmp_path, "OT_MIL", 2025, [(0.8, 0.2), (0.1, 0.9)])
    write_predictions(tmp_path, "MO_MIL", 2025, [(0.7, 0.3), (0.2, 0.8)])

    pairs = paired_frames(tmp_path, 512)

    assert pairs[0][0] == 2025
