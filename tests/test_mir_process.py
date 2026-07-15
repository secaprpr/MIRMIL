import torch

from process.MIR_MIL.process_mir_mil import (
    ExponentialMovingAverage,
    load_distillation_targets,
)


def test_ema_copies_first_update_then_smooths():
    model = torch.nn.Linear(2, 1)
    ema = ExponentialMovingAverage(model, decay=0.5)

    with torch.no_grad():
        model.weight.fill_(2.0)
        model.bias.fill_(1.0)
    ema.update(model)
    torch.testing.assert_close(ema.model.weight, model.weight)
    torch.testing.assert_close(ema.model.bias, model.bias)

    with torch.no_grad():
        model.weight.fill_(4.0)
        model.bias.fill_(3.0)
    ema.update(model)
    torch.testing.assert_close(
        ema.model.weight, torch.full_like(model.weight, 3.0)
    )
    torch.testing.assert_close(
        ema.model.bias, torch.full_like(model.bias, 2.0)
    )


def test_load_distillation_targets_aligns_by_slide_path(tmp_path):
    prob_path = tmp_path / "teacher.csv"
    prob_path.write_text(
        "slide_path,label,prob_0,prob_1,prob_2\n"
        "b.pt,1,0.1,0.8,0.1\n"
        "a.pt,0,2.0,1.0,1.0\n",
        encoding="utf-8",
    )

    class Dataset:
        slide_path_list = ["a.pt", "b.pt"]

    targets = load_distillation_targets(str(prob_path), Dataset(), 3)

    torch.testing.assert_close(
        targets,
        torch.tensor(
            [
                [0.5, 0.25, 0.25],
                [0.1, 0.8, 0.1],
            ]
        ),
    )
