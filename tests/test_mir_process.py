import torch

from process.MIR_MIL.process_mir_mil import ExponentialMovingAverage


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
