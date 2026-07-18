import math

import pytest
import torch

from modules.MIR_MIL.ordinal_risk_mil import OrdinalRiskMIL


def make_model(aggregation="entropic", prediction_head="ordinal"):
    return OrdinalRiskMIL(
        in_dim=8,
        num_classes=3,
        hidden_dim=12,
        dropout=0.0,
        aggregation=aggregation,
        prediction_head=prediction_head,
        risk_temperature=0.5,
    )


def test_forward_is_permutation_invariant():
    torch.manual_seed(3)
    model = make_model().eval()
    bag = torch.randn(17, 8)
    first = model(bag, return_state=True)
    second = model(bag[torch.randperm(17)], return_state=True)
    assert torch.allclose(first["logits"], second["logits"], atol=1e-6)
    assert torch.allclose(first["risk"], second["risk"], atol=1e-6)


def test_entropic_risk_is_invariant_to_exact_bag_duplication():
    model = make_model()
    scores = torch.tensor([-1.0, 0.2, 1.7, 0.4])
    original = model.aggregate_scores(scores)
    duplicated = model.aggregate_scores(scores.repeat(5))
    assert torch.allclose(original, duplicated, atol=1e-6)


def test_mean_and_entropic_risk_match_on_constant_measure():
    mean_model = make_model(aggregation="mean")
    risk_model = make_model(aggregation="entropic")
    scores = torch.full((11,), 2.25)
    assert mean_model.aggregate_scores(scores).item() == pytest.approx(2.25)
    assert risk_model.aggregate_scores(scores).item() == pytest.approx(
        2.25, abs=1e-6
    )


def test_ordered_boundaries_produce_valid_probabilities():
    model = make_model()
    thresholds = model.ordered_thresholds()
    probabilities = model.ordinal_probabilities(torch.tensor(0.3))
    assert torch.all(thresholds[1:] > thresholds[:-1])
    assert torch.all(probabilities > 0)
    assert probabilities.sum().item() == pytest.approx(1.0)


def test_initial_threshold_gap_is_centered_and_configurable():
    model = OrdinalRiskMIL(
        in_dim=8,
        num_classes=3,
        hidden_dim=12,
        dropout=0.0,
        initial_threshold_center=0.25,
        initial_threshold_gap=1.5,
    )
    thresholds = model.ordered_thresholds().detach()
    assert thresholds.mean().item() == pytest.approx(0.25, abs=1e-4)
    assert (thresholds[1] - thresholds[0]).item() == pytest.approx(
        1.5001, abs=1e-4
    )


def test_severity_shift_monotonically_increases_risk_and_survival():
    model = make_model()
    scores = torch.tensor([-0.7, 0.1, 0.5, 1.2])
    low_risk = model.aggregate_scores(scores)
    high_risk = model.aggregate_scores(scores + 0.8)
    thresholds = model.ordered_thresholds().detach()
    low_survival = torch.sigmoid(low_risk - thresholds)
    high_survival = torch.sigmoid(high_risk - thresholds)
    assert high_risk > low_risk
    assert torch.all(high_survival > low_survival)


@pytest.mark.parametrize(
    "aggregation,prediction_head",
    [
        ("mean", "ordinal"),
        ("entropic", "softmax"),
        ("entropic", "ordinal"),
    ],
)
def test_ablation_variants_have_finite_gradients(
    aggregation, prediction_head
):
    torch.manual_seed(5)
    model = make_model(aggregation, prediction_head)
    bag = torch.randn(1, 13, 8)
    label = torch.tensor([1])
    output, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()
    assert output["logits"].shape == (1, 3)
    assert math.isfinite(losses["loss"].item())
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
