import math

import pytest
import torch

from modules.MIR_MIL.boundary_risk_mil import BoundaryRiskMIL


def make_model(aggregation="entropic"):
    return BoundaryRiskMIL(
        in_dim=8,
        num_classes=3,
        hidden_dim=12,
        dropout=0.0,
        aggregation=aggregation,
        risk_temperature=0.5,
        score_bound=2.0,
    )


def test_forward_is_permutation_invariant():
    torch.manual_seed(7)
    model = make_model().eval()
    bag = torch.randn(19, 8)
    first = model(bag, return_state=True)
    second = model(bag[torch.randperm(19)], return_state=True)
    assert torch.allclose(first["logits"], second["logits"], atol=1e-6)
    assert torch.allclose(
        first["boundary_risks"], second["boundary_risks"], atol=1e-6
    )


def test_entropic_boundary_risk_is_duplication_invariant():
    model = make_model()
    evidence = torch.tensor(
        [[-1.0, 0.2], [0.3, 1.5], [1.1, -0.4]], dtype=torch.float32
    )
    assert torch.allclose(
        model.aggregate_evidence(evidence),
        model.aggregate_evidence(evidence.repeat(4, 1)),
        atol=1e-6,
    )


def test_bounded_witness_has_finite_range():
    model = make_model()
    encoded = torch.randn(31, 12) * 1000
    evidence = model.bounded_evidence(encoded)
    assert evidence.abs().max().item() <= model.score_bound + 1e-6


def test_continuation_probabilities_are_ordered_and_all_classes_reachable():
    model = make_model()
    for risks, expected in [
        ([-8.0, 0.0], 0),
        ([8.0, -8.0], 1),
        ([8.0, 8.0], 2),
    ]:
        probabilities, _, cumulative = model.class_probabilities(
            torch.tensor(risks)
        )
        assert probabilities.argmax().item() == expected
        assert probabilities.sum().item() == pytest.approx(1.0)
        assert torch.all(probabilities > 0)
        assert cumulative[1] <= cumulative[0]


@pytest.mark.parametrize("aggregation", ["mean", "entropic"])
def test_boundary_variants_have_finite_gradients(aggregation):
    torch.manual_seed(11)
    model = make_model(aggregation)
    output, losses = model.compute_loss(
        torch.randn(1, 17, 8),
        torch.tensor([1]),
        torch.nn.CrossEntropyLoss(),
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
