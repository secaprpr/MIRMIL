import torch
import pytest

from modules.MIR_MIL.mir_mil import MIR_MIL
from utils.model_utils import get_criterion, get_model_from_yaml
from utils.yaml_utils import read_yaml


def make_model(num_classes=3, **overrides):
    options = {
        "in_dim": 6,
        "num_classes": num_classes,
        "hidden_dim": 12,
        "sketch_dim": 5,
        "num_tail_scores": 3,
        "tail_temperature": 0.4,
        "potential_hidden_dim": 7,
        "dropout": 0.0,
        "stability_weight": 0.0,
    }
    options.update(overrides)
    torch.manual_seed(7)
    return MIR_MIL(**options).double()


def test_forward_is_permutation_invariant():
    model = make_model()
    bag = torch.randn(11, 6, dtype=torch.double)
    permutation = torch.randperm(bag.shape[0])

    first = model(bag)["logits"]
    second = model(bag[permutation])["logits"]

    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_measure_influence_is_centered_under_current_measure():
    model = make_model()
    bag = torch.randn(13, 6, dtype=torch.double)

    result = model.measure_influence_response(bag, target_class=1)

    torch.testing.assert_close(
        result["response"].mean(),
        torch.zeros((), dtype=torch.double),
        atol=1e-10,
        rtol=1e-10,
    )


def test_closed_form_response_matches_finite_difference():
    model = make_model()
    bag = torch.randn(9, 6, dtype=torch.double)
    result = model.measure_influence_response(bag, target_class=2)

    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=2, epsilon=1e-6
            )
            for point in bag
        ]
    )

    torch.testing.assert_close(
        result["response"], finite, atol=2e-5, rtol=2e-5
    )


def test_second_order_response_matches_finite_difference():
    model = make_model(moment_order=2)
    bag = torch.randn(9, 6, dtype=torch.double)
    result = model.measure_influence_response(bag, target_class=2)
    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=2, epsilon=1e-6
            )
            for point in bag
        ]
    )

    assert result["variance_response"].abs().max() > 0
    torch.testing.assert_close(
        result["response"], finite, atol=3e-5, rtol=3e-5
    )


def test_integrated_functional_attribution_is_complete():
    model = make_model()
    bag = torch.randn(8, 6, dtype=torch.double)
    baseline = torch.randn(7, 6, dtype=torch.double)

    result = model.integrated_functional_attribution(
        bag, baseline, target_class=0, steps=257
    )

    torch.testing.assert_close(
        result["decomposition"],
        result["score_difference"],
        atol=2e-5,
        rtol=2e-5,
    )


def test_second_order_integrated_attribution_is_complete():
    model = make_model(moment_order=2)
    bag = torch.randn(8, 6, dtype=torch.double)
    baseline = torch.randn(7, 6, dtype=torch.double)

    result = model.integrated_functional_attribution(
        bag, baseline, target_class=0, steps=257
    )

    torch.testing.assert_close(
        result["decomposition"],
        result["score_difference"],
        atol=3e-5,
        rtol=3e-5,
    )


def test_multiclass_score_is_class_margin():
    model = make_model(num_classes=4)
    logits = torch.tensor([[1.0, 2.0, -1.0, 0.5]], dtype=torch.double)

    score = model.explained_score(logits, target_class=1)
    expected = logits[0, 1] - torch.logsumexp(
        logits[0, torch.tensor([0, 2, 3])], dim=0
    )

    torch.testing.assert_close(score, expected)


def test_stability_loss_backpropagates():
    model = make_model(
        num_classes=2,
        stability_weight=0.2,
        patch_dropout=0.2,
        feature_noise_std=0.01,
    ).float()
    bag = torch.randn(1, 12, 6)
    label = torch.tensor([1])
    criterion = torch.nn.CrossEntropyLoss()

    _, losses = model.compute_loss(bag, label, criterion)
    losses["loss"].backward()

    assert torch.isfinite(losses["loss"])
    assert any(
        parameter.grad is not None
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def test_ordinal_cdf_loss_respects_class_distance():
    model = make_model(num_classes=4)
    label = torch.tensor([0])
    adjacent = torch.tensor(
        [[0.0, 5.0, 0.0, 0.0]], dtype=torch.double
    )
    distant = torch.tensor(
        [[0.0, 0.0, 0.0, 5.0]], dtype=torch.double
    )

    assert (
        model.ordinal_cdf_loss(adjacent, label)
        < model.ordinal_cdf_loss(distant, label)
    )


def test_zero_ordinal_weight_preserves_total_loss():
    model = make_model(num_classes=4, ordinal_weight=0.0).float()
    bag = torch.randn(1, 12, 6)
    label = torch.tensor([2])
    criterion = torch.nn.CrossEntropyLoss()

    _, losses = model.compute_loss(bag, label, criterion)

    torch.testing.assert_close(
        losses["loss"], losses["classification_loss"]
    )


def test_mixture_prototype_potential_preserves_mir_properties():
    model = make_model(
        num_classes=4,
        potential_type="mixture_prototype",
        prototype_embedding_dim=6,
        prototypes_per_class=3,
        prototype_regularization_weight=0.1,
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    response = model.measure_influence_response(bag, target_class=2)

    assert response["logits"].shape == (1, 4)
    torch.testing.assert_close(
        response["response"].mean(),
        torch.zeros((), dtype=torch.double),
        atol=1e-10,
        rtol=1e-10,
    )
    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=2, epsilon=1e-6
            )
            for point in bag
        ]
    )
    torch.testing.assert_close(
        response["response"], finite, atol=3e-5, rtol=3e-5
    )


def test_mixture_prototype_regularization_backpropagates():
    model = make_model(
        num_classes=4,
        potential_type="mixture_prototype",
        prototype_embedding_dim=6,
        prototypes_per_class=3,
        prototype_regularization_weight=0.1,
    ).float()
    bag = torch.randn(1, 12, 6)
    label = torch.tensor([2])
    criterion = torch.nn.CrossEntropyLoss()

    _, losses = model.compute_loss(bag, label, criterion)
    losses["loss"].backward()

    assert losses["prototype_loss"] > 0
    assert model.potential.prototypes.grad is not None


def test_residual_prototype_starts_from_base_potential():
    model = make_model(
        num_classes=4,
        potential_type="residual_prototype",
        prototype_embedding_dim=6,
        prototypes_per_class=3,
        prototype_residual_initial_scale=0.0,
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    state = model.state_from_weighted_points(bag)[0]

    logits = model.potential(state.unsqueeze(0))
    base_logits = model.potential.base(state.unsqueeze(0))

    torch.testing.assert_close(logits, base_logits)


def test_residual_prototype_preserves_shared_initialization():
    base = make_model(num_classes=4)
    residual = make_model(
        num_classes=4,
        potential_type="residual_prototype",
        prototype_embedding_dim=6,
        prototypes_per_class=3,
    )

    for base_parameter, residual_parameter in zip(
        base.encoder.parameters(), residual.encoder.parameters()
    ):
        torch.testing.assert_close(base_parameter, residual_parameter)
    for base_parameter, residual_parameter in zip(
        base.potential.parameters(), residual.potential.base.parameters()
    ):
        torch.testing.assert_close(base_parameter, residual_parameter)


def test_residual_prototype_preserves_mir_response_properties():
    model = make_model(
        num_classes=4,
        potential_type="residual_prototype",
        prototype_embedding_dim=6,
        prototypes_per_class=3,
        prototype_residual_initial_scale=0.1,
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    response = model.measure_influence_response(bag, target_class=2)
    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=2, epsilon=1e-6
            )
            for point in bag
        ]
    )

    torch.testing.assert_close(
        response["response"].mean(),
        torch.zeros((), dtype=torch.double),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        response["response"], finite, atol=3e-5, rtol=3e-5
    )


def test_local_route_response_matches_finite_difference():
    model = make_model(
        num_local_routes=3,
        local_route_dim=4,
        local_route_temperature=0.4,
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    response = model.measure_influence_response(bag, target_class=2)
    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=2, epsilon=1e-6
            )
            for point in bag
        ]
    )

    assert response["local_response"].abs().max() > 0
    torch.testing.assert_close(
        response["local_response"].mean(),
        torch.zeros((), dtype=torch.double),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        response["response"], finite, atol=4e-5, rtol=4e-5
    )


def test_local_route_integrated_attribution_is_complete():
    model = make_model(
        num_local_routes=3,
        local_route_dim=4,
        local_route_temperature=0.4,
    )
    bag = torch.randn(8, 6, dtype=torch.double)
    baseline = torch.randn(7, 6, dtype=torch.double)

    result = model.integrated_functional_attribution(
        bag, baseline, target_class=0, steps=257
    )

    torch.testing.assert_close(
        result["decomposition"],
        result["score_difference"],
        atol=4e-5,
        rtol=4e-5,
    )


def test_adaptive_multiscale_response_matches_finite_difference():
    model = make_model(
        num_local_routes=3,
        local_route_dim=4,
        local_route_temperature=0.4,
        potential_type="adaptive_multiscale",
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    response = model.measure_influence_response(bag, target_class=2)
    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=2, epsilon=1e-6
            )
            for point in bag
        ]
    )

    torch.testing.assert_close(
        response["response"], finite, atol=4e-5, rtol=4e-5
    )


def test_adaptive_multiscale_starts_with_conservative_local_gate():
    model = make_model(
        num_local_routes=3,
        local_route_dim=4,
        potential_type="adaptive_multiscale",
        multiscale_gate_initial_bias=-2.0,
    )
    state = torch.randn(2, 20, dtype=torch.double)
    global_state = state[:, :8]
    gate = torch.sigmoid(model.potential.local_gate(global_state))

    torch.testing.assert_close(
        gate,
        torch.full_like(gate, torch.sigmoid(torch.tensor(-2.0)).item()),
    )


def test_class_conditional_multiscale_response_matches_finite_difference():
    model = make_model(
        num_classes=3,
        num_local_routes=6,
        local_route_dim=4,
        local_route_temperature=0.4,
        potential_type="class_conditional_multiscale",
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    response = model.measure_influence_response(bag, target_class=2)
    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=2, epsilon=1e-6
            )
            for point in bag
        ]
    )

    assert len(model.potential.class_local_potentials) == 3
    torch.testing.assert_close(
        response["response"].mean(),
        torch.zeros((), dtype=torch.double),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        response["response"], finite, atol=5e-5, rtol=5e-5
    )


def test_class_conditional_multiscale_requires_route_groups():
    with pytest.raises(ValueError, match="divisible by num_classes"):
        make_model(
            num_classes=3,
            num_local_routes=4,
            local_route_dim=4,
            potential_type="class_conditional_multiscale",
        )


def test_hybrid_multiscale_response_matches_finite_difference():
    model = make_model(
        num_classes=3,
        num_local_routes=6,
        local_route_dim=4,
        local_route_temperature=0.4,
        potential_type="hybrid_multiscale",
        multiscale_class_mix_initial=0.5,
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    response = model.measure_influence_response(bag, target_class=1)
    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=1, epsilon=1e-6
            )
            for point in bag
        ]
    )

    torch.testing.assert_close(
        torch.sigmoid(model.potential.class_mix_logit),
        torch.full((3,), 0.5, dtype=torch.double),
    )
    torch.testing.assert_close(
        response["response"], finite, atol=5e-5, rtol=5e-5
    )


def test_hybrid_multiscale_preserves_shared_path_initialization():
    shared = make_model(
        num_classes=3,
        num_local_routes=6,
        local_route_dim=4,
        potential_type="adaptive_multiscale",
    )
    hybrid = make_model(
        num_classes=3,
        num_local_routes=6,
        local_route_dim=4,
        potential_type="hybrid_multiscale",
    )

    pairs = [
        (
            shared.potential.global_potential,
            hybrid.potential.global_potential,
        ),
        (
            shared.potential.local_potential,
            hybrid.potential.shared_local_potential,
        ),
        (shared.potential.local_gate, hybrid.potential.local_gate),
    ]
    for shared_module, hybrid_module in pairs:
        for shared_parameter, hybrid_parameter in zip(
            shared_module.parameters(), hybrid_module.parameters()
        ):
            torch.testing.assert_close(
                shared_parameter, hybrid_parameter
            )


def test_residual_class_multiscale_response_matches_finite_difference():
    model = make_model(
        num_classes=3,
        num_local_routes=6,
        local_route_dim=4,
        local_route_temperature=0.4,
        potential_type="residual_class_multiscale",
        multiscale_class_residual_initial_scale=0.05,
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    response = model.measure_influence_response(bag, target_class=1)
    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=1, epsilon=1e-6
            )
            for point in bag
        ]
    )

    torch.testing.assert_close(
        model.potential.class_residual_scale,
        torch.full((3,), 0.05, dtype=torch.double),
    )
    torch.testing.assert_close(
        response["response"], finite, atol=5e-5, rtol=5e-5
    )


def test_residual_class_preserves_shared_path_initialization():
    shared = make_model(
        num_classes=3,
        num_local_routes=6,
        local_route_dim=4,
        potential_type="adaptive_multiscale",
    )
    residual = make_model(
        num_classes=3,
        num_local_routes=6,
        local_route_dim=4,
        potential_type="residual_class_multiscale",
    )

    pairs = [
        (
            shared.potential.global_potential,
            residual.potential.global_potential,
        ),
        (
            shared.potential.local_potential,
            residual.potential.shared_local_potential,
        ),
        (shared.potential.local_gate, residual.potential.local_gate),
    ]
    for shared_module, residual_module in pairs:
        for shared_parameter, residual_parameter in zip(
            shared_module.parameters(), residual_module.parameters()
        ):
            torch.testing.assert_close(
                shared_parameter, residual_parameter
            )


def test_adaptive_multiscale_prototype_response_matches_finite_difference():
    model = make_model(
        num_local_routes=3,
        local_route_dim=4,
        local_route_temperature=0.4,
        potential_type="adaptive_multiscale_prototype",
        prototype_embedding_dim=6,
        prototypes_per_class=2,
        multiscale_prototype_initial_scale=0.05,
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    response = model.measure_influence_response(bag, target_class=1)
    finite = torch.stack(
        [
            model.finite_difference_response(
                bag, point, target_class=1, epsilon=1e-6
            )
            for point in bag
        ]
    )

    torch.testing.assert_close(
        response["response"], finite, atol=5e-5, rtol=5e-5
    )


def test_adaptive_multiscale_prototype_regularization_backpropagates():
    model = make_model(
        num_local_routes=3,
        local_route_dim=4,
        potential_type="adaptive_multiscale_prototype",
        prototype_embedding_dim=6,
        prototypes_per_class=2,
        prototype_regularization_weight=0.1,
    ).float()
    bag = torch.randn(12, 6)
    label = torch.tensor([1])

    _, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    assert losses["prototype_loss"] > 0
    assert model.potential.prototype_scale.grad is not None
    assert model.potential.prototype_potential.prototypes.grad is not None


def test_model_is_constructed_from_repository_yaml():
    args = read_yaml("configs/MIR_MIL.yaml")
    model = get_model_from_yaml(args)

    assert isinstance(model, MIR_MIL)
    assert model.input_dim == args.Model.in_dim
    assert model.num_classes == args.General.num_classes
    assert model.num_local_routes == 12
    assert model.potential_type == "adaptive_multiscale"


def test_cross_entropy_supports_label_smoothing():
    criterion = get_criterion("ce", label_smoothing=0.1)

    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.1
