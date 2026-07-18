import torch
import torch.nn.functional as F
import pytest

from modules.MIR_MIL.mir_mil import MIR_MIL
from utils.general_utils import add_epoch_info_log, init_epoch_info_log
from utils.loop_utils import _ranking_memory_scores, cal_scores, mir_train_loop
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


def test_sparse_class_readout_is_permutation_invariant():
    model = make_model(
        sparse_class_weight=0.1,
        sparse_class_query_count=3,
        sparse_class_topk_fraction=0.2,
    )
    bag = torch.randn(17, 6, dtype=torch.double)
    permutation = torch.randperm(bag.shape[0])

    first = model(bag)["logits"]
    second = model(bag[permutation])["logits"]

    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_focus_sparse_readout_is_permutation_invariant():
    model = make_model(
        focus_sparse_head_weight=0.1,
        focus_sparse_loss_weight=0.2,
        focus_sparse_query_count=3,
        focus_sparse_topk_fraction=0.2,
    )
    bag = torch.randn(17, 6, dtype=torch.double)
    permutation = torch.randperm(bag.shape[0])

    first = model(bag)["logits"]
    second = model(bag[permutation])["logits"]

    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_spatial_region_readout_is_permutation_invariant():
    model = make_model(
        coordinate_dim=2,
        spatial_region_weight=0.1,
        spatial_region_grid_size=3,
        spatial_region_value_dim=7,
        spatial_region_dim=8,
        spatial_region_attention_dim=5,
    )
    features = torch.randn(23, 6, dtype=torch.double)
    coordinates = torch.rand(23, 2, dtype=torch.double)
    bag = torch.cat((features, coordinates), dim=1)
    permutation = torch.randperm(bag.shape[0])

    first = model(bag)["logits"]
    second = model(bag[permutation])["logits"]

    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_spatial_region_readout_backpropagates():
    model = make_model(
        coordinate_dim=2,
        spatial_region_weight=0.1,
        spatial_region_grid_size=3,
        spatial_region_value_dim=7,
        spatial_region_dim=8,
        spatial_region_attention_dim=5,
    ).float()
    bag = torch.cat((torch.randn(1, 23, 6), torch.rand(1, 23, 2)), dim=2)
    label = torch.tensor([1])

    _, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    head = model.spatial_region_head
    assert head.value.weight.grad is not None
    assert torch.isfinite(head.value.weight.grad).all()
    assert head.attention_score.weight.grad is not None
    assert head.classifiers.grad is not None


def test_spatial_region_readout_requires_coordinates():
    with pytest.raises(ValueError, match="coordinate_dim=2"):
        make_model(spatial_region_weight=0.1)


def test_zero_coordinate_encoder_scale_blocks_direct_coordinate_signal():
    model = make_model(coordinate_dim=2, coordinate_encoder_scale=0.0)
    features = torch.randn(23, 6, dtype=torch.double)
    first_coordinates = torch.rand(23, 2, dtype=torch.double)
    second_coordinates = torch.rand(23, 2, dtype=torch.double)

    first = model(torch.cat((features, first_coordinates), dim=1))["logits"]
    second = model(torch.cat((features, second_coordinates), dim=1))["logits"]

    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_center_free_spatial_regions_are_reflection_invariant():
    model = make_model(
        coordinate_dim=2,
        coordinate_encoder_scale=0.0,
        spatial_region_weight=0.1,
        spatial_region_grid_size=3,
        spatial_region_value_dim=7,
        spatial_region_dim=8,
        spatial_region_attention_dim=5,
        spatial_region_include_centers=False,
        spatial_region_include_mass=True,
    )
    features = torch.randn(23, 6, dtype=torch.double)
    coordinates = 0.05 + 0.9 * torch.rand(23, 2, dtype=torch.double)
    reflected = coordinates.clone()
    reflected[:, 0] = 1.0 - reflected[:, 0]

    first = model(torch.cat((features, coordinates), dim=1))["logits"]
    second = model(torch.cat((features, reflected), dim=1))["logits"]

    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_residual_class_moment_readout_is_permutation_invariant():
    model = make_model(
        residual_class_moment_token_weight=0.1,
        residual_class_moment_token_count=3,
        residual_class_moment_token_readout_dim=8,
        residual_class_moment_token_rank_dim=5,
    )
    bag = torch.randn(17, 6, dtype=torch.double)
    permutation = torch.randperm(bag.shape[0])

    first = model(bag)["logits"]
    second = model(bag[permutation])["logits"]

    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_residual_class_moment_readout_backpropagates_through_residual():
    model = make_model(
        residual_class_moment_token_weight=0.1,
        residual_class_moment_token_count=3,
        residual_class_moment_token_readout_dim=8,
        residual_class_moment_token_rank_dim=5,
    ).float()
    bag = torch.randn(1, 19, 6)
    label = torch.tensor([1])

    _, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    head = model.residual_class_moment_token_head
    assert head.tokens.grad is not None
    assert torch.isfinite(head.tokens.grad).all()
    assert head.class_factors.grad is not None
    assert head.residual_scale.grad is not None


def test_pairwise_boundary_readout_is_permutation_invariant_and_zero_sum():
    model = make_model(
        pairwise_boundary_weight=0.1,
        pairwise_boundary_query_dim=5,
        pairwise_boundary_value_dim=7,
        pairwise_boundary_rank_dim=4,
    )
    bag = torch.randn(17, 6, dtype=torch.double)
    permutation = torch.randperm(bag.shape[0])

    first = model(bag)
    second = model(bag[permutation])

    torch.testing.assert_close(
        first["logits"], second["logits"], atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        first["pairwise_boundary_logits"],
        second["pairwise_boundary_logits"],
        atol=1e-10,
        rtol=1e-10,
    )
    encoded = model.state_from_weighted_points(bag)[1]
    class_residual, _ = model.pairwise_boundary_head(encoded)
    torch.testing.assert_close(
        class_residual.sum(dim=1), torch.zeros(1, dtype=torch.double)
    )


def test_pairwise_boundary_loss_supervises_pairs_and_backpropagates():
    model = make_model(
        pairwise_boundary_weight=0.1,
        pairwise_boundary_loss_weight=0.2,
        pairwise_boundary_query_dim=5,
        pairwise_boundary_value_dim=7,
        pairwise_boundary_rank_dim=4,
    ).float()
    bag = torch.randn(1, 19, 6)
    label = torch.tensor([1])

    output, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    assert output["pairwise_boundary_logits"].shape == (1, 3)
    assert losses["pairwise_boundary_loss"].item() > 0
    head = model.pairwise_boundary_head
    assert head.class_queries.grad is not None
    assert torch.isfinite(head.class_queries.grad).all()
    assert head.class_factors.grad is not None
    assert torch.isfinite(head.class_factors.grad).all()


def test_pairwise_boundary_loss_ignores_pairs_without_target_class():
    model = make_model(pairwise_boundary_loss_weight=0.2).float()
    pairwise_logits = torch.tensor([[0.2, 100.0, -0.4]])
    label = torch.tensor([1])

    loss = model.pairwise_boundary_loss(pairwise_logits, label)
    expected = F.binary_cross_entropy_with_logits(
        torch.tensor([0.2, -0.4]), torch.tensor([1.0, 0.0])
    )

    torch.testing.assert_close(loss, expected)


def test_pairwise_boundary_alignment_updates_only_deployed_logits():
    model = make_model(
        pairwise_boundary_loss_weight=0.2,
        pairwise_boundary_alignment_weight=0.3,
    ).float()
    logits = torch.tensor([[0.1, -0.2, 0.4]], requires_grad=True)
    pairwise_logits = torch.tensor(
        [[1.0, -1.0, 0.7]], requires_grad=True
    )
    label = torch.tensor([1])

    loss = model.pairwise_boundary_alignment_loss(
        logits, pairwise_logits, label
    )
    loss.backward()

    assert loss.item() > 0
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert pairwise_logits.grad is None


def test_pairwise_boundary_alignment_has_no_inference_residual():
    base = make_model()
    aligned = make_model(
        pairwise_boundary_loss_weight=0.2,
        pairwise_boundary_alignment_weight=0.3,
    )
    aligned.load_state_dict(base.state_dict(), strict=False)
    bag = torch.randn(17, 6, dtype=torch.double)

    torch.testing.assert_close(
        aligned(bag)["logits"], base(bag)["logits"]
    )


def test_logit_calibration_defaults_to_identity():
    base = make_model()
    calibrated = make_model(use_logit_calibration=True)
    bag = torch.randn(13, 6, dtype=torch.double)

    base_logits = base(bag)["logits"]
    calibrated_output = calibrated(bag)

    torch.testing.assert_close(
        calibrated_output["raw_logits"], base_logits, atol=1e-10, rtol=1e-10
    )
    torch.testing.assert_close(
        calibrated_output["logits"], base_logits, atol=1e-10, rtol=1e-10
    )


def test_logit_calibration_applies_temperature_and_bias():
    model = make_model(
        use_logit_calibration=True,
        logit_calibration_initial_temperature=2.0,
        logit_calibration_bias_init=[0.1, -0.2, 0.3],
    )
    logits = torch.tensor([[1.0, 2.0, -1.0]], dtype=torch.double)

    calibrated = model.calibrate_logits(logits)

    expected = logits / 2.0 + torch.tensor(
        [[0.1, -0.2, 0.3]], dtype=torch.double
    )
    torch.testing.assert_close(calibrated, expected)


def test_sparse_class_readout_backpropagates_through_queries_and_gate():
    model = make_model(
        sparse_class_weight=0.1,
        sparse_class_query_count=3,
        sparse_class_topk_fraction=0.2,
    ).float()
    bag = torch.randn(1, 19, 6)
    label = torch.tensor([1])

    _, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    head = model.sparse_class_head
    assert head.queries.grad is not None
    assert torch.isfinite(head.queries.grad).all()
    assert head.context_gate[-1].weight.grad is not None


def test_sparse_class_gate_has_configured_initial_value():
    model = make_model(
        sparse_class_weight=0.1,
        sparse_class_gate_initial_bias=-1.0,
    )
    bag = torch.randn(9, 6, dtype=torch.double)
    state = model.state_from_weighted_points(bag)[0]

    gate = torch.sigmoid(
        model.sparse_class_head.context_gate(state.unsqueeze(0))
    )
    expected = torch.sigmoid(torch.tensor(-1.0, dtype=torch.double))
    torch.testing.assert_close(gate, torch.full_like(gate, expected))


def test_sparse_class_readout_handles_single_patch_bag():
    model = make_model(
        sparse_class_weight=0.1,
        sparse_class_topk_fraction=0.001,
    )
    output = model(torch.randn(1, 6, dtype=torch.double))["logits"]

    assert output.shape == (1, 3)
    assert torch.isfinite(output).all()


def test_gated_attention_residual_is_permutation_invariant():
    model = make_model(
        gated_attention_weight=0.1,
        gated_attention_dim=8,
        gated_attention_value_dim=9,
    )
    bag = torch.randn(17, 6, dtype=torch.double)
    permutation = torch.randperm(bag.shape[0])

    first = model(bag)["logits"]
    second = model(bag[permutation])["logits"]

    torch.testing.assert_close(first, second, atol=1e-10, rtol=1e-10)


def test_gated_attention_residual_backpropagates():
    model = make_model(
        gated_attention_weight=0.1,
        gated_attention_dim=8,
        gated_attention_value_dim=9,
    ).float()
    bag = torch.randn(1, 19, 6)
    label = torch.tensor([1])

    _, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    head = model.gated_attention_head
    assert head.attention_score.weight.grad is not None
    assert torch.isfinite(head.attention_score.weight.grad).all()
    assert head.classifiers.grad is not None


def test_boundary_heads_are_absent_by_default():
    model = make_model()
    output = model(torch.randn(9, 6, dtype=torch.double))

    assert "ovr_logits" not in output
    assert "adjacent_logits" not in output
    assert "pairwise_boundary_logits" not in output


def test_ovr_boundary_head_adds_supervised_loss_and_gradients():
    model = make_model(ovr_loss_weight=0.2, ovr_head_weight=0.05).float()
    bag = torch.randn(1, 19, 6)
    label = torch.tensor([1])

    output, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    assert output["ovr_logits"].shape == (1, 3)
    assert losses["ovr_loss"].item() > 0
    assert model.ovr_head.net[-1].weight.grad is not None
    assert torch.isfinite(model.ovr_head.net[-1].weight.grad).all()


def test_ovr_boundary_loss_accepts_per_class_pos_weights():
    model = make_model(
        ovr_loss_weight=0.2,
        ovr_loss_pos_weight=2.0,
        ovr_loss_pos_weights=[1.0, 5.0, 1.0],
    ).float()
    logits = torch.tensor([[0.1, -0.2, 0.3]])
    label = torch.tensor([1])

    loss = model.ovr_boundary_loss(logits, label)
    expected = F.binary_cross_entropy_with_logits(
        logits,
        torch.tensor([[0.0, 1.0, 0.0]]),
        pos_weight=torch.tensor([1.0, 5.0, 1.0]),
    )

    torch.testing.assert_close(loss, expected)
    assert model.ovr_loss_pos_weights == [1.0, 5.0, 1.0]


def test_adjacent_boundary_head_adds_supervised_loss_and_gradients():
    model = make_model(adjacent_loss_weight=0.2).float()
    bag = torch.randn(1, 19, 6)
    label = torch.tensor([2])

    output, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    assert output["adjacent_logits"].shape == (1, 2)
    assert losses["adjacent_loss"].item() > 0
    assert model.adjacent_head.net[-1].weight.grad is not None
    assert torch.isfinite(model.adjacent_head.net[-1].weight.grad).all()


def test_focus_class_head_adds_supervised_loss_and_gradients():
    model = make_model(
        focus_class_index=1,
        focus_class_head_weight=0.1,
        focus_class_loss_weight=0.2,
        focus_class_loss_pos_weight=3.0,
    ).float()
    bag = torch.randn(1, 19, 6)
    label = torch.tensor([1])

    output, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    assert output["focus_class_logits"].shape == (1, 1)
    assert losses["focus_class_loss"].item() > 0
    assert model.focus_class_head.net[-1].weight.grad is not None
    assert torch.isfinite(model.focus_class_head.net[-1].weight.grad).all()


def test_focus_sparse_head_adds_supervised_loss_and_gradients():
    model = make_model(
        focus_class_index=1,
        focus_sparse_head_weight=0.1,
        focus_sparse_loss_weight=0.2,
        focus_sparse_loss_pos_weight=3.0,
        focus_sparse_query_count=3,
        focus_sparse_topk_fraction=0.2,
    ).float()
    bag = torch.randn(1, 19, 6)
    label = torch.tensor([1])

    output, losses = model.compute_loss(
        bag, label, torch.nn.CrossEntropyLoss()
    )
    losses["loss"].backward()

    assert output["focus_sparse_logits"].shape == (1, 1)
    assert losses["focus_sparse_loss"].item() > 0
    assert model.focus_sparse_head.queries.grad is not None
    assert torch.isfinite(model.focus_sparse_head.queries.grad).all()
    mixer = model.focus_sparse_head.context_mixer[-1]
    assert mixer.weight.grad is not None
    assert torch.isfinite(mixer.weight.grad).all()


def test_focus_sparse_readout_handles_single_patch_bag():
    model = make_model(
        focus_sparse_head_weight=0.1,
        focus_sparse_topk_fraction=0.001,
    )
    output = model(torch.randn(1, 6, dtype=torch.double))

    assert output["logits"].shape == (1, 3)
    assert torch.isfinite(output["logits"]).all()


@pytest.mark.parametrize("fraction", [0.0, 1.1])
def test_sparse_class_readout_rejects_invalid_topk_fraction(fraction):
    with pytest.raises(ValueError, match="topk_fraction"):
        make_model(
            sparse_class_weight=0.1,
            sparse_class_topk_fraction=fraction,
        )


def test_optional_input_group_normalization_preserves_state_structure():
    baseline = make_model()
    normalized = make_model(
        input_group_l2_normalize=True,
        input_group_size=3,
    )
    bag = torch.tensor(
        [[3.0, 4.0, 0.0, 0.0, 0.0, 2.0]],
        dtype=torch.double,
    )

    transformed = normalized._normalize_bag(bag)

    torch.testing.assert_close(
        transformed.reshape(1, 2, 3).norm(dim=2),
        torch.ones((1, 2), dtype=torch.double),
    )
    assert baseline.state_dict().keys() == normalized.state_dict().keys()


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


def test_anchor_route_response_matches_finite_difference():
    model = make_model(
        num_local_routes=3,
        local_route_dim=4,
        anchor_route_dim=7,
        anchor_route_temperature=0.8,
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

    assert response["anchor_response"].abs().max() > 0
    torch.testing.assert_close(
        response["anchor_response"].mean(),
        torch.zeros((), dtype=torch.double),
        atol=1e-10,
        rtol=1e-10,
    )
    torch.testing.assert_close(
        response["response"], finite, atol=5e-5, rtol=5e-5
    )


def test_anchor_route_integrated_attribution_is_complete():
    model = make_model(
        num_local_routes=3,
        local_route_dim=4,
        anchor_route_dim=7,
        anchor_route_temperature=0.8,
    )
    bag = torch.randn(8, 6, dtype=torch.double)
    baseline = torch.randn(7, 6, dtype=torch.double)

    result = model.integrated_functional_attribution(
        bag, baseline, target_class=0, steps=1025
    )

    torch.testing.assert_close(
        result["decomposition"],
        result["score_difference"],
        atol=5e-5,
        rtol=5e-5,
    )


def test_anchored_multiscale_response_matches_finite_difference():
    model = make_model(
        hidden_dim=12,
        num_local_routes=3,
        local_route_dim=4,
        anchor_route_dim=12,
        anchor_route_identity=True,
        potential_type="anchored_multiscale",
        anchor_global_initial_scale=0.1,
        anchor_local_initial_scale=0.1,
    )
    bag = torch.randn(10, 6, dtype=torch.double)
    output = model(bag, return_state=True)
    encoded = model.encoder(model._normalize_bag(bag))
    torch.testing.assert_close(output["anchor_basis"], encoded)

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
        response["response"], finite, atol=6e-5, rtol=6e-5
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


def test_sparse_class_options_are_constructed_from_yaml():
    args = read_yaml("configs/MIR_MIL.yaml")
    args.Model.sparse_class_weight = 0.2
    args.Model.sparse_class_query_count = 3
    args.Model.sparse_class_topk_fraction = 0.125
    args.Model.sparse_class_gate_initial_bias = -1.5

    model = get_model_from_yaml(args)

    assert model.sparse_class_weight == 0.2
    assert model.sparse_class_query_count == 3
    assert model.sparse_class_topk_fraction == 0.125
    assert model.sparse_class_head is not None
    assert model.sparse_class_head.gate_initial_bias == -1.5


def test_residual_class_moment_options_are_constructed_from_yaml():
    args = read_yaml("configs/MIR_MIL.yaml")
    args.Model.residual_class_moment_token_weight = 0.2
    args.Model.residual_class_moment_token_count = 3
    args.Model.residual_class_moment_token_rank_dim = 5
    args.Model.residual_class_moment_token_initial_scale = 0.025

    model = get_model_from_yaml(args)

    assert model.residual_class_moment_token_weight == 0.2
    assert model.residual_class_moment_token_count == 3
    assert model.residual_class_moment_token_head is not None
    torch.testing.assert_close(
        model.residual_class_moment_token_head.residual_scale,
        torch.full_like(
            model.residual_class_moment_token_head.residual_scale, 0.025
        ),
    )


def test_pairwise_boundary_options_are_constructed_from_yaml():
    args = read_yaml("configs/MIR_MIL.yaml")
    args.General.num_classes = 3
    args.Model.pairwise_boundary_weight = 0.1
    args.Model.pairwise_boundary_loss_weight = 0.2
    args.Model.pairwise_boundary_alignment_weight = 0.3
    args.Model.pairwise_boundary_query_dim = 17
    args.Model.pairwise_boundary_value_dim = 19
    args.Model.pairwise_boundary_rank_dim = 11

    model = get_model_from_yaml(args)

    assert model.pairwise_boundary_weight == 0.1
    assert model.pairwise_boundary_loss_weight == 0.2
    assert model.pairwise_boundary_alignment_weight == 0.3
    assert model.pairwise_boundary_head is not None
    assert model.pairwise_boundary_head.query_dim == 17
    assert model.pairwise_boundary_head.value_dim == 19
    assert model.pairwise_boundary_head.rank_dim == 11


def test_logit_calibration_options_are_constructed_from_yaml():
    args = read_yaml("configs/MIR_MIL.yaml")
    args.General.num_classes = 3
    args.Model.use_logit_calibration = True
    args.Model.logit_calibration_learn_temperature = True
    args.Model.logit_calibration_initial_temperature = 1.5
    args.Model.logit_calibration_bias_init = [0.0, -0.4, -0.3]

    model = get_model_from_yaml(args)

    assert model.use_logit_calibration
    assert model.logit_calibration_learn_temperature
    assert model.logit_calibration_bias.requires_grad
    assert model.logit_calibration_log_temperature.requires_grad
    torch.testing.assert_close(
        model.logit_calibration_bias,
        torch.tensor([0.0, -0.4, -0.3]),
    )
    torch.testing.assert_close(
        model.logit_calibration_log_temperature.exp(),
        torch.tensor(1.5),
    )


def test_boundary_head_options_are_constructed_from_yaml():
    args = read_yaml("configs/MIR_MIL.yaml")
    args.General.num_classes = 3
    args.Model.ovr_head_weight = 0.05
    args.Model.ovr_loss_weight = 0.1
    args.Model.ovr_loss_pos_weights = [1.0, 5.0, 1.0]
    args.Model.ovr_head_hidden_dim = 17
    args.Model.adjacent_head_weight = 0.0
    args.Model.adjacent_loss_weight = 0.05
    args.Model.adjacent_head_hidden_dim = 19
    args.Model.focus_class_index = 1
    args.Model.focus_class_head_weight = 0.07
    args.Model.focus_class_loss_weight = 0.3
    args.Model.focus_class_loss_pos_weight = 4.0
    args.Model.focus_class_head_hidden_dim = 23
    args.Model.focus_sparse_head_weight = 0.09
    args.Model.focus_sparse_loss_weight = 0.25
    args.Model.focus_sparse_loss_pos_weight = 3.5
    args.Model.focus_sparse_query_count = 5
    args.Model.focus_sparse_topk_fraction = 0.04

    model = get_model_from_yaml(args)

    assert model.ovr_head_weight == 0.05
    assert model.ovr_loss_weight == 0.1
    assert model.ovr_loss_pos_weights == [1.0, 5.0, 1.0]
    assert model.ovr_head is not None
    assert model.ovr_head.net[1].out_features == 17
    assert model.adjacent_head_weight == 0.0
    assert model.adjacent_loss_weight == 0.05
    assert model.adjacent_head is not None
    assert model.adjacent_head.net[1].out_features == 19
    assert model.focus_class_index == 1
    assert model.focus_class_head_weight == 0.07
    assert model.focus_class_loss_weight == 0.3
    assert model.focus_class_loss_pos_weight == 4.0
    assert model.focus_class_head is not None
    assert model.focus_class_head.net[1].out_features == 23
    assert model.focus_sparse_head_weight == 0.09
    assert model.focus_sparse_loss_weight == 0.25
    assert model.focus_sparse_loss_pos_weight == 3.5
    assert model.focus_sparse_query_count == 5
    assert model.focus_sparse_topk_fraction == 0.04
    assert model.focus_sparse_head is not None


def test_cross_entropy_supports_label_smoothing():
    criterion = get_criterion("ce", label_smoothing=0.1)

    assert isinstance(criterion, torch.nn.CrossEntropyLoss)
    assert criterion.label_smoothing == 0.1


def test_cal_scores_exposes_per_class_auc_for_model_selection():
    labels = [0, 1, 2, 0, 1, 2]
    probs = [
        [0.8, 0.1, 0.1],
        [0.2, 0.7, 0.1],
        [0.1, 0.2, 0.7],
        [0.7, 0.2, 0.1],
        [0.1, 0.6, 0.3],
        [0.2, 0.2, 0.6],
    ]

    metrics = cal_scores(probs, labels, num_classes=3)

    assert metrics["auc_class_0"] == 1.0
    assert metrics["auc_class_1"] == 1.0
    assert metrics["auc_class_2"] == 1.0
    assert metrics["min_class_auc"] == 1.0
    assert metrics["macro_auc_hmean_auc_class_1"] == 1.0


def test_epoch_log_accepts_dynamic_metrics():
    epoch_log = init_epoch_info_log()
    metrics = {
        "macro_auc": 0.8,
        "auc_class_1": 0.7,
        "macro_auc_hmean_auc_class_1": 0.7466666667,
    }

    add_epoch_info_log(
        epoch_log,
        epoch=0,
        train_loss=1.0,
        val_loss=0.9,
        test_loss=None,
        val_metrics=metrics,
        test_metrics=None,
    )

    assert epoch_log["val_auc_class_1"] == [0.7]
    assert epoch_log["test_auc_class_1"] == [None]
    assert epoch_log["val_macro_auc_hmean_auc_class_1"] == [0.7466666667]


def test_mir_train_loop_applies_distillation_loss():
    class TinyMir(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)

        def compute_loss(self, bag, label, criterion):
            logits = self.linear(bag.mean(dim=1))
            classification_loss = criterion(logits, label)
            return {
                "logits": logits
            }, {
                "loss": classification_loss,
                "classification_loss": classification_loss,
            }

    model = TinyMir()
    loader = [
        (
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
    ]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    teacher = torch.tensor([[0.1, 0.9]])

    train_loss, _, components = mir_train_loop(
        "cpu",
        model,
        loader,
        criterion,
        optimizer,
        None,
        distillation_targets=teacher,
        distillation_weight=0.5,
        distillation_temperature=1.0,
    )

    assert components["distillation_loss"] > 0
    assert train_loss > components["classification_loss"]


def test_mir_train_loop_can_mask_low_entropy_distillation_targets():
    class TinyMir(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)

        def compute_loss(self, bag, label, criterion):
            logits = self.linear(bag.mean(dim=1))
            classification_loss = criterion(logits, label)
            return {
                "logits": logits
            }, {
                "loss": classification_loss,
                "classification_loss": classification_loss,
            }

    model = TinyMir()
    loader = [
        (
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
    ]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    teacher = torch.tensor([[0.999, 0.001]])

    train_loss, _, components = mir_train_loop(
        "cpu",
        model,
        loader,
        criterion,
        optimizer,
        None,
        distillation_targets=teacher,
        distillation_weight=0.5,
        distillation_temperature=1.0,
        distillation_min_entropy=0.5,
    )

    assert components["distillation_loss"] == 0.0
    assert train_loss == components["classification_loss"]


def test_mir_train_loop_supports_entropy_weighted_distillation():
    class TinyMir(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)

        def compute_loss(self, bag, label, criterion):
            logits = self.linear(bag.mean(dim=1))
            classification_loss = criterion(logits, label)
            return {
                "logits": logits
            }, {
                "loss": classification_loss,
                "classification_loss": classification_loss,
            }

    model = TinyMir()
    loader = [
        (
            torch.tensor(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[2.0, 0.0], [0.0, 1.0]],
                ]
            ),
            torch.tensor([0, 1]),
            torch.tensor([0, 1]),
        )
    ]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    teacher = torch.tensor([[0.5, 0.5], [0.999, 0.001]])

    train_loss, _, components = mir_train_loop(
        "cpu",
        model,
        loader,
        criterion,
        optimizer,
        None,
        distillation_targets=teacher,
        distillation_weight=0.5,
        distillation_temperature=1.0,
        distillation_entropy_weight_power=1.0,
    )

    assert components["distillation_loss"] > 0
    assert train_loss > components["classification_loss"]


def test_mir_train_loop_applies_ranking_memory_loss():
    class TinyMir(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 2, bias=False)

        def compute_loss(self, bag, label, criterion):
            logits = self.linear(bag.mean(dim=1))
            classification_loss = criterion(logits, label)
            return {
                "logits": logits
            }, {
                "loss": classification_loss,
                "classification_loss": classification_loss,
            }

    model = TinyMir()
    loader = [
        (
            torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
            torch.tensor([0]),
            torch.tensor([0]),
        )
    ]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    ranking_memory = torch.tensor([[0.0, 0.0], [2.0, 2.0]])
    ranking_labels = torch.tensor([0, 1])
    ranking_valid = torch.tensor([False, True])

    train_loss, _, components = mir_train_loop(
        "cpu",
        model,
        loader,
        criterion,
        optimizer,
        None,
        ranking_memory=ranking_memory,
        ranking_memory_labels=ranking_labels,
        ranking_memory_valid=ranking_valid,
        ranking_memory_weight=0.5,
        ranking_memory_margin=0.1,
        ranking_memory_momentum=0.5,
    )

    assert components["ranking_memory_loss"] > 0
    assert train_loss > components["classification_loss"]
    assert ranking_valid.tolist() == [True, True]


def test_ranking_memory_can_focus_one_class_and_mine_hard_pairs():
    class FixedLogits(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[0.0, 0.5, 0.0]]))

        def compute_loss(self, bag, label, criterion):
            logits = self.logits.expand(label.shape[0], -1)
            classification_loss = criterion(logits, label)
            return {"logits": logits}, {
                "loss": classification_loss,
                "classification_loss": classification_loss,
            }

    model = FixedLogits()
    loader = [(torch.zeros(1, 1, 1), torch.tensor([1]), torch.tensor([0]))]
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    memory = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.4, 0.0],
            [0.0, 1.5, 0.0],
        ]
    )
    labels = torch.tensor([1, 0, 2])
    valid = torch.tensor([False, True, True])

    _, _, components = mir_train_loop(
        "cpu",
        model,
        loader,
        criterion,
        optimizer,
        None,
        ranking_memory=memory,
        ranking_memory_labels=labels,
        ranking_memory_valid=valid,
        ranking_memory_weight=1.0,
        ranking_memory_max_pairs=1,
        ranking_memory_class_indices=[1],
        ranking_memory_hard_mining=True,
    )

    expected = torch.nn.functional.softplus(torch.tensor(0.1 - (0.5 - 1.5)))
    torch.testing.assert_close(
        torch.tensor(components["ranking_memory_loss"]), expected
    )

    negative_model = FixedLogits()
    negative_model.logits.data.copy_(torch.tensor([[0.5, 0.5, 0.0]]))
    negative_memory = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.3, 0.0],
        ]
    )
    _, _, negative_components = mir_train_loop(
        "cpu",
        negative_model,
        [(torch.zeros(1, 1, 1), torch.tensor([0]), torch.tensor([0]))],
        criterion,
        torch.optim.SGD(negative_model.parameters(), lr=0.0),
        None,
        ranking_memory=negative_memory,
        ranking_memory_labels=torch.tensor([0, 1, 1]),
        ranking_memory_valid=torch.tensor([False, True, True]),
        ranking_memory_weight=1.0,
        ranking_memory_max_pairs=1,
        ranking_memory_class_indices=[1],
        ranking_memory_hard_mining=True,
    )
    negative_expected = torch.nn.functional.softplus(
        torch.tensor(0.1 - (-1.0 - 0.5))
    )
    torch.testing.assert_close(
        torch.tensor(negative_components["ranking_memory_loss"]),
        negative_expected,
    )


def test_ranking_memory_warmup_updates_memory_without_applying_loss():
    class TinyMir(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.tensor([[0.2, -0.2]]))

        def compute_loss(self, bag, label, criterion):
            classification_loss = criterion(self.logits, label)
            return {"logits": self.logits}, {
                "loss": classification_loss,
                "classification_loss": classification_loss,
            }

    model = TinyMir()
    loader = [(torch.zeros(1, 1, 1), torch.tensor([0]), torch.tensor([0]))]
    memory = torch.zeros(1, 2)
    valid = torch.zeros(1, dtype=torch.bool)
    criterion = torch.nn.CrossEntropyLoss()

    train_loss, _, components = mir_train_loop(
        "cpu",
        model,
        loader,
        criterion,
        torch.optim.SGD(model.parameters(), lr=0.0),
        None,
        ranking_memory=memory,
        ranking_memory_labels=torch.tensor([0]),
        ranking_memory_valid=valid,
        ranking_memory_weight=1.0,
        ranking_memory_apply_loss=False,
    )

    assert components["ranking_memory_loss"] == 0.0
    assert train_loss == components["classification_loss"]
    assert valid.tolist() == [True]
    torch.testing.assert_close(memory, model.logits.detach())


def test_ovr_log_odds_ranking_matches_softmax_probability_order():
    logits = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [2.0, 1.2, 0.0],
            [-1.0, 0.5, 1.5],
        ]
    )
    scores = _ranking_memory_scores(logits, "ovr_log_odds")
    probabilities = torch.softmax(logits, dim=-1)

    for class_index in range(logits.shape[1]):
        score_order = torch.argsort(scores[:, class_index])
        probability_order = torch.argsort(probabilities[:, class_index])
        torch.testing.assert_close(score_order, probability_order)

    expected_class_1 = logits[:, 1] - torch.logsumexp(
        logits[:, [0, 2]], dim=-1
    )
    torch.testing.assert_close(scores[:, 1], expected_class_1)
