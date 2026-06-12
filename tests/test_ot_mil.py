import unittest

import torch
from addict import Dict

from modules.OT_MIL.ot_mil import OT_MIL
from utils.model_utils import WarmUpLR, get_model_from_yaml


class OTMILTest(unittest.TestCase):
    def _make_model(self):
        return OT_MIL(
            in_dim=32,
            hidden_dim=16,
            num_classes=3,
            num_prototypes=4,
            dropout=0.0,
            sinkhorn_iterations=8,
            gate_temperature=0.5,
            max_instances=128,
        )

    def test_forward_and_backward(self):
        torch.manual_seed(7)
        model = self._make_model()
        bag = torch.randn(1, 40, 32)
        label = torch.tensor([2])

        output = model(bag, return_WSI_feature=True, return_controls=True)
        losses = model.compute_loss(output, label)
        losses["loss"].backward()

        self.assertEqual(output["logits"].shape, (1, 3))
        self.assertEqual(output["WSI_feature"].shape, (1, 100))
        self.assertEqual(output["complement_logits"].shape, (1, 3))
        self.assertEqual(output["full_logits"].shape, (1, 3))
        self.assertEqual(output["random_logits"].shape, (1, 3))
        self.assertEqual(output["transport"].shape, (40, 4))
        self.assertEqual(output["selection_gate"].shape, (40,))
        self.assertTrue(torch.isfinite(losses["loss"]))
        self.assertTrue(
            torch.allclose(
                output["selected_ratio"], output["random_selected_ratio"]
            )
        )
        self.assertIsNotNone(model.prototypes.grad)
        self.assertTrue(torch.isfinite(model.prototypes.grad).all())

    def test_permutation_invariance(self):
        torch.manual_seed(11)
        model = self._make_model().eval()
        bag = torch.randn(1, 30, 32)
        permutation = torch.randperm(30)

        with torch.no_grad():
            logits = model(bag)["logits"]
            permuted_logits = model(bag[:, permutation])["logits"]

        self.assertTrue(torch.allclose(logits, permuted_logits, atol=1e-5, rtol=1e-5))

    def test_attention_interface(self):
        model = self._make_model().eval()
        output = model(torch.randn(1, 12, 32), return_WSI_attn=True)
        self.assertEqual(output["WSI_attn"].shape, (12, 1))
        self.assertTrue(((output["WSI_attn"] >= 0) & (output["WSI_attn"] <= 1)).all())

    def test_quantile_gate_reduces_selected_mass(self):
        torch.manual_seed(13)
        dense_model = self._make_model().eval()
        sparse_model = self._make_model().eval()
        sparse_model.load_state_dict(dense_model.state_dict())
        sparse_model.selection_fraction = 0.1
        bag = torch.randn(1, 100, 32)

        with torch.no_grad():
            dense_ratio = dense_model(bag)["selected_ratio"]
            sparse_ratio = sparse_model(bag)["selected_ratio"]

        self.assertLess(sparse_ratio.item(), dense_ratio.item())
        self.assertLess(sparse_ratio.item(), 0.25)

    def test_invalid_selection_fraction_is_rejected(self):
        with self.assertRaises(ValueError):
            OT_MIL(selection_fraction=1.0)

    def test_instance_evidence_branch_is_trainable(self):
        torch.manual_seed(17)
        model = OT_MIL(
            in_dim=32,
            hidden_dim=16,
            num_classes=3,
            num_prototypes=4,
            dropout=0.0,
            sinkhorn_iterations=8,
            gate_temperature=0.5,
            max_instances=128,
            instance_evidence_weight=0.5,
        )
        output = model(torch.randn(1, 24, 32))
        loss = output["logits"].sum()
        loss.backward()

        self.assertIsNotNone(model.instance_classifier.weight.grad)
        self.assertTrue(
            torch.isfinite(model.instance_classifier.weight.grad).all()
        )

    def test_legacy_config_uses_optional_defaults(self):
        config = Dict(
            {
                "General": {"MODEL_NAME": "OT_MIL", "num_classes": 2},
                "Model": {
                    "in_dim": 32,
                    "hidden_dim": 16,
                    "num_prototypes": 4,
                    "dropout": 0.0,
                    "sinkhorn_iterations": 8,
                    "epsilon": 0.1,
                    "tau_source": 0.5,
                    "tau_target": 0.5,
                    "gate_temperature": 0.5,
                    "max_instances": 128,
                    "necessity_weight": 0.1,
                    "minimality_weight": 0.01,
                    "diversity_weight": 0.01,
                    "full_classification_weight": 0.25,
                    "consistency_weight": 0.05,
                    "necessity_margin": 1.0,
                },
            }
        )

        model = get_model_from_yaml(config)

        self.assertEqual(model.selection_fraction, 0.0)
        self.assertEqual(model.instance_evidence_weight, 0.0)
        self.assertIsNone(model.instance_classifier)
        self.assertEqual(model.rare_instance_weight, 0.0)
        self.assertIsNone(model.rare_instance_classifier)
        self.assertFalse(model.mass_faithful_transport)
        self.assertFalse(model.learned_evidence_gate)
        self.assertFalse(model.class_conditional_gate)
        self.assertFalse(model.residual_evidence_logits)
        self.assertFalse(model.binary_likelihood_ratio)

    def test_mass_faithful_gate_uses_transport_retention(self):
        model = OT_MIL(
            in_dim=32,
            hidden_dim=16,
            num_classes=2,
            num_prototypes=4,
            dropout=0.0,
            mass_faithful_transport=True,
        )
        row_mass = torch.tensor([1.0 / 6.0, 1.0 / 3.0, 2.0 / 3.0])
        gate = model._selection_gate(row_mass)

        self.assertTrue(torch.all(gate[1:] > gate[:-1]))
        self.assertAlmostEqual(gate[1].item(), 0.5, places=6)

    def test_low_rank_prototype_embedding_reduces_representation_size(self):
        model = OT_MIL(
            in_dim=32,
            hidden_dim=16,
            num_classes=3,
            num_prototypes=4,
            prototype_rank_dim=5,
            dropout=0.0,
        )
        output = model(torch.randn(1, 12, 32), return_WSI_feature=True)

        self.assertEqual(output["WSI_feature"].shape, (1, 56))
        self.assertEqual(
            model.classifier[0].normalized_shape,
            (56,),
        )

    def test_learned_evidence_gate_receives_classification_gradient(self):
        torch.manual_seed(23)
        model = OT_MIL(
            in_dim=32,
            hidden_dim=16,
            num_classes=2,
            num_prototypes=4,
            dropout=0.0,
            sinkhorn_iterations=8,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
        )
        output = model(torch.randn(1, 24, 32))
        torch.nn.functional.cross_entropy(
            output["logits"], torch.tensor([1])
        ).backward()

        self.assertIsNotNone(model.evidence_scorer.weight.grad)
        self.assertGreater(model.evidence_scorer.weight.grad.abs().sum().item(), 0)

    def test_probability_necessity_is_invariant_to_logit_shift(self):
        model = OT_MIL(
            in_dim=32,
            hidden_dim=16,
            num_classes=2,
            num_prototypes=4,
            necessity_log_probability=True,
        )
        output = model(torch.randn(1, 12, 32))
        labels = torch.tensor([1])
        original = model.compute_loss(output, labels)["necessity_loss"]
        shifted = dict(output)
        shifted["logits"] = output["logits"] + 100.0
        shifted["complement_logits"] = output["complement_logits"] - 50.0
        after_shift = model.compute_loss(shifted, labels)["necessity_loss"]

        self.assertTrue(torch.allclose(original, after_shift, atol=1e-5))

    def test_class_conditional_gate_builds_one_submeasure_per_class(self):
        torch.manual_seed(29)
        model = OT_MIL(
            in_dim=32,
            hidden_dim=16,
            num_classes=3,
            num_prototypes=4,
            dropout=0.0,
            sinkhorn_iterations=8,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
        )
        output = model(
            torch.randn(1, 24, 32),
            return_WSI_feature=True,
            return_controls=True,
        )
        torch.nn.functional.cross_entropy(
            output["logits"], torch.tensor([2])
        ).backward()

        self.assertEqual(output["class_selection_gates"].shape, (24, 3))
        self.assertEqual(output["selection_gate"].shape, (24,))
        self.assertEqual(output["WSI_feature"].shape, (3, 100))
        self.assertEqual(output["logits"].shape, (1, 3))
        self.assertIsNotNone(model.evidence_scorer.weight.grad)
        self.assertGreater(model.evidence_scorer.weight.grad.abs().sum().item(), 0)

    def test_class_conditional_gate_requires_learned_scores(self):
        with self.assertRaisesRegex(ValueError, "requires a learned"):
            OT_MIL(class_conditional_gate=True)

    def test_residual_evidence_starts_from_full_bag_prediction(self):
        torch.manual_seed(31)
        model = OT_MIL(
            in_dim=32,
            hidden_dim=16,
            num_classes=3,
            num_prototypes=4,
            dropout=0.0,
            sinkhorn_iterations=8,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
        )
        output = model(torch.randn(1, 24, 32), return_controls=True)

        self.assertTrue(torch.allclose(output["logits"], output["full_logits"]))
        self.assertTrue(
            torch.allclose(output["complement_logits"], output["full_logits"])
        )
        self.assertTrue(
            torch.allclose(output["random_logits"], output["full_logits"])
        )

        loss = torch.nn.functional.cross_entropy(
            output["logits"], torch.tensor([1])
        )
        loss.backward()
        self.assertGreater(
            model.evidence_residual_classifier.weight.grad.abs().sum().item(),
            0,
        )

    def test_binary_likelihood_ratio_is_antisymmetric(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=2,
            num_prototypes=2,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
            binary_likelihood_ratio=True,
        )
        with torch.no_grad():
            model.evidence_scorer.weight.fill_(0.25)
            model.evidence_residual_classifier.weight.fill_(0.5)
        features = torch.ones(3, 4)
        row_mass = torch.full((3,), 1.0 / 3.0)
        gates = model._selection_gate(row_mass, features)

        self.assertTrue(
            torch.allclose(gates.sum(dim=1), torch.ones(3), atol=1e-6)
        )

        representations = torch.randn(2, 14)
        full_representation = torch.randn(1, 14)
        full_logits = torch.randn(1, 2)
        logits = model._evidence_residual_logits(
            representations, full_representation, full_logits
        )
        residual = logits - full_logits
        self.assertTrue(
            torch.allclose(residual[:, 0], -residual[:, 1], atol=1e-6)
        )

    def test_binary_likelihood_ratio_breaks_gate_symmetry(self):
        torch.manual_seed(37)
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=2,
            num_prototypes=2,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
            binary_likelihood_ratio=True,
        )
        output = model(torch.randn(1, 16, 8))
        self.assertTrue(torch.allclose(output["logits"], output["full_logits"]))

        torch.nn.functional.cross_entropy(
            output["logits"], torch.tensor([1])
        ).backward()
        self.assertGreater(model.evidence_scorer.weight.grad.abs().sum(), 0)

    def test_binary_common_gate_adds_shared_diagnostic_quality(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=2,
            num_prototypes=2,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
            binary_likelihood_ratio=True,
            binary_common_gate_weight=0.5,
        )
        with torch.no_grad():
            model.evidence_scorer.weight.zero_()
            model.evidence_scorer.bias.copy_(torch.tensor([2.0, 0.0]))
        gates = model._selection_gate(
            torch.full((3,), 1.0 / 3.0), torch.ones(3, 4)
        )

        self.assertTrue(torch.allclose(gates[:, 0], gates[:, 1]))
        self.assertTrue(torch.all(gates > 0.5))

    def test_binary_common_gate_penalty_regularizes_only_shared_potential(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=2,
            num_prototypes=2,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
            binary_likelihood_ratio=True,
            binary_common_gate_weight=1.0,
            binary_common_gate_penalty_weight=0.5,
        )
        with torch.no_grad():
            model.evidence_scorer.weight[0].fill_(0.25)
            model.evidence_scorer.weight[1].fill_(-0.1)
        output = model(torch.randn(1, 12, 8))
        losses = model.compute_loss(output, torch.tensor([1]))
        losses["loss"].backward()

        self.assertGreater(output["common_gate_energy"], 0)
        self.assertGreater(model.evidence_scorer.weight.grad[0].abs().sum(), 0)

    def test_binary_common_gate_balancing_suppresses_dominant_shared_potential(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=2,
            num_prototypes=2,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
            binary_likelihood_ratio=True,
            binary_common_gate_weight=1.0,
            binary_common_gate_balance_power=2.0,
        )
        with torch.no_grad():
            model.evidence_scorer.weight.zero_()
            model.evidence_scorer.bias.copy_(torch.tensor([4.0, 1.0]))
        output = model(torch.randn(1, 12, 8))

        self.assertTrue(
            torch.allclose(
                output["common_gate_scale"],
                torch.tensor(0.04),
                atol=1e-5,
            )
        )

    def test_binary_common_gate_learns_task_level_coupling(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=2,
            num_prototypes=2,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
            binary_likelihood_ratio=True,
            binary_common_gate_weight=1.0,
            binary_common_gate_learnable_scale=True,
        )
        with torch.no_grad():
            model.evidence_scorer.weight[0].fill_(0.25)
        output = model(torch.randn(1, 12, 8))
        model.compute_loss(output, torch.tensor([1]))["loss"].backward()

        self.assertTrue(
            torch.allclose(output["common_gate_scale"], torch.tensor(0.5))
        )
        self.assertGreater(model.binary_common_gate_logit.grad.abs(), 0)

    def test_binary_dual_gate_mixes_endpoint_probabilities(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=2,
            num_prototypes=2,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
            binary_likelihood_ratio=True,
            binary_common_gate_weight=1.0,
            binary_dual_gate_mix=0.6,
        )
        with torch.no_grad():
            model.evidence_scorer.weight[0].fill_(0.25)
            model.evidence_scorer.weight[1].fill_(-0.1)
        output = model(torch.randn(1, 12, 8))
        expected = (
            0.4 * torch.softmax(output["contrast_logits"], dim=-1)
            + 0.6 * torch.softmax(output["common_logits"], dim=-1)
        )

        self.assertTrue(
            torch.allclose(torch.softmax(output["logits"], dim=-1), expected)
        )

    def test_binary_dual_gate_supervises_both_endpoints(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=2,
            num_prototypes=2,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
            binary_likelihood_ratio=True,
            binary_common_gate_weight=1.0,
            binary_dual_gate_mix=0.6,
            binary_dual_endpoint_weight=1.0,
        )
        output = model(torch.randn(1, 12, 8))
        losses = model.compute_loss(output, torch.tensor([1]))
        expected = 0.5 * (
            torch.nn.functional.cross_entropy(
                output["contrast_logits"], torch.tensor([1])
            )
            + torch.nn.functional.cross_entropy(
                output["common_logits"], torch.tensor([1])
            )
        )

        self.assertTrue(
            torch.allclose(losses["endpoint_classification_loss"], expected)
        )

    def test_binary_likelihood_ratio_rejects_invalid_modes(self):
        with self.assertRaisesRegex(ValueError, "requires two classes"):
            OT_MIL(
                num_classes=3,
                learned_evidence_gate=True,
                class_conditional_gate=True,
                residual_evidence_logits=True,
                binary_likelihood_ratio=True,
            )
        with self.assertRaisesRegex(ValueError, "requires class-conditional"):
            OT_MIL(num_classes=2, binary_likelihood_ratio=True)
        with self.assertRaisesRegex(ValueError, "requires likelihood-ratio"):
            OT_MIL(num_classes=2, binary_common_gate_weight=0.5)
        with self.assertRaisesRegex(ValueError, "requires common-gate"):
            OT_MIL(
                num_classes=2,
                binary_common_gate_penalty_weight=0.5,
            )
        with self.assertRaisesRegex(ValueError, "requires common-gate"):
            OT_MIL(
                num_classes=2,
                binary_common_gate_balance_power=1.0,
            )
        with self.assertRaisesRegex(ValueError, "requires common-gate"):
            OT_MIL(
                num_classes=2,
                binary_common_gate_learnable_scale=True,
            )
        with self.assertRaisesRegex(ValueError, "requires common-gate"):
            OT_MIL(num_classes=2, binary_dual_gate_mix=0.5)
        with self.assertRaisesRegex(ValueError, "requires a dual-gate"):
            OT_MIL(num_classes=2, binary_dual_endpoint_weight=0.5)

    def test_class_prototype_routing_preserves_mass_scale_and_is_trainable(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=3,
            num_prototypes=4,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            class_prototype_routing=True,
            class_prototype_separation_weight=0.1,
            residual_evidence_logits=True,
        )
        weights = model._class_prototype_weights()
        output = model(torch.randn(1, 12, 8))
        losses = model.compute_loss(output, torch.tensor([1]))
        losses["loss"].backward()

        self.assertEqual(weights.shape, (3, 4))
        self.assertTrue(
            torch.allclose(weights.mean(dim=1), torch.ones(3), atol=1e-6)
        )
        self.assertEqual(output["class_prototype_weights"].shape, (3, 4))
        self.assertGreater(model.class_prototype_logits.grad.abs().sum(), 0)

    def test_class_prototype_routing_rejects_invalid_modes(self):
        with self.assertRaisesRegex(ValueError, "requires class-conditional"):
            OT_MIL(num_classes=3, class_prototype_routing=True)
        with self.assertRaisesRegex(ValueError, "requires prototype routing"):
            OT_MIL(
                num_classes=3,
                class_prototype_separation_weight=0.1,
            )
        with self.assertRaisesRegex(ValueError, "requires prototype routing"):
            OT_MIL(
                num_classes=3,
                class_prototype_information_weight=0.1,
            )
        with self.assertRaisesRegex(ValueError, "requires prototype routing"):
            OT_MIL(
                num_classes=3,
                class_prototype_init_strength=1.0,
            )

    def test_class_gate_competition_preserves_uniform_gate_initialization(self):
        common = dict(
            in_dim=8,
            hidden_dim=4,
            num_classes=3,
            num_prototypes=4,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
        )
        baseline = OT_MIL(**common)
        competitive = OT_MIL(**common, class_gate_competition=True)
        features = torch.randn(10, 4)
        row_mass = torch.full((10,), 0.1)

        self.assertTrue(
            torch.allclose(
                baseline._selection_gate(row_mass, features),
                competitive._selection_gate(row_mass, features),
            )
        )

    def test_class_gate_competition_allocates_evidence_between_classes(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=3,
            num_prototypes=4,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            class_gate_competition=True,
            residual_evidence_logits=True,
        )
        with torch.no_grad():
            model.evidence_scorer.bias.copy_(torch.tensor([3.0, 0.0, -3.0]))
        gates = model._selection_gate(
            torch.full((6,), 1.0 / 6.0), torch.ones(6, 4)
        )

        self.assertTrue(torch.all(gates[:, 0] > gates[:, 1]))
        self.assertTrue(torch.all(gates[:, 1] > gates[:, 2]))
        self.assertTrue(torch.all((gates >= 0) & (gates <= 1)))

    def test_class_gate_competition_rejects_invalid_modes(self):
        with self.assertRaisesRegex(ValueError, "requires multiclass"):
            OT_MIL(num_classes=3, class_gate_competition=True)
        with self.assertRaisesRegex(ValueError, "requires multiclass"):
            OT_MIL(
                num_classes=2,
                learned_evidence_gate=True,
                class_conditional_gate=True,
                class_gate_competition=True,
            )
        with self.assertRaisesRegex(ValueError, "temperature must be positive"):
            OT_MIL(num_classes=3, class_gate_competition_temperature=0.0)
        with self.assertRaisesRegex(ValueError, "strength must be"):
            OT_MIL(num_classes=3, class_gate_competition_strength=1.1)

    def test_zero_class_gate_competition_strength_is_exact_baseline(self):
        common = dict(
            in_dim=8,
            hidden_dim=4,
            num_classes=3,
            num_prototypes=4,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
        )
        baseline = OT_MIL(**common)
        competitive = OT_MIL(
            **common,
            class_gate_competition=True,
            class_gate_competition_strength=0.0,
        )
        with torch.no_grad():
            scores = torch.tensor([2.0, 0.0, -2.0])
            baseline.evidence_scorer.bias.copy_(scores)
            competitive.evidence_scorer.bias.copy_(scores)
        features = torch.randn(10, 4)
        row_mass = torch.full((10,), 0.1)

        self.assertTrue(
            torch.allclose(
                baseline._selection_gate(row_mass, features),
                competitive._selection_gate(row_mass, features),
            )
        )

    def test_class_prototype_routing_does_not_shift_shared_initialization(self):
        common = dict(
            in_dim=8,
            hidden_dim=4,
            num_classes=3,
            num_prototypes=4,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            residual_evidence_logits=True,
        )
        torch.manual_seed(53)
        baseline = OT_MIL(**common)
        torch.manual_seed(53)
        routed = OT_MIL(**common, class_prototype_routing=True)

        baseline_state = baseline.state_dict()
        routed_state = routed.state_dict()
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, routed_state[name]), name)

    def test_class_prototype_information_rewards_specialized_coverage(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=3,
            num_prototypes=4,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            class_prototype_routing=True,
            class_prototype_information_weight=1.0,
            residual_evidence_logits=True,
        )
        bag = torch.randn(1, 12, 8)
        uniform_loss = model.compute_loss(
            model(bag), torch.tensor([1])
        )["class_prototype_information_loss"]
        with torch.no_grad():
            model.class_prototype_logits.fill_(-8.0)
            model.class_prototype_logits[0, 0] = 8.0
            model.class_prototype_logits[1, 1] = 8.0
            model.class_prototype_logits[2, 2] = 8.0
        specialized_loss = model.compute_loss(
            model(bag), torch.tensor([1])
        )["class_prototype_information_loss"]

        self.assertLess(specialized_loss, uniform_loss - 0.5)

    def test_class_prototype_initialization_balances_class_supports(self):
        model = OT_MIL(
            in_dim=8,
            hidden_dim=4,
            num_classes=4,
            num_prototypes=8,
            prototype_rank_dim=2,
            dropout=0.0,
            mass_faithful_transport=True,
            learned_evidence_gate=True,
            class_conditional_gate=True,
            class_prototype_routing=True,
            class_prototype_init_strength=2.0,
            residual_evidence_logits=True,
        )
        routes = model.class_prototype_logits.argmax(dim=0)

        self.assertTrue(
            torch.equal(routes, torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]))
        )
        self.assertTrue(
            torch.equal(
                torch.bincount(routes, minlength=4),
                torch.full((4,), 2),
            )
        )

    def test_rare_instance_branch_is_trainable(self):
        torch.manual_seed(19)
        model = OT_MIL(
            in_dim=32,
            hidden_dim=16,
            num_classes=2,
            num_prototypes=4,
            dropout=0.0,
            sinkhorn_iterations=8,
            gate_temperature=0.5,
            max_instances=128,
            rare_instance_weight=0.5,
            rare_instance_topk=4,
            rare_gate_weight=0.25,
        )
        output = model(torch.randn(1, 24, 32), return_controls=True)
        loss = output["logits"].sum()
        loss.backward()

        self.assertEqual(output["rare_instance_scores"].shape, (24,))
        self.assertIsNotNone(model.rare_instance_classifier.weight.grad)
        self.assertTrue(
            torch.isfinite(model.rare_instance_classifier.weight.grad).all()
        )

    def test_rare_instance_branch_rejects_multiclass_use(self):
        with self.assertRaisesRegex(ValueError, "requires two classes"):
            OT_MIL(num_classes=3, rare_instance_weight=0.5)

    def test_warmup_scheduler_matches_optimizer_groups(self):
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        scheduler = WarmUpLR(optimizer, warmup_epochs=2, base_lr=2e-4)
        self.assertEqual(len(scheduler.get_last_lr()), len(optimizer.param_groups))
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-4)


if __name__ == "__main__":
    unittest.main()
