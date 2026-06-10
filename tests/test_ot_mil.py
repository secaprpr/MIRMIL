import unittest

import torch

from modules.OT_MIL.ot_mil import OT_MIL
from utils.model_utils import WarmUpLR


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

    def test_warmup_scheduler_matches_optimizer_groups(self):
        model = self._make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)
        scheduler = WarmUpLR(optimizer, warmup_epochs=2, base_lr=2e-4)
        self.assertEqual(len(scheduler.get_last_lr()), len(optimizer.param_groups))
        self.assertAlmostEqual(scheduler.get_last_lr()[0], 1e-4)


if __name__ == "__main__":
    unittest.main()
