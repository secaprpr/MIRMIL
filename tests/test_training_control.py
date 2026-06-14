import unittest
from types import SimpleNamespace

import torch

from utils.general_utils import cal_is_stopping
from utils.model_utils import ClampedCosineAnnealingLR


class ConfigNamespace(SimpleNamespace):
    def __contains__(self, key):
        return hasattr(self, key)


def earlystop_args(patience=3, min_delta=0.0):
    return SimpleNamespace(
        General=SimpleNamespace(
            earlystop=ConfigNamespace(
                use=True,
                patience=patience,
                metric="acc",
                min_delta=min_delta,
            )
        )
    )


class TrainingControlTest(unittest.TestCase):
    def test_earlystop_uses_global_best(self):
        log = {
            "epoch": [1, 2, 3, 4, 5],
            "val_acc": [0.8, 0.5, 0.6, 0.7, 0.75],
        }
        self.assertTrue(cal_is_stopping(earlystop_args(), log, "Train_Val"))

    def test_earlystop_respects_min_delta(self):
        log = {
            "epoch": [1, 2, 3, 4],
            "val_acc": [0.8, 0.8004, 0.8005, 0.8009],
        }
        self.assertTrue(
            cal_is_stopping(
                earlystop_args(patience=3, min_delta=0.001),
                log,
                "Train_Val",
            )
        )

    def test_clamped_cosine_does_not_reheat(self):
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=1.0)
        scheduler = ClampedCosineAnnealingLR(
            optimizer, T_max=4, eta_min=0.1
        )
        learning_rates = []
        for _ in range(8):
            optimizer.step()
            scheduler.step()
            learning_rates.append(optimizer.param_groups[0]["lr"])

        self.assertAlmostEqual(learning_rates[3], 0.1)
        self.assertEqual(learning_rates[3:], [0.1] * 5)


if __name__ == "__main__":
    unittest.main()
