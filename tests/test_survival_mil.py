import os
import tempfile
import unittest

import pandas as pd
import torch
from addict import Dict

from modules.AB_MIL.ab_mil import AB_MIL
from modules.SURVIVAL_MIL import SurvivalMILWrapper
from process.SURVIVAL_MIL.process_survival_mil import process_SURVIVAL_MIL
from utils.survival_dataset import SurvivalWSIDataset
from utils.survival_utils import (
    NLLSurvLoss,
    concordance_index,
    discretize_survival_times,
    fit_discrete_time_cutpoints,
)


class SurvivalMILTest(unittest.TestCase):
    def test_cutpoints_and_c_index(self):
        times = [5, 10, 15, 20, 25]
        events = [1, 0, 1, 1, 0]
        cutpoints = fit_discrete_time_cutpoints(times, events, num_bins=4)
        labels = discretize_survival_times(times, cutpoints)
        self.assertEqual(len(labels), len(times))
        self.assertTrue(labels.max() <= 3)

        c_index = concordance_index(
            event_times=[1, 2, 3],
            events=[1, 1, 0],
            risks=[3.0, 2.0, 1.0],
        )
        self.assertEqual(c_index, 1.0)

    def test_survival_wrapper_outputs_hazards(self):
        backbone = AB_MIL(L=8, D=4, num_classes=4, in_dim=6)
        model = SurvivalMILWrapper(backbone, num_bins=4)
        output = model(torch.randn(1, 5, 6))
        self.assertEqual(output["hazards"].shape, (1, 4))
        loss = NLLSurvLoss()(
            output["hazards"],
            output["survival"],
            torch.tensor([1]),
            torch.tensor([1.0]),
        )
        self.assertTrue(torch.isfinite(loss))

    def test_dataset_and_tiny_process(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = []
            for index in range(6):
                path = os.path.join(tmpdir, f"slide_{index}.pt")
                torch.save(torch.randn(8, 6), path)
                paths.append(path)

            df = pd.DataFrame(
                {
                    "train_slide_path": paths[:4] + [None, None],
                    "train_time_months": [5, 7, 9, 11, None, None],
                    "train_event": [1, 0, 1, 1, None, None],
                    "val_slide_path": [paths[4], None, None, None, None, None],
                    "val_time_months": [13, None, None, None, None, None],
                    "val_event": [1, None, None, None, None, None],
                    "test_slide_path": [paths[5], None, None, None, None, None],
                    "test_time_months": [15, None, None, None, None, None],
                    "test_event": [0, None, None, None, None, None],
                }
            )
            csv_path = os.path.join(tmpdir, "survival.csv")
            df.to_csv(csv_path, index=False)

            dataset = SurvivalWSIDataset(
                csv_path, "train", fit_cutpoints=True, num_bins=4
            )
            self.assertEqual(len(dataset), 4)

            args = Dict(
                {
                    "General": {
                        "MODEL_NAME": "SURVIVAL_MIL",
                        "seed": 2024,
                        "num_classes": 4,
                        "num_epochs": 1,
                        "device": 0,
                        "num_workers": 0,
                        "best_model_metric": "c_index",
                        "earlystop": {
                            "use": False,
                            "patience": 2,
                            "metric": "c_index",
                        },
                    },
                    "Dataset": {
                        "DATASET_NAME": "tiny_survival",
                        "dataset_csv_path": csv_path,
                        "balanced_sampler": {"use": False, "replacement": True},
                    },
                    "Logs": {"now_log_dir": tmpdir},
                    "Model": {
                        "backbone": "AB_MIL",
                        "in_dim": 6,
                        "L": 8,
                        "D": 4,
                        "dropout": 0.0,
                        "act": "relu",
                        "max_instances": 4,
                        "sampling": "uniform",
                        "survival": {
                            "num_bins": 4,
                            "alpha": 0.0,
                            "time_column": "time_months",
                            "event_column": "event",
                            "representation": "auto",
                            "backbone_num_outputs": 4,
                        },
                        "optimizer": {
                            "which": "adam",
                            "adam_config": {"lr": 0.001, "weight_decay": 0.0},
                            "adamw_config": {"lr": 0.001, "weight_decay": 0.0},
                        },
                        "scheduler": {
                            "warmup": 0,
                            "which": "none",
                        },
                    },
                }
            )
            if not torch.cuda.is_available():
                self.skipTest(
                    "process_SURVIVAL_MIL follows the project CUDA convention"
                )
            process_SURVIVAL_MIL(args)


if __name__ == "__main__":
    unittest.main()
