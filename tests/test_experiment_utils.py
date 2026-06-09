import unittest
from argparse import Namespace
from tempfile import NamedTemporaryFile

import pandas as pd

from experiments.prepare_split import deterministic_stratified_split
from experiments.run_benchmark import build_command, file_sha256


class ExperimentUtilsTest(unittest.TestCase):
    def test_split_is_deterministic_and_stratified(self):
        frame = pd.DataFrame(
            {
                "slide_id": [f"slide_{index:02d}" for index in range(20)],
                "label": [0] * 10 + [1] * 10,
                "slide_path": [f"/tmp/{index}.pt" for index in range(20)],
            }
        )
        first = deterministic_stratified_split(frame, 2024, 0.6, 0.2)
        second = deterministic_stratified_split(frame, 2024, 0.6, 0.2)
        self.assertTrue(first.equals(second))
        counts = first.groupby(["label", "split"]).size()
        for label in (0, 1):
            self.assertEqual(counts[label, "train"], 6)
            self.assertEqual(counts[label, "val"], 2)
            self.assertEqual(counts[label, "test"], 2)

    def test_benchmark_command_uses_shared_budget(self):
        args = Namespace(
            python="python",
            num_classes=2,
            epochs=30,
            device=0,
            num_workers=4,
            patience=8,
            dataset_name="CAMELYON16",
            split="/tmp/split.csv",
            balanced=True,
            log_root="/tmp/logs",
            in_dim=1024,
            max_instances=4096,
        )
        ot_command = build_command(args, "OT_MIL", 2024)
        mo_command = build_command(args, "MO_MIL", 2024)

        for command in (ot_command, mo_command):
            self.assertIn("Model.max_instances=4096", command)
            self.assertIn("Model.sampling=random", command)
            self.assertIn("General.num_epochs=30", command)
            self.assertIn("Dataset.balanced_sampler.use=true", command)
        self.assertIn("Model.scheduler.cosine_config.T_max=28", ot_command)

    def test_file_sha256_is_stable(self):
        with NamedTemporaryFile() as file:
            file.write(b"fixed split content")
            file.flush()
            first = file_sha256(file.name)
            second = file_sha256(file.name)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)


if __name__ == "__main__":
    unittest.main()
