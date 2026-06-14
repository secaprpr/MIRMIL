import unittest
from argparse import Namespace
from tempfile import NamedTemporaryFile

import pandas as pd

from experiments.prepare_split import (
    deterministic_group_stratified_split,
    deterministic_stratified_split,
)
from experiments.evaluate_checkpoints import (
    experiment_variant,
    file_sha256 as evaluation_file_sha256,
)
from experiments.run_benchmark import build_command, file_sha256
from utils.yaml_utils import read_yaml, update_config_from_options


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
            best_model_metric="macro_auc",
            earlystop_metric="macro_auc",
            earlystop_min_delta=0.001,
            scheduler_t_max=40,
            clamp_cosine=True,
            model_option=[],
            dataset_name="CAMELYON16",
            split="/tmp/split.csv",
            balanced=True,
            log_root="/tmp/logs",
            in_dim=1024,
            max_instances=4096,
        )
        ab_command = build_command(args, "AB_MIL", 2024)
        mir_command = build_command(args, "MIR_MIL", 2024)
        ot_command = build_command(args, "OT_MIL_CLASS_MASS", 2024)
        mo_command = build_command(args, "MO_MIL", 2024)

        for command in (ab_command, mir_command, ot_command, mo_command):
            self.assertIn("Model.max_instances=4096", command)
            self.assertIn("Model.sampling=random", command)
            self.assertIn("General.num_epochs=30", command)
            self.assertIn("Dataset.balanced_sampler.use=true", command)
        self.assertIn("General.experiment_variant=AB_MIL", ab_command)
        self.assertIn("General.experiment_variant=MIR_MIL", mir_command)
        self.assertIn(
            "General.experiment_variant=OT_MIL_CLASS_MASS", ot_command
        )
        self.assertIn(
            "Model.class_mass_classification_weight=0.1", ot_command
        )
        self.assertIn("Model.scheduler.cosine_config.T_max=40", ot_command)
        self.assertIn("Model.scheduler.cosine_config.T_max=40", mir_command)
        self.assertIn(
            "Model.scheduler.cosine_config.clamp_after_t_max=true",
            mir_command,
        )
        self.assertIn("General.best_model_metric=macro_auc", mir_command)
        self.assertIn("General.earlystop.metric=macro_auc", mir_command)
        self.assertIn("General.earlystop.min_delta=0.001", mir_command)

    def test_group_split_prevents_patient_leakage(self):
        frame = pd.DataFrame(
            {
                "slide_id": [f"slide_{index:02d}" for index in range(24)],
                "patient_id": [
                    f"patient_{label}_{patient}"
                    for label in (0, 1)
                    for patient in range(6)
                    for _ in range(2)
                ],
                "label": [0] * 12 + [1] * 12,
                "slide_path": [f"/tmp/{index}.pt" for index in range(24)],
            }
        )
        first = deterministic_group_stratified_split(
            frame, 2024, 0.5, 0.25, "patient_id"
        )
        second = deterministic_group_stratified_split(
            frame, 2024, 0.5, 0.25, "patient_id"
        )

        self.assertTrue(first.equals(second))
        self.assertEqual(first.groupby("patient_id")["split"].nunique().max(), 1)
        patient_counts = (
            first.drop_duplicates("patient_id")
            .groupby(["label", "split"])
            .size()
        )
        for label in (0, 1):
            self.assertEqual(patient_counts[label, "train"], 3)
            self.assertEqual(patient_counts[label, "val"], 2)
            self.assertEqual(patient_counts[label, "test"], 1)

    def test_group_split_rejects_mixed_labels(self):
        frame = pd.DataFrame(
            {
                "slide_id": ["a", "b"],
                "patient_id": ["patient", "patient"],
                "label": [0, 1],
            }
        )
        with self.assertRaisesRegex(ValueError, "multiple labels"):
            deterministic_group_stratified_split(
                frame, 2024, 0.6, 0.2, "patient_id"
            )

    def test_file_sha256_is_stable(self):
        with NamedTemporaryFile() as file:
            file.write(b"fixed split content")
            file.flush()
            first = file_sha256(file.name)
            second = file_sha256(file.name)
            evaluation_hash = evaluation_file_sha256(file.name)
        self.assertEqual(first, second)
        self.assertEqual(first, evaluation_hash)
        self.assertEqual(len(first), 64)

    def test_benchmark_variant_survives_saved_config(self):
        args = read_yaml("configs/OT_MIL_MULTICLASS.yaml")
        update_config_from_options(
            args,
            ["General.experiment_variant=OT_MIL_CLASS_MASS"],
        )
        self.assertEqual(experiment_variant(args), "OT_MIL_CLASS_MASS")


if __name__ == "__main__":
    unittest.main()
