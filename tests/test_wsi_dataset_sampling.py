import os
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
import torch
from addict import Dict

from utils.wsi_utils import (
    WSI_Coord_Dataset,
    WSI_Dataset,
    build_wsi_datasets,
)


class WSIDatasetSamplingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.features = np.arange(200, dtype=np.float32).reshape(100, 2)
        self.h5_path = os.path.join(self.temp_dir.name, "slide.h5")
        with h5py.File(self.h5_path, "w") as file:
            file.create_dataset("features", data=self.features)
            file.create_dataset(
                "coords",
                data=np.arange(200, dtype=np.float32).reshape(100, 2),
            )
        self.uni_h5_path = os.path.join(self.temp_dir.name, "uni_slide.h5")
        with h5py.File(self.uni_h5_path, "w") as file:
            file.create_dataset("features", data=self.features[None, ...])
        self.pt_path = os.path.join(self.temp_dir.name, "slide.pt")
        torch.save(torch.from_numpy(self.features), self.pt_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _csv(self, path):
        csv_path = os.path.join(self.temp_dir.name, os.path.basename(path) + ".csv")
        pd.DataFrame(
            {
                "train_slide_path": [path],
                "train_label": [1],
                "val_slide_path": [path],
                "val_label": [1],
                "test_slide_path": [path],
                "test_label": [1],
            }
        ).to_csv(csv_path, index=False)
        return csv_path

    def test_uniform_sampling_matches_for_h5_and_pt(self):
        h5_dataset = WSI_Dataset(
            self._csv(self.h5_path), "val", max_instances=10, sampling="uniform"
        )
        pt_dataset = WSI_Dataset(
            self._csv(self.pt_path), "val", max_instances=10, sampling="uniform"
        )
        h5_features, _ = h5_dataset[0]
        pt_features, _ = pt_dataset[0]
        self.assertEqual(h5_features.shape, (10, 2))
        self.assertTrue(torch.equal(h5_features, pt_features))

    def test_head_sampling(self):
        dataset = WSI_Dataset(
            self._csv(self.h5_path), "test", max_instances=8, sampling="head"
        )
        features, _ = dataset[0]
        self.assertTrue(torch.equal(features, torch.from_numpy(self.features[:8])))

    def test_uniform_sampling_supports_uni_batched_h5(self):
        dataset = WSI_Dataset(
            self._csv(self.uni_h5_path),
            "val",
            max_instances=10,
            sampling="uniform",
        )
        features, _ = dataset[0]
        indices = np.linspace(0, 99, 10, dtype=np.int64)
        expected = torch.from_numpy(self.features[indices])

        self.assertEqual(features.shape, (10, 2))
        self.assertTrue(torch.equal(features, expected))

    def test_coordinate_dataset_samples_and_normalizes_coordinates(self):
        dataset = WSI_Coord_Dataset(
            self._csv(self.h5_path),
            "val",
            max_instances=10,
            sampling="uniform",
        )

        values, _ = dataset[0]

        self.assertEqual(values.shape, (10, 4))
        self.assertLessEqual(values[:, -2:].abs().max().item(), 1.0)

    def test_build_wsi_datasets_uses_random_train_and_uniform_eval(self):
        config = Dict(
            {
                "Dataset": {"dataset_csv_path": self._csv(self.h5_path)},
                "Model": {
                    "max_instances": 11,
                    "sampling": "random",
                },
            }
        )
        train, val, test = build_wsi_datasets(config)
        self.assertEqual(train.max_instances, 11)
        self.assertEqual(train.sampling, "random")
        self.assertEqual(val.sampling, "uniform")
        self.assertEqual(test.sampling, "uniform")


if __name__ == "__main__":
    unittest.main()
