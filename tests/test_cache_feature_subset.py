import os
import tempfile
import unittest

import h5py
import numpy as np
import pandas as pd
import torch

from experiments.cache_feature_subset import (
    cache_feature,
    candidate_indices,
    feature_paths,
)


class CacheFeatureSubsetTest(unittest.TestCase):
    def test_candidate_indices_include_bag_endpoints(self):
        indices = candidate_indices(10, 4)

        np.testing.assert_array_equal(indices, np.array([0, 3, 6, 9]))

    def test_cache_is_deterministic_and_reusable(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, "slide.h5")
            output_dir = os.path.join(directory, "cache")
            os.makedirs(output_dir)
            features = np.arange(40, dtype=np.float32).reshape(10, 4)
            with h5py.File(source, "w") as file:
                file.create_dataset("features", data=features)

            first_path, rows, dimensions, first_created = cache_feature(
                source, output_dir, 4
            )
            second_path, _, _, second_created = cache_feature(
                source, output_dir, 4
            )
            cached = torch.load(
                first_path, map_location="cpu", weights_only=True
            )

            self.assertEqual(first_path, second_path)
            self.assertEqual((rows, dimensions), (4, 4))
            self.assertTrue(first_created)
            self.assertFalse(second_created)
            np.testing.assert_array_equal(
                cached.numpy(), features[[0, 3, 6, 9]]
            )

    def test_feature_paths_are_unique_and_ordered(self):
        frame = pd.DataFrame(
            {
                "train_slide_path": ["a.h5", "b.h5"],
                "val_slide_path": ["b.h5", "c.h5"],
                "test_slide_path": [None, "d.h5"],
            }
        )

        self.assertEqual(
            feature_paths(frame), ["a.h5", "b.h5", "c.h5", "d.h5"]
        )


if __name__ == "__main__":
    unittest.main()
