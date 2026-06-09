import unittest

import pandas as pd

from experiments.prepare_split import deterministic_stratified_split


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


if __name__ == "__main__":
    unittest.main()
