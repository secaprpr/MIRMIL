import os
import tempfile
import unittest

import yaml

from utils.yaml_utils import change_yaml_by_options, read_yaml, update_config_from_options


class YamlUtilsTest(unittest.TestCase):
    def test_zero_float_and_boolean_overrides(self):
        config = read_yaml("configs/OT_MIL.yaml")
        updated = update_config_from_options(
            config,
            ["Model.necessity_weight=0", "Dataset.balanced_sampler.use=true"],
        )
        self.assertEqual(updated.Model.necessity_weight, 0)
        self.assertIs(updated.Dataset.balanced_sampler.use, True)

    def test_saved_override_is_plain_yaml(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "config.yaml")
            with open(path, "w", encoding="utf-8") as file:
                yaml.safe_dump({"Model": {"weight": 0.5}}, file)
            change_yaml_by_options(path, ["Model.weight=0"])
            with open(path, encoding="utf-8") as file:
                saved = yaml.safe_load(file)
            self.assertEqual(saved["Model"]["weight"], 0)


if __name__ == "__main__":
    unittest.main()
