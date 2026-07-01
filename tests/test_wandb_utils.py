from addict import Dict

from utils.wandb_utils import (
    build_training_config,
    build_training_identity,
    hash_manifest,
    metric_payload,
)


def make_config(tmp_path):
    split = tmp_path / "split.csv"
    split.write_text(
        "train_slide_path,train_label\nslide.pt,0\n",
        encoding="utf-8",
    )
    return Dict(
        {
            "General": {
                "MODEL_NAME": "MIR_MIL",
                "seed": 2024,
                "num_classes": 6,
                "num_epochs": 30,
                "best_model_metric": "macro_auc",
                "earlystop": {
                    "use": True,
                    "patience": 8,
                    "metric": "macro_auc",
                },
            },
            "Dataset": {
                "DATASET_NAME": "PANDA",
                "dataset_csv_path": str(split),
                "balanced_sampler": {
                    "use": True,
                    "replacement": True,
                },
            },
            "Model": {
                "in_dim": 1024,
                "max_instances": 512,
                "sampling": "random",
            },
            "Tracking": {
                "wandb": {
                    "enabled": False,
                    "project": "MIR-MIL",
                    "feature": "uni",
                    "variant": "base",
                    "protocol": "protocol_v1",
                    "split_id": "panda_v1",
                    "comparison_id": "encoder_comparison_v1",
                    "tags": [],
                }
            },
        }
    )


def test_training_identity_is_stable_and_seed_specific(tmp_path):
    config = make_config(tmp_path)
    _, name, group, experiment_key = build_training_identity(config)
    assert name == "PANDA_uni_MIR_MIL_base_seed2024_train"
    assert group == "PANDA_uni_MIR_MIL_base_protocol_v1_panda_v1"
    assert experiment_key.startswith(name)


def test_training_config_records_split_hash_and_feature(tmp_path):
    config = make_config(tmp_path)
    payload = build_training_config(config, "Train_Val")
    assert len(payload["split"]["sha256"]) == 64
    assert payload["features"]["encoder"] == "uni"
    assert payload["training"]["seed"] == 2024
    assert payload["split"]["process_pipeline"] == "Train_Val"


def test_metric_payload_uses_unambiguous_names():
    payload = metric_payload(
        "val",
        {
            "acc": 0.7,
            "bacc": 0.6,
            "macro_auc": 0.8,
            "macro_f1": 0.65,
            "confusion_mat": [[1, 0], [0, 1]],
        },
    )
    assert payload == {
        "val/accuracy": 0.7,
        "val/balanced_accuracy": 0.6,
        "val/macro_auc_ovr": 0.8,
        "val/macro_f1": 0.65,
    }


def test_hash_manifest_is_order_independent(tmp_path):
    first = tmp_path / "a.txt"
    second = tmp_path / "b.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")
    forward = hash_manifest([first, second])
    reverse = hash_manifest([second, first])
    assert forward["sha256"] == reverse["sha256"]
    assert len(forward["files"]) == 2
