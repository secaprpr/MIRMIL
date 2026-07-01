import hashlib
import json
import math
import os
from pathlib import Path


SCHEMA_VERSION = 1
_ACTIVE_TRACKER = None


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value):
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if hasattr(value, "to_dict"):
        return _plain(value.to_dict())
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _finite_scalar(value):
    value = _plain(value)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value if math.isfinite(float(value)) else None
    return value


def metric_payload(prefix, metrics):
    mapping = {
        "acc": "accuracy",
        "bacc": "balanced_accuracy",
        "macro_auc": "macro_auc_ovr",
        "macro_f1": "macro_f1",
    }
    return {
        f"{prefix}/{target}": _finite_scalar(metrics[source])
        for source, target in mapping.items()
        if source in metrics
    }


def tracking_settings(config):
    tracking = getattr(config, "Tracking", {})
    wandb_config = getattr(tracking, "wandb", {})
    return {
        "enabled": bool(getattr(wandb_config, "enabled", False)),
        "project": str(getattr(wandb_config, "project", "MIR-MIL")),
        "entity": getattr(wandb_config, "entity", None),
        "mode": str(getattr(wandb_config, "mode", "online")),
        "tags": list(getattr(wandb_config, "tags", [])),
        "feature": str(getattr(wandb_config, "feature", "unknown")),
        "variant": str(
            getattr(
                wandb_config,
                "variant",
                getattr(config.General, "experiment_variant", "base"),
            )
        ),
        "protocol": str(getattr(wandb_config, "protocol", "default")),
        "split_id": str(getattr(wandb_config, "split_id", "unspecified")),
        "comparison_id": getattr(wandb_config, "comparison_id", None),
        "source_manifest_sha256": getattr(
            wandb_config, "source_manifest_sha256", None
        ),
        "feature_manifest_sha256": getattr(
            wandb_config, "feature_manifest_sha256", None
        ),
        "coordinate_manifest_sha256": getattr(
            wandb_config, "coordinate_manifest_sha256", None
        ),
        "encoder_checkpoint_sha256": getattr(
            wandb_config, "encoder_checkpoint_sha256", None
        ),
        "upload_checkpoints": bool(
            getattr(wandb_config, "upload_checkpoints", False)
        ),
        "max_artifact_mb": float(wandb_config.get("max_artifact_mb", 50)),
    }


def build_training_identity(config, job_type="train"):
    settings = tracking_settings(config)
    dataset = str(config.Dataset.DATASET_NAME)
    model = str(config.General.MODEL_NAME)
    seed = int(config.General.seed)
    name = (
        f"{dataset}_{settings['feature']}_{model}_"
        f"{settings['variant']}_seed{seed}_{job_type}"
    )
    group = (
        f"{dataset}_{settings['feature']}_{model}_{settings['variant']}_"
        f"{settings['protocol']}_{settings['split_id']}"
    )
    experiment_key = name + "_" + settings["protocol"] + "_" + settings["split_id"]
    return settings, name, group, experiment_key


def build_training_config(config, process_pipeline=None):
    settings, _, _, experiment_key = build_training_identity(config)
    split_path = os.path.abspath(config.Dataset.dataset_csv_path)
    split_hash = file_sha256(split_path)
    model_settings = _plain(config.Model)
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_key": experiment_key,
        "comparison_id": settings["comparison_id"],
        "dataset": {
            "name": str(config.Dataset.DATASET_NAME),
            "num_classes": int(config.General.num_classes),
            "source_manifest_sha256": settings[
                "source_manifest_sha256"
            ],
        },
        "split": {
            "id": settings["split_id"],
            "path": split_path,
            "sha256": split_hash,
            "process_pipeline": process_pipeline,
        },
        "features": {
            "encoder": settings["feature"],
            "checkpoint_sha256": settings[
                "encoder_checkpoint_sha256"
            ],
            "manifest_sha256": settings["feature_manifest_sha256"],
            "coordinate_manifest_sha256": settings[
                "coordinate_manifest_sha256"
            ],
            "feature_dim": int(config.Model.in_dim),
        },
        "model": {
            "name": str(config.General.MODEL_NAME),
            "variant": settings["variant"],
            "settings": model_settings,
        },
        "training": {
            "seed": int(config.General.seed),
            "epochs_cap": int(config.General.num_epochs),
            "best_model_metric": str(config.General.best_model_metric),
            "earlystop": _plain(config.General.earlystop),
            "balanced_sampler": _plain(config.Dataset.balanced_sampler),
        },
        "evaluation": {
            "protocol": settings["protocol"],
            "sampling": "uniform",
            "patch_budget": (
                int(config.Model.max_instances)
                if "max_instances" in config.Model
                else 0
            ),
        },
    }


class WandbTracker:
    def __init__(self, run=None, max_artifact_mb=50):
        self.run = run
        self.max_artifact_bytes = int(float(max_artifact_mb) * 1024 * 1024)

    @property
    def enabled(self):
        return self.run is not None

    @classmethod
    def for_training(cls, config, process_pipeline=None):
        settings, name, group, _ = build_training_identity(config)
        if not settings["enabled"]:
            return cls()
        wandb = _import_wandb()
        tags = _normalized_tags(
            settings["tags"],
            config.Dataset.DATASET_NAME,
            settings["feature"],
            config.General.MODEL_NAME,
        )
        run = wandb.init(
            project=settings["project"],
            entity=settings["entity"] or None,
            name=name,
            group=group,
            job_type="train",
            tags=tags,
            mode=settings["mode"],
            dir=config.Logs.now_log_dir,
            config=build_training_config(config, process_pipeline),
        )
        run.define_metric("epoch")
        run.define_metric("train/*", step_metric="epoch")
        run.define_metric("val/*", step_metric="epoch")
        return cls(run, settings["max_artifact_mb"])

    @classmethod
    def for_job(
        cls,
        *,
        enabled,
        project,
        entity,
        mode,
        name,
        group,
        job_type,
        tags,
        config,
        output_dir,
        max_artifact_mb=50,
    ):
        if not enabled:
            return cls()
        wandb = _import_wandb()
        run = wandb.init(
            project=project,
            entity=entity or None,
            name=name,
            group=group,
            job_type=job_type,
            tags=list(tags),
            mode=mode,
            dir=output_dir,
            config=_plain(config),
        )
        return cls(run, max_artifact_mb)

    def log_epoch(
        self,
        epoch,
        train_loss,
        train_components,
        val_loss,
        val_metrics,
        elapsed_seconds,
    ):
        if not self.enabled:
            return
        payload = {
            "epoch": int(epoch),
            "train/loss": float(train_loss),
            "train/elapsed_seconds": float(elapsed_seconds),
            "val/loss": float(val_loss),
            **{
                f"train/{key}": float(value)
                for key, value in train_components.items()
            },
            **metric_payload("val", val_metrics),
        }
        self.run.log(payload)

    def summary(self, values):
        if not self.enabled:
            return
        for key, value in values.items():
            value = _finite_scalar(value)
            if value is not None:
                self.run.summary[key] = value

    def log_artifact(self, name, artifact_type, files, metadata=None):
        if not self.enabled:
            return
        wandb = _import_wandb()
        artifact = wandb.Artifact(
            name=name,
            type=artifact_type,
            metadata=_plain(metadata or {}),
        )
        added = False
        for path in files:
            if path and os.path.isfile(path):
                if os.path.getsize(path) > self.max_artifact_bytes:
                    print(
                        "Skipping oversized W&B artifact file "
                        f"({os.path.getsize(path)} bytes): {path}"
                    )
                    continue
                artifact.add_file(path)
                added = True
        if added:
            self.run.log_artifact(artifact)

    def finish(self, exit_code=0):
        if self.enabled:
            self.run.finish(exit_code=exit_code)


def job_options(parser):
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="MIR-MIL")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb-tag", action="append", default=[])
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-comparison-id")


def hash_manifest(paths):
    records = []
    for path in sorted(os.path.abspath(item) for item in paths):
        records.append(
            {
                "path": path,
                "size": os.path.getsize(path),
                "sha256": file_sha256(path),
            }
        )
    encoded = json.dumps(records, sort_keys=True).encode("utf-8")
    return {
        "files": records,
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _normalized_tags(tags, dataset, feature, model):
    result = list(tags)
    for tag in (
        f"dataset:{str(dataset).lower()}",
        f"feature:{str(feature).lower()}",
        f"model:{str(model).lower()}",
    ):
        if tag not in result:
            result.append(tag)
    return result


def _import_wandb():
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "W&B tracking is enabled but wandb is not installed. "
            "Install it with `pip install wandb`."
        ) from exc
    return wandb


def start_training_tracker(config, process_pipeline=None):
    global _ACTIVE_TRACKER
    if _ACTIVE_TRACKER is not None and _ACTIVE_TRACKER.enabled:
        raise RuntimeError("A W&B training tracker is already active")
    _ACTIVE_TRACKER = WandbTracker.for_training(
        config,
        process_pipeline=process_pipeline,
    )
    return _ACTIVE_TRACKER


def active_training_tracker():
    return _ACTIVE_TRACKER


def finish_training_tracker(
    args,
    epoch_info_log,
    best_epoch,
    process_pipeline,
):
    global _ACTIVE_TRACKER
    tracker = _ACTIVE_TRACKER
    if tracker is None:
        return
    if tracker.enabled and epoch_info_log["epoch"]:
        best_index = max(
            min(best_epoch - 1, len(epoch_info_log["epoch"]) - 1),
            0,
        )
        metrics = {
            "acc": epoch_info_log["val_acc"][best_index],
            "bacc": epoch_info_log["val_bacc"][best_index],
            "macro_auc": epoch_info_log["val_macro_auc"][best_index],
            "macro_f1": epoch_info_log["val_macro_f1"][best_index],
        }
        tracker.summary(
            {
                **{
                    key.replace("val/", "val/best_"): value
                    for key, value in metric_payload("val", metrics).items()
                },
                "train/best_epoch": int(best_epoch),
                "train/stop_epoch": int(epoch_info_log["epoch"][-1]),
                "provenance/process_pipeline": process_pipeline,
            }
        )
        import glob

        checkpoints = glob.glob(
            os.path.join(args.Logs.now_log_dir, "Best*.pth")
        )
        if checkpoints:
            tracker.summary(
                {
                    "provenance/checkpoint_path": os.path.abspath(
                        checkpoints[0]
                    ),
                    "provenance/checkpoint_sha256": file_sha256(
                        checkpoints[0]
                    ),
                }
            )
        tracker.log_artifact(
            name=f"{tracker.run.id}-split",
            artifact_type="split",
            files=[args.Dataset.dataset_csv_path],
            metadata={
                "sha256": file_sha256(args.Dataset.dataset_csv_path)
            },
        )
    tracker.finish()
    _ACTIVE_TRACKER = None
