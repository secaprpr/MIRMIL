import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


MODEL_SPECS = {
    "MIR_MIL_MT_V1": {
        "config": "configs/releases/MIR_MIL_MT_V1.yaml",
        "model": "MIR_MIL",
        "options": [],
    },
    "MIR_MIL": {
        "config": "configs/MIR_MIL.yaml",
        "model": "MIR_MIL",
        "options": [],
    },
    "AB_MIL": {
        "config": "configs/AB_MIL.yaml",
        "model": "AB_MIL",
        "options": [],
    },
    "CLAM_SB_MIL": {
        "config": "configs/CLAM_SB_MIL.yaml",
        "model": "CLAM_SB_MIL",
        "options": [],
    },
    "CLAM_MB_MIL": {
        "config": "configs/CLAM_MB_MIL.yaml",
        "model": "CLAM_MB_MIL",
        "options": [],
    },
    "DS_MIL": {
        "config": "configs/DS_MIL.yaml",
        "model": "DS_MIL",
        "options": [],
    },
    "DTFD_MIL": {
        "config": "configs/DTFD_MIL.yaml",
        "model": "DTFD_MIL",
        "options": [],
    },
    "TRANS_MIL": {
        "config": "configs/TRANS_MIL.yaml",
        "model": "TRANS_MIL",
        "options": [],
    },
    "RRT_MIL": {
        "config": "configs/RRT_MIL.yaml",
        "model": "RRT_MIL",
        "options": [],
    },
    "WIKG_MIL": {
        "config": "configs/WIKG_MIL.yaml",
        "model": "WIKG_MIL",
        "options": [],
    },
    "AC_MIL": {
        "config": "configs/AC_MIL.yaml",
        "model": "AC_MIL",
        "options": [],
    },
    "MAMBA2D_MIL": {
        "config": "configs/MAMBA2D_MIL.yaml",
        "model": "MAMBA2D_MIL",
        "options": [],
    },
    "MO_MIL": {
        "config": "configs/MO_MIL.yaml",
        "model": "MO_MIL",
        "options": [],
    },
    "OT_MIL_ORIGINAL": {
        "config": "configs/OT_MIL_MULTICLASS.yaml",
        "model": "OT_MIL",
        "options": ["Model.class_mass_classification_weight=0.0"],
    },
    "OT_MIL_CLASS_MASS": {
        "config": "configs/OT_MIL_MULTICLASS.yaml",
        "model": "OT_MIL",
        "options": ["Model.class_mass_classification_weight=0.1"],
    },
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def build_command(args, variant, seed):
    spec = MODEL_SPECS[variant]
    options = [
        f"General.seed={seed}",
        f"General.num_classes={args.num_classes}",
        f"General.num_epochs={args.epochs}",
        f"General.device={args.device}",
        f"General.num_workers={args.num_workers}",
        f"General.best_model_metric={args.best_model_metric}",
        "General.earlystop.use=true",
        f"General.earlystop.patience={args.patience}",
        f"General.earlystop.metric={args.earlystop_metric}",
        f"General.earlystop.min_delta={args.earlystop_min_delta}",
        f"Dataset.DATASET_NAME={args.dataset_name}",
        f"Dataset.dataset_csv_path={os.path.abspath(args.split)}",
        f"Dataset.balanced_sampler.use={str(args.balanced).lower()}",
        f"Logs.log_root_dir={os.path.abspath(args.log_root)}",
        f"Model.in_dim={args.in_dim}",
        f"Model.max_instances={args.max_instances}",
        "Model.sampling=random",
        f"General.experiment_variant={variant}",
        f"Tracking.wandb.enabled={str(getattr(args, 'wandb', False)).lower()}",
        f"Tracking.wandb.project={getattr(args, 'wandb_project', 'MIR-MIL')}",
        f"Tracking.wandb.mode={getattr(args, 'wandb_mode', 'online')}",
        f"Tracking.wandb.feature={getattr(args, 'feature', 'unknown')}",
        f"Tracking.wandb.variant={variant}",
        f"Tracking.wandb.protocol={getattr(args, 'protocol', 'default')}",
        f"Tracking.wandb.split_id={getattr(args, 'split_id', 'default')}",
        "Tracking.wandb.source_manifest_sha256="
        f"{getattr(args, 'source_manifest_sha256', None)}",
        "Tracking.wandb.feature_manifest_sha256="
        f"{getattr(args, 'feature_manifest_sha256', None)}",
        "Tracking.wandb.coordinate_manifest_sha256="
        f"{getattr(args, 'coordinate_manifest_sha256', None)}",
        "Tracking.wandb.encoder_checkpoint_sha256="
        f"{getattr(args, 'encoder_checkpoint_sha256', None)}",
        "Tracking.wandb.upload_checkpoints=false",
        f"Tracking.wandb.max_artifact_mb={getattr(args, 'max_artifact_mb', 50)}",
    ]
    if getattr(args, "wandb_entity", None):
        options.append(f"Tracking.wandb.entity={args.wandb_entity}")
    if getattr(args, "comparison_id", None):
        options.append(
            f"Tracking.wandb.comparison_id={args.comparison_id}"
        )
    options.extend(spec["options"])
    options.extend(args.model_option)
    if spec["model"] in {"MIR_MIL", "OT_MIL"}:
        scheduler_t_max = (
            args.scheduler_t_max
            if args.scheduler_t_max is not None
            else max(args.epochs - 2, 1)
        )
        options.append(
            f"Model.scheduler.cosine_config.T_max="
            f"{scheduler_t_max}"
        )
        options.append(
            "Model.scheduler.cosine_config.clamp_after_t_max="
            f"{str(args.clamp_cosine).lower()}"
        )
    return [
        args.python,
        "train_mil.py",
        "--yaml_path",
        spec["config"],
        "--options",
        *options,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--log-root", required=True)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_SPECS,
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--best-model-metric", default="macro_auc")
    parser.add_argument("--earlystop-metric", default="macro_auc")
    parser.add_argument("--earlystop-min-delta", type=float, default=0.0)
    parser.add_argument("--scheduler-t-max", type=int)
    parser.add_argument(
        "--clamp-cosine",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--model-option",
        action="append",
        default=[],
        help="Repeat KEY=VALUE to pass an additional model option",
    )
    parser.add_argument("--max-instances", type=int, default=4096)
    parser.add_argument("--in-dim", type=int, default=1024)
    parser.add_argument("--feature", default="unknown")
    parser.add_argument("--protocol", default="default")
    parser.add_argument("--split-id", default="unspecified")
    parser.add_argument("--comparison-id")
    parser.add_argument("--source-manifest-sha256")
    parser.add_argument("--feature-manifest-sha256")
    parser.add_argument("--coordinate-manifest-sha256")
    parser.add_argument("--encoder-checkpoint-sha256")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="MIR-MIL")
    parser.add_argument("--wandb-entity")
    parser.add_argument(
        "--wandb-mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--max-artifact-mb", type=float, default=50)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--balanced", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.split):
        raise FileNotFoundError(args.split)
    os.makedirs(args.log_root, exist_ok=True)
    commands = [
        build_command(args, model, seed)
        for seed in args.seeds
        for model in args.models
    ]
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "dataset_name": args.dataset_name,
        "split": os.path.abspath(args.split),
        "split_sha256": file_sha256(args.split),
        "models": args.models,
        "seeds": args.seeds,
        "epochs": args.epochs,
        "patience": args.patience,
        "best_model_metric": args.best_model_metric,
        "earlystop_metric": args.earlystop_metric,
        "earlystop_min_delta": args.earlystop_min_delta,
        "scheduler_t_max": args.scheduler_t_max,
        "clamp_cosine": args.clamp_cosine,
        "model_options": args.model_option,
        "max_instances": args.max_instances,
        "balanced": args.balanced,
        "feature": args.feature,
        "protocol": args.protocol,
        "split_id": args.split_id,
        "comparison_id": args.comparison_id,
        "source_manifest_sha256": args.source_manifest_sha256,
        "feature_manifest_sha256": args.feature_manifest_sha256,
        "coordinate_manifest_sha256": args.coordinate_manifest_sha256,
        "encoder_checkpoint_sha256": args.encoder_checkpoint_sha256,
        "wandb": args.wandb,
        "commands": commands,
    }
    manifest_path = os.path.join(
        args.log_root, f"{args.dataset_name}_benchmark_manifest.json"
    )
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))

    if args.dry_run:
        return
    for command in commands:
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
