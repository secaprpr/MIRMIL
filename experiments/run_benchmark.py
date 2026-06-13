import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone


MODEL_SPECS = {
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
        "General.earlystop.use=true",
        f"General.earlystop.patience={args.patience}",
        f"Dataset.DATASET_NAME={args.dataset_name}",
        f"Dataset.dataset_csv_path={os.path.abspath(args.split)}",
        f"Dataset.balanced_sampler.use={str(args.balanced).lower()}",
        f"Logs.log_root_dir={os.path.abspath(args.log_root)}",
        f"Model.in_dim={args.in_dim}",
        f"Model.max_instances={args.max_instances}",
        "Model.sampling=random",
        f"General.experiment_variant={variant}",
    ]
    options.extend(spec["options"])
    if spec["model"] in {"MIR_MIL", "OT_MIL"}:
        options.append(
            f"Model.scheduler.cosine_config.T_max="
            f"{max(args.epochs - 2, 1)}"
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
    parser.add_argument("--max-instances", type=int, default=4096)
    parser.add_argument("--in-dim", type=int, default=1024)
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
        "max_instances": args.max_instances,
        "balanced": args.balanced,
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
