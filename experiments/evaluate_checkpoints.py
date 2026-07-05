import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.general_utils import set_global_seed
from utils.loop_utils import cal_scores
from utils.model_utils import get_model_from_yaml
from utils.wsi_utils import WSI_Dataset
from utils.yaml_utils import read_yaml
from utils.wandb_utils import (
    SCHEMA_VERSION,
    WandbTracker,
    job_options,
    metric_payload,
    tracking_settings,
)


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


def find_run_files(run_dir, checkpoint_kind="best"):
    config_paths = glob.glob(os.path.join(run_dir, "*.yaml"))
    checkpoint_prefix = "Best" if checkpoint_kind == "best" else "Last"
    checkpoint_paths = glob.glob(
        os.path.join(run_dir, f"{checkpoint_prefix}*.pth")
    )
    if len(config_paths) != 1 or len(checkpoint_paths) != 1:
        raise ValueError(
            f"{run_dir} must contain exactly one YAML and one "
            f"{checkpoint_prefix} checkpoint"
        )
    return config_paths[0], checkpoint_paths[0]


def discover_runs(root, models, checkpoint_kind="best"):
    run_dirs = []
    checkpoint_prefix = "Best" if checkpoint_kind == "best" else "Last"
    for checkpoint in glob.glob(
        os.path.join(root, "**", f"{checkpoint_prefix}*.pth"), recursive=True
    ):
        run_dir = os.path.dirname(checkpoint)
        config_paths = glob.glob(os.path.join(run_dir, "*.yaml"))
        if len(config_paths) != 1:
            continue
        config = read_yaml(config_paths[0])
        if config.General.MODEL_NAME in models:
            run_dirs.append(run_dir)
    return sorted(run_dirs)


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def experiment_variant(args):
    if "experiment_variant" in args.General:
        return str(args.General.experiment_variant)
    return str(args.General.MODEL_NAME)


def evaluate_run(
    run_dir,
    budget,
    device,
    num_workers,
    split_override=None,
    group="test",
    checkpoint_kind="best",
):
    config_path, checkpoint_path = find_run_files(
        run_dir, checkpoint_kind=checkpoint_kind
    )
    args = read_yaml(config_path)
    if split_override:
        args.Dataset.dataset_csv_path = os.path.abspath(split_override)
    args.Model.max_instances = budget
    args.General.device = device
    set_global_seed(args.General.seed)

    dataset = WSI_Dataset(
        args.Dataset.dataset_csv_path,
        group,
        max_instances=budget,
        sampling="uniform",
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
    )
    torch_device = torch.device(f"cuda:{device}")
    model = get_model_from_yaml(args).to(torch_device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=torch_device, weights_only=True)
    )
    model.eval()

    labels = []
    selected_probabilities = []
    full_probabilities = []
    complement_probabilities = []
    random_probabilities = []
    selected_ratios = []
    with torch.no_grad():
        for bag, label in loader:
            bag = bag.float().to(torch_device)
            if args.General.MODEL_NAME == "OT_MIL":
                output = model(bag, return_controls=True)
            else:
                output = model(bag)
            labels.append(int(label.item()))
            selected_probabilities.append(
                torch.softmax(output["logits"].squeeze(0), dim=0).cpu().numpy()
            )
            if args.General.MODEL_NAME == "OT_MIL":
                full_probabilities.append(
                    torch.softmax(output["full_logits"].squeeze(0), dim=0)
                    .cpu()
                    .numpy()
                )
                complement_probabilities.append(
                    torch.softmax(output["complement_logits"].squeeze(0), dim=0)
                    .cpu()
                    .numpy()
                )
                random_probabilities.append(
                    torch.softmax(output["random_logits"].squeeze(0), dim=0)
                    .cpu()
                    .numpy()
                )
                selected_ratios.append(float(output["selected_ratio"].item()))

    metrics = cal_scores(
        selected_probabilities, labels, args.General.num_classes
    )
    result = {
        "run_dir": os.path.abspath(run_dir),
        "model": args.General.MODEL_NAME,
        "variant": experiment_variant(args),
        "seed": int(args.General.seed),
        "budget": int(budget),
        "num_slides": len(dataset),
        "macro_auc": metrics["macro_auc"],
        "acc": metrics["acc"],
        "bacc": metrics["bacc"],
        "macro_f1": metrics["macro_f1"],
    }
    if args.General.MODEL_NAME == "OT_MIL":
        full_metrics = cal_scores(
            full_probabilities, labels, args.General.num_classes
        )
        complement_metrics = cal_scores(
            complement_probabilities, labels, args.General.num_classes
        )
        random_metrics = cal_scores(
            random_probabilities, labels, args.General.num_classes
        )
        result.update(
            {
                "selected_ratio": sum(selected_ratios) / len(selected_ratios),
                "full_macro_auc": full_metrics["macro_auc"],
                "complement_macro_auc": complement_metrics["macro_auc"],
                "random_macro_auc": random_metrics["macro_auc"],
            }
        )

    prediction_rows = []
    for slide_path, label, probabilities in zip(
        dataset.slide_path_list,
        labels,
        selected_probabilities,
    ):
        row = {
            "slide_path": slide_path,
            "label": label,
            "model": args.General.MODEL_NAME,
            "variant": experiment_variant(args),
            "seed": int(args.General.seed),
            "budget": int(budget),
        }
        row.update(
            {
                f"prob_{class_index}": float(probability)
                for class_index, probability in enumerate(probabilities)
            }
        )
        prediction_rows.append(row)
    return json_safe(result), prediction_rows, config_path, checkpoint_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--models", nargs="+", default=["OT_MIL", "MO_MIL"]
    )
    parser.add_argument("--budgets", nargs="+", type=int, default=[128, 256, 512])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--group",
        choices=["train", "val", "test"],
        default="test",
        help="Dataset column group to evaluate (default: test).",
    )
    parser.add_argument(
        "--checkpoint-kind",
        choices=["best", "last"],
        default="best",
        help="Evaluate the selected Best or final Last checkpoint.",
    )
    parser.add_argument(
        "--split-override",
        help=(
            "Evaluate frozen checkpoints on this split CSV instead of the "
            "training-time config split"
        ),
    )
    job_options(parser)
    args = parser.parse_args()

    if args.split_override and not os.path.isfile(args.split_override):
        raise FileNotFoundError(args.split_override)
    run_dirs = discover_runs(
        args.run_root,
        set(args.models),
        checkpoint_kind=args.checkpoint_kind,
    )
    if not run_dirs:
        raise FileNotFoundError(f"No completed runs found under {args.run_root}")
    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    eval_run_records = []
    provenance = {
        "git_commit": git_commit(),
        "run_root": os.path.abspath(args.run_root),
        "split_override": (
            os.path.abspath(args.split_override)
            if args.split_override
            else None
        ),
        "split_override_sha256": (
            file_sha256(args.split_override)
            if args.split_override
            else None
        ),
        "group": args.group,
        "checkpoint_kind": args.checkpoint_kind,
        "budgets": args.budgets,
        "runs": [],
    }
    for run_dir in run_dirs:
        for budget in args.budgets:
            result, predictions, config_path, checkpoint_path = evaluate_run(
                run_dir,
                budget,
                args.device,
                args.num_workers,
                split_override=args.split_override,
                group=args.group,
                checkpoint_kind=args.checkpoint_kind,
            )
            results.append(result)
            prediction_name = (
                f"{result['variant']}_seed{result['seed']}_budget{budget}.csv"
            )
            prediction_path = os.path.join(args.output_dir, prediction_name)
            pd.DataFrame(predictions).to_csv(prediction_path, index=False)
            run_config = read_yaml(config_path)
            settings = tracking_settings(run_config)
            split_path = (
                os.path.abspath(args.split_override)
                if args.split_override
                else os.path.abspath(run_config.Dataset.dataset_csv_path)
            )
            split_hash = file_sha256(split_path)
            dataset_name = str(run_config.Dataset.DATASET_NAME)
            model_name = str(run_config.General.MODEL_NAME)
            protocol = settings["protocol"]
            split_id = settings["split_id"]
            name = (
                f"{dataset_name}_{settings['feature']}_{model_name}_"
                f"{result['variant']}_seed{result['seed']}_eval"
            )
            group = args.wandb_group or (
                f"{dataset_name}_{settings['feature']}_{model_name}_"
                f"{result['variant']}_{protocol}_{split_id}"
            )
            tracker = WandbTracker.for_job(
                enabled=args.wandb,
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                name=name,
                group=group,
                job_type="eval",
                tags=[
                    *args.wandb_tag,
                    f"dataset:{dataset_name.lower()}",
                    f"feature:{settings['feature'].lower()}",
                    f"model:{model_name.lower()}",
                    "test:sealed" if args.split_override else "test:configured",
                ],
                config={
                    "schema_version": SCHEMA_VERSION,
                    "comparison_id": (
                        args.wandb_comparison_id
                        or settings["comparison_id"]
                    ),
                    "parent": {
                        "run_dir": os.path.abspath(run_dir),
                        "config_sha256": file_sha256(config_path),
                        "checkpoint_sha256": file_sha256(checkpoint_path),
                    },
                    "dataset": {
                        "name": dataset_name,
                        "num_classes": int(
                            run_config.General.num_classes
                        ),
                    },
                    "split": {
                        "id": split_id,
                        "path": split_path,
                        "sha256": split_hash,
                        "test_sealed": bool(args.split_override),
                    },
                    "features": {
                        "encoder": settings["feature"],
                        "manifest_sha256": settings[
                            "feature_manifest_sha256"
                        ],
                        "coordinate_manifest_sha256": settings[
                            "coordinate_manifest_sha256"
                        ],
                    },
                    "model": {
                        "name": model_name,
                        "variant": result["variant"],
                    },
                    "evaluation": {
                        "protocol": protocol,
                        "sampling": "uniform",
                        "patch_budget": int(budget),
                        "num_slides": int(result["num_slides"]),
                    },
                },
                output_dir=args.output_dir,
            )
            tracker.summary(
                {
                    **metric_payload(
                        "test",
                        {
                            "acc": result["acc"],
                            "bacc": result["bacc"],
                            "macro_auc": result["macro_auc"],
                            "macro_f1": result["macro_f1"],
                        },
                    ),
                    "test/num_slides": int(result["num_slides"]),
                }
            )
            tracker.log_artifact(
                name=f"{tracker.run.id}-predictions"
                if tracker.enabled
                else "predictions",
                artifact_type="predictions",
                files=[prediction_path],
                metadata={
                    "split_sha256": split_hash,
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                },
            )
            if tracker.enabled:
                eval_run_records.append(
                    {
                        "run_id": tracker.run.id,
                        "variant": result["variant"],
                        "budget": int(budget),
                    }
                )
            tracker.finish()
            provenance["runs"].append(
                {
                    **result,
                    "config_sha256": file_sha256(config_path),
                    "checkpoint_sha256": file_sha256(checkpoint_path),
                }
            )
            print(json.dumps(result, indent=2))

    result_frame = pd.DataFrame(results).sort_values(
        ["budget", "variant", "seed"]
    )
    result_frame.to_csv(
        os.path.join(args.output_dir, "budget_results.csv"), index=False
    )
    aggregate = (
        result_frame.groupby(["budget", "variant"])
        .agg(
            macro_auc_mean=("macro_auc", "mean"),
            macro_auc_std=("macro_auc", "std"),
            acc_mean=("acc", "mean"),
            bacc_mean=("bacc", "mean"),
            macro_f1_mean=("macro_f1", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(
        os.path.join(args.output_dir, "budget_aggregate.csv"), index=False
    )
    if args.wandb:
        first_config_path, _ = find_run_files(run_dirs[0])
        first_config = read_yaml(first_config_path)
        settings = tracking_settings(first_config)
        dataset_name = str(first_config.Dataset.DATASET_NAME)
        for row in aggregate.to_dict(orient="records"):
            variant = str(row["variant"])
            budget = int(row["budget"])
            child_ids = [
                item["run_id"]
                for item in eval_run_records
                if item["variant"] == variant
                and item["budget"] == budget
            ]
            tracker = WandbTracker.for_job(
                enabled=True,
                project=args.wandb_project,
                entity=args.wandb_entity,
                mode=args.wandb_mode,
                name=(
                    f"{dataset_name}_{settings['feature']}_{variant}_"
                    f"budget{budget}_aggregate"
                ),
                group=args.wandb_group
                or (
                    f"{dataset_name}_{settings['feature']}_{variant}_"
                    f"{settings['protocol']}_{settings['split_id']}"
                ),
                job_type="aggregate",
                tags=[
                    *args.wandb_tag,
                    f"dataset:{dataset_name.lower()}",
                    f"feature:{settings['feature'].lower()}",
                    "job:aggregate",
                ],
                config={
                    "schema_version": SCHEMA_VERSION,
                    "comparison_id": (
                        args.wandb_comparison_id
                        or settings["comparison_id"]
                    ),
                    "aggregate": {
                        "variant": variant,
                        "budget": budget,
                        "child_eval_run_ids": child_ids,
                        "num_seeds": len(child_ids),
                    },
                },
                output_dir=args.output_dir,
            )
            tracker.summary(
                {
                    "test/macro_auc_ovr": row["macro_auc_mean"],
                    "test/macro_auc_ovr_std": row["macro_auc_std"],
                    "test/accuracy": row["acc_mean"],
                    "test/balanced_accuracy": row["bacc_mean"],
                    "test/macro_f1": row["macro_f1_mean"],
                    "aggregate/num_seeds": len(child_ids),
                }
            )
            tracker.finish()
    with open(
        os.path.join(args.output_dir, "provenance.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(provenance, file, indent=2)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
