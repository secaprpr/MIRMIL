import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.model_utils import get_model_from_yaml
from utils.wsi_utils import WSI_Coord_Dataset, WSI_Dataset
from utils.yaml_utils import read_yaml
from utils.wandb_utils import (
    SCHEMA_VERSION,
    WandbTracker,
    job_options,
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
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def find_run_files(run_dir):
    configs = glob.glob(os.path.join(run_dir, "*.yaml"))
    checkpoints = glob.glob(os.path.join(run_dir, "Best*.pth"))
    if len(configs) != 1 or len(checkpoints) != 1:
        raise ValueError(
            "run-dir must contain exactly one YAML and one Best checkpoint"
        )
    return configs[0], checkpoints[0]


def correlation(first, second):
    first = np.asarray(first, dtype=float)
    second = np.asarray(second, dtype=float)
    if first.size < 2 or np.std(first) == 0 or np.std(second) == 0:
        return float("nan")
    return float(np.corrcoef(first, second)[0, 1])


def faithfulness_metrics(response, finite_difference, topk):
    response = np.asarray(response, dtype=float)
    finite_difference = np.asarray(finite_difference, dtype=float)
    if response.shape != finite_difference.shape:
        raise ValueError("response and finite_difference shapes must match")
    ranks_response = pd.Series(response).rank(method="average").to_numpy()
    ranks_finite = pd.Series(finite_difference).rank(
        method="average"
    ).to_numpy()
    count = min(max(int(topk), 1), response.size)
    response_top = set(np.argsort(np.abs(response))[-count:])
    finite_top = set(np.argsort(np.abs(finite_difference))[-count:])
    return {
        "pearson": correlation(response, finite_difference),
        "spearman": correlation(ranks_response, ranks_finite),
        "mse": float(np.mean((response - finite_difference) ** 2)),
        "mae": float(np.mean(np.abs(response - finite_difference))),
        "topk_overlap": len(response_top & finite_top) / count,
    }


def choose_target(output, label, mode, target_class):
    if mode == "predicted":
        return int(output["logits"].argmax(dim=1).item())
    if mode == "label":
        return int(label)
    return int(target_class)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--group", choices=["train", "val", "test"], default="val")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--max-slides", type=int, default=20)
    parser.add_argument("--patches-per-slide", type=int, default=64)
    parser.add_argument("--epsilon", type=float, default=1e-4)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument(
        "--target",
        choices=["predicted", "label", "class"],
        default="predicted",
    )
    parser.add_argument("--target-class", type=int)
    job_options(parser)
    args = parser.parse_args()
    if args.target == "class" and args.target_class is None:
        parser.error("--target-class is required when --target=class")

    config_path, checkpoint_path = find_run_files(args.run_dir)
    config = read_yaml(config_path)
    if config.General.MODEL_NAME != "MIR_MIL":
        raise ValueError("The checkpoint is not a MIR_MIL run")
    config.Model.max_instances = args.budget
    device = torch.device(f"cuda:{args.device}")
    model = get_model_from_yaml(config).to(device)
    model.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)
    )
    model.eval()

    dataset_class = (
        WSI_Coord_Dataset
        if int(getattr(config.Model, "coordinate_dim", 0)) == 2
        else WSI_Dataset
    )
    dataset = dataset_class(
        args.split,
        args.group,
        max_instances=args.budget,
        sampling="uniform",
    )
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    slide_rows = []
    patch_rows = []
    for slide_index in range(min(len(dataset), args.max_slides)):
        bag, label_tensor = dataset[slide_index]
        bag = bag.float().to(device)
        label = int(label_tensor.item())
        with torch.no_grad():
            prediction = model(bag)
        target_class = choose_target(
            prediction, label, args.target, args.target_class
        )
        with torch.enable_grad():
            response_output = model.measure_influence_response(
                bag, target_class=target_class
            )
            count = min(args.patches_per_slide, bag.shape[0])
            indices = np.sort(
                rng.choice(bag.shape[0], size=count, replace=False)
            )
            finite = torch.stack(
                [
                    model.finite_difference_response(
                        bag,
                        bag[index],
                        target_class,
                        epsilon=args.epsilon,
                    )
                    for index in indices
                ]
            )
        response = (
            response_output["response"][indices].detach().cpu().numpy()
        )
        finite = finite.detach().cpu().numpy()
        metrics = faithfulness_metrics(response, finite, args.topk)
        slide_path = dataset.slide_path_list[slide_index]
        slide_rows.append(
            {
                "slide_index": slide_index,
                "slide_path": slide_path,
                "label": label,
                "target_class": target_class,
                "num_instances": int(bag.shape[0]),
                "response_mean": float(
                    response_output["response"].mean().detach().cpu()
                ),
                **metrics,
            }
        )
        for patch_index, response_value, finite_value in zip(
            indices, response, finite
        ):
            patch_rows.append(
                {
                    "slide_index": slide_index,
                    "slide_path": slide_path,
                    "patch_index": int(patch_index),
                    "label": label,
                    "target_class": target_class,
                    "response": float(response_value),
                    "finite_difference": float(finite_value),
                    "absolute_error": float(
                        abs(response_value - finite_value)
                    ),
                }
            )

    slide_frame = pd.DataFrame(slide_rows)
    patch_frame = pd.DataFrame(patch_rows)
    slide_frame.to_csv(
        os.path.join(args.output_dir, "slide_faithfulness.csv"), index=False
    )
    patch_frame.to_csv(
        os.path.join(args.output_dir, "patch_faithfulness.csv"), index=False
    )
    aggregate = {
        "num_slides": len(slide_frame),
        "num_patches": len(patch_frame),
        "pearson_mean": float(slide_frame["pearson"].mean()),
        "spearman_mean": float(slide_frame["spearman"].mean()),
        "mse_mean": float(slide_frame["mse"].mean()),
        "mae_mean": float(slide_frame["mae"].mean()),
        "topk_overlap_mean": float(slide_frame["topk_overlap"].mean()),
        "absolute_response_mean": float(
            patch_frame["response"].abs().mean()
        ),
        "centered_response_mean": float(
            slide_frame["response_mean"].abs().mean()
        ),
    }
    provenance = {
        "git_commit": git_commit(),
        "run_dir": os.path.abspath(args.run_dir),
        "config_sha256": file_sha256(config_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "split": os.path.abspath(args.split),
        "split_sha256": file_sha256(args.split),
        "group": args.group,
        "budget": args.budget,
        "epsilon": args.epsilon,
        "target": args.target,
        "target_class": args.target_class,
        "aggregate": aggregate,
    }
    with open(
        os.path.join(args.output_dir, "provenance.json"),
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(provenance, file, indent=2)
    settings = tracking_settings(config)
    dataset_name = str(config.Dataset.DATASET_NAME)
    model_name = str(config.General.MODEL_NAME)
    variant = settings["variant"]
    name = (
        f"{dataset_name}_{settings['feature']}_{model_name}_{variant}_"
        f"seed{args.seed}_faithfulness"
    )
    group = args.wandb_group or (
        f"{dataset_name}_{settings['feature']}_{model_name}_{variant}_"
        f"{settings['protocol']}_{settings['split_id']}"
    )
    tracker = WandbTracker.for_job(
        enabled=args.wandb,
        project=args.wandb_project,
        entity=args.wandb_entity,
        mode=args.wandb_mode,
        name=name,
        group=group,
        job_type="faithfulness",
        tags=[
            *args.wandb_tag,
            f"dataset:{dataset_name.lower()}",
            f"feature:{settings['feature'].lower()}",
            f"model:{model_name.lower()}",
            "audit:faithfulness",
        ],
        config={
            "schema_version": SCHEMA_VERSION,
            "comparison_id": (
                args.wandb_comparison_id or settings["comparison_id"]
            ),
            "parent": {
                "run_dir": os.path.abspath(args.run_dir),
                "config_sha256": provenance["config_sha256"],
                "checkpoint_sha256": provenance["checkpoint_sha256"],
            },
            "dataset": {
                "name": dataset_name,
                "num_classes": int(config.General.num_classes),
            },
            "split": {
                "id": settings["split_id"],
                "path": provenance["split"],
                "sha256": provenance["split_sha256"],
                "group": args.group,
            },
            "features": {
                "encoder": settings["feature"],
                "manifest_sha256": settings["feature_manifest_sha256"],
                "coordinate_manifest_sha256": settings[
                    "coordinate_manifest_sha256"
                ],
            },
            "model": {"name": model_name, "variant": variant},
            "faithfulness": {
                "budget": args.budget,
                "max_slides": args.max_slides,
                "patches_per_slide": args.patches_per_slide,
                "epsilon": args.epsilon,
                "topk": args.topk,
                "seed": args.seed,
                "target": args.target,
                "target_class": args.target_class,
            },
        },
        output_dir=args.output_dir,
    )
    tracker.summary(
        {
            "faithfulness/pearson": aggregate["pearson_mean"],
            "faithfulness/spearman": aggregate["spearman_mean"],
            "faithfulness/fd_mae": aggregate["mae_mean"],
            "faithfulness/fd_mse": aggregate["mse_mean"],
            "faithfulness/topk_overlap": aggregate[
                "topk_overlap_mean"
            ],
            "faithfulness/centered_mean_abs": aggregate[
                "centered_response_mean"
            ],
            "faithfulness/num_slides": aggregate["num_slides"],
            "faithfulness/num_patches": aggregate["num_patches"],
        }
    )
    tracker.log_artifact(
        name=f"{tracker.run.id}-faithfulness"
        if tracker.enabled
        else "faithfulness",
        artifact_type="faithfulness",
        files=[
            os.path.join(args.output_dir, "slide_faithfulness.csv"),
            os.path.join(args.output_dir, "patch_faithfulness.csv"),
            os.path.join(args.output_dir, "provenance.json"),
        ],
        metadata={
            "checkpoint_sha256": provenance["checkpoint_sha256"],
            "split_sha256": provenance["split_sha256"],
        },
    )
    tracker.finish()
    print(json.dumps(aggregate, indent=2))


if __name__ == "__main__":
    main()
