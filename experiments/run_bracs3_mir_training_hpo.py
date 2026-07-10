"""BRACS3 MIR-MIL training-only hyperparameter screening.

This runner intentionally keeps the MIR-MIL architecture and pre-extracted
features unchanged. It varies only training/sampling/loss/regularization knobs
and records validation results. It does not evaluate the test split.
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


TRAINING_ONLY_VARIANTS = {
    "baseline_noes30": {
        "balanced": True,
        "options": {},
    },
    "lr1e4_wd1e5": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-5,
        },
    },
    "lr1e4_wd5e5": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 5e-5,
        },
    },
    "lr1e4_wd1e4": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
        },
    },
    "lr2e4_wd5e5": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 2e-4,
            "Model.optimizer.adamw_config.weight_decay": 5e-5,
        },
    },
    "lr3e4_wd5e5": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 3e-4,
            "Model.optimizer.adamw_config.weight_decay": 5e-5,
        },
    },
    "clean_noaug": {
        "balanced": True,
        "options": {
            "Model.stability_weight": 0.0,
            "Model.patch_dropout": 0.0,
            "Model.feature_noise_std": 0.0,
        },
    },
    "mild_reg": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "strong_reg": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 5e-4,
            "Model.dropout": 0.25,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.1,
            "Model.feature_noise_std": 0.01,
        },
    },
    "smooth005": {
        "balanced": True,
        "options": {
            "Model.label_smoothing": 0.05,
        },
    },
    "smooth010": {
        "balanced": True,
        "options": {
            "Model.label_smoothing": 0.1,
        },
    },
    "mild_smooth005": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
            "Model.label_smoothing": 0.05,
        },
    },
    "loss_sqrt_inverse": {
        "balanced": True,
        "options": {
            "Model.class_weighting": "sqrt_inverse",
        },
    },
    "loss_effective099": {
        "balanced": True,
        "options": {
            "Model.class_weighting": "effective",
            "Model.class_weight_beta": 0.99,
        },
    },
    "loss_focal1": {
        "balanced": True,
        "options": {
            "Model.focal_gamma": 1.0,
        },
    },
    "unbalanced_mild": {
        "balanced": False,
        "options": {
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
            "Model.label_smoothing": 0.05,
        },
    },
}


def serialize(value):
    if isinstance(value, float):
        text = f"{value:.10f}".rstrip("0").rstrip(".")
        return "0.0" if text == "0" else text
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def build_command(args, feature, seed, variant, spec, output):
    split = args.feature_root / "metadata" / (
        f"BRACS3_{feature}_split_official_train_val.csv"
    )
    cmd = [
        str(args.python),
        "experiments/run_benchmark.py",
        "--split",
        str(split),
        "--dataset-name",
        "BRACS3",
        "--num-classes",
        "3",
        "--log-root",
        str(output),
        "--models",
        "MIR_MIL",
        "--seeds",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--patience",
        "999",
        "--max-instances",
        str(args.max_instances),
        "--in-dim",
        "1024",
        "--feature",
        feature,
        "--protocol",
        args.protocol,
        "--split-id",
        "official-train-val-3class",
        "--comparison-id",
        f"bracs3-{feature}-mir-hpo-{variant}-seed{seed}",
        "--device",
        "0",
        "--num-workers",
        str(args.num_workers),
        "--model-option",
        "General.earlystop.use=false",
        "--model-option",
        "General.earlystop.patience=999",
    ]
    cmd.append("--balanced" if spec["balanced"] else "--no-balanced")
    if args.wandb:
        cmd.extend(["--wandb", "--wandb-project", args.wandb_project])
    for key, value in spec["options"].items():
        cmd.extend(["--model-option", f"{key}={serialize(value)}"])
    return cmd


def best_result(output):
    matches = list(output.glob("BRACS3/MIR_MIL/time_*/Best_Log*.csv"))
    if not matches:
        raise FileNotFoundError(f"no Best_Log under {output}")
    path = max(matches, key=lambda item: item.stat().st_mtime)
    row = pd.read_csv(path).iloc[0]
    checkpoint = None
    ckpts = list(path.parent.glob("Best_EPOCH_*.pth"))
    if ckpts:
        checkpoint = str(max(ckpts, key=lambda item: item.stat().st_mtime).resolve())
    return {
        "best_log": str(path.resolve()),
        "checkpoint": checkpoint,
        "best_epoch": int(row["epoch"]),
        "val_acc": float(row["val_acc"]),
        "val_bacc": float(row["val_bacc"]),
        "val_macro_auc": float(row["val_macro_auc"]),
        "val_macro_f1": float(row["val_macro_f1"]),
    }


def epoch_summary(output):
    logs = list(output.glob("**/Log_seed*_BRACS3_MIR_MIL.csv"))
    if not logs:
        return {}
    log = max(logs, key=lambda item: item.stat().st_mtime)
    frame = pd.read_csv(log)
    return {
        "epochs_ran": int(frame["epoch"].max()),
        "last_val_macro_auc": float(frame.iloc[-1]["val_macro_auc"]),
        "last_val_macro_f1": float(frame.iloc[-1]["val_macro_f1"]),
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "artifacts/bracs3_mir_hpo_training_only",
    )
    parser.add_argument("--features", nargs="+", default=["uni"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025])
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--max-instances", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--protocol", default="bracs3-mir-hpo-training-v1")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="MIR-MIL")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()

    variants = args.variants or list(TRAINING_ONLY_VARIANTS)
    unknown = sorted(set(variants) - set(TRAINING_ONLY_VARIANTS))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_queue = queue.Queue()
    manifest = []
    for feature in args.features:
        for variant in variants:
            spec = TRAINING_ONLY_VARIANTS[variant]
            for seed in args.seeds:
                output = args.output_dir / feature / variant / f"seed{seed}"
                cmd = build_command(args, feature, seed, variant, spec, output)
                task = {
                    "feature": feature,
                    "variant": variant,
                    "seed": seed,
                    "balanced": spec["balanced"],
                    "options": spec["options"],
                    "output": output,
                    "command": cmd,
                }
                task_queue.put(task)
                manifest.append(
                    {
                        **{k: v for k, v in task.items() if k not in {"output"}},
                        "output": str(output),
                    }
                )
    with open(args.output_dir / "hpo_manifest.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now(timezone.utc).isoformat(),
                "protocol": args.protocol,
                "epochs": args.epochs,
                "earlystop": "disabled",
                "features": args.features,
                "seeds": args.seeds,
                "variants": variants,
                "tasks": manifest,
            },
            f,
            indent=2,
        )

    records = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            output = task["output"]
            output.mkdir(parents=True, exist_ok=True)
            completed = output / "completed.json"
            result_path = output / "result.json"
            if completed.exists() and result_path.exists():
                status = "skipped"
                with open(result_path, encoding="utf-8") as f:
                    record = json.load(f)
            else:
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = str(gpu)
                env["PYTHONPATH"] = str(project_root)
                with open(output / "train.stdout.log", "a", encoding="utf-8") as log:
                    log.write(
                        "\nSTART "
                        + datetime.now(timezone.utc).isoformat()
                        + f" GPU={gpu} COMMAND="
                        + json.dumps(task["command"])
                        + "\n"
                    )
                    log.flush()
                    rc = subprocess.run(
                        task["command"],
                        cwd=project_root,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    ).returncode
                record = {
                    "feature": task["feature"],
                    "variant": task["variant"],
                    "seed": task["seed"],
                    "return_code": rc,
                    "gpu": str(gpu),
                    "balanced": task["balanced"],
                    "options": json.dumps(task["options"], sort_keys=True),
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "command": task["command"],
                }
                if rc == 0:
                    record.update(best_result(output))
                    record.update(epoch_summary(output))
                    with open(completed, "w", encoding="utf-8") as f:
                        json.dump(record, f, indent=2)
                    status = "completed"
                else:
                    with open(output / "failed.json", "w", encoding="utf-8") as f:
                        json.dump(record, f, indent=2)
                    status = "failed"
                with open(result_path, "w", encoding="utf-8") as f:
                    json.dump(record, f, indent=2)
            with lock:
                records.append(record)
                frame = pd.DataFrame(records)
                if "val_macro_auc" in frame:
                    frame = frame.sort_values(
                        ["val_macro_auc", "val_macro_f1"],
                        ascending=False,
                        na_position="last",
                    )
                frame.to_csv(args.output_dir / "hpo_results.csv", index=False)
            print(
                f"[{status}] gpu={gpu} feature={task['feature']} "
                f"variant={task['variant']} seed={task['seed']}",
                flush=True,
            )
            task_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu.strip(),))
        for gpu in args.gpus.split(",")
        if gpu.strip()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    frame = pd.DataFrame(records)
    if "val_macro_auc" in frame:
        frame = frame.sort_values(
            ["val_macro_auc", "val_macro_f1"],
            ascending=False,
            na_position="last",
        )
    frame.to_csv(args.output_dir / "hpo_results.csv", index=False)
    print(frame.to_string(index=False))
    if (frame["return_code"] != 0).any():
        raise SystemExit("one or more HPO tasks failed")


if __name__ == "__main__":
    main()
