"""Parallel held-out evaluation and three-seed baseline aggregation."""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import wandb


def latest_file(root, pattern):
    matches = list(root.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{pattern} under {root}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def build_tasks(args):
    training_manifest = json.load(open(args.training_manifest))
    tasks = []
    for item in training_manifest["tasks"]:
        feature, model, seed = (
            item["feature"], item["model"], int(item["seed"])
        )
        run_dir = Path(item["run_dir"])
        output = args.output_dir / feature / model / f"seed{seed}"
        split_suffix = args.full_h5_suffix if model == "MAMBA2D_MIL" else args.full_suffix
        split = args.feature_root / "metadata" / (
            f"{args.split_prefix}_{feature}_{split_suffix}.csv"
        )
        if model == "C2AUG":
            checkpoint = latest_file(
                run_dir / "checkpoints", "**/*.ckpt"
            )
            if checkpoint.name == "last.ckpt":
                best = [
                    path for path in (run_dir / "checkpoints").glob("**/*.ckpt")
                    if path.name != "last.ckpt"
                ]
                checkpoint = max(best, key=lambda path: path.stat().st_mtime)
            command = [
                str(args.python),
                str(args.project_root / "third_party_adapters/c2aug_evaluate.py"),
                "--checkpoint", str(checkpoint),
                "--split", str(split),
                "--output-dir", str(output),
                "--feature", feature,
                "--seed", str(seed),
                "--max-instances", str(args.max_instances),
                "--num-workers", str(args.num_workers),
                "--wandb-project", args.wandb_project,
                "--dataset-name", args.dataset_name,
                "--num-classes", str(args.num_classes),
            ]
            cwd = args.project_root / "baselines/c2aug"
        else:
            checkpoint = latest_file(
                run_dir,
                f"{args.dataset_name}/*/time_*/Best_EPOCH_*.pth",
            )
            config = checkpoint.parent / f"{model}.yaml"
            if not config.exists():
                config = latest_file(checkpoint.parent, "*.yaml")
            command = [
                str(args.python),
                "experiments/evaluate_panda_task.py",
                "--config", str(config),
                "--checkpoint", str(checkpoint),
                "--split", str(split),
                "--output-dir", str(output),
                "--feature", feature,
                "--model", model,
                "--seed", str(seed),
                "--max-instances", str(args.max_instances),
                "--num-workers", str(args.num_workers),
                "--wandb-project", args.wandb_project,
                "--dataset-name", args.dataset_name,
            ]
            cwd = args.project_root
        tasks.append(
            {
                "feature": feature, "model": model, "seed": seed,
                "output": output, "command": command, "cwd": cwd,
            }
        )
    return tasks


def run_task(task, gpu):
    task["output"].mkdir(parents=True, exist_ok=True)
    completed = task["output"] / "completed.json"
    if completed.exists() and (task["output"] / "result.json").exists():
        return "skipped"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    if task["model"] == "C2AUG":
        environment["PYTHONPATH"] = str(task["cwd"])
    started = datetime.now(timezone.utc).isoformat()
    with open(task["output"] / "eval.stdout.log", "a", encoding="utf-8") as log:
        result = subprocess.run(
            task["command"], cwd=task["cwd"], env=environment,
            stdout=log, stderr=subprocess.STDOUT,
        )
    record = {
        "feature": task["feature"], "model": task["model"],
        "seed": task["seed"], "gpu": str(gpu),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "return_code": result.returncode,
        "command": task["command"],
    }
    marker = completed if result.returncode == 0 else task["output"] / "failed.json"
    with open(marker, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return "completed" if result.returncode == 0 else "failed"


def aggregate(args, tasks):
    rows = []
    for task in tasks:
        path = task["output"] / "result.json"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(json.load(open(path)))
    frame = pd.DataFrame(rows).sort_values(["feature", "model", "seed"])
    frame.to_csv(args.output_dir / "seed_results.csv", index=False)
    aggregate = (
        frame.groupby(["feature", "model"])
        .agg(
            seeds=("seed", "count"),
            acc_mean=("acc", "mean"), acc_std=("acc", "std"),
            bacc_mean=("bacc", "mean"), bacc_std=("bacc", "std"),
            macro_auc_mean=("macro_auc", "mean"),
            macro_auc_std=("macro_auc", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
        )
        .reset_index()
    )
    aggregate.to_csv(args.output_dir / "aggregate_results.csv", index=False)
    for row in aggregate.to_dict("records"):
        run = wandb.init(
            project=args.wandb_project,
            name=(
                f"{args.dataset_name}_{row['feature']}_"
                f"{row['model']}_aggregate"
            ),
            group=(
                f"{args.dataset_name}_{row['feature']}_{row['model']}_"
                f"protocol-v1_{args.split_id}"
            ),
            job_type="aggregate",
            tags=[
                f"dataset:{args.dataset_name.lower()}",
                f"feature:{row['feature']}",
                f"model:{row['model']}", "job:aggregate",
            ],
            config={
                "dataset": args.dataset_name, "feature": row["feature"],
                "model": row["model"], "num_seeds": int(row["seeds"]),
                "test": "held-out", "split_id": args.split_id,
            },
            reinit="finish_previous",
        )
        run.summary.update(
            {
                "test/accuracy_mean": row["acc_mean"],
                "test/accuracy_std": row["acc_std"],
                "test/balanced_accuracy_mean": row["bacc_mean"],
                "test/balanced_accuracy_std": row["bacc_std"],
                "test/macro_auc_ovr_mean": row["macro_auc_mean"],
                "test/macro_auc_ovr_std": row["macro_auc_std"],
                "test/macro_f1_mean": row["macro_f1_mean"],
                "test/macro_f1_std": row["macro_f1_std"],
                "aggregate/num_seeds": int(row["seeds"]),
            }
        )
        run.finish()
    print(aggregate.to_string(index=False))


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-manifest", type=Path,
        default=project_root / "artifacts/panda_baselines/matrix_manifest.json",
    )
    parser.add_argument(
        "--feature-root", type=Path,
        default=Path("/data15/data15_5/fanhao/datasets/PANDA/MIRMIL_FEATURES"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "artifacts/panda_evaluation",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--max-instances", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-project", default="MIR-MIL")
    parser.add_argument("--dataset-name", default="PANDA")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--split-prefix", default="PANDA")
    parser.add_argument("--full-suffix", default="split_v1_full_qc")
    parser.add_argument("--full-h5-suffix", default="split_v1_full_qc_h5")
    parser.add_argument("--split-id", default="split-v1-qc")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    args.project_root = project_root
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args)
    with open(args.output_dir / "evaluation_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(
            [
                {
                    **{k: v for k, v in task.items() if k not in {"output", "cwd"}},
                    "output": str(task["output"]), "cwd": str(task["cwd"]),
                }
                for task in tasks
            ],
            handle, indent=2,
        )
    task_queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    failures, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            status = run_task(task, gpu)
            print(
                f"[{status}] gpu={gpu} {task['feature']} "
                f"{task['model']} seed={task['seed']}",
                flush=True,
            )
            if status == "failed":
                with lock:
                    failures.append(task)
            task_queue.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu.strip(),))
        for gpu in args.gpus.split(",") if gpu.strip()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"{len(failures)} evaluation tasks failed")
    aggregate(args, tasks)


if __name__ == "__main__":
    main()
