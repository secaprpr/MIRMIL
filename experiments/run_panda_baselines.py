"""Run a formal R50/UNI baseline matrix across local GPUs."""

import argparse
import hashlib
import json
import os
import queue
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path


GENERIC_MODELS = [
    "AB_MIL",
    "CLAM_SB_MIL",
    "CLAM_MB_MIL",
    "DS_MIL",
    "DTFD_MIL",
    "TRANS_MIL",
    "RRT_MIL",
    "WIKG_MIL",
    "AC_MIL",
    "MAMBA2D_MIL",
    "MIR_MIL",
]
ALL_MODELS = GENERIC_MODELS + ["C2AUG"]


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generic_command(args, feature, model, seed, run_dir, hashes):
    train_suffix = getattr(args, "train_suffix", "split_v1_train_val_qc")
    train_h5_suffix = getattr(
        args, "train_h5_suffix", "split_v1_train_val_qc_h5"
    )
    split_prefix = getattr(args, "split_prefix", "PANDA")
    dataset_name = getattr(args, "dataset_name", "PANDA")
    num_classes = getattr(args, "num_classes", 6)
    split_id = getattr(args, "split_id", "split-v1-qc")
    split_suffix = (
        train_h5_suffix if model == "MAMBA2D_MIL" else train_suffix
    )
    max_instances = (
        getattr(args, "spatial_max_instances", args.max_instances)
        if model == "MAMBA2D_MIL"
        else args.max_instances
    )
    split = args.metadata_dir / (
        f"{split_prefix}_{feature}_{split_suffix}.csv"
    )
    command = [
        str(args.python),
        "experiments/run_benchmark.py",
        "--split", str(split),
        "--dataset-name", dataset_name,
        "--num-classes", str(num_classes),
        "--log-root", str(run_dir),
        "--models", model,
        "--seeds", str(seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--max-instances", str(max_instances),
        "--in-dim", "1024",
        "--feature", feature,
        "--protocol", args.protocol,
        "--split-id", split_id,
        "--comparison-id",
        f"{dataset_name.lower()}-{feature}-{model}-seed{seed}",
        "--wandb",
        "--wandb-project", args.wandb_project,
        "--wandb-mode", "online",
        "--device", "0",
        "--num-workers", str(args.num_workers),
        "--source-manifest-sha256", hashes["source"],
        "--feature-manifest-sha256", hashes[f"{feature}_feature"],
        "--coordinate-manifest-sha256", hashes["coordinates"],
        "--encoder-checkpoint-sha256", hashes[f"{feature}_checkpoint"],
    ]
    if model == "MAMBA2D_MIL":
        command.extend(
            [
                "--model-option",
                "Model.coord_scale="
                f"{getattr(args, 'spatial_coord_scale', 512.0)}",
                "--model-option",
                "Model.d_model="
                f"{getattr(args, 'spatial_d_model', 512)}",
            ]
        )
    return command


def c2aug_command(args, feature, seed, run_dir, hashes):
    train_suffix = getattr(args, "train_suffix", "split_v1_train_val_qc")
    split_prefix = getattr(args, "split_prefix", "PANDA")
    dataset_name = getattr(args, "dataset_name", "PANDA")
    num_classes = getattr(args, "num_classes", 6)
    split_id = getattr(args, "split_id", "split-v1-qc")
    split = args.metadata_dir / (
        f"{split_prefix}_{feature}_{train_suffix}.csv"
    )
    feature_dir = args.feature_root / feature
    feat_name = "res50" if feature == "r50" else "uni"
    return [
        str(args.python),
        "train.py",
        "--model", "transmil",
        "--fold", "0",
        "--dataset", dataset_name,
        "--gpus", "0",
        "--feat", feat_name,
        "--enc", "attn",
        "--views", "2",
        "--split-csv", str(split),
        "--feature-dir", str(feature_dir),
        "--num-classes", str(num_classes),
        "--seed", str(seed),
        "--max-epochs", str(args.epochs),
        "--max-instances", str(args.max_instances),
        "--num-workers", "0",
        "--monitor", "val_auc",
        "--patience", str(args.patience),
        "--log-root", str(run_dir / "logs"),
        "--checkpoint-root", str(run_dir / "checkpoints"),
        "--wandb",
        "--wandb-project", args.wandb_project,
        "--feature-name", feature,
        "--protocol", args.protocol,
        "--split-id", split_id,
        "--split-sha256", hashes[f"{feature}_split"],
        "--source-manifest-sha256", hashes["source"],
        "--feature-manifest-sha256", hashes[f"{feature}_feature"],
        "--coordinate-manifest-sha256", hashes["coordinates"],
        "--encoder-checkpoint-sha256", hashes[f"{feature}_checkpoint"],
    ]


def build_tasks(args, hashes):
    tasks = []
    for feature in args.features:
        for seed in args.seeds:
            for model in args.models:
                run_dir = (
                    args.output_dir / feature / model / f"seed{seed}"
                )
                if model == "C2AUG":
                    command = c2aug_command(
                        args, feature, seed, run_dir, hashes
                    )
                    cwd = args.c2aug_dir
                else:
                    command = generic_command(
                        args, feature, model, seed, run_dir, hashes
                    )
                    cwd = args.project_root
                tasks.append(
                    {
                        "feature": feature,
                        "model": model,
                        "seed": seed,
                        "run_dir": run_dir,
                        "cwd": cwd,
                        "command": command,
                    }
                )
    return tasks


def run_task(task, gpu):
    run_dir = task["run_dir"]
    run_dir.mkdir(parents=True, exist_ok=True)
    completed = run_dir / "completed.json"
    if completed.exists():
        return "skipped"
    log_path = run_dir / "train.stdout.log"
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    environment.setdefault(
        "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True"
    )
    started = datetime.now(timezone.utc).isoformat()
    with open(log_path, "a", encoding="utf-8") as log:
        log.write(
            f"\nSTART {started} GPU={gpu} "
            f"COMMAND={json.dumps(task['command'])}\n"
        )
        log.flush()
        result = subprocess.run(
            task["command"],
            cwd=task["cwd"],
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    record = {
        "feature": task["feature"],
        "model": task["model"],
        "seed": task["seed"],
        "gpu": str(gpu),
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "return_code": result.returncode,
        "command": task["command"],
    }
    destination = completed if result.returncode == 0 else run_dir / "failed.json"
    with open(destination, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
    return "completed" if result.returncode == 0 else "failed"


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", nargs="+", choices=["r50", "uni"], default=["r50", "uni"])
    parser.add_argument("--models", nargs="+", choices=ALL_MODELS, default=ALL_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2024, 2025, 2026])
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--max-instances", type=int, default=4096)
    parser.add_argument("--spatial-max-instances", type=int, default=4096)
    parser.add_argument("--spatial-coord-scale", type=float, default=512.0)
    parser.add_argument("--spatial-d-model", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--protocol", default="protocol-v1")
    parser.add_argument("--dataset-name", default="PANDA")
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--split-prefix", default="PANDA")
    parser.add_argument("--train-suffix", default="split_v1_train_val_qc")
    parser.add_argument("--train-h5-suffix", default="split_v1_train_val_qc_h5")
    parser.add_argument("--split-id", default="split-v1-qc")
    parser.add_argument("--source-manifest", default="panda_source_manifest_qc.csv")
    parser.add_argument("--coordinate-manifest", default="panda_patch_manifest_qc.csv")
    parser.add_argument("--wandb-project", default="MIR-MIL")
    parser.add_argument("--manifest-name", default="matrix_manifest.json")
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--feature-root",
        type=Path,
        default=Path("/data15/data15_5/fanhao/datasets/PANDA/MIRMIL_FEATURES"),
    )
    parser.add_argument("--output-dir", type=Path, default=project_root / "artifacts/panda_baselines")
    args = parser.parse_args()
    args.project_root = project_root
    args.metadata_dir = args.feature_root / "metadata"
    args.c2aug_dir = project_root / "baselines/c2aug"
    args.output_dir = args.output_dir.resolve()

    hashes = {
        "source": file_sha256(args.metadata_dir / args.source_manifest),
        "coordinates": file_sha256(args.metadata_dir / args.coordinate_manifest),
        "r50_feature": file_sha256(args.feature_root / "r50/logs/feature_audit.csv"),
        "uni_feature": file_sha256(args.feature_root / "uni/logs/feature_audit.csv"),
        "r50_split": file_sha256(
            args.metadata_dir
            / f"{args.split_prefix}_r50_{args.train_suffix}.csv"
        ),
        "uni_split": file_sha256(
            args.metadata_dir
            / f"{args.split_prefix}_uni_{args.train_suffix}.csv"
        ),
        "r50_checkpoint": file_sha256(project_root / "artifacts/pretrained/resnet50_imagenet/resnet50-19c8e357.pth"),
        "uni_checkpoint": file_sha256(project_root / "artifacts/pretrained/uni/pytorch_model.bin"),
    }
    tasks = build_tasks(args, hashes)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "features": args.features,
        "models": args.models,
        "seeds": args.seeds,
        "gpus": args.gpus,
        "epochs": args.epochs,
        "patience": args.patience,
        "max_instances": args.max_instances,
        "spatial_max_instances": args.spatial_max_instances,
        "spatial_coord_scale": args.spatial_coord_scale,
        "spatial_d_model": args.spatial_d_model,
        "protocol": args.protocol,
        "dataset_name": args.dataset_name,
        "num_classes": args.num_classes,
        "split_id": args.split_id,
        "hashes": hashes,
        "tasks": [
            {
                **{k: v for k, v in task.items() if k not in {"run_dir", "cwd"}},
                "run_dir": str(task["run_dir"]),
                "cwd": str(task["cwd"]),
            }
            for task in tasks
        ],
    }
    with open(args.output_dir / args.manifest_name, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    task_queue = queue.Queue()
    for task in tasks:
        task_queue.put(task)
    failures = []
    lock = threading.Lock()

    def worker(gpu):
        while True:
            try:
                task = task_queue.get_nowait()
            except queue.Empty:
                return
            status = run_task(task, gpu)
            print(
                f"[{status}] gpu={gpu} feature={task['feature']} "
                f"model={task['model']} seed={task['seed']}",
                flush=True,
            )
            if status == "failed":
                with lock:
                    failures.append(task)
            task_queue.task_done()

    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in gpus]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if failures:
        raise SystemExit(f"{len(failures)} tasks failed")


if __name__ == "__main__":
    main()
