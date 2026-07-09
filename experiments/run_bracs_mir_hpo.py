"""Run validation-only BRACS MIR-MIL hyperparameter screening on local GPUs."""

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


VARIANTS = {
    "default_refit": {
        "balanced": True,
        "options": {},
    },
    "lr1e4_mild": {
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
    "lr5e5_clean": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 5e-5,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.0,
            "Model.patch_dropout": 0.0,
            "Model.feature_noise_std": 0.0,
            "Model.label_smoothing": 0.05,
        },
    },
    "lr1e4_smooth": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 5e-4,
            "Model.dropout": 0.2,
            "Model.stability_weight": 0.0,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
            "Model.label_smoothing": 0.1,
        },
    },
    "lr2e4_smooth": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 2e-4,
            "Model.optimizer.adamw_config.weight_decay": 5e-4,
            "Model.dropout": 0.2,
            "Model.stability_weight": 0.0,
            "Model.patch_dropout": 0.0,
            "Model.feature_noise_std": 0.0,
            "Model.label_smoothing": 0.1,
        },
    },
    "lr1e4_ema": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
            "Model.label_smoothing": 0.05,
            "Model.ema_decay": 0.99,
        },
    },
    "lr2e4_ema": {
        "balanced": True,
        "options": {
            "Model.optimizer.adamw_config.lr": 2e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.0,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
            "Model.label_smoothing": 0.05,
            "Model.ema_decay": 0.99,
        },
    },
    "lr1e4_unbalanced": {
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
    "lr2e4_unbalanced": {
        "balanced": False,
        "options": {
            "Model.optimizer.adamw_config.lr": 2e-4,
            "Model.optimizer.adamw_config.weight_decay": 5e-4,
            "Model.dropout": 0.2,
            "Model.stability_weight": 0.0,
            "Model.patch_dropout": 0.0,
            "Model.feature_noise_std": 0.0,
            "Model.label_smoothing": 0.1,
        },
    },
    "global_mlp": {
        "balanced": True,
        "options": {
            "Model.potential_type": "mlp",
            "Model.stability_weight": 0.1,
            "Model.patch_dropout": 0.1,
            "Model.feature_noise_std": 0.01,
        },
    },
    "global_mlp_reg": {
        "balanced": True,
        "options": {
            "Model.potential_type": "mlp",
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
            "Model.label_smoothing": 0.05,
        },
    },
    "gate_global": {
        "balanced": True,
        "options": {
            "Model.multiscale_gate_initial_bias": -2.0,
            "Model.multiscale_local_initial_scale": 0.1,
        },
    },
    "gate_mid": {
        "balanced": True,
        "options": {
            "Model.multiscale_gate_initial_bias": -1.0,
            "Model.multiscale_local_initial_scale": 0.25,
        },
    },
    "residual_prototype": {
        "balanced": True,
        "options": {
            "Model.potential_type": "residual_prototype",
            "Model.prototype_residual_initial_scale": 0.1,
            "Model.prototype_regularization_weight": 0.01,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
        },
    },
    "mixture_prototype": {
        "balanced": True,
        "options": {
            "Model.potential_type": "mixture_prototype",
            "Model.prototype_regularization_weight": 0.01,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
        },
    },
    "adaptive_prototype": {
        "balanced": True,
        "options": {
            "Model.potential_type": "adaptive_multiscale_prototype",
            "Model.multiscale_prototype_initial_scale": 0.05,
            "Model.prototype_regularization_weight": 0.01,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
        },
    },
    "moment_order2": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_default": {
        "balanced": True,
        "options": {"Model.moment_order": 2},
    },
    "m2_clean": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.stability_weight": 0.0,
            "Model.patch_dropout": 0.0,
            "Model.feature_noise_std": 0.0,
        },
    },
    "m2_smooth": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.stability_weight": 0.0,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
            "Model.label_smoothing": 0.05,
        },
    },
    "m2_lr1e4_default_aug": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-5,
        },
    },
    "m2_lr5e5_mild": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.optimizer.adamw_config.lr": 5e-5,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_gate_mid": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.multiscale_gate_initial_bias": -1.0,
            "Model.multiscale_local_initial_scale": 0.25,
            "Model.optimizer.adamw_config.lr": 1e-4,
        },
    },
    "m2_tail_cold": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.tail_temperature": 0.1,
            "Model.optimizer.adamw_config.lr": 1e-4,
        },
    },
    "m2_tail_warm": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.tail_temperature": 0.5,
            "Model.optimizer.adamw_config.lr": 1e-4,
        },
    },
    "m2_tail4": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.num_tail_scores": 4,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_tail16": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.num_tail_scores": 16,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_route_cold": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.local_route_temperature": 0.1,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_route_warm": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.local_route_temperature": 0.5,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_global_mlp": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.potential_type": "mlp",
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_class_conditional": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.potential_type": "class_conditional_multiscale",
            "Model.num_local_routes": 14,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_hybrid": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.potential_type": "hybrid_multiscale",
            "Model.num_local_routes": 14,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_residual_class": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.potential_type": "residual_class_multiscale",
            "Model.num_local_routes": 14,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_dropout10": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.1,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_dropout20": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.2,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "m2_no_stability": {
        "balanced": True,
        "options": {
            "Model.moment_order": 2,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.0,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "ordinal_002": {
        "balanced": True,
        "options": {"Model.ordinal_weight": 0.02},
    },
    "ordinal_005": {
        "balanced": True,
        "options": {"Model.ordinal_weight": 0.05},
    },
    "ordinal_010": {
        "balanced": True,
        "options": {"Model.ordinal_weight": 0.1},
    },
    "ordinal_020": {
        "balanced": True,
        "options": {"Model.ordinal_weight": 0.2},
    },
    "ordinal_050": {
        "balanced": True,
        "options": {"Model.ordinal_weight": 0.5},
    },
    "ordinal_010_mild": {
        "balanced": True,
        "options": {
            "Model.ordinal_weight": 0.1,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "ordinal_020_mild": {
        "balanced": True,
        "options": {
            "Model.ordinal_weight": 0.2,
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "ordinal_010_unbalanced": {
        "balanced": False,
        "options": {"Model.ordinal_weight": 0.1},
    },
    "unbalanced_default": {
        "balanced": False,
        "options": {},
    },
    "ordinal_005_unbalanced": {
        "balanced": False,
        "options": {"Model.ordinal_weight": 0.05},
    },
    "ordinal_020_unbalanced": {
        "balanced": False,
        "options": {"Model.ordinal_weight": 0.2},
    },
    "fusion_norm_default": {
        "balanced": True,
        "options": {
            "Model.input_group_l2_normalize": True,
            "Model.input_group_size": 1024,
        },
    },
    "fusion_norm_mlp": {
        "balanced": True,
        "options": {
            "Model.input_group_l2_normalize": True,
            "Model.input_group_size": 1024,
            "Model.potential_type": "mlp",
        },
    },
    "fusion_norm_mlp_mild": {
        "balanced": True,
        "options": {
            "Model.input_group_l2_normalize": True,
            "Model.input_group_size": 1024,
            "Model.potential_type": "mlp",
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "fusion_norm_adaptive": {
        "balanced": True,
        "options": {
            "Model.input_group_l2_normalize": True,
            "Model.input_group_size": 1024,
            "Model.potential_type": "adaptive_multiscale",
            "Model.optimizer.adamw_config.lr": 1e-4,
            "Model.optimizer.adamw_config.weight_decay": 1e-4,
            "Model.dropout": 0.15,
            "Model.stability_weight": 0.05,
            "Model.patch_dropout": 0.05,
            "Model.feature_noise_std": 0.005,
        },
    },
    "capacity_h128_s64": {
        "balanced": True,
        "options": {
            "Model.hidden_dim": 128,
            "Model.sketch_dim": 64,
            "Model.potential_hidden_dim": 64,
        },
    },
    "capacity_h384_s128": {
        "balanced": True,
        "options": {
            "Model.hidden_dim": 384,
            "Model.sketch_dim": 128,
            "Model.potential_hidden_dim": 192,
        },
    },
    "capacity_h384_s192": {
        "balanced": True,
        "options": {
            "Model.hidden_dim": 384,
            "Model.sketch_dim": 192,
            "Model.potential_hidden_dim": 192,
        },
    },
    "capacity_h512_s128": {
        "balanced": True,
        "options": {
            "Model.hidden_dim": 512,
            "Model.sketch_dim": 128,
            "Model.potential_hidden_dim": 256,
        },
    },
    "capacity_h512_s256": {
        "balanced": True,
        "options": {
            "Model.hidden_dim": 512,
            "Model.sketch_dim": 256,
            "Model.potential_hidden_dim": 256,
        },
    },
    "capacity_routes8": {
        "balanced": True,
        "options": {"Model.num_local_routes": 8},
    },
    "capacity_routes16": {
        "balanced": True,
        "options": {"Model.num_local_routes": 16},
    },
    "capacity_potential256": {
        "balanced": True,
        "options": {"Model.potential_hidden_dim": 256},
    },
}


def command(args, name, spec, output):
    result = [
        str(args.python),
        "experiments/run_benchmark.py",
        "--split", str(args.split),
        "--dataset-name", "BRACS",
        "--num-classes", "7",
        "--log-root", str(output),
        "--models", "MIR_MIL",
        "--seeds", str(args.seed),
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--max-instances", str(args.max_instances),
        "--in-dim", str(args.in_dim),
        "--feature", args.feature,
        "--protocol", args.protocol,
        "--split-id", "official-train-val",
        "--comparison-id", f"bracs-mir-hpo-{name}-seed{args.seed}",
        "--device", "0",
        "--num-workers", str(args.num_workers),
    ]
    result.append("--balanced" if spec["balanced"] else "--no-balanced")
    if args.wandb:
        result.extend(
            ["--wandb", "--wandb-project", args.wandb_project]
        )
    for key, value in spec["options"].items():
        if isinstance(value, float):
            serialized = f"{value:.10f}".rstrip("0").rstrip(".")
            if serialized == "0":
                serialized = "0.0"
        else:
            serialized = str(value)
        result.extend(["--model-option", f"{key}={serialized}"])
    return result


def best_result(output):
    matches = list(output.glob("BRACS/MIR_MIL/time_*/Best_Log*.csv"))
    if not matches:
        raise FileNotFoundError(f"no Best_Log under {output}")
    path = max(matches, key=lambda item: item.stat().st_mtime)
    row = pd.read_csv(path).iloc[0]
    return {
        "best_log": str(path.resolve()),
        "best_epoch": int(row["epoch"]),
        "val_acc": float(row["val_acc"]),
        "val_bacc": float(row["val_bacc"]),
        "val_macro_auc": float(row["val_macro_auc"]),
        "val_macro_f1": float(row["val_macro_f1"]),
    }


def main():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=Path,
        default=Path(
            "/data15/data15_5/fanhao/datasets/BRACS/MIRMIL_FEATURES/"
            "metadata/BRACS_uni_split_official_train_val.csv"
        ),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=project_root / "artifacts/bracs_mir_hpo",
    )
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-instances", type=int, default=4096)
    parser.add_argument("--in-dim", type=int, default=1024)
    parser.add_argument("--feature", default="uni")
    parser.add_argument("--protocol", default="bracs-mir-hpo-v1")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="MIR-MIL")
    parser.add_argument(
        "--variants", nargs="+", choices=sorted(VARIANTS),
        default=None,
    )
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    work = queue.Queue()
    selected = args.variants or list(VARIANTS)
    for name in selected:
        spec = VARIANTS[name]
        work.put((name, spec))
    records, lock = [], threading.Lock()

    def worker(gpu):
        while True:
            try:
                name, spec = work.get_nowait()
            except queue.Empty:
                return
            output = args.output_dir / name
            output.mkdir(parents=True, exist_ok=True)
            cmd = command(args, name, spec, output)
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            with open(output / "train.stdout.log", "a", encoding="utf-8") as log:
                return_code = subprocess.run(
                    cmd, cwd=project_root, env=environment,
                    stdout=log, stderr=subprocess.STDOUT,
                ).returncode
            record = {
                "variant": name,
                "return_code": return_code,
                "gpu": gpu,
                "balanced": spec["balanced"],
                "options": json.dumps(spec["options"], sort_keys=True),
                "finished_at": datetime.now(timezone.utc).isoformat(),
            }
            if return_code == 0:
                record.update(best_result(output))
            with lock:
                records.append(record)
                pd.DataFrame(records).sort_values("variant").to_csv(
                    args.output_dir / "hpo_results.csv", index=False
                )
            print(name, "completed" if return_code == 0 else "failed", flush=True)
            work.task_done()

    threads = [
        threading.Thread(target=worker, args=(gpu.strip(),))
        for gpu in args.gpus.split(",") if gpu.strip()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    frame = pd.DataFrame(records).sort_values(
        "val_macro_auc", ascending=False, na_position="last"
    )
    frame.to_csv(args.output_dir / "hpo_results.csv", index=False)
    print(frame.to_string(index=False))
    if (frame["return_code"] != 0).any():
        raise SystemExit("one or more HPO variants failed")


if __name__ == "__main__":
    main()
