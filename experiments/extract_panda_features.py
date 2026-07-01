"""Launch and audit sharded PANDA feature extraction."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import h5py
import pandas as pd
import torch


BACKBONES = {
    "r50": ("resnet50_imagenet", 1024),
    "uni": ("uni", 1024),
}


def build_command(args, shard_index, num_shards):
    backbone, _ = BACKBONES[args.encoder]
    extractor = args.project_root / "feature_extractor" / "create_pt_features.py"
    return [
        str(args.python),
        str(extractor),
        "--data_h5_dir",
        str(args.patch_dir),
        "--process_wsi_paths_csv",
        str(args.source_csv),
        "--feat_dir",
        str(args.output_dir),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--target_patch_size",
        "224",
        "--backbone",
        backbone,
        "--pretrained_weights_dir",
        str(args.weights_dir),
        "--num_shards",
        str(num_shards),
        "--shard_index",
        str(shard_index),
        "--failure_log",
        str(args.output_dir / "logs" / f"failures_shard{shard_index}.csv"),
    ]


def audit_outputs(source_csv, patch_dir, output_dir, expected_dim):
    source = pd.read_csv(source_csv)
    records = []
    for wsi_path in source["wsi_path"]:
        slide_id = Path(wsi_path).stem
        coord_path = patch_dir / "patches" / f"{slide_id}.h5"
        feature_path = output_dir / "h5_files" / f"{slide_id}.h5"
        pt_path = output_dir / "pt_files" / f"{slide_id}.pt"
        status = "ok"
        detail = ""
        try:
            with h5py.File(coord_path, "r") as coord_file:
                coord_count = int(coord_file["coords"].shape[0])
            with h5py.File(feature_path, "r") as feature_file:
                shape = tuple(feature_file["features"].shape)
                feature_coords = tuple(feature_file["coords"].shape)
            pt = torch.load(pt_path, map_location="cpu", weights_only=True)
            if shape != (coord_count, expected_dim):
                raise ValueError(f"H5 feature shape {shape}, coords={coord_count}")
            if feature_coords != (coord_count, 2):
                raise ValueError(f"H5 coordinate shape {feature_coords}")
            if tuple(pt.shape) != shape:
                raise ValueError(f"PT shape {tuple(pt.shape)}, H5={shape}")
            if not torch.isfinite(pt).all():
                raise ValueError("PT contains non-finite values")
        except (OSError, KeyError, RuntimeError, ValueError, EOFError) as exc:
            status = "failed"
            detail = str(exc)
        records.append(
            {"slide_id": slide_id, "status": status, "detail": detail}
        )
    return pd.DataFrame(records)


def parse_args():
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--encoder", choices=sorted(BACKBONES), required=True)
    parser.add_argument("--source-csv", type=Path, required=True)
    parser.add_argument("--patch-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument(
        "--python", type=Path, default=Path(sys.executable)
    )
    parser.set_defaults(project_root=project_root)
    return parser.parse_args()


def main():
    args = parse_args()
    args.source_csv = args.source_csv.resolve()
    args.patch_dir = args.patch_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.weights_dir = args.weights_dir.resolve()
    args.project_root = args.project_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"
    log_dir.mkdir(exist_ok=True)

    if not args.audit_only:
        gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
        if not gpus:
            raise ValueError("--gpus must contain at least one GPU")
        processes = []
        handles = []
        for shard_index, gpu in enumerate(gpus):
            log_path = log_dir / f"extract_shard{shard_index}.log"
            handle = open(log_path, "a", encoding="utf-8")
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = gpu
            command = build_command(args, shard_index, len(gpus))
            process = subprocess.Popen(
                command,
                cwd=args.project_root / "feature_extractor",
                env=environment,
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
            processes.append((shard_index, gpu, process))
            handles.append(handle)
        failures = []
        for shard_index, gpu, process in processes:
            return_code = process.wait()
            if return_code:
                failures.append((shard_index, gpu, return_code))
        for handle in handles:
            handle.close()
        if failures:
            raise RuntimeError(f"feature extraction shards failed: {failures}")

    _, expected_dim = BACKBONES[args.encoder]
    audit = audit_outputs(
        args.source_csv, args.patch_dir, args.output_dir, expected_dim
    )
    audit_path = log_dir / "feature_audit.csv"
    audit.to_csv(audit_path, index=False)
    failed = audit[audit["status"] != "ok"]
    print(f"audited={len(audit)} ok={len(audit) - len(failed)} failed={len(failed)}")
    print(f"audit={audit_path}")
    if len(failed):
        raise RuntimeError(f"{len(failed)} feature outputs failed audit")


if __name__ == "__main__":
    main()
