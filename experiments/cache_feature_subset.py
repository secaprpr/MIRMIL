import argparse
import hashlib
import json
import os
import pickle
import tempfile
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone

import h5py
import numpy as np
import pandas as pd
import torch


SPLITS = ("train", "val", "test")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def candidate_indices(num_instances, max_candidates):
    if max_candidates <= 0 or num_instances <= max_candidates:
        return None
    return np.linspace(
        0, num_instances - 1, max_candidates, dtype=np.int64
    )


def output_name(source_path):
    path_hash = hashlib.sha1(
        os.path.abspath(source_path).encode("utf-8")
    ).hexdigest()[:12]
    stem = os.path.splitext(os.path.basename(source_path))[0]
    return f"{stem}_{path_hash}.pt"


def load_subset(source_path, max_candidates):
    if source_path.endswith(".h5"):
        with h5py.File(source_path, "r") as file:
            dataset = file["features"]
            r50_source = file.attrs.get("r50_source")
            uni_source = file.attrs.get("uni_source")
            if r50_source and uni_source:
                with h5py.File(r50_source, "r") as r50_file, h5py.File(
                    uni_source, "r"
                ) as uni_file:
                    r50_features = r50_file["features"][:]
                    uni_features = uni_file["features"][:]
                if r50_features.shape[0] != uni_features.shape[0]:
                    raise ValueError(
                        "Paired H5 feature row counts differ: "
                        f"{r50_features.shape} vs {uni_features.shape}"
                    )
                indices = candidate_indices(
                    r50_features.shape[0], max_candidates
                )
                if indices is not None:
                    r50_features = r50_features[indices]
                    uni_features = uni_features[indices]
                return torch.from_numpy(
                    np.concatenate((r50_features, uni_features), axis=1)
                )
            if dataset.ndim == 2:
                num_instances = dataset.shape[0]
                indices = candidate_indices(num_instances, max_candidates)
                features = dataset[:] if indices is None else dataset[indices]
            elif dataset.ndim == 3 and dataset.shape[0] == 1:
                num_instances = dataset.shape[1]
                indices = candidate_indices(num_instances, max_candidates)
                features = (
                    dataset[0]
                    if indices is None
                    else dataset[0, indices, :]
                )
            else:
                raise ValueError(
                    f"Unexpected H5 feature shape {dataset.shape} in "
                    f"{source_path}"
                )
        return torch.from_numpy(features)

    loaded = torch.load(source_path, map_location="cpu", weights_only=False)
    if isinstance(loaded, dict):
        if "feats" in loaded:
            loaded = loaded["feats"]
        elif "features" in loaded:
            loaded = loaded["features"]
        else:
            raise ValueError(
                f"Unknown feature dictionary keys in {source_path}: "
                f"{list(loaded.keys())}"
            )
    if loaded.ndim == 3:
        loaded = loaded.squeeze(0)
    indices = candidate_indices(loaded.shape[0], max_candidates)
    return loaded if indices is None else loaded[torch.from_numpy(indices)]


def cache_feature(source_path, output_dir, max_candidates, overwrite=False):
    destination = os.path.join(output_dir, output_name(source_path))
    if os.path.isfile(destination) and not overwrite:
        try:
            cached = torch.load(
                destination, map_location="cpu", weights_only=True
            )
            valid = cached.ndim == 2 and (
                max_candidates <= 0 or cached.shape[0] <= max_candidates
            )
            if valid:
                return (
                    destination,
                    int(cached.shape[0]),
                    int(cached.shape[1]),
                    False,
                )
        except (
            OSError,
            RuntimeError,
            EOFError,
            pickle.UnpicklingError,
            AttributeError,
        ):
            pass

    features = load_subset(source_path, max_candidates).cpu().contiguous()
    if features.ndim != 2:
        raise ValueError(
            f"Expected two-dimensional features in {source_path}, "
            f"got shape {tuple(features.shape)}"
        )
    handle, temporary = tempfile.mkstemp(
        prefix=f"{os.path.basename(destination)}.",
        suffix=".tmp",
        dir=output_dir,
    )
    os.close(handle)
    try:
        torch.save(features, temporary)
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return destination, int(features.shape[0]), int(features.shape[1]), True


def feature_paths(split_frame):
    paths = []
    for split in SPLITS:
        column = f"{split}_slide_path"
        if column in split_frame:
            paths.extend(split_frame[column].dropna().astype(str).tolist())
    return list(dict.fromkeys(paths))


def cache_feature_task(arguments):
    index, total, source_path, output_dir, max_candidates, overwrite = (
        arguments
    )
    destination, rows, dimensions, created = cache_feature(
        source_path,
        output_dir,
        max_candidates,
        overwrite=overwrite,
    )
    return {
        "index": index,
        "total": total,
        "source_path": source_path,
        "cached_path": os.path.abspath(destination),
        "instances": rows,
        "dimensions": dimensions,
        "created": created,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-split", required=True)
    parser.add_argument("--max-candidates", type=int, default=4096)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.max_candidates <= 0:
        raise ValueError("--max-candidates must be positive")
    split_frame = pd.read_csv(args.split)
    source_paths = feature_paths(split_frame)
    if not source_paths:
        raise ValueError(f"No feature paths found in {args.split}")
    missing = [path for path in source_paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} source files are missing; first: {missing[0]}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(
        os.path.dirname(os.path.abspath(args.output_split)), exist_ok=True
    )
    tasks = [
        (
            index,
            len(source_paths),
            source_path,
            args.output_dir,
            args.max_candidates,
            args.overwrite,
        )
        for index, source_path in enumerate(source_paths, start=1)
    ]
    if args.workers == 1:
        records = [cache_feature_task(task) for task in tasks]
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            records = list(executor.map(cache_feature_task, tasks))
    path_mapping = {
        record["source_path"]: record["cached_path"]
        for record in records
    }
    for record in records:
        print(
            f"[{record['index']}/{record['total']}] "
            f"{record['source_path']} -> {record['cached_path']} "
            f"{record['instances']}x{record['dimensions']}"
        )

    cached_split = split_frame.copy()
    for split in SPLITS:
        column = f"{split}_slide_path"
        if column in cached_split:
            cached_split[column] = cached_split[column].map(
                lambda path: (
                    path_mapping[str(path)] if pd.notna(path) else path
                )
            )
    cached_split.to_csv(args.output_split, index=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_split": os.path.abspath(args.split),
        "source_split_sha256": file_sha256(args.split),
        "output_split": os.path.abspath(args.output_split),
        "output_split_sha256": file_sha256(args.output_split),
        "output_dir": os.path.abspath(args.output_dir),
        "max_candidates": args.max_candidates,
        "sampling": "uniform",
        "num_slides": len(records),
        "total_instances": sum(record["instances"] for record in records),
        "total_bytes": sum(
            os.path.getsize(record["cached_path"]) for record in records
        ),
        "records": [
            {
                key: value
                for key, value in record.items()
                if key not in {"index", "total"}
            }
            for record in records
        ],
    }
    manifest_path = f"{args.output_split}.manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    summary = {
        key: value for key, value in manifest.items() if key != "records"
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
