"""Read-only BRACS split and result protocol audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


GROUPS = ("train", "val", "test")


def _paths_and_labels(df, group):
    paths = df[f"{group}_slide_path"].dropna().astype(str).tolist()
    labels = df[f"{group}_label"].dropna().astype(int).tolist()
    return paths, labels


def _stem(path):
    return Path(path).stem


def audit_splits(r50_split, uni_split):
    r50 = pd.read_csv(r50_split)
    uni = pd.read_csv(uni_split)
    report = {}
    for group in GROUPS:
        r_paths, r_labels = _paths_and_labels(r50, group)
        u_paths, u_labels = _paths_and_labels(uni, group)
        r_stems = [_stem(path) for path in r_paths]
        u_stems = [_stem(path) for path in u_paths]
        report[group] = {
            "r50_count": len(r_paths),
            "uni_count": len(u_paths),
            "path_stems_match": r_stems == u_stems,
            "labels_match": r_labels == u_labels,
            "label_counts": (
                pd.Series(r_labels)
                .value_counts()
                .sort_index()
                .astype(int)
                .to_dict()
            ),
            "duplicate_stems": sorted(
                stem for stem in set(r_stems) if r_stems.count(stem) > 1
            ),
        }
    return report


def audit_seed_results(path):
    df = pd.read_csv(path)
    required = {
        "feature",
        "model",
        "seed",
        "num_slides",
        "acc",
        "bacc",
        "macro_auc",
        "macro_f1",
    }
    missing = sorted(required - set(df.columns))
    metric_cols = ["acc", "bacc", "macro_auc", "macro_f1"]
    finite = df[metric_cols].notna().all().all()
    ranges_ok = {
        col: bool(((df[col] >= 0) & (df[col] <= 1)).all())
        for col in metric_cols
    }
    seeds_per_group = (
        df.groupby(["feature", "model"])["seed"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    slide_counts = (
        df.groupby(["feature", "model"])["num_slides"]
        .nunique()
        .astype(int)
        .to_dict()
    )
    return {
        "rows": int(len(df)),
        "missing_columns": missing,
        "metrics_finite": bool(finite),
        "metric_ranges_ok": ranges_ok,
        "seeds_per_feature_model_min": int(min(seeds_per_group.values())),
        "seeds_per_feature_model_max": int(max(seeds_per_group.values())),
        "unique_num_slide_counts_per_feature_model": {
            f"{feature}/{model}": int(value)
            for (feature, model), value in slide_counts.items()
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r50-split", type=Path, required=True)
    parser.add_argument("--uni-split", type=Path, required=True)
    parser.add_argument("--baseline-seed-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = {
        "r50_split": str(args.r50_split),
        "uni_split": str(args.uni_split),
        "baseline_seed_results": str(args.baseline_seed_results),
        "split_audit": audit_splits(args.r50_split, args.uni_split),
        "seed_result_audit": audit_seed_results(args.baseline_seed_results),
    }
    result["passed"] = (
        all(
            item["path_stems_match"]
            and item["labels_match"]
            and not item["duplicate_stems"]
            for item in result["split_audit"].values()
        )
        and not result["seed_result_audit"]["missing_columns"]
        and result["seed_result_audit"]["metrics_finite"]
        and all(result["seed_result_audit"]["metric_ranges_ok"].values())
        and result["seed_result_audit"]["seeds_per_feature_model_min"] == 3
        and result["seed_result_audit"]["seeds_per_feature_model_max"] == 3
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
