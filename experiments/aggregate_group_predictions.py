import argparse
import glob
import json
import os
import re
import sys

import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from utils.loop_utils import cal_scores


PREDICTION_PATTERN = re.compile(
    r"^(?P<model>.+)_seed(?P<seed>\d+)_budget(?P<budget>\d+)\.csv$"
)


def load_cache_mapping(cache_manifest):
    with open(cache_manifest, "r", encoding="utf-8") as file:
        manifest = json.load(file)
    return {
        os.path.abspath(record["source_path"]): os.path.abspath(
            record["cached_path"]
        )
        for record in manifest["records"]
    }


def build_group_mapping(assignments_path, cache_manifest=None, split="test"):
    assignments = pd.read_csv(assignments_path)
    assignments = assignments[assignments["split"] == split].copy()

    direct_columns = {"slide_path", "patient_id", "label", "split"}
    cached_columns = {"feature_path", "case_id", "label", "split"}
    if direct_columns.issubset(assignments.columns):
        assignments["slide_path"] = assignments["slide_path"].map(os.path.abspath)
        if cache_manifest is not None:
            source_to_cache = load_cache_mapping(cache_manifest)
            assignments["slide_path"] = assignments["slide_path"].map(
                source_to_cache.get
            )
            if assignments["slide_path"].isna().any():
                raise ValueError(
                    "At least one direct slide path is absent from the "
                    "cache manifest"
                )
        assignments["case_id"] = assignments["patient_id"].astype(str)
    elif cached_columns.issubset(assignments.columns):
        if cache_manifest is None:
            raise ValueError(
                "A cache manifest is required for feature_path assignments"
            )
        source_to_cache = load_cache_mapping(cache_manifest)
        assignments["slide_path"] = assignments["feature_path"].map(
            lambda path: source_to_cache.get(os.path.abspath(path))
        )
        if assignments["slide_path"].isna().any():
            first = assignments.loc[
                assignments["slide_path"].isna(), "feature_path"
            ].iloc[0]
            raise ValueError(f"Feature is absent from cache manifest: {first}")
    else:
        expected = sorted(direct_columns) + sorted(cached_columns)
        raise ValueError(
            "Assignments must contain either direct or cached mapping "
            f"columns; expected: {expected}"
        )

    if assignments["slide_path"].duplicated().any():
        raise ValueError("Duplicate cached slide paths in assignments")
    if assignments.groupby("case_id")["label"].nunique().max() > 1:
        raise ValueError("At least one group contains conflicting labels")
    return assignments[["slide_path", "case_id", "label"]]


def aggregate_prediction_frame(predictions, group_mapping):
    frame = predictions.copy()
    frame["slide_path"] = frame["slide_path"].map(os.path.abspath)
    merged = frame.merge(
        group_mapping,
        on="slide_path",
        how="left",
        suffixes=("_prediction", ""),
        validate="one_to_one",
    )
    if merged["case_id"].isna().any():
        first = merged.loc[merged["case_id"].isna(), "slide_path"].iloc[0]
        raise ValueError(f"Prediction slide has no group mapping: {first}")
    if not (
        merged["label_prediction"].astype(int) == merged["label"].astype(int)
    ).all():
        raise ValueError("Prediction and assignment labels disagree")

    probability_columns = sorted(
        [column for column in merged if column.startswith("prob_")],
        key=lambda value: int(value.split("_")[1]),
    )
    grouped = (
        merged.groupby("case_id", sort=True)
        .agg(
            label=("label", "first"),
            num_slides=("slide_path", "size"),
            **{
                column: (column, "mean")
                for column in probability_columns
            },
        )
        .reset_index()
    )
    return grouped


def evaluate_grouped_frame(grouped):
    probability_columns = sorted(
        [column for column in grouped if column.startswith("prob_")],
        key=lambda value: int(value.split("_")[1]),
    )
    probabilities = grouped[probability_columns].to_numpy()
    labels = grouped["label"].astype(int).to_numpy()
    return cal_scores(probabilities, labels, probabilities.shape[1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument(
        "--cache-manifest",
        help=(
            "Map source feature paths to cached prediction paths; required "
            "for feature_path/case_id assignments"
        ),
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    prediction_paths = sorted(
        glob.glob(os.path.join(args.input_dir, "*_seed*_budget*.csv"))
    )
    if not prediction_paths:
        raise FileNotFoundError(
            f"No prediction CSV files found in {args.input_dir}"
        )
    group_mapping = build_group_mapping(
        args.assignments, args.cache_manifest, split=args.split
    )
    os.makedirs(args.output_dir, exist_ok=True)
    results = []
    for path in prediction_paths:
        match = PREDICTION_PATTERN.match(os.path.basename(path))
        if not match:
            continue
        predictions = pd.read_csv(path)
        grouped = aggregate_prediction_frame(predictions, group_mapping)
        grouped["seed"] = int(match.group("seed"))
        grouped["budget"] = int(match.group("budget"))
        metrics = evaluate_grouped_frame(grouped)
        result = {
            "model": match.group("model"),
            "seed": int(match.group("seed")),
            "budget": int(match.group("budget")),
            "num_groups": len(grouped),
            "num_slides": int(grouped["num_slides"].sum()),
            "macro_auc": metrics["macro_auc"],
            "acc": metrics["acc"],
            "bacc": metrics["bacc"],
            "macro_f1": metrics["macro_f1"],
        }
        results.append(result)
        output_name = os.path.basename(path)
        grouped.to_csv(os.path.join(args.output_dir, output_name), index=False)
        print(json.dumps(result, indent=2))

    result_frame = pd.DataFrame(results).sort_values(
        ["budget", "model", "seed"]
    )
    result_frame.to_csv(
        os.path.join(args.output_dir, "grouped_results.csv"), index=False
    )
    aggregate = (
        result_frame.groupby(["budget", "model"])
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
        os.path.join(args.output_dir, "grouped_aggregate.csv"), index=False
    )
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
