import argparse
import json
import math

import numpy as np
import pandas as pd


def parse_cohort(value):
    if "=" not in value or "," not in value:
        raise argparse.ArgumentTypeError(
            "Cohorts must use NAME=BINARY_RESULTS,MULTICLASS_RESULTS"
        )
    name, paths = value.split("=", 1)
    binary_path, multiclass_path = paths.split(",", 1)
    if not name or not binary_path or not multiclass_path:
        raise argparse.ArgumentTypeError(
            "Cohorts must use NAME=BINARY_RESULTS,MULTICLASS_RESULTS"
        )
    return name, binary_path, multiclass_path


def model_difference(path, metric):
    frame = pd.read_csv(path)
    required = {"seed", "model", metric}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing result columns: {sorted(missing)}")
    pivot = frame.pivot(index="seed", columns="model", values=metric)
    for model in ("OT_MIL", "MO_MIL"):
        if model not in pivot:
            raise ValueError(f"Missing model results: {model}")
    return pivot["OT_MIL"] - pivot["MO_MIL"]


def cohort_preference(name, binary_path, multiclass_path, metric):
    binary = model_difference(binary_path, metric)
    multiclass = model_difference(multiclass_path, metric)
    seeds = binary.index.intersection(multiclass.index)
    if len(seeds) != len(binary) or len(seeds) != len(multiclass):
        raise ValueError(f"Training seeds differ for cohort {name}")
    preference = multiclass.loc[seeds] - binary.loc[seeds]
    return {
        "cohort": name,
        "seeds": [int(seed) for seed in seeds],
        "binary_differences": [float(value) for value in binary.loc[seeds]],
        "multiclass_differences": [
            float(value) for value in multiclass.loc[seeds]
        ],
        "task_type_preferences": [float(value) for value in preference],
        "mean_task_type_preference": float(preference.mean()),
    }


def binomial_sign_pvalue(positive, total):
    return sum(
        math.comb(total, successes) * 0.5**total
        for successes in range(positive, total + 1)
    )


def hierarchical_bootstrap(cohorts, iterations, seed):
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations, dtype=np.float64)
    for iteration in range(iterations):
        cohort_indices = rng.choice(
            len(cohorts), size=len(cohorts), replace=True
        )
        cohort_means = []
        for index in cohort_indices:
            values = np.asarray(
                cohorts[index]["task_type_preferences"], dtype=np.float64
            )
            seed_indices = rng.choice(
                len(values), size=len(values), replace=True
            )
            cohort_means.append(float(values[seed_indices].mean()))
        samples[iteration] = np.mean(cohort_means)
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        action="append",
        type=parse_cohort,
        required=True,
        help="Repeat NAME=BINARY_RESULTS,MULTICLASS_RESULTS",
    )
    parser.add_argument(
        "--metric",
        choices=["macro_auc", "acc", "bacc", "macro_f1"],
        default="macro_auc",
    )
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")

    cohorts = [
        cohort_preference(name, binary, multiclass, args.metric)
        for name, binary, multiclass in args.cohort
    ]
    bootstrap = hierarchical_bootstrap(
        cohorts, args.iterations, args.seed
    )
    cohort_means = [
        cohort["mean_task_type_preference"] for cohort in cohorts
    ]
    positive_cohorts = sum(value > 0 for value in cohort_means)
    all_preferences = [
        value
        for cohort in cohorts
        for value in cohort["task_type_preferences"]
    ]
    result = {
        "metric": args.metric,
        "iterations": args.iterations,
        "bootstrap_seed": args.seed,
        "num_cohorts": len(cohorts),
        "num_seed_pairs": len(all_preferences),
        "cohorts": cohorts,
        "mean_task_type_preference": float(np.mean(cohort_means)),
        "ci_95_low": float(np.quantile(bootstrap, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap, 0.975)),
        "probability_multiclass_preference_positive": float(
            np.mean(bootstrap > 0)
        ),
        "positive_cohorts": positive_cohorts,
        "cohort_level_one_sided_sign_pvalue": binomial_sign_pvalue(
            positive_cohorts, len(cohorts)
        ),
        "positive_seed_pairs": sum(value > 0 for value in all_preferences),
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
