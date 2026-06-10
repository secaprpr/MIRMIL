import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def macro_auc(labels, probabilities):
    if probabilities.shape[1] == 2:
        return roc_auc_score(labels, probabilities[:, 1])
    return roc_auc_score(
        labels,
        probabilities,
        average="macro",
        multi_class="ovr",
    )


def load_prediction(path):
    frame = pd.read_csv(path)
    probability_columns = sorted(
        [column for column in frame if column.startswith("prob_")],
        key=lambda column: int(column.split("_")[1]),
    )
    return frame, probability_columns


def paired_frames(input_dir, budget):
    pairs = []
    pattern = os.path.join(input_dir, f"OT_MIL_seed*_budget{budget}.csv")
    for ot_path in sorted(glob.glob(pattern)):
        seed = int(os.path.basename(ot_path).split("_seed")[1].split("_")[0])
        mo_path = os.path.join(
            input_dir, f"MO_MIL_seed{seed}_budget{budget}.csv"
        )
        if not os.path.isfile(mo_path):
            raise FileNotFoundError(mo_path)
        ot_frame, probability_columns = load_prediction(ot_path)
        mo_frame, mo_probability_columns = load_prediction(mo_path)
        if probability_columns != mo_probability_columns:
            raise ValueError(f"Probability columns differ for seed {seed}")
        merged = ot_frame.merge(
            mo_frame,
            on=["slide_path", "label", "seed", "budget"],
            suffixes=("_ot", "_mo"),
            validate="one_to_one",
        )
        if len(merged) != len(ot_frame) or len(merged) != len(mo_frame):
            raise ValueError(f"Prediction rows differ for seed {seed}")
        pairs.append((seed, merged, probability_columns))
    if not pairs:
        raise FileNotFoundError(
            f"No paired predictions for budget {budget} in {input_dir}"
        )
    return pairs


def stratified_indices(labels, rng):
    sampled = []
    for label in np.unique(labels):
        class_indices = np.flatnonzero(labels == label)
        sampled.append(
            rng.choice(class_indices, size=len(class_indices), replace=True)
        )
    indices = np.concatenate(sampled)
    rng.shuffle(indices)
    return indices


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--budget", type=int, default=512)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--output")
    args = parser.parse_args()

    pairs = paired_frames(args.input_dir, args.budget)
    rng = np.random.default_rng(args.seed)
    observed = []
    bootstrap_differences = np.empty(args.iterations, dtype=np.float64)
    prepared = []
    for seed, frame, probability_columns in pairs:
        labels = frame["label"].to_numpy()
        ot_probabilities = frame[
            [f"{column}_ot" for column in probability_columns]
        ].to_numpy()
        mo_probabilities = frame[
            [f"{column}_mo" for column in probability_columns]
        ].to_numpy()
        difference = macro_auc(labels, ot_probabilities) - macro_auc(
            labels, mo_probabilities
        )
        observed.append({"seed": seed, "macro_auc_difference": difference})
        prepared.append((labels, ot_probabilities, mo_probabilities))

    for iteration in range(args.iterations):
        seed_differences = []
        for labels, ot_probabilities, mo_probabilities in prepared:
            indices = stratified_indices(labels, rng)
            seed_differences.append(
                macro_auc(labels[indices], ot_probabilities[indices])
                - macro_auc(labels[indices], mo_probabilities[indices])
            )
        bootstrap_differences[iteration] = np.mean(seed_differences)

    result = {
        "budget": args.budget,
        "iterations": args.iterations,
        "bootstrap_seed": args.seed,
        "num_training_seeds": len(pairs),
        "per_seed": observed,
        "mean_macro_auc_difference": float(
            np.mean([item["macro_auc_difference"] for item in observed])
        ),
        "ci_95_low": float(np.quantile(bootstrap_differences, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap_differences, 0.975)),
        "probability_ot_better": float(
            np.mean(bootstrap_differences > 0)
        ),
    }
    output_path = args.output or os.path.join(
        args.input_dir, f"paired_bootstrap_budget{args.budget}.json"
    )
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
