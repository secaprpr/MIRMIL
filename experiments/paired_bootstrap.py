import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)


def macro_auc(labels, probabilities):
    if probabilities.shape[1] == 2:
        return roc_auc_score(labels, probabilities[:, 1])
    return roc_auc_score(
        labels,
        probabilities,
        average="macro",
        multi_class="ovr",
    )


def metric_score(metric, labels, probabilities):
    if metric == "macro_auc":
        return macro_auc(labels, probabilities)
    predictions = probabilities.argmax(axis=1)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(labels, predictions)
    if metric == "macro_f1":
        return f1_score(labels, predictions, average="macro")
    if metric == "accuracy":
        return accuracy_score(labels, predictions)
    raise ValueError(f"Unknown metric: {metric}")


def load_prediction(path):
    frame = pd.read_csv(path)
    probability_columns = sorted(
        [column for column in frame if column.startswith("prob_")],
        key=lambda column: int(column.split("_")[1]),
    )
    return frame, probability_columns


def paired_frames(
    input_dir,
    budget,
    id_column="slide_path",
    model_a="OT_MIL",
    model_b="MO_MIL",
):
    pairs = []
    pattern = os.path.join(
        input_dir, f"{model_a}_seed*_budget{budget}.csv"
    )
    for model_a_path in sorted(glob.glob(pattern)):
        seed = int(
            os.path.basename(model_a_path).split("_seed")[1].split("_")[0]
        )
        model_b_path = os.path.join(
            input_dir, f"{model_b}_seed{seed}_budget{budget}.csv"
        )
        if not os.path.isfile(model_b_path):
            raise FileNotFoundError(model_b_path)
        model_a_frame, probability_columns = load_prediction(model_a_path)
        model_b_frame, model_b_probability_columns = load_prediction(
            model_b_path
        )
        if probability_columns != model_b_probability_columns:
            raise ValueError(f"Probability columns differ for seed {seed}")
        if id_column not in model_a_frame or id_column not in model_b_frame:
            raise ValueError(f"Missing prediction ID column: {id_column}")
        merged = model_a_frame.merge(
            model_b_frame,
            on=[id_column, "label", "seed", "budget"],
            suffixes=("_a", "_b"),
            validate="one_to_one",
        )
        if (
            len(merged) != len(model_a_frame)
            or len(merged) != len(model_b_frame)
        ):
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
    parser.add_argument(
        "--metric",
        choices=[
            "macro_auc",
            "balanced_accuracy",
            "macro_f1",
            "accuracy",
        ],
        default="macro_auc",
    )
    parser.add_argument("--output")
    parser.add_argument(
        "--id-column",
        default="slide_path",
        help="Row identifier used to pair model predictions",
    )
    parser.add_argument("--model-a", default="OT_MIL")
    parser.add_argument("--model-b", default="MO_MIL")
    args = parser.parse_args()

    pairs = paired_frames(
        args.input_dir,
        args.budget,
        args.id_column,
        args.model_a,
        args.model_b,
    )
    rng = np.random.default_rng(args.seed)
    observed = []
    bootstrap_differences = np.empty(args.iterations, dtype=np.float64)
    prepared = []
    for seed, frame, probability_columns in pairs:
        labels = frame["label"].to_numpy()
        model_a_probabilities = frame[
            [f"{column}_a" for column in probability_columns]
        ].to_numpy()
        model_b_probabilities = frame[
            [f"{column}_b" for column in probability_columns]
        ].to_numpy()
        difference = metric_score(
            args.metric, labels, model_a_probabilities
        ) - metric_score(
            args.metric, labels, model_b_probabilities
        )
        observed.append(
            {"seed": seed, f"{args.metric}_difference": difference}
        )
        prepared.append(
            (labels, model_a_probabilities, model_b_probabilities)
        )

    for iteration in range(args.iterations):
        seed_differences = []
        for (
            labels,
            model_a_probabilities,
            model_b_probabilities,
        ) in prepared:
            indices = stratified_indices(labels, rng)
            seed_differences.append(
                metric_score(
                    args.metric,
                    labels[indices],
                    model_a_probabilities[indices],
                )
                - metric_score(
                    args.metric,
                    labels[indices],
                    model_b_probabilities[indices],
                )
            )
        bootstrap_differences[iteration] = np.mean(seed_differences)

    result = {
        "budget": args.budget,
        "iterations": args.iterations,
        "bootstrap_seed": args.seed,
        "metric": args.metric,
        "model_a": args.model_a,
        "model_b": args.model_b,
        "num_training_seeds": len(pairs),
        "per_seed": observed,
        f"mean_{args.metric}_difference": float(
            np.mean(
                [
                    item[f"{args.metric}_difference"]
                    for item in observed
                ]
            )
        ),
        "ci_95_low": float(np.quantile(bootstrap_differences, 0.025)),
        "ci_95_high": float(np.quantile(bootstrap_differences, 0.975)),
        "probability_model_a_better": float(
            np.mean(bootstrap_differences > 0)
        ),
    }
    if args.model_a == "OT_MIL" and args.model_b == "MO_MIL":
        result["probability_ot_better"] = result[
            "probability_model_a_better"
        ]
    default_name = (
        f"paired_bootstrap_budget{args.budget}.json"
        if args.metric == "macro_auc"
        else f"paired_bootstrap_{args.metric}_budget{args.budget}.json"
    )
    output_path = args.output or os.path.join(args.input_dir, default_name)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
