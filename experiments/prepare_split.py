import argparse
import hashlib
import json
import os

import pandas as pd


def stable_score(slide_id, seed):
    digest = hashlib.sha256(f"{seed}:{slide_id}".encode("utf-8")).hexdigest()
    return int(digest, 16)


def deterministic_stratified_split(df, seed, train_ratio, val_ratio):
    assignments = []
    for label, group in df.groupby("label", sort=True):
        ordered = group.assign(
            _score=group["slide_id"].map(lambda value: stable_score(value, seed))
        ).sort_values(["_score", "slide_id"])
        count = len(ordered)
        train_count = round(count * train_ratio)
        val_count = round(count * val_ratio)
        train_count = min(max(train_count, 1), count)
        val_count = min(max(val_count, 1), max(count - train_count, 0))
        split = (
            ["train"] * train_count
            + ["val"] * val_count
            + ["test"] * (count - train_count - val_count)
        )
        ordered = ordered.assign(split=split)
        assignments.append(ordered.drop(columns="_score"))
    return pd.concat(assignments, ignore_index=True).sort_values("slide_id")


def build_training_csv(assignments, output_path):
    columns = {}
    for split in ("train", "val", "test"):
        subset = assignments[assignments["split"] == split].reset_index(drop=True)
        columns[f"{split}_slide_path"] = subset["slide_path"]
        columns[f"{split}_label"] = subset["label"]
    pd.DataFrame(columns).to_csv(output_path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--slide-column", default="slide_id")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--feature-extension", default=".pt")
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--train-ratio", type=float, default=0.6)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--max-per-class", type=int, default=0)
    args = parser.parse_args()

    if args.train_ratio <= 0 or args.val_ratio <= 0:
        raise ValueError("train_ratio and val_ratio must be positive")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be less than 1")

    source = pd.read_csv(args.labels)
    frame = source[[args.slide_column, args.label_column]].rename(
        columns={args.slide_column: "slide_id", args.label_column: "label"}
    )
    frame["slide_id"] = frame["slide_id"].astype(str)
    if not pd.api.types.is_numeric_dtype(frame["label"]):
        label_names = sorted(frame["label"].unique())
        label_mapping = {name: index for index, name in enumerate(label_names)}
        frame["label"] = frame["label"].map(label_mapping)
    else:
        label_mapping = {
            str(label): int(label) for label in sorted(frame["label"].unique())
        }
        frame["label"] = frame["label"].astype(int)

    if args.max_per_class > 0:
        sampled = []
        for _, group in frame.groupby("label", sort=True):
            ordered = group.assign(
                _score=group["slide_id"].map(
                    lambda value: stable_score(value, args.seed)
                )
            ).sort_values(["_score", "slide_id"])
            sampled.append(ordered.head(args.max_per_class).drop(columns="_score"))
        frame = pd.concat(sampled, ignore_index=True)

    frame["slide_path"] = frame["slide_id"].map(
        lambda slide_id: os.path.join(
            args.feature_dir, slide_id + args.feature_extension
        )
    )
    missing = frame.loc[~frame["slide_path"].map(os.path.isfile), "slide_path"]
    if len(missing):
        raise FileNotFoundError(
            f"{len(missing)} feature files are missing; first: {missing.iloc[0]}"
        )
    if frame["slide_id"].duplicated().any():
        raise ValueError("Duplicate slide IDs found")

    assignments = deterministic_stratified_split(
        frame, args.seed, args.train_ratio, args.val_ratio
    )
    os.makedirs(args.output_dir, exist_ok=True)
    prefix = os.path.join(args.output_dir, args.dataset_name)
    assignment_path = prefix + "_assignments.csv"
    training_path = prefix + "_split.csv"
    manifest_path = prefix + "_manifest.json"
    assignments.to_csv(assignment_path, index=False)
    build_training_csv(assignments, training_path)

    counts = (
        assignments.groupby(["split", "label"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    manifest = {
        "dataset": args.dataset_name,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": 1 - args.train_ratio - args.val_ratio,
        },
        "num_slides": len(assignments),
        "num_classes": int(assignments["label"].nunique()),
        "label_mapping": label_mapping,
        "counts": counts,
        "assignment_sha256": hashlib.sha256(
            assignments[["slide_id", "label", "split"]]
            .to_csv(index=False)
            .encode("utf-8")
        ).hexdigest(),
    }
    with open(manifest_path, "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)
    print(json.dumps(manifest, indent=2))
    print(f"training_csv={training_path}")


if __name__ == "__main__":
    main()
