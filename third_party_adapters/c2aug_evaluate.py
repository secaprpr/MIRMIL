"""Evaluate a C2Aug Lightning checkpoint on held-out test features."""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import wandb
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TestBags(Dataset):
    def __init__(self, split, max_instances):
        frame = pd.read_csv(split)
        self.paths = frame["test_slide_path"].dropna().tolist()
        self.labels = frame["test_label"].dropna().astype(int).tolist()
        self.max_instances = max_instances
        if len(self.paths) != len(self.labels):
            raise ValueError("test paths and labels differ")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        features = torch.load(
            self.paths[index], map_location="cpu", weights_only=True
        )
        if features.ndim == 3:
            features = features.squeeze(0)
        if len(features) > self.max_instances:
            indices = torch.linspace(
                0, len(features) - 1, self.max_instances
            ).long()
            features = features[indices]
        return features, self.labels[index], self.paths[index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature", choices=["r50", "uni"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-instances", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-project", default="MIR-MIL")
    parser.add_argument("--dataset-name", default="PANDA")
    parser.add_argument("--num-classes", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from models import ModelInterface

    feat_name = "res50" if args.feature == "r50" else "uni"
    model = ModelInterface.load_from_checkpoint(
        args.checkpoint,
        model="transmil",
        feat=feat_name,
        debug=False,
        encoder="attn",
        num_classes=args.num_classes,
        input_dim=1024,
        map_location="cuda:0",
    ).cuda().eval()
    dataset = TestBags(args.split, args.max_instances)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    probabilities, labels, paths = [], [], []
    with torch.no_grad():
        for features, label, path in loader:
            output = model.encoder(
                data=features.float().cuda(),
                label=label.long().cuda(),
            )
            probabilities.append(
                torch.softmax(output["logits"].squeeze(0), dim=0)
                .cpu().numpy()
            )
            labels.append(int(label.item()))
            paths.append(path[0])
    probabilities = np.asarray(probabilities)
    labels_array = np.asarray(labels)
    predictions = probabilities.argmax(axis=1)
    result = {
        "feature": args.feature,
        "model": "C2AUG",
        "seed": args.seed,
        "num_slides": len(labels),
        "acc": float(accuracy_score(labels_array, predictions)),
        "bacc": float(balanced_accuracy_score(labels_array, predictions)),
        "macro_auc": float(
            roc_auc_score(
                labels_array, probabilities,
                average="macro", multi_class="ovr",
            )
        ),
        "macro_f1": float(
            f1_score(labels_array, predictions, average="macro")
        ),
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": sha256(args.checkpoint),
        "split_sha256": sha256(args.split),
    }
    rows = []
    for path, label, probability in zip(paths, labels, probabilities):
        row = {"slide_path": path, "label": label}
        row.update(
            {f"prob_{index}": float(value) for index, value in enumerate(probability)}
        )
        rows.append(row)
    prediction_path = args.output_dir / "predictions.csv"
    pd.DataFrame(rows).to_csv(prediction_path, index=False)
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    run = wandb.init(
        project=args.wandb_project,
        name=(
            f"{args.dataset_name}_{args.feature}_TransMIL_"
            f"C2Aug_seed{args.seed}_eval"
        ),
        group=(
            f"{args.dataset_name}_{args.feature}_TransMIL_"
            "C2Aug_protocol-v1_official"
        ),
        job_type="eval",
        tags=[
            f"dataset:{args.dataset_name.lower()}",
            f"feature:{args.feature}", "model:c2aug",
            "test:sealed", f"seed:{args.seed}",
        ],
        config={
            "dataset": args.dataset_name, "feature": args.feature,
            "model": "C2AUG", "seed": args.seed,
            "split_sha256": result["split_sha256"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "patch_budget": args.max_instances, "sampling": "uniform",
        },
    )
    run.summary.update(
        {
            "test/accuracy": result["acc"],
            "test/balanced_accuracy": result["bacc"],
            "test/macro_auc_ovr": result["macro_auc"],
            "test/macro_f1": result["macro_f1"],
            "test/num_slides": result["num_slides"],
        }
    )
    try:
        artifact = wandb.Artifact(
            f"{run.id}-predictions", type="predictions",
            metadata={
                "split_sha256": result["split_sha256"],
                "checkpoint_sha256": result["checkpoint_sha256"],
            },
        )
        artifact.add_file(str(prediction_path))
        run.log_artifact(artifact)
    except Exception as exc:
        print(f"W&B prediction artifact upload failed: {exc}")
    run.finish()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
