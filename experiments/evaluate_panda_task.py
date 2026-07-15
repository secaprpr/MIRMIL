"""Evaluate one trained repository model on a held-out test split."""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.general_utils import set_global_seed
from utils.loop_utils import cal_scores, get_cam_1d
from utils.model_utils import get_model_from_yaml
from utils.wandb_utils import WandbTracker, file_sha256, metric_payload
from utils.wsi_utils import WSI_Coord_Dataset, WSI_Dataset
from utils.yaml_utils import read_yaml


def cal_independent_scores(scores, labels, num_classes):
    scores = torch.tensor(scores)
    labels = torch.tensor(labels).view(-1)
    predicted_classes = torch.argmax(scores, dim=1)
    class_auc = {
        f"auc_class_{class_index}": roc_auc_score(
            y_true=(labels.numpy() == class_index).astype(int),
            y_score=scores[:, class_index].numpy(),
        )
        for class_index in range(num_classes)
    }
    macro_auc = sum(class_auc.values()) / len(class_auc)
    min_class_auc = min(class_auc.values())
    class_1_auc = class_auc.get("auc_class_1", macro_auc)
    eps = 1e-12
    return {
        "acc": accuracy_score(labels.numpy(), predicted_classes.numpy()),
        "bacc": balanced_accuracy_score(labels.numpy(), predicted_classes.numpy()),
        "macro_auc": macro_auc,
        "micro_auc": macro_auc,
        "weighted_auc": macro_auc,
        "min_class_auc": min_class_auc,
        "macro_auc_hmean_auc_class_1": (
            2.0 * macro_auc * class_1_auc / max(macro_auc + class_1_auc, eps)
        ),
        "macro_f1": f1_score(labels.numpy(), predicted_classes.numpy(), average="macro"),
        "micro_f1": f1_score(labels.numpy(), predicted_classes.numpy(), average="micro"),
        "weighted_f1": f1_score(labels.numpy(), predicted_classes.numpy(), average="weighted"),
        "macro_recall": recall_score(labels.numpy(), predicted_classes.numpy(), average="macro"),
        "micro_recall": recall_score(labels.numpy(), predicted_classes.numpy(), average="micro"),
        "weighted_recall": recall_score(labels.numpy(), predicted_classes.numpy(), average="weighted"),
        "macro_pre": precision_score(labels.numpy(), predicted_classes.numpy(), average="macro"),
        "micro_pre": precision_score(labels.numpy(), predicted_classes.numpy(), average="micro"),
        "weighted_pre": precision_score(labels.numpy(), predicted_classes.numpy(), average="weighted"),
        "quadratic_kappa": cohen_kappa_score(
            labels.numpy(), predicted_classes.numpy(), weights="quadratic"
        ),
        "linear_kappa": cohen_kappa_score(
            labels.numpy(), predicted_classes.numpy(), weights="linear"
        ),
        "confusion_mat": confusion_matrix(labels.numpy(), predicted_classes.numpy()),
        **class_auc,
    }


def prediction_scores(output, score_mode):
    if score_mode == "softmax":
        return torch.softmax(output["logits"].squeeze(0), dim=0)
    if score_mode == "ovr_sigmoid":
        if "ovr_logits" not in output:
            raise ValueError("score_mode=ovr_sigmoid requires output['ovr_logits']")
        return torch.sigmoid(output["ovr_logits"].squeeze(0))
    if score_mode == "raw_logits":
        return output["logits"].squeeze(0)
    raise ValueError(f"Unsupported score_mode: {score_mode}")


def evaluate(args):
    config = read_yaml(args.config)
    config.Dataset.dataset_csv_path = str(args.split.resolve())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.General.device = 0 if device.type == "cuda" else "cpu"
    config.Model.max_instances = args.max_instances
    set_global_seed(int(config.General.seed))
    spatial = str(config.General.MODEL_NAME) == "MAMBA2D_MIL"
    dataset_class = WSI_Coord_Dataset if spatial else WSI_Dataset
    dataset = dataset_class(
        str(args.split),
        "test",
        max_instances=args.max_instances,
        sampling="uniform",
    )
    loader = DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    model = get_model_from_yaml(config)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    is_dtfd = str(config.General.MODEL_NAME) == "DTFD_MIL"
    if is_dtfd:
        checkpoint_keys = ("classifier", "attention", "dimReduction", "attCls")
        for module, key in zip(model, checkpoint_keys):
            module.to(device)
            module.load_state_dict(checkpoint[key])
            module.eval()
    else:
        model = model.to(device)
        model.load_state_dict(checkpoint)
        model.eval()
    probabilities, labels = [], []
    with torch.no_grad():
        for bag, label in loader:
            bag = bag.float().to(device)
            if is_dtfd:
                classifier, attention, dim_reduction, attention_classifier = model
                pseudo_features = []
                instance_per_group = (
                    int(config.Model.total_instance) // int(config.Model.num_Group)
                )
                for sub_features in torch.chunk(
                    bag.squeeze(0), int(config.Model.num_Group), dim=0
                ):
                    middle_features = dim_reduction(sub_features)
                    attention_scores = attention(middle_features).squeeze(0)
                    weighted_features = torch.einsum(
                        "ns,n->ns", middle_features, attention_scores
                    )
                    attention_feature = weighted_features.sum(dim=0, keepdim=True)
                    patch_logits = get_cam_1d(
                        classifier, weighted_features.unsqueeze(0)
                    ).squeeze(0).transpose(0, 1)
                    sort_index = torch.sort(
                        torch.softmax(patch_logits, dim=1)[:, -1],
                        descending=True,
                    ).indices
                    top_max = sort_index[:instance_per_group].long()
                    top_min = sort_index[-instance_per_group:].long()
                    if str(config.Model.distill) == "MaxMinS":
                        selected = torch.cat((top_max, top_min), dim=0)
                        pseudo_features.append(
                            middle_features.index_select(0, selected)
                        )
                    elif str(config.Model.distill) == "MaxS":
                        pseudo_features.append(
                            middle_features.index_select(0, top_max)
                        )
                    elif str(config.Model.distill) == "AFS":
                        pseudo_features.append(attention_feature)
                    else:
                        raise ValueError(
                            f"Unsupported DTFD distill mode: {config.Model.distill}"
                        )
                output = attention_classifier(torch.cat(pseudo_features, dim=0))
            elif spatial:
                in_dim = int(config.Model.in_dim)
                features = bag[..., :in_dim]
                coords = bag[..., in_dim:in_dim + 2].squeeze(0)
                output = model(features, coords=coords)
            else:
                output = model(bag)
            probability = prediction_scores(output, args.score_mode)
            probabilities.append(probability.cpu().numpy())
            labels.append(int(label.item()))
    if args.score_mode == "softmax":
        metrics = cal_scores(probabilities, labels, int(config.General.num_classes))
    else:
        metrics = cal_independent_scores(
            probabilities, labels, int(config.General.num_classes)
        )
    rows = []
    for slide_path, label, probability in zip(
        dataset.slide_path_list, labels, probabilities
    ):
        row = {"slide_path": slide_path, "label": label}
        row.update(
            {f"prob_{index}": float(value) for index, value in enumerate(probability)}
        )
        rows.append(row)
    return config, metrics, rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--max-instances", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--wandb-project", default="MIR-MIL")
    parser.add_argument("--dataset-name", default="PANDA")
    parser.add_argument(
        "--score-mode",
        choices=["softmax", "ovr_sigmoid", "raw_logits"],
        default="softmax",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config, metrics, predictions = evaluate(args)
    prediction_path = args.output_dir / "predictions.csv"
    pd.DataFrame(predictions).to_csv(prediction_path, index=False)
    result = {
        "feature": args.feature,
        "model": args.model,
        "seed": args.seed,
        "num_slides": len(predictions),
        "acc": metrics["acc"],
        "bacc": metrics["bacc"],
        "macro_auc": metrics["macro_auc"],
        "macro_f1": metrics["macro_f1"],
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "split_sha256": file_sha256(args.split),
    }
    with open(args.output_dir / "result.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    tracker = WandbTracker.for_job(
        enabled=True,
        project=args.wandb_project,
        entity=None,
        mode="online",
        name=(
            f"{args.dataset_name}_{args.feature}_{args.model}_"
            f"{args.model}_seed{args.seed}_eval"
        ),
        group=(
            f"{args.dataset_name}_{args.feature}_{args.model}_"
            "protocol-v1_official"
        ),
        job_type="eval",
        tags=[
            f"dataset:{args.dataset_name.lower()}", f"feature:{args.feature}",
            f"model:{args.model}", "test:sealed", f"seed:{args.seed}",
        ],
        config={
            "dataset": args.dataset_name,
            "feature": args.feature,
            "model": args.model,
            "seed": args.seed,
            "split_sha256": result["split_sha256"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "patch_budget": args.max_instances,
            "sampling": "uniform",
        },
        output_dir=str(args.output_dir),
    )
    tracker.summary(
        {
            **metric_payload(
                "test",
                {
                    "acc": result["acc"],
                    "bacc": result["bacc"],
                    "macro_auc": result["macro_auc"],
                    "macro_f1": result["macro_f1"],
                },
            ),
            "test/num_slides": result["num_slides"],
        }
    )
    tracker.log_artifact(
        name=f"{tracker.run.id}-predictions",
        artifact_type="predictions",
        files=[str(prediction_path)],
        metadata={
            "split_sha256": result["split_sha256"],
            "checkpoint_sha256": result["checkpoint_sha256"],
        },
    )
    tracker.finish()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
