import glob
import os
from copy import deepcopy

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from utils.general_utils import (
    add_epoch_info_log,
    attach_test_result,
    cal_is_stopping,
    init_epoch_info_log,
    set_global_seed,
)
from utils.loop_utils import mir_train_loop, mir_val_loop
from utils.model_utils import (
    get_criterion,
    get_model_from_yaml,
    get_optimizer,
    get_scheduler,
    model_select,
    save_last_model,
    save_log,
)
from utils.process_utils import get_process_pipeline
from utils.wsi_utils import WSI_Coord_Dataset, WSI_Dataset


class ExponentialMovingAverage:
    def __init__(self, model, decay):
        if not 0 < decay < 1:
            raise ValueError("EMA decay must be between zero and one")
        self.model = deepcopy(model).eval()
        self.decay = float(decay)
        self.updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        target = self.model.state_dict()
        if self.updates == 0:
            self.model.load_state_dict(source)
        else:
            for name, value in target.items():
                source_value = source[name].detach()
                if value.is_floating_point():
                    value.mul_(self.decay).add_(
                        source_value, alpha=1.0 - self.decay
                    )
                else:
                    value.copy_(source_value)
        self.updates += 1


class StochasticWeightAverage:
    def __init__(self, model):
        self.model = deepcopy(model).eval()
        self.updates = 0
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        source = model.state_dict()
        target = self.model.state_dict()
        if self.updates == 0:
            self.model.load_state_dict(source)
        else:
            weight = 1.0 / float(self.updates + 1)
            for name, value in target.items():
                source_value = source[name].detach()
                if value.is_floating_point():
                    value.mul_(1.0 - weight).add_(source_value, alpha=weight)
                else:
                    value.copy_(source_value)
        self.updates += 1


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        bag, label = self.dataset[index]
        return bag, label, torch.tensor(index, dtype=torch.long)

    def __getattr__(self, name):
        return getattr(self.dataset, name)


def load_distillation_targets(path, dataset, num_classes):
    if not path:
        return None
    frame = pd.read_csv(path)
    required_columns = ["slide_path", *[f"prob_{idx}" for idx in range(num_classes)]]
    missing_columns = [column for column in required_columns if column not in frame]
    if missing_columns:
        raise ValueError(
            "Distillation probability file is missing columns: "
            f"{missing_columns}"
        )
    if frame["slide_path"].duplicated().any():
        duplicated = frame.loc[
            frame["slide_path"].duplicated(), "slide_path"
        ].head(5)
        raise ValueError(
            "Distillation probability file contains duplicate slide_path "
            f"entries, e.g. {duplicated.tolist()}"
        )
    by_slide = frame.set_index("slide_path")
    rows = []
    missing_slides = []
    prob_columns = [f"prob_{idx}" for idx in range(num_classes)]
    for slide_path in dataset.slide_path_list:
        if slide_path not in by_slide.index:
            missing_slides.append(slide_path)
            continue
        rows.append(by_slide.loc[slide_path, prob_columns].astype(float).to_numpy())
    if missing_slides:
        raise ValueError(
            "Distillation probability file is missing train slides, e.g. "
            f"{missing_slides[:5]}"
        )
    targets = torch.tensor(rows, dtype=torch.float32)
    if torch.any(targets < 0):
        raise ValueError("Distillation probabilities must be non-negative")
    row_sums = targets.sum(dim=1, keepdim=True)
    if torch.any(row_sums <= 0):
        raise ValueError("Distillation probability rows must have positive mass")
    return targets / row_sums


def _build_class_weights(args, train_dataset, device):
    mode = str(getattr(args.Model, "class_weighting", "none")).lower()
    if mode in {"", "none", "false", "off"}:
        return None
    labels = torch.tensor(train_dataset.labels_list, dtype=torch.long)
    num_classes = int(args.General.num_classes)
    counts = torch.bincount(labels, minlength=num_classes).float()
    if torch.any(counts == 0):
        raise ValueError(
            "Cannot build class weights with missing train classes: "
            f"counts={counts.tolist()}"
        )
    if mode == "inverse":
        weights = counts.sum() / (num_classes * counts)
    elif mode == "sqrt_inverse":
        weights = torch.sqrt(counts.sum() / (num_classes * counts))
    elif mode == "effective":
        beta = float(getattr(args.Model, "class_weight_beta", 0.999))
        effective_num = 1.0 - torch.pow(
            torch.full_like(counts, beta), counts
        )
        weights = (1.0 - beta) / effective_num
        weights = weights / weights.mean()
    else:
        raise ValueError(
            "Unknown Model.class_weighting="
            f"{mode}. Supported: none, inverse, sqrt_inverse, effective"
        )
    return weights.to(device)


def process_MIR_MIL(args):
    dataset_class = (
        WSI_Coord_Dataset
        if int(getattr(args.Model, "coordinate_dim", 0)) == 2
        else WSI_Dataset
    )
    dataset_kwargs = {
        "max_instances": args.Model.max_instances,
        "sampling": args.Model.sampling,
    }
    train_dataset = dataset_class(
        args.Dataset.dataset_csv_path, "train", **dataset_kwargs
    )
    val_dataset = dataset_class(
        args.Dataset.dataset_csv_path,
        "val",
        max_instances=args.Model.max_instances,
        sampling="uniform",
    )
    test_dataset = dataset_class(
        args.Dataset.dataset_csv_path,
        "test",
        max_instances=args.Model.max_instances,
        sampling="uniform",
    )
    process_pipeline = get_process_pipeline(val_dataset, test_dataset)
    args.General.process_pipeline = process_pipeline

    generator = torch.Generator()
    generator.manual_seed(args.General.seed)
    set_global_seed(args.General.seed)
    distillation_weight = float(
        getattr(args.Model, "distillation_weight", 0.0)
    )
    ranking_memory_weight = float(
        getattr(args.Model, "ranking_memory_weight", 0.0)
    )
    ranking_memory_warmup_epochs = int(
        getattr(args.Model, "ranking_memory_warmup_epochs", 0)
    )
    if ranking_memory_warmup_epochs < 0:
        raise ValueError("ranking_memory_warmup_epochs must be non-negative")
    distillation_targets = None
    if distillation_weight > 0:
        distillation_targets = load_distillation_targets(
            getattr(args.Model, "distillation_prob_path", None),
            train_dataset,
            int(args.General.num_classes),
        )
    if distillation_weight > 0 or ranking_memory_weight > 0:
        train_dataset_for_loader = IndexedDataset(train_dataset)
    else:
        train_dataset_for_loader = train_dataset
    ranking_memory = None
    ranking_memory_labels = None
    ranking_memory_valid = None
    if ranking_memory_weight > 0:
        ranking_memory = torch.zeros(
            len(train_dataset),
            int(args.General.num_classes),
            dtype=torch.float32,
        )
        ranking_memory_labels = torch.tensor(
            train_dataset.labels_list,
            dtype=torch.long,
        )
        ranking_memory_valid = torch.zeros(
            len(train_dataset),
            dtype=torch.bool,
        )
    if args.Dataset.balanced_sampler.use:
        sampler = train_dataset.get_balanced_sampler(
            replacement=args.Dataset.balanced_sampler.replacement,
            strategy=getattr(args.Dataset.balanced_sampler, "strategy", "weighted"),
            samples_per_class=getattr(
                args.Dataset.balanced_sampler,
                "samples_per_class",
                0,
            ),
            seed=args.General.seed,
        )
        train_loader = DataLoader(
            train_dataset_for_loader,
            batch_size=1,
            sampler=sampler,
            num_workers=args.General.num_workers,
            generator=generator,
        )
    else:
        train_loader = DataLoader(
            train_dataset_for_loader,
            batch_size=1,
            shuffle=True,
            num_workers=args.General.num_workers,
            generator=generator,
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.General.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.General.num_workers,
    )

    device = torch.device(f"cuda:{args.General.device}")
    model = get_model_from_yaml(args).to(device)
    ema_decay = float(getattr(args.Model, "ema_decay", 0.0))
    ema = (
        ExponentialMovingAverage(model, ema_decay)
        if ema_decay > 0
        else None
    )
    swa_start_epoch = int(getattr(args.Model, "swa_start_epoch", 0) or 0)
    swa = (
        StochasticWeightAverage(model)
        if swa_start_epoch > 0
        else None
    )
    if ema is not None and swa is not None:
        raise ValueError("Use EMA or SWA, not both, for MIR_MIL training")
    optimizer, base_lr = get_optimizer(args, model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    class_weights = _build_class_weights(args, train_dataset, device)
    criterion = get_criterion(
        args.Model.criterion,
        label_smoothing=getattr(args.Model, "label_smoothing", 0.0),
        class_weights=class_weights,
        focal_gamma=getattr(args.Model, "focal_gamma", 0.0),
    )

    epoch_log = init_epoch_info_log()
    reverse = args.General.best_model_metric == "val_loss"
    best_value = 9999 if reverse else 0
    best_epoch = 1
    for epoch in tqdm(range(args.General.num_epochs), colour="GREEN"):
        active_scheduler = (
            warmup_scheduler
            if epoch + 1 <= args.Model.scheduler.warmup
            else scheduler
        )
        train_loss, elapsed, components = mir_train_loop(
            device,
            model,
            train_loader,
            criterion,
            optimizer,
            active_scheduler,
            distillation_targets=distillation_targets,
            distillation_weight=distillation_weight,
            distillation_temperature=float(
                getattr(args.Model, "distillation_temperature", 1.0)
            ),
            distillation_min_entropy=float(
                getattr(args.Model, "distillation_min_entropy", 0.0)
            ),
            distillation_max_confidence=float(
                getattr(args.Model, "distillation_max_confidence", 1.0)
            ),
            distillation_entropy_weight_power=float(
                getattr(args.Model, "distillation_entropy_weight_power", 0.0)
            ),
            ranking_memory=ranking_memory,
            ranking_memory_labels=ranking_memory_labels,
            ranking_memory_valid=ranking_memory_valid,
            ranking_memory_weight=ranking_memory_weight,
            ranking_memory_margin=float(
                getattr(args.Model, "ranking_memory_margin", 0.1)
            ),
            ranking_memory_momentum=float(
                getattr(args.Model, "ranking_memory_momentum", 0.9)
            ),
            ranking_memory_max_pairs=int(
                getattr(args.Model, "ranking_memory_max_pairs", 64)
            ),
            ranking_memory_class_indices=getattr(
                args.Model, "ranking_memory_class_indices", None
            ),
            ranking_memory_hard_mining=bool(
                getattr(args.Model, "ranking_memory_hard_mining", False)
            ),
            ranking_memory_apply_loss=epoch >= ranking_memory_warmup_epochs,
            ranking_memory_score_type=str(
                getattr(args.Model, "ranking_memory_score_type", "logit")
            ),
        )
        if ema is not None:
            ema.update(model)
        if swa is not None and epoch + 1 >= swa_start_epoch:
            swa.update(model)
        if ema is not None:
            validation_model = ema.model
        elif swa is not None and swa.updates > 0:
            validation_model = swa.model
        else:
            validation_model = model
        val_loss, val_metrics = mir_val_loop(
            device,
            args.General.num_classes,
            validation_model,
            val_loader,
            criterion,
        )
        print(
            f"EPOCH:{epoch + 1}, Train_Loss:{train_loss}, "
            f"Val_Loss:{val_loss}, Cost_Time:{elapsed}"
        )
        print("Train_Loss_Components:", components)
        print("Val_Metrics:", val_metrics)
        add_epoch_info_log(
            epoch_log,
            epoch,
            train_loss,
            val_loss,
            None,
            val_metrics,
            None,
        )
        best_value, best_epoch = model_select(
            reverse,
            args,
            validation_model.state_dict(),
            val_metrics,
            args.General.best_model_metric,
            best_value,
            epoch,
            best_epoch,
        )
        if cal_is_stopping(args, epoch_log, process_pipeline):
            print(f"Early Stop In EPOCH {epoch + 1}!")
            break

    last_epoch = epoch_log["epoch"][-1]
    if ema is not None:
        final_model = ema.model
    elif swa is not None and swa.updates > 0:
        final_model = swa.model
    else:
        final_model = model
    save_last_model(args, final_model.state_dict(), last_epoch)
    if not test_dataset.is_None_Dataset():
        checkpoints = glob.glob(
            os.path.join(args.Logs.now_log_dir, "Best*.pth")
        )
        if checkpoints:
            model.load_state_dict(
                torch.load(
                    checkpoints[0], map_location=device, weights_only=True
                )
            )
            result_epoch = best_epoch
        else:
            result_epoch = last_epoch
        test_loss, test_metrics = mir_val_loop(
            device,
            args.General.num_classes,
            model,
            test_loader,
            criterion,
        )
        attach_test_result(
            epoch_log, result_epoch - 1, test_loss, test_metrics
        )
        print("Final_Test_Metrics:", test_metrics)
    save_log(args, epoch_log, best_epoch, process_pipeline)
