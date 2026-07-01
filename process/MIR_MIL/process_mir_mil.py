import glob
import os
from copy import deepcopy

import torch
from torch.utils.data import DataLoader
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
from utils.wandb_utils import WandbTracker, file_sha256, metric_payload


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
    tracker = WandbTracker.for_training(args, process_pipeline)

    generator = torch.Generator()
    generator.manual_seed(args.General.seed)
    set_global_seed(args.General.seed)
    if args.Dataset.balanced_sampler.use:
        sampler = train_dataset.get_balanced_sampler(
            replacement=args.Dataset.balanced_sampler.replacement
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=1,
            sampler=sampler,
            num_workers=args.General.num_workers,
            generator=generator,
        )
    else:
        train_loader = DataLoader(
            train_dataset,
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
    optimizer, base_lr = get_optimizer(args, model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    criterion = get_criterion(
        args.Model.criterion,
        label_smoothing=getattr(args.Model, "label_smoothing", 0.0),
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
        )
        if ema is not None:
            ema.update(model)
        validation_model = ema.model if ema is not None else model
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
        tracker.log_epoch(
            epoch + 1,
            train_loss,
            components,
            val_loss,
            val_metrics,
            elapsed,
        )
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
    final_model = ema.model if ema is not None else model
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
    best_index = max(min(best_epoch - 1, len(epoch_log["epoch"]) - 1), 0)
    best_val_metrics = {
        "acc": epoch_log["val_acc"][best_index],
        "bacc": epoch_log["val_bacc"][best_index],
        "macro_auc": epoch_log["val_macro_auc"][best_index],
        "macro_f1": epoch_log["val_macro_f1"][best_index],
    }
    tracker.summary(
        {
            **{
                key.replace("val/", "val/best_"): value
                for key, value in metric_payload(
                    "val", best_val_metrics
                ).items()
            },
            "train/best_epoch": int(best_epoch),
            "train/stop_epoch": int(last_epoch),
        }
    )
    best_checkpoints = glob.glob(
        os.path.join(args.Logs.now_log_dir, "Best*.pth")
    )
    tracking = getattr(getattr(args, "Tracking", {}), "wandb", {})
    upload_checkpoints = bool(
        getattr(tracking, "upload_checkpoints", False)
    )
    if best_checkpoints and upload_checkpoints:
        checkpoint_path = best_checkpoints[0]
        tracker.log_artifact(
            name=f"{tracker.run.id}-best-checkpoint"
            if tracker.enabled
            else "best-checkpoint",
            artifact_type="model",
            files=[checkpoint_path],
            metadata={
                "sha256": file_sha256(checkpoint_path),
                "best_epoch": int(best_epoch),
            },
        )
    elif best_checkpoints:
        checkpoint_path = best_checkpoints[0]
        tracker.summary(
            {
                "provenance/checkpoint_path": os.path.abspath(
                    checkpoint_path
                ),
                "provenance/checkpoint_sha256": file_sha256(
                    checkpoint_path
                ),
            }
        )
    tracker.log_artifact(
        name=f"{tracker.run.id}-split"
        if tracker.enabled
        else "split",
        artifact_type="split",
        files=[args.Dataset.dataset_csv_path],
        metadata={"sha256": file_sha256(args.Dataset.dataset_csv_path)},
    )
    tracker.finish()
