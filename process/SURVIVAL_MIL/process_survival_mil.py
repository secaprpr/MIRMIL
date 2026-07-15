import copy
import glob
import os
import time

import numpy as np
import pandas as pd
import torch
from addict import Dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from modules.SURVIVAL_MIL import SurvivalMILWrapper
from utils.model_utils import (
    get_model_from_yaml,
    get_optimizer,
    get_scheduler,
    save_best_model,
    save_last_model,
)
from utils.process_utils import get_process_pipeline
from utils.survival_dataset import SurvivalWSIDataset
from utils.survival_utils import (
    NLLSurvLoss,
    bootstrap_c_index,
    concordance_index,
)
from utils.general_utils import set_global_seed
from utils.yaml_utils import read_yaml


def process_SURVIVAL_MIL(args):
    train_dataset, val_dataset, test_dataset = build_survival_datasets(args)
    process_pipeline = get_process_pipeline(val_dataset, test_dataset)
    args.General.process_pipeline = process_pipeline

    generator = torch.Generator()
    generator.manual_seed(args.General.seed)
    set_global_seed(args.General.seed)
    num_workers = args.General.num_workers

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=num_workers,
        generator=generator,
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=num_workers
    )
    print("Survival DataLoader Ready!")
    print(f"Survival cutpoints: {train_dataset.cutpoints}")

    device = torch.device(f"cuda:{args.General.device}")
    model = build_survival_model(args).to(device)
    print("Survival Model Ready!")

    optimizer, base_lr = get_optimizer(args, model)
    scheduler, warmup_scheduler = get_scheduler(args, optimizer, base_lr)
    criterion = NLLSurvLoss(alpha=_survival_option(args, "alpha", 0.0))
    warmup_epoch = args.Model.scheduler.warmup

    epoch_info_log = init_survival_epoch_log()
    best_model_metric = args.General.best_model_metric
    reverse = best_model_metric == "val_loss"
    best_val_metric = np.inf if reverse else -np.inf
    best_epoch = 1

    print("Start Survival Process!")
    print("Using Process Pipeline:", process_pipeline)
    for epoch in tqdm(range(args.General.num_epochs), colour="GREEN"):
        now_scheduler = warmup_scheduler if epoch + 1 <= warmup_epoch else scheduler
        train_loss, cost_time = survival_train_loop(
            device, model, train_dataloader, criterion, optimizer, now_scheduler
        )

        if process_pipeline == "Train_Val_Test":
            val_loss, val_metrics = survival_val_loop(
                device, model, val_dataloader, criterion
            )
            test_loss, test_metrics = None, None
        elif process_pipeline == "Train_Val":
            val_loss, val_metrics = survival_val_loop(
                device, model, val_dataloader, criterion
            )
            test_loss, test_metrics = None, None
        else:
            val_loss, val_metrics, test_loss, test_metrics = None, None, None, None

        print("----------------SURVIVAL INFO----------------")
        print(
            f"EPOCH:{epoch + 1}, Train_Loss:{train_loss}, "
            f"Val_Loss:{val_loss}, Test_Loss:{test_loss}, Cost_Time:{cost_time}"
        )
        print(f"Val_Metrics: {val_metrics}")
        print(f"Test_Metrics: {test_metrics}")

        add_survival_epoch_log(
            epoch_info_log,
            epoch,
            train_loss,
            val_loss,
            test_loss,
            val_metrics,
            test_metrics,
        )
        best_val_metric, best_epoch = survival_model_select(
            reverse,
            args,
            model.state_dict(),
            val_metrics,
            best_model_metric,
            best_val_metric,
            epoch,
            best_epoch,
        )

        if survival_early_stop(args, epoch_info_log, process_pipeline, epoch):
            print(f"Early Stop In EPOCH {epoch + 1}!")
            save_last_model(args, model.state_dict(), epoch + 1)
            break

        if epoch + 1 == args.General.num_epochs:
            save_last_model(args, model.state_dict(), epoch + 1)

    final_test_loss, final_test_metrics = evaluate_test_once(
        args,
        device,
        model,
        test_dataset,
        test_dataloader,
        criterion,
        process_pipeline,
    )
    attach_final_test_result(
        epoch_info_log,
        best_epoch,
        final_test_loss,
        final_test_metrics,
    )
    save_survival_log(args, epoch_info_log, best_epoch)


def build_survival_datasets(args):
    survival_cfg = _survival_cfg(args)
    max_instances = int(args.Model.max_instances) if "max_instances" in args.Model else 0
    sampling = str(args.Model.sampling) if "sampling" in args.Model else "random"
    num_bins = int(_survival_option(args, "num_bins", args.General.num_classes))
    time_column = str(_survival_option(args, "time_column", "time_months"))
    event_column = str(_survival_option(args, "event_column", "event"))
    label_column = str(_survival_option(args, "label_column", "survival_label"))
    patient_column = str(_survival_option(args, "patient_column", "patient_id"))
    patient_level = bool(_survival_option(args, "patient_level", True))
    path = args.Dataset.dataset_csv_path

    train_dataset = SurvivalWSIDataset(
        path,
        "train",
        time_column=time_column,
        event_column=event_column,
        label_column=label_column,
        patient_column=patient_column,
        patient_level=patient_level,
        num_bins=num_bins,
        fit_cutpoints=True,
        max_instances=max_instances,
        sampling=sampling,
    )
    val_dataset = SurvivalWSIDataset(
        path,
        "val",
        time_column=time_column,
        event_column=event_column,
        label_column=label_column,
        patient_column=patient_column,
        patient_level=patient_level,
        num_bins=num_bins,
        cutpoints=train_dataset.cutpoints,
        max_instances=max_instances,
        sampling="uniform",
    )
    test_dataset = SurvivalWSIDataset(
        path,
        "test",
        time_column=time_column,
        event_column=event_column,
        label_column=label_column,
        patient_column=patient_column,
        patient_level=patient_level,
        num_bins=num_bins,
        cutpoints=train_dataset.cutpoints,
        max_instances=max_instances,
        sampling="uniform",
    )
    survival_cfg.cutpoints = train_dataset.cutpoints
    return train_dataset, val_dataset, test_dataset


def build_survival_model(args):
    backbone_args = build_backbone_args(args)
    backbone = get_model_from_yaml(backbone_args)
    num_bins = int(_survival_option(args, "num_bins", args.General.num_classes))
    return SurvivalMILWrapper(
        backbone=backbone,
        num_bins=num_bins,
        representation=_survival_option(args, "representation", "auto"),
        head_hidden_dim=int(_survival_option(args, "head_hidden_dim", 0)),
        dropout=float(_survival_option(args, "head_dropout", 0.0)),
        require_wsi_feature=bool(
            _survival_option(args, "require_wsi_feature", True)
        ),
    )


def build_backbone_args(args):
    backbone_config = args.Model.backbone_config if "backbone_config" in args.Model else None
    if backbone_config:
        backbone_args = read_yaml(backbone_config)
    else:
        backbone_args = copy.deepcopy(args)
    backbone_args.General.MODEL_NAME = str(args.Model.backbone)
    backbone_args.General.seed = args.General.seed
    backbone_args.General.device = args.General.device
    backbone_args.General.num_workers = args.General.num_workers
    backbone_args.General.num_classes = int(
        _survival_option(args, "backbone_num_outputs", args.General.num_classes)
    )
    if "in_dim" in args.Model:
        backbone_args.Model.in_dim = args.Model.in_dim
    return backbone_args


def survival_train_loop(device, model, loader, criterion, optimizer, scheduler):
    start = time.time()
    model.train()
    train_loss_log = 0.0
    for data in loader:
        optimizer.zero_grad()
        bag = data[0].to(device).float()
        labels = data[1].to(device).long()
        events = data[3].to(device).float()
        output = model(bag)
        loss = criterion(output["hazards"], output["survival"], labels, events)
        train_loss_log += loss.item()
        loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    return train_loss_log / max(len(loader), 1), time.time() - start


def survival_val_loop(device, model, loader, criterion, return_predictions=False):
    model.eval()
    loss_log = 0.0
    event_times = []
    events = []
    risks = []
    records = []
    with torch.no_grad():
        for data in loader:
            bag = data[0].to(device).float()
            labels = data[1].to(device).long()
            event_time = data[2].to(device).float()
            event = data[3].to(device).float()
            patient_id = data[4][0] if len(data) > 4 else None
            output = model(bag)
            loss = criterion(output["hazards"], output["survival"], labels, event)
            loss_log += loss.item()
            event_times.extend(event_time.cpu().numpy().reshape(-1).tolist())
            events.extend(event.cpu().numpy().reshape(-1).tolist())
            risks.extend(output["risk"].cpu().numpy().reshape(-1).tolist())
            if return_predictions:
                records.append(
                    build_prediction_record(
                        patient_id,
                        labels,
                        event_time,
                        event,
                        output,
                    )
                )

    prefer_sksurv = True
    c_result = concordance_index(
        event_times, events, risks, prefer_sksurv=prefer_sksurv
    )
    ci_low, ci_high = bootstrap_c_index(
        event_times,
        events,
        risks,
        n_bootstraps=0,
        prefer_sksurv=prefer_sksurv,
    )
    metrics = {
        "c_index": c_result.c_index,
        "c_index_source": c_result.source,
        "c_index_ci_low": ci_low,
        "c_index_ci_high": ci_high,
        "event_count": int(np.sum(events)),
        "sample_count": int(len(events)),
    }
    if return_predictions:
        return loss_log / max(len(loader), 1), metrics, records
    return loss_log / max(len(loader), 1), metrics


def build_prediction_record(patient_id, labels, event_time, event, output):
    hazards = output["hazards"].detach().cpu().numpy().reshape(-1)
    survival = output["survival"].detach().cpu().numpy().reshape(-1)
    record = {
        "patient_id": patient_id,
        "time": float(event_time.detach().cpu().numpy().reshape(-1)[0]),
        "event": int(event.detach().cpu().numpy().reshape(-1)[0]),
        "label": int(labels.detach().cpu().numpy().reshape(-1)[0]),
        "risk": float(output["risk"].detach().cpu().numpy().reshape(-1)[0]),
    }
    for index, value in enumerate(hazards):
        record[f"hazard_{index}"] = float(value)
    for index, value in enumerate(survival):
        record[f"survival_{index}"] = float(value)
    return record


def init_survival_epoch_log():
    return {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "test_loss": [],
        "val_c_index": [],
        "val_c_index_source": [],
        "val_c_index_ci_low": [],
        "val_c_index_ci_high": [],
        "val_event_count": [],
        "val_sample_count": [],
        "test_c_index": [],
        "test_c_index_source": [],
        "test_c_index_ci_low": [],
        "test_c_index_ci_high": [],
        "test_event_count": [],
        "test_sample_count": [],
    }


def add_survival_epoch_log(
    log, epoch, train_loss, val_loss, test_loss, val_metrics, test_metrics
):
    log["epoch"].append(epoch + 1)
    log["train_loss"].append(train_loss)
    log["val_loss"].append(val_loss)
    log["test_loss"].append(test_loss)
    for prefix, metrics in [("val", val_metrics), ("test", test_metrics)]:
        for key in [
            "c_index",
            "c_index_source",
            "c_index_ci_low",
            "c_index_ci_high",
            "event_count",
            "sample_count",
        ]:
            log[f"{prefix}_{key}"].append(None if metrics is None else metrics[key])


def evaluate_test_once(
    args,
    device,
    model,
    test_dataset,
    test_dataloader,
    criterion,
    process_pipeline,
):
    if test_dataset.is_None_Dataset():
        return None, None
    load_best_or_last_checkpoint(args, model, device)
    test_loss, test_metrics, records = survival_val_loop(
        device, model, test_dataloader, criterion, return_predictions=True
    )
    test_metrics = add_bootstrap_ci(args, test_metrics, records)
    export_patient_predictions(args, records, "test")
    print("Final Test_Metrics:", test_metrics)
    return test_loss, test_metrics


def load_best_or_last_checkpoint(args, model, device):
    best_paths = sorted(glob.glob(os.path.join(args.Logs.now_log_dir, "Best*.pth")))
    last_paths = sorted(glob.glob(os.path.join(args.Logs.now_log_dir, "Last*.pth")))
    checkpoint_path = best_paths[-1] if best_paths else (last_paths[-1] if last_paths else None)
    if checkpoint_path is None:
        print("No saved checkpoint found; evaluating current model state.")
        return
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint for final test: {checkpoint_path}")


def add_bootstrap_ci(args, metrics, records):
    if metrics is None:
        return metrics
    n_bootstraps = int(_survival_option(args, "bootstrap_n", 1000))
    confidence = float(_survival_option(args, "bootstrap_confidence", 0.95))
    ci_low, ci_high = bootstrap_c_index(
        [record["time"] for record in records],
        [record["event"] for record in records],
        [record["risk"] for record in records],
        n_bootstraps=n_bootstraps,
        confidence=confidence,
        seed=int(args.General.seed),
    )
    metrics["c_index_ci_low"] = ci_low
    metrics["c_index_ci_high"] = ci_high
    return metrics


def export_patient_predictions(args, records, split):
    if not records:
        return
    path = os.path.join(
        args.Logs.now_log_dir,
        f"{split}_patient_predictions_seed{args.General.seed}_"
        f"{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv",
    )
    pd.DataFrame(records).to_csv(path, index=False)
    print(f"Patient-level predictions saved: {path}")


def attach_final_test_result(log, best_epoch, test_loss, test_metrics):
    if test_metrics is None or not log["epoch"]:
        return
    index = max(min(best_epoch - 1, len(log["epoch"]) - 1), 0)
    log["test_loss"][index] = test_loss
    for key, value in test_metrics.items():
        log_key = "test_" + key
        if log_key in log:
            log[log_key][index] = value


def save_survival_log(args, epoch_info_log, best_epoch):
    log_df = pd.DataFrame(epoch_info_log)
    log_path = os.path.join(
        args.Logs.now_log_dir,
        f"Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv",
    )
    log_df.to_csv(log_path, index=False)
    print("Survival Global Log CSV Saved!")
    best_df = log_df[log_df["epoch"] == best_epoch]
    best_path = os.path.join(
        args.Logs.now_log_dir,
        f"Best_Log_seed{args.General.seed}_{args.Dataset.DATASET_NAME}_{args.General.MODEL_NAME}.csv",
    )
    best_df.to_csv(best_path, index=False)
    print("Survival Best Log CSV Saved!")
    finish_survival_tracker(args, epoch_info_log, best_epoch)


def finish_survival_tracker(args, epoch_info_log, best_epoch):
    from utils import wandb_utils

    tracker = wandb_utils.active_training_tracker()
    if tracker is None:
        return
    if tracker.enabled and epoch_info_log["epoch"]:
        best_index = max(
            min(best_epoch - 1, len(epoch_info_log["epoch"]) - 1),
            0,
        )
        tracker.summary(
            {
                "val/best_c_index": epoch_info_log["val_c_index"][best_index],
                "train/best_epoch": int(best_epoch),
                "train/stop_epoch": int(epoch_info_log["epoch"][-1]),
                "provenance/process_pipeline": args.General.process_pipeline,
            }
        )
    tracker.finish()
    wandb_utils._ACTIVE_TRACKER = None


def survival_model_select(
    reverse,
    args,
    model_state_dict,
    val_metrics,
    best_model_metric,
    best_val_metric,
    epoch,
    best_epoch,
):
    if val_metrics is None:
        return best_val_metric, best_epoch
    metric_name = best_model_metric.replace("val_", "")
    current = val_metrics[metric_name] if metric_name in val_metrics else None
    if current is None or np.isnan(current):
        return best_val_metric, best_epoch
    if reverse and current < best_val_metric:
        best_epoch = epoch + 1
        best_val_metric = current
        save_best_model(args, model_state_dict, best_epoch)
    elif not reverse and current > best_val_metric:
        best_epoch = epoch + 1
        best_val_metric = current
        save_best_model(args, model_state_dict, best_epoch)
    return best_val_metric, best_epoch


def survival_early_stop(args, epoch_info_log, process_pipeline, epoch):
    if process_pipeline == "Train_Test":
        return False
    if args.General.earlystop.use is None or args.General.earlystop.use is False:
        return False
    patience = int(args.General.earlystop.patience)
    if epoch_info_log["epoch"][-1] <= patience:
        return False
    judge_metric = args.General.earlystop.metric
    if not str(judge_metric).startswith("val_"):
        judge_metric = "val_" + str(judge_metric)
    values = np.asarray(epoch_info_log[judge_metric], dtype=float)
    if judge_metric == "val_loss":
        values = -values
    min_delta = float(
        args.General.earlystop.min_delta if "min_delta" in args.General.earlystop else 0.0
    )
    previous_best = np.nanmax(values[:-patience])
    recent_best = np.nanmax(values[-patience:])
    return recent_best <= previous_best + min_delta


def _survival_cfg(args):
    if "survival" not in args.Model:
        args.Model.survival = Dict()
    return args.Model.survival


def _survival_option(args, name, default):
    survival_cfg = _survival_cfg(args)
    return survival_cfg[name] if name in survival_cfg else default
