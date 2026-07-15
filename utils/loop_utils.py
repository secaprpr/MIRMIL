import torch
import time
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,roc_curve,precision_recall_fscore_support,balanced_accuracy_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, cohen_kappa_score, confusion_matrix
import time
import torch.nn as nn

def cal_scores(probs, labels, num_classes):       # probs:[batch_size, num_classes]   labels:[batch_size, ]
    probs = torch.tensor(probs)
    labels = torch.tensor(labels)
    predicted_classes = torch.argmax(probs, dim=1)
    accuracy = accuracy_score(labels.numpy(), predicted_classes.numpy())
    if num_classes > 2:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='macro', multi_class='ovr')
        micro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='micro', multi_class='ovr')
        weighted_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs.numpy(), average='weighted', multi_class='ovr')
        class_auc = {
            f"auc_class_{class_index}": roc_auc_score(
                y_true=(labels.numpy() == class_index).astype(int),
                y_score=probs[:, class_index].numpy(),
            )
            for class_index in range(num_classes)
        }
    else:
        macro_auc = roc_auc_score(y_true=labels.numpy(), y_score=probs[:,1].numpy())
        weighted_auc = micro_auc = macro_auc
        class_auc = {
            "auc_class_0": macro_auc,
            "auc_class_1": macro_auc,
        }
    min_class_auc = min(class_auc.values())
    class_1_auc = class_auc.get("auc_class_1", macro_auc)
    eps = 1e-12
    macro_auc_hmean_auc_class_1 = (
        2.0
        * macro_auc
        * class_1_auc
        / max(macro_auc + class_1_auc, eps)
    )
    weighted_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    weighted_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    weighted_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='weighted')
    macro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    macro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    macro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='macro')
    micro_f1 = f1_score(labels.numpy(), predicted_classes.numpy(), average='micro')
    micro_recall = recall_score(labels.numpy(), predicted_classes.numpy(), average='micro')
    micro_precision = precision_score(labels.numpy(), predicted_classes.numpy(), average='micro') 
    baccuracy = balanced_accuracy_score(labels.numpy(), predicted_classes.numpy())
    quadratic_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='quadratic')
    linear_kappa = cohen_kappa_score(labels.numpy(), predicted_classes.numpy(), weights='linear')
    confusion_mat = confusion_matrix(labels.numpy(), predicted_classes.numpy())
    metrics = {'acc': accuracy,  'bacc': baccuracy, 
               'macro_auc': macro_auc, 'micro_auc': micro_auc, 'weighted_auc':weighted_auc,
               'min_class_auc': min_class_auc,
               'macro_auc_hmean_auc_class_1': macro_auc_hmean_auc_class_1,
                'macro_f1': macro_f1, 'micro_f1': micro_f1, 'weighted_f1': weighted_f1, 
                 'macro_recall': macro_recall, 'micro_recall': micro_recall,'weighted_recall': weighted_recall, 
                 'macro_pre': macro_precision, 'micro_pre': micro_precision,'weighted_pre': weighted_precision,
                 'quadratic_kappa': quadratic_kappa,'linear_kappa':linear_kappa,  
                 'confusion_mat': confusion_mat}
    metrics.update(class_auc)
    return metrics

def train_loop(device,model,loader,criterion,optimizer,scheduler):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        train_logits = model(bag)['logits']
        train_loss = criterion(train_logits, label)
        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def val_loop(device,num_classes,model,loader,criterion,retrun_WSI_feature = False,return_WSI_attn=False):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            if retrun_WSI_feature:
                WSI_feature = model(bag,return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(bag,return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            val_logits = model(bag)['logits']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            
            # Handle BCE loss - convert label to one-hot if needed
            if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                import torch.nn.functional as F
                label_for_loss = F.one_hot(label.long(), num_classes=num_classes).float()
            else:
                label_for_loss = label
            
            val_loss = criterion(val_logits, label_for_loss)
            val_loss_log += val_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics


def _ranking_memory_scores(logits, score_type):
    if score_type == "logit":
        return logits
    if score_type != "ovr_log_odds":
        raise ValueError(
            "ranking_memory_score_type must be 'logit' or 'ovr_log_odds'"
        )
    if logits.shape[-1] < 2:
        raise ValueError("ovr_log_odds ranking requires at least two classes")
    scores = []
    for class_index in range(logits.shape[-1]):
        other_logits = torch.cat(
            (logits[..., :class_index], logits[..., class_index + 1:]),
            dim=-1,
        )
        scores.append(
            logits[..., class_index]
            - torch.logsumexp(other_logits, dim=-1)
        )
    return torch.stack(scores, dim=-1)


def mir_train_loop(
    device,
    model,
    loader,
    criterion,
    optimizer,
    scheduler,
    distillation_targets=None,
    distillation_weight=0.0,
    distillation_temperature=1.0,
    distillation_min_entropy=0.0,
    distillation_max_confidence=1.0,
    distillation_entropy_weight_power=0.0,
    ranking_memory=None,
    ranking_memory_labels=None,
    ranking_memory_valid=None,
    ranking_memory_weight=0.0,
    ranking_memory_margin=0.1,
    ranking_memory_momentum=0.9,
    ranking_memory_max_pairs=64,
    ranking_memory_class_indices=None,
    ranking_memory_hard_mining=False,
    ranking_memory_apply_loss=True,
    ranking_memory_score_type="logit",
):
    if distillation_weight < 0:
        raise ValueError("distillation_weight must be non-negative")
    if distillation_temperature <= 0:
        raise ValueError("distillation_temperature must be positive")
    if distillation_min_entropy < 0:
        raise ValueError("distillation_min_entropy must be non-negative")
    if not 0 < distillation_max_confidence <= 1:
        raise ValueError("distillation_max_confidence must be in (0, 1]")
    if distillation_entropy_weight_power < 0:
        raise ValueError(
            "distillation_entropy_weight_power must be non-negative"
        )
    if ranking_memory_weight < 0:
        raise ValueError("ranking_memory_weight must be non-negative")
    if ranking_memory_margin < 0:
        raise ValueError("ranking_memory_margin must be non-negative")
    if not 0 <= ranking_memory_momentum < 1:
        raise ValueError("ranking_memory_momentum must be in [0, 1)")
    if ranking_memory_max_pairs <= 0:
        raise ValueError("ranking_memory_max_pairs must be positive")
    if ranking_memory_score_type not in {"logit", "ovr_log_odds"}:
        raise ValueError(
            "ranking_memory_score_type must be 'logit' or 'ovr_log_odds'"
        )
    if ranking_memory_class_indices is not None:
        ranking_memory_class_indices = [
            int(index) for index in ranking_memory_class_indices
        ]
        if not ranking_memory_class_indices:
            raise ValueError("ranking_memory_class_indices must not be empty")
        if len(set(ranking_memory_class_indices)) != len(
            ranking_memory_class_indices
        ):
            raise ValueError("ranking_memory_class_indices must be unique")
        if min(ranking_memory_class_indices) < 0:
            raise ValueError("ranking_memory_class_indices must be non-negative")
    start = time.time()
    model.train()
    totals = {
        "loss": 0.0,
        "classification_loss": 0.0,
        "distillation_loss": 0.0,
        "logit_margin_loss": 0.0,
        "ordinal_loss": 0.0,
        "ovr_loss": 0.0,
        "adjacent_loss": 0.0,
        "focus_class_loss": 0.0,
        "focus_sparse_loss": 0.0,
        "stability_loss": 0.0,
        "subset_consistency_loss": 0.0,
        "subset_supervised_loss": 0.0,
        "lipschitz_loss": 0.0,
        "prototype_loss": 0.0,
        "ranking_memory_loss": 0.0,
    }
    for batch in loader:
        if len(batch) == 3:
            bag, label, sample_index = batch
        else:
            bag, label = batch
            sample_index = None
        optimizer.zero_grad()
        bag = bag.float().to(device)
        label = label.long().to(device)
        output, losses = model.compute_loss(bag, label, criterion)
        loss = losses["loss"]
        if ranking_memory_weight > 0:
            if (
                ranking_memory is None
                or ranking_memory_labels is None
                or ranking_memory_valid is None
                or sample_index is None
            ):
                raise ValueError(
                    "ranking memory tensors and sample indices are required "
                    "when ranking_memory_weight is positive"
                )
            logits = output["logits"]
            memory = ranking_memory.to(device)
            memory_labels = ranking_memory_labels.to(device)
            memory_valid = ranking_memory_valid.to(device)
            current_ranking_scores = _ranking_memory_scores(
                logits, ranking_memory_score_type
            )
            memory_ranking_scores = _ranking_memory_scores(
                memory, ranking_memory_score_type
            )
            ranking_terms = []
            num_classes = logits.shape[1]
            if ranking_memory_class_indices is None:
                ranking_classes = range(num_classes)
            else:
                if max(ranking_memory_class_indices) >= num_classes:
                    raise ValueError(
                        "ranking_memory_class_indices contains an out-of-range class"
                    )
                ranking_classes = ranking_memory_class_indices
            batch_indices = sample_index.view(-1).to(device)
            if ranking_memory_apply_loss:
                for row, sample_label in enumerate(label.view(-1)):
                    current_scores = current_ranking_scores[row]
                    for class_index in ranking_classes:
                        current_is_positive = sample_label.item() == class_index
                        if current_is_positive:
                            pair_mask = memory_valid & (memory_labels != class_index)
                            if torch.any(pair_mask):
                                other_scores = memory_ranking_scores[
                                    pair_mask, class_index
                                ]
                                if other_scores.numel() > ranking_memory_max_pairs:
                                    if ranking_memory_hard_mining:
                                        other_scores = torch.topk(
                                            other_scores,
                                            ranking_memory_max_pairs,
                                            largest=True,
                                        ).values
                                    else:
                                        other_scores = other_scores[
                                            -ranking_memory_max_pairs:
                                        ]
                                ranking_terms.append(
                                    F.softplus(
                                        ranking_memory_margin
                                        - (
                                            current_scores[class_index]
                                            - other_scores
                                        )
                                    ).mean()
                                )
                        else:
                            pair_mask = memory_valid & (memory_labels == class_index)
                            if torch.any(pair_mask):
                                other_scores = memory_ranking_scores[
                                    pair_mask, class_index
                                ]
                                if other_scores.numel() > ranking_memory_max_pairs:
                                    if ranking_memory_hard_mining:
                                        other_scores = torch.topk(
                                            other_scores,
                                            ranking_memory_max_pairs,
                                            largest=False,
                                        ).values
                                    else:
                                        other_scores = other_scores[
                                            -ranking_memory_max_pairs:
                                        ]
                                ranking_terms.append(
                                    F.softplus(
                                        ranking_memory_margin
                                        - (
                                            other_scores
                                            - current_scores[class_index]
                                        )
                                    ).mean()
                                )
            if ranking_memory_apply_loss and ranking_terms:
                ranking_memory_loss = torch.stack(ranking_terms).mean()
            else:
                ranking_memory_loss = logits.sum() * 0.0
            loss = loss + ranking_memory_weight * ranking_memory_loss
            losses["ranking_memory_loss"] = ranking_memory_loss
            losses["loss"] = loss
        if distillation_weight > 0:
            if distillation_targets is None or sample_index is None:
                raise ValueError(
                    "distillation_targets and sample indices are required "
                    "when distillation_weight is positive"
                )
            teacher = distillation_targets[sample_index.view(-1)].to(device)
            teacher = teacher.clamp_min(1e-8)
            teacher = teacher / teacher.sum(dim=1, keepdim=True)
            if distillation_temperature != 1.0:
                teacher = torch.softmax(
                    torch.log(teacher) / distillation_temperature,
                    dim=1,
                )
            confidence = teacher.max(dim=1).values
            entropy = -(
                teacher * torch.log(teacher.clamp_min(1e-8))
            ).sum(dim=1) / torch.log(
                torch.tensor(
                    teacher.shape[1],
                    dtype=teacher.dtype,
                    device=teacher.device,
                )
            )
            distillation_mask = (
                (entropy >= distillation_min_entropy)
                & (confidence <= distillation_max_confidence)
            )
            student_log_probs = F.log_softmax(
                output["logits"] / distillation_temperature,
                dim=1,
            )
            if torch.any(distillation_mask):
                per_sample_loss = F.kl_div(
                    student_log_probs[distillation_mask],
                    teacher[distillation_mask],
                    reduction="none",
                ).sum(dim=1) * (distillation_temperature ** 2)
                if distillation_entropy_weight_power > 0:
                    weights = entropy[distillation_mask].clamp_min(
                        1e-8
                    ).pow(distillation_entropy_weight_power)
                    distillation_loss = (
                        per_sample_loss * weights
                    ).sum() / weights.sum().clamp_min(1e-8)
                else:
                    distillation_loss = per_sample_loss.mean()
            else:
                distillation_loss = output["logits"].sum() * 0.0
            loss = loss + distillation_weight * distillation_loss
            losses["distillation_loss"] = distillation_loss
            losses["loss"] = loss
        loss.backward()
        optimizer.step()
        if ranking_memory_weight > 0:
            with torch.no_grad():
                batch_indices_cpu = sample_index.view(-1).cpu().long()
                logits_cpu = output["logits"].detach().cpu()
                previous_valid = ranking_memory_valid[batch_indices_cpu]
                ranking_memory[batch_indices_cpu[~previous_valid]] = logits_cpu[
                    ~previous_valid
                ]
                if torch.any(previous_valid):
                    valid_indices = batch_indices_cpu[previous_valid]
                    ranking_memory[valid_indices].mul_(ranking_memory_momentum).add_(
                        logits_cpu[previous_valid],
                        alpha=1.0 - ranking_memory_momentum,
                    )
                ranking_memory_valid[batch_indices_cpu] = True
        for key in totals:
            if key in losses:
                totals[key] += float(losses[key].detach().item())
    if scheduler is not None:
        scheduler.step()
    count = max(len(loader), 1)
    components = {
        key: value / count for key, value in totals.items() if key != "loss"
    }
    return totals["loss"] / count, time.time() - start, components


def mir_val_loop(device, num_classes, model, loader, criterion):
    model.eval()
    losses = 0.0
    labels = []
    probabilities = []
    with torch.no_grad():
        for bag, label in loader:
            bag = bag.float().to(device)
            label = label.long().to(device)
            logits = model(bag)["logits"]
            losses += float(criterion(logits, label).item())
            labels.append(label.cpu().numpy())
            probabilities.append(
                torch.softmax(logits.squeeze(0), dim=0).cpu().numpy()
            )
    metrics = cal_scores(probabilities, labels, num_classes)
    return losses / max(len(loader), 1), metrics


def ot_train_loop(device, model, loader, criterion, optimizer, scheduler):
    start = time.time()
    model.train()
    loss_log = 0.0
    component_log = {
        "classification_loss": 0.0,
        "endpoint_classification_loss": 0.0,
        "class_mass_classification_loss": 0.0,
        "full_classification_loss": 0.0,
        "consistency_loss": 0.0,
        "necessity_loss": 0.0,
        "minimality_loss": 0.0,
        "common_gate_energy": 0.0,
        "class_prototype_separation_loss": 0.0,
        "class_prototype_information_loss": 0.0,
        "complement_uniformity_loss": 0.0,
        "diversity_loss": 0.0,
    }
    for data in loader:
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].float().to(device)
        output = model(bag)
        losses = model.compute_loss(output, label, criterion)
        losses["loss"].backward()
        optimizer.step()

        loss_log += losses["loss"].item()
        for key in component_log:
            component_log[key] += losses[key].item()

    if scheduler is not None:
        scheduler.step()
    num_batches = max(len(loader), 1)
    loss_log /= num_batches
    component_log = {key: value / num_batches for key, value in component_log.items()}
    return loss_log, time.time() - start, component_log


def ot_val_loop(
    device,
    num_classes,
    model,
    loader,
    criterion,
    retrun_WSI_feature=False,
    return_WSI_attn=False,
    return_diagnostics=False,
):
    model.eval()
    loss_log = 0.0
    labels = []
    probabilities = []
    WSI_features = []
    WSI_attns = []
    selected_ratios = []
    complement_probabilities = []
    full_probabilities = []
    random_probabilities = []
    with torch.no_grad():
        for data in loader:
            label = data[1].long().to(device)
            bag = data[0].float().to(device)
            output = model(
                bag,
                return_WSI_feature=retrun_WSI_feature,
                return_WSI_attn=return_WSI_attn,
                return_controls=return_diagnostics,
            )
            if retrun_WSI_feature:
                WSI_features.append(output["WSI_feature"])
                continue
            if return_WSI_attn:
                WSI_attns.append(output["WSI_attn"].cpu())
                continue

            losses = model.compute_loss(output, label, criterion)
            loss_log += losses["loss"].item()
            labels.append(label.cpu().numpy())
            probabilities.append(
                torch.softmax(output["logits"].squeeze(0), dim=0).cpu().numpy()
            )
            complement_probabilities.append(
                torch.softmax(output["complement_logits"].squeeze(0), dim=0)
                .cpu()
                .numpy()
            )
            full_probabilities.append(
                torch.softmax(output["full_logits"].squeeze(0), dim=0).cpu().numpy()
            )
            selected_ratios.append(output["selected_ratio"].item())
            if return_diagnostics:
                random_probabilities.append(
                    torch.softmax(output["random_logits"].squeeze(0), dim=0)
                    .cpu()
                    .numpy()
                )

    if retrun_WSI_feature:
        return torch.cat(WSI_features, dim=0).cpu().numpy()
    if return_WSI_attn:
        return WSI_attns
    loss_log /= max(len(loader), 1)
    selected_metrics = cal_scores(probabilities, labels, num_classes)
    if not return_diagnostics:
        return loss_log, selected_metrics
    complement_metrics = cal_scores(complement_probabilities, labels, num_classes)
    full_metrics = cal_scores(full_probabilities, labels, num_classes)
    random_metrics = cal_scores(random_probabilities, labels, num_classes)
    label_indices = torch.tensor(labels).view(-1).long()
    selected_true = torch.tensor(probabilities).gather(
        1, label_indices[:, None]
    ).squeeze(1)
    complement_true = torch.tensor(complement_probabilities).gather(
        1, label_indices[:, None]
    ).squeeze(1)
    random_true = torch.tensor(random_probabilities).gather(
        1, label_indices[:, None]
    ).squeeze(1)
    diagnostics = {
        "selected_ratio_mean": float(torch.tensor(selected_ratios).mean()),
        "selected_ratio_std": float(torch.tensor(selected_ratios).std(unbiased=False)),
        "full_macro_auc": full_metrics["macro_auc"],
        "complement_macro_auc": complement_metrics["macro_auc"],
        "random_macro_auc": random_metrics["macro_auc"],
        "full_acc": full_metrics["acc"],
        "complement_acc": complement_metrics["acc"],
        "random_acc": random_metrics["acc"],
        "necessity_confidence_drop": float(
            (selected_true - complement_true).mean()
        ),
        "selection_vs_random_confidence_gain": float(
            (selected_true - random_true).mean()
        ),
    }
    return loss_log, selected_metrics, diagnostics

def ac_train_loop(device,model,loader,criterion,optimizer,scheduler,n_token):
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag)
        train_logits = forward_return['logits']
        sub_preds = forward_return['sub_preds']
        attns = forward_return['attns']
        if n_token > 1:
            loss0 = criterion(sub_preds, label.repeat_interleave(n_token))
        else:
            loss0 = torch.tensor(0.)
        diff_loss = torch.tensor(0).to(device, dtype=torch.float)
        attns = torch.softmax(attns, dim=-1)

        for i in range(n_token):
            for j in range(i + 1, n_token): 
                diff_loss += torch.cosine_similarity(attns[:, i], attns[:, j], dim=-1).mean() / (
                            n_token * (n_token - 1) / 2)
        train_loss = criterion(train_logits, label)
        train_loss = diff_loss + loss0 + train_loss
        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def ac_val_loop(device,num_classes,model,loader,criterion,n_token,retrun_WSI_feature = False,return_WSI_attn=False):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].long().to(device)
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            forward_return = model(bag)
            val_logits = forward_return['logits']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            sub_preds = forward_return['sub_preds']
            attns = forward_return['attns']
            if n_token > 1:
                loss0 = criterion(sub_preds, label.repeat_interleave(n_token))
            else:
                loss0 = torch.tensor(0.)
            diff_loss = torch.tensor(0).to(device, dtype=torch.float)
            attns = torch.softmax(attns, dim=-1)
            for i in range(n_token):
                for j in range(i + 1, n_token): 
                    diff_loss += torch.cosine_similarity(attns[:, i], attns[:, j], dim=-1).mean() / (
                                n_token * (n_token - 1) / 2)
            val_loss = criterion(val_logits, label)
            val_loss = diff_loss + loss0 + val_loss
            val_loss_log += val_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics


# clam has instance-loss defferent from other mil models
def clam_train_loop(device,model,loader,criterion,optimizer,scheduler,bag_weight):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag,label=label)
        instance_loss = forward_return['instance_loss']
        train_logits = forward_return['logits']
        train_loss = criterion(train_logits, label)
        total_loss = train_loss * bag_weight + instance_loss * (1-bag_weight)
        train_loss_log += total_loss.item()
        total_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time


def tripleloss(golabal,p_center,nc_center):
    golabal = golabal.squeeze(0)
    n_globallesionrepresente, _ = golabal.shape
    p_center = p_center.repeat(n_globallesionrepresente, 1)
    nc_center = nc_center.repeat(n_globallesionrepresente, 1)
    triple_loss = nn.TripletMarginWithDistanceLoss(distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y) ,margin=1)
    loss = triple_loss(golabal,p_center,nc_center)
    return loss

def dgr_train_loop(device,model,loader,criterion,optimizer,scheduler,now_epoch,epoch_des,n_lesion):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        if torch.argmax(label)==0:
            forward_return = model(bag,bag_mode='normal')
            train_logits, A,H,p_center,nc_center,lesion = forward_return['logits'],forward_return['A'],forward_return['H'],forward_return['postivecenter'],forward_return['normalcenter'],forward_return['lesion_enhacing']
        else:
            train_logits, A,H,p_center,nc_center,lesion= model(bag,bag_mode='abnormal')
        if now_epoch < epoch_des:
            train_loss = criterion(train_logits, label)
        else:
            train_loss = criterion(train_logits, label)
            lesion_norm = lesion.squeeze(0)
            lesion_norm = torch.nn.functional.normalize(lesion_norm)
            div_loss = -torch.logdet(lesion_norm@lesion_norm.T+1e-10*torch.eye(n_lesion).to(device))
            sim_loss = tripleloss(lesion,p_center,nc_center)
            train_loss = train_loss + 0.1*div_loss + 0.1*sim_loss 

        train_loss_log += train_loss.item()
        train_loss.backward()
        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time

def clam_val_loop(device,num_classes,model,loader,criterion,bag_weight,retrun_WSI_feature = False,return_WSI_attn=False):
    model.eval()
    val_loss_log = 0
    labels = []
    bag_predictions_after_normal = []
    model = model.to(device)
    WSI_features = []
    WSI_attns = []
    with torch.no_grad():
        for i, data in enumerate(loader):
            label = data[1].to(device).long()
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            if retrun_WSI_feature:
                WSI_feature = model(bag,label = label, return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(bag,label = label, return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            forward_return = model(bag,label=label)
            instance_loss = forward_return['instance_loss']
            val_logits = forward_return['logits']
            val_logits = val_logits.squeeze(0)
            bag_predictions_after_normal.append(torch.softmax(val_logits,0).cpu().numpy())
            val_logits = val_logits.unsqueeze(0)
            val_loss = criterion(val_logits,label)
            total_loss = val_loss * bag_weight + instance_loss * (1-bag_weight)
            val_loss_log += total_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    val_loss_log /= len(loader)
    return val_loss_log,val_metrics

def ds_train_loop(device,model,loader,criterion,optimizer,scheduler):
    
    start = time.time()
    model.train()
    train_loss_log = 0
    model = model.to(device)
    for i, data in enumerate(loader):
        optimizer.zero_grad()
        label = data[1].long().to(device)
        bag = data[0].to(device).float()
        forward_return = model(bag)
        max_prediction = forward_return['max_prediction']
        train_logits = forward_return['logits']
        loss_bag = criterion(train_logits, label)
        loss_max = criterion(max_prediction, label)
        train_loss = 0.5*loss_bag + 0.5*loss_max
        train_loss_log += train_loss.item()

        train_loss.backward()

        optimizer.step()
    if scheduler is not None:
        scheduler.step()
    train_loss_log /= len(loader)
    end = time.time()
    total_time = end - start
    return train_loss_log,total_time

def ds_val_loop(device,num_classes,model,loader,criterion,retrun_WSI_feature = False,return_WSI_attn=False):
    WSI_features = []
    WSI_attns = []
    labels = []
    bag_predictions_after_normal = []
    val_loss_log = 0
    model.eval()
    model = model.to(device)
    with torch.autograd.set_detect_anomaly(True):
        for i, data in enumerate(loader):
            label = data[1].long().to(device)
            labels.append(label.cpu().numpy())
            bag = data[0].to(device).float()
            forward_return = model(bag)
            if retrun_WSI_feature:
                WSI_feature = model(bag,return_WSI_feature=True)['WSI_feature']
                WSI_features.append(WSI_feature)
                continue
            if return_WSI_attn:
                WSI_attn = model(bag,return_WSI_attn=True)['WSI_attn']
                WSI_attns.append(WSI_attn)
                continue
            max_prediction = forward_return['max_prediction']
            val_logits = forward_return['logits']
            bag_predictions_after_normal.append(torch.softmax(val_logits[0],0).cpu().detach().numpy())
            loss_bag = criterion(val_logits, label)
            loss_max = criterion(max_prediction, label)
            val_loss = 0.5*loss_bag + 0.5*loss_max
            val_loss_log += val_loss.item()
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().detach().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    val_loss_log /= len(loader)
    val_metrics= cal_scores(bag_predictions_after_normal,labels,num_classes)
    return val_loss_log,val_metrics

def get_cam_1d(classifier, features):
    tweight = list(classifier.parameters())[-2]
    cam_maps = torch.einsum('bgf,cf->bcg', [features, tweight])
    return cam_maps

def dtfd_train_loop(device, model_list, loader, criterion, optimizer_list, scheduler_list, num_Group, grad_clipping,distill,total_instance):
    train_loss_log = 0
    start = time.time()
    instance_per_group = total_instance // num_Group
    # Unpack model list
    classifier, attention, dimReduction, attCls = model_list
    classifier.train()
    attention.train()
    dimReduction.train()
    attCls.train()

    # Unpack optimizer and scheduler lists
    optimizer_A, optimizer_B = optimizer_list
    scheduler_A, scheduler_B = scheduler_list

    total_loss = 0
    for i, data in enumerate(loader):
        label = data[1].long().to(device)
        bag = data[0].to(device).float()

        slide_sub_preds = []
        slide_sub_labels = []
        slide_pseudo_feat = []

        # Split bag into chunks
        inputs_pseudo_bags = torch.chunk(bag.squeeze(0), num_Group, dim=0)

        for subFeat_tensor in inputs_pseudo_bags:
            slide_sub_labels.append(label)
            subFeat_tensor = subFeat_tensor.to(device)

            # Forward pass through models
            tmidFeat = dimReduction(subFeat_tensor)
            tAA = attention(tmidFeat).squeeze(0)
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x fs
            tPredict = classifier(tattFeat_tensor)  # 1 x 2
            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)
            slide_sub_preds.append(tPredict)


        # Concatenate tensors
        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        slide_sub_preds = torch.cat(slide_sub_preds, dim=0)  # numGroup x fs
        slide_sub_labels = torch.cat(slide_sub_labels, dim=0)  # numGroup

        # Calculate and backpropagate loss for the first tier
        loss_A = criterion(slide_sub_preds, slide_sub_labels)
        optimizer_A.zero_grad()
        loss_A.backward(retain_graph=True)
        total_loss += loss_A.item()

        # Clip gradients and update weights
        torch.nn.utils.clip_grad_norm_(dimReduction.parameters(), grad_clipping)
        torch.nn.utils.clip_grad_norm_(attention.parameters(), grad_clipping)
        torch.nn.utils.clip_grad_norm_(classifier.parameters(), grad_clipping)


        # Second tier optimization
        gSlidePred = attCls(slide_pseudo_feat)['logits']
        loss_B = criterion(gSlidePred, label).mean()
        optimizer_B.zero_grad()
        loss_B.backward()
        total_loss += loss_B.item()

        # Clip gradients and update weights
        torch.nn.utils.clip_grad_norm_(attCls.parameters(), grad_clipping)
        optimizer_A.step()
        optimizer_B.step()

    # Step schedulers
    scheduler_A.step()
    scheduler_B.step()

    end = time.time()
    total_loss /= len(loader)
    total_time = end - start

    return total_loss, total_time


def dtfd_val_loop(device,num_classes,model_list,loader,criterion,num_Group,grad_clipping,distill,total_instance,retrun_WSI_feature = False,return_WSI_attn=False):
    WSI_features = []
    WSI_attns = []
    instance_per_group = total_instance // num_Group
    classifier,attention,dimReduction,attCls = model_list
    classifier.eval()
    attention.eval()
    dimReduction.eval()
    attCls.eval()
    total_loss = 0
    y_score=[]
    y_true=[]
    for i, data in enumerate(loader):
        label = data[1].long().to(device)
        bag = data[0].to(device).float()

        slide_sub_preds = []
        slide_sub_labels = []
        slide_pseudo_feat = []
        inputs_pseudo_bags=torch.chunk(bag.squeeze(0), num_Group,dim=0)
        
        for subFeat_tensor in inputs_pseudo_bags:
            subFeat_tensor=subFeat_tensor.to(device)
            with torch.no_grad():
                tmidFeat = dimReduction(subFeat_tensor)
                tAA = attention(tmidFeat).squeeze(0)
                tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  # n x fs
                tattFeat_tensor = torch.sum(tattFeats, dim=0, keepdim=True)  # 1 x fs
                tPredict = classifier(tattFeat_tensor)  # 1 x 2
            tattFeats = torch.einsum('ns,n->ns', tmidFeat, tAA)  ### n x fs
            tattFeat_tensor = torch.sum(tattFeats, dim=0).unsqueeze(0)  ## 1 x fs
            patch_pred_logits = get_cam_1d(classifier, tattFeats.unsqueeze(0)).squeeze(0)  ###  cls x n
            patch_pred_logits = torch.transpose(patch_pred_logits, 0, 1)  ## n x cls
            patch_pred_softmax = torch.softmax(patch_pred_logits, dim=1)  ## n x cls

            _, sort_idx = torch.sort(patch_pred_softmax[:,-1], descending=True)
            topk_idx_max = sort_idx[:instance_per_group].long()
            topk_idx_min = sort_idx[-instance_per_group:].long()
            topk_idx = torch.cat([topk_idx_max, topk_idx_min], dim=0)
            MaxMin_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx)   
            max_inst_feat = tmidFeat.index_select(dim=0, index=topk_idx_max)
            af_inst_feat = tattFeat_tensor

            if distill == 'MaxMinS':
                slide_pseudo_feat.append(MaxMin_inst_feat)
            elif distill == 'MaxS':
                slide_pseudo_feat.append(max_inst_feat)
            elif distill == 'AFS':
                slide_pseudo_feat.append(af_inst_feat)
            slide_sub_preds.append(tPredict)

        slide_pseudo_feat = torch.cat(slide_pseudo_feat, dim=0)
        gSlidePred = torch.softmax(attCls(slide_pseudo_feat)['logits'], dim=1)
        forward_return = attCls(slide_pseudo_feat, return_WSI_attn = return_WSI_attn, return_WSI_feature = retrun_WSI_feature)
        if retrun_WSI_feature:
            WSI_feature = forward_return['WSI_feature']
            WSI_features.append(WSI_feature)
            continue
        if return_WSI_attn:
            WSI_attn = forward_return['WSI_attn']
            WSI_attns.append(WSI_attn)
            continue
        loss = criterion(forward_return['logits'], label)
        total_loss += loss.item()
        pred=(gSlidePred.cpu().data.numpy()).tolist()
        y_score.extend(pred)
        y_true.extend(label)
    if retrun_WSI_feature:
        WSI_features = torch.cat(WSI_features, dim=0).cpu().detach().numpy()
        return WSI_features
    if return_WSI_attn:
        return WSI_attns
    
    total_loss /= len(loader)
    val_metrics= cal_scores(y_score,y_true,num_classes)
    return total_loss,val_metrics


# ============================================
# Mixup Training Loop Utilities
# ============================================
import numpy as np

def train_loop_with_mixup(device, model, dataloader, criterion, optimizer, scheduler, mixup_config, mix_fn):
    """
    Generic training loop with mixup augmentation
    Args:
        device: torch device
        model: MIL model
        dataloader: training dataloader
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler
        mixup_config: dict with mixup parameters (prob, alpha, etc.)
        mix_fn: mixing function from the specific module (e.g., mixup_data, insmix_data, etc.)
    Returns:
        train_loss, cost_time
    """
    model.train()
    train_loss = 0.
    start_time = time.time()
    
    all_features = []
    all_labels = []
    
    for idx, batch in enumerate(dataloader):
        if len(batch) == 3:
            _, data, label = batch
        else:
            data, label = batch
        all_features.append(data.squeeze(0).to(device))
        all_labels.append(label.to(device))
    
    prob = mixup_config.get('prob', 0.5)
    n_samples = len(all_features)
    perm = torch.randperm(n_samples)
    
    for i in range(n_samples):
        optimizer.zero_grad()
        
        if np.random.rand() < prob:
            j = perm[i].item()
            if i != j:
                # Call the specific mix function with appropriate kwargs
                mix_kwargs = {k: v for k, v in mixup_config.items() if k != 'prob'}
                # For rankmix, we need to pass the model
                if 'model' in mix_fn.__code__.co_varnames:
                    mixed_feat, mixed_label, _ = mix_fn(
                        all_features[i], all_labels[i],
                        all_features[j], all_labels[j],
                        model, **mix_kwargs
                    )
                else:
                    mixed_feat, mixed_label, _ = mix_fn(
                        all_features[i], all_labels[i],
                        all_features[j], all_labels[j],
                        **mix_kwargs
                    )
            else:
                mixed_feat, mixed_label = all_features[i], all_labels[i]
        else:
            mixed_feat, mixed_label = all_features[i], all_labels[i]
        
        forward_return = model(mixed_feat.unsqueeze(0))
        logits = forward_return['logits']
        
        if mixed_label.dim() == 0 or mixed_label.size(0) == 1:
            if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
                label_for_loss = F.one_hot(mixed_label.long(), num_classes=logits.size(1)).float()
            else:
                label_for_loss = mixed_label.long()
        else:
            label_for_loss = mixed_label.float()
        
        if criterion.__class__.__name__ == 'BCEWithLogitsLoss':
            loss = criterion(logits, label_for_loss)
        else:
            loss = criterion(logits, label_for_loss.argmax(dim=-1) if label_for_loss.dim() > 1 else label_for_loss)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    if scheduler is not None:
        scheduler.step()
    
    train_loss /= n_samples
    cost_time = time.time() - start_time
    
    return train_loss, cost_time
