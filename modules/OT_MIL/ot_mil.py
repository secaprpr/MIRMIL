import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)


class OT_MIL(nn.Module):
    """Minimal sufficient submeasure learning with unbalanced OT."""

    def __init__(
        self,
        in_dim=1024,
        hidden_dim=256,
        num_classes=2,
        num_prototypes=16,
        dropout=0.1,
        sinkhorn_iterations=20,
        epsilon=0.1,
        tau_source=0.5,
        tau_target=0.5,
        gate_temperature=0.1,
        max_instances=4096,
        necessity_weight=0.5,
        minimality_weight=0.05,
        diversity_weight=0.01,
        full_classification_weight=0.5,
        consistency_weight=0.1,
        necessity_margin=1.0,
    ):
        super().__init__()
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be positive")
        if epsilon <= 0 or tau_source <= 0 or tau_target <= 0:
            raise ValueError("OT regularization parameters must be positive")

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.tau_source = tau_source
        self.tau_target = tau_target
        self.gate_temperature = gate_temperature
        self.max_instances = max_instances
        self.necessity_weight = necessity_weight
        self.minimality_weight = minimality_weight
        self.diversity_weight = diversity_weight
        self.full_classification_weight = full_classification_weight
        self.consistency_weight = consistency_weight
        self.necessity_margin = necessity_margin
        self.regularization_progress = 1.0

        self.projector = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.prototypes = nn.Parameter(torch.empty(num_prototypes, hidden_dim))
        nn.init.normal_(self.prototypes, std=0.02)
        self.prototype_logits = nn.Parameter(torch.zeros(num_prototypes))
        self.selection_threshold = nn.Parameter(torch.tensor(0.0))

        representation_dim = num_prototypes * hidden_dim + num_prototypes
        self.classifier = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.apply(_init_weights)

    def _sample_instances(self, x):
        num_instances = x.size(0)
        if self.max_instances <= 0 or num_instances <= self.max_instances:
            indices = torch.arange(num_instances, device=x.device)
            return x, indices
        if self.training:
            indices = torch.randperm(num_instances, device=x.device)[: self.max_instances]
            indices = indices.sort().values
        else:
            indices = torch.linspace(
                0, num_instances - 1, self.max_instances, device=x.device
            ).long()
        return x.index_select(0, indices), indices

    def _cost_matrix(self, features):
        features = F.normalize(features, dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        return 1.0 - features @ prototypes.transpose(0, 1)

    def _unbalanced_sinkhorn(self, cost):
        num_instances = cost.size(0)
        dtype = cost.dtype
        device = cost.device
        tiny = torch.finfo(dtype).eps

        source = torch.full(
            (num_instances,), 1.0 / num_instances, dtype=dtype, device=device
        )
        target = F.softmax(self.prototype_logits, dim=0).to(dtype=dtype)
        log_kernel = -cost / self.epsilon
        log_source = source.clamp_min(tiny).log()
        log_target = target.clamp_min(tiny).log()
        source_power = self.tau_source / (self.tau_source + self.epsilon)
        target_power = self.tau_target / (self.tau_target + self.epsilon)

        log_u = torch.zeros_like(source)
        log_v = torch.zeros_like(target)
        for _ in range(self.sinkhorn_iterations):
            log_u = source_power * (
                log_source - torch.logsumexp(log_kernel + log_v[None, :], dim=1)
            )
            log_v = target_power * (
                log_target
                - torch.logsumexp(log_kernel + log_u[:, None], dim=0)
            )
        return torch.exp(log_u[:, None] + log_kernel + log_v[None, :])

    def _selection_gate(self, row_mass):
        # Standardized log-mass removes UOT's global scaling ambiguity while
        # retaining relative evidence concentration across patches.
        log_mass = row_mass.clamp_min(torch.finfo(row_mass.dtype).eps).log()
        evidence_score = (log_mass - log_mass.mean()) / log_mass.std(
            unbiased=False
        ).clamp_min(1e-4)
        return torch.sigmoid(
            (evidence_score - self.selection_threshold)
            / max(self.gate_temperature, 1e-6)
        )

    def _build_representation(self, features, conditional_plan, weights):
        tiny = torch.finfo(features.dtype).eps
        weighted_plan = conditional_plan * weights[:, None]
        prototype_mass = weighted_plan.sum(dim=0)
        barycenters = (
            weighted_plan.transpose(0, 1) @ features
        ) / prototype_mass[:, None].clamp_min(tiny)
        normalized_mass = prototype_mass / prototype_mass.sum().clamp_min(tiny)
        return torch.cat((barycenters.flatten(), normalized_mass), dim=0).unsqueeze(0)

    def forward(self, x, return_WSI_attn=False, return_WSI_feature=False):
        if x.dim() == 3:
            if x.size(0) != 1:
                raise ValueError("OT_MIL currently expects batch_size=1")
            x = x.squeeze(0)
        if x.dim() != 2 or x.size(0) == 0:
            raise ValueError("Expected a non-empty [num_instances, feature_dim] bag")

        x, sampled_indices = self._sample_instances(x)
        features = self.projector(x)
        cost = self._cost_matrix(features)
        transport = self._unbalanced_sinkhorn(cost)
        row_mass = transport.sum(dim=1)
        conditional_plan = transport / row_mass[:, None].clamp_min(
            torch.finfo(transport.dtype).eps
        )
        gate = self._selection_gate(row_mass)

        selected_repr = self._build_representation(features, conditional_plan, gate)
        complement_repr = self._build_representation(
            features, conditional_plan, 1.0 - gate
        )
        full_repr = self._build_representation(
            features, conditional_plan, torch.ones_like(gate)
        )
        logits = self.classifier(selected_repr)
        complement_logits = self.classifier(complement_repr)
        full_logits = self.classifier(full_repr)

        result = {
            "logits": logits,
            "complement_logits": complement_logits,
            "full_logits": full_logits,
            "transport": transport,
            "selection_gate": gate,
            "selected_ratio": gate.mean(),
            "selection_threshold": self.selection_threshold,
            "prototype_usage": transport.sum(dim=0),
            "sampled_indices": sampled_indices,
        }
        if return_WSI_feature:
            result["WSI_feature"] = selected_repr
        if return_WSI_attn:
            result["WSI_attn"] = gate.unsqueeze(-1)
        return result

    def set_regularization_progress(self, progress):
        self.regularization_progress = float(min(max(progress, 0.0), 1.0))

    def compute_loss(self, output, labels, criterion=None):
        if criterion is None:
            criterion = F.cross_entropy
        classification = criterion(output["logits"], labels)
        full_classification = criterion(output["full_logits"], labels)
        consistency = F.kl_div(
            F.log_softmax(output["logits"], dim=-1),
            F.softmax(output["full_logits"].detach(), dim=-1),
            reduction="batchmean",
        )

        true_class = labels.view(-1, 1)
        selected_score = output["logits"].gather(1, true_class).squeeze(1)
        complement_score = output["complement_logits"].gather(
            1, true_class
        ).squeeze(1)
        necessity = F.relu(
            self.necessity_margin - selected_score + complement_score
        ).mean()
        minimality = output["selected_ratio"]

        normalized_prototypes = F.normalize(self.prototypes, dim=-1)
        similarity = normalized_prototypes @ normalized_prototypes.transpose(0, 1)
        off_diagonal = similarity - torch.eye(
            self.num_prototypes, device=similarity.device, dtype=similarity.dtype
        )
        diversity = off_diagonal.square().sum() / max(
            self.num_prototypes * (self.num_prototypes - 1), 1
        )

        regularization_progress = self.regularization_progress
        total = (
            classification
            + self.full_classification_weight * full_classification
            + self.consistency_weight * consistency
            + regularization_progress * self.necessity_weight * necessity
            + regularization_progress * self.minimality_weight * minimality
            + self.diversity_weight * diversity
        )
        return {
            "loss": total,
            "classification_loss": classification.detach(),
            "full_classification_loss": full_classification.detach(),
            "consistency_loss": consistency.detach(),
            "necessity_loss": necessity.detach(),
            "minimality_loss": minimality.detach(),
            "diversity_loss": diversity.detach(),
        }
