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
        prototype_rank_dim=0,
        dropout=0.1,
        sinkhorn_iterations=20,
        epsilon=0.1,
        tau_source=0.5,
        tau_target=0.5,
        gate_temperature=0.1,
        selection_fraction=0.0,
        max_instances=4096,
        necessity_weight=0.5,
        minimality_weight=0.05,
        diversity_weight=0.01,
        full_classification_weight=0.5,
        consistency_weight=0.1,
        necessity_margin=1.0,
        instance_evidence_weight=0.0,
        instance_evidence_temperature=1.0,
        rare_instance_weight=0.0,
        rare_instance_topk=32,
        rare_instance_temperature=1.0,
        rare_gate_weight=0.0,
        mass_faithful_transport=False,
        learned_evidence_gate=False,
        evidence_gate_weight=1.0,
        class_conditional_gate=False,
        residual_evidence_logits=False,
        binary_likelihood_ratio=False,
        binary_common_gate_weight=0.0,
        binary_common_gate_penalty_weight=0.0,
        binary_common_gate_balance_power=0.0,
        binary_common_gate_learnable_scale=False,
        necessity_log_probability=False,
        complement_uniformity_weight=0.0,
    ):
        super().__init__()
        if num_prototypes < 1:
            raise ValueError("num_prototypes must be positive")
        if prototype_rank_dim < 0:
            raise ValueError("prototype_rank_dim must be non-negative")
        if epsilon <= 0 or tau_source <= 0 or tau_target <= 0:
            raise ValueError("OT regularization parameters must be positive")
        if not 0.0 <= selection_fraction < 1.0:
            raise ValueError("selection_fraction must be in [0, 1)")
        if instance_evidence_weight < 0 or instance_evidence_temperature <= 0:
            raise ValueError("Instance evidence parameters must be positive")
        if (
            rare_instance_weight < 0
            or rare_instance_topk < 1
            or rare_instance_temperature <= 0
            or not 0.0 <= rare_gate_weight <= 1.0
        ):
            raise ValueError("Rare-instance parameters are invalid")
        if (rare_instance_weight > 0 or rare_gate_weight > 0) and num_classes != 2:
            raise ValueError("Rare-instance evidence currently requires two classes")
        if (
            evidence_gate_weight < 0
            or binary_common_gate_weight < 0
            or binary_common_gate_penalty_weight < 0
            or binary_common_gate_balance_power < 0
            or complement_uniformity_weight < 0
        ):
            raise ValueError("Evidence-gate and complement weights must be non-negative")
        if class_conditional_gate and not learned_evidence_gate:
            raise ValueError("Class-conditional gating requires a learned evidence gate")
        if class_conditional_gate and (
            rare_instance_weight > 0 or rare_gate_weight > 0
        ):
            raise ValueError(
                "Class-conditional and binary rare-instance gates cannot be combined"
            )
        if residual_evidence_logits and (
            instance_evidence_weight > 0 or rare_instance_weight > 0
        ):
            raise ValueError(
                "Residual evidence logits cannot be combined with auxiliary "
                "instance-logit branches"
            )
        if binary_likelihood_ratio and num_classes != 2:
            raise ValueError("Binary likelihood-ratio evidence requires two classes")
        if binary_likelihood_ratio and not (
            class_conditional_gate and residual_evidence_logits
        ):
            raise ValueError(
                "Binary likelihood-ratio evidence requires class-conditional "
                "gating and residual logits"
            )
        if binary_common_gate_weight > 0 and not binary_likelihood_ratio:
            raise ValueError(
                "Binary common-gate evidence requires likelihood-ratio mode"
            )
        if binary_common_gate_penalty_weight > 0 and binary_common_gate_weight <= 0:
            raise ValueError(
                "Binary common-gate penalty requires common-gate evidence"
            )
        if binary_common_gate_balance_power > 0 and binary_common_gate_weight <= 0:
            raise ValueError(
                "Binary common-gate balancing requires common-gate evidence"
            )
        if binary_common_gate_learnable_scale and binary_common_gate_weight <= 0:
            raise ValueError(
                "Learnable binary common-gate scaling requires common-gate evidence"
            )

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.prototype_rank_dim = prototype_rank_dim
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.tau_source = tau_source
        self.tau_target = tau_target
        self.gate_temperature = gate_temperature
        self.selection_fraction = selection_fraction
        self.max_instances = max_instances
        self.necessity_weight = necessity_weight
        self.minimality_weight = minimality_weight
        self.diversity_weight = diversity_weight
        self.full_classification_weight = full_classification_weight
        self.consistency_weight = consistency_weight
        self.necessity_margin = necessity_margin
        self.instance_evidence_weight = instance_evidence_weight
        self.instance_evidence_temperature = instance_evidence_temperature
        self.rare_instance_weight = rare_instance_weight
        self.rare_instance_topk = rare_instance_topk
        self.rare_instance_temperature = rare_instance_temperature
        self.rare_gate_weight = rare_gate_weight
        self.mass_faithful_transport = mass_faithful_transport
        self.learned_evidence_gate = learned_evidence_gate
        self.evidence_gate_weight = evidence_gate_weight
        self.class_conditional_gate = class_conditional_gate
        self.residual_evidence_logits = residual_evidence_logits
        self.binary_likelihood_ratio = binary_likelihood_ratio
        self.binary_common_gate_weight = binary_common_gate_weight
        self.binary_common_gate_penalty_weight = (
            binary_common_gate_penalty_weight
        )
        self.binary_common_gate_balance_power = binary_common_gate_balance_power
        self.binary_common_gate_learnable_scale = (
            binary_common_gate_learnable_scale
        )
        self.necessity_log_probability = necessity_log_probability
        self.complement_uniformity_weight = complement_uniformity_weight
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
        evidence_output_dim = num_classes if class_conditional_gate else 1
        if binary_likelihood_ratio:
            evidence_output_dim = 2 if binary_common_gate_weight > 0 else 1
        self.evidence_scorer = (
            nn.Linear(hidden_dim, evidence_output_dim)
            if learned_evidence_gate
            else None
        )
        self.binary_common_gate_logit = (
            nn.Parameter(torch.tensor(0.0))
            if binary_common_gate_learnable_scale
            else None
        )
        self.prototype_projector = (
            nn.Sequential(
                nn.Linear(hidden_dim, prototype_rank_dim),
                nn.LayerNorm(prototype_rank_dim),
                nn.GELU(),
            )
            if prototype_rank_dim > 0
            else None
        )

        prototype_feature_dim = prototype_rank_dim or hidden_dim
        representation_dim = (
            num_prototypes * prototype_feature_dim
            + num_prototypes
            + 2 * hidden_dim
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(representation_dim),
            nn.Linear(representation_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.evidence_residual_classifier = (
            nn.Linear(
                representation_dim,
                1 if binary_likelihood_ratio else num_classes,
            )
            if residual_evidence_logits
            else None
        )
        self.instance_classifier = (
            nn.Linear(hidden_dim, num_classes)
            if instance_evidence_weight > 0
            else None
        )
        self.rare_instance_classifier = (
            nn.Linear(hidden_dim, 1)
            if rare_instance_weight > 0 or rare_gate_weight > 0
            else None
        )
        self.apply(_init_weights)
        if self.evidence_scorer is not None:
            nn.init.zeros_(self.evidence_scorer.weight)
            nn.init.zeros_(self.evidence_scorer.bias)
        if self.evidence_residual_classifier is not None:
            if self.binary_likelihood_ratio:
                nn.init.normal_(
                    self.evidence_residual_classifier.weight, std=1e-3
                )
            else:
                nn.init.zeros_(self.evidence_residual_classifier.weight)
            nn.init.zeros_(self.evidence_residual_classifier.bias)

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

    def _selection_gate(self, row_mass, features=None):
        if self.mass_faithful_transport:
            source_mass = 1.0 / row_mass.numel()
            evidence_score = (
                row_mass.clamp_min(torch.finfo(row_mass.dtype).eps)
                / source_mass
            ).log()
            if self.evidence_scorer is not None:
                if features is None:
                    raise ValueError("features are required by the learned evidence gate")
                learned_score = self.evidence_scorer(features)
                if self.class_conditional_gate:
                    if self.binary_likelihood_ratio:
                        if self.binary_common_gate_weight > 0:
                            common_score = (
                                self.binary_common_gate_weight
                                * learned_score[:, :1]
                            )
                            log_likelihood_ratio = learned_score[:, 1:2]
                            if self.binary_common_gate_logit is not None:
                                common_score = (
                                    torch.sigmoid(self.binary_common_gate_logit)
                                    * common_score
                                )
                            if self.binary_common_gate_balance_power > 0:
                                tiny = torch.finfo(common_score.dtype).eps
                                common_magnitude = (
                                    common_score.detach().abs().mean()
                                )
                                contrast_magnitude = (
                                    log_likelihood_ratio.detach().abs().mean()
                                )
                                common_scale = (
                                    (contrast_magnitude + tiny)
                                    / (
                                        common_magnitude
                                        + contrast_magnitude
                                        + 2.0 * tiny
                                    )
                                ).pow(self.binary_common_gate_balance_power)
                                common_score = common_scale * common_score
                        else:
                            common_score = 0.0
                            log_likelihood_ratio = learned_score
                        learned_score = torch.cat(
                            (
                                common_score - log_likelihood_ratio,
                                common_score + log_likelihood_ratio,
                            ),
                            dim=1,
                        )
                    evidence_score = evidence_score[:, None] + (
                        self.evidence_gate_weight * learned_score
                    )
                else:
                    evidence_score = evidence_score + self.evidence_gate_weight * (
                        learned_score.squeeze(-1)
                    )
        else:
            # Legacy mode retains only within-bag UOT mass rank.
            log_mass = row_mass.clamp_min(torch.finfo(row_mass.dtype).eps).log()
            evidence_score = (log_mass - log_mass.mean()) / log_mass.std(
                unbiased=False
            ).clamp_min(1e-4)
        threshold = self.selection_threshold
        if self.selection_fraction > 0.0 and evidence_score.numel() > 1:
            sparse_threshold = torch.quantile(
                evidence_score.detach(),
                1.0 - self.selection_fraction,
                dim=0 if evidence_score.dim() == 2 else None,
            )
            threshold = threshold + sparse_threshold
        return torch.sigmoid(
            (evidence_score - threshold)
            / max(self.gate_temperature, 1e-6)
        )

    def _class_conditional_representations(self, features, transport, gates):
        return torch.cat(
            [
                self._build_transport_representation(
                    features, transport * gates[:, class_index, None]
                )
                for class_index in range(self.num_classes)
            ],
            dim=0,
        )

    def _class_conditional_logits(self, representations):
        class_score_matrix = self.classifier(representations)
        return class_score_matrix.diagonal().unsqueeze(0)

    def _class_conditional_instance_evidence(self, features, gates):
        return torch.stack(
            [
                self._aggregate_instance_evidence(
                    features, gates[:, class_index]
                )[0, class_index]
                for class_index in range(self.num_classes)
            ]
        ).unsqueeze(0)

    def _evidence_residual_logits(
        self, representations, full_representation, full_logits
    ):
        if self.evidence_residual_classifier is None:
            raise RuntimeError("Residual evidence classifier is disabled")
        if self.class_conditional_gate:
            residual_features = representations - full_representation.expand(
                self.num_classes, -1
            )
            if self.binary_likelihood_ratio:
                log_odds_residual = self.evidence_residual_classifier(
                    residual_features[1:2] - residual_features[0:1]
                )
                residual_logits = torch.cat(
                    (-0.5 * log_odds_residual, 0.5 * log_odds_residual),
                    dim=1,
                )
            else:
                residual_matrix = self.evidence_residual_classifier(
                    residual_features
                )
                residual_logits = residual_matrix.diagonal().unsqueeze(0)
        else:
            residual_logits = self.evidence_residual_classifier(
                representations - full_representation
            )
        return full_logits + residual_logits

    def _build_transport_representation(self, features, weighted_plan):
        tiny = torch.finfo(features.dtype).eps
        prototype_mass = weighted_plan.sum(dim=0)
        barycenters = (
            weighted_plan.transpose(0, 1) @ features
        ) / prototype_mass[:, None].clamp_min(tiny)
        normalized_mass = prototype_mass / prototype_mass.sum().clamp_min(tiny)
        instance_mass = weighted_plan.sum(dim=1)
        total_mass = instance_mass.sum().clamp_min(tiny)
        global_mean = (instance_mass[:, None] * features).sum(dim=0) / total_mass
        global_variance = (
            instance_mass[:, None] * (features - global_mean).square()
        ).sum(dim=0) / total_mass
        global_std = torch.sqrt(global_variance.clamp_min(tiny))
        prototype_features = (
            self.prototype_projector(barycenters)
            if self.prototype_projector is not None
            else barycenters
        )
        return torch.cat(
            (
                prototype_features.flatten(),
                normalized_mass,
                global_mean,
                global_std,
            ),
            dim=0,
        ).unsqueeze(0)

    def _build_representation(self, features, conditional_plan, weights):
        weighted_plan = conditional_plan * weights[:, None]
        return self._build_transport_representation(features, weighted_plan)

    def _aggregate_instance_evidence(self, features, weights):
        if self.instance_classifier is None:
            raise RuntimeError("Instance evidence branch is disabled")
        temperature = self.instance_evidence_temperature
        instance_logits = self.instance_classifier(features) / temperature
        log_weights = weights.clamp_min(
            torch.finfo(weights.dtype).eps
        ).log()
        pooled = torch.logsumexp(
            instance_logits + log_weights[:, None], dim=0
        ) - torch.logsumexp(log_weights, dim=0)
        return (temperature * pooled).unsqueeze(0)

    def _rare_instance_scores(self, features):
        if self.rare_instance_classifier is None:
            raise RuntimeError("Rare-instance evidence branch is disabled")
        return self.rare_instance_classifier(features).squeeze(-1)

    def _aggregate_rare_evidence(self, scores, weights):
        topk = min(self.rare_instance_topk, scores.numel())
        weighted_scores = scores + weights.clamp_min(
            torch.finfo(weights.dtype).eps
        ).log()
        top_scores = torch.topk(weighted_scores, topk).values
        temperature = self.rare_instance_temperature
        positive_logit = temperature * (
            torch.logsumexp(top_scores / temperature, dim=0)
            - torch.log(
                torch.tensor(
                    topk,
                    device=top_scores.device,
                    dtype=top_scores.dtype,
                )
            )
        )
        return torch.stack((-0.5 * positive_logit, 0.5 * positive_logit)).unsqueeze(0)

    def forward(
        self,
        x,
        return_WSI_attn=False,
        return_WSI_feature=False,
        return_controls=False,
    ):
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
        gate = self._selection_gate(row_mass, features)
        common_gate_energy = features.new_zeros(())
        common_gate_scale = features.new_ones(())
        if self.binary_common_gate_weight > 0:
            evidence_scores = self.evidence_scorer(features)
            common_score = (
                self.binary_common_gate_weight
                * evidence_scores[:, 0]
            )
            if self.binary_common_gate_logit is not None:
                common_gate_scale = torch.sigmoid(
                    self.binary_common_gate_logit
                )
                common_score = common_gate_scale * common_score
            if self.binary_common_gate_balance_power > 0:
                tiny = torch.finfo(common_score.dtype).eps
                common_magnitude = common_score.detach().abs().mean()
                contrast_magnitude = evidence_scores[:, 1].detach().abs().mean()
                balance_scale = (
                    (contrast_magnitude + tiny)
                    / (
                        common_magnitude
                        + contrast_magnitude
                        + 2.0 * tiny
                    )
                ).pow(self.binary_common_gate_balance_power)
                common_gate_scale = common_gate_scale * balance_scale
                common_score = balance_scale * common_score
            common_gate_energy = common_score.square().mean()
        rare_scores = None
        if self.rare_instance_classifier is not None:
            rare_scores = self._rare_instance_scores(features)
            if self.rare_gate_weight > 0:
                rare_gate = torch.sigmoid(
                    rare_scores / self.rare_instance_temperature
                )
                gate = (
                    (1.0 - self.rare_gate_weight) * gate
                    + self.rare_gate_weight * rare_gate
                )

        if self.class_conditional_gate:
            selected_repr = self._class_conditional_representations(
                features, transport, gate
            )
            complement_repr = self._class_conditional_representations(
                features, transport, 1.0 - gate
            )
            full_repr = self._build_transport_representation(features, transport)
            class_mass = (row_mass[:, None] * gate).sum(dim=0)
            selected_ratio = (
                class_mass
                / row_mass.sum().clamp_min(torch.finfo(row_mass.dtype).eps)
            ).mean()
        elif self.mass_faithful_transport:
            selected_repr = self._build_transport_representation(
                features, transport * gate[:, None]
            )
            complement_repr = self._build_transport_representation(
                features, transport * (1.0 - gate)[:, None]
            )
            full_repr = self._build_transport_representation(features, transport)
            selected_ratio = (
                row_mass * gate
            ).sum() / row_mass.sum().clamp_min(torch.finfo(row_mass.dtype).eps)
        else:
            selected_repr = self._build_representation(
                features, conditional_plan, gate
            )
            complement_repr = self._build_representation(
                features, conditional_plan, 1.0 - gate
            )
            full_repr = self._build_representation(
                features, conditional_plan, torch.ones_like(gate)
            )
            selected_ratio = gate.mean()
        full_logits = self.classifier(full_repr)
        if self.residual_evidence_logits:
            logits = self._evidence_residual_logits(
                selected_repr, full_repr, full_logits
            )
            complement_logits = self._evidence_residual_logits(
                complement_repr, full_repr, full_logits
            )
        elif self.class_conditional_gate:
            logits = self._class_conditional_logits(selected_repr)
            complement_logits = self._class_conditional_logits(complement_repr)
        else:
            logits = self.classifier(selected_repr)
            complement_logits = self.classifier(complement_repr)
        if self.instance_evidence_weight > 0:
            if self.class_conditional_gate:
                logits = logits + self.instance_evidence_weight * (
                    self._class_conditional_instance_evidence(features, gate)
                )
                complement_logits = complement_logits + (
                    self.instance_evidence_weight
                    * self._class_conditional_instance_evidence(
                        features, 1.0 - gate
                    )
                )
            else:
                logits = logits + self.instance_evidence_weight * (
                    self._aggregate_instance_evidence(features, gate)
                )
                complement_logits = complement_logits + (
                    self.instance_evidence_weight
                    * self._aggregate_instance_evidence(features, 1.0 - gate)
                )
            full_logits = full_logits + self.instance_evidence_weight * (
                self._aggregate_instance_evidence(
                    features, torch.ones_like(gate)
                )
            )
        if self.rare_instance_weight > 0:
            logits = logits + self.rare_instance_weight * (
                self._aggregate_rare_evidence(rare_scores, gate)
            )
            complement_logits = complement_logits + self.rare_instance_weight * (
                self._aggregate_rare_evidence(rare_scores, 1.0 - gate)
            )
            full_logits = full_logits + self.rare_instance_weight * (
                self._aggregate_rare_evidence(
                    rare_scores, torch.ones_like(gate)
                )
            )

        result = {
            "logits": logits,
            "complement_logits": complement_logits,
            "full_logits": full_logits,
            "transport": transport,
            "selection_gate": (
                gate.mean(dim=1) if self.class_conditional_gate else gate
            ),
            "selected_ratio": selected_ratio,
            "gate_mean": gate.mean(),
            "common_gate_energy": common_gate_energy,
            "common_gate_scale": common_gate_scale,
            "selection_threshold": self.selection_threshold,
            "prototype_usage": transport.sum(dim=0),
            "sampled_indices": sampled_indices,
        }
        if self.class_conditional_gate:
            result["class_selection_gates"] = gate
        if rare_scores is not None:
            result["rare_instance_scores"] = rare_scores
        if return_controls:
            generator = torch.Generator()
            generator.manual_seed(features.size(0))
            permutation = torch.randperm(
                features.size(0), generator=generator, device="cpu"
            ).to(features.device)
            random_gate = gate.index_select(0, permutation)
            if self.class_conditional_gate:
                random_repr = self._class_conditional_representations(
                    features, transport, random_gate
                )
            elif self.mass_faithful_transport:
                random_repr = self._build_transport_representation(
                    features, transport * random_gate[:, None]
                )
            else:
                random_repr = self._build_representation(
                    features, conditional_plan, random_gate
                )
            if self.residual_evidence_logits:
                result["random_logits"] = self._evidence_residual_logits(
                    random_repr, full_repr, full_logits
                )
            elif self.class_conditional_gate:
                result["random_logits"] = self._class_conditional_logits(
                    random_repr
                )
            else:
                result["random_logits"] = self.classifier(random_repr)
            if self.instance_evidence_weight > 0:
                if self.class_conditional_gate:
                    random_evidence = self._class_conditional_instance_evidence(
                        features, random_gate
                    )
                else:
                    random_evidence = self._aggregate_instance_evidence(
                        features, random_gate
                    )
                result["random_logits"] = result[
                    "random_logits"
                ] + self.instance_evidence_weight * random_evidence
            if self.rare_instance_weight > 0:
                result["random_logits"] = result[
                    "random_logits"
                ] + self.rare_instance_weight * (
                    self._aggregate_rare_evidence(rare_scores, random_gate)
                )
            result["random_selected_ratio"] = random_gate.mean()
        if return_WSI_feature:
            result["WSI_feature"] = selected_repr
        if return_WSI_attn:
            result["WSI_attn"] = result["selection_gate"].unsqueeze(-1)
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
        if self.necessity_log_probability:
            selected_score = F.log_softmax(output["logits"], dim=-1).gather(
                1, true_class
            ).squeeze(1)
            complement_score = F.log_softmax(
                output["complement_logits"], dim=-1
            ).gather(1, true_class).squeeze(1)
        else:
            selected_score = output["logits"].gather(
                1, true_class
            ).squeeze(1)
            complement_score = output["complement_logits"].gather(
                1, true_class
            ).squeeze(1)
        necessity = F.relu(
            self.necessity_margin - selected_score + complement_score
        ).mean()
        minimality = output["selected_ratio"]
        common_gate_energy = output["common_gate_energy"]
        complement_probabilities = F.softmax(
            output["complement_logits"], dim=-1
        )
        complement_uniformity = (
            complement_probabilities
            * (
                complement_probabilities.clamp_min(
                    torch.finfo(complement_probabilities.dtype).eps
                ).log()
                + torch.log(
                    torch.tensor(
                        self.num_classes,
                        device=complement_probabilities.device,
                        dtype=complement_probabilities.dtype,
                    )
                )
            )
        ).sum(dim=-1).mean()

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
            + regularization_progress
            * self.binary_common_gate_penalty_weight
            * common_gate_energy
            + regularization_progress
            * self.complement_uniformity_weight
            * complement_uniformity
            + self.diversity_weight * diversity
        )
        return {
            "loss": total,
            "classification_loss": classification.detach(),
            "full_classification_loss": full_classification.detach(),
            "consistency_loss": consistency.detach(),
            "necessity_loss": necessity.detach(),
            "minimality_loss": minimality.detach(),
            "common_gate_energy": common_gate_energy.detach(),
            "complement_uniformity_loss": complement_uniformity.detach(),
            "diversity_loss": diversity.detach(),
        }
