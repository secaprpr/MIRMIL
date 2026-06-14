import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name):
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    raise ValueError(f"Unsupported activation: {name}")


class MixturePrototypePotential(nn.Module):
    """Class energy from a smooth mixture of measure-state prototypes."""

    def __init__(
        self,
        state_dim,
        num_classes,
        embedding_dim,
        prototypes_per_class,
        temperature,
        mixture_temperature,
        hidden_dim,
        dropout,
        act,
        diversity_margin,
        separation_margin,
    ):
        super().__init__()
        if embedding_dim <= 0 or prototypes_per_class <= 0:
            raise ValueError(
                "embedding_dim and prototypes_per_class must be positive"
            )
        if temperature <= 0 or mixture_temperature <= 0:
            raise ValueError("prototype temperatures must be positive")
        self.num_classes = int(num_classes)
        self.prototypes_per_class = int(prototypes_per_class)
        self.temperature = float(temperature)
        self.mixture_temperature = float(mixture_temperature)
        self.diversity_margin = float(diversity_margin)
        self.separation_margin = float(separation_margin)
        self.projector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.prototypes = nn.Parameter(
            torch.empty(
                self.num_classes,
                self.prototypes_per_class,
                embedding_dim,
            )
        )
        # Keep auxiliary prototype creation from shifting shared-model RNG.
        with torch.random.fork_rng():
            nn.init.normal_(
                self.prototypes, std=1.0 / math.sqrt(embedding_dim)
            )

    def forward(self, state):
        embedding = F.normalize(self.projector(state), dim=-1)
        prototypes = F.normalize(self.prototypes, dim=-1)
        similarities = torch.einsum(
            "bd,ckd->bck", embedding, prototypes
        )
        scaled = similarities / self.temperature
        logits = self.mixture_temperature * torch.logsumexp(
            scaled / self.mixture_temperature,
            dim=-1,
        )
        logits = logits - self.mixture_temperature * math.log(
            self.prototypes_per_class
        )
        return logits

    def regularization(self):
        prototypes = F.normalize(self.prototypes, dim=-1)
        loss = prototypes.new_zeros(())
        if self.prototypes_per_class > 1:
            within = torch.einsum(
                "ckd,cld->ckl", prototypes, prototypes
            )
            mask = ~torch.eye(
                self.prototypes_per_class,
                dtype=torch.bool,
                device=prototypes.device,
            )
            loss = loss + F.relu(
                within[:, mask] - self.diversity_margin
            ).square().mean()
        centers = F.normalize(prototypes.mean(dim=1), dim=-1)
        between = centers @ centers.T
        class_mask = ~torch.eye(
            self.num_classes,
            dtype=torch.bool,
            device=prototypes.device,
        )
        loss = loss + F.relu(
            between[class_mask] - self.separation_margin
        ).square().mean()
        return loss


class ResidualPrototypePotential(nn.Module):
    """MLP potential with an optional class-wise prototype residual."""

    def __init__(
        self,
        state_dim,
        num_classes,
        hidden_dim,
        dropout,
        act,
        prototype_embedding_dim,
        prototypes_per_class,
        prototype_temperature,
        prototype_mixture_temperature,
        prototype_diversity_margin,
        prototype_separation_margin,
        initial_scale,
    ):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        with torch.random.fork_rng():
            self.residual = MixturePrototypePotential(
                state_dim=state_dim,
                num_classes=num_classes,
                embedding_dim=prototype_embedding_dim,
                prototypes_per_class=prototypes_per_class,
                temperature=prototype_temperature,
                mixture_temperature=prototype_mixture_temperature,
                hidden_dim=hidden_dim,
                dropout=dropout,
                act=act,
                diversity_margin=prototype_diversity_margin,
                separation_margin=prototype_separation_margin,
            )
        self.residual_scale = nn.Parameter(
            torch.full((num_classes,), float(initial_scale))
        )

    def forward(self, state):
        return self.base(state) + self.residual_scale * self.residual(state)

    def regularization(self):
        return self.residual.regularization()


class MIR_MIL(nn.Module):
    """Neural measure potential with closed-form measure influence response."""

    def __init__(
        self,
        in_dim,
        num_classes,
        hidden_dim=256,
        sketch_dim=128,
        moment_order=1,
        num_tail_scores=8,
        tail_temperature=0.25,
        potential_hidden_dim=128,
        dropout=0.1,
        act="gelu",
        coordinate_dim=0,
        stability_weight=0.0,
        patch_dropout=0.0,
        feature_noise_std=0.0,
        coordinate_jitter_std=0.0,
        lipschitz_weight=0.0,
        lipschitz_target=1.0,
        lipschitz_samples=64,
        ordinal_weight=0.0,
        potential_type="mlp",
        prototype_embedding_dim=64,
        prototypes_per_class=4,
        prototype_temperature=0.2,
        prototype_mixture_temperature=1.0,
        prototype_regularization_weight=0.0,
        prototype_diversity_margin=0.0,
        prototype_separation_margin=0.0,
        prototype_residual_initial_scale=0.0,
    ):
        super().__init__()
        if in_dim <= 0 or num_classes < 2:
            raise ValueError("in_dim must be positive and num_classes >= 2")
        if coordinate_dim not in {0, 2}:
            raise ValueError("coordinate_dim must be 0 or 2")
        if num_tail_scores <= 0 or tail_temperature <= 0:
            raise ValueError(
                "num_tail_scores and tail_temperature must be positive"
            )
        self.in_dim = int(in_dim)
        self.coordinate_dim = int(coordinate_dim)
        self.input_dim = self.in_dim + self.coordinate_dim
        self.num_classes = int(num_classes)
        self.sketch_dim = int(sketch_dim)
        self.moment_order = int(moment_order)
        if self.moment_order not in {1, 2}:
            raise ValueError("moment_order must be 1 or 2")
        self.composition_state_dim = self.sketch_dim * self.moment_order
        self.num_tail_scores = int(num_tail_scores)
        self.tail_temperature = float(tail_temperature)
        self.stability_weight = float(stability_weight)
        self.patch_dropout = float(patch_dropout)
        self.feature_noise_std = float(feature_noise_std)
        self.coordinate_jitter_std = float(coordinate_jitter_std)
        self.lipschitz_weight = float(lipschitz_weight)
        self.lipschitz_target = float(lipschitz_target)
        self.lipschitz_samples = int(lipschitz_samples)
        self.ordinal_weight = float(ordinal_weight)
        if self.ordinal_weight < 0:
            raise ValueError("ordinal_weight must be non-negative")
        self.potential_type = str(potential_type)
        self.prototype_regularization_weight = float(
            prototype_regularization_weight
        )
        if self.prototype_regularization_weight < 0:
            raise ValueError(
                "prototype_regularization_weight must be non-negative"
            )

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
        )
        self.response_basis = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            _activation(act),
            nn.Linear(hidden_dim, self.sketch_dim),
        )
        self.tail_scorer = nn.Linear(hidden_dim, self.num_tail_scores)
        state_dim = self.composition_state_dim + self.num_tail_scores
        if self.potential_type == "mlp":
            self.potential = nn.Sequential(
                nn.Linear(state_dim, potential_hidden_dim),
                _activation(act),
                nn.Dropout(dropout),
                nn.Linear(potential_hidden_dim, self.num_classes),
            )
        elif self.potential_type == "mixture_prototype":
            self.potential = MixturePrototypePotential(
                state_dim=state_dim,
                num_classes=self.num_classes,
                embedding_dim=prototype_embedding_dim,
                prototypes_per_class=prototypes_per_class,
                temperature=prototype_temperature,
                mixture_temperature=prototype_mixture_temperature,
                hidden_dim=potential_hidden_dim,
                dropout=dropout,
                act=act,
                diversity_margin=prototype_diversity_margin,
                separation_margin=prototype_separation_margin,
            )
        elif self.potential_type == "residual_prototype":
            self.potential = ResidualPrototypePotential(
                state_dim=state_dim,
                num_classes=self.num_classes,
                hidden_dim=potential_hidden_dim,
                dropout=dropout,
                act=act,
                prototype_embedding_dim=prototype_embedding_dim,
                prototypes_per_class=prototypes_per_class,
                prototype_temperature=prototype_temperature,
                prototype_mixture_temperature=(
                    prototype_mixture_temperature
                ),
                prototype_diversity_margin=prototype_diversity_margin,
                prototype_separation_margin=prototype_separation_margin,
                initial_scale=prototype_residual_initial_scale,
            )
        else:
            raise ValueError(
                f"Unsupported potential_type: {self.potential_type}"
            )
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _normalize_bag(self, bag):
        if bag.ndim == 3:
            if bag.shape[0] != 1:
                raise ValueError("MIR_MIL expects batch size one")
            bag = bag.squeeze(0)
        if bag.ndim != 2 or bag.shape[0] == 0:
            raise ValueError("bag must have shape [N, D] with N > 0")
        if bag.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension {self.input_dim}, got "
                f"{bag.shape[1]}"
            )
        return bag

    def _point_statistics(self, points):
        encoded = self.encoder(points)
        return self.response_basis(encoded), self.tail_scorer(encoded)

    def state_from_weighted_points(self, points, weights=None):
        points = self._normalize_bag(points)
        basis, tail_scores = self._point_statistics(points)
        if weights is None:
            weights = torch.full(
                (points.shape[0],),
                1.0 / points.shape[0],
                dtype=points.dtype,
                device=points.device,
            )
        else:
            weights = weights.to(device=points.device, dtype=points.dtype)
            if weights.ndim != 1 or weights.shape[0] != points.shape[0]:
                raise ValueError("weights must have shape [N]")
            if torch.any(weights < 0):
                raise ValueError("weights must be non-negative")
            weights = weights / weights.sum().clamp_min(
                torch.finfo(points.dtype).eps
            )

        composition = torch.sum(weights[:, None] * basis, dim=0)
        composition_state = [composition]
        if self.moment_order == 2:
            composition_state.append(
                torch.sum(
                    weights[:, None]
                    * (basis - composition[None, :]).square(),
                    dim=0,
                )
            )
        log_weights = torch.log(
            weights.clamp_min(torch.finfo(points.dtype).tiny)
        )
        tail_state = self.tail_temperature * torch.logsumexp(
            tail_scores / self.tail_temperature + log_weights[:, None],
            dim=0,
        )
        state = torch.cat([*composition_state, tail_state], dim=0)
        return state, basis, tail_scores, weights

    def forward(self, bag, return_state=False):
        state, basis, tail_scores, weights = self.state_from_weighted_points(
            bag
        )
        logits = self.potential(state.unsqueeze(0))
        output = {"logits": logits}
        if return_state:
            output.update(
                {
                    "state": state,
                    "basis": basis,
                    "tail_scores": tail_scores,
                    "weights": weights,
                }
            )
        return output

    def explained_score(self, logits, target_class):
        logits = logits.squeeze(0)
        target_class = int(target_class)
        if not 0 <= target_class < self.num_classes:
            raise ValueError("target_class is out of range")
        other = torch.cat(
            [logits[:target_class], logits[target_class + 1 :]], dim=0
        )
        return logits[target_class] - torch.logsumexp(other, dim=0)

    def _state_gradient(self, state, target_class, create_graph=False):
        logits = self.potential(state.unsqueeze(0))
        score = self.explained_score(logits, target_class)
        gradient = torch.autograd.grad(
            score,
            state,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        return score, gradient

    def measure_influence_response(
        self,
        bag,
        target_class=None,
        create_graph=False,
    ):
        state, basis, tail_scores, weights = self.state_from_weighted_points(
            bag
        )
        logits = self.potential(state.unsqueeze(0))
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = self.explained_score(logits, target_class)
        gradient = torch.autograd.grad(
            score,
            state,
            create_graph=create_graph,
            retain_graph=True,
        )[0]
        composition = state[: self.sketch_dim]
        composition_gradient = gradient[: self.sketch_dim]
        offset = self.sketch_dim
        variance_response = torch.zeros(
            basis.shape[0], dtype=basis.dtype, device=basis.device
        )
        if self.moment_order == 2:
            variance = state[offset : offset + self.sketch_dim]
            variance_gradient = gradient[
                offset : offset + self.sketch_dim
            ]
            variance_response = (
                (
                    (basis - composition[None, :]).square()
                    - variance[None, :]
                )
                * variance_gradient[None, :]
            ).sum(dim=1)
            offset += self.sketch_dim
        tail_gradient = gradient[offset:]

        log_normalizers = torch.logsumexp(
            tail_scores / self.tail_temperature
            + torch.log(
                weights.clamp_min(torch.finfo(weights.dtype).tiny)
            )[:, None],
            dim=0,
        )
        density_ratio = torch.exp(
            tail_scores / self.tail_temperature - log_normalizers[None, :]
        )
        composition_response = (
            (basis - composition[None, :]) * composition_gradient[None, :]
        ).sum(dim=1)
        tail_response = (
            self.tail_temperature
            * (density_ratio - 1.0)
            * tail_gradient[None, :]
        ).sum(dim=1)
        return {
            "logits": logits,
            "score": score,
            "target_class": target_class,
            "response": (
                composition_response + variance_response + tail_response
            ),
            "composition_response": composition_response,
            "variance_response": variance_response,
            "tail_response": tail_response,
        }

    def functional_derivative(
        self,
        evaluation_points,
        reference_points,
        reference_weights=None,
        target_class=None,
    ):
        reference_state, _, reference_tail, reference_weights = (
            self.state_from_weighted_points(
                reference_points, reference_weights
            )
        )
        logits = self.potential(reference_state.unsqueeze(0))
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = self.explained_score(logits, target_class)
        gradient = torch.autograd.grad(
            score, reference_state, retain_graph=True
        )[0]
        evaluation_points = self._normalize_bag(evaluation_points)
        basis, tail_scores = self._point_statistics(evaluation_points)
        composition = reference_state[: self.sketch_dim]
        composition_gradient = gradient[: self.sketch_dim]
        offset = self.sketch_dim
        variance_derivative = basis.new_zeros(basis.shape[0])
        if self.moment_order == 2:
            variance_gradient = gradient[
                offset : offset + self.sketch_dim
            ]
            variance_derivative = (
                (basis.square() - 2.0 * composition[None, :] * basis)
                * variance_gradient[None, :]
            ).sum(dim=1)
            offset += self.sketch_dim
        tail_gradient = gradient[offset:]
        log_normalizers = torch.logsumexp(
            reference_tail / self.tail_temperature
            + torch.log(
                reference_weights.clamp_min(
                    torch.finfo(reference_weights.dtype).tiny
                )
            )[:, None],
            dim=0,
        )
        density_ratio = torch.exp(
            tail_scores / self.tail_temperature - log_normalizers[None, :]
        )
        return (
            basis * composition_gradient[None, :]
        ).sum(dim=1) + variance_derivative + (
            self.tail_temperature
            * density_ratio
            * tail_gradient[None, :]
        ).sum(
            dim=1
        )

    def integrated_functional_attribution(
        self,
        bag,
        baseline_bag,
        target_class=None,
        steps=32,
    ):
        bag = self._normalize_bag(bag)
        baseline_bag = self._normalize_bag(baseline_bag)
        if steps < 2:
            raise ValueError("steps must be at least 2")

        with torch.enable_grad():
            if target_class is None:
                target_class = int(
                    self.forward(bag)["logits"].argmax(dim=1).item()
                )
            support = torch.cat([bag, baseline_bag], dim=0)
            target_values = []
            baseline_values = []
            grid = torch.linspace(
                0.0,
                1.0,
                steps,
                device=bag.device,
                dtype=bag.dtype,
            )
            for t in grid:
                weights = torch.cat(
                    [
                        torch.full(
                            (bag.shape[0],),
                            t / bag.shape[0],
                            device=bag.device,
                            dtype=bag.dtype,
                        ),
                        torch.full(
                            (baseline_bag.shape[0],),
                            (1.0 - t) / baseline_bag.shape[0],
                            device=bag.device,
                            dtype=bag.dtype,
                        ),
                    ]
                )
                values = self.functional_derivative(
                    support,
                    support,
                    weights,
                    target_class,
                )
                target_values.append(values[: bag.shape[0]])
                baseline_values.append(values[bag.shape[0] :])
            target_values = torch.stack(target_values)
            baseline_values = torch.stack(baseline_values)
            target_integrated = torch.trapezoid(
                target_values, grid, dim=0
            )
            baseline_integrated = torch.trapezoid(
                baseline_values, grid, dim=0
            )
            decomposition = (
                target_integrated.mean() - baseline_integrated.mean()
            )
            bag_score = self.explained_score(
                self.forward(bag)["logits"], target_class
            )
            baseline_score = self.explained_score(
                self.forward(baseline_bag)["logits"], target_class
            )
        return {
            "target_class": target_class,
            "bag_attribution": target_integrated,
            "baseline_attribution": baseline_integrated,
            "decomposition": decomposition,
            "score_difference": bag_score - baseline_score,
        }

    def finite_difference_response(
        self,
        bag,
        point,
        target_class,
        epsilon=1e-3,
    ):
        bag = self._normalize_bag(bag)
        point = point.reshape(1, -1)
        support = torch.cat([bag, point], dim=0)
        weights = torch.cat(
            [
                torch.full(
                    (bag.shape[0],),
                    (1.0 - epsilon) / bag.shape[0],
                    device=bag.device,
                    dtype=bag.dtype,
                ),
                torch.tensor(
                    [epsilon], device=bag.device, dtype=bag.dtype
                ),
            ]
        )
        perturbed_state, _, _, _ = self.state_from_weighted_points(
            support, weights
        )
        perturbed_score = self.explained_score(
            self.potential(perturbed_state.unsqueeze(0)), target_class
        )
        base_score = self.explained_score(
            self.forward(bag)["logits"], target_class
        )
        return (perturbed_score - base_score) / epsilon

    def augment_bag(self, bag):
        bag = self._normalize_bag(bag)
        augmented = bag
        if self.patch_dropout > 0 and bag.shape[0] > 1:
            keep = torch.rand(bag.shape[0], device=bag.device)
            keep = keep >= self.patch_dropout
            if not keep.any():
                keep[torch.randint(bag.shape[0], (1,), device=bag.device)] = True
            augmented = augmented[keep]
        augmented = augmented.clone()
        if self.feature_noise_std > 0:
            augmented[:, : self.in_dim] += (
                torch.randn_like(augmented[:, : self.in_dim])
                * self.feature_noise_std
            )
        if self.coordinate_dim and self.coordinate_jitter_std > 0:
            augmented[:, -self.coordinate_dim :] += (
                torch.randn_like(augmented[:, -self.coordinate_dim :])
                * self.coordinate_jitter_std
            )
        return augmented

    def lipschitz_penalty(self, bag):
        if self.lipschitz_weight <= 0:
            return bag.new_zeros(())
        bag = self._normalize_bag(bag)
        count = min(self.lipschitz_samples, bag.shape[0])
        indices = torch.randperm(bag.shape[0], device=bag.device)[:count]
        points = bag[indices].detach().requires_grad_(True)
        basis, _ = self._point_statistics(points)
        probe = torch.randn_like(basis)
        probe = probe / probe.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scalar = (basis * probe).sum()
        gradients = torch.autograd.grad(
            scalar, points, create_graph=True
        )[0]
        norms = gradients.norm(dim=1)
        return F.relu(norms - self.lipschitz_target).square().mean()

    def compute_loss(self, bag, label, criterion):
        output = self.forward(bag)
        classification_loss = criterion(output["logits"], label)
        ordinal_loss = self.ordinal_cdf_loss(output["logits"], label)
        stability_loss = bag.new_zeros(())
        if self.stability_weight > 0:
            augmented_logits = self.forward(self.augment_bag(bag))["logits"]
            stability_loss = F.mse_loss(
                output["logits"], augmented_logits
            )
        lipschitz_loss = self.lipschitz_penalty(bag)
        prototype_loss = bag.new_zeros(())
        if hasattr(self.potential, "regularization"):
            prototype_loss = self.potential.regularization()
        loss = (
            classification_loss
            + self.ordinal_weight * ordinal_loss
            + self.stability_weight * stability_loss
            + self.lipschitz_weight * lipschitz_loss
            + self.prototype_regularization_weight * prototype_loss
        )
        return output, {
            "loss": loss,
            "classification_loss": classification_loss,
            "ordinal_loss": ordinal_loss,
            "stability_loss": stability_loss,
            "lipschitz_loss": lipschitz_loss,
            "prototype_loss": prototype_loss,
        }

    def ordinal_cdf_loss(self, logits, label):
        probabilities = torch.softmax(logits, dim=1)
        predicted_cdf = probabilities.cumsum(dim=1)[:, :-1]
        thresholds = torch.arange(
            self.num_classes - 1,
            device=logits.device,
        ).unsqueeze(0)
        target_cdf = (label.unsqueeze(1) <= thresholds).to(logits.dtype)
        return F.mse_loss(predicted_cdf, target_cdf)
