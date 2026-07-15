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


class AdaptiveMultiscalePotential(nn.Module):
    """Global potential with a sample-adaptive local residual."""

    def __init__(
        self,
        global_state_dim,
        local_state_dim,
        num_classes,
        hidden_dim,
        dropout,
        act,
        gate_initial_bias,
        local_initial_scale,
    ):
        super().__init__()
        if global_state_dim <= 0 or local_state_dim <= 0:
            raise ValueError(
                "global_state_dim and local_state_dim must be positive"
            )
        self.global_state_dim = int(global_state_dim)
        self.local_state_dim = int(local_state_dim)
        self.gate_initial_bias = float(gate_initial_bias)
        self.global_potential = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.local_potential = nn.Sequential(
            nn.Linear(self.local_state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.local_gate = nn.Linear(self.global_state_dim, num_classes)
        self.local_scale = nn.Parameter(
            torch.full((num_classes,), float(local_initial_scale))
        )

    def reset_gate(self):
        nn.init.zeros_(self.local_gate.weight)
        nn.init.constant_(self.local_gate.bias, self.gate_initial_bias)

    def forward(self, state):
        global_state = state[:, : self.global_state_dim]
        local_state = state[:, self.global_state_dim :]
        if local_state.shape[1] != self.local_state_dim:
            raise ValueError(
                f"Expected local state dimension {self.local_state_dim}, "
                f"got {local_state.shape[1]}"
            )
        gate = torch.sigmoid(self.local_gate(global_state))
        return self.global_potential(global_state) + (
            gate * self.local_scale * self.local_potential(local_state)
        )


class AnchoredMultiscalePotential(nn.Module):
    """Full-rank attention potential with global and routed residuals."""

    def __init__(
        self,
        global_state_dim,
        local_state_dim,
        anchor_state_dim,
        num_classes,
        hidden_dim,
        dropout,
        act,
        gate_initial_bias,
        global_initial_scale,
        local_initial_scale,
    ):
        super().__init__()
        if min(
            global_state_dim, local_state_dim, anchor_state_dim
        ) <= 0:
            raise ValueError("anchored state dimensions must be positive")
        self.global_state_dim = int(global_state_dim)
        self.local_state_dim = int(local_state_dim)
        self.anchor_state_dim = int(anchor_state_dim)
        self.gate_initial_bias = float(gate_initial_bias)
        self.anchor_potential = nn.Linear(
            self.anchor_state_dim, num_classes
        )
        self.global_potential = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.local_potential = nn.Sequential(
            nn.Linear(self.local_state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        self.local_gate = nn.Linear(self.global_state_dim, num_classes)
        self.global_scale = nn.Parameter(
            torch.full((num_classes,), float(global_initial_scale))
        )
        self.local_scale = nn.Parameter(
            torch.full((num_classes,), float(local_initial_scale))
        )

    def reset_gate(self):
        nn.init.zeros_(self.local_gate.weight)
        nn.init.constant_(
            self.local_gate.bias, self.gate_initial_bias
        )

    def forward(self, state):
        global_end = self.global_state_dim
        local_end = global_end + self.local_state_dim
        global_state = state[:, :global_end]
        local_state = state[:, global_end:local_end]
        anchor_state = state[:, local_end:]
        if anchor_state.shape[1] != self.anchor_state_dim:
            raise ValueError(
                f"Expected anchor state dimension "
                f"{self.anchor_state_dim}, got {anchor_state.shape[1]}"
            )
        gate = torch.sigmoid(self.local_gate(global_state))
        return (
            self.anchor_potential(anchor_state)
            + self.global_scale * self.global_potential(global_state)
            + gate * self.local_scale * self.local_potential(local_state)
        )


class ClassConditionalMultiscalePotential(nn.Module):
    """Global potential with class-owned local measure routes."""

    def __init__(
        self,
        global_state_dim,
        local_state_dim,
        num_classes,
        hidden_dim,
        dropout,
        act,
        gate_initial_bias,
        local_initial_scale,
    ):
        super().__init__()
        if global_state_dim <= 0 or local_state_dim <= 0:
            raise ValueError(
                "global_state_dim and local_state_dim must be positive"
            )
        if local_state_dim % num_classes:
            raise ValueError(
                "local_state_dim must be divisible by num_classes"
            )
        self.global_state_dim = int(global_state_dim)
        self.local_state_dim = int(local_state_dim)
        self.num_classes = int(num_classes)
        self.class_local_state_dim = (
            self.local_state_dim // self.num_classes
        )
        self.gate_initial_bias = float(gate_initial_bias)
        self.global_potential = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        with torch.random.fork_rng():
            self.class_local_potentials = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.class_local_state_dim, hidden_dim
                        ),
                        _activation(act),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(self.num_classes)
                ]
            )
        self.local_gate = nn.Linear(
            self.global_state_dim, self.num_classes
        )
        self.local_scale = nn.Parameter(
            torch.full(
                (self.num_classes,), float(local_initial_scale)
            )
        )

    def reset_gate(self):
        nn.init.zeros_(self.local_gate.weight)
        nn.init.constant_(
            self.local_gate.bias, self.gate_initial_bias
        )

    def forward(self, state):
        global_state = state[:, : self.global_state_dim]
        local_state = state[:, self.global_state_dim :]
        if local_state.shape[1] != self.local_state_dim:
            raise ValueError(
                f"Expected local state dimension {self.local_state_dim}, "
                f"got {local_state.shape[1]}"
            )
        class_states = local_state.reshape(
            local_state.shape[0],
            self.num_classes,
            self.class_local_state_dim,
        )
        local_logits = torch.cat(
            [
                potential(class_states[:, class_index])
                for class_index, potential in enumerate(
                    self.class_local_potentials
                )
            ],
            dim=1,
        )
        gate = torch.sigmoid(self.local_gate(global_state))
        return self.global_potential(global_state) + (
            gate * self.local_scale * local_logits
        )


class HybridMultiscalePotential(nn.Module):
    """Adaptive mixture of shared and class-owned local potentials."""

    def __init__(
        self,
        global_state_dim,
        local_state_dim,
        num_classes,
        hidden_dim,
        dropout,
        act,
        gate_initial_bias,
        local_initial_scale,
        class_mix_initial,
    ):
        super().__init__()
        if not 0 < class_mix_initial < 1:
            raise ValueError("class_mix_initial must be between zero and one")
        if global_state_dim <= 0 or local_state_dim <= 0:
            raise ValueError(
                "global_state_dim and local_state_dim must be positive"
            )
        if local_state_dim % num_classes:
            raise ValueError(
                "local_state_dim must be divisible by num_classes"
            )
        self.global_state_dim = int(global_state_dim)
        self.local_state_dim = int(local_state_dim)
        self.num_classes = int(num_classes)
        self.class_local_state_dim = (
            self.local_state_dim // self.num_classes
        )
        self.gate_initial_bias = float(gate_initial_bias)
        # Keep the shared path in the same creation order as
        # AdaptiveMultiscalePotential so it has the same seeded initialization.
        self.global_potential = nn.Sequential(
            nn.Linear(self.global_state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.shared_local_potential = nn.Sequential(
            nn.Linear(self.local_state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.num_classes),
        )
        self.local_gate = nn.Linear(
            self.global_state_dim, self.num_classes
        )
        self.local_scale = nn.Parameter(
            torch.full(
                (self.num_classes,), float(local_initial_scale)
            )
        )
        with torch.random.fork_rng():
            self.class_local_potentials = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(
                            self.class_local_state_dim, hidden_dim
                        ),
                        _activation(act),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, 1),
                    )
                    for _ in range(self.num_classes)
                ]
            )
        initial_logit = math.log(
            class_mix_initial / (1.0 - class_mix_initial)
        )
        self.class_mix_logit = nn.Parameter(
            torch.full((self.num_classes,), initial_logit)
        )

    def reset_gate(self):
        nn.init.zeros_(self.local_gate.weight)
        nn.init.constant_(
            self.local_gate.bias, self.gate_initial_bias
        )

    def forward(self, state):
        global_state = state[:, : self.global_state_dim]
        local_state = state[:, self.global_state_dim :]
        if local_state.shape[1] != self.local_state_dim:
            raise ValueError(
                f"Expected local state dimension {self.local_state_dim}, "
                f"got {local_state.shape[1]}"
            )
        class_states = local_state.reshape(
            local_state.shape[0],
            self.num_classes,
            self.class_local_state_dim,
        )
        class_logits = torch.cat(
            [
                potential(class_states[:, class_index])
                for class_index, potential in enumerate(
                    self.class_local_potentials
                )
            ],
            dim=1,
        )
        shared_logits = self.shared_local_potential(local_state)
        class_mix = torch.sigmoid(self.class_mix_logit)
        local_logits = (
            (1.0 - class_mix) * shared_logits
            + class_mix * class_logits
        )
        gate = torch.sigmoid(self.local_gate(global_state))
        return self.global_potential(global_state) + (
            gate * self.local_scale * local_logits
        )


class ResidualClassMultiscalePotential(HybridMultiscalePotential):
    """Shared local potential with a class-owned residual."""

    def __init__(
        self,
        global_state_dim,
        local_state_dim,
        num_classes,
        hidden_dim,
        dropout,
        act,
        gate_initial_bias,
        local_initial_scale,
        class_residual_initial_scale,
    ):
        super().__init__(
            global_state_dim=global_state_dim,
            local_state_dim=local_state_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            act=act,
            gate_initial_bias=gate_initial_bias,
            local_initial_scale=local_initial_scale,
            class_mix_initial=0.5,
        )
        del self.class_mix_logit
        self.class_residual_scale = nn.Parameter(
            torch.full(
                (self.num_classes,),
                float(class_residual_initial_scale),
            )
        )

    def forward(self, state):
        global_state = state[:, : self.global_state_dim]
        local_state = state[:, self.global_state_dim :]
        if local_state.shape[1] != self.local_state_dim:
            raise ValueError(
                f"Expected local state dimension {self.local_state_dim}, "
                f"got {local_state.shape[1]}"
            )
        class_states = local_state.reshape(
            local_state.shape[0],
            self.num_classes,
            self.class_local_state_dim,
        )
        class_logits = torch.cat(
            [
                potential(class_states[:, class_index])
                for class_index, potential in enumerate(
                    self.class_local_potentials
                )
            ],
            dim=1,
        )
        local_logits = (
            self.shared_local_potential(local_state)
            + self.class_residual_scale * class_logits
        )
        gate = torch.sigmoid(self.local_gate(global_state))
        return self.global_potential(global_state) + (
            gate * self.local_scale * local_logits
        )


class AdaptiveMultiscalePrototypePotential(AdaptiveMultiscalePotential):
    """Adaptive multiscale potential with class-wise state prototypes."""

    def __init__(
        self,
        state_dim,
        global_state_dim,
        local_state_dim,
        num_classes,
        hidden_dim,
        dropout,
        act,
        gate_initial_bias,
        local_initial_scale,
        prototype_embedding_dim,
        prototypes_per_class,
        prototype_temperature,
        prototype_mixture_temperature,
        prototype_diversity_margin,
        prototype_separation_margin,
        prototype_initial_scale,
    ):
        super().__init__(
            global_state_dim=global_state_dim,
            local_state_dim=local_state_dim,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            dropout=dropout,
            act=act,
            gate_initial_bias=gate_initial_bias,
            local_initial_scale=local_initial_scale,
        )
        with torch.random.fork_rng():
            self.prototype_potential = MixturePrototypePotential(
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
        self.prototype_scale = nn.Parameter(
            torch.full((num_classes,), float(prototype_initial_scale))
        )

    def forward(self, state):
        return super().forward(state) + (
            self.prototype_scale * self.prototype_potential(state)
        )

    def regularization(self):
        return self.prototype_potential.regularization()


class ClassAwareEvidenceHead(nn.Module):
    """Class-wise patch evidence readout over existing encoded instances."""

    def __init__(
        self,
        hidden_dim,
        num_classes,
        query_dim,
        value_dim,
        temperature,
        dropout,
    ):
        super().__init__()
        if query_dim <= 0 or value_dim <= 0:
            raise ValueError("query_dim and value_dim must be positive")
        if temperature <= 0:
            raise ValueError("evidence temperature must be positive")
        self.num_classes = int(num_classes)
        self.query_dim = int(query_dim)
        self.temperature = float(temperature)
        self.key = nn.Linear(hidden_dim, self.query_dim)
        self.value = nn.Linear(hidden_dim, int(value_dim))
        self.query = nn.Parameter(
            torch.empty(self.num_classes, self.query_dim)
        )
        self.logit_vector = nn.Parameter(
            torch.empty(self.num_classes, int(value_dim))
        )
        self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.dropout = nn.Dropout(dropout)
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.query, std=1.0 / math.sqrt(self.query_dim)
            )
            nn.init.normal_(
                self.logit_vector, std=1.0 / math.sqrt(value_dim)
            )

    def forward(self, encoded):
        keys = self.key(encoded)
        values = self.dropout(self.value(encoded))
        scale = math.sqrt(self.query_dim) * self.temperature
        scores = torch.einsum("nq,cq->nc", keys, self.query) / scale
        attention = torch.softmax(scores, dim=0)
        evidence = torch.einsum("nc,nv->cv", attention, values)
        logits = (
            evidence * self.logit_vector
        ).sum(dim=1) + self.logit_bias
        return logits.unsqueeze(0)


class MultiTokenAttentionReadout(nn.Module):
    """Dataset-agnostic multi-token attention readout over encoded patches."""

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_tokens,
        token_dim,
        readout_dim,
        temperature,
        dropout,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if token_dim <= 0 or readout_dim <= 0:
            raise ValueError("token_dim and readout_dim must be positive")
        if temperature <= 0:
            raise ValueError("multi-token temperature must be positive")
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.temperature = float(temperature)
        self.key = nn.Linear(hidden_dim, self.token_dim)
        self.value = nn.Linear(hidden_dim, int(readout_dim))
        self.tokens = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(int(readout_dim) * self.num_tokens),
            nn.Dropout(dropout),
            nn.Linear(int(readout_dim) * self.num_tokens, num_classes),
        )
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.tokens, std=1.0 / math.sqrt(self.token_dim)
            )

    def forward(self, encoded):
        keys = self.key(encoded)
        values = self.value(encoded)
        scale = math.sqrt(self.token_dim) * self.temperature
        scores = torch.einsum("nd,td->nt", keys, self.tokens) / scale
        attention = torch.softmax(scores, dim=0)
        token_values = torch.einsum("nt,nv->tv", attention, values)
        return self.output(token_values.flatten().unsqueeze(0))


class MomentMultiTokenAttentionReadout(nn.Module):
    """Multi-token attention readout with per-token mean and variance.

    Fixed multi-token attention extracts one weighted value mean per token.
    This variant preserves that generic evidence retrieval path while adding
    second-order token statistics. It remains dataset-agnostic and works for
    arbitrary class counts.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_tokens,
        token_dim,
        readout_dim,
        temperature,
        dropout,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if token_dim <= 0 or readout_dim <= 0:
            raise ValueError("token_dim and readout_dim must be positive")
        if temperature <= 0:
            raise ValueError("moment-token temperature must be positive")
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.readout_dim = int(readout_dim)
        self.temperature = float(temperature)
        self.key = nn.Linear(hidden_dim, self.token_dim)
        self.value = nn.Linear(hidden_dim, self.readout_dim)
        self.tokens = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(2 * self.readout_dim * self.num_tokens),
            nn.Dropout(dropout),
            nn.Linear(2 * self.readout_dim * self.num_tokens, num_classes),
        )
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.tokens, std=1.0 / math.sqrt(self.token_dim)
            )

    def forward(self, encoded):
        keys = self.key(encoded)
        values = self.value(encoded)
        scale = math.sqrt(self.token_dim) * self.temperature
        scores = torch.einsum("nd,td->nt", keys, self.tokens) / scale
        attention = torch.softmax(scores, dim=0)
        token_mean = torch.einsum("nt,nv->tv", attention, values)
        token_second = torch.einsum(
            "nt,nv->tv", attention, values.square()
        )
        token_variance = (token_second - token_mean.square()).clamp_min(
            0.0
        )
        token_statistics = torch.cat((token_mean, token_variance), dim=1)
        return self.output(token_statistics.flatten().unsqueeze(0))


class TailTokenAttentionReadout(nn.Module):
    """Multi-token readout that preserves both average and tail evidence.

    A soft attention mean captures distributed evidence, while a per-token
    top-k mean captures rare high-response patches that can be diluted by
    whole-bag averaging. The module is dataset-agnostic: tokens are shared
    across classes, the top-k fraction is independent of dataset identity, and
    the final linear readout works for arbitrary class counts.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_tokens,
        token_dim,
        readout_dim,
        temperature,
        topk_fraction,
        dropout,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if token_dim <= 0 or readout_dim <= 0:
            raise ValueError("token_dim and readout_dim must be positive")
        if temperature <= 0:
            raise ValueError("tail-token temperature must be positive")
        if not 0 < topk_fraction <= 1:
            raise ValueError("tail-token topk_fraction must be in (0, 1]")
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.readout_dim = int(readout_dim)
        self.temperature = float(temperature)
        self.topk_fraction = float(topk_fraction)
        self.key = nn.Linear(hidden_dim, self.token_dim)
        self.value = nn.Linear(hidden_dim, self.readout_dim)
        self.tokens = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim)
        )
        self.output = nn.Sequential(
            nn.LayerNorm(2 * self.readout_dim * self.num_tokens),
            nn.Dropout(dropout),
            nn.Linear(2 * self.readout_dim * self.num_tokens, num_classes),
        )
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.tokens, std=1.0 / math.sqrt(self.token_dim)
            )

    def forward(self, encoded):
        keys = self.key(encoded)
        values = self.value(encoded)
        scale = math.sqrt(self.token_dim) * self.temperature
        scores = torch.einsum("nd,td->nt", keys, self.tokens) / scale
        attention = torch.softmax(scores, dim=0)
        token_mean = torch.einsum("nt,nv->tv", attention, values)
        k = max(1, math.ceil(encoded.shape[0] * self.topk_fraction))
        k = min(k, encoded.shape[0])
        top_indices = torch.topk(scores, k=k, dim=0).indices
        top_values = values[top_indices]
        token_tail_mean = top_values.mean(dim=0)
        token_statistics = torch.cat((token_mean, token_tail_mean), dim=1)
        return self.output(token_statistics.flatten().unsqueeze(0))


class LowRankClassMomentTokenReadout(nn.Module):
    """Class-conditioned readout over shared moment-token evidence.

    Shared tokens retrieve evidence modes from the bag. For each token, the
    module computes first- and second-order value statistics, projects them
    into a low-rank space, then lets each class form logits from generic
    class factors. Classes do not own attention queries, so retrieval remains
    dataset-agnostic while boundaries can be class-specific.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_tokens,
        token_dim,
        value_dim,
        rank_dim,
        temperature,
        dropout,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if min(token_dim, value_dim, rank_dim) <= 0:
            raise ValueError(
                "token_dim, value_dim, and rank_dim must be positive"
            )
        if temperature <= 0:
            raise ValueError(
                "class-moment-token temperature must be positive"
            )
        self.num_classes = int(num_classes)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.value_dim = int(value_dim)
        self.rank_dim = int(rank_dim)
        self.temperature = float(temperature)
        self.key = nn.Linear(hidden_dim, self.token_dim)
        self.value = nn.Linear(hidden_dim, self.value_dim)
        self.tokens = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim)
        )
        self.token_projector = nn.Sequential(
            nn.LayerNorm(2 * self.value_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * self.value_dim, self.rank_dim),
        )
        self.class_factors = nn.Parameter(
            torch.empty(
                self.num_classes,
                self.num_tokens,
                self.rank_dim,
            )
        )
        self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.tokens, std=1.0 / math.sqrt(self.token_dim)
            )
            nn.init.normal_(
                self.class_factors,
                std=1.0 / math.sqrt(self.num_tokens * self.rank_dim),
            )

    def forward(self, encoded):
        keys = self.key(encoded)
        values = self.value(encoded)
        scale = math.sqrt(self.token_dim) * self.temperature
        scores = torch.einsum("nd,td->nt", keys, self.tokens) / scale
        attention = torch.softmax(scores, dim=0)
        token_mean = torch.einsum("nt,nv->tv", attention, values)
        token_second = torch.einsum(
            "nt,nv->tv", attention, values.square()
        )
        token_variance = (token_second - token_mean.square()).clamp_min(
            0.0
        )
        token_statistics = torch.cat((token_mean, token_variance), dim=1)
        token_evidence = self.token_projector(token_statistics)
        logits = torch.einsum(
            "tr,ctr->c", token_evidence, self.class_factors
        ) + self.logit_bias
        return logits.unsqueeze(0)


class ResidualClassMomentTokenReadout(nn.Module):
    """Shared moment-token readout with a class-conditioned residual.

    The shared path keeps the same generic evidence extraction bias as the
    successful moment-token head. A small low-rank class residual can then
    adjust decision boundaries without giving each class its own patch query.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_tokens,
        token_dim,
        readout_dim,
        rank_dim,
        temperature,
        dropout,
        residual_initial_scale,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if min(token_dim, readout_dim, rank_dim) <= 0:
            raise ValueError(
                "token_dim, readout_dim, and rank_dim must be positive"
            )
        if temperature <= 0:
            raise ValueError(
                "residual-class-moment temperature must be positive"
            )
        self.num_classes = int(num_classes)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.readout_dim = int(readout_dim)
        self.rank_dim = int(rank_dim)
        self.temperature = float(temperature)
        self.key = nn.Linear(hidden_dim, self.token_dim)
        self.value = nn.Linear(hidden_dim, self.readout_dim)
        self.tokens = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim)
        )
        statistics_dim = 2 * self.readout_dim * self.num_tokens
        self.shared_output = nn.Sequential(
            nn.LayerNorm(statistics_dim),
            nn.Dropout(dropout),
            nn.Linear(statistics_dim, self.num_classes),
        )
        self.residual_projector = nn.Sequential(
            nn.LayerNorm(2 * self.readout_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * self.readout_dim, self.rank_dim),
            nn.GELU(),
        )
        self.class_factors = nn.Parameter(
            torch.empty(
                self.num_classes,
                self.num_tokens,
                self.rank_dim,
            )
        )
        self.residual_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.residual_scale = nn.Parameter(
            torch.full(
                (self.num_classes,), float(residual_initial_scale)
            )
        )
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.tokens, std=1.0 / math.sqrt(self.token_dim)
            )
            nn.init.normal_(
                self.class_factors,
                std=1.0 / math.sqrt(self.num_tokens * self.rank_dim),
            )

    def forward(self, encoded):
        keys = self.key(encoded)
        values = self.value(encoded)
        scale = math.sqrt(self.token_dim) * self.temperature
        scores = torch.einsum("nd,td->nt", keys, self.tokens) / scale
        attention = torch.softmax(scores, dim=0)
        token_mean = torch.einsum("nt,nv->tv", attention, values)
        token_second = torch.einsum(
            "nt,nv->tv", attention, values.square()
        )
        token_variance = (token_second - token_mean.square()).clamp_min(
            0.0
        )
        token_statistics = torch.cat((token_mean, token_variance), dim=1)
        shared_logits = self.shared_output(
            token_statistics.flatten().unsqueeze(0)
        )
        token_evidence = self.residual_projector(token_statistics)
        residual_logits = torch.einsum(
            "tr,ctr->c", token_evidence, self.class_factors
        ) + self.residual_bias
        return shared_logits + self.residual_scale * residual_logits


class LowRankClassTokenReadout(nn.Module):
    """Shared evidence tokens with a low-rank class-specific readout.

    This differs from ClassAwareEvidenceHead: classes do not own patch
    attention queries. A small shared token bank first extracts evidence
    modes from the bag, then each class learns a low-rank combination of
    those shared modes. This keeps the evidence retrieval dataset-agnostic
    while allowing class boundaries to use different mixtures of evidence.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_tokens,
        token_dim,
        value_dim,
        rank_dim,
        temperature,
        dropout,
    ):
        super().__init__()
        if num_tokens <= 0:
            raise ValueError("num_tokens must be positive")
        if min(token_dim, value_dim, rank_dim) <= 0:
            raise ValueError(
                "token_dim, value_dim, and rank_dim must be positive"
            )
        if temperature <= 0:
            raise ValueError("class-token temperature must be positive")
        self.num_classes = int(num_classes)
        self.num_tokens = int(num_tokens)
        self.token_dim = int(token_dim)
        self.value_dim = int(value_dim)
        self.rank_dim = int(rank_dim)
        self.temperature = float(temperature)
        self.key = nn.Linear(hidden_dim, self.token_dim)
        self.value = nn.Linear(hidden_dim, self.value_dim)
        self.tokens = nn.Parameter(
            torch.empty(self.num_tokens, self.token_dim)
        )
        self.token_projector = nn.Sequential(
            nn.LayerNorm(self.value_dim),
            nn.Dropout(dropout),
            nn.Linear(self.value_dim, self.rank_dim),
        )
        self.class_factors = nn.Parameter(
            torch.empty(
                self.num_classes,
                self.num_tokens,
                self.rank_dim,
            )
        )
        self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.tokens, std=1.0 / math.sqrt(self.token_dim)
            )
            nn.init.normal_(
                self.class_factors,
                std=1.0 / math.sqrt(self.num_tokens * self.rank_dim),
            )

    def forward(self, encoded):
        keys = self.key(encoded)
        values = self.value(encoded)
        scale = math.sqrt(self.token_dim) * self.temperature
        scores = torch.einsum("nd,td->nt", keys, self.tokens) / scale
        attention = torch.softmax(scores, dim=0)
        token_values = torch.einsum("nt,nv->tv", attention, values)
        token_evidence = self.token_projector(token_values)
        logits = torch.einsum(
            "tr,ctr->c", token_evidence, self.class_factors
        ) + self.logit_bias
        return logits.unsqueeze(0)


class SparseClassEvidenceReadout(nn.Module):
    """Class-specific multi-query readout for rare local evidence.

    Soft summaries preserve distributed evidence and hard top-k summaries
    retain small diagnostic regions. The global MIR state gates each class,
    so local evidence is interpreted in whole-slide context.
    """

    def __init__(
        self,
        hidden_dim,
        state_dim,
        num_classes,
        num_queries,
        query_dim,
        value_dim,
        rank_dim,
        gate_hidden_dim,
        temperature,
        topk_fraction,
        dropout,
        gate_initial_bias,
    ):
        super().__init__()
        if num_queries <= 0:
            raise ValueError("sparse-class num_queries must be positive")
        if min(query_dim, value_dim, rank_dim, gate_hidden_dim) <= 0:
            raise ValueError("sparse-class dimensions must be positive")
        if temperature <= 0:
            raise ValueError("sparse-class temperature must be positive")
        if not 0 < topk_fraction <= 1:
            raise ValueError(
                "sparse-class topk_fraction must be in (0, 1]"
            )
        if not 0 <= dropout < 1:
            raise ValueError("sparse-class dropout must be in [0, 1)")

        self.num_classes = int(num_classes)
        self.num_queries = int(num_queries)
        self.query_dim = int(query_dim)
        self.value_dim = int(value_dim)
        self.temperature = float(temperature)
        self.topk_fraction = float(topk_fraction)
        self.key = nn.Linear(hidden_dim, self.query_dim)
        self.value = nn.Linear(hidden_dim, self.value_dim)
        self.queries = nn.Parameter(
            torch.empty(
                self.num_classes, self.num_queries, self.query_dim
            )
        )
        self.evidence_projector = nn.Sequential(
            nn.LayerNorm(2 * self.value_dim),
            nn.Dropout(dropout),
            nn.Linear(2 * self.value_dim, int(rank_dim)),
            nn.GELU(),
        )
        self.class_factors = nn.Parameter(
            torch.empty(self.num_classes, self.num_queries, int(rank_dim))
        )
        self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        self.context_gate = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, int(gate_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(gate_hidden_dim), self.num_classes),
        )
        self.gate_initial_bias = float(gate_initial_bias)
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.queries, std=1.0 / math.sqrt(self.query_dim)
            )
            nn.init.normal_(
                self.class_factors,
                std=1.0 / math.sqrt(self.num_queries * int(rank_dim)),
            )

    def reset_gate(self):
        final_layer = self.context_gate[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.constant_(final_layer.bias, self.gate_initial_bias)

    def forward(self, encoded, state):
        keys = self.key(encoded)
        values = self.value(encoded)
        scale = math.sqrt(self.query_dim) * self.temperature
        scores = torch.einsum("nd,ctd->nct", keys, self.queries) / scale

        attention = torch.softmax(scores, dim=0)
        soft_evidence = torch.einsum(
            "nct,nv->ctv", attention, values
        )
        k = max(1, math.ceil(encoded.shape[0] * self.topk_fraction))
        k = min(k, encoded.shape[0])
        top_indices = torch.topk(scores, k=k, dim=0).indices
        sparse_evidence = values[top_indices].mean(dim=0)

        statistics = torch.cat((soft_evidence, sparse_evidence), dim=-1)
        evidence = self.evidence_projector(statistics)
        raw_logits = torch.einsum(
            "ctr,ctr->c", evidence, self.class_factors
        ) + self.logit_bias
        gate = torch.sigmoid(self.context_gate(state.unsqueeze(0))).squeeze(0)
        return (gate * raw_logits).unsqueeze(0)


class FocusedSparseEvidenceReadout(nn.Module):
    """Patch-level mixture readout for one difficult class."""

    def __init__(
        self,
        hidden_dim,
        state_dim,
        num_queries,
        query_dim,
        value_dim,
        readout_dim,
        temperature,
        topk_fraction,
        dropout,
    ):
        super().__init__()
        if num_queries <= 0:
            raise ValueError("focus-sparse num_queries must be positive")
        if min(query_dim, value_dim, readout_dim) <= 0:
            raise ValueError("focus-sparse dimensions must be positive")
        if temperature <= 0:
            raise ValueError("focus-sparse temperature must be positive")
        if not 0 < topk_fraction <= 1:
            raise ValueError("focus-sparse topk_fraction must be in (0, 1]")
        if not 0 <= dropout < 1:
            raise ValueError("focus-sparse dropout must be in [0, 1)")

        self.num_queries = int(num_queries)
        self.query_dim = int(query_dim)
        self.value_dim = int(value_dim)
        self.temperature = float(temperature)
        self.topk_fraction = float(topk_fraction)
        self.key = nn.Linear(hidden_dim, self.query_dim)
        self.value = nn.Linear(hidden_dim, self.value_dim)
        self.queries = nn.Parameter(
            torch.empty(self.num_queries, self.query_dim)
        )
        self.state_projector = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, self.value_dim),
            nn.GELU(),
        )
        self.evidence_projector = nn.Sequential(
            nn.LayerNorm(3 * self.value_dim),
            nn.Dropout(dropout),
            nn.Linear(3 * self.value_dim, int(readout_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query_scorer = nn.Linear(int(readout_dim), 1)
        self.context_mixer = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, int(readout_dim)),
            nn.GELU(),
            nn.Linear(int(readout_dim), self.num_queries),
        )
        self.logit_bias = nn.Parameter(torch.zeros(1))
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(self.queries, std=1.0 / math.sqrt(self.query_dim))

    def reset_mixer(self):
        final_layer = self.context_mixer[-1]
        nn.init.zeros_(final_layer.weight)
        nn.init.zeros_(final_layer.bias)

    def forward(self, encoded, state):
        keys = self.key(encoded)
        values = self.value(encoded)
        scale = math.sqrt(self.query_dim) * self.temperature
        scores = torch.einsum("nd,qd->nq", keys, self.queries) / scale

        attention = torch.softmax(scores, dim=0)
        soft_evidence = torch.einsum("nq,nv->qv", attention, values)
        k = max(1, math.ceil(encoded.shape[0] * self.topk_fraction))
        k = min(k, encoded.shape[0])
        top_indices = torch.topk(scores, k=k, dim=0).indices
        sparse_evidence = values[top_indices].mean(dim=0)
        global_evidence = values.mean(dim=0, keepdim=True)
        state_context = self.state_projector(state.unsqueeze(0))
        statistics = torch.cat(
            (
                soft_evidence,
                sparse_evidence - global_evidence,
                state_context.expand(self.num_queries, -1),
            ),
            dim=-1,
        )
        query_logits = self.query_scorer(
            self.evidence_projector(statistics)
        ).squeeze(-1)
        mixture = torch.softmax(
            self.context_mixer(state.unsqueeze(0)), dim=1
        ).squeeze(0)
        logit = torch.sum(mixture * query_logits) + self.logit_bias.squeeze(0)
        return logit.reshape(1, 1)


class GatedAttentionResidualReadout(nn.Module):
    """ABMIL-style gated attention residual over encoded patches."""

    def __init__(
        self,
        hidden_dim,
        num_classes,
        attention_dim,
        value_dim,
        dropout,
        temperature,
        class_specific,
    ):
        super().__init__()
        if min(attention_dim, value_dim) <= 0:
            raise ValueError("gated attention dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("gated attention dropout must be in [0, 1)")
        if temperature <= 0:
            raise ValueError("gated attention temperature must be positive")
        self.num_classes = int(num_classes)
        self.attention_dim = int(attention_dim)
        self.value_dim = int(value_dim)
        self.temperature = float(temperature)
        self.class_specific = bool(class_specific)
        self.attention_v = nn.Sequential(
            nn.Linear(hidden_dim, self.attention_dim),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(hidden_dim, self.attention_dim),
            nn.Sigmoid(),
        )
        self.attention_dropout = nn.Dropout(dropout)
        attention_out = self.num_classes if self.class_specific else 1
        self.attention_score = nn.Linear(self.attention_dim, attention_out)
        self.value = nn.Sequential(
            nn.Linear(hidden_dim, self.value_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        if self.class_specific:
            self.classifiers = nn.Parameter(
                torch.empty(self.num_classes, self.value_dim)
            )
            self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
            with torch.random.fork_rng(devices=[]):
                nn.init.normal_(
                    self.classifiers, std=1.0 / math.sqrt(self.value_dim)
                )
        else:
            self.classifier = nn.Linear(self.value_dim, self.num_classes)

    def forward(self, encoded):
        gated = self.attention_v(encoded) * self.attention_u(encoded)
        gated = self.attention_dropout(gated)
        scores = self.attention_score(gated) / self.temperature
        values = self.value(encoded)
        if self.class_specific:
            attention = torch.softmax(scores, dim=0)
            pooled = torch.einsum("nc,nv->cv", attention, values)
            logits = torch.einsum(
                "cv,cv->c", pooled, self.classifiers
            ) + self.logit_bias
            return logits.unsqueeze(0)
        attention = torch.softmax(scores.squeeze(-1), dim=0)
        pooled = torch.sum(attention[:, None] * values, dim=0)
        return self.classifier(pooled.unsqueeze(0))


class SpatialRegionMomentReadout(nn.Module):
    """Hierarchical region readout over features and true patch coordinates."""

    def __init__(
        self,
        hidden_dim,
        num_classes,
        grid_size,
        value_dim,
        region_dim,
        attention_dim,
        dropout,
        temperature,
        include_centers=True,
        include_mass=False,
    ):
        super().__init__()
        if grid_size <= 0:
            raise ValueError("spatial region grid_size must be positive")
        if min(value_dim, region_dim, attention_dim) <= 0:
            raise ValueError("spatial region dimensions must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("spatial region dropout must be in [0, 1)")
        if temperature <= 0:
            raise ValueError("spatial region temperature must be positive")
        self.num_classes = int(num_classes)
        self.grid_size = int(grid_size)
        self.value_dim = int(value_dim)
        self.temperature = float(temperature)
        self.include_centers = bool(include_centers)
        self.include_mass = bool(include_mass)
        self.value = nn.Linear(hidden_dim, self.value_dim)
        statistic_dim = 2 * self.value_dim
        statistic_dim += 2 if self.include_centers else 0
        statistic_dim += 1 if self.include_mass else 0
        self.region_projector = nn.Sequential(
            nn.LayerNorm(statistic_dim),
            nn.Linear(statistic_dim, int(region_dim)),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_v = nn.Sequential(
            nn.Linear(int(region_dim), int(attention_dim)),
            nn.Tanh(),
        )
        self.attention_u = nn.Sequential(
            nn.Linear(int(region_dim), int(attention_dim)),
            nn.Sigmoid(),
        )
        self.attention_score = nn.Linear(
            int(attention_dim), self.num_classes
        )
        self.classifiers = nn.Parameter(
            torch.empty(self.num_classes, int(region_dim))
        )
        self.logit_bias = nn.Parameter(torch.zeros(self.num_classes))
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.classifiers, std=1.0 / math.sqrt(int(region_dim))
            )

    def region_statistics(self, encoded, coordinates):
        if coordinates.ndim != 2 or coordinates.shape != (
            encoded.shape[0],
            2,
        ):
            raise ValueError("coordinates must have shape [N, 2]")
        coordinates = coordinates.to(dtype=encoded.dtype)
        if coordinates.amin() >= 0 and coordinates.amax() <= 1:
            normalized = coordinates
        else:
            minimum = coordinates.amin(dim=0)
            span = (coordinates.amax(dim=0) - minimum).clamp_min(1e-6)
            normalized = (coordinates - minimum) / span
        normalized = normalized.clamp(0.0, 1.0)
        bins = torch.floor(normalized * self.grid_size).long()
        bins = bins.clamp(max=self.grid_size - 1)
        region_ids = bins[:, 1] * self.grid_size + bins[:, 0]
        occupied, inverse = torch.unique(
            region_ids, sorted=True, return_inverse=True
        )
        del occupied

        values = self.value(encoded)
        region_count = int(inverse.max().item()) + 1
        counts = values.new_zeros((region_count, 1))
        counts.index_add_(
            0, inverse, values.new_ones((values.shape[0], 1))
        )
        sums = values.new_zeros((region_count, self.value_dim))
        sums.index_add_(0, inverse, values)
        square_sums = values.new_zeros((region_count, self.value_dim))
        square_sums.index_add_(0, inverse, values.square())
        coordinate_sums = values.new_zeros((region_count, 2))
        coordinate_sums.index_add_(0, inverse, normalized)

        mean = sums / counts
        variance = (square_sums / counts - mean.square()).clamp_min(0.0)
        statistics = [mean, variance]
        if self.include_centers:
            statistics.append(coordinate_sums / counts)
        if self.include_mass:
            denominator = math.log1p(values.shape[0])
            statistics.append(torch.log1p(counts) / denominator)
        return torch.cat(statistics, dim=1)

    def forward(self, encoded, coordinates):
        statistics = self.region_statistics(encoded, coordinates)
        regions = self.region_projector(statistics)
        gated = self.attention_v(regions) * self.attention_u(regions)
        scores = self.attention_score(gated) / self.temperature
        attention = torch.softmax(scores, dim=0)
        pooled = torch.einsum("rc,rd->cd", attention, regions)
        logits = torch.einsum(
            "cd,cd->c", pooled, self.classifiers
        ) + self.logit_bias
        return logits.unsqueeze(0)


class LatentEvidenceTransformerReadout(nn.Module):
    """Perceiver-style latent evidence readout over encoded patches.

    Latent tokens cross-attend to patches in O(NK), then interact through
    self-attention over only K latent tokens before classification. This
    keeps the branch practical for large WSI bags while allowing retrieved
    evidence modes to condition each other.
    """

    def __init__(
        self,
        hidden_dim,
        num_classes,
        num_latents,
        latent_dim,
        num_heads,
        num_layers,
        mlp_ratio,
        temperature,
        dropout,
    ):
        super().__init__()
        if num_latents <= 0:
            raise ValueError("num_latents must be positive")
        if latent_dim <= 0:
            raise ValueError("latent_dim must be positive")
        if num_heads <= 0 or latent_dim % num_heads:
            raise ValueError(
                "num_heads must be positive and divide latent_dim"
            )
        if num_layers < 0:
            raise ValueError("num_layers must be non-negative")
        if mlp_ratio <= 0:
            raise ValueError("mlp_ratio must be positive")
        if temperature <= 0:
            raise ValueError("latent readout temperature must be positive")
        self.num_latents = int(num_latents)
        self.latent_dim = int(latent_dim)
        self.temperature = float(temperature)
        self.key = nn.Linear(hidden_dim, self.latent_dim)
        self.value = nn.Linear(hidden_dim, self.latent_dim)
        self.latents = nn.Parameter(
            torch.empty(self.num_latents, self.latent_dim)
        )
        self.blocks = nn.ModuleList()
        ffn_dim = max(int(round(self.latent_dim * mlp_ratio)), 1)
        for _ in range(int(num_layers)):
            self.blocks.append(
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(self.latent_dim),
                        "attn": nn.MultiheadAttention(
                            embed_dim=self.latent_dim,
                            num_heads=int(num_heads),
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm2": nn.LayerNorm(self.latent_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(self.latent_dim, ffn_dim),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(ffn_dim, self.latent_dim),
                            nn.Dropout(dropout),
                        ),
                    }
                )
            )
        self.output = nn.Sequential(
            nn.LayerNorm(self.latent_dim * self.num_latents),
            nn.Dropout(dropout),
            nn.Linear(self.latent_dim * self.num_latents, num_classes),
        )
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.latents, std=1.0 / math.sqrt(self.latent_dim)
            )

    def forward(self, encoded):
        keys = self.key(encoded)
        values = self.value(encoded)
        scale = math.sqrt(self.latent_dim) * self.temperature
        scores = torch.einsum("nd,kd->nk", keys, self.latents) / scale
        attention = torch.softmax(scores, dim=0)
        latent_values = torch.einsum("nk,nd->kd", attention, values)
        latent_values = latent_values.unsqueeze(0)
        for block in self.blocks:
            normalized = block["norm1"](latent_values)
            attended, _ = block["attn"](
                normalized,
                normalized,
                normalized,
                need_weights=False,
            )
            latent_values = latent_values + attended
            latent_values = latent_values + block["ffn"](
                block["norm2"](latent_values)
            )
        return self.output(latent_values.flatten(start_dim=1))


class OrdinalCalibrationHead(nn.Module):
    """Monotonic-threshold ordinal residual over the MIR-MIL slide state."""

    def __init__(
        self,
        state_dim,
        num_classes,
        hidden_dim,
        dropout,
        act,
        temperature,
    ):
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be >= 2")
        if state_dim <= 0 or hidden_dim <= 0:
            raise ValueError("state_dim and hidden_dim must be positive")
        if temperature <= 0:
            raise ValueError("ordinal_head_temperature must be positive")
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.score = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        threshold_count = self.num_classes - 1
        initial_increment = math.log(math.exp(1.0) - 1.0)
        self.raw_threshold_increments = nn.Parameter(
            torch.full((threshold_count,), initial_increment)
        )
        self.threshold_shift = nn.Parameter(torch.zeros(()))

    def thresholds(self):
        increments = F.softplus(self.raw_threshold_increments)
        thresholds = torch.cumsum(increments, dim=0)
        thresholds = thresholds - thresholds.mean()
        return thresholds + self.threshold_shift

    def forward(self, state):
        score = self.score(state) / self.temperature
        thresholds = self.thresholds().to(dtype=state.dtype)
        survival = torch.sigmoid(score - thresholds.unsqueeze(0))
        eps = torch.finfo(state.dtype).eps
        probabilities = []
        probabilities.append((1.0 - survival[:, :1]).clamp_min(eps))
        if self.num_classes > 2:
            probabilities.append(
                (survival[:, :-1] - survival[:, 1:]).clamp_min(eps)
            )
        probabilities.append(survival[:, -1:].clamp_min(eps))
        class_probabilities = torch.cat(probabilities, dim=1)
        class_probabilities = class_probabilities / (
            class_probabilities.sum(dim=1, keepdim=True).clamp_min(eps)
        )
        logits = torch.log(class_probabilities)
        return logits - logits.mean(dim=1, keepdim=True)


class CosineStateResidualHead(nn.Module):
    """Normalized cosine classifier over the MIR-MIL slide state."""

    def __init__(
        self,
        state_dim,
        num_classes,
        embedding_dim,
        hidden_dim,
        dropout,
        act,
        initial_scale,
    ):
        super().__init__()
        if state_dim <= 0 or num_classes < 2:
            raise ValueError("state_dim must be positive and classes >= 2")
        if embedding_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                "embedding_dim and hidden_dim must be positive"
            )
        if initial_scale <= 0:
            raise ValueError("cosine_head_initial_scale must be positive")
        self.projector = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.class_prototypes = nn.Parameter(
            torch.empty(num_classes, embedding_dim)
        )
        self.log_scale = nn.Parameter(
            torch.tensor(math.log(float(initial_scale)))
        )
        with torch.random.fork_rng(devices=[]):
            nn.init.normal_(
                self.class_prototypes, std=1.0 / math.sqrt(embedding_dim)
            )

    def forward(self, state):
        embedding = F.normalize(self.projector(state), dim=-1)
        prototypes = F.normalize(self.class_prototypes, dim=-1)
        logits = embedding @ prototypes.T
        scale = self.log_scale.exp().clamp(max=100.0)
        return scale * logits


class StateBoundaryHead(nn.Module):
    """Small state head for explicit one-vs-rest or threshold supervision."""

    def __init__(
        self,
        state_dim,
        output_dim,
        hidden_dim,
        dropout,
        act,
        temperature,
    ):
        super().__init__()
        if state_dim <= 0 or output_dim <= 0 or hidden_dim <= 0:
            raise ValueError(
                "state_dim, output_dim, and hidden_dim must be positive"
            )
        if temperature <= 0:
            raise ValueError("boundary head temperature must be positive")
        self.temperature = float(temperature)
        self.net = nn.Sequential(
            nn.LayerNorm(state_dim),
            nn.Linear(state_dim, hidden_dim),
            _activation(act),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, state):
        return self.net(state) / self.temperature


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
        num_local_routes=0,
        local_route_dim=32,
        local_route_temperature=0.25,
        anchor_route_dim=0,
        anchor_route_temperature=1.0,
        anchor_route_identity=False,
        potential_hidden_dim=128,
        dropout=0.1,
        act="gelu",
        coordinate_dim=0,
        coordinate_encoder_scale=1.0,
        input_group_l2_normalize=False,
        input_group_size=0,
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
        multiscale_gate_initial_bias=-2.0,
        multiscale_local_initial_scale=0.1,
        multiscale_class_mix_initial=0.5,
        multiscale_class_residual_initial_scale=0.05,
        multiscale_prototype_initial_scale=0.05,
        anchor_global_initial_scale=0.1,
        anchor_local_initial_scale=0.1,
        evidence_weight=0.0,
        evidence_query_dim=64,
        evidence_value_dim=128,
        evidence_temperature=1.0,
        evidence_dropout=0.0,
        multi_token_weight=0.0,
        multi_token_count=4,
        multi_token_dim=64,
        multi_token_readout_dim=128,
        multi_token_temperature=1.0,
        multi_token_dropout=0.0,
        multi_token_gated=False,
        multi_token_gate_hidden_dim=64,
        multi_token_gate_initial_bias=0.0,
        moment_token_weight=0.0,
        moment_token_count=4,
        moment_token_dim=64,
        moment_token_readout_dim=128,
        moment_token_temperature=1.0,
        moment_token_dropout=0.0,
        tail_token_weight=0.0,
        tail_token_count=4,
        tail_token_dim=64,
        tail_token_readout_dim=128,
        tail_token_temperature=1.0,
        tail_token_topk_fraction=0.05,
        tail_token_dropout=0.0,
        class_moment_token_weight=0.0,
        class_moment_token_count=4,
        class_moment_token_dim=64,
        class_moment_token_value_dim=128,
        class_moment_token_rank_dim=32,
        class_moment_token_temperature=1.0,
        class_moment_token_dropout=0.0,
        residual_class_moment_token_weight=0.0,
        residual_class_moment_token_count=4,
        residual_class_moment_token_dim=64,
        residual_class_moment_token_readout_dim=128,
        residual_class_moment_token_rank_dim=32,
        residual_class_moment_token_temperature=1.0,
        residual_class_moment_token_dropout=0.0,
        residual_class_moment_token_initial_scale=0.05,
        class_token_weight=0.0,
        class_token_count=4,
        class_token_dim=64,
        class_token_value_dim=128,
        class_token_rank_dim=32,
        class_token_temperature=1.0,
        class_token_dropout=0.0,
        sparse_class_weight=0.0,
        sparse_class_query_count=4,
        sparse_class_query_dim=64,
        sparse_class_value_dim=128,
        sparse_class_rank_dim=32,
        sparse_class_gate_hidden_dim=64,
        sparse_class_temperature=1.0,
        sparse_class_topk_fraction=0.02,
        sparse_class_dropout=0.0,
        sparse_class_gate_initial_bias=0.0,
        gated_attention_weight=0.0,
        gated_attention_dim=128,
        gated_attention_value_dim=128,
        gated_attention_dropout=0.0,
        gated_attention_temperature=1.0,
        gated_attention_class_specific=True,
        spatial_region_weight=0.0,
        spatial_region_grid_size=4,
        spatial_region_value_dim=64,
        spatial_region_dim=96,
        spatial_region_attention_dim=64,
        spatial_region_dropout=0.0,
        spatial_region_temperature=1.0,
        spatial_region_include_centers=True,
        spatial_region_include_mass=False,
        latent_readout_weight=0.0,
        latent_readout_count=4,
        latent_readout_dim=128,
        latent_readout_heads=4,
        latent_readout_layers=1,
        latent_readout_mlp_ratio=2.0,
        latent_readout_temperature=1.0,
        latent_readout_dropout=0.0,
        ordinal_head_weight=0.0,
        ordinal_head_hidden_dim=64,
        ordinal_head_dropout=0.0,
        ordinal_head_temperature=1.0,
        cosine_head_weight=0.0,
        cosine_head_dim=64,
        cosine_head_hidden_dim=128,
        cosine_head_dropout=0.0,
        cosine_head_initial_scale=8.0,
        ovr_head_weight=0.0,
        ovr_loss_weight=0.0,
        ovr_loss_pos_weight=1.0,
        ovr_loss_pos_weights=None,
        ovr_head_hidden_dim=64,
        ovr_head_dropout=0.0,
        ovr_head_temperature=1.0,
        adjacent_head_weight=0.0,
        adjacent_loss_weight=0.0,
        adjacent_head_hidden_dim=64,
        adjacent_head_dropout=0.0,
        adjacent_head_temperature=1.0,
        focus_class_index=1,
        focus_class_head_weight=0.0,
        focus_class_loss_weight=0.0,
        focus_class_loss_pos_weight=1.0,
        focus_class_head_hidden_dim=64,
        focus_class_head_dropout=0.0,
        focus_class_head_temperature=1.0,
        focus_sparse_head_weight=0.0,
        focus_sparse_loss_weight=0.0,
        focus_sparse_loss_pos_weight=1.0,
        focus_sparse_query_count=4,
        focus_sparse_query_dim=64,
        focus_sparse_value_dim=128,
        focus_sparse_readout_dim=64,
        focus_sparse_temperature=1.0,
        focus_sparse_topk_fraction=0.02,
        focus_sparse_dropout=0.0,
        logit_margin_loss_weight=0.0,
        logit_margin=1.0,
        subset_consistency_weight=0.0,
        subset_consistency_supervised_weight=0.0,
        subset_consistency_fraction=0.75,
        subset_consistency_views=1,
        subset_consistency_temperature=1.0,
        use_logit_calibration=False,
        logit_calibration_learn_temperature=False,
        logit_calibration_initial_temperature=1.0,
        logit_calibration_bias_init=None,
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
        self.coordinate_encoder_scale = float(coordinate_encoder_scale)
        if self.coordinate_encoder_scale < 0:
            raise ValueError("coordinate_encoder_scale must be non-negative")
        self.input_dim = self.in_dim + self.coordinate_dim
        self.input_group_l2_normalize = bool(input_group_l2_normalize)
        self.input_group_size = int(input_group_size)
        if self.input_group_l2_normalize and (
            self.input_group_size <= 0
            or self.in_dim % self.input_group_size != 0
        ):
            raise ValueError(
                "input_group_size must be positive and divide in_dim when "
                "input_group_l2_normalize is enabled"
            )
        self.num_classes = int(num_classes)
        self.sketch_dim = int(sketch_dim)
        self.moment_order = int(moment_order)
        if self.moment_order not in {1, 2}:
            raise ValueError("moment_order must be 1 or 2")
        self.composition_state_dim = self.sketch_dim * self.moment_order
        self.num_tail_scores = int(num_tail_scores)
        self.tail_temperature = float(tail_temperature)
        self.num_local_routes = int(num_local_routes)
        self.local_route_dim = int(local_route_dim)
        self.local_route_temperature = float(local_route_temperature)
        self.anchor_route_dim = int(anchor_route_dim)
        self.anchor_route_temperature = float(anchor_route_temperature)
        self.anchor_route_identity = bool(anchor_route_identity)
        if self.num_local_routes < 0 or self.local_route_dim <= 0:
            raise ValueError(
                "num_local_routes must be non-negative and "
                "local_route_dim must be positive"
            )
        if self.local_route_temperature <= 0:
            raise ValueError("local_route_temperature must be positive")
        if self.anchor_route_dim < 0:
            raise ValueError("anchor_route_dim must be non-negative")
        if self.anchor_route_temperature <= 0:
            raise ValueError("anchor_route_temperature must be positive")
        if self.anchor_route_identity and (
            self.anchor_route_dim != hidden_dim
        ):
            raise ValueError(
                "identity anchor dimension must equal hidden_dim"
            )
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
        self.use_logit_calibration = bool(use_logit_calibration)
        self.logit_calibration_learn_temperature = bool(
            logit_calibration_learn_temperature
        )
        self.logit_calibration_initial_temperature = float(
            logit_calibration_initial_temperature
        )
        if self.logit_calibration_initial_temperature <= 0:
            raise ValueError(
                "logit_calibration_initial_temperature must be positive"
            )
        if logit_calibration_bias_init is None:
            logit_calibration_bias_init = [0.0] * self.num_classes
        if (
            len(logit_calibration_bias_init) != self.num_classes
            and self.use_logit_calibration
        ):
            raise ValueError(
                "logit_calibration_bias_init must have one value per class"
            )
        if len(logit_calibration_bias_init) != self.num_classes:
            logit_calibration_bias_init = [0.0] * self.num_classes
        calibration_bias = torch.tensor(
            logit_calibration_bias_init, dtype=torch.float32
        )
        if self.use_logit_calibration:
            self.logit_calibration_bias = nn.Parameter(calibration_bias)
            if self.logit_calibration_learn_temperature:
                self.logit_calibration_log_temperature = nn.Parameter(
                    torch.tensor(
                        math.log(self.logit_calibration_initial_temperature),
                        dtype=torch.float32,
                    )
                )
            else:
                self.register_buffer(
                    "logit_calibration_log_temperature",
                    torch.tensor(
                        math.log(self.logit_calibration_initial_temperature),
                        dtype=torch.float32,
                    ),
                )
        else:
            self.register_buffer(
                "logit_calibration_bias", calibration_bias
            )
            self.register_buffer(
                "logit_calibration_log_temperature",
                torch.tensor(0.0, dtype=torch.float32),
            )
        self.potential_type = str(potential_type)
        self.prototype_regularization_weight = float(
            prototype_regularization_weight
        )
        if self.prototype_regularization_weight < 0:
            raise ValueError(
                "prototype_regularization_weight must be non-negative"
            )
        self.evidence_weight = float(evidence_weight)
        if self.evidence_weight < 0:
            raise ValueError("evidence_weight must be non-negative")
        self.multi_token_weight = float(multi_token_weight)
        self.multi_token_count = int(multi_token_count)
        self.multi_token_dim = int(multi_token_dim)
        self.multi_token_readout_dim = int(multi_token_readout_dim)
        self.multi_token_temperature = float(multi_token_temperature)
        self.multi_token_dropout = float(multi_token_dropout)
        self.multi_token_gated = bool(multi_token_gated)
        self.multi_token_gate_hidden_dim = int(multi_token_gate_hidden_dim)
        self.multi_token_gate_initial_bias = float(
            multi_token_gate_initial_bias
        )
        if self.multi_token_weight < 0:
            raise ValueError("multi_token_weight must be non-negative")
        if self.multi_token_count <= 0:
            raise ValueError("multi_token_count must be positive")
        if self.multi_token_dim <= 0 or self.multi_token_readout_dim <= 0:
            raise ValueError(
                "multi_token_dim and multi_token_readout_dim must be "
                "positive"
            )
        if self.multi_token_temperature <= 0:
            raise ValueError("multi_token_temperature must be positive")
        if not 0 <= self.multi_token_dropout < 1:
            raise ValueError("multi_token_dropout must be in [0, 1)")
        if self.multi_token_gate_hidden_dim <= 0:
            raise ValueError(
                "multi_token_gate_hidden_dim must be positive"
            )
        self.moment_token_weight = float(moment_token_weight)
        self.moment_token_count = int(moment_token_count)
        self.moment_token_dim = int(moment_token_dim)
        self.moment_token_readout_dim = int(moment_token_readout_dim)
        self.moment_token_temperature = float(moment_token_temperature)
        self.moment_token_dropout = float(moment_token_dropout)
        if self.moment_token_weight < 0:
            raise ValueError("moment_token_weight must be non-negative")
        if self.moment_token_count <= 0:
            raise ValueError("moment_token_count must be positive")
        if (
            self.moment_token_dim <= 0
            or self.moment_token_readout_dim <= 0
        ):
            raise ValueError(
                "moment_token_dim and moment_token_readout_dim must be "
                "positive"
            )
        if self.moment_token_temperature <= 0:
            raise ValueError("moment_token_temperature must be positive")
        if not 0 <= self.moment_token_dropout < 1:
            raise ValueError("moment_token_dropout must be in [0, 1)")
        self.tail_token_weight = float(tail_token_weight)
        self.tail_token_count = int(tail_token_count)
        self.tail_token_dim = int(tail_token_dim)
        self.tail_token_readout_dim = int(tail_token_readout_dim)
        self.tail_token_temperature = float(tail_token_temperature)
        self.tail_token_topk_fraction = float(tail_token_topk_fraction)
        self.tail_token_dropout = float(tail_token_dropout)
        if self.tail_token_weight < 0:
            raise ValueError("tail_token_weight must be non-negative")
        if self.tail_token_count <= 0:
            raise ValueError("tail_token_count must be positive")
        if self.tail_token_dim <= 0 or self.tail_token_readout_dim <= 0:
            raise ValueError(
                "tail_token_dim and tail_token_readout_dim must be positive"
            )
        if self.tail_token_temperature <= 0:
            raise ValueError("tail_token_temperature must be positive")
        if not 0 < self.tail_token_topk_fraction <= 1:
            raise ValueError(
                "tail_token_topk_fraction must be in (0, 1]"
            )
        if not 0 <= self.tail_token_dropout < 1:
            raise ValueError("tail_token_dropout must be in [0, 1)")
        self.class_moment_token_weight = float(
            class_moment_token_weight
        )
        self.class_moment_token_count = int(class_moment_token_count)
        self.class_moment_token_dim = int(class_moment_token_dim)
        self.class_moment_token_value_dim = int(
            class_moment_token_value_dim
        )
        self.class_moment_token_rank_dim = int(
            class_moment_token_rank_dim
        )
        self.class_moment_token_temperature = float(
            class_moment_token_temperature
        )
        self.class_moment_token_dropout = float(
            class_moment_token_dropout
        )
        if self.class_moment_token_weight < 0:
            raise ValueError(
                "class_moment_token_weight must be non-negative"
            )
        if self.class_moment_token_count <= 0:
            raise ValueError("class_moment_token_count must be positive")
        if min(
            self.class_moment_token_dim,
            self.class_moment_token_value_dim,
            self.class_moment_token_rank_dim,
        ) <= 0:
            raise ValueError(
                "class_moment_token_dim, class_moment_token_value_dim, "
                "and class_moment_token_rank_dim must be positive"
            )
        if self.class_moment_token_temperature <= 0:
            raise ValueError(
                "class_moment_token_temperature must be positive"
            )
        if not 0 <= self.class_moment_token_dropout < 1:
            raise ValueError(
                "class_moment_token_dropout must be in [0, 1)"
            )
        self.residual_class_moment_token_weight = float(
            residual_class_moment_token_weight
        )
        self.residual_class_moment_token_count = int(
            residual_class_moment_token_count
        )
        self.residual_class_moment_token_dim = int(
            residual_class_moment_token_dim
        )
        self.residual_class_moment_token_readout_dim = int(
            residual_class_moment_token_readout_dim
        )
        self.residual_class_moment_token_rank_dim = int(
            residual_class_moment_token_rank_dim
        )
        self.residual_class_moment_token_temperature = float(
            residual_class_moment_token_temperature
        )
        self.residual_class_moment_token_dropout = float(
            residual_class_moment_token_dropout
        )
        self.residual_class_moment_token_initial_scale = float(
            residual_class_moment_token_initial_scale
        )
        if self.residual_class_moment_token_weight < 0:
            raise ValueError(
                "residual_class_moment_token_weight must be non-negative"
            )
        if self.residual_class_moment_token_count <= 0:
            raise ValueError(
                "residual_class_moment_token_count must be positive"
            )
        if min(
            self.residual_class_moment_token_dim,
            self.residual_class_moment_token_readout_dim,
            self.residual_class_moment_token_rank_dim,
        ) <= 0:
            raise ValueError(
                "residual_class_moment_token dimensions must be positive"
            )
        if self.residual_class_moment_token_temperature <= 0:
            raise ValueError(
                "residual_class_moment_token_temperature must be positive"
            )
        if not 0 <= self.residual_class_moment_token_dropout < 1:
            raise ValueError(
                "residual_class_moment_token_dropout must be in [0, 1)"
            )
        self.class_token_weight = float(class_token_weight)
        self.class_token_count = int(class_token_count)
        self.class_token_dim = int(class_token_dim)
        self.class_token_value_dim = int(class_token_value_dim)
        self.class_token_rank_dim = int(class_token_rank_dim)
        self.class_token_temperature = float(class_token_temperature)
        self.class_token_dropout = float(class_token_dropout)
        if self.class_token_weight < 0:
            raise ValueError("class_token_weight must be non-negative")
        if self.class_token_count <= 0:
            raise ValueError("class_token_count must be positive")
        if min(
            self.class_token_dim,
            self.class_token_value_dim,
            self.class_token_rank_dim,
        ) <= 0:
            raise ValueError(
                "class_token_dim, class_token_value_dim, and "
                "class_token_rank_dim must be positive"
            )
        if self.class_token_temperature <= 0:
            raise ValueError("class_token_temperature must be positive")
        if not 0 <= self.class_token_dropout < 1:
            raise ValueError("class_token_dropout must be in [0, 1)")
        self.sparse_class_weight = float(sparse_class_weight)
        self.sparse_class_query_count = int(sparse_class_query_count)
        self.sparse_class_query_dim = int(sparse_class_query_dim)
        self.sparse_class_value_dim = int(sparse_class_value_dim)
        self.sparse_class_rank_dim = int(sparse_class_rank_dim)
        self.sparse_class_gate_hidden_dim = int(
            sparse_class_gate_hidden_dim
        )
        self.sparse_class_temperature = float(sparse_class_temperature)
        self.sparse_class_topk_fraction = float(
            sparse_class_topk_fraction
        )
        self.sparse_class_dropout = float(sparse_class_dropout)
        self.sparse_class_gate_initial_bias = float(
            sparse_class_gate_initial_bias
        )
        if self.sparse_class_weight < 0:
            raise ValueError("sparse_class_weight must be non-negative")
        if self.sparse_class_query_count <= 0:
            raise ValueError("sparse_class_query_count must be positive")
        if min(
            self.sparse_class_query_dim,
            self.sparse_class_value_dim,
            self.sparse_class_rank_dim,
            self.sparse_class_gate_hidden_dim,
        ) <= 0:
            raise ValueError("sparse_class dimensions must be positive")
        if self.sparse_class_temperature <= 0:
            raise ValueError("sparse_class_temperature must be positive")
        if not 0 < self.sparse_class_topk_fraction <= 1:
            raise ValueError(
                "sparse_class_topk_fraction must be in (0, 1]"
            )
        if not 0 <= self.sparse_class_dropout < 1:
            raise ValueError("sparse_class_dropout must be in [0, 1)")
        self.gated_attention_weight = float(gated_attention_weight)
        self.gated_attention_dim = int(gated_attention_dim)
        self.gated_attention_value_dim = int(gated_attention_value_dim)
        self.gated_attention_dropout = float(gated_attention_dropout)
        self.gated_attention_temperature = float(
            gated_attention_temperature
        )
        self.gated_attention_class_specific = bool(
            gated_attention_class_specific
        )
        if self.gated_attention_weight < 0:
            raise ValueError(
                "gated_attention_weight must be non-negative"
            )
        if (
            self.gated_attention_dim <= 0
            or self.gated_attention_value_dim <= 0
        ):
            raise ValueError(
                "gated_attention_dim and gated_attention_value_dim "
                "must be positive"
            )
        if not 0 <= self.gated_attention_dropout < 1:
            raise ValueError("gated_attention_dropout must be in [0, 1)")
        if self.gated_attention_temperature <= 0:
            raise ValueError(
                "gated_attention_temperature must be positive"
            )
        self.spatial_region_weight = float(spatial_region_weight)
        self.spatial_region_grid_size = int(spatial_region_grid_size)
        self.spatial_region_value_dim = int(spatial_region_value_dim)
        self.spatial_region_dim = int(spatial_region_dim)
        self.spatial_region_attention_dim = int(
            spatial_region_attention_dim
        )
        self.spatial_region_dropout = float(spatial_region_dropout)
        self.spatial_region_temperature = float(
            spatial_region_temperature
        )
        self.spatial_region_include_centers = bool(
            spatial_region_include_centers
        )
        self.spatial_region_include_mass = bool(
            spatial_region_include_mass
        )
        if self.spatial_region_weight < 0:
            raise ValueError("spatial_region_weight must be non-negative")
        if self.spatial_region_weight > 0 and self.coordinate_dim != 2:
            raise ValueError(
                "spatial_region_weight requires coordinate_dim=2"
            )
        if self.spatial_region_grid_size <= 0:
            raise ValueError("spatial_region_grid_size must be positive")
        if min(
            self.spatial_region_value_dim,
            self.spatial_region_dim,
            self.spatial_region_attention_dim,
        ) <= 0:
            raise ValueError("spatial region dimensions must be positive")
        if not 0 <= self.spatial_region_dropout < 1:
            raise ValueError("spatial_region_dropout must be in [0, 1)")
        if self.spatial_region_temperature <= 0:
            raise ValueError(
                "spatial_region_temperature must be positive"
            )
        self.latent_readout_weight = float(latent_readout_weight)
        self.latent_readout_count = int(latent_readout_count)
        self.latent_readout_dim = int(latent_readout_dim)
        self.latent_readout_heads = int(latent_readout_heads)
        self.latent_readout_layers = int(latent_readout_layers)
        self.latent_readout_mlp_ratio = float(latent_readout_mlp_ratio)
        self.latent_readout_temperature = float(
            latent_readout_temperature
        )
        self.latent_readout_dropout = float(latent_readout_dropout)
        if self.latent_readout_weight < 0:
            raise ValueError("latent_readout_weight must be non-negative")
        if self.latent_readout_count <= 0:
            raise ValueError("latent_readout_count must be positive")
        if self.latent_readout_dim <= 0:
            raise ValueError("latent_readout_dim must be positive")
        if (
            self.latent_readout_heads <= 0
            or self.latent_readout_dim % self.latent_readout_heads
        ):
            raise ValueError(
                "latent_readout_heads must be positive and divide "
                "latent_readout_dim"
            )
        if self.latent_readout_layers < 0:
            raise ValueError("latent_readout_layers must be non-negative")
        if self.latent_readout_mlp_ratio <= 0:
            raise ValueError("latent_readout_mlp_ratio must be positive")
        if self.latent_readout_temperature <= 0:
            raise ValueError(
                "latent_readout_temperature must be positive"
            )
        if not 0 <= self.latent_readout_dropout < 1:
            raise ValueError("latent_readout_dropout must be in [0, 1)")
        self.ordinal_head_weight = float(ordinal_head_weight)
        self.ordinal_head_hidden_dim = int(ordinal_head_hidden_dim)
        self.ordinal_head_dropout = float(ordinal_head_dropout)
        self.ordinal_head_temperature = float(ordinal_head_temperature)
        if self.ordinal_head_weight < 0:
            raise ValueError("ordinal_head_weight must be non-negative")
        if self.ordinal_head_hidden_dim <= 0:
            raise ValueError("ordinal_head_hidden_dim must be positive")
        if not 0 <= self.ordinal_head_dropout < 1:
            raise ValueError("ordinal_head_dropout must be in [0, 1)")
        if self.ordinal_head_temperature <= 0:
            raise ValueError("ordinal_head_temperature must be positive")
        self.cosine_head_weight = float(cosine_head_weight)
        self.cosine_head_dim = int(cosine_head_dim)
        self.cosine_head_hidden_dim = int(cosine_head_hidden_dim)
        self.cosine_head_dropout = float(cosine_head_dropout)
        self.cosine_head_initial_scale = float(cosine_head_initial_scale)
        if self.cosine_head_weight < 0:
            raise ValueError("cosine_head_weight must be non-negative")
        if self.cosine_head_dim <= 0 or self.cosine_head_hidden_dim <= 0:
            raise ValueError(
                "cosine_head_dim and cosine_head_hidden_dim must be "
                "positive"
            )
        if not 0 <= self.cosine_head_dropout < 1:
            raise ValueError("cosine_head_dropout must be in [0, 1)")
        if self.cosine_head_initial_scale <= 0:
            raise ValueError("cosine_head_initial_scale must be positive")
        self.ovr_head_weight = float(ovr_head_weight)
        self.ovr_loss_weight = float(ovr_loss_weight)
        self.ovr_loss_pos_weight = float(ovr_loss_pos_weight)
        if ovr_loss_pos_weights is None:
            self.ovr_loss_pos_weights = None
        else:
            self.ovr_loss_pos_weights = [
                float(weight) for weight in ovr_loss_pos_weights
            ]
        self.ovr_head_hidden_dim = int(ovr_head_hidden_dim)
        self.ovr_head_dropout = float(ovr_head_dropout)
        self.ovr_head_temperature = float(ovr_head_temperature)
        if self.ovr_head_weight < 0 or self.ovr_loss_weight < 0:
            raise ValueError("ovr weights must be non-negative")
        if self.ovr_loss_pos_weight <= 0:
            raise ValueError("ovr_loss_pos_weight must be positive")
        if self.ovr_loss_pos_weights is not None:
            if len(self.ovr_loss_pos_weights) != self.num_classes:
                raise ValueError(
                    "ovr_loss_pos_weights must match num_classes"
                )
            if any(weight <= 0 for weight in self.ovr_loss_pos_weights):
                raise ValueError("ovr_loss_pos_weights must be positive")
        if self.ovr_head_hidden_dim <= 0:
            raise ValueError("ovr_head_hidden_dim must be positive")
        if not 0 <= self.ovr_head_dropout < 1:
            raise ValueError("ovr_head_dropout must be in [0, 1)")
        if self.ovr_head_temperature <= 0:
            raise ValueError("ovr_head_temperature must be positive")
        self.adjacent_head_weight = float(adjacent_head_weight)
        self.adjacent_loss_weight = float(adjacent_loss_weight)
        self.adjacent_head_hidden_dim = int(adjacent_head_hidden_dim)
        self.adjacent_head_dropout = float(adjacent_head_dropout)
        self.adjacent_head_temperature = float(adjacent_head_temperature)
        if self.adjacent_head_weight < 0 or self.adjacent_loss_weight < 0:
            raise ValueError("adjacent weights must be non-negative")
        if self.adjacent_head_hidden_dim <= 0:
            raise ValueError("adjacent_head_hidden_dim must be positive")
        if not 0 <= self.adjacent_head_dropout < 1:
            raise ValueError("adjacent_head_dropout must be in [0, 1)")
        if self.adjacent_head_temperature <= 0:
            raise ValueError("adjacent_head_temperature must be positive")
        self.focus_class_index = int(focus_class_index)
        self.focus_class_head_weight = float(focus_class_head_weight)
        self.focus_class_loss_weight = float(focus_class_loss_weight)
        self.focus_class_loss_pos_weight = float(focus_class_loss_pos_weight)
        self.focus_class_head_hidden_dim = int(focus_class_head_hidden_dim)
        self.focus_class_head_dropout = float(focus_class_head_dropout)
        self.focus_class_head_temperature = float(
            focus_class_head_temperature
        )
        if not 0 <= self.focus_class_index < self.num_classes:
            raise ValueError("focus_class_index must identify a class")
        if self.focus_class_head_weight < 0 or self.focus_class_loss_weight < 0:
            raise ValueError("focus class weights must be non-negative")
        if self.focus_class_loss_pos_weight <= 0:
            raise ValueError("focus_class_loss_pos_weight must be positive")
        if self.focus_class_head_hidden_dim <= 0:
            raise ValueError("focus_class_head_hidden_dim must be positive")
        if not 0 <= self.focus_class_head_dropout < 1:
            raise ValueError("focus_class_head_dropout must be in [0, 1)")
        if self.focus_class_head_temperature <= 0:
            raise ValueError("focus_class_head_temperature must be positive")
        self.focus_sparse_head_weight = float(focus_sparse_head_weight)
        self.focus_sparse_loss_weight = float(focus_sparse_loss_weight)
        self.focus_sparse_loss_pos_weight = float(
            focus_sparse_loss_pos_weight
        )
        self.focus_sparse_query_count = int(focus_sparse_query_count)
        self.focus_sparse_query_dim = int(focus_sparse_query_dim)
        self.focus_sparse_value_dim = int(focus_sparse_value_dim)
        self.focus_sparse_readout_dim = int(focus_sparse_readout_dim)
        self.focus_sparse_temperature = float(focus_sparse_temperature)
        self.focus_sparse_topk_fraction = float(
            focus_sparse_topk_fraction
        )
        self.focus_sparse_dropout = float(focus_sparse_dropout)
        if (
            self.focus_sparse_head_weight < 0
            or self.focus_sparse_loss_weight < 0
        ):
            raise ValueError("focus-sparse weights must be non-negative")
        if self.focus_sparse_loss_pos_weight <= 0:
            raise ValueError("focus_sparse_loss_pos_weight must be positive")
        if self.focus_sparse_query_count <= 0:
            raise ValueError("focus_sparse_query_count must be positive")
        if min(
            self.focus_sparse_query_dim,
            self.focus_sparse_value_dim,
            self.focus_sparse_readout_dim,
        ) <= 0:
            raise ValueError("focus-sparse dimensions must be positive")
        if self.focus_sparse_temperature <= 0:
            raise ValueError("focus_sparse_temperature must be positive")
        if not 0 < self.focus_sparse_topk_fraction <= 1:
            raise ValueError(
                "focus_sparse_topk_fraction must be in (0, 1]"
            )
        if not 0 <= self.focus_sparse_dropout < 1:
            raise ValueError("focus_sparse_dropout must be in [0, 1)")
        self.logit_margin_loss_weight = float(logit_margin_loss_weight)
        self.logit_margin = float(logit_margin)
        if self.logit_margin_loss_weight < 0:
            raise ValueError(
                "logit_margin_loss_weight must be non-negative"
            )
        if self.logit_margin <= 0:
            raise ValueError("logit_margin must be positive")
        self.subset_consistency_weight = float(subset_consistency_weight)
        self.subset_consistency_supervised_weight = float(
            subset_consistency_supervised_weight
        )
        self.subset_consistency_fraction = float(
            subset_consistency_fraction
        )
        self.subset_consistency_views = int(subset_consistency_views)
        self.subset_consistency_temperature = float(
            subset_consistency_temperature
        )
        if self.subset_consistency_weight < 0:
            raise ValueError(
                "subset_consistency_weight must be non-negative"
            )
        if self.subset_consistency_supervised_weight < 0:
            raise ValueError(
                "subset_consistency_supervised_weight must be non-negative"
            )
        if not 0 < self.subset_consistency_fraction <= 1:
            raise ValueError(
                "subset_consistency_fraction must be in (0, 1]"
            )
        if self.subset_consistency_views < 1:
            raise ValueError("subset_consistency_views must be positive")
        if self.subset_consistency_temperature <= 0:
            raise ValueError(
                "subset_consistency_temperature must be positive"
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
        if self.num_local_routes > 0:
            self.local_route_basis = nn.Linear(
                hidden_dim, self.local_route_dim
            )
            self.local_route_scorer = nn.Linear(
                hidden_dim, self.num_local_routes
            )
        else:
            self.local_route_basis = None
            self.local_route_scorer = None
        if self.anchor_route_dim > 0:
            self.anchor_route_basis = (
                None
                if self.anchor_route_identity
                else nn.Linear(hidden_dim, self.anchor_route_dim)
            )
            self.anchor_route_scorer = nn.Linear(hidden_dim, 1)
        else:
            self.anchor_route_basis = None
            self.anchor_route_scorer = None
        local_state_dim = (
            self.num_local_routes * self.local_route_dim
            + self.anchor_route_dim
        )
        state_dim = (
            self.composition_state_dim
            + self.num_tail_scores
            + local_state_dim
        )
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
        elif self.potential_type == "adaptive_multiscale":
            if self.num_local_routes <= 0:
                raise ValueError(
                    "adaptive_multiscale requires num_local_routes > 0"
                )
            self.potential = AdaptiveMultiscalePotential(
                global_state_dim=(
                    self.composition_state_dim + self.num_tail_scores
                ),
                local_state_dim=(
                    local_state_dim
                ),
                num_classes=self.num_classes,
                hidden_dim=potential_hidden_dim,
                dropout=dropout,
                act=act,
                gate_initial_bias=multiscale_gate_initial_bias,
                local_initial_scale=multiscale_local_initial_scale,
            )
        elif self.potential_type == "anchored_multiscale":
            if self.num_local_routes <= 0:
                raise ValueError(
                    "anchored_multiscale requires num_local_routes > 0"
                )
            if self.anchor_route_dim <= 0:
                raise ValueError(
                    "anchored_multiscale requires anchor_route_dim > 0"
                )
            self.potential = AnchoredMultiscalePotential(
                global_state_dim=(
                    self.composition_state_dim + self.num_tail_scores
                ),
                local_state_dim=(
                    self.num_local_routes * self.local_route_dim
                ),
                anchor_state_dim=self.anchor_route_dim,
                num_classes=self.num_classes,
                hidden_dim=potential_hidden_dim,
                dropout=dropout,
                act=act,
                gate_initial_bias=multiscale_gate_initial_bias,
                global_initial_scale=anchor_global_initial_scale,
                local_initial_scale=anchor_local_initial_scale,
            )
        elif self.potential_type == "class_conditional_multiscale":
            if self.num_local_routes <= 0:
                raise ValueError(
                    "class_conditional_multiscale requires "
                    "num_local_routes > 0"
                )
            if self.num_local_routes % self.num_classes:
                raise ValueError(
                    "class_conditional_multiscale requires "
                    "num_local_routes divisible by num_classes"
                )
            if self.anchor_route_dim:
                raise ValueError(
                    "class_conditional_multiscale does not support "
                    "an anchor route"
                )
            self.potential = ClassConditionalMultiscalePotential(
                global_state_dim=(
                    self.composition_state_dim + self.num_tail_scores
                ),
                local_state_dim=(
                    self.num_local_routes * self.local_route_dim
                ),
                num_classes=self.num_classes,
                hidden_dim=potential_hidden_dim,
                dropout=dropout,
                act=act,
                gate_initial_bias=multiscale_gate_initial_bias,
                local_initial_scale=multiscale_local_initial_scale,
            )
        elif self.potential_type == "hybrid_multiscale":
            if self.num_local_routes <= 0:
                raise ValueError(
                    "hybrid_multiscale requires num_local_routes > 0"
                )
            if self.num_local_routes % self.num_classes:
                raise ValueError(
                    "hybrid_multiscale requires num_local_routes "
                    "divisible by num_classes"
                )
            if self.anchor_route_dim:
                raise ValueError(
                    "hybrid_multiscale does not support an anchor route"
                )
            self.potential = HybridMultiscalePotential(
                global_state_dim=(
                    self.composition_state_dim + self.num_tail_scores
                ),
                local_state_dim=(
                    self.num_local_routes * self.local_route_dim
                ),
                num_classes=self.num_classes,
                hidden_dim=potential_hidden_dim,
                dropout=dropout,
                act=act,
                gate_initial_bias=multiscale_gate_initial_bias,
                local_initial_scale=multiscale_local_initial_scale,
                class_mix_initial=multiscale_class_mix_initial,
            )
        elif self.potential_type == "residual_class_multiscale":
            if self.num_local_routes <= 0:
                raise ValueError(
                    "residual_class_multiscale requires "
                    "num_local_routes > 0"
                )
            if self.num_local_routes % self.num_classes:
                raise ValueError(
                    "residual_class_multiscale requires "
                    "num_local_routes divisible by num_classes"
                )
            if self.anchor_route_dim:
                raise ValueError(
                    "residual_class_multiscale does not support "
                    "an anchor route"
                )
            self.potential = ResidualClassMultiscalePotential(
                global_state_dim=(
                    self.composition_state_dim + self.num_tail_scores
                ),
                local_state_dim=(
                    local_state_dim
                ),
                num_classes=self.num_classes,
                hidden_dim=potential_hidden_dim,
                dropout=dropout,
                act=act,
                gate_initial_bias=multiscale_gate_initial_bias,
                local_initial_scale=multiscale_local_initial_scale,
                class_residual_initial_scale=(
                    multiscale_class_residual_initial_scale
                ),
            )
        elif self.potential_type == "adaptive_multiscale_prototype":
            if self.num_local_routes <= 0:
                raise ValueError(
                    "adaptive_multiscale_prototype requires "
                    "num_local_routes > 0"
                )
            self.potential = AdaptiveMultiscalePrototypePotential(
                state_dim=state_dim,
                global_state_dim=(
                    self.composition_state_dim + self.num_tail_scores
                ),
                local_state_dim=(
                    self.num_local_routes * self.local_route_dim
                ),
                num_classes=self.num_classes,
                hidden_dim=potential_hidden_dim,
                dropout=dropout,
                act=act,
                gate_initial_bias=multiscale_gate_initial_bias,
                local_initial_scale=multiscale_local_initial_scale,
                prototype_embedding_dim=prototype_embedding_dim,
                prototypes_per_class=prototypes_per_class,
                prototype_temperature=prototype_temperature,
                prototype_mixture_temperature=(
                    prototype_mixture_temperature
                ),
                prototype_diversity_margin=prototype_diversity_margin,
                prototype_separation_margin=prototype_separation_margin,
                prototype_initial_scale=(
                    multiscale_prototype_initial_scale
                ),
            )
        else:
            raise ValueError(
                f"Unsupported potential_type: {self.potential_type}"
            )
        self.evidence_head = None
        if self.evidence_weight > 0:
            self.evidence_head = ClassAwareEvidenceHead(
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                query_dim=evidence_query_dim,
                value_dim=evidence_value_dim,
                temperature=evidence_temperature,
                dropout=evidence_dropout,
            )
        self.multi_token_head = None
        self.multi_token_gate = None
        self.moment_token_head = None
        self.tail_token_head = None
        self.class_moment_token_head = None
        self.residual_class_moment_token_head = None
        self.class_token_head = None
        self.sparse_class_head = None
        self.gated_attention_head = None
        self.spatial_region_head = None
        self.latent_readout_head = None
        self.ordinal_head = None
        self.cosine_head = None
        self.ovr_head = None
        self.adjacent_head = None
        self.focus_class_head = None
        self.focus_sparse_head = None
        if self.multi_token_weight > 0:
            self.multi_token_head = MultiTokenAttentionReadout(
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                num_tokens=self.multi_token_count,
                token_dim=self.multi_token_dim,
                readout_dim=self.multi_token_readout_dim,
                temperature=self.multi_token_temperature,
                dropout=self.multi_token_dropout,
            )
            if self.multi_token_gated:
                self.multi_token_gate = nn.Sequential(
                    nn.LayerNorm(state_dim),
                    nn.Linear(state_dim, self.multi_token_gate_hidden_dim),
                    _activation(act),
                    nn.Linear(
                        self.multi_token_gate_hidden_dim,
                        self.num_classes,
                    ),
                )
        if self.class_token_weight > 0:
            self.class_token_head = LowRankClassTokenReadout(
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                num_tokens=self.class_token_count,
                token_dim=self.class_token_dim,
                value_dim=self.class_token_value_dim,
                rank_dim=self.class_token_rank_dim,
                temperature=self.class_token_temperature,
                dropout=self.class_token_dropout,
            )
        if self.sparse_class_weight > 0:
            self.sparse_class_head = SparseClassEvidenceReadout(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_classes=self.num_classes,
                num_queries=self.sparse_class_query_count,
                query_dim=self.sparse_class_query_dim,
                value_dim=self.sparse_class_value_dim,
                rank_dim=self.sparse_class_rank_dim,
                gate_hidden_dim=self.sparse_class_gate_hidden_dim,
                temperature=self.sparse_class_temperature,
                topk_fraction=self.sparse_class_topk_fraction,
                dropout=self.sparse_class_dropout,
                gate_initial_bias=self.sparse_class_gate_initial_bias,
            )
        if self.gated_attention_weight > 0:
            self.gated_attention_head = GatedAttentionResidualReadout(
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                attention_dim=self.gated_attention_dim,
                value_dim=self.gated_attention_value_dim,
                dropout=self.gated_attention_dropout,
                temperature=self.gated_attention_temperature,
                class_specific=self.gated_attention_class_specific,
            )
        if self.spatial_region_weight > 0:
            self.spatial_region_head = SpatialRegionMomentReadout(
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                grid_size=self.spatial_region_grid_size,
                value_dim=self.spatial_region_value_dim,
                region_dim=self.spatial_region_dim,
                attention_dim=self.spatial_region_attention_dim,
                dropout=self.spatial_region_dropout,
                temperature=self.spatial_region_temperature,
                include_centers=self.spatial_region_include_centers,
                include_mass=self.spatial_region_include_mass,
            )
        if self.moment_token_weight > 0:
            self.moment_token_head = MomentMultiTokenAttentionReadout(
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                num_tokens=self.moment_token_count,
                token_dim=self.moment_token_dim,
                readout_dim=self.moment_token_readout_dim,
                temperature=self.moment_token_temperature,
                dropout=self.moment_token_dropout,
            )
        if self.tail_token_weight > 0:
            self.tail_token_head = TailTokenAttentionReadout(
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                num_tokens=self.tail_token_count,
                token_dim=self.tail_token_dim,
                readout_dim=self.tail_token_readout_dim,
                temperature=self.tail_token_temperature,
                topk_fraction=self.tail_token_topk_fraction,
                dropout=self.tail_token_dropout,
            )
        if self.class_moment_token_weight > 0:
            self.class_moment_token_head = LowRankClassMomentTokenReadout(
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                num_tokens=self.class_moment_token_count,
                token_dim=self.class_moment_token_dim,
                value_dim=self.class_moment_token_value_dim,
                rank_dim=self.class_moment_token_rank_dim,
                temperature=self.class_moment_token_temperature,
                dropout=self.class_moment_token_dropout,
            )
        if self.residual_class_moment_token_weight > 0:
            self.residual_class_moment_token_head = (
                ResidualClassMomentTokenReadout(
                    hidden_dim=hidden_dim,
                    num_classes=self.num_classes,
                    num_tokens=self.residual_class_moment_token_count,
                    token_dim=self.residual_class_moment_token_dim,
                    readout_dim=(
                        self.residual_class_moment_token_readout_dim
                    ),
                    rank_dim=self.residual_class_moment_token_rank_dim,
                    temperature=(
                        self.residual_class_moment_token_temperature
                    ),
                    dropout=self.residual_class_moment_token_dropout,
                    residual_initial_scale=(
                        self.residual_class_moment_token_initial_scale
                    ),
                )
            )
        if self.latent_readout_weight > 0:
            self.latent_readout_head = LatentEvidenceTransformerReadout(
                hidden_dim=hidden_dim,
                num_classes=self.num_classes,
                num_latents=self.latent_readout_count,
                latent_dim=self.latent_readout_dim,
                num_heads=self.latent_readout_heads,
                num_layers=self.latent_readout_layers,
                mlp_ratio=self.latent_readout_mlp_ratio,
                temperature=self.latent_readout_temperature,
                dropout=self.latent_readout_dropout,
            )
        if self.ordinal_head_weight > 0:
            self.ordinal_head = OrdinalCalibrationHead(
                state_dim=state_dim,
                num_classes=self.num_classes,
                hidden_dim=self.ordinal_head_hidden_dim,
                dropout=self.ordinal_head_dropout,
                act=act,
                temperature=self.ordinal_head_temperature,
            )
        if self.cosine_head_weight > 0:
            self.cosine_head = CosineStateResidualHead(
                state_dim=state_dim,
                num_classes=self.num_classes,
                embedding_dim=self.cosine_head_dim,
                hidden_dim=self.cosine_head_hidden_dim,
                dropout=self.cosine_head_dropout,
                act=act,
                initial_scale=self.cosine_head_initial_scale,
            )
        if self.ovr_head_weight > 0 or self.ovr_loss_weight > 0:
            self.ovr_head = StateBoundaryHead(
                state_dim=state_dim,
                output_dim=self.num_classes,
                hidden_dim=self.ovr_head_hidden_dim,
                dropout=self.ovr_head_dropout,
                act=act,
                temperature=self.ovr_head_temperature,
            )
        if self.adjacent_head_weight > 0 or self.adjacent_loss_weight > 0:
            self.adjacent_head = StateBoundaryHead(
                state_dim=state_dim,
                output_dim=self.num_classes - 1,
                hidden_dim=self.adjacent_head_hidden_dim,
                dropout=self.adjacent_head_dropout,
                act=act,
                temperature=self.adjacent_head_temperature,
            )
        if self.focus_class_head_weight > 0 or self.focus_class_loss_weight > 0:
            self.focus_class_head = StateBoundaryHead(
                state_dim=state_dim,
                output_dim=1,
                hidden_dim=self.focus_class_head_hidden_dim,
                dropout=self.focus_class_head_dropout,
                act=act,
                temperature=self.focus_class_head_temperature,
            )
        if (
            self.focus_sparse_head_weight > 0
            or self.focus_sparse_loss_weight > 0
        ):
            self.focus_sparse_head = FocusedSparseEvidenceReadout(
                hidden_dim=hidden_dim,
                state_dim=state_dim,
                num_queries=self.focus_sparse_query_count,
                query_dim=self.focus_sparse_query_dim,
                value_dim=self.focus_sparse_value_dim,
                readout_dim=self.focus_sparse_readout_dim,
                temperature=self.focus_sparse_temperature,
                topk_fraction=self.focus_sparse_topk_fraction,
                dropout=self.focus_sparse_dropout,
            )
        self.apply(self._initialize)
        if self.multi_token_gate is not None:
            final_gate_layer = self.multi_token_gate[-1]
            nn.init.zeros_(final_gate_layer.weight)
            nn.init.constant_(
                final_gate_layer.bias, self.multi_token_gate_initial_bias
            )
        if self.sparse_class_head is not None:
            self.sparse_class_head.reset_gate()
        if self.focus_sparse_head is not None:
            self.focus_sparse_head.reset_mixer()
        if isinstance(
            self.potential,
            (
                AdaptiveMultiscalePotential,
                AnchoredMultiscalePotential,
                ClassConditionalMultiscalePotential,
                HybridMultiscalePotential,
                ResidualClassMultiscalePotential,
            ),
        ):
            self.potential.reset_gate()

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
        if self.input_group_l2_normalize:
            feature_groups = bag[:, : self.in_dim].reshape(
                bag.shape[0], -1, self.input_group_size
            )
            feature_groups = F.normalize(
                feature_groups, p=2, dim=2, eps=1e-12
            )
            features = feature_groups.reshape(bag.shape[0], self.in_dim)
            if self.coordinate_dim:
                bag = torch.cat((features, bag[:, self.in_dim :]), dim=1)
            else:
                bag = features
        return bag

    def _point_statistics(self, points):
        encoded = self.encoder(points)
        basis = self.response_basis(encoded)
        tail_scores = self.tail_scorer(encoded)
        if self.num_local_routes > 0:
            local_basis = self.local_route_basis(encoded)
            local_scores = self.local_route_scorer(encoded)
        else:
            local_basis = encoded.new_empty((encoded.shape[0], 0))
            local_scores = encoded.new_empty((encoded.shape[0], 0))
        if self.anchor_route_dim > 0:
            anchor_basis = (
                encoded
                if self.anchor_route_identity
                else self.anchor_route_basis(encoded)
            )
            anchor_scores = self.anchor_route_scorer(encoded)
        else:
            anchor_basis = encoded.new_empty((encoded.shape[0], 0))
            anchor_scores = encoded.new_empty((encoded.shape[0], 0))
        return (
            encoded,
            basis,
            tail_scores,
            local_basis,
            local_scores,
            anchor_basis,
            anchor_scores,
        )

    def _encoder_points(self, points):
        if self.coordinate_dim == 0 or self.coordinate_encoder_scale == 1.0:
            return points
        points = points.clone()
        points[:, self.in_dim :] *= self.coordinate_encoder_scale
        return points

    def state_from_weighted_points(self, points, weights=None):
        points = self._normalize_bag(points)
        points = self._encoder_points(points)
        (
            encoded,
            basis,
            tail_scores,
            local_basis,
            local_scores,
            anchor_basis,
            anchor_scores,
        ) = self._point_statistics(points)
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
        local_state = points.new_empty((0,))
        if self.num_local_routes > 0:
            local_log_weights = (
                local_scores / self.local_route_temperature
                + log_weights[:, None]
            )
            local_weights = torch.softmax(local_log_weights, dim=0)
            local_state = torch.einsum(
                "nr,nd->rd", local_weights, local_basis
            ).flatten()
        anchor_state = points.new_empty((0,))
        if self.anchor_route_dim > 0:
            anchor_log_weights = (
                anchor_scores[:, 0] / self.anchor_route_temperature
                + log_weights
            )
            anchor_weights = torch.softmax(anchor_log_weights, dim=0)
            anchor_state = torch.sum(
                anchor_weights[:, None] * anchor_basis, dim=0
            )
        state = torch.cat(
            [
                *composition_state,
                tail_state,
                local_state,
                anchor_state,
            ],
            dim=0,
        )
        return (
            state,
            encoded,
            basis,
            tail_scores,
            weights,
            local_basis,
            local_scores,
            anchor_basis,
            anchor_scores,
        )

    @staticmethod
    def adjacent_logits_to_class_logits(adjacent_logits):
        survival = torch.sigmoid(adjacent_logits)
        eps = torch.finfo(adjacent_logits.dtype).eps
        probabilities = []
        probabilities.append((1.0 - survival[:, :1]).clamp_min(eps))
        if adjacent_logits.shape[1] > 1:
            probabilities.append(
                (survival[:, :-1] - survival[:, 1:]).clamp_min(eps)
            )
        probabilities.append(survival[:, -1:].clamp_min(eps))
        probabilities = torch.cat(probabilities, dim=1)
        probabilities = probabilities / probabilities.sum(
            dim=1, keepdim=True
        ).clamp_min(eps)
        logits = torch.log(probabilities)
        return logits - logits.mean(dim=1, keepdim=True)

    def residual_readout_logits(
        self,
        encoded,
        state=None,
        coordinates=None,
        return_auxiliary=False,
    ):
        logits = encoded.new_zeros((1, self.num_classes))
        auxiliary = {}
        if self.evidence_head is not None:
            logits = logits + self.evidence_weight * self.evidence_head(
                encoded
            )
        if self.multi_token_head is not None:
            multi_token_logits = self.multi_token_head(encoded)
            if self.multi_token_gate is not None:
                if state is None:
                    raise ValueError(
                        "state is required when multi_token_gated is enabled"
                    )
                gate = torch.sigmoid(
                    self.multi_token_gate(state.unsqueeze(0))
                )
                multi_token_logits = gate * multi_token_logits
            logits = logits + self.multi_token_weight * multi_token_logits
        if self.class_token_head is not None:
            logits = logits + self.class_token_weight * self.class_token_head(
                encoded
            )
        if self.sparse_class_head is not None:
            if state is None:
                raise ValueError(
                    "state is required when sparse_class_head is enabled"
                )
            logits = logits + (
                self.sparse_class_weight
                * self.sparse_class_head(encoded, state)
            )
        if self.gated_attention_head is not None:
            logits = logits + (
                self.gated_attention_weight
                * self.gated_attention_head(encoded)
            )
        if self.spatial_region_head is not None:
            if coordinates is None:
                raise ValueError(
                    "coordinates are required when spatial regions are enabled"
                )
            logits = logits + (
                self.spatial_region_weight
                * self.spatial_region_head(encoded, coordinates)
            )
        if self.moment_token_head is not None:
            logits = logits + (
                self.moment_token_weight * self.moment_token_head(encoded)
            )
        if self.tail_token_head is not None:
            logits = logits + (
                self.tail_token_weight * self.tail_token_head(encoded)
            )
        if self.class_moment_token_head is not None:
            logits = logits + (
                self.class_moment_token_weight
                * self.class_moment_token_head(encoded)
            )
        if self.residual_class_moment_token_head is not None:
            logits = logits + (
                self.residual_class_moment_token_weight
                * self.residual_class_moment_token_head(encoded)
            )
        if self.latent_readout_head is not None:
            logits = logits + (
                self.latent_readout_weight
                * self.latent_readout_head(encoded)
            )
        if self.ordinal_head is not None:
            if state is None:
                raise ValueError(
                    "state is required when ordinal_head is enabled"
                )
            logits = logits + self.ordinal_head_weight * self.ordinal_head(
                state.unsqueeze(0)
            )
        if self.cosine_head is not None:
            if state is None:
                raise ValueError(
                    "state is required when cosine_head is enabled"
                )
            logits = logits + self.cosine_head_weight * self.cosine_head(
                state.unsqueeze(0)
            )
        if self.ovr_head is not None:
            if state is None:
                raise ValueError("state is required when ovr_head is enabled")
            ovr_logits = self.ovr_head(state.unsqueeze(0))
            auxiliary["ovr_logits"] = ovr_logits
            logits = logits + self.ovr_head_weight * ovr_logits
        if self.adjacent_head is not None:
            if state is None:
                raise ValueError(
                    "state is required when adjacent_head is enabled"
                )
            adjacent_logits = self.adjacent_head(state.unsqueeze(0))
            auxiliary["adjacent_logits"] = adjacent_logits
            logits = logits + self.adjacent_head_weight * (
                self.adjacent_logits_to_class_logits(adjacent_logits)
            )
        if self.focus_class_head is not None:
            if state is None:
                raise ValueError(
                    "state is required when focus_class_head is enabled"
                )
            focus_logits = self.focus_class_head(state.unsqueeze(0))
            auxiliary["focus_class_logits"] = focus_logits
            focus_residual = logits.new_full(
                (1, self.num_classes),
                -1.0 / max(self.num_classes - 1, 1),
            )
            focus_residual[:, self.focus_class_index] = 1.0
            logits = logits + (
                self.focus_class_head_weight
                * focus_logits
                * focus_residual
            )
        if self.focus_sparse_head is not None:
            if state is None:
                raise ValueError(
                    "state is required when focus_sparse_head is enabled"
                )
            focus_sparse_logits = self.focus_sparse_head(encoded, state)
            auxiliary["focus_sparse_logits"] = focus_sparse_logits
            focus_sparse_residual = logits.new_full(
                (1, self.num_classes),
                -1.0 / max(self.num_classes - 1, 1),
            )
            focus_sparse_residual[:, self.focus_class_index] = 1.0
            logits = logits + (
                self.focus_sparse_head_weight
                * focus_sparse_logits
                * focus_sparse_residual
            )
        if return_auxiliary:
            return logits, auxiliary
        return logits

    def calibrate_logits(self, logits):
        if not self.use_logit_calibration:
            return logits
        temperature = self.logit_calibration_log_temperature.exp()
        temperature = temperature.clamp(min=0.05, max=20.0)
        return logits / temperature + self.logit_calibration_bias.to(
            dtype=logits.dtype,
            device=logits.device,
        )

    def forward(self, bag, return_state=False):
        normalized_bag = self._normalize_bag(bag)
        coordinates = (
            normalized_bag[:, self.in_dim : self.in_dim + 2]
            if self.coordinate_dim == 2
            else None
        )
        (
            state,
            encoded,
            basis,
            tail_scores,
            weights,
            local_basis,
            local_scores,
            anchor_basis,
            anchor_scores,
        ) = self.state_from_weighted_points(normalized_bag)
        logits = self.potential(state.unsqueeze(0))
        residual_logits, auxiliary = self.residual_readout_logits(
            encoded,
            state,
            coordinates=coordinates,
            return_auxiliary=True,
        )
        logits = logits + residual_logits
        raw_logits = logits
        logits = self.calibrate_logits(raw_logits)
        output = {"logits": logits}
        output.update(auxiliary)
        if self.use_logit_calibration:
            output["raw_logits"] = raw_logits
        if return_state:
            output.update(
                {
                    "state": state,
                    "encoded": encoded,
                    "basis": basis,
                    "tail_scores": tail_scores,
                    "weights": weights,
                    "local_basis": local_basis,
                    "local_scores": local_scores,
                    "anchor_basis": anchor_basis,
                    "anchor_scores": anchor_scores,
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
        (
            state,
            _encoded,
            basis,
            tail_scores,
            weights,
            local_basis,
            local_scores,
            anchor_basis,
            anchor_scores,
        ) = self.state_from_weighted_points(bag)
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
        tail_gradient = gradient[
            offset : offset + self.num_tail_scores
        ]
        offset += self.num_tail_scores

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
        local_response = basis.new_zeros(basis.shape[0])
        if self.num_local_routes > 0:
            local_size = self.num_local_routes * self.local_route_dim
            local_state = state[offset : offset + local_size].reshape(
                self.num_local_routes, self.local_route_dim
            )
            local_gradient = gradient[
                offset : offset + local_size
            ].reshape(
                self.num_local_routes, self.local_route_dim
            )
            local_log_normalizers = torch.logsumexp(
                local_scores / self.local_route_temperature
                + torch.log(
                    weights.clamp_min(torch.finfo(weights.dtype).tiny)
                )[:, None],
                dim=0,
            )
            local_density_ratio = torch.exp(
                local_scores / self.local_route_temperature
                - local_log_normalizers[None, :]
            )
            local_response = torch.einsum(
                "nr,nrd,rd->n",
                local_density_ratio,
                local_basis[:, None, :] - local_state[None, :, :],
                local_gradient,
            )
            offset += local_size
        anchor_response = basis.new_zeros(basis.shape[0])
        if self.anchor_route_dim > 0:
            anchor_state = state[
                offset : offset + self.anchor_route_dim
            ]
            anchor_gradient = gradient[
                offset : offset + self.anchor_route_dim
            ]
            anchor_log_normalizer = torch.logsumexp(
                anchor_scores[:, 0] / self.anchor_route_temperature
                + torch.log(
                    weights.clamp_min(
                        torch.finfo(weights.dtype).tiny
                    )
                ),
                dim=0,
            )
            anchor_density_ratio = torch.exp(
                anchor_scores[:, 0] / self.anchor_route_temperature
                - anchor_log_normalizer
            )
            anchor_response = (
                anchor_density_ratio[:, None]
                * (anchor_basis - anchor_state[None, :])
                * anchor_gradient[None, :]
            ).sum(dim=1)
        return {
            "logits": logits,
            "score": score,
            "target_class": target_class,
            "response": (
                composition_response
                + variance_response
                + tail_response
                + local_response
                + anchor_response
            ),
            "composition_response": composition_response,
            "variance_response": variance_response,
            "tail_response": tail_response,
            "local_response": local_response,
            "anchor_response": anchor_response,
        }

    def functional_derivative(
        self,
        evaluation_points,
        reference_points,
        reference_weights=None,
        target_class=None,
    ):
        (
            reference_state,
            _reference_encoded,
            _,
            reference_tail,
            reference_weights,
            _,
            reference_local_scores,
            _,
            reference_anchor_scores,
        ) = self.state_from_weighted_points(
            reference_points, reference_weights
        )
        logits = self.potential(reference_state.unsqueeze(0))
        if target_class is None:
            target_class = int(logits.argmax(dim=1).item())
        score = self.explained_score(logits, target_class)
        gradient = torch.autograd.grad(
            score, reference_state, retain_graph=True
        )[0]
        evaluation_points = self._normalize_bag(evaluation_points)
        (
            _encoded,
            basis,
            tail_scores,
            local_basis,
            local_scores,
            anchor_basis,
            anchor_scores,
        ) = self._point_statistics(evaluation_points)
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
        tail_gradient = gradient[
            offset : offset + self.num_tail_scores
        ]
        offset += self.num_tail_scores
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
        local_derivative = basis.new_zeros(basis.shape[0])
        if self.num_local_routes > 0:
            local_size = self.num_local_routes * self.local_route_dim
            local_state = reference_state[
                offset : offset + local_size
            ].reshape(
                self.num_local_routes, self.local_route_dim
            )
            local_gradient = gradient[
                offset : offset + local_size
            ].reshape(
                self.num_local_routes, self.local_route_dim
            )
            local_log_normalizers = torch.logsumexp(
                reference_local_scores / self.local_route_temperature
                + torch.log(
                    reference_weights.clamp_min(
                        torch.finfo(reference_weights.dtype).tiny
                    )
                )[:, None],
                dim=0,
            )
            local_density_ratio = torch.exp(
                local_scores / self.local_route_temperature
                - local_log_normalizers[None, :]
            )
            local_derivative = torch.einsum(
                "nr,nrd,rd->n",
                local_density_ratio,
                local_basis[:, None, :] - local_state[None, :, :],
                local_gradient,
            )
            offset += local_size
        anchor_derivative = basis.new_zeros(basis.shape[0])
        if self.anchor_route_dim > 0:
            anchor_state = reference_state[
                offset : offset + self.anchor_route_dim
            ]
            anchor_gradient = gradient[
                offset : offset + self.anchor_route_dim
            ]
            anchor_log_normalizer = torch.logsumexp(
                reference_anchor_scores[:, 0]
                / self.anchor_route_temperature
                + torch.log(
                    reference_weights.clamp_min(
                        torch.finfo(reference_weights.dtype).tiny
                    )
                ),
                dim=0,
            )
            anchor_density_ratio = torch.exp(
                anchor_scores[:, 0] / self.anchor_route_temperature
                - anchor_log_normalizer
            )
            anchor_derivative = (
                anchor_density_ratio[:, None]
                * (anchor_basis - anchor_state[None, :])
                * anchor_gradient[None, :]
            ).sum(dim=1)
        return (
            basis * composition_gradient[None, :]
        ).sum(
            dim=1
        ) + variance_derivative + local_derivative + anchor_derivative + (
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
        perturbed_state, perturbed_encoded = self.state_from_weighted_points(
            support, weights
        )[:2]
        perturbed_logits = self.potential(perturbed_state.unsqueeze(0))
        perturbed_logits = perturbed_logits + self.residual_readout_logits(
            perturbed_encoded,
            perturbed_state,
            coordinates=(
                support[:, self.in_dim : self.in_dim + 2]
                if self.coordinate_dim == 2
                else None
            ),
        )
        perturbed_score = self.explained_score(
            perturbed_logits, target_class
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

    def subset_view(self, bag):
        bag = self._normalize_bag(bag)
        if bag.shape[0] <= 1 or self.subset_consistency_fraction >= 1:
            return bag
        count = max(
            1,
            int(math.ceil(bag.shape[0] * self.subset_consistency_fraction)),
        )
        if count >= bag.shape[0]:
            return bag
        indices = torch.randperm(bag.shape[0], device=bag.device)[:count]
        return bag[indices]

    def prediction_consistency_loss(self, teacher_logits, student_logits):
        temperature = self.subset_consistency_temperature
        teacher_probabilities = torch.softmax(
            teacher_logits.detach() / temperature,
            dim=1,
        )
        student_log_probabilities = torch.log_softmax(
            student_logits / temperature,
            dim=1,
        )
        return (
            F.kl_div(
                student_log_probabilities,
                teacher_probabilities,
                reduction="batchmean",
            )
            * temperature
            * temperature
        )

    def lipschitz_penalty(self, bag):
        if self.lipschitz_weight <= 0:
            return bag.new_zeros(())
        bag = self._normalize_bag(bag)
        count = min(self.lipschitz_samples, bag.shape[0])
        indices = torch.randperm(bag.shape[0], device=bag.device)[:count]
        points = bag[indices].detach().requires_grad_(True)
        basis = self._point_statistics(points)[1]
        probe = torch.randn_like(basis)
        probe = probe / probe.norm(dim=1, keepdim=True).clamp_min(1e-8)
        scalar = (basis * probe).sum()
        gradients = torch.autograd.grad(
            scalar, points, create_graph=True
        )[0]
        norms = gradients.norm(dim=1)
        return F.relu(norms - self.lipschitz_target).square().mean()

    def logit_margin_loss(self, logits, label):
        if self.logit_margin_loss_weight <= 0:
            return logits.new_zeros(())
        if logits.ndim != 2:
            raise ValueError("logits must have shape [batch, num_classes]")
        if logits.shape[1] < 2:
            raise ValueError("logit margin loss requires at least 2 classes")
        label = label.long().view(-1)
        true_logits = logits.gather(1, label.unsqueeze(1))
        negative_mask = torch.ones_like(logits, dtype=torch.bool)
        negative_mask.scatter_(1, label.unsqueeze(1), False)
        negative_logits = logits[negative_mask].view(logits.shape[0], -1)
        violations = self.logit_margin + negative_logits - true_logits
        return F.relu(violations).square().mean()

    def ovr_boundary_loss(self, ovr_logits, label):
        if self.ovr_loss_weight <= 0 or ovr_logits is None:
            return ovr_logits.new_zeros(()) if ovr_logits is not None else None
        label = label.long().view(-1)
        targets = F.one_hot(label, num_classes=self.num_classes).to(
            dtype=ovr_logits.dtype
        )
        if self.ovr_loss_pos_weights is None:
            pos_weight = torch.full(
                (self.num_classes,),
                self.ovr_loss_pos_weight,
                dtype=ovr_logits.dtype,
                device=ovr_logits.device,
            )
        else:
            pos_weight = torch.tensor(
                self.ovr_loss_pos_weights,
                dtype=ovr_logits.dtype,
                device=ovr_logits.device,
            )
        return F.binary_cross_entropy_with_logits(
            ovr_logits, targets, pos_weight=pos_weight
        )

    def adjacent_boundary_loss(self, adjacent_logits, label):
        if self.adjacent_loss_weight <= 0 or adjacent_logits is None:
            if adjacent_logits is not None:
                return adjacent_logits.new_zeros(())
            return None
        label = label.long().view(-1)
        thresholds = torch.arange(
            self.num_classes - 1, device=adjacent_logits.device
        ).unsqueeze(0)
        targets = (label.unsqueeze(1) > thresholds).to(
            dtype=adjacent_logits.dtype
        )
        return F.binary_cross_entropy_with_logits(
            adjacent_logits, targets
        )

    def focus_class_boundary_loss(self, focus_logits, label):
        if self.focus_class_loss_weight <= 0 or focus_logits is None:
            if focus_logits is not None:
                return focus_logits.new_zeros(())
            return None
        targets = (label.long().view(-1) == self.focus_class_index).to(
            dtype=focus_logits.dtype
        ).unsqueeze(1)
        pos_weight = torch.tensor(
            self.focus_class_loss_pos_weight,
            dtype=focus_logits.dtype,
            device=focus_logits.device,
        )
        return F.binary_cross_entropy_with_logits(
            focus_logits, targets, pos_weight=pos_weight
        )

    def focus_sparse_boundary_loss(self, focus_logits, label):
        if self.focus_sparse_loss_weight <= 0 or focus_logits is None:
            if focus_logits is not None:
                return focus_logits.new_zeros(())
            return None
        targets = (label.long().view(-1) == self.focus_class_index).to(
            dtype=focus_logits.dtype
        ).unsqueeze(1)
        pos_weight = torch.tensor(
            self.focus_sparse_loss_pos_weight,
            dtype=focus_logits.dtype,
            device=focus_logits.device,
        )
        return F.binary_cross_entropy_with_logits(
            focus_logits, targets, pos_weight=pos_weight
        )

    def compute_loss(self, bag, label, criterion):
        output = self.forward(bag)
        classification_loss = criterion(output["logits"], label)
        logit_margin_loss = self.logit_margin_loss(
            output["logits"], label
        )
        ordinal_loss = self.ordinal_cdf_loss(output["logits"], label)
        ovr_loss = bag.new_zeros(())
        if "ovr_logits" in output:
            ovr_loss = self.ovr_boundary_loss(output["ovr_logits"], label)
        adjacent_loss = bag.new_zeros(())
        if "adjacent_logits" in output:
            adjacent_loss = self.adjacent_boundary_loss(
                output["adjacent_logits"], label
            )
        focus_class_loss = bag.new_zeros(())
        if "focus_class_logits" in output:
            focus_class_loss = self.focus_class_boundary_loss(
                output["focus_class_logits"], label
            )
        focus_sparse_loss = bag.new_zeros(())
        if "focus_sparse_logits" in output:
            focus_sparse_loss = self.focus_sparse_boundary_loss(
                output["focus_sparse_logits"], label
            )
        stability_loss = bag.new_zeros(())
        if self.stability_weight > 0:
            augmented_logits = self.forward(self.augment_bag(bag))["logits"]
            stability_loss = F.mse_loss(
                output["logits"], augmented_logits
            )
        subset_consistency_loss = bag.new_zeros(())
        subset_supervised_loss = bag.new_zeros(())
        if (
            self.subset_consistency_weight > 0
            or self.subset_consistency_supervised_weight > 0
        ):
            view_consistency_losses = []
            view_supervised_losses = []
            for _ in range(self.subset_consistency_views):
                view_logits = self.forward(self.subset_view(bag))["logits"]
                if self.subset_consistency_weight > 0:
                    view_consistency_losses.append(
                        self.prediction_consistency_loss(
                            output["logits"], view_logits
                        )
                    )
                if self.subset_consistency_supervised_weight > 0:
                    view_supervised_losses.append(
                        criterion(view_logits, label)
                    )
            if view_consistency_losses:
                subset_consistency_loss = torch.stack(
                    view_consistency_losses
                ).mean()
            if view_supervised_losses:
                subset_supervised_loss = torch.stack(
                    view_supervised_losses
                ).mean()
        lipschitz_loss = self.lipschitz_penalty(bag)
        prototype_loss = bag.new_zeros(())
        if hasattr(self.potential, "regularization"):
            prototype_loss = self.potential.regularization()
        loss = (
            classification_loss
            + self.logit_margin_loss_weight * logit_margin_loss
            + self.ordinal_weight * ordinal_loss
            + self.ovr_loss_weight * ovr_loss
            + self.adjacent_loss_weight * adjacent_loss
            + self.focus_class_loss_weight * focus_class_loss
            + self.focus_sparse_loss_weight * focus_sparse_loss
            + self.stability_weight * stability_loss
            + self.subset_consistency_weight * subset_consistency_loss
            + self.subset_consistency_supervised_weight
            * subset_supervised_loss
            + self.lipschitz_weight * lipschitz_loss
            + self.prototype_regularization_weight * prototype_loss
        )
        return output, {
            "loss": loss,
            "classification_loss": classification_loss,
            "logit_margin_loss": logit_margin_loss,
            "ordinal_loss": ordinal_loss,
            "ovr_loss": ovr_loss,
            "adjacent_loss": adjacent_loss,
            "focus_class_loss": focus_class_loss,
            "focus_sparse_loss": focus_sparse_loss,
            "stability_loss": stability_loss,
            "subset_consistency_loss": subset_consistency_loss,
            "subset_supervised_loss": subset_supervised_loss,
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
