import math

import torch
import torch.nn as nn

from .ordinal_risk_mil import _activation


class BoundaryRiskMIL(nn.Module):
    """Continuation-ratio MIL over adjacent, distributional boundaries.

    A shared patch representation produces one bounded evidence field for each
    adjacent class boundary. Each field is aggregated as a bag functional and
    parameterizes the probability of continuing beyond that boundary. The
    resulting cumulative probabilities are ordered by construction.
    """

    VALID_AGGREGATIONS = {"mean", "entropic"}

    def __init__(
        self,
        in_dim,
        num_classes,
        hidden_dim=256,
        dropout=0.1,
        act="gelu",
        aggregation="entropic",
        risk_temperature=0.5,
        decision_temperature=1.0,
        score_bound=3.0,
    ):
        super().__init__()
        if int(in_dim) <= 0:
            raise ValueError("in_dim must be positive")
        if int(num_classes) < 2:
            raise ValueError("num_classes must be at least two")
        if int(hidden_dim) <= 0:
            raise ValueError("hidden_dim must be positive")
        if aggregation not in self.VALID_AGGREGATIONS:
            raise ValueError(
                f"aggregation must be one of {sorted(self.VALID_AGGREGATIONS)}"
            )
        if float(risk_temperature) <= 0:
            raise ValueError("risk_temperature must be positive")
        if float(decision_temperature) <= 0:
            raise ValueError("decision_temperature must be positive")
        if float(score_bound) <= 0:
            raise ValueError("score_bound must be positive")

        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.num_boundaries = self.num_classes - 1
        self.aggregation = str(aggregation)
        self.risk_temperature = float(risk_temperature)
        self.decision_temperature = float(decision_temperature)
        self.score_bound = float(score_bound)

        self.encoder = nn.Sequential(
            nn.Linear(self.in_dim, int(hidden_dim)),
            _activation(act),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            _activation(act),
            nn.Dropout(float(dropout)),
        )
        self.boundary_witness = nn.Linear(
            int(hidden_dim), self.num_boundaries
        )
        self.boundary_bias = nn.Parameter(torch.zeros(self.num_boundaries))

    def normalize_bag(self, bag):
        if bag.ndim == 3:
            if bag.shape[0] != 1:
                raise ValueError("BoundaryRiskMIL expects batch size one")
            bag = bag.squeeze(0)
        if bag.ndim != 2 or bag.shape[0] == 0:
            raise ValueError("bag must have shape [N, D] with N > 0")
        if bag.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected input dimension {self.in_dim}, got {bag.shape[1]}"
            )
        return bag

    def bounded_evidence(self, encoded):
        return self.score_bound * torch.tanh(
            self.boundary_witness(encoded) / self.score_bound
        )

    def aggregate_evidence(self, evidence):
        if evidence.ndim != 2 or evidence.shape[0] == 0:
            raise ValueError("evidence must have shape [N, C-1] with N > 0")
        if evidence.shape[1] != self.num_boundaries:
            raise ValueError(
                f"Expected {self.num_boundaries} boundaries, got "
                f"{evidence.shape[1]}"
            )
        if self.aggregation == "mean":
            return evidence.mean(dim=0)
        temperature = torch.as_tensor(
            self.risk_temperature,
            dtype=evidence.dtype,
            device=evidence.device,
        )
        return temperature * (
            torch.logsumexp(evidence / temperature, dim=0)
            - math.log(evidence.shape[0])
        )

    def class_probabilities(self, risks):
        continuation = torch.sigmoid(
            (risks + self.boundary_bias) / self.decision_temperature
        )
        cumulative = torch.cumprod(continuation, dim=0)
        probabilities = torch.cat(
            (
                1.0 - cumulative[:1],
                cumulative[:-1] - cumulative[1:],
                cumulative[-1:],
            )
        )
        probabilities = probabilities.clamp_min(1e-8)
        return probabilities / probabilities.sum(), continuation, cumulative

    def forward(self, bag, return_state=False):
        bag = self.normalize_bag(bag)
        encoded = self.encoder(bag)
        evidence = self.bounded_evidence(encoded)
        risks = self.aggregate_evidence(evidence)
        probabilities, continuation, cumulative = self.class_probabilities(
            risks
        )
        output = {"logits": probabilities.log().unsqueeze(0)}
        if return_state:
            output.update(
                {
                    "probabilities": probabilities,
                    "boundary_evidence": evidence,
                    "boundary_risks": risks,
                    "continuation_probabilities": continuation,
                    "cumulative_probabilities": cumulative,
                }
            )
        return output

    def compute_loss(self, bag, label, criterion):
        output = self.forward(bag, return_state=True)
        classification_loss = criterion(output["logits"], label)
        return output, {
            "loss": classification_loss,
            "classification_loss": classification_loss,
        }
