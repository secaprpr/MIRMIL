import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation(name):
    name = str(name).lower()
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported activation: {name}")


class OrdinalRiskMIL(nn.Module):
    """A single-path MIL predictor built from an ordered severity risk.

    Each instance is mapped to a scalar severity. The empirical bag measure is
    reduced by either its mean or a normalized entropic risk, after which the
    bag is classified by ordered cumulative boundaries or an unconstrained
    softmax ablation. No parallel or residual bag-level predictor is used.
    """

    VALID_AGGREGATIONS = {"mean", "entropic"}
    VALID_HEADS = {"ordinal", "softmax"}

    def __init__(
        self,
        in_dim,
        num_classes,
        hidden_dim=256,
        dropout=0.1,
        act="gelu",
        aggregation="entropic",
        prediction_head="ordinal",
        risk_temperature=0.5,
        learnable_risk_temperature=False,
        decision_temperature=1.0,
        min_temperature=0.05,
        initial_threshold_center=0.0,
        initial_threshold_gap=1.0,
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
        if prediction_head not in self.VALID_HEADS:
            raise ValueError(
                f"prediction_head must be one of {sorted(self.VALID_HEADS)}"
            )
        if float(risk_temperature) <= 0:
            raise ValueError("risk_temperature must be positive")
        if float(decision_temperature) <= 0:
            raise ValueError("decision_temperature must be positive")
        if float(min_temperature) <= 0:
            raise ValueError("min_temperature must be positive")
        if float(initial_threshold_gap) <= 0:
            raise ValueError("initial_threshold_gap must be positive")

        self.in_dim = int(in_dim)
        self.num_classes = int(num_classes)
        self.aggregation = str(aggregation)
        self.prediction_head = str(prediction_head)
        self.learnable_risk_temperature = bool(
            learnable_risk_temperature
        )
        self.min_temperature = float(min_temperature)
        self.decision_temperature = float(decision_temperature)

        self.severity = nn.Sequential(
            nn.Linear(self.in_dim, int(hidden_dim)),
            _activation(act),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            _activation(act),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

        initial_log_temperature = math.log(float(risk_temperature))
        if self.learnable_risk_temperature:
            self.log_risk_temperature = nn.Parameter(
                torch.tensor(initial_log_temperature)
            )
        else:
            self.register_buffer(
                "log_risk_temperature",
                torch.tensor(initial_log_temperature),
            )

        if self.prediction_head == "ordinal":
            initial_origin = float(initial_threshold_center) - (
                float(initial_threshold_gap) * (self.num_classes - 2) / 2.0
            )
            self.threshold_origin = nn.Parameter(
                torch.tensor(initial_origin)
            )
            if self.num_classes > 2:
                initial_gap = torch.full(
                    (self.num_classes - 2,),
                    float(initial_threshold_gap),
                )
                self.raw_threshold_gaps = nn.Parameter(
                    torch.log(torch.expm1(initial_gap))
                )
            else:
                self.register_parameter("raw_threshold_gaps", None)
            self.softmax_head = None
        else:
            self.softmax_head = nn.Linear(1, self.num_classes)
            self.register_parameter("threshold_origin", None)
            self.register_parameter("raw_threshold_gaps", None)

    @property
    def risk_temperature(self):
        return self.log_risk_temperature.exp().clamp_min(
            self.min_temperature
        )

    def ordered_thresholds(self):
        if self.prediction_head != "ordinal":
            raise RuntimeError("ordered thresholds require the ordinal head")
        if self.raw_threshold_gaps is None:
            return self.threshold_origin.reshape(1)
        gaps = F.softplus(self.raw_threshold_gaps) + 1e-4
        return torch.cat(
            (
                self.threshold_origin.reshape(1),
                self.threshold_origin + torch.cumsum(gaps, dim=0),
            )
        )

    def normalize_bag(self, bag):
        if bag.ndim == 3:
            if bag.shape[0] != 1:
                raise ValueError("OrdinalRiskMIL expects batch size one")
            bag = bag.squeeze(0)
        if bag.ndim != 2 or bag.shape[0] == 0:
            raise ValueError("bag must have shape [N, D] with N > 0")
        if bag.shape[1] != self.in_dim:
            raise ValueError(
                f"Expected input dimension {self.in_dim}, got {bag.shape[1]}"
            )
        return bag

    def aggregate_scores(self, scores):
        scores = scores.reshape(-1)
        if scores.numel() == 0:
            raise ValueError("scores must not be empty")
        if self.aggregation == "mean":
            return scores.mean()
        temperature = self.risk_temperature.to(
            dtype=scores.dtype, device=scores.device
        )
        return temperature * (
            torch.logsumexp(scores / temperature, dim=0)
            - math.log(scores.numel())
        )

    def ordinal_probabilities(self, risk):
        thresholds = self.ordered_thresholds().to(
            dtype=risk.dtype, device=risk.device
        )
        cumulative = torch.sigmoid(
            (risk.reshape(1) - thresholds) / self.decision_temperature
        )
        probabilities = torch.cat(
            (
                1.0 - cumulative[:1],
                cumulative[:-1] - cumulative[1:],
                cumulative[-1:],
            )
        )
        probabilities = probabilities.clamp_min(1e-8)
        return probabilities / probabilities.sum()

    def forward(self, bag, return_state=False):
        bag = self.normalize_bag(bag)
        instance_severity = self.severity(bag).squeeze(-1)
        risk = self.aggregate_scores(instance_severity)
        if self.prediction_head == "ordinal":
            probabilities = self.ordinal_probabilities(risk)
            logits = probabilities.log().unsqueeze(0)
        else:
            logits = self.softmax_head(risk.reshape(1, 1))
            probabilities = torch.softmax(logits.squeeze(0), dim=0)
        output = {"logits": logits}
        if return_state:
            output.update(
                {
                    "risk": risk,
                    "instance_severity": instance_severity,
                    "probabilities": probabilities,
                }
            )
            if self.prediction_head == "ordinal":
                output["thresholds"] = self.ordered_thresholds()
        return output

    def compute_loss(self, bag, label, criterion):
        output = self.forward(bag, return_state=True)
        classification_loss = criterion(output["logits"], label)
        return output, {
            "loss": classification_loss,
            "classification_loss": classification_loss,
        }
