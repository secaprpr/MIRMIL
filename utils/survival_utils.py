import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


def fit_discrete_time_cutpoints(times, events, num_bins=4):
    """Fit RRT-style quantile cutpoints from uncensored survival times."""
    times = np.asarray(times, dtype=float)
    events = np.asarray(events, dtype=int)
    if times.ndim != 1:
        raise ValueError("times must be a 1-D array")
    if len(times) == 0:
        raise ValueError("Cannot fit survival cutpoints from an empty split")

    event_times = times[events == 1]
    source = event_times if len(event_times) >= num_bins else times
    quantiles = np.linspace(0.0, 1.0, int(num_bins) + 1)[1:-1]
    if len(quantiles) == 0:
        return []
    cutpoints = np.quantile(source, quantiles)
    return sorted(float(value) for value in np.unique(cutpoints))


def discretize_survival_times(times, cutpoints):
    times = np.asarray(times, dtype=float)
    cutpoints = np.asarray(cutpoints, dtype=float)
    return np.searchsorted(cutpoints, times, side="right").astype(np.int64)


def survival_from_hazards(hazards):
    hazards = hazards.clamp(min=1e-7, max=1.0 - 1e-7)
    return torch.cumprod(1.0 - hazards, dim=1)


def risk_from_survival(survival):
    return -torch.sum(survival, dim=1)


class NLLSurvLoss(nn.Module):
    """Discrete-time survival NLL used by RRT-MIL survival experiments.

    events use the natural convention: 1 means observed event, 0 means censored.
    """

    def __init__(self, alpha=0.0, eps=1e-7):
        super().__init__()
        self.alpha = float(alpha)
        self.eps = float(eps)

    def forward(self, hazards, survival, labels, events):
        hazards = hazards.clamp(min=self.eps, max=1.0 - self.eps)
        survival = survival.clamp(min=self.eps, max=1.0)
        labels = labels.view(-1, 1).long()
        events = events.view(-1, 1).float()
        censorship = 1.0 - events

        padded_survival = torch.cat(
            [torch.ones_like(survival[:, :1]), survival], dim=1
        )
        s_before = torch.gather(padded_survival, 1, labels).clamp_min(self.eps)
        s_after = torch.gather(
            padded_survival, 1, labels + 1
        ).clamp_min(self.eps)
        h_this = torch.gather(hazards, 1, labels).clamp_min(self.eps)

        uncensored_loss = -events * (torch.log(s_before) + torch.log(h_this))
        censored_loss = -censorship * torch.log(s_after)
        loss = censored_loss + uncensored_loss
        if self.alpha > 0:
            loss = (1.0 - self.alpha) * loss + self.alpha * uncensored_loss
        return loss.mean()


@dataclass
class ConcordanceResult:
    c_index: float
    source: str


def concordance_index(event_times, events, risks, prefer_sksurv=True):
    """Harrell's C-index with higher risk meaning shorter predicted survival."""
    event_times = np.asarray(event_times, dtype=float)
    events = np.asarray(events, dtype=bool)
    risks = np.asarray(risks, dtype=float)
    if prefer_sksurv:
        try:
            from sksurv.metrics import concordance_index_censored

            value = concordance_index_censored(events, event_times, risks)[0]
            return ConcordanceResult(float(value), "scikit-survival")
        except Exception:
            pass

    return ConcordanceResult(
        _harrell_concordance_index(event_times, events, risks),
        "internal",
    )


def _harrell_concordance_index(event_times, events, risks):
    event_times = np.asarray(event_times, dtype=float)
    events = np.asarray(events, dtype=bool)
    risks = np.asarray(risks, dtype=float)

    comparable = 0
    concordant = 0.0
    for i in range(len(event_times)):
        for j in range(i + 1, len(event_times)):
            if event_times[i] == event_times[j]:
                continue
            if event_times[i] < event_times[j] and events[i]:
                comparable += 1
                concordant += _compare_risk(risks[i], risks[j])
            elif event_times[j] < event_times[i] and events[j]:
                comparable += 1
                concordant += _compare_risk(risks[j], risks[i])

    if comparable == 0:
        return math.nan
    return concordant / comparable


def bootstrap_c_index(
    event_times,
    events,
    risks,
    n_bootstraps=1000,
    confidence=0.95,
    seed=2024,
    prefer_sksurv=True,
):
    event_times = np.asarray(event_times, dtype=float)
    events = np.asarray(events, dtype=bool)
    risks = np.asarray(risks, dtype=float)
    if len(event_times) < 2 or int(n_bootstraps or 0) <= 0:
        return math.nan, math.nan

    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(n_bootstraps)):
        indices = rng.integers(0, len(event_times), size=len(event_times))
        if len(np.unique(event_times[indices])) < 2 or not np.any(events[indices]):
            continue
        value = concordance_index(
            event_times[indices],
            events[indices],
            risks[indices],
            prefer_sksurv=prefer_sksurv,
        ).c_index
        if not math.isnan(value):
            values.append(value)

    if not values:
        return math.nan, math.nan
    alpha = (1.0 - float(confidence)) / 2.0
    return (
        float(np.quantile(values, alpha)),
        float(np.quantile(values, 1.0 - alpha)),
    )


def _compare_risk(shorter_risk, longer_risk):
    if shorter_risk > longer_risk:
        return 1.0
    if shorter_risk == longer_risk:
        return 0.5
    return 0.0
