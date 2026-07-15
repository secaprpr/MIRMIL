import torch
import torch.nn as nn

from utils.survival_utils import risk_from_survival, survival_from_hazards


class SurvivalMILWrapper(nn.Module):
    """Attach a discrete-time hazard head to an arbitrary MIL backbone."""

    def __init__(
        self,
        backbone,
        num_bins=4,
        representation="auto",
        head_hidden_dim=0,
        dropout=0.0,
    ):
        super().__init__()
        self.backbone = backbone
        self.num_bins = int(num_bins)
        self.representation = str(representation or "auto")

        layers = []
        if dropout:
            layers.append(nn.Dropout(float(dropout)))
        if int(head_hidden_dim or 0) > 0:
            layers.extend(
                [
                    nn.LazyLinear(int(head_hidden_dim)),
                    nn.ReLU(),
                    nn.Dropout(float(dropout)) if dropout else nn.Identity(),
                    nn.Linear(int(head_hidden_dim), self.num_bins),
                ]
            )
        else:
            layers.append(nn.LazyLinear(self.num_bins))
        self.hazard_head = nn.Sequential(*layers)

    def forward(self, x):
        output = self._forward_backbone(x)
        representation = self._select_representation(output)
        if representation.dim() == 1:
            representation = representation.unsqueeze(0)
        representation = representation.flatten(start_dim=1)
        hazard_logits = self.hazard_head(representation)
        hazards = torch.sigmoid(hazard_logits)
        survival = survival_from_hazards(hazards)
        risk = risk_from_survival(survival)
        return {
            "logits": hazard_logits,
            "hazards": hazards,
            "survival": survival,
            "risk": risk,
            "backbone_output": output,
        }

    def _forward_backbone(self, x):
        try:
            return self.backbone(x, return_WSI_feature=True)
        except TypeError:
            return self.backbone(x)

    def _select_representation(self, output):
        if isinstance(output, dict):
            if self.representation in output:
                return output[self.representation]
            if self.representation == "auto" and "WSI_feature" in output:
                return output["WSI_feature"]
            if "logits" in output:
                return output["logits"]
            raise KeyError(
                "Backbone output dict must contain logits or the configured "
                f"representation={self.representation}"
            )
        if isinstance(output, (tuple, list)):
            return output[0]
        return output
