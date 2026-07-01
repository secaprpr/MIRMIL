import torch

from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL
from modules.CLAM_SB_MIL.clam_sb_mil import CLAM_SB_MIL


def test_clam_models_support_bags_smaller_than_instance_sample_count():
    bag = torch.randn(1, 4, 1024)
    label = torch.tensor([2])
    models = (
        CLAM_SB_MIL(num_classes=6, k_sample=8, instance_eval=True),
        CLAM_MB_MIL(num_classes=6, k_sample=8, instance_eval=True),
    )
    for model in models:
        output = model(bag, label=label)
        assert output["logits"].shape == (1, 6)
        assert torch.isfinite(output["instance_loss"])
