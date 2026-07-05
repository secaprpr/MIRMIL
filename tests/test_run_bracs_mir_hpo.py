from argparse import Namespace
from pathlib import Path

from experiments.run_bracs_mir_hpo import VARIANTS, command


def test_hpo_variants_preserve_mir_parameter_shapes(tmp_path):
    forbidden = {
        "Model.hidden_dim",
        "Model.sketch_dim",
        "Model.num_local_routes",
        "Model.local_route_dim",
        "Model.prototype_embedding_dim",
        "Model.prototypes_per_class",
    }
    nonstructural = [
        name for name in VARIANTS
        if name.startswith(("lr1e4_", "lr2e4_", "lr5e5_"))
    ]
    for name in nonstructural:
        spec = VARIANTS[name]
        assert forbidden.isdisjoint(spec["options"])

    args = Namespace(
        python=Path("/env/python"),
        split=Path("/data/split.csv"),
        seed=2024,
        epochs=40,
        patience=10,
        max_instances=4096,
        num_workers=4,
        wandb=False,
        wandb_project="MIR-MIL",
    )
    result = command(
        args, "trial", VARIANTS["lr1e4_mild"], tmp_path
    )
    assert "--no-balanced" not in result
    assert "Model.optimizer.adamw_config.lr=0.0001" in result
