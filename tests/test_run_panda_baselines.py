from argparse import Namespace
from pathlib import Path

from experiments.run_panda_baselines import build_tasks


def test_matrix_uses_h5_only_for_spatial_model(tmp_path):
    args = Namespace(
        features=["r50"],
        seeds=[2024],
        models=["AB_MIL", "MAMBA2D_MIL", "C2AUG"],
        output_dir=tmp_path,
        metadata_dir=Path("/metadata"),
        feature_root=Path("/features"),
        project_root=Path("/project"),
        c2aug_dir=Path("/project/baselines/c2aug"),
        python=Path("/env/python"),
        epochs=30,
        patience=8,
        max_instances=4096,
        num_workers=4,
        protocol="protocol-v1",
        wandb_project="MIR-MIL",
    )
    hashes = {
        "source": "s",
        "coordinates": "c",
        "r50_feature": "f",
        "r50_checkpoint": "w",
        "r50_split": "p",
    }
    tasks = build_tasks(args, hashes)
    commands = {task["model"]: task["command"] for task in tasks}
    assert "train_val_qc_h5.csv" in " ".join(commands["MAMBA2D_MIL"])
    assert "train_val_qc.csv" in " ".join(commands["AB_MIL"])
    assert commands["C2AUG"][commands["C2AUG"].index("--views") + 1] == "2"
