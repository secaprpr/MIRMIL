from argparse import Namespace
from pathlib import Path

import h5py
import pandas as pd
import torch

from experiments.extract_panda_features import audit_outputs, build_command


def test_build_command_assigns_disjoint_shard(tmp_path):
    args = Namespace(
        encoder="r50",
        python=Path("/env/bin/python"),
        project_root=Path("/project"),
        patch_dir=Path("/patches"),
        source_csv=Path("/source.csv"),
        output_dir=tmp_path,
        batch_size=64,
        num_workers=2,
        weights_dir=Path("/weights"),
    )
    command = build_command(args, shard_index=3, num_shards=8)
    assert command[command.index("--shard_index") + 1] == "3"
    assert command[command.index("--num_shards") + 1] == "8"
    assert command[command.index("--backbone") + 1] == "resnet50_imagenet"


def test_audit_outputs_checks_feature_coordinate_and_pt_shapes(tmp_path):
    patch_dir = tmp_path / "patch"
    output_dir = tmp_path / "features"
    (patch_dir / "patches").mkdir(parents=True)
    (output_dir / "h5_files").mkdir(parents=True)
    (output_dir / "pt_files").mkdir()
    source_csv = tmp_path / "source.csv"
    pd.DataFrame({"wsi_path": ["/wsi/case.tiff"]}).to_csv(
        source_csv, index=False
    )
    coords = torch.tensor([[0, 0], [256, 0]]).numpy()
    features = torch.ones(2, 4)
    with h5py.File(patch_dir / "patches/case.h5", "w") as file:
        file.create_dataset("coords", data=coords)
    with h5py.File(output_dir / "h5_files/case.h5", "w") as file:
        file.create_dataset("coords", data=coords)
        file.create_dataset("features", data=features.numpy())
    torch.save(features, output_dir / "pt_files/case.pt")

    audit = audit_outputs(source_csv, patch_dir, output_dir, expected_dim=4)
    assert audit.iloc[0]["status"] == "ok"
