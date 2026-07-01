import torch

from modules.MAMBA2D_MIL.mamba2d_mil import Mamba2D_MIL
from process.MAMBA2D_MIL.process_mamba2d_mil import _split_spatial_bag


def test_coordinate_grid_is_compact_and_tracks_occupied_patches():
    features = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    coords = torch.tensor([[0.0, 0.0], [1000.0, 0.0], [1000.0, 5000.0]])

    grid, occupied = Mamba2D_MIL._features_to_grid(
        None, features, coords
    )

    assert grid.shape == (2, 2, 4)
    assert occupied.sum().item() == 3
    assert torch.equal(grid[0, 0], features[0])
    assert torch.equal(grid[0, 1], features[1])
    assert torch.equal(grid[1, 1], features[2])


def test_spatial_bag_separates_features_and_coordinates():
    bag = torch.randn(1, 7, 10)
    features, coords = _split_spatial_bag(bag, in_dim=8)
    assert features.shape == (1, 7, 8)
    assert coords.shape == (7, 2)
