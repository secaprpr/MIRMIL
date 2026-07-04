import torch

from modules.MAMBA2D_MIL.mamba2d_mil import (
    Mamba2D_MIL,
    run_scan_in_chunks,
)
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


def test_coordinate_scale_bins_and_averages_colliding_patches():
    model = object.__new__(Mamba2D_MIL)
    model.coord_scale = 1024.0
    features = torch.tensor([[1.0, 3.0], [3.0, 5.0], [7.0, 9.0]])
    coords = torch.tensor([[0.0, 0.0], [256.0, 256.0], [1024.0, 0.0]])

    grid, occupied = model._features_to_grid(features, coords)

    assert grid.shape == (1, 2, 2)
    assert occupied.sum().item() == 2
    assert torch.equal(grid[0, 0], torch.tensor([2.0, 4.0]))
    assert torch.equal(grid[0, 1], features[2])


def test_spatial_bag_separates_features_and_coordinates():
    bag = torch.randn(1, 7, 10)
    features, coords = _split_spatial_bag(bag, in_dim=8)
    assert features.shape == (1, 7, 8)
    assert coords.shape == (7, 2)


def test_chunked_scan_preserves_outputs_and_gradients():
    layer = torch.nn.Linear(4, 4)
    full_input = torch.randn(11, 3, 4, requires_grad=True)
    chunk_input = full_input.detach().clone().requires_grad_(True)
    full = layer(full_input)
    chunked = run_scan_in_chunks(
        layer, chunk_input, chunk_size=3, training=True
    )
    assert torch.allclose(full, chunked)
    full.sum().backward()
    expected_gradient = full_input.grad.clone()
    layer.zero_grad(set_to_none=True)
    chunked.sum().backward()
    assert torch.allclose(expected_gradient, chunk_input.grad)
