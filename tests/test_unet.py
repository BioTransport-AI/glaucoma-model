from __future__ import annotations

import torch

from glaucoma_model.models import UNet


def test_forward_output_shape_matches_input_spatial_size() -> None:
    model = UNet(in_channels=3, base_channels=4)
    x = torch.randn(2, 3, 128, 128)
    out = model(x)
    assert out.shape == (2, 1, 128, 128)


def test_forward_produces_finite_values() -> None:
    model = UNet(in_channels=3, base_channels=4).eval()
    x = torch.randn(2, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert torch.isfinite(out).all()


def test_forward_works_on_smaller_divisible_by_16_input() -> None:
    model = UNet(in_channels=3, base_channels=4)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_batch_size_one_works_in_train_mode_unlike_tabular_cnn() -> None:
    # Unlike BatchNorm1d on TabularCNN's (batch, channels) tensors (see
    # test_tabular_cnn.py -- batch=1 there has one value per channel, zero
    # variance, and PyTorch raises), BatchNorm2d normalizes over (batch x H x W)
    # per channel. A batch of 1 still has H*W>1 spatial positions to compute
    # variance over, so train mode works fine here -- checked empirically, not
    # assumed by analogy to the 1D case, since that assumption turned out wrong.
    model = UNet(in_channels=3, base_channels=4)
    x = torch.randn(1, 3, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)
    assert torch.isfinite(out).all()
