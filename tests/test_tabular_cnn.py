"""Shape and behavior tests for the tabular 1D-CNN baseline.

No real training data needed -- these just confirm the architecture is wired
correctly (right output shape, no NaN/Inf, handles the feature count we
actually have) before any training loop depends on it.
"""

from __future__ import annotations

import pytest
import torch

from glaucoma_model.models import TabularCNN


def test_forward_output_shape_matches_batch_size() -> None:
    model = TabularCNN(num_features=23)
    x = torch.randn(8, 23)
    out = model(x)
    assert out.shape == (8, 2)


def test_forward_produces_finite_values() -> None:
    model = TabularCNN(num_features=23)
    x = torch.randn(8, 23)
    out = model(x)
    assert torch.isfinite(out).all()


def test_rejects_too_few_features() -> None:
    with pytest.raises(ValueError):
        TabularCNN(num_features=3)


def test_batch_size_one_fails_in_train_mode_but_works_in_eval() -> None:
    """BatchNorm1d needs more than one sample per channel to compute batch
    statistics while training -- this is inherent to the paper's own
    architecture, not a bug here, but it's the kind of thing that's
    confusing to hit by surprise later. Documented via a test instead of a
    comment so it can't silently go stale.
    """
    model = TabularCNN(num_features=23)
    single = torch.randn(1, 23)

    model.train()
    with pytest.raises(ValueError):
        model(single)

    model.eval()
    out = model(single)
    assert out.shape == (1, 2)
