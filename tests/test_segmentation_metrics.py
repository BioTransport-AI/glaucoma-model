from __future__ import annotations

import pytest
import torch

from glaucoma_model.segmentation_metrics import (
    bce_dice_loss,
    dice_coefficient,
    dice_loss,
    iou_score,
    pixel_accuracy,
)


def _logits_for(mask: torch.Tensor, confidence: float = 10.0) -> torch.Tensor:
    """Build logits that sigmoid to ~0/1 matching `mask` exactly -- lets tests
    assert on "perfect prediction" cases without hand-deriving a separate
    logit tensor each time."""
    return (mask * 2 - 1) * confidence


def test_dice_and_iou_are_perfect_for_exact_prediction() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0:2, 0:2] = 1.0
    logits = _logits_for(target)

    assert dice_coefficient(logits, target) == pytest.approx(1.0, abs=1e-4)
    assert iou_score(logits, target) == pytest.approx(1.0, abs=1e-4)
    assert pixel_accuracy(logits, target) == pytest.approx(1.0, abs=1e-4)


def test_dice_and_iou_are_zero_for_disjoint_prediction() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0:2, 0:2] = 1.0
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 2:4, 2:4] = 1.0
    logits = _logits_for(pred)

    assert dice_coefficient(logits, target) == pytest.approx(0.0, abs=1e-4)
    assert iou_score(logits, target) == pytest.approx(0.0, abs=1e-4)


def test_dice_matches_hand_computed_partial_overlap() -> None:
    # 4x4 grid. Target is the left half (8 px). Pred is the top half (8 px).
    # Overlap is the top-left 2x4 quadrant (4 px).
    # Dice = 2*|A∩B| / (|A|+|B|) = 2*4 / (8+8) = 0.5
    # IoU  = |A∩B| / |A∪B|       = 4 / (8+8-4) = 4/12 = 0.3333...
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, :, 0:2] = 1.0
    pred = torch.zeros(1, 1, 4, 4)
    pred[0, 0, 0:2, :] = 1.0
    logits = _logits_for(pred)

    assert dice_coefficient(logits, target) == pytest.approx(0.5, abs=1e-3)
    assert iou_score(logits, target) == pytest.approx(4 / 12, abs=1e-3)


def test_pixel_accuracy_counts_background_agreement_too() -> None:
    # Both masks empty -- every pixel (all background) agrees, accuracy should
    # be 1.0 even though Dice's own numerator/denominator would both be zero
    # (handled by the eps smoothing, not asserted here).
    target = torch.zeros(1, 1, 4, 4)
    logits = _logits_for(target)
    assert pixel_accuracy(logits, target) == pytest.approx(1.0, abs=1e-4)


def test_dice_loss_is_near_zero_for_confident_correct_prediction() -> None:
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    logits = _logits_for(target, confidence=15.0)
    loss = dice_loss(logits, target)
    assert loss.item() < 0.01


def test_dice_loss_is_near_one_for_confidently_wrong_prediction() -> None:
    target = torch.zeros(1, 1, 8, 8)
    target[:, :, 2:6, 2:6] = 1.0
    wrong = 1.0 - target
    logits = _logits_for(wrong, confidence=15.0)
    loss = dice_loss(logits, target)
    assert loss.item() > 0.95


def test_dice_loss_is_differentiable() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0:2, 0:2] = 1.0
    logits = torch.randn(1, 1, 4, 4, requires_grad=True)
    loss = dice_loss(logits, target)
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_bce_dice_loss_combines_both_terms() -> None:
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0:2, 0:2] = 1.0
    logits = torch.randn(1, 1, 4, 4)

    combined = bce_dice_loss(logits, target)
    dice_only = dice_loss(logits, target)

    # Not a tight bound -- just confirms bce_dice_loss isn't silently equal to
    # (or smaller than) the dice term alone, which would mean the BCE half got
    # dropped somewhere.
    assert combined.item() > dice_only.item()
