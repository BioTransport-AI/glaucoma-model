"""Loss and evaluation metrics for the U-Net segmentation stage, split out from
train_segmentation.py so they're independently testable -- a silent bug in Dice
math would show up as "the model trained fine but the numbers don't mean what
Table 2/3/4 think they mean," which is a much worse failure mode than a loud
test failure.

All functions take raw logits (pre-sigmoid) for predictions, matching
nn.BCEWithLogitsLoss's convention -- apply sigmoid internally, never expect the
caller to have already done it (that's the easiest way to silently double-apply
or skip the activation).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

EPS = 1e-6


def dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Soft Dice loss (1 - Dice), on sigmoid probabilities, not binarized --
    needs to stay differentiable for training. See dice_coefficient() for the
    binarized version used at evaluation time."""
    probs = torch.sigmoid(logits).flatten(1)
    targets = targets.flatten(1)
    intersection = (probs * targets).sum(dim=1)
    union = probs.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + EPS) / (union + EPS)
    return 1.0 - dice.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """BCE + Dice, equal-weighted sum. The paper (Section 3.2) describes
    "combining these two loss functions (BCE-Dice loss)" without stating an
    exact weighting -- equal weight is the standard default for this
    combination in the segmentation literature, not independently verified
    against the paper's own code."""
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return bce + dice_loss(logits, targets)


def _binarize(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (torch.sigmoid(logits) > threshold).float()


def dice_coefficient(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Binarized Dice coefficient -- matches the "Dice coefficient" column in
    the paper's Tables 2/3/4, which report a single evaluation number, not a
    differentiable loss."""
    preds = _binarize(logits, threshold).flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    union = preds.sum(dim=1) + targets.sum(dim=1)
    dice = (2.0 * intersection + EPS) / (union + EPS)
    return dice.mean().item()


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Binarized Intersection-over-Union -- matches the "IoU" column in the
    paper's Tables 2/3/4."""
    preds = _binarize(logits, threshold).flatten(1)
    targets = targets.flatten(1)
    intersection = (preds * targets).sum(dim=1)
    union = ((preds + targets) > 0).float().sum(dim=1)
    iou = (intersection + EPS) / (union + EPS)
    return iou.mean().item()


def pixel_accuracy(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    """Fraction of correctly classified pixels -- matches the "Accuracy" column
    in the paper's Tables 2/3/4. Dominated by background pixels for a small
    structure like the optic cup (that's exactly why Dice/IoU, not accuracy,
    are the metrics that actually distinguish a good mask from a blank one)."""
    preds = _binarize(logits, threshold)
    correct = (preds == targets).float().mean()
    return correct.item()
