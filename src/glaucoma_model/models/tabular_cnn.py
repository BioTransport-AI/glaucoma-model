"""Tabular-only 1D-CNN baseline (Parikh, Nguyen & Penkova 2026, numeric branch).

Per the paper's own ablation, this alone reaches ~75% accuracy and ~91% when
paired with a Swin backbone -- the cheapest way to validate the data pipeline
end to end before the image branches exist. Architecture: two sequential
conv-bn-relu-maxpool blocks (no shared parameters) over the engineered
biomarkers, flattened into fully connected layers.
"""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.relu(self.bn(self.conv(x))))


class TabularCNN(nn.Module):
    """2-block 1D-CNN over the 23 engineered biomarkers, binary glaucoma output."""

    def __init__(self, num_features: int = 23, hidden_channels: int = 32, fc_dim: int = 64) -> None:
        super().__init__()
        if num_features < 4:
            raise ValueError("num_features must be at least 4 to survive two maxpool(2) layers")
        self.block1 = ConvBlock1D(1, hidden_channels)
        self.block2 = ConvBlock1D(hidden_channels, hidden_channels * 2)
        flattened_dim = (hidden_channels * 2) * (num_features // 4)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_dim, fc_dim),
            nn.BatchNorm1d(fc_dim),
            nn.ReLU(),
            nn.Linear(fc_dim, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, num_features) -> (batch, 1, num_features) for Conv1d
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        return self.classifier(x)
