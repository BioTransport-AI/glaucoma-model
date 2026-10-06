"""Standard U-Net for binary structure segmentation (optic disc/cup, vasculature --
Parikh, Nguyen & Penkova 2026, Sections 3.2-3.4). The paper names the architecture
family ("The U-Net model...") without specifying channel widths or depth, so this
follows the original U-Net's own design (Ronneberger et al. 2015): 4 downsampling
levels, double-conv blocks, skip connections -- just at a narrower default channel
width (32 base, not 64) since training runs on CPU here, not the paper's own compute.
Widen via `base_channels` if GPU time becomes available later.
"""

from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    """(Conv3x3 -> BatchNorm -> ReLU) x2, same padding -- the repeated unit at
    every U-Net level, both encoder and decoder sides."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """Maxpool(2) then DoubleConv -- one encoder step."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    """Transposed-conv upsample, concat with the matching encoder skip connection,
    then DoubleConv -- one decoder step."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        # Skip and upsampled feature maps are the same size whenever the input's
        # height/width are both divisible by 16 (4 maxpool(2) levels) -- the pad
        # below only matters for odd input sizes, which the pipeline never
        # actually produces (features.py always resizes to 128x128 first).
        diff_y = skip.shape[2] - x.shape[2]
        diff_x = skip.shape[3] - x.shape[3]
        if diff_y or diff_x:
            x = nn.functional.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """4-level U-Net. Input (batch, in_channels, H, W) with H and W divisible by
    16; output (batch, 1, H, W) raw logits -- apply sigmoid yourself (BCEWithLogitsLoss
    expects logits directly, and so does the Dice-loss term in train_segmentation.py)."""

    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c = base_channels
        self.in_conv = DoubleConv(in_channels, c)
        self.down1 = Down(c, c * 2)
        self.down2 = Down(c * 2, c * 4)
        self.down3 = Down(c * 4, c * 8)
        self.down4 = Down(c * 8, c * 16)
        self.up1 = Up(c * 16, c * 8)
        self.up2 = Up(c * 8, c * 4)
        self.up3 = Up(c * 4, c * 2)
        self.up4 = Up(c * 2, c)
        self.out_conv = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.out_conv(x)
