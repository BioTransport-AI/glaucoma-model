"""Paired fundus-image / binary-mask dataset for training the U-Net structures
(optic disc, optic cup, vasculature -- Parikh, Nguyen & Penkova 2026, Sections
3.2-3.4). Resize-to-128x128 and the augmentation list (both sections, applied to
the training split only) are taken directly from the paper's own description:
horizontal flip, vertical flip, random rotation by 90 degrees, shift-scale-rotation
by 15 degrees, random brightness/contrast, Gaussian blur, Gaussian noise, and
hue/saturation jitter.

Uses albumentations rather than torchvision.transforms because the image and its
mask must receive the *same* random geometric transform (flip the image but not
the mask, or vice versa, and every pixel-level biomarker downstream is wrong) --
albumentations' `Compose(..., image=..., mask=...)` guarantees that; synchronizing
two independent torchvision transform calls correctly is easy to get subtly wrong.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

IMAGE_SIZE = 128


def build_transforms(image_size: int = IMAGE_SIZE, train: bool = True) -> A.Compose:
    """Train: paper's augmentation list after a resize. Val/test: resize only --
    augmenting evaluation data would make the reported Dice/IoU not comparable
    to the paper's own Table 2/3/4 numbers, which evaluate on clean resizes."""
    if not train:
        return A.Compose([A.Resize(image_size, image_size)])

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(scale=(0.85, 1.15), translate_percent=(-0.15, 0.15), rotate=(-15, 15), p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.GaussianBlur(p=0.3),
            A.GaussNoise(p=0.3),
            A.HueSaturationValue(p=0.3),
        ]
    )


class SegmentationDataset(Dataset):
    """One row per (fundus image, structure mask) pair. `df` must have `file_path`
    and a mask-path column (see glaucoma_model.data.SEGMENTATION_STRUCTURES) --
    normally the output of `load_segmentation_split()`, already filtered to rows
    that actually have a mask for this structure.

    Returns (image: float32 tensor [3, H, W] in [0, 1], mask: float32 tensor
    [1, H, W] in {0, 1}).
    """

    def __init__(
        self,
        df: pd.DataFrame,
        mask_column: str,
        image_size: int = IMAGE_SIZE,
        train: bool = True,
        transform: Optional[A.Compose] = None,
    ) -> None:
        if mask_column not in df.columns:
            raise ValueError(f"{mask_column!r} not in dataframe columns: {list(df.columns)}")
        self.df = df.reset_index(drop=True)
        self.mask_column = mask_column
        self.transform = transform or build_transforms(image_size, train=train)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.df.iloc[idx]

        image_bgr = cv2.imread(row["file_path"], cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(f"Could not read fundus image at {row['file_path']}")
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        mask_path = row[self.mask_column]
        mask_raw = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise FileNotFoundError(f"Could not read mask at {mask_path}")
        mask = (mask_raw > 0).astype(np.uint8)

        transformed = self.transform(image=image, mask=mask)
        image_t = torch.from_numpy(transformed["image"]).permute(2, 0, 1).float() / 255.0
        mask_t = torch.from_numpy(transformed["mask"]).float().unsqueeze(0)
        return image_t, mask_t
