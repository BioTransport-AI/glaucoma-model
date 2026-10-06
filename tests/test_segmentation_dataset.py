from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from glaucoma_model.segmentation_dataset import SegmentationDataset, build_transforms


def _make_fake_pair(tmp_path: Path, idx: int) -> tuple[str, str]:
    rng = np.random.default_rng(idx)
    image = rng.integers(0, 255, size=(96, 96, 3), dtype=np.uint8)
    mask = np.zeros((96, 96), dtype=np.uint8)
    mask[20:50, 20:50] = 255  # a plain square -- easy to sanity-check after resize

    image_path = tmp_path / f"fundus_{idx}.png"
    mask_path = tmp_path / f"mask_{idx}.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    return str(image_path), str(mask_path)


def _make_df(tmp_path: Path, n: int = 3) -> pd.DataFrame:
    rows = []
    for i in range(n):
        image_path, mask_path = _make_fake_pair(tmp_path, i)
        rows.append({"file_path": image_path, "optic_cup_mask_path": mask_path})
    return pd.DataFrame(rows)


def test_getitem_returns_expected_shapes_and_dtypes(tmp_path: Path) -> None:
    df = _make_df(tmp_path)
    ds = SegmentationDataset(df, mask_column="optic_cup_mask_path", image_size=128, train=False)

    image, mask = ds[0]
    assert image.shape == (3, 128, 128)
    assert mask.shape == (1, 128, 128)
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32


def test_image_values_normalized_to_unit_range(tmp_path: Path) -> None:
    df = _make_df(tmp_path)
    ds = SegmentationDataset(df, mask_column="optic_cup_mask_path", image_size=128, train=False)
    image, _ = ds[0]
    assert image.min() >= 0.0
    assert image.max() <= 1.0


def test_mask_stays_binary_through_resize_and_augmentation(tmp_path: Path) -> None:
    df = _make_df(tmp_path)
    # train=True exercises the full augmentation pipeline (flips, Affine rotate/
    # scale, blur, noise) -- mask_interpolation=NEAREST on every geometric op is
    # what keeps this binary; a mask with interpolation artifacts (any value
    # other than exactly 0.0/1.0) would mean a transform is using the wrong
    # interpolation mode and silently corrupting ground truth.
    ds = SegmentationDataset(df, mask_column="optic_cup_mask_path", image_size=128, train=True)
    for i in range(len(ds)):
        _, mask = ds[i]
        uniques = torch.unique(mask)
        assert set(uniques.tolist()) <= {0.0, 1.0}, f"non-binary mask values: {uniques}"


def test_len_matches_dataframe_length(tmp_path: Path) -> None:
    df = _make_df(tmp_path, n=5)
    ds = SegmentationDataset(df, mask_column="optic_cup_mask_path", train=False)
    assert len(ds) == 5


def test_missing_mask_column_raises(tmp_path: Path) -> None:
    df = _make_df(tmp_path)
    with pytest.raises(ValueError):
        SegmentationDataset(df, mask_column="not_a_real_column")


def test_val_transform_has_no_augmentation_randomness(tmp_path: Path) -> None:
    # Same image run through the val/test transform twice should be identical --
    # augmenting eval data would make reported Dice not comparable run to run,
    # let alone to the paper's own Table 2/3/4 (evaluated on clean resizes).
    df = _make_df(tmp_path, n=1)
    ds = SegmentationDataset(df, mask_column="optic_cup_mask_path", train=False)
    image_a, mask_a = ds[0]
    image_b, mask_b = ds[0]
    assert torch.equal(image_a, image_b)
    assert torch.equal(mask_a, mask_b)


def test_build_transforms_train_vs_eval_differ() -> None:
    train_t = build_transforms(train=True)
    eval_t = build_transforms(train=False)
    assert len(train_t.transforms) > len(eval_t.transforms)
