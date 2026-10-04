"""Sanity tests for geometric feature extraction against synthetic masks/images.

Uses drawn circles of known radius (not real fundus images) so the expected
diameter is known exactly -- these catch extraction bugs, they don't validate
against the paper's real data since that requires SMDG-19 to be fetched first.
"""

from __future__ import annotations

import numpy as np
import pytest

from glaucoma_model.features import (
    FEATURE_NAMES,
    GlaucomaFeatures,
    extract_features,
    features_to_vector,
    mask_diameter_px,
    retina_diameter_px,
    vasculature_density,
)


def _draw_circle_mask(size: int, radius: int) -> np.ndarray:
    import cv2

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(mask, (size // 2, size // 2), radius, color=1, thickness=-1)
    return mask


def _draw_circle_image(size: int, radius: int) -> np.ndarray:
    """A grayscale image with a bright filled circle on a dark background --
    stands in for a fundus photo's retinal field for `retina_diameter_px`,
    which expects image intensities, not an already-binary mask."""
    import cv2

    image = np.zeros((size, size), dtype=np.uint8)
    cv2.circle(image, (size // 2, size // 2), radius, color=255, thickness=-1)
    return image


@pytest.mark.parametrize("radius", [20, 50, 100])
def test_mask_diameter_px_matches_known_radius(radius: int) -> None:
    mask = _draw_circle_mask(size=radius * 4, radius=radius)
    detected = mask_diameter_px(mask)
    assert detected == pytest.approx(2 * radius, rel=0.1)


@pytest.mark.parametrize("radius", [20, 50, 100])
def test_retina_diameter_px_matches_known_radius(radius: int) -> None:
    image = _draw_circle_image(size=radius * 4, radius=radius)
    detected = retina_diameter_px(image)
    assert detected == pytest.approx(2 * radius, rel=0.1)


def test_mask_diameter_px_empty_mask_returns_zero() -> None:
    mask = np.zeros((50, 50), dtype=np.uint8)
    assert mask_diameter_px(mask) == 0.0


def test_vasculature_density_bounds() -> None:
    empty = np.zeros((10, 10), dtype=np.uint8)
    full = np.ones((10, 10), dtype=np.uint8)
    assert vasculature_density(empty) == 0.0
    assert vasculature_density(full) == 1.0


def test_extract_features_cup_smaller_than_disc_gives_ratio_below_one() -> None:
    fundus = _draw_circle_image(400, 150)
    disc = _draw_circle_mask(400, 40)
    cup = _draw_circle_mask(400, 20)
    vessels = np.random.default_rng(0).integers(0, 2, size=(400, 400)).astype(np.uint8)
    macula = _draw_circle_mask(400, 15)

    features = extract_features(fundus, disc, cup, vessels, macula)

    assert 0 < features.optic_cup_to_disc_diameter_ratio < 1
    assert 0 < features.optic_disc_to_retina_diameter_ratio < 1
    assert len(features.to_dict()) == 23


def test_features_to_vector_matches_feature_names_order() -> None:
    """A wrong order here wouldn't crash anything -- it'd just silently train
    a model on garbled features. That makes this the one test worth being
    paranoid about."""
    values = {name: float(i) for i, name in enumerate(FEATURE_NAMES)}
    features = GlaucomaFeatures(**values)

    vector = features_to_vector(features)

    assert vector == [values[name] for name in FEATURE_NAMES]
    assert len(vector) == len(FEATURE_NAMES) == 23
    for name, value in zip(FEATURE_NAMES, vector):
        assert getattr(features, name) == value
