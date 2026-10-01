"""Deterministic geometric biomarker extraction from glaucoma segmentation masks.

Reproduces the feature set in Parikh, Nguyen & Penkova (2026), Table 6: diameters,
areas, and circumferences extracted via Canny edge detection, contour detection,
and minimum-enclosing-circle fitting -- not learned embeddings, so the features
can't leak information from the segmentation stage into the classification stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import cv2
import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "retina_diameter",
    "optic_disc_diameter",
    "optic_cup_diameter",
    "vasculature_density",
    "macular_area",
    "retina_radius",
    "retina_area",
    "retina_circumference",
    "optic_cup_radius",
    "optic_cup_area",
    "optic_cup_circumference",
    "optic_disc_radius",
    "optic_disc_area",
    "optic_disc_circumference",
    "macular_radius",
    "macular_diameter",
    "macular_circumference",
    "optic_disc_to_retina_diameter_ratio",
    "optic_disc_to_retina_area_ratio",
    "optic_cup_to_disc_diameter_ratio",
    "optic_cup_to_disc_area_ratio",
    "optic_cup_to_retina_diameter_ratio",
    "optic_cup_to_retina_area_ratio",
)


def circle_diameter_px(mask: np.ndarray) -> float:
    """Diameter (pixels) of the minimum enclosing circle around a mask's largest contour.

    `mask` is a single-channel array where nonzero means foreground. Returns 0.0
    for an empty mask rather than raising, since a missing structure is valid
    input (e.g. a macula mask that didn't segment anything).
    """
    binary = (mask > 0).astype(np.uint8) * 255
    edges = cv2.Canny(binary, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    _, radius = cv2.minEnclosingCircle(largest)
    return 2.0 * float(radius)


def vasculature_density(mask: np.ndarray) -> float:
    """Fraction of foreground pixels in a vasculature mask.

    The paper lists this as an "Extracted" feature without giving its exact
    formula in the part of the text we have -- this is the standard definition
    used in retinal-imaging literature, not independently verified against the
    paper's own code.
    """
    if mask.size == 0:
        return 0.0
    return float(np.count_nonzero(mask)) / float(mask.size)


def _circle_area(diameter: float) -> float:
    return math.pi * (diameter / 2.0) ** 2


def _circle_circumference(diameter: float) -> float:
    return math.pi * diameter


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


@dataclass(frozen=True)
class GlaucomaFeatures:
    """The 23 features from Table 6: 5 extracted, 18 engineered."""

    retina_diameter: float
    optic_disc_diameter: float
    optic_cup_diameter: float
    vasculature_density: float
    macular_area: float

    retina_radius: float
    retina_area: float
    retina_circumference: float
    optic_cup_radius: float
    optic_cup_area: float
    optic_cup_circumference: float
    optic_disc_radius: float
    optic_disc_area: float
    optic_disc_circumference: float
    macular_radius: float
    macular_diameter: float
    macular_circumference: float
    optic_disc_to_retina_diameter_ratio: float
    optic_disc_to_retina_area_ratio: float
    optic_cup_to_disc_diameter_ratio: float
    optic_cup_to_disc_area_ratio: float
    optic_cup_to_retina_diameter_ratio: float
    optic_cup_to_retina_area_ratio: float

    def to_dict(self) -> dict[str, float]:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def extract_features(
    retina_mask: np.ndarray,
    optic_disc_mask: np.ndarray,
    optic_cup_mask: np.ndarray,
    vasculature_mask: np.ndarray,
    macula_mask: np.ndarray,
) -> GlaucomaFeatures:
    """Build the full Table 6 feature set from five binary segmentation masks."""

    retina_d = circle_diameter_px(retina_mask)
    disc_d = circle_diameter_px(optic_disc_mask)
    cup_d = circle_diameter_px(optic_cup_mask)
    macula_d = circle_diameter_px(macula_mask)

    retina_area = _circle_area(retina_d)
    disc_area = _circle_area(disc_d)
    cup_area = _circle_area(cup_d)

    return GlaucomaFeatures(
        retina_diameter=retina_d,
        optic_disc_diameter=disc_d,
        optic_cup_diameter=cup_d,
        vasculature_density=vasculature_density(vasculature_mask),
        macular_area=_circle_area(macula_d),
        retina_radius=retina_d / 2.0,
        retina_area=retina_area,
        retina_circumference=_circle_circumference(retina_d),
        optic_cup_radius=cup_d / 2.0,
        optic_cup_area=cup_area,
        optic_cup_circumference=_circle_circumference(cup_d),
        optic_disc_radius=disc_d / 2.0,
        optic_disc_area=disc_area,
        optic_disc_circumference=_circle_circumference(disc_d),
        macular_radius=macula_d / 2.0,
        macular_diameter=macula_d,
        macular_circumference=_circle_circumference(macula_d),
        optic_disc_to_retina_diameter_ratio=_safe_ratio(disc_d, retina_d),
        optic_disc_to_retina_area_ratio=_safe_ratio(disc_area, retina_area),
        optic_cup_to_disc_diameter_ratio=_safe_ratio(cup_d, disc_d),
        optic_cup_to_disc_area_ratio=_safe_ratio(cup_area, disc_area),
        optic_cup_to_retina_diameter_ratio=_safe_ratio(cup_d, retina_d),
        optic_cup_to_retina_area_ratio=_safe_ratio(cup_area, retina_area),
    )
