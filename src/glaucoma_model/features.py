"""Deterministic geometric biomarker extraction (Parikh, Nguyen & Penkova 2026, Table 6).

Two input types, matching the paper's actual pipeline -- not one function
applied uniformly to five masks:

- The retinal boundary comes from Canny edge detection directly on the raw
  fundus photograph (paper section 3.6). There is no dedicated retina
  segmentation model in the architecture -- optic disc, optic cup, and
  vasculature come from U-Net, the macula from a custom SegFormer, and that's
  the full list of trained segmentation models.
- Optic disc, optic cup, vasculature, and macula masks are already clean
  binary segmentation-model outputs, so their boundaries come from contour
  detection directly, with no edge-detection pass.

Diameters/areas are deterministic geometry, not learned embeddings, so they
can't leak information from the segmentation stage into the classification
stage.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import cv2
import numpy as np


def retina_diameter_px(fundus_image: np.ndarray) -> float:
    """Diameter (pixels) of the retinal field in a raw fundus photograph.

    `fundus_image` is the original photo (grayscale or BGR, as loaded by
    `cv2.imread`) -- NOT a segmentation mask. Canny edge detection -> contour
    detection -> minimum enclosing circle, per the paper's own pipeline.
    """
    gray = fundus_image if fundus_image.ndim == 2 else cv2.cvtColor(fundus_image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    _, radius = cv2.minEnclosingCircle(largest)
    return 2.0 * float(radius)


def mask_diameter_px(mask: np.ndarray) -> float:
    """Diameter (pixels) of the minimum enclosing circle around a binary mask's
    largest contour -- for segmentation-model outputs (disc/cup/macula), which
    are already clean binary regions and need no edge-detection pass first.

    Returns 0.0 for an empty mask (a structure that didn't segment) rather
    than raising.
    """
    binary = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(contours, key=cv2.contourArea)
    _, radius = cv2.minEnclosingCircle(largest)
    return 2.0 * float(radius)


def vasculature_density(mask: np.ndarray) -> float:
    """Fraction of foreground pixels in a vasculature mask.

    The paper lists this as an "Extracted" feature without giving its exact
    formula in the part of the text we have -- this is the standard
    definition used in retinal-imaging literature, not independently
    verified against the paper's own code.
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


# Derived from the dataclass itself, not hand-duplicated -- a field added or
# renamed on GlaucomaFeatures can't silently drift out of sync with this.
FEATURE_NAMES: tuple[str, ...] = tuple(f.name for f in fields(GlaucomaFeatures))


def features_to_vector(features: GlaucomaFeatures) -> list[float]:
    """Order a GlaucomaFeatures instance into the fixed column order models expect.

    Returns a plain list, not a torch.Tensor -- this module has no hard
    dependency on torch; wrap with `torch.tensor(...)` at the call site.
    """
    values = features.to_dict()
    return [values[name] for name in FEATURE_NAMES]


def extract_features(
    fundus_image: np.ndarray,
    optic_disc_mask: np.ndarray,
    optic_cup_mask: np.ndarray,
    vasculature_mask: np.ndarray,
    macula_mask: np.ndarray,
) -> GlaucomaFeatures:
    """Build the full Table 6 feature set.

    `fundus_image` is the raw (or grayscale) photograph, used for the
    retinal boundary. The other four are binary segmentation masks from
    their respective trained models (U-Net x3, SegFormer for the macula).
    """

    retina_d = retina_diameter_px(fundus_image)
    disc_d = mask_diameter_px(optic_disc_mask)
    cup_d = mask_diameter_px(optic_cup_mask)
    macula_d = mask_diameter_px(macula_mask)

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
