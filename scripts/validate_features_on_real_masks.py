"""Validate extract_features() against SMDG-19's real ground-truth masks.

These are the masks used to TRAIN the paper's segmentation models (the
`used_for_segmentation` rows the classification split excludes) -- using
them here to sanity-check the feature-extraction code against real images
instead of just synthetic circles, before any segmentation model exists to
produce masks for the rest of the dataset.

Two real limitations surfaced by the data itself, not bugs in this script:
  - Optic disc/cup masks and blood-vessel masks come from disjoint sets of
    images (zero overlap) -- SMDG-19 draws them from different source
    datasets within its standardized 19. Validated separately below.
  - No macula masks exist anywhere in SMDG-19 -- those come from the paper's
    custom-trained SegFormer, which doesn't exist yet. Not validated here.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from glaucoma_model.features import (
    extract_features,
    mask_diameter_px,
    retina_diameter_px,
    vasculature_density,
)

RAW = Path(__file__).resolve().parents[1] / "data" / "smdg-19-glaucoma"
INDEX_PATH = RAW / "index.parquet"
FUNDUS_DIR = RAW / "full-fundus/full-fundus"
DISC_DIR = RAW / "optic-disc/optic-disc"
CUP_DIR = RAW / "optic-cup/optic-cup"
VESSEL_DIR = RAW / "blood-vessel/blood-vessel"


def _load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def validate_disc_cup(index: pd.DataFrame) -> pd.DataFrame:
    disc_ids = {p.stem for p in DISC_DIR.glob("*")}
    cup_ids = {p.stem for p in CUP_DIR.glob("*")}
    usable = index[index["image_id"].isin(disc_ids & cup_ids)]

    rows = []
    for _, row in usable.iterrows():
        image_id = row["image_id"]
        fundus = _load_gray(FUNDUS_DIR / f"{image_id}.png")
        disc = _load_gray(DISC_DIR / f"{image_id}.png")
        cup = _load_gray(CUP_DIR / f"{image_id}.png")

        retina_d = retina_diameter_px(fundus)
        disc_d = mask_diameter_px(disc)
        cup_d = mask_diameter_px(cup)
        rows.append(
            {
                "image_id": image_id,
                "label": row["label"],
                "retina_diameter_px": retina_d,
                "optic_disc_diameter_px": disc_d,
                "optic_cup_diameter_px": cup_d,
                "cup_to_disc_diameter_ratio": cup_d / disc_d if disc_d else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def validate_vasculature() -> pd.DataFrame:
    rows = []
    for p in sorted(VESSEL_DIR.glob("*")):
        mask = _load_gray(p)
        rows.append({"image_id": p.stem, "vasculature_density": vasculature_density(mask)})
    return pd.DataFrame(rows)


def main() -> None:
    index = pd.read_parquet(INDEX_PATH)

    print("=== Disc/cup/retina, on real ground-truth masks ===")
    disc_cup = validate_disc_cup(index)
    print(f"{len(disc_cup)} images with both disc and cup masks available")
    print(
        disc_cup[
            ["retina_diameter_px", "optic_disc_diameter_px", "optic_cup_diameter_px", "cup_to_disc_diameter_ratio"]
        ].describe()
    )

    print()
    print("=== Cup-to-disc ratio by label, vs. the paper's Table 7 ===")
    print("Paper reports: glaucoma 0.602, non-glaucoma 0.498 (unitless ratio -- no px/mm calibration needed to compare)")
    by_label = disc_cup.groupby("label")["cup_to_disc_diameter_ratio"].agg(["mean", "std", "count"])
    print(by_label)

    print()
    print("=== Vasculature density, on real ground-truth masks ===")
    vessel = validate_vasculature()
    print(f"{len(vessel)} images with blood-vessel masks available")
    print(vessel["vasculature_density"].describe())

    print()
    print("=== Full extract_features() wrapper, 3 real samples ===")
    print("(vessel/macula zeroed -- no SMDG-19 images have vessel+disc+cup together, and no macula masks exist at all)")
    zero_fill = np.zeros((10, 10), dtype=np.uint8)
    for image_id in disc_cup["image_id"].head(3):
        fundus = _load_gray(FUNDUS_DIR / f"{image_id}.png")
        disc = _load_gray(DISC_DIR / f"{image_id}.png")
        cup = _load_gray(CUP_DIR / f"{image_id}.png")
        feats = extract_features(fundus, disc, cup, zero_fill, zero_fill)
        print(f"  {image_id}: cup/disc ratio = {feats.optic_cup_to_disc_diameter_ratio:.3f}, "
              f"disc area = {feats.optic_disc_area:.1f}, field count = {len(feats.to_dict())}")


if __name__ == "__main__":
    main()
