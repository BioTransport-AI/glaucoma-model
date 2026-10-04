"""Tests for the classification-split loader, specifically the suspect-label filter.

Split in two: a fast unit test against a synthetic index (no real data needed,
always runs), and an integration test against the real fetched SMDG-19 data
that pins the exact row count the paper reports -- skipped automatically if
the dataset hasn't been fetched on this machine.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from glaucoma_model.data import SUSPECT_LABEL, VALID_LABELS, load_classification_split

REAL_INDEX = Path(__file__).resolve().parents[1] / "data" / "smdg-19-glaucoma" / "index.parquet"


def test_valid_labels_excludes_suspect_constant() -> None:
    assert SUSPECT_LABEL not in VALID_LABELS
    assert set(VALID_LABELS) == {0, 1}


def test_load_classification_split_drops_suspect_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = pd.DataFrame(
        {
            "image_id": ["a", "b", "c", "d"],
            "label": [0, 1, -1, 0],
            "used_for_segmentation": [False, False, False, False],
        }
    )
    monkeypatch.setattr("glaucoma_model.data.load_split", lambda *a, **k: fake)

    result = load_classification_split()

    assert len(result) == 3
    assert set(result["label"]) == {0, 1}
    assert "c" not in set(result["image_id"])


@pytest.mark.skipif(not REAL_INDEX.exists(), reason="SMDG-19 not fetched on this machine")
def test_load_classification_split_matches_paper_exactly() -> None:
    """The number this whole module exists for. If this ever stops being
    exactly 8818, something upstream changed -- a dataset version, the
    leakage logic, or the label filter -- and it needs investigating, not
    silently re-pinning to whatever the new number is."""
    df = load_classification_split()
    assert len(df) == 8818
    assert set(df["label"].unique()) == {0, 1}
