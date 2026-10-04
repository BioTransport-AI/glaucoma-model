"""Dataset loading for the glaucoma classification task.

Thin wrapper over biotransport_datasets.load_split() that applies the one
filter the paper's pipeline needs but SMDG-19's raw index doesn't encode:
dropping glaucoma-suspect rows (label == -1). The paper's architecture is a
strict 2-class problem -- nothing in its methodology, benchmark tables, or
confusion matrix mentions a third class, and TabularCNN's output head is a
2-way classifier. A -1 target against a 2-way CrossEntropyLoss head is
undefined behavior, not a loud failure -- exactly the kind of thing that's
obvious in hindsight and easy to hit face-first otherwise.

This isn't a guess dressed up as a fix. The paper states a starting count of
12,316 samples before leakage-exclusion. That's exactly 12,449 (today's full
SMDG-19) minus the 133 suspect rows -- and applying leakage-exclusion to
that reduced set lands at exactly the paper's reported 8,818, not an
approximation. See test_data.py for the regression test on that exact
number.
"""

from __future__ import annotations

import pandas as pd
from biotransport_datasets import load_split

SUSPECT_LABEL = -1
VALID_LABELS = (0, 1)

DATASET_NAME = "smdg-19-glaucoma"


def load_classification_split(exclude_segmentation_leakage: bool = True) -> pd.DataFrame:
    """The glaucoma classification split: leakage-safe, suspect-excluded, 2-class only.

    Matches the paper's reported 8,818-sample set exactly with the default
    exclude_segmentation_leakage=True.
    """
    df = load_split(DATASET_NAME, exclude_segmentation_leakage=exclude_segmentation_leakage)
    return df[df["label"].isin(VALID_LABELS)].reset_index(drop=True)
