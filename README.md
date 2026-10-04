# Glaucoma Model

Reproducing and extending the multimodal glaucoma classification pipeline from
[Parikh, Nguyen & Penkova (2026), *Multimodal glaucoma classification using
segmentation based biomarker extraction*, Knowledge-Based Systems
349](https://doi.org/10.1016/j.knosys.2026.116449) — Dr. Penkova is corresponding
author. Full paper PDF lives in the `biotransport-ai` repo's `papers/` directory.

Background, full architecture detail, and the live development plan are tracked
in the team's Funding & Data Tracker doc (Glaucoma Model tab) — this repo is
where that plan actually gets implemented.

## Dataset

[SMDG-19](https://www.kaggle.com/datasets/deathtrooper/multichannel-glaucoma-benchmark-dataset)
(Standardized Multi-Channel Dataset for Glaucoma) standardizes 19 public
glaucoma datasets (REFUGE, ORIGA, AIROGS, DRISHTI-GS1, and others) into one
format. 12,449 labeled fundus images total.

**Labels are 0 (non-glaucoma), 1 (glaucoma), and -1 (glaucoma-suspect, 133
rows)** — not clean binary. The paper's pipeline is strictly 2-class, with
no mention anywhere of a suspect category. Use
`glaucoma_model.load_classification_split()`
(`src/glaucoma_model/data.py`), not `biotransport_datasets.load_split()`
directly — it drops the -1 rows for you. This was confirmed empirically, not
assumed: the paper states a starting count of 12,316 samples before
leakage-exclusion, which is exactly 12,449 minus the 133 suspects. Applying
leakage-exclusion to that reduced set lands at **exactly 8,818** — the
paper's reported figure precisely, not approximately. (An earlier version of
this README attributed the gap to generic dataset-version drift at ~8,883
rows; that was the *unfiltered* count and the explanation was wrong — this
is the real cause, pinned by `tests/test_data.py`.) The 133 suspects aren't
only a classification-stage concern either: 68 of them sit inside the
3,566-row pool Phase 4 segmentation training will draw from.

Raw dataset files are never committed here — see `data/README.md` to fetch
it, and `scripts/validate_features_on_real_masks.py` to sanity-check
`extract_features()` against the real ground-truth masks once it's fetched
(validates cup-to-disc ratio by label against the paper's own Table 7).

## Architecture

Three stages:

1. **Segmentation → engineered biomarkers.** U-Net for optic disc and optic
   cup, U-Net for retinal vasculature, a custom SegFormer (FPN + Squeeze-
   Excitation) for the macula — there is no separate retina segmentation
   model. The retinal boundary instead comes from Canny edge detection
   directly on the raw fundus photograph; optic disc/cup/macula boundaries
   come from contour detection directly on their (already binary)
   segmentation masks, no edge detection needed. All of it is deliberately
   deterministic geometry, not learned embeddings, to avoid leakage between
   the segmentation and classification stages. 23 total features (5
   extracted, 18 engineered ratios/radii/areas) — see
   `src/glaucoma_model/features.py` for the exact spec.
2. **Image branch.** Swin-b and Swin-V2-b in parallel, features concatenated.
3. **Fusion.** Bidirectional cross-modal attention (image attends to numeric
   biomarkers and vice versa) — this beat both feature concatenation and
   gated fusion in the paper's own ablation study.

**Benchmark to match or beat:** 92.63% test accuracy, 0.975 ROC-AUC, 0.904
precision, 0.930 recall. Numeric biomarkers alone (1D-CNN, no images) already
reach ~91% paired with either Swin backbone (~75% fully alone) — the tabular
signal is doing most of the work.

## Development plan

1. ~~Transcribe the full feature spec and training config from the paper~~ —
   done; see `src/glaucoma_model/features.py` and the Funding & Data Tracker.
2. ~~Pull SMDG-19 and reproduce the leakage-safe split~~ — done, and exactly
   matches the paper's 8,818 via `load_classification_split()` (see the
   Dataset section above). `extract_features()` is also validated against
   SMDG-19's real ground-truth masks now, not just synthetic circles — cup-
   to-disc ratio by label matches the paper's Table 7 within 2%.
3. **Reproduce the tabular-only 1D-CNN baseline first**
   (`src/glaucoma_model/models/tabular_cnn.py`) — the cheapest way to
   validate the full pipeline end to end before spending compute on two
   vision transformers.
4. Reproduce the segmentation stage (U-Net for optic disc/cup/vasculature,
   SegFormer for macula) — start with optic cup specifically, since cup
   radius/area/circumference alone account for over a third of the paper's
   total feature importance. Real training-pool sizes available now differ
   slightly from the paper's (disc 3,103 / cup 2,874 here vs. 2,805 reported
   for both in the paper) — same version-drift pattern as above, not a new
   problem to debug when you see it.
5. Add the Swin-b + Swin-V2-b image branches and bidirectional cross-attention
   fusion once the tabular baseline is validated.
6. Extend to OHTS/ORIGA once that outreach lands.
7. Wrap the trained model as an agent-accessible tool.

## Setup

```bash
pip install -r requirements.txt
pip install -e .
```

The editable install (`pip install -e .`) is what makes `glaucoma_model`
importable from `src/` — skip it and `pytest` / scripts will fail with
`ModuleNotFoundError`.

For an exact reproduction of the environment today's validated results came
from (recommended for anything you want numerically comparable later), use
`pip install -r requirements-lock.txt` instead of the loose `requirements.txt`.

See `data/README.md` for fetching SMDG-19. Run tests with `pytest`.

## The cross-repo dependency

Fetching and indexing SMDG-19 still depends on the sibling `ai-research`
repo being cloned alongside this one (`datasets/scripts/fetch_kaggle.sh` for
the fetch, `biotransport_datasets` for indexing/loading) — that part hasn't
changed, and still won't work from a fresh clone without `ai-research` as a
sibling directory. What changed: it's no longer an undeclared path hack.
`biotransport_datasets` is now a real, versioned, pip-installable package
(`ai-research/datasets/pyproject.toml`) — `requirements.txt` installs it in
editable mode from `../ai-research/datasets`, resolves correctly from any
working directory (verified by importing it from outside both repos
entirely), and shows up properly in `pip show biotransport-datasets`.
