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
format. 12,449 labeled fundus images; the paper's classification set excludes
images used to train its own segmentation models, leaving a leakage-safe
split the paper reports as 8,818 samples. **Fetched and indexed for real as
of 2026-10-03: this live copy of the dataset gives 8,883** (a ~0.7% drift,
consistent with the same small version drift visible in the label counts
below — Kaggle's copy has moved since the paper was published, not a bug in
our indexing).

**Label values are 0 (non-glaucoma, 7,549), 1 (glaucoma, 4,767), and -1
(glaucoma-suspect, 133)** — not a clean binary 0/1. The paper's own pipeline
is a 2-class problem with no mention of a suspect class, so **the -1 rows
need to be explicitly excluded before training** (`TabularCNN`'s output head
is 2-way; feeding a raw -1 into `CrossEntropyLoss` as-is will error or
corrupt a batch, not fail loudly in an obvious way). This affects the
segmentation-training pool too: 68 of the 3,566 `used_for_segmentation=True`
rows are suspect-labeled.

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
2. ~~Pull SMDG-19 and reproduce the leakage-safe split~~ — done (8,883
   samples; see the Dataset section above). `extract_features()` is also
   validated against SMDG-19's real ground-truth masks now, not just
   synthetic circles — cup-to-disc ratio by label matches the paper's
   Table 7 within 2%. **Before any training: filter out the 133 (68 in the
   segmentation pool) suspect-labeled rows** -- see the Dataset section.
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

## Known architectural debt

Fetching and indexing SMDG-19 currently depends on the sibling
`ai-research` repo (`datasets/scripts/fetch_kaggle.sh` and the
`biotransport_datasets` package) being cloned alongside this one — nothing
in this repo enforces or declares that dependency. Fine for a two-person
team working from the same machine layout; worth revisiting (vendor the
needed pieces in, or package `biotransport_datasets` properly) before
anyone else tries to run this from a fresh clone.
