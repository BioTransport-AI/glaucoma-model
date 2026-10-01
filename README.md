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
images used to train its own segmentation models, leaving a leakage-safe 8,818
samples (80/10/10 split).

Raw dataset files are never committed here — see `data/README.md` to fetch it
yourself (same read-only-source convention as the `biotransport-ai` repo's
`datasets/`).

## Architecture (confirmed from the full paper)

Three stages:

1. **Segmentation → engineered biomarkers.** U-Net for optic disc and optic
   cup, U-Net for retinal vasculature, a custom SegFormer (FPN + Squeeze-
   Excitation) for the macula. Diameters/areas are extracted via Canny edge
   detection → contour detection → minimum enclosing circle — deliberately
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

**Reality check:** this is a known, reproducible recipe now, not an open
research question — that's real progress — but reproducing it (4 segmentation
models plus a dual-transformer classifier) is still a multi-week effort, not
something to rush. Building incrementally, cheapest validation first.

## Development plan

1. ~~Transcribe the full feature spec and training config from the paper~~ —
   done; see `src/glaucoma_model/features.py` and the Funding & Data Tracker.
2. Pull SMDG-19 and reproduce the leakage-safe 8,818-sample split.
3. **Reproduce the tabular-only 1D-CNN baseline first**
   (`src/glaucoma_model/models/tabular_cnn.py`) — the cheapest way to
   validate the full pipeline end to end before spending compute on two
   vision transformers.
4. Reproduce the segmentation stage (U-Net for optic disc/cup/vasculature,
   SegFormer for macula) — start with optic cup specifically, since cup
   radius/area/circumference alone account for over a third of the paper's
   total feature importance.
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

See `data/README.md` for fetching SMDG-19. Run tests with `pytest`.
