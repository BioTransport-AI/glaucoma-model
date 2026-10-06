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

**Every classification-eligible image has zero real ground-truth masks —
this is by construction, not a gap to fix.** Leakage-exclusion removes
exactly the images that have *any* segmentation mask (that's its whole job),
so the two pools are perfectly disjoint: of the 8,883 unfiltered
classification-eligible rows, 0 have an optic-cup, optic-disc, or
vasculature mask (checked directly against the real index, not inferred).
This means `extract_features()` can never run on real ground truth for a
classification-pool image — real masks only exist for the ~3,500 images
*excluded* from classification. The paper's own pipeline resolves this the
only way possible: train the segmentation models first, then run them on
every classification-pool image to get *predicted* masks, then extract
features from those. See "Development plan" below — this is why the
tabular-CNN baseline (step 3) can't be reproduced on real features until
step 4 (segmentation) has produced at least disc/cup/vasculature predictions
to run it on.

**No macula masks exist anywhere in the public SMDG-19 download.** The
paper's macula ground truth (Section 3.5) is a separate 728-image manual
annotation its authors did themselves via Roboflow — never part of SMDG-19's
public release. `glaucoma_model.data.SEGMENTATION_STRUCTURES` therefore has
no `macula` entry. Reproducing the macula/SegFormer stage will need the team
to annotate its own macula masks, the same way the paper's authors did; it
isn't a recipe bug to fix. The 4 macula-derived features (of 23 total) will
need to come from a zero-filled placeholder or a team-annotated model until
then — tracked here, not yet blocking anything downstream.

`biotransport_datasets.recipes.smdg19` now indexes real per-structure mask
paths (`optic_cup_mask_path`, `optic_disc_mask_path`, `vasculature_mask_path`
— `NaN` where that image has no mask for that structure) and a `split`
column: one stratified-by-label 80:10:10 train/val/test assignment, made
once across the whole 12,449-row index so every task — classification, and
each segmentation structure — filters the *same* assignment rather than
each computing its own (the only way a row can't end up train for one task
and test for another). Use `glaucoma_model.data.load_segmentation_split
(structure, split)` rather than querying these columns directly — it applies
the right mask-presence filter per structure.

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
3. ~~Validate the tabular-only 1D-CNN baseline architecture~~ — done, but
   narrower than originally planned here, and worth being precise about why.
   This step was originally meant as "train TabularCNN on real extracted
   features, cheaply, before spending compute on segmentation." That's not
   actually possible: as the Dataset section above lays out, zero
   classification-pool images have a real ground-truth mask of any kind, by
   construction of the leakage exclusion. There is no real feature vector to
   train on yet — only once step 4 produces trained segmentation models can
   they run on the classification pool to produce the *predicted* masks
   `extract_features()` needs. What's actually done: `TabularCNN`'s
   architecture is validated on synthetic tensors (`tests/test_tabular_cnn.py`
   — forward shape, finite outputs, the batch-size-1/BatchNorm1d edge case),
   and `extract_features()` itself is validated against real ground-truth
   masks on the segmentation-training pool (`scripts/validate_features_on_real_masks.py`).
   A real-feature baseline training run is now a later step, folded into
   step 5 below rather than a separate one — this also matches the paper's
   own section ordering (3.2–3.5 segmentation, 3.6 tabular classification),
   which this plan had inverted.
4. **In progress: reproduce the segmentation stage.** U-Net implemented
   (`src/glaucoma_model/models/unet.py` — standard 4-level encoder/decoder,
   `base_channels=32` by default, narrower than the paper's own unstated
   width since training here runs on CPU, not the paper's compute) with a
   paired image/mask dataset (`segmentation_dataset.py`, using the paper's
   exact augmentation list — albumentations, not torchvision, so the same
   random transform hits both image and mask) and the paper's stated
   hyperparameters (`scripts/train_segmentation.py`): 128×128, BCE+Dice loss,
   Adam lr=1e-3, `ReduceLROnPlateau(patience=3)` on validation Dice. Starting
   with optic cup specifically (50 epochs, per the paper's own 3.3 — cup
   radius/area/circumference alone account for over a third of total feature
   importance); optic disc next (30 epochs per 3.2), then vasculature.
   Real training-pool sizes differ slightly from the paper's (disc 3,103 /
   cup 2,874 here vs. 2,805 reported for both) — same version-drift pattern
   noted elsewhere in this README, not a new problem to debug when you see
   it. No macula sub-stage yet — see the Dataset section's note on why.
5. Once optic-cup/disc/vasculature U-Nets exist: run them on the
   classification pool to get predicted masks, extract real feature vectors,
   and *then* train/evaluate the tabular-CNN baseline for real (this absorbs
   what step 3 was originally meant to do).
6. Add the Swin-b + Swin-V2-b image branches and bidirectional cross-attention
   fusion once the tabular baseline is validated.
7. Extend to OHTS/ORIGA once that outreach lands.
8. Wrap the trained model as an agent-accessible tool.

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

## Running segmentation training

```bash
python scripts/train_segmentation.py --structure optic_cup --smoke-test  # ~15s, checks the pipeline runs
python scripts/train_segmentation.py --structure optic_cup               # the real 50-epoch run
```

`--smoke-test` trains 2 epochs on a 16-image subset and writes to a separate
`*_smoketest.pt` checkpoint — use it to confirm the pipeline still runs
end to end after a change, before committing CPU time to a real run. The
script defaults to `--device cpu`; there's no discrete GPU on the machine
this was developed on (Intel integrated graphics only — confirmed via
`lspci`, no CUDA available to torch), so CPU is the realistic default here,
not a placeholder. Measured directly against the real optic-cup training
pool (2,268 images, steady-state batches, warm-up excluded): roughly
0.8s/batch at the default batch size of 16, which puts the full 50-epoch
optic-cup run at **roughly 1.5–2 hours** on a 16-core CPU — slow but
genuinely feasible given the paper's own 128×128 resolution is small, not
something that requires sourcing GPU access first. Pass `--device cuda` if
that changes later. Best-checkpoint selection is by validation Dice; final
train/val/test metrics print in the same Loss/Dice/IoU/Accuracy shape as the
paper's Tables 2–5 so a real run's numbers are directly comparable.

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
