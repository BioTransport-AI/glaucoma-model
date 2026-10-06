"""Train a U-Net for one segmentation structure (optic disc, optic cup, or
vasculature -- Parikh, Nguyen & Penkova 2026, Sections 3.2-3.4). No macula: see
glaucoma_model.data.SEGMENTATION_STRUCTURES for why.

Hyperparameters default to what the paper states for the optic cup specifically
(50 epochs; optic disc only needed 30 -- override with --epochs if training that
structure instead): 128x128 input, Adam lr=1e-3, ReduceLROnPlateau(patience=3)
monitoring validation Dice, combined BCE-Dice loss. Batch size isn't stated in
the paper; 16 is this script's own default, not a reproduction of a documented
value.

    python scripts/train_segmentation.py --structure optic_cup --smoke-test
    python scripts/train_segmentation.py --structure optic_cup

Run from the glaucoma-model repo root with its venv active (needs the real
SMDG-19 data fetched and indexed -- see data/README.md).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from glaucoma_model.data import SEGMENTATION_STRUCTURES, load_segmentation_split
from glaucoma_model.models import UNet
from glaucoma_model.segmentation_dataset import SegmentationDataset
from glaucoma_model.segmentation_metrics import bce_dice_loss, dice_coefficient, iou_score, pixel_accuracy

CHECKPOINT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"


def build_dataloaders(
    structure: str, batch_size: int, image_size: int, num_workers: int, limit: "int | None" = None
) -> tuple[DataLoader, DataLoader, DataLoader]:
    mask_col = SEGMENTATION_STRUCTURES[structure]
    splits = {}
    for name in ("train", "val", "test"):
        df = load_segmentation_split(structure, split=name)
        if limit is not None:
            df = df.head(limit if name == "train" else max(1, limit // 2))
        is_train = name == "train"
        ds = SegmentationDataset(df, mask_col, image_size=image_size, train=is_train)
        splits[name] = DataLoader(ds, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)
    return splits["train"], splits["val"], splits["test"]


def run_epoch(
    model: UNet, loader: DataLoader, device: torch.device, optimizer: "torch.optim.Optimizer | None" = None
) -> dict[str, float]:
    """One pass over `loader`. Training (gradients, backward, optimizer step)
    when `optimizer` is given; eval (no_grad) otherwise."""
    model.train(optimizer is not None)
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "accuracy": 0.0}
    n_batches = 0

    with torch.set_grad_enabled(optimizer is not None):
        for images, masks in loader:
            images, masks = images.to(device), masks.to(device)
            logits = model(images)
            loss = bce_dice_loss(logits, masks)

            if optimizer is not None:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            totals["loss"] += loss.item()
            totals["dice"] += dice_coefficient(logits, masks)
            totals["iou"] += iou_score(logits, masks)
            totals["accuracy"] += pixel_accuracy(logits, masks)
            n_batches += 1

    if n_batches == 0:
        raise ValueError("loader produced zero batches -- empty split or batch_size larger than the data")
    return {k: v / n_batches for k, v in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--structure", default="optic_cup", choices=sorted(SEGMENTATION_STRUCTURES))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--base-channels", type=int, default=32, help="U-Net width; see models/unet.py")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT_DIR))
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="2 epochs on a small subset -- verifies the pipeline runs and reports a real "
        "per-epoch time estimate, without overwriting a real run's checkpoint.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    limit = 16 if args.smoke_test else None
    epochs = 2 if args.smoke_test else args.epochs

    train_dl, val_dl, test_dl = build_dataloaders(
        args.structure, args.batch_size, args.image_size, args.num_workers, limit=limit
    )
    print(
        f"structure={args.structure} device={device} "
        f"train={len(train_dl.dataset)} val={len(val_dl.dataset)} test={len(test_dl.dataset)} images"
    )
    if args.smoke_test:
        print("--smoke-test: 2 epochs on a 16-image subset, to check correctness and timing only")

    model = UNet(in_channels=3, base_channels=args.base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoketest" if args.smoke_test else ""
    best_path = checkpoint_dir / f"{args.structure}_unet_best{suffix}.pt"
    best_val_dice = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = run_epoch(model, train_dl, device, optimizer=optimizer)
        val_metrics = run_epoch(model, val_dl, device, optimizer=None)
        scheduler.step(val_metrics["dice"])
        elapsed = time.time() - t0

        print(
            f"epoch {epoch}/{epochs} ({elapsed:.1f}s) "
            f"train loss={train_metrics['loss']:.4f} dice={train_metrics['dice']:.4f} | "
            f"val loss={val_metrics['loss']:.4f} dice={val_metrics['dice']:.4f} "
            f"iou={val_metrics['iou']:.4f} acc={val_metrics['accuracy']:.4f}"
        )

        if val_metrics["dice"] > best_val_dice:
            best_val_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "epoch": epoch,
                    "val_dice": best_val_dice,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"  new best val dice {best_val_dice:.4f} -> saved to {best_path}")

    print(f"\ntraining done. loading best checkpoint (val dice {best_val_dice:.4f}) for final evaluation")
    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])

    print(f"\n=== Final evaluation: {args.structure} U-Net (cf. paper's Table 2/3/4) ===")
    print(f"{'Set':<12}{'Loss':<10}{'Dice':<10}{'IoU':<10}{'Accuracy':<10}")
    for name, loader in (("Train", train_dl), ("Validation", val_dl), ("Test", test_dl)):
        metrics = run_epoch(model, loader, device, optimizer=None)
        print(
            f"{name:<12}{metrics['loss']:<10.5f}{metrics['dice']:<10.5f}"
            f"{metrics['iou']:<10.5f}{metrics['accuracy']:<10.5f}"
        )


if __name__ == "__main__":
    main()
