"""Fine-tune a separate DWI+ADC+FLAIR model with best-checkpoint early stopping."""

import argparse
import gc
import os
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from datasets.soop_dataset import PATCH_SIZE
from datasets.soop_flair_dataset import build_flair_manifest, SOOPFlairDataset
from models.unet3d import create_unet3d
from training.loss import SegmentationLoss, dice_score, iou_score


ROOT = Path(__file__).resolve().parent
BEST = ROOT / "weights" / "soop_flair_best.pth"
LAST = ROOT / "weights" / "soop_flair_last.pth"
SOURCE = ROOT / "weights" / "best_unet3d.pth"


def save_atomic(data, destination):
    temporary = destination.with_suffix(".tmp")
    torch.save(data, temporary)
    os.replace(temporary, destination)


def train(total_epochs, patience):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.backends.cudnn.benchmark = True
    manifest = build_flair_manifest()
    if len(manifest) < 1400:
        raise RuntimeError(f"Registered FLAIR cohort incomplete: {len(manifest)}/1449")
    manifest.to_csv(ROOT / "split_dataset" / "soop_flair_manifest.csv", index=False)
    train_set = SOOPFlairDataset(manifest, "train")
    val_set = SOOPFlairDataset(manifest, "val")
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    model = create_unet3d(in_channels=3).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    criterion = SegmentationLoss()
    BEST.parent.mkdir(exist_ok=True)
    start_epoch, best_dice, stale = 1, -1.0, 0
    if LAST.exists():
        checkpoint = torch.load(LAST, map_location="cuda", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_dice = float(checkpoint["best_dice"])
        stale = int(checkpoint.get("stale", 0))
        print(f"Resuming at epoch {start_epoch}; best Dice {best_dice:.2%}", flush=True)
    else:
        model.load_state_dict(torch.load(SOURCE, map_location="cpu", weights_only=True))
        print("Initialized from ISLES 3-channel weights", flush=True)
    print(f"Device {torch.cuda.get_device_name(0)}; train {len(train_set)}; val {len(val_set)}; "
          f"max epochs {total_epochs}; early-stop patience {patience}", flush=True)
    for epoch in range(start_epoch, total_epochs + 1):
        started = time.monotonic()
        model.train()
        train_loss = 0.0
        print(f"\n=== Epoch {epoch}/{total_epochs} ===", flush=True)
        for step, (image, mask) in enumerate(train_loader, 1):
            image, mask = image.cuda(), mask.cuda()
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            loss = criterion(prediction, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if step == 1 or step % 50 == 0 or step == len(train_loader):
                print(f"Train {step}/{len(train_loader)} ({100*step/len(train_loader):.0f}%) "
                      f"loss {train_loss/step:.4f}", flush=True)
            del image, mask, prediction, loss
            if step % 50 == 0:
                gc.collect()
        model.eval()
        val_loss = val_dice = val_iou = 0.0
        with torch.inference_mode():
            for step, (image, mask) in enumerate(val_loader, 1):
                image, mask = image.cuda(), mask.cuda()
                prediction = sliding_window_inference(image, PATCH_SIZE, 1, model)
                val_loss += criterion(prediction, mask).item()
                val_dice += dice_score(prediction, mask).item()
                val_iou += iou_score(prediction, mask).item()
                if step == 1 or step % 25 == 0 or step == len(val_loader):
                    print(f"Validation {step}/{len(val_loader)} ({100*step/len(val_loader):.0f}%)", flush=True)
                del image, mask, prediction
                if step % 10 == 0:
                    gc.collect()
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        val_loss /= len(val_loader)
        print(f"Epoch {epoch}/{total_epochs}: train loss {train_loss/len(train_loader):.4f} | "
              f"val loss {val_loss:.4f} | Dice {val_dice:.2%} | IoU {val_iou:.2%} | "
              f"{time.monotonic()-started:.0f} sec", flush=True)
        if val_dice > best_dice:
            best_dice, stale = val_dice, 0
            save_atomic(model.state_dict(), BEST)
            print(f"Saved best weights: {BEST}", flush=True)
        else:
            stale += 1
        save_atomic({"epoch": epoch, "best_dice": best_dice, "stale": stale,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict()}, LAST)
        print(f"Saved resume checkpoint: {LAST}; epochs without improvement: {stale}", flush=True)
        if stale >= patience:
            print(f"Early stopping after {epoch} epochs. Best Dice {best_dice:.2%}", flush=True)
            break
        gc.collect()
    print(f"SOOP FLAIR training complete. Best Dice: {best_dice:.2%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()
    train(args.epochs, args.patience)
