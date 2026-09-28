"""Train the project's 3D U-Net on the locally extracted ISLES 2022 data."""

import argparse
import gc
from pathlib import Path
import time

import pandas as pd
import torch
from torch.utils.data import DataLoader
from monai.inferers import sliding_window_inference

from datasets.mri_dataset import MRIDataset
from models.unet3d import create_unet3d
from training.loss import SegmentationLoss, dice_score, iou_score


ROOT = Path(__file__).resolve().parent
MODALITIES = ("dwi_path", "adc_path", "flair_path", "mask_path")
PREPARED_FLAIR = Path(r"D:\ISLES-2022-prepared")


def local_csv(split, data_root):
    source = ROOT / "split_dataset" / f"{split}.csv"
    target = ROOT / "split_dataset" / f"{split}_local.csv"
    frame = pd.read_csv(source)
    for column in MODALITIES:
        frame[column] = frame[column].map(
            lambda value: str(data_root.joinpath(*str(value).replace("\\", "/").split("/")[1:]))
        )
    frame["flair_path"] = frame.apply(
        lambda row: str(PREPARED_FLAIR / f"{row['subject_id']}_{row['session_id']}_FLAIR.nii.gz"),
        axis=1,
    )
    missing = [path for column in MODALITIES for path in frame[column] if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} dataset files; first: {missing[0]}")
    frame.to_csv(target, index=False)
    return target


def run(epochs, data_root, resume=False, initial_best_dice=-1.0, start_epoch=1):
    train_csv = local_csv("train", data_root)
    val_csv = local_csv("val", data_root)
    train_set = MRIDataset(train_csv, train=True)
    val_set = MRIDataset(val_csv, train=False)
    train_loader = DataLoader(train_set, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU training will not fall back to CPU")
    device = torch.device("cuda")
    model = create_unet3d().to(device)
    output = ROOT / "weights" / "best_unet3d.pth"
    if resume:
        model.load_state_dict(torch.load(output, map_location=device, weights_only=True))
        print(f"Resumed weights: {output}", flush=True)
    criterion = SegmentationLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    output.parent.mkdir(exist_ok=True)
    best_dice = initial_best_dice
    print(f"Device: {device}; train: {len(train_set)} cases; validation: {len(val_set)} cases", flush=True)
    print(f"Weights: {output}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        started = time.monotonic()
        print(f"\n=== Epoch {epoch}/{epochs} ===", flush=True)
        model.train()
        train_loss = 0.0
        for step, (image, mask) in enumerate(train_loader, 1):
            image, mask = image.to(device), mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(image)
            loss = criterion(prediction, mask)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if step == 1 or step % 5 == 0 or step == len(train_loader):
                print(f"Train: {step}/{len(train_loader)} ({100*step/len(train_loader):.0f}%) | loss {train_loss/step:.4f}", flush=True)
            del image, mask, prediction, loss
            if step % 20 == 0:
                gc.collect()

        model.eval()
        val_loss = val_dice = val_iou = 0.0
        with torch.no_grad():
            for step, (image, mask) in enumerate(val_loader, 1):
                image, mask = image.to(device), mask.to(device)
                prediction = sliding_window_inference(image, (96, 96, 96), 1, model)
                val_loss += criterion(prediction, mask).item()
                val_dice += dice_score(prediction, mask).item()
                val_iou += iou_score(prediction, mask).item()
                if step == 1 or step % 5 == 0 or step == len(val_loader):
                    print(f"Validation: {step}/{len(val_loader)} ({100*step/len(val_loader):.0f}%)", flush=True)
                del image, mask, prediction
                gc.collect()

        val_loss /= len(val_loader)
        val_dice /= len(val_loader)
        val_iou /= len(val_loader)
        print(f"Epoch {epoch}/{epochs}: train loss {train_loss/len(train_loader):.4f} | val loss {val_loss:.4f} | Dice {val_dice:.2%} | IoU {val_iou:.2%} | {time.monotonic()-started:.0f} sec", flush=True)
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(model.state_dict(), output)
            print(f"Saved best weights: {output}", flush=True)
    print(f"Training complete. Best Dice: {best_dice:.2%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\ISLES-2022"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--initial-best-dice", type=float, default=-1.0)
    parser.add_argument("--start-epoch", type=int, default=1)
    args = parser.parse_args()
    run(args.epochs, args.data_root, args.resume, args.initial_best_dice, args.start_epoch)
