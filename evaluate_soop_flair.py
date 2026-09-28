"""Evaluate the FLAIR model on held-out SOOP and independent ISLES cases."""

import json
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference

from datasets.mri_dataset import MRIDataset
from datasets.soop_dataset import PATCH_SIZE
from datasets.soop_flair_dataset import build_flair_manifest, SOOPFlairDataset
from models.unet3d import create_unet3d


ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "soop_flair_evaluation.json"


def evaluate(dataset, model):
    rows = []
    with torch.inference_mode():
        for index in range(len(dataset)):
            image, truth = dataset[index]
            logits = sliding_window_inference(image[None].cuda(), PATCH_SIZE, 1, model)
            probability = torch.sigmoid(logits)[0, 0].cpu().numpy()
            target = truth[0].numpy() > 0
            prediction = probability >= 0.5
            tp = int(np.count_nonzero(prediction & target))
            fp = int(np.count_nonzero(prediction & ~target))
            fn = int(np.count_nonzero(~prediction & target))
            dice = 2 * tp / max(2 * tp + fp + fn, 1)
            rows.append({"dice": dice, "tp": tp, "fp": fp, "fn": fn})
            if (index + 1) % 25 == 0 or index + 1 == len(dataset):
                print(f"Evaluated {index + 1}/{len(dataset)}", flush=True)
    return {
        "cases": len(rows),
        "mean_dice": float(np.mean([row["dice"] for row in rows])),
        "total_false_positive_voxels": int(sum(row["fp"] for row in rows)),
        "total_true_positive_voxels": int(sum(row["tp"] for row in rows)),
    }


def main():
    model = create_unet3d(in_channels=3).cuda().eval()
    model.load_state_dict(torch.load(ROOT / "weights" / "soop_flair_best.pth",
                                     map_location="cpu", weights_only=True))
    soop = SOOPFlairDataset(build_flair_manifest(), "val")
    isles = MRIDataset(ROOT / "split_dataset" / "val_local.csv", train=False)
    report = {"checkpoint": str(ROOT / "weights" / "soop_flair_best.pth"),
              "threshold": 0.5, "postprocessing": "none"}
    print("SOOP held-out evaluation", flush=True)
    report["soop"] = evaluate(soop, model)
    print("ISLES external evaluation", flush=True)
    report["isles"] = evaluate(isles, model)
    OUTPUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
