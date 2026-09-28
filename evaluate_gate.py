"""Check whether modality agreement can safely reduce false segmentations."""

import gc
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy.ndimage import label

from datasets.mri_dataset import MRIDataset
from models.unet3d import create_unet3d

ROOT = Path(__file__).resolve().parent


def main():
    dataset = MRIDataset(ROOT / "split_dataset" / "val_local.csv", train=False)
    model = create_unet3d().cuda().eval()
    model.load_state_dict(torch.load(ROOT / "weights" / "best_unet3d.pth", map_location="cuda", weights_only=True))
    thresholds = (0.4, 0.5, 0.6, 0.7, 0.8)
    methods = ("baseline", "dwi>0", "dwi>0.5")
    results = {(threshold, method): [] for threshold in thresholds for method in methods}
    for index in range(len(dataset)):
        image, truth = dataset[index]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = sliding_window_inference(image[None].cuda(), (96, 96, 96), 1, model)
            probability = torch.sigmoid(prediction)[0, 0].float().cpu().numpy()
        target = truth[0].numpy() > 0
        dwi, adc = image[0].numpy(), image[1].numpy()
        gates = {
            "baseline": np.ones_like(probability, dtype=bool),
            "dwi>0": dwi > 0,
            "dwi>0.5": dwi > 0.5,
        }
        for threshold in thresholds:
            binary = probability >= threshold
            for method, gate in gates.items():
                candidate = binary & gate
                components, count = label(candidate)
                sizes = np.bincount(components.ravel(), minlength=count + 1)
                candidate &= sizes[components] >= 10
                tp = np.count_nonzero(candidate & target)
                fp = np.count_nonzero(candidate & ~target)
                fn = np.count_nonzero(~candidate & target)
                results[(threshold, method)].append((2 * tp / max(2 * tp + fp + fn, 1), tp, fp, fn, tp / max(tp + fp + fn, 1)))
        if (index + 1) % 10 == 0:
            print(f"Validated {index + 1}/{len(dataset)}", flush=True)
        del image, truth, prediction, probability
        gc.collect()
    for (threshold, method), values in results.items():
        array = np.asarray(values)
        tp, fp, fn = array[:, 1:4].sum(axis=0)
        print(f"threshold={threshold:.1f} {method}: Dice={array[:,0].mean():.3f} IoU={array[:,4].mean():.3f} precision={tp/max(tp+fp,1):.3f} recall={tp/max(tp+fn,1):.3f} false_voxels={int(fp)}")


if __name__ == "__main__":
    main()
