"""Compare inference thresholds and 3D component filters on held-out cases."""

import gc
from pathlib import Path

import numpy as np
import torch
from monai.inferers import sliding_window_inference
from scipy.ndimage import label

from datasets.mri_dataset import MRIDataset
from models.unet3d import create_unet3d


ROOT = Path(__file__).resolve().parent
THRESHOLDS = (0.5, 0.6, 0.7, 0.8, 0.9)
MIN_SIZES = (0, 10, 30, 100)


def main():
    dataset = MRIDataset(ROOT / "split_dataset" / "val_local.csv", train=False)
    model = create_unet3d().cuda().eval()
    model.load_state_dict(torch.load(ROOT / "weights" / "best_unet3d.pth", map_location="cuda", weights_only=True))
    scores = {(threshold, minimum): [] for threshold in THRESHOLDS for minimum in MIN_SIZES}
    for index in range(len(dataset)):
        image, truth = dataset[index]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            prediction = sliding_window_inference(image[None].cuda(), (96, 96, 96), 1, model)
            probability = torch.sigmoid(prediction)[0, 0].float().cpu().numpy()
        target = truth[0].numpy() > 0
        for threshold in THRESHOLDS:
            binary = probability >= threshold
            components, count = label(binary)
            sizes = np.bincount(components.ravel(), minlength=count + 1)
            for minimum in MIN_SIZES:
                candidate = binary if minimum == 0 else binary & (sizes[components] >= minimum)
                tp = np.count_nonzero(candidate & target)
                fp = np.count_nonzero(candidate & ~target)
                fn = np.count_nonzero(~candidate & target)
                dice = 2 * tp / max(2 * tp + fp + fn, 1)
                iou = tp / max(tp + fp + fn, 1)
                scores[(threshold, minimum)].append((dice, tp, fp, fn, iou))
        if (index + 1) % 5 == 0 or index + 1 == len(dataset):
            print(f"Validated {index + 1}/{len(dataset)}", flush=True)
        del image, truth, prediction, probability
        gc.collect()
    for (threshold, minimum), values in scores.items():
        array = np.asarray(values)
        tp, fp, fn = array[:, 1:4].sum(axis=0)
        print(f"threshold={threshold:.1f} min_voxels={minimum:3d} mean_dice={array[:, 0].mean():.3f} mean_iou={array[:, 4].mean():.3f} precision={tp/max(tp+fp,1):.3f} recall={tp/max(tp+fn,1):.3f} false_voxels={int(fp)}", flush=True)


if __name__ == "__main__":
    main()
