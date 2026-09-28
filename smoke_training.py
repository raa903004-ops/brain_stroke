"""Smoke check the MONAI data pipeline and one 3D U-Net training step."""

from pathlib import Path
import tempfile
import time

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from datasets.mri_dataset import MRIDataset
from models.unet3d import create_unet3d
from training.loss import SegmentationLoss


with tempfile.TemporaryDirectory() as folder:
    root = Path(folder)
    paths = {}
    for name in ("dwi", "adc", "flair", "label"):
        data = np.zeros((32, 32, 32), dtype=np.float32)
        data[12:20, 12:20, 12:20] = 1
        path = root / f"{name}.nii.gz"
        nib.save(nib.Nifti1Image(data, np.eye(4)), path)
        paths[f"{name if name != 'label' else 'mask'}_path"] = str(path)
    csv = root / "case.csv"
    pd.DataFrame([paths]).to_csv(csv, index=False)
    started = time.monotonic()
    image, mask = MRIDataset(csv, train=True)[0]
    assert image.shape == (3, 96, 96, 96), image.shape
    assert mask.shape == (1, 96, 96, 96), mask.shape
    model = create_unet3d()
    output = model(image.unsqueeze(0))
    loss = SegmentationLoss()(output, mask.unsqueeze(0))
    loss.backward()
    print(f"smoke training step passed; loss={loss.item():.4f}; seconds={time.monotonic()-started:.1f}")
