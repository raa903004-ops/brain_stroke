"""DWI/ADC acute lesion volumes from the independent SOOP cohort."""

import hashlib
import json
from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ResampleToMatchd, NormalizeIntensityd, ConcatItemsd, SpatialPadd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, EnsureTyped,
)


ROOT = Path(r"D:\SOOP")
PATCH_SIZE = (96, 96, 32)


def build_manifest():
    subjects = [json.loads(line) for line in (ROOT / "subjects.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = []
    for subject in subjects:
        if not subject.get("evaluable_lesionAcute") or not subject.get("adc_usable"):
            continue
        paths = {key: ROOT / subject[key] for key in ("image_trace", "image_adc", "mask_lesionAcute")}
        if not all(path.is_file() for path in paths.values()):
            continue
        subject_id = subject["subject_id"]
        bucket = int.from_bytes(hashlib.sha256(subject_id.encode()).digest()[:4], "big") % 10
        rows.append({"subject_id": subject_id, "split": "val" if bucket == 0 else "train",
                     "dwi_path": str(paths["image_trace"]), "adc_path": str(paths["image_adc"]),
                     "mask_path": str(paths["mask_lesionAcute"])})
    return pd.DataFrame(rows)


class SOOPDataset(Dataset):
    def __init__(self, manifest, split):
        self.data = manifest.loc[manifest.split == split].reset_index(drop=True)
        transforms = [
            LoadImaged(keys=("dwi", "adc", "label")),
            EnsureChannelFirstd(keys=("dwi", "adc", "label")),
            Orientationd(keys=("dwi", "adc", "label"), axcodes="RAS"),
            ResampleToMatchd(keys=("adc", "label"), key_dst="dwi", mode=("bilinear", "nearest")),
            NormalizeIntensityd(keys=("dwi", "adc"), nonzero=True, channel_wise=True),
            ConcatItemsd(keys=("dwi", "adc"), name="image", dim=0),
            SpatialPadd(keys=("image", "label"), spatial_size=PATCH_SIZE),
        ]
        if split == "train":
            transforms.extend([
                RandCropByPosNegLabeld(keys=("image", "label"), label_key="label",
                                       spatial_size=PATCH_SIZE, pos=1, neg=1, num_samples=1),
                RandFlipd(keys=("image", "label"), prob=0.5, spatial_axis=0),
                RandRotate90d(keys=("image", "label"), prob=0.5, max_k=3, spatial_axes=(0, 1)),
            ])
        transforms.append(EnsureTyped(keys=("image", "label")))
        self.transforms = Compose(transforms)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        result = self.transforms({"dwi": row.dwi_path, "adc": row.adc_path,
                                  "label": row.mask_path})
        if isinstance(result, list):
            result = result[0]
        return result["image"], (result["label"] > 0).float()
