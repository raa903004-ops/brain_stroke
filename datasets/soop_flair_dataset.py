"""Three-channel SOOP cohort after rigid FLAIR-to-TRACE registration."""

from pathlib import Path

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    ResampleToMatchd, NormalizeIntensityd, ConcatItemsd, SpatialPadd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, EnsureTyped,
)
from torch.utils.data import Dataset

from datasets.soop_dataset import build_manifest, PATCH_SIZE


REGISTERED = Path(r"D:\SOOP-registered-flair")


def build_flair_manifest():
    manifest = build_manifest().copy()
    manifest["flair_path"] = manifest.subject_id.map(
        lambda subject: str(REGISTERED / f"{subject}_flair_on_trace.nii.gz"))
    manifest = manifest[manifest.flair_path.map(lambda path: Path(path).is_file())]
    return manifest.reset_index(drop=True)


class SOOPFlairDataset(Dataset):
    def __init__(self, manifest, split):
        self.data = manifest.loc[manifest.split == split].reset_index(drop=True)
        keys = ("dwi", "adc", "flair", "label")
        transforms = [
            LoadImaged(keys=keys),
            EnsureChannelFirstd(keys=keys),
            Orientationd(keys=keys, axcodes="RAS"),
            ResampleToMatchd(keys=("adc", "flair", "label"), key_dst="dwi",
                             mode=("bilinear", "bilinear", "nearest")),
            NormalizeIntensityd(keys=("dwi", "adc", "flair"), nonzero=True, channel_wise=True),
            ConcatItemsd(keys=("dwi", "adc", "flair"), name="image", dim=0),
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
                                  "flair": row.flair_path, "label": row.mask_path})
        if isinstance(result, list):
            result = result[0]
        return result["image"], (result["label"] > 0).float()
