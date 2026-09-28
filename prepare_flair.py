"""Resample large FLAIR volumes to the DWI grid without loading them into RAM."""

import gc
import gzip
from pathlib import Path
import shutil
import tempfile

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.ndimage import affine_transform


ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(r"D:\ISLES-2022")
OUTPUT_ROOT = Path(r"D:\ISLES-2022-prepared")
TEMP_ROOT = Path(r"D:\brain-stroke-install")


def dataset_path(value):
    return DATA_ROOT.joinpath(*str(value).replace("\\", "/").split("/")[1:])


def output_path(row):
    return OUTPUT_ROOT / f"{row.subject_id}_{row.session_id}_FLAIR.nii.gz"


def prepare_case(row):
    output = output_path(row)
    if output.is_file():
        return output
    OUTPUT_ROOT.mkdir(exist_ok=True)
    TEMP_ROOT.mkdir(exist_ok=True)
    resample_flair_to_dwi(dataset_path(row.flair_path), dataset_path(row.dwi_path), output)
    return output


def resample_flair_to_dwi(flair_path, dwi_path, output):
    """Write a compact FLAIR aligned to DWI while streaming the original volume."""
    flair_path = Path(flair_path)
    output = Path(output)
    output.parent.mkdir(exist_ok=True)
    TEMP_ROOT.mkdir(exist_ok=True)
    dwi = nib.load(str(dwi_path))
    with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
        plain = Path(folder) / "flair.nii"
        with gzip.open(flair_path, "rb") as source, plain.open("wb") as target:
            shutil.copyfileobj(source, target, length=4 * 1024 * 1024)
        flair = nib.load(str(plain), mmap="r")
        proxy = flair.dataobj
        source = np.memmap(plain, dtype=proxy.dtype, mode="r", offset=proxy.offset,
                           shape=proxy.shape, order="F")
        matrix = np.linalg.inv(flair.affine) @ dwi.affine
        sampled = affine_transform(source, matrix[:3, :3], offset=matrix[:3, 3],
                                   output_shape=dwi.shape[:3], output=np.float32,
                                   order=1, mode="constant", cval=0, prefilter=False)
        if proxy.slope != 1 or proxy.inter != 0:
            sampled = sampled * proxy.slope + proxy.inter
        del source, flair
        gc.collect()
    image = nib.Nifti1Image(sampled, dwi.affine)
    image.header.set_data_dtype(np.float32)
    nib.save(image, str(output))
    return output


def main():
    OUTPUT_ROOT.mkdir(exist_ok=True)
    TEMP_ROOT.mkdir(exist_ok=True)
    frames = [pd.read_csv(ROOT / "split_dataset" / f"{split}.csv") for split in ("train", "val")]
    cases = pd.concat(frames, ignore_index=True).drop_duplicates(["subject_id", "session_id"])
    print(f"Preparing FLAIR for {len(cases)} cases", flush=True)
    for index, row in enumerate(cases.itertuples(index=False), 1):
        output = prepare_case(row)
        print(f"FLAIR {index}/{len(cases)} ({100*index/len(cases):.0f}%): {output.name}", flush=True)
    print(f"Prepared FLAIR volumes: {OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
