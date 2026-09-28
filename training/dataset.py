import os

import nibabel as nib
import numpy as np
import torch

from torch.utils.data import Dataset
from skimage.transform import resize



class ISLESDataset(Dataset):

    def __init__(self, root_dir):

        self.samples = []

        self.root_dir = root_dir


        patients = [
            p for p in os.listdir(root_dir)
            if p.startswith("sub-strokecase")
        ]


        for patient in patients:


            session_path = os.path.join(
                root_dir,
                patient,
                "ses-0001"
            )


            # пути к изображениям

            dwi_path = os.path.join(
                session_path,
                "dwi",
                f"{patient}_ses-0001_dwi.nii.gz"
            )


            adc_path = os.path.join(
                session_path,
                "dwi",
                f"{patient}_ses-0001_adc.nii.gz"
            )


            flair_path = os.path.join(
                session_path,
                "anat",
                f"{patient}_ses-0001_FLAIR.nii.gz"
            )


            # маска

            mask_path = os.path.join(
                root_dir,
                "derivatives",
                patient,
                "ses-0001",
                f"{patient}_ses-0001_msk.nii.gz"
            )


            if (
                os.path.exists(dwi_path)
                and os.path.exists(adc_path)
                and os.path.exists(flair_path)
                and os.path.exists(mask_path)
            ):

                self.samples.append(
                    {
                        "dwi": dwi_path,
                        "adc": adc_path,
                        "flair": flair_path,
                        "mask": mask_path
                    }
                )


        print(
            "Найдено пациентов:",
            len(self.samples)
        )



    def load_nii(self, path):

        img = nib.load(path)

        data = img.get_fdata()

        return data.astype(np.float32)



    def normalize(self, img):

        img = img - np.min(img)

        img = img / (
            np.max(img) + 1e-8
        )

        return img.astype(np.float32)



    def resize_image(
            self,
            img,
            size=(256,256)
    ):

        img = resize(
            img,
            size,
            preserve_range=True
        )

        return img.astype(np.float32)



    def __len__(self):

        return len(self.samples)



    def __getitem__(self,index):


        sample = self.samples[index]


        # =====================
        # загрузка MRI
        # =====================

        dwi = self.load_nii(
            sample["dwi"]
        )

        adc = self.load_nii(
            sample["adc"]
        )

        flair = self.load_nii(
            sample["flair"]
        )

        mask = self.load_nii(
            sample["mask"]
        )


        # =====================
        # нормализация
        # =====================

        dwi = self.normalize(dwi)

        adc = self.normalize(adc)

        flair = self.normalize(flair)



        # =====================
        # берём центральный срез
        # =====================

        num_slices = min(
            dwi.shape[2],
            adc.shape[2],
            flair.shape[2],
            mask.shape[2]
        )

        slice_id = num_slices // 2


        dwi_slice = dwi[:,:,slice_id]

        adc_slice = adc[:,:,slice_id]

        flair_slice = flair[:,:,slice_id]

        mask_slice = mask[:,:,slice_id]



        # =====================
        # resize
        # =====================

        dwi_slice = self.resize_image(
            dwi_slice
        )

        adc_slice = self.resize_image(
            adc_slice
        )

        flair_slice = self.resize_image(
            flair_slice
        )


        mask_slice = self.resize_image(
            mask_slice
        )


        # =====================
        # собираем вход U-Net
        # =====================


        image = np.stack(
            [
                dwi_slice,
                adc_slice,
                flair_slice
            ],
            axis=0
        )


        mask_slice = np.expand_dims(
            mask_slice,
            axis=0
        )


        return (
            torch.tensor(
                image,
                dtype=torch.float32
            ),

            torch.tensor(
                mask_slice,
                dtype=torch.float32
            )
        )