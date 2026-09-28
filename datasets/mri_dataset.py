import pandas as pd
import torch

from torch.utils.data import Dataset
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ResampleToMatchd,
    NormalizeIntensityd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    ConcatItemsd,
    EnsureTyped,
    SpatialPadd
)


class MRIDataset(Dataset):
    """
    Dataset для ISLES-2022.

    Вход:
        DWI
        ADC
        FLAIR

    Выход:
        image:
            [DWI, ADC, FLAIR]

        label:
            mask ишемического очага
    """

    def __init__(
            self,
            csv_file,
            patch_size=(96, 96, 96),
            train=True
    ):

        self.data = pd.read_csv(csv_file)

        self.train = train


        transforms = [

            # -----------------------------
            # загрузка NIfTI
            # -----------------------------

            LoadImaged(
                keys=[
                    "dwi",
                    "adc",
                    "flair",
                    "label"
                ]
            ),


            # -----------------------------
            # добавляем размер канала
            # -----------------------------

            EnsureChannelFirstd(
                keys=[
                    "dwi",
                    "adc",
                    "flair",
                    "label"
                ]
            ),


            # -----------------------------
            # единая ориентация
            # -----------------------------

            Orientationd(
                keys=[
                    "dwi",
                    "adc",
                    "flair",
                    "label"
                ],
                axcodes="RAS"
            ),

            # используем компактную сетку DWI, чтобы избежать переполнения памяти

            ResampleToMatchd(
                keys=[
                    "adc",
                    "flair",
                    "label"
                ],

                key_dst="dwi",

                mode=[
                    "bilinear",
                    "bilinear",
                    "nearest"
                ]
            ),


            # -----------------------------
            # нормализация МРТ
            # -----------------------------

            NormalizeIntensityd(
                keys=[
                    "dwi",
                    "adc",
                    "flair"
                ],

                nonzero=True,

                channel_wise=True
            ),


            # -----------------------------
            # объединяем 3 модальности
            # -----------------------------

            ConcatItemsd(
                keys=[
                    "dwi",
                    "adc",
                    "flair"
                ],

                name="image",

                dim=0
            ),
            SpatialPadd(
                keys=[
                    "image",
                    "label"
                ],
                spatial_size=(
                    96,
                    96,
                    96
                )
            ),

        ]


        # только при обучении делаем crop
        if train:

            transforms.append(

                RandCropByPosNegLabeld(
                    keys=[
                        "image",
                        "label"
                    ],

                    label_key="label",

                    spatial_size=(
                        96,
                        96,
                        96
                    ),

                    pos=1,

                    neg=1,

                    num_samples=1
                )
            )


            transforms.append(

                RandFlipd(

                    keys=[
                        "image",
                        "label"
                    ],

                    prob=0.5,

                    spatial_axis=0
                )
            )


            transforms.append(

                RandRotate90d(

                    keys=[
                        "image",
                        "label"
                    ],

                    prob=0.5,

                    max_k=3
                )
            )


        transforms.append(

            EnsureTyped(
                keys=[
                    "image",
                    "label"
                ]
            )
        )


        self.transforms = Compose(transforms)



    def __len__(self):

        return len(self.data)



    def __getitem__(self, index):

        row = self.data.iloc[index]


        patient = {

            "dwi":
                row["dwi_path"],

            "adc":
                row["adc_path"],

            "flair":
                row["flair_path"],

            "label":
                row["mask_path"]

        }


        data = self.transforms(patient)


        # RandCropByPosNegLabeld
        # возвращает список

        if isinstance(data, list):
            data = data[0]


        image = data["image"]

        label = data["label"]


        # делаем маску бинарной

        label = (label > 0).float()


        return image, label
