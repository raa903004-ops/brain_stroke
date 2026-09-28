import torch
import torch.nn as nn

from monai.losses import DiceLoss


class SegmentationLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.dice = DiceLoss(
            sigmoid=True
        )

        self.bce = nn.BCEWithLogitsLoss()


    def forward(self, prediction, target):

        dice_loss = self.dice(
            prediction,
            target
        )

        bce_loss = self.bce(
            prediction,
            target
        )

        return dice_loss + bce_loss



def dice_score(prediction, target, threshold=0.5):

    prediction = torch.sigmoid(prediction)

    prediction = (
        prediction > threshold
    ).float()


    intersection = (
        prediction * target
    ).sum()


    dice = (
        2.0 * intersection
        /
        (
            prediction.sum()
            +
            target.sum()
            + 1e-8
        )
    )


    return dice



def iou_score(prediction, target, threshold=0.5):

    prediction = torch.sigmoid(prediction)

    prediction = (
        prediction > threshold
    ).float()


    intersection = (
        prediction * target
    ).sum()


    union = (
        prediction.sum()
        +
        target.sum()
        -
        intersection
    )


    iou = (
        intersection
        /
        (union + 1e-8)
    )


    return iou