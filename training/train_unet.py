import torch

from torch.utils.data import DataLoader

from monai.inferers import sliding_window_inference


from datasets.mri_dataset import MRIDataset

from models.unet3d import create_unet3d

from training.loss import (
    SegmentationLoss,
    dice_score,
    iou_score
)



# ==========================
# настройки
# ==========================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


EPOCHS = 20

BATCH_SIZE = 1

LEARNING_RATE = 1e-4


PATCH_SIZE = (
    96,
    96,
    96
)


TRAIN_CSV = (
    "split_dataset/train.csv"
)


VAL_CSV = (
    "split_dataset/val.csv"
)


MODEL_PATH = "/content/drive/MyDrive/stroke_project/models/best_unet3d.pth"



# ==========================
# Dataset
# ==========================


train_dataset = MRIDataset(
    TRAIN_CSV,
    train=True
)


val_dataset = MRIDataset(
    VAL_CSV,
    train=False
)



train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)



val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False
)



# ==========================
# model
# ==========================


model = create_unet3d()

model.to(DEVICE)



criterion = SegmentationLoss()



optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)



best_dice = 0



# ==========================
# обучение
# ==========================


for epoch in range(EPOCHS):


    print(
        f"\nEpoch {epoch+1}/{EPOCHS}"
    )


    # ----------------------
    # TRAIN
    # ----------------------

    model.train()


    train_loss = 0



    for image, mask in train_loader:


        image = image.to(DEVICE)

        mask = mask.to(DEVICE)



        prediction = model(image)



        loss = criterion(
            prediction,
            mask
        )



        optimizer.zero_grad()

        loss.backward()

        optimizer.step()



        train_loss += loss.item()



    train_loss /= len(train_loader)



    print(
        "Train loss:",
        train_loss
    )



    # ----------------------
    # VALIDATION
    # ----------------------

    model.eval()


    val_loss = 0

    val_dice = 0

    val_iou = 0



    with torch.no_grad():


        for image, mask in val_loader:


            image = image.to(DEVICE)

            mask = mask.to(DEVICE)



            # ВАЖНО:
            # полный MRI проходит через sliding window


            prediction = sliding_window_inference(

                inputs=image,

                roi_size=PATCH_SIZE,

                sw_batch_size=1,

                predictor=model

            )



            loss = criterion(
                prediction,
                mask
            )



            val_loss += loss.item()



            val_dice += dice_score(
                prediction,
                mask
            ).item()



            val_iou += iou_score(
                prediction,
                mask
            ).item()



    val_loss /= len(val_loader)

    val_dice /= len(val_loader)

    val_iou /= len(val_loader)



    print(
        "Val loss:",
        val_loss
    )


    print(
        "Dice:",
        val_dice
    )


    print(
        "IoU:",
        val_iou
    )



    # ----------------------
    # сохраняем модель
    # ----------------------


    if val_dice > best_dice:


        best_dice = val_dice


        torch.save(
            model.state_dict(),
            MODEL_PATH
        )


        print(
            "Сохранены лучшие веса"
        )



print("\nОбучение завершено")

print(
    "Лучший Dice:",
    best_dice
)