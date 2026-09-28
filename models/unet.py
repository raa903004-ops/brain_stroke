import torch
import torch.nn as nn


class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True),


            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(out_channels),

            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        return self.block(x)



class UNet(nn.Module):

    def __init__(
            self,
            in_channels=3,
            out_channels=1
    ):

        super().__init__()


        # =====================
        # Encoder
        # =====================

        self.down1 = DoubleConv(
            in_channels,
            64
        )

        self.pool1 = nn.MaxPool2d(2)


        self.down2 = DoubleConv(
            64,
            128
        )

        self.pool2 = nn.MaxPool2d(2)


        self.down3 = DoubleConv(
            128,
            256
        )

        self.pool3 = nn.MaxPool2d(2)



        # =====================
        # Bottleneck
        # =====================

        self.bottom = DoubleConv(
            256,
            512
        )



        # =====================
        # Decoder
        # =====================

        self.up3 = nn.ConvTranspose2d(
            512,
            256,
            kernel_size=2,
            stride=2
        )


        self.conv3 = DoubleConv(
            512,
            256
        )



        self.up2 = nn.ConvTranspose2d(
            256,
            128,
            kernel_size=2,
            stride=2
        )


        self.conv2 = DoubleConv(
            256,
            128
        )



        self.up1 = nn.ConvTranspose2d(
            128,
            64,
            kernel_size=2,
            stride=2
        )


        self.conv1 = DoubleConv(
            128,
            64
        )



        # последний слой
        # получаем карту вероятности инсульта

        self.out = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )



    def forward(self, x):


        # =====================
        # Encoder
        # =====================

        x1 = self.down1(x)


        x2 = self.down2(
            self.pool1(x1)
        )


        x3 = self.down3(
            self.pool2(x2)
        )



        # =====================
        # Bottleneck
        # =====================

        x4 = self.bottom(
            self.pool3(x3)
        )



        # =====================
        # Decoder
        # =====================


        x = self.up3(x4)


        x = torch.cat(
            [
                x,
                x3
            ],
            dim=1
        )


        x = self.conv3(x)



        x = self.up2(x)


        x = torch.cat(
            [
                x,
                x2
            ],
            dim=1
        )


        x = self.conv2(x)



        x = self.up1(x)


        x = torch.cat(
            [
                x,
                x1
            ],
            dim=1
        )


        x = self.conv1(x)



        # ВАЖНО:
        # без sigmoid
        # будем использовать BCEWithLogitsLoss

        return self.out(x)