import torch

from monai.networks.nets import UNet


def create_unet3d(in_channels=3):
    return UNet(
        spatial_dims=3,
        in_channels=in_channels,
        out_channels=1,
        channels=(
            8,
            16,
            32,
            64,
            128
        ),
        strides=(
            2,
            2,
            2,
            2
        ),
        num_res_units=2
    )


    return model



if __name__ == "__main__":


    model = create_unet3d()


    x = torch.randn(
        1,
        3,
        96,
        96,
        96
    )


    y = model(x)


    print(
        "Input:",
        x.shape
    )


    print(
        "Output:",
        y.shape
    )
