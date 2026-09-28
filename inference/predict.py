import torch
import numpy as np
from scipy.ndimage import label

from models.unet3d import create_unet3d


def count_lesions(mask, min_size=1):
    """Count connected regions and return a compact label map and pixel sizes."""
    regions, total = label(np.asarray(mask, dtype=bool))
    filtered = np.zeros_like(regions, dtype=np.int32)
    sizes = []
    for region_id in range(1, total + 1):
        region = regions == region_id
        size = int(region.sum())
        if size >= min_size:
            sizes.append(size)
            filtered[region] = len(sizes)
    return len(sizes), filtered, sizes



def load_model(path):

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = create_unet3d()

    checkpoint = torch.load(
        path,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)

    model.eval()

    return model, device



def predict(
        model,
        image,
        device
):

    image = torch.tensor(
        image,
        dtype=torch.float32
    )

    # добавляем batch dimension
    image = image.unsqueeze(0)

    # отправляем на GPU
    image = image.to(device)


    with torch.no_grad():

        pred = model(image)


    probability = (
        torch.sigmoid(pred)
        .squeeze()
        .cpu()
        .numpy()
    )


    mask = (
        probability > 0.5
    )


    return mask, probability
