import nibabel as nib
import numpy as np


def load_nifti(path):

    img = nib.load(path)

    data = img.get_fdata()

    return data.astype(np.float32)



def normalize(img):

    img = img - img.min()

    img = img / (img.max()+1e-8)

    return img



def create_input(
        dwi,
        adc,
        flair
):

    dwi = normalize(dwi)
    adc = normalize(adc)
    flair = normalize(flair)


    x = np.stack(
        [
            dwi,
            adc,
            flair
        ],
        axis=0
    )


    return x