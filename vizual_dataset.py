import matplotlib.pyplot as plt

from datasets.mri_dataset import MRIDataset


dataset = MRIDataset(
    "preprocessing/split_dataset/train.csv",
    train=True
)


image, mask = dataset[0]


# берём средний срез по глубине

slice_id = image.shape[-1] // 2


dwi = image[0, :, :, slice_id]

adc = image[1, :, :, slice_id]

flair = image[2, :, :, slice_id]

mask_slice = mask[0, :, :, slice_id]


plt.figure(figsize=(12,4))


plt.subplot(1,4,1)
plt.imshow(dwi, cmap="gray")
plt.title("DWI")


plt.subplot(1,4,2)
plt.imshow(adc, cmap="gray")
plt.title("ADC")


plt.subplot(1,4,3)
plt.imshow(flair, cmap="gray")
plt.title("FLAIR")


plt.subplot(1,4,4)
plt.imshow(flair, cmap="gray")
plt.imshow(
    mask_slice,
    alpha=0.5
)
plt.title("FLAIR + MASK")


plt.show()