from datasets.mri_dataset import MRIDataset


dataset = MRIDataset(
    "preprocessing/split_dataset/train.csv",
    train=True
)


image, mask = dataset[0]

for i in range(10):

    image, mask = dataset[0]

    print(
        i,
        mask.unique()
    )