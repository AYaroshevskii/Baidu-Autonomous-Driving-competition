from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd


class CarDataset(Dataset):
    def __init__(self, dataframe, root_dir="train", transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # Get image name
        idx, labels = self.df.values[idx]

        # Read image
        img = np.load("resized_{}/{}.npy".format(self.root_dir, idx)).astype("float32")
        img = (img / 255).astype("float32")

        # Get mask and regression maps
        if self.root_dir == "train":

            mask = np.load("resized_labels/mask_{}.npy".format(idx)).astype("float32")
            regr = np.load("resized_labels/regr_{}.npy".format(idx)).astype("float32")

            if self.transform is not None:
                img, mask, regr = self.transform((img, mask, regr))

            regr = np.rollaxis(regr, 2, 0)
            img = np.rollaxis(img, 2, 0)

        else:

            mask, regr = 0, 0
            # TTA if needed
            if self.transform is not None:
                img, mask, regr = self.transform((img, mask, regr))
            img = np.rollaxis(img, 2, 0)

        return [img, mask, regr]
