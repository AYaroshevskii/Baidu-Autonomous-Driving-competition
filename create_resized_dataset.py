""" For speed up training processes, convert images and masks to numpy """

from tqdm import tqdm_notebook
import sys

sys.path.append("..")
from torch.utils.data import Dataset
from preprocessing import *
from utils import *


class CarDataset(Dataset):
    def __init__(self, dataframe, root_dir, transform=None):
        self.df = dataframe
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        # Get image name
        idx, labels = self.df.values[idx]
        img_name = self.root_dir.format(idx)

        # Read image
        img0 = imread(img_name, True)
        img = preprocess_image(img0)

        # Get mask and regression maps
        if self.training:
            mask, regr = get_mask_and_regr(img0, labels)
        else:
            mask, regr = 0, 0

        return [img, mask, regr, idx]


train_images_dir = "train_images/{}.jpg"
test_images_dir = "test_images/{}.jpg"


df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("sample_submission.csv")
train = train.loc[~train["ImageId"].isin(bad_list)]

# Create dataset objects
train_dataset = CarDataset(df_train, train_images_dir)
test_dataset = CarDataset(df_test, test_images_dir)

for i in tqdm_notebook(range(len(train_dataset))):
    x = train_dataset.__getitem__(i)

    np.save("resized_train/{}.npy".format(x[-1]), x[0].astype("uint8"))
    np.save("resized_labels2/mask_{}.npy".format(x[-1]), x[1].astype("float16"))
    np.save("resized_labels2/regr_{}.npy".format(x[-1]), x[2].astype("float16"))

for i in tqdm_notebook(range(len(test_dataset))):
    x = test_dataset.__getitem__(i)

    np.save("resized_test/{}.npy".format(x[-1]), x[0].astype("float16"))
