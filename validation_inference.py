from models import *
from Augmentations import *
from DataLoader import *
from utils import *
from criterion import *
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--encoder', type=str, default='b2')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--model_checkpoint', type=str, default='')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--df_train_path', type=str, default='train.csv')
args = parser.parse_args()

#Load train/test dataframes and remove bad samples
train = pd.read_csv(args.df_train_path)
train = train.loc[~train['ImageId'].isin(bad_list)]

#KFOLD split
GLOBAL_FOLD = args.fold
kf = KFold(n_splits=args.kfold, random_state = SEED)

for f, (train_index, test_index) in enumerate(kf.split(train)):
  if f == GLOBAL_FOLD:
    df_train = train.iloc[train_index]
    df_dev = train.iloc[test_index]
    break

valid_dataset = CarDataset(df_dev, transform = valid_transforms)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
enc_name = args.encoder
model = Model(enc_name).cuda()

if args.model_checkpoint:
    load_checkpoint(args.model_checkpoint, model, None)
else:
    load_checkpoint("models/{}_fold{}.pth".format(enc_name, GLOBAL_FOLD), model, None)

BATCH_SIZE = args.batch_size

with torch.no_grad():
    preds = []
    
    for (img_batch, _, _) in tqdm(data.DataLoader(valid_dataset, batch_size = BATCH_SIZE, shuffle = False,
                                                  num_workers=4), disable=True):

        img_batch = img_batch.to(device)
        y_pred = model(img_batch)
        y_pred = y_pred.cpu().data.numpy()
        
        preds.extend(y_pred)
    
np.save('validation_inferences/{}_fold{}.npy'.format(enc_name, fold), np.array(preds))