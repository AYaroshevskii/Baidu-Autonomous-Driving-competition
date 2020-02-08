from models import *
from Augmentations import *
from DataLoader import *
from utils import *
from criterion import *
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--encoder", type=str, default="b2")
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--model_checkpoint", type=str, default="")
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--df_test_path", type=str, default="sample_submission.csv")
args = parser.parse_args()


GLOBAL_FOLD = args.fold
df_test = pd.read_csv(args.df_test_path)
test_dataset = CarDataset(df_test, root_dir="test", transform=test_transforms)

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

    for (img_batch, _, _) in tqdm(
        data.DataLoader(
            test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
        ),
        disable=True,
    ):

        img_batch = img_batch.to(device)
        y_pred = model(img_batch)
        y_pred = y_pred.cpu().data.numpy()

        preds.extend(y_pred)


np.save("inferences/{}_fold{}.npy".format(enc_name, fold), np.array(preds))
