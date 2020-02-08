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
parser.add_argument("--kfold", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--model_checkpoint", type=str, default="")
parser.add_argument("--max_epoch", type=int, default=50)
parser.add_argument("--start_epoch", type=int, default=0)
parser.add_argument("--seed", type=int, default=123)
parser.add_argument("--df_train_path", type=str, default="train.csv")
parser.add_argument("--df_test_path", type=str, default="sample_submission.csv")
args = parser.parse_args()

# Seed everything
SEED = args.seed

torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

np.random.seed(SEED)
random.seed(SEED)


def _worker_init_fn(worker_id):
    np.random.seed(worker_id)


# Load train/test dataframes and remove bad samples
train = pd.read_csv(args.df_train_path)
df_test = pd.read_csv(args.df_test_path)
train = train.loc[~train["ImageId"].isin(bad_list)]


# KFOLD split
GLOBAL_FOLD = args.fold
kf = KFold(n_splits=args.kfold, random_state=SEED)

for f, (train_index, test_index) in enumerate(kf.split(train)):
    if f == GLOBAL_FOLD:
        df_train = train.iloc[train_index]
        df_dev = train.iloc[test_index]
        break

# Create dataset objects
train_dataset = CarDataset(df_train, transform=train_transforms)
valid_dataset = CarDataset(df_dev)
test_dataset = CarDataset(df_test, root_dir="test", transform=test_transforms)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Model(enc_name).to(device)
multi_loss = criterion

optimizer = torch.optim.Adam(model.parameters(), 1e-5)
scheduler_decay = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=1 / np.sqrt(10),
    threshold=0,
    patience=1,
    verbose=True,
    min_lr=1e-5,
)
scheduler_warmup = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=10)

if args.model_checkpoint:
    load_checkpoint(args.model_checkpoint, model, optimizer)

start_epoch = args.start_epoch
NUM_EPOCHS = args.max_epoch
BATCH_SIZE = args.batch_size
WARMUP_epoch = 2
best_score = 100000

for epoch in range(start_epoch, NUM_EPOCHS):
    print("Epoch : ", epoch)

    # Train
    train_loss = []
    train_regr_loss = []
    model.train()

    for (img_batch, mask_batch, regr_batch) in tqdm(
        data.DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            worker_init_fn=_worker_init_fn,
        ),
        disable=True,
    ):

        img_batch = img_batch.to(device)
        mask_batch = mask_batch.to(device)
        regr_batch = regr_batch.to(device)

        y_pred = model(img_batch)

        loss, mask_loss, regr_loss = multi_loss(y_pred, mask_batch, regr_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())
        train_regr_loss.append(regr_loss.item())

    if epoch < WARMUP_epoch:
        scheduler_warmup.step()
        print("Loss : {}".format(np.mean(train_loss)))
        print("Regr Loss : {}".format(np.mean(train_regr_loss)))
        continue

    # Validation
    with torch.no_grad():

        unet_score = []
        regr_val_score = []
        mean_ap_metric = 0
        mean_f_score = 0

        for (img_batch, mask_batch, regr_batch) in tqdm(
            data.DataLoader(
                valid_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=2,
                worker_init_fn=_worker_init_fn,
            ),
            disable=True,
        ):

            img_batch = img_batch.to(device)
            mask_batch = mask_batch.to(device)
            regr_batch = regr_batch.to(device)

            y_pred = model(img_batch)

            loss, mask_loss, regr_loss = multi_loss(y_pred, mask_batch, regr_batch)

            unet_score.append(loss.item())
            regr_val_score.append(regr_loss.item())

            mask_pred = torch.sigmoid(y_pred[:, 0]) > 0.5

            # Per Image
            for s in range(y_pred.shape[0]):
                mean_ap_metric += average_precision_score(
                    mask_batch[s, :, :].cpu().data.numpy().astype(int).reshape(-1),
                    mask_pred[s, :, :].cpu().data.numpy().astype(int).reshape(-1),
                )
                mean_f_score += f1_score(
                    mask_batch[s, :, :].cpu().data.numpy().astype(int).reshape(-1),
                    mask_pred[s, :, :].cpu().data.numpy().astype(int).reshape(-1),
                )

        mean_ap_metric /= len(valid_dataset)
        mean_f_score /= len(valid_dataset)
        val_loss = np.mean(unet_score)

        if np.mean(regr_val_score) < best_score:
            save_checkpoint(
                "models/{}_fold{}.pth".format(enc_name, GLOBAL_FOLD), model, optimizer
            )
            best_score = np.mean(regr_val_score)

        print("Val regr Loss : {}".format(np.mean(regr_val_score)))
        print("Regr Loss : {}".format(np.mean(train_regr_loss)))
        print(
            "Train Loss : {}, Validation Loss : {}".format(
                np.mean(train_loss), val_loss
            )
        )
        print("mAP Metric : {}, mF1_score : {}".format(mean_ap_metric, mean_f_score))

        scheduler_decay.step(np.mean(regr_val_score))
