import glob
import os
import random
import cv2
import numpy as np
import timm
import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

os.makedirs('classfication', exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import wandb


wandb.init(project="pesticide")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


class BollwormDataset(Dataset):
    def __init__(self, path, ids, val, transforms=None):
        super().__init__()
        self.path = path
        self.ids = ids
        self.val = val
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index, ):
        id = self.ids[index]

        if self.val:
            img = cv2.imread(f'{self.path}/images/val/{id}.jpg')
            if os.path.exists(f'{self.path}/labels/val/{id}.txt'):
                label = 1
            else:
                label = 0



        else:
            img = cv2.imread(f'{self.path}/images/train/{id}.jpg')
            if os.path.exists(f'{self.path}/labels/train/{id}.txt'):
                label = 1
            else:
                label = 0
        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, torch.tensor(label)


def transform(DIM):
    return Compose([
        Resize(DIM, DIM),
        Normalize(),
        ToTensorV2()

    ])


class Model(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x):
        x = self.model(x)
        return x


def roc_auc_pytorch(y_true, y_pred):
    """Return the roc auc given a pytorch input and a pytorch target"""
    return roc_auc_score(y_true.detach().cpu().numpy(), y_pred.detach().cpu().numpy())


def accuracy(output, target, threshold=0.6):
    output = (output > threshold).int()
    return (output == target).float().mean()


def train(model, optimizer, loader, criterion, scheduler, device, fold):
    model.train()
    final_loss = 0
    scaler = GradScaler()
    stream = tqdm(loader, total=len(loader))
    preds = []
    targets = []
    for data, target in stream:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        with autocast():
            output = model(data)
            loss = criterion(output, target.unsqueeze(1).float())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        final_loss += loss.item()

        scheduler.step()
        preds.append(output.sigmoid().detach().cpu())
        targets.append(target.detach().cpu())
        stream.set_description(f"Train Loss {loss.item() / len(stream):.4f}")
    wandb.log({f"Train Loss Fold {fold}": final_loss / len(loader)})


def val(model, loader, criterion, device, fold):
    model.eval()
    final_loss = 0
    preds = []
    targets = []
    stream = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for data, target in stream:
            data, target = data.to(device), target.to(device)
            with autocast():
                output = model(data)
                loss = criterion(output, target.unsqueeze(1).float())
            final_loss += loss.item()
            preds.append(output.sigmoid().detach().cpu())
            targets.append(target.detach().cpu())
            stream.set_description(f"Val Loss {loss.item() / len(stream):.4f}")
        roc_auc = roc_auc_pytorch(torch.cat(targets), torch.cat(preds))
        accuracy_score = accuracy(torch.cat(preds), torch.cat(targets))
        print(f"Val Loss {final_loss / len(loader)} ROC AUC {roc_auc:.6f} Accuracy {accuracy_score:.6f}")
        wandb.log({f"Val Loss fold {fold}": final_loss / len(loader), f"ROC AUC {fold} ": roc_auc})
        return roc_auc


for i in range(5):
    print(f'Fold {i}')
    path = f"/notebooks/yolov5/dataset/fold_{i}"
    train_ids = glob.glob(f"{path}/images/train/*.jpg")
    train_ids = [i.split('/')[-1].split('.')[0] for i in train_ids]
    val_ids = glob.glob(f"{path}/images/val/*.jpg")
    val_ids = [i.split('/')[-1].split('.')[0] for i in val_ids]
    train_ds = BollwormDataset(path, train_ids, False, transforms=transform(1024))
    val_ds = BollwormDataset(path, val_ids, True, transforms=transform(1024))
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)
    model = Model('tf_efficientnet_b2_ns', pretrained=True)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3, steps_per_epoch=len(train_dl), epochs=10)

    for epoch in range(10):
        print(f"Epoch {epoch} Started")
        train(model, optimizer, train_dl, criterion, scheduler, device, i)
        val(model, val_dl, criterion, device, i)

    torch.save(model.state_dict(), f"classfication/model_{i}.pth")
