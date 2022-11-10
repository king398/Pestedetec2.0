import torch
import torch.nn as nn
import pandas as pd
import glob
import os
import random
import cv2
import numpy as np
import timm
import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)


class Model(nn.Module):
    def __init__(self, model_name, pretrained=True):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=1)

    def forward(self, x):
        x = self.model(x)
        return x


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
        path = self.ids[index]
        id = path.split('/')[-1].split('.')[0]


        img = cv2.imread(path)
        if os.path.exists(f'{self.path}/labels/val/{id}.txt'):
            label = 1
        else:
            label = 0
        if self.transforms:
            img = self.transforms(image=img)['image']

        return img, torch.tensor(label)


def oof_fn(model, device, loader, ):
    model.eval()

    preds = []
    loader = tqdm(loader)
    with torch.no_grad():
        for img, label in loader:
            img = img.to(device)
            with torch.cuda.amp.autocast():
                pred = model(img)
            preds.append(pred.sigmoid().detach().cpu().numpy())
    preds = np.concatenate(preds)
    preds = preds.reshape(-1)
    return preds


final_val = []
final_pred = []
for i in range(5):
    print(f'Fold {i}')
    path = f"/home/mithil/PycharmProjects/Pestedetec2.0/notebooks/yolov5/dataset/fold_{i}"
    val_ids = glob.glob(f"{path}/images/val/*")

    val_dataset = BollwormDataset(path, val_ids, val=True, transforms=Compose([
        Resize(512, 512),
        Normalize(),
        ToTensorV2()
    ]))
    val_ids = [i.split('/')[-1].split('.')[0] for i in val_ids]
    final_val.extend(val_ids)
    val_dl = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)
    model = Model('tf_efficientnet_b0_ns', pretrained=False)
    model.load_state_dict(torch.load(f'/home/mithil/PycharmProjects/Pestedetec2.0/models/classfication/model_{i}.pth'))
    model = model.cuda()
    device = torch.device('cuda')
    val_preds = oof_fn(model, device, val_dl)
    final_pred.extend(val_preds)

oof_df = pd.DataFrame()
oof_df['id'] = final_val
oof_df['pred'] = final_pred
oof_df.to_csv('/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/oof.csv', index=False)
