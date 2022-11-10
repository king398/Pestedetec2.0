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
    def __init__(self, path, ids, transforms=None):
        super().__init__()
        self.path = path
        self.ids = ids
        self.transforms = transforms

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index, ):
        id = self.ids[index]
        img = cv2.imread(f'{self.path}/{id}.jpg')
        if img is None:
            img = cv2.imread(f'{self.path}/{id}.jpeg')

        if self.transforms:
            img = self.transforms(image=img)['image']

        return img


def inference(model, loader, device):
    model.eval()
    preds = []
    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(device)
            pred = model(images)
            preds.append(pred.sigmoid().detach().cpu().numpy())

    return np.concatenate(preds).reshape(-1)


preds = None
ids = glob.glob('/home/mithil/PycharmProjects/Pestedetec2.0/notebooks/yolov5/test_images/*')
ids = [os.path.basename(id) for id in ids]
ids = [id.split('.')[0] for id in ids]
dataset = BollwormDataset('/home/mithil/PycharmProjects/Pestedetec2.0/notebooks/yolov5/test_images', ids,
                          transforms=Compose([
                              Resize(512, 512),
                              Normalize(),
                              ToTensorV2()
                          ]))
loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)
for i in range(5):
    model = Model('tf_efficientnet_b0_ns', pretrained=False)
    model.load_state_dict(
        torch.load(f'/home/mithil/PycharmProjects/Pestedetec2.0/models/classfication/model_{i}.pth'))
    model.to('cuda')
    pred = inference(model, loader, 'cuda')
    if preds is None:
        preds = pred / 5
    else:
        preds += pred / 5

df = pd.DataFrame()
df['id'] = ids
df['label'] = preds
df.to_csv('/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/inference_classifier.csv', index=False)
