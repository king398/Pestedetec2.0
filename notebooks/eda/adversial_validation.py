import pandas as pd
import os
from fastai.vision import *
from fastai.vision.data import *
from fastai.vision.all import *

import numpy as np
from fastai import *
import torch.optim as optim
from torchvision import transforms
from torch.optim import lr_scheduler
import os
import torch
import glob

device = torch.device("cuda:0")

train_path = glob.glob(f"/home/mithil/PycharmProjects/Pestedetec2.0/notebooks/yolov5/dataset/fold_0/images/*/*")
train_labels = np.zeros(len(train_path))
test_path = glob.glob(f"/home/mithil/PycharmProjects/Pestedetec2.0/notebooks/yolov5/test_images/*")
test_labels = np.ones(len(test_path))
df = pd.DataFrame.from_dict(
    {'path': np.concatenate((train_path, test_path)), 'label': np.concatenate((train_labels, test_labels))})
df = df.sample(frac=1).reset_index(drop=True)
df.head()
dls = ImageDataLoaders.from_df(df, bs=64, item_tfms=[Resize(256, Normalize.from_stats(*imagenet_stats))], path='/',
                               valid_pct=0.4)
learn = cnn_learner(dls, models.resnet34, metrics=RocAucBinary)
print(learn.loss_func)
# lr = learn.lr_find()
