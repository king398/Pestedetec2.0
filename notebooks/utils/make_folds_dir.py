import shutil
import os
import pandas as pd
from tqdm.auto import tqdm

train_kf = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_mskf.csv')
for i in tqdm(range(5)):
    val_df = train_kf[train_kf['fold'] == i].reset_index(drop=True)
    os.makedirs('/home/mithil/PycharmProjects/Pestedetec2.0/data/folds/fold_{}/images/'.format(i), exist_ok=True)
    os.makedirs('/home/mithil/PycharmProjects/Pestedetec2.0/data/folds/fold_{}/labels/'.format(i), exist_ok=True)
    for j in tqdm(val_df['image_id'].unique()):
        shutil.copy(f'/home/mithil/PycharmProjects/Pestedetec2.0/data/train_images/{j}',
                    f'/home/mithil/PycharmProjects/Pestedetec2.0/data/folds/fold_{i}/images')
        if os.path.exists(f'/home/mithil/PycharmProjects/Pestedetec2.0/data/labels/{j.split(".")[0]}.txt'):
            shutil.copy(f'/home/mithil/PycharmProjects/Pestedetec2.0/data/labels/{j.split(".")[0]}.txt',
                        f'/home/mithil/PycharmProjects/Pestedetec2.0/data/folds/fold_{i}/labels')
