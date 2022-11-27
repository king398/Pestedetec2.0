import shutil
import pandas as pd
from tqdm.auto import tqdm
import os

test_pseudo_kf = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/data/yolov5l6-1536-image-size-25-epoch-pseudo.csv')
test_image_path = '/home/mithil/PycharmProjects/Pestedetec2.0/notebooks/yolov5/test_images'
test_labels_path = '/home/mithil/PycharmProjects/Pestedetec2.0/data/pseudo_labels/yolov5l6-1536-image-size-25-epoch-mskf'
for i in tqdm(range(5)):
    val_df = test_pseudo_kf[test_pseudo_kf['fold'] == i].reset_index(drop=True)
    os.makedirs(f'/home/mithil/PycharmProjects/Pestedetec2.0/data/dataset/pseudo/fold_{i}/images/', exist_ok=True)
    os.makedirs(f'/home/mithil/PycharmProjects/Pestedetec2.0/data/dataset/pseudo/fold_{i}/labels/', exist_ok=True)
    for j in tqdm(val_df['image_id'].unique()):
        shutil.copy(f'{test_image_path}/{j}',
                    f'/home/mithil/PycharmProjects/Pestedetec2.0/data/dataset/pseudo/fold_{i}/images')
        if os.path.exists(f'{test_labels_path}/{j.split(".")[0]}.txt'):
            shutil.copy(f'{test_labels_path}/{j.split(".")[0]}.txt',
                        f'/home/mithil/PycharmProjects/Pestedetec2.0/data/dataset/pseudo/fold_{i}/labels')
