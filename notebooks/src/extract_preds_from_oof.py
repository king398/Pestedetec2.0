import pandas as pd
import os
import numpy as np
from statistics import mean

train_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
train_labels_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_modified.csv')
ids = []
labels = []
pred_labels_path = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5s6_image_size_1024_oof'
id_label_dict = dict(zip(train_labels_df['image_id'].values, train_labels_df['number_of_worms'].values))


def make_labels(id):
    id = id.split('.')[0]
    ids.extend([f"{id}_pbw.jpg", f"{id}_abw.jpg"])
    pbw = 0
    abw = 0
    if os.path.exists(
            f'{pred_labels_path}/{id}.txt'):
        with open(
                f'{pred_labels_path}/{id}.txt') as f:
            preds_per_line = f.readlines()
            for i in preds_per_line:
                if i.split(' ')[0] == '0':
                    pbw += 1
                else:
                    abw += 1
    labels.extend([pbw, abw])


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


list(map(make_labels, train_df['image_id_worm'].values))
oof = pd.DataFrame({'image_id_worm': ids, 'label': labels}, index=None)
oof.to_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/oof_df/yolov5s6_image_size_1024_oof.csv',
    index=False)
pred_label_dict = dict(zip(oof['image_id_worm'].values, oof['label'].values))


def return_error(id):
    id = id.split('.')[0]
    pbw_label = id_label_dict[f'{id}_pbw.jpg']
    abw_label = id_label_dict[f'{id}_abw.jpg']
    pbw_pred = pred_label_dict[f'{id}_pbw.jpg']
    abw_pred = pred_label_dict[f'{id}_abw.jpg']
    error = mae(np.array(pbw_label), np.array(pbw_pred)) + mae(np.array(abw_label), np.array(abw_pred))
    return error


error = list(map(return_error, train_df['image_id_worm'].values))
print(mean(error))
