import pandas as pd
import os
import numpy as np
from statistics import mean, mode
import random
from ensemble_boxes import soft_nms

train_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
train_labels_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_modified.csv')
ids = []
labels = []
pred_labels_path = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5m6-1536-image-size-25-epoch-mskf'
pred_labels_path_2 = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5m6-1536-image-size-20-epoch-mskf'
id_label_dict = dict(zip(train_labels_df['image_id'].values, train_labels_df['number_of_worms'].values))

classifier_pred = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/tf_effnet_b2_1024_image_size.csv')
classifier_pred_dict = dict(zip(classifier_pred['id'].values, classifier_pred['pred'].values))


def make_labels(id):
    id = id.split('.')[0]
    ids.extend([f"{id}_pbw.jpg", f"{id}_abw.jpg"])
    pbw_1 = 0
    abw_1 = 0

    classifier_pred = classifier_pred_dict[id] * 1.0

    if os.path.exists(
            f'{pred_labels_path}/{id}.txt'):
        with open(
                f'{pred_labels_path}/{id}.txt') as f:
            preds_per_line = f.readlines()

            for i in preds_per_line:
                conf = float(i.split(' ')[5])
                if i.split(' ')[0] == '0' and conf > 0.35:
                    pbw_1 += 1
                elif i.split(' ')[0] == '1' and conf > 0.35:
                    abw_1 += 1
    pbw_2 = 0
    abw_2 = 0
    if os.path.exists(f'{pred_labels_path_2}/{id}.txt'):
        with open(
                f'{pred_labels_path_2}/{id}.txt') as f:
            preds_per_line = f.readlines()
            for i in preds_per_line:
                conf = float(i.split(' ')[5])
                if i.split(' ')[0] == '0' and conf > 0.35:
                    pbw_2 += 1
                elif i.split(' ')[0] == '1' and conf > 0.35:
                    abw_2 += 1
    pbw = int(pbw_2 * 0.4 + pbw_1 * 0.6)
    abw = int(abw_2 * 0.4 + abw_1 * 0.6)

    labels.extend([pbw, abw])


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)


list(map(make_labels, train_df['image_id_worm'].values))
oof = pd.DataFrame({'image_id_worm': ids, 'label': labels}, index=None)
oof.to_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/oof_df/yolov5m6-1536-image-size-25-epoch-mskf-yolov5m6-1536-image-size-20-epoch-mskf.csv',
    index=False)
pred_label_dict = dict(zip(oof['image_id_worm'].values, oof['label'].values))


def return_error(id):
    id = id.split('.')[0]
    pbw_label = id_label_dict[f'{id}_pbw.jpg']
    abw_label = id_label_dict[f'{id}_abw.jpg']
    pbw_pred = pred_label_dict[f'{id}_pbw.jpg']
    abw_pred = pred_label_dict[f'{id}_abw.jpg']
    error = float(mae(np.array(pbw_label), np.array(pbw_pred)) + float(mae(np.array(abw_label), np.array(abw_pred))))
    return error


error = list(map(return_error, train_df['image_id_worm'].values))
print(mean(error))
error_random_picked = []
for i in range(69420):
    error_random_picked.append(np.mean(random.choices(error, k=3000)))
print(min(error_random_picked))
