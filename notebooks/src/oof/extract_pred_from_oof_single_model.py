import pandas as pd
import os
import numpy as np
from statistics import mean, mode
from ensemble_boxes import *
import cv2
import matplotlib.pyplot as plt
from pybboxes import BoundingBox
from joblib import Parallel, delayed
from tqdm import tqdm

train_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
train_labels_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_modified.csv')
ids = []
labels = []
pred_labels_path = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/mskf/yolov5l6-1536-image-size-25-epoch-mskf'
id_label_dict = dict(zip(train_labels_df['image_id'].values, train_labels_df['number_of_worms'].values))

classifier_pred = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/tf_effnet_b2_1024_image_size.csv')
classifier_pred_dict = dict(zip(classifier_pred['id'].values, classifier_pred['pred'].values))


def gen_color_list(model_num, labels_num):
    color_list = np.zeros((model_num, labels_num, 3))
    colors_to_use = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 0, 0)]
    total = 0
    for i in range(model_num):
        for j in range(labels_num):
            color_list[i, j, :] = colors_to_use[total]
            total = (total + 1) % len(colors_to_use)
    return color_list


def show_boxes(boxes_list, scores_list, labels_list, image_size=800):
    thickness = 5
    color_list = gen_color_list(len(boxes_list), len(np.unique(labels_list)))
    image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    image[...] = 255
    for i in range(len(boxes_list)):
        for j in range(len(boxes_list[i])):
            x1 = int(image_size * boxes_list[i][j][0])
            y1 = int(image_size * boxes_list[i][j][1])
            x2 = int(image_size * boxes_list[i][j][2])
            y2 = int(image_size * boxes_list[i][j][3])
            lbl = labels_list[i][j]
            cv2.rectangle(image, (x1, y1), (x2, y2), color_list[i][lbl], int(thickness * scores_list[i][j]))
    show_image(image)


def show_image(im, name='image'):
    plt.figure(figsize=(20, 20))
    plt.imshow(im)
    plt.show()


def make_labels(id):
    id = id.split('.')[0]
    pbw = 0
    abw = 0

    classifier_pred = classifier_pred_dict[id] * 1.0

    if os.path.exists(
            f'{pred_labels_path}/{id}.txt') and classifier_pred > 0.35:
        with open(
                f'{pred_labels_path}/{id}.txt') as f:
            preds_per_line = f.readlines()
            bboxes = []
            scores = []
            label = []

            for i in preds_per_line:
                i = i.split(' ')
                bbox = [float(i[1]), float(i[2]), float(i[3]), float(i[4])]
                bbox = BoundingBox.from_yolo(*bbox, image_size=(1536, 1536))
                bbox = bbox.to_albumentations().values

                bboxes.append(list(bbox))
                scores.append(float(i[5]))

                label.append(int(i[0]))

            bboxes, scores, label = soft_nms([bboxes], [scores], [label], iou_thr=0.4, sigma=0.8, thresh=0.35,
                                             method='nms', )

            for i in range(len(label)):
                if label[i] == 0:
                    pbw += 1
                else:
                    abw += 1

    return pbw, abw, f"{id}_pbw.jpg", f"{id}_abw.jpg"


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)


# list(map(make_labels, tqdm(train_df['image_id_worm'].values)))
pred = Parallel(n_jobs=4)(delayed(make_labels)(id) for id in tqdm(train_df['image_id_worm'].values))
for i in pred:
    ids.append(i[2])
    ids.append(i[3])
    labels.append(i[0])
    labels.append(i[1])
oof = pd.DataFrame({'image_id_worm': ids, 'label': labels}, index=None)
oof.to_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/oof_df/yolov5m6-2000-image-size-mskf.csv',
    index=False)
pred_label_dict = dict(zip(oof['image_id_worm'].values, oof['label'].values))


def return_error(id):
    id = id.split('.')[0]
    pbw_label = id_label_dict[f'{id}_pbw.jpg']
    abw_label = id_label_dict[f'{id}_abw.jpg']
    pbw_pred = pred_label_dict[f'{id}_pbw.jpg']
    abw_pred = pred_label_dict[f'{id}_abw.jpg']
    error = float(mae(np.array(pbw_label), np.array(pbw_pred)) + float(mae(np.array(abw_label), np.array(abw_pred))))
    return error / 2


error = list(map(return_error, train_df['image_id_worm'].values))
print(mean(error))
