import pandas as pd
import os
import numpy as np
from statistics import mean, mode
import random
from ensemble_boxes import soft_nms, weighted_boxes_fusion
from pybboxes import BoundingBox

train_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
train_labels_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_modified.csv')
ids = []
labels = []
pred_labels_path = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5m6-1536-image-size-25-epoch-mskf'
pred_labels_path_2 = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5m6-1536-image-size-25-epoch-mskf-tta'
id_label_dict = dict(zip(train_labels_df['image_id'].values, train_labels_df['number_of_worms'].values))

classifier_pred = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/tf_effnet_b2_1024_image_size.csv')
classifier_pred_dict = dict(zip(classifier_pred['id'].values, classifier_pred['pred'].values))


def make_labels(id):
    id = id.split('.')[0]
    ids.extend([f"{id}_pbw.jpg", f"{id}_abw.jpg"])
    pbw_1 = 0
    abw_1 = 0
    bbox_final = []
    score = []
    label_final = []

    classifier_pred = classifier_pred_dict[id] * 1.0

    if os.path.exists(
            f'{pred_labels_path}/{id}.txt') and classifier_pred > 0.35:
        with open(
                f'{pred_labels_path}/{id}.txt') as f:
            preds_per_line = f.readlines()
            bboxes = []
            scores = []
            label_list = []

            for i in preds_per_line:
                i = i.split(' ')
                bbox = [float(i[1]), float(i[2]), float(i[3]), float(i[4])]
                bbox = BoundingBox.from_yolo(*bbox, image_size=(1536, 1536))
                bbox = bbox.to_albumentations().values

                bboxes.append(list(bbox))
                scores.append(float(i[5]))

                label_list.append(int(i[0]))
            bboxes, scores, label_list = soft_nms([bboxes], [scores], [label_list], iou_thr=0.3, sigma=0.9, thresh=0.35,
                                                  method='nms', )
            bbox_final.append(bboxes)
            score.append(scores)
            label_final.append(label_list)
            for i in range(len(bboxes)):
                if label_list[i] == 0:
                    pbw_1 += 1
                else:
                    abw_1 += 1
    pbw_2 = 0
    abw_2 = 0

    if os.path.exists(f'{pred_labels_path_2}/{id}.txt') and classifier_pred > 0.35:
        with open(
                f'{pred_labels_path_2}/{id}.txt') as f:
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
            bboxes, scores, label_list = soft_nms([bboxes], [scores], [label], iou_thr=0.3, sigma=0.9, thresh=0.4,
                                                  method='nms', )
            for i in range(len(bboxes)):
                if label[i] == 0:
                    pbw_2 += 1
                else:
                    abw_2 += 1
            bbox_final.append(bboxes)
            score.append(scores)
            label_final.append(label_list)
    pbw = 0
    abw = 0
    if len(bbox_final) > 0:
        bbox_final, score, label_final = weighted_boxes_fusion(bbox_final, score, label_final, iou_thr=0.3, skip_box_thr=0.35)

        for i in range(len(bbox_final)):
            if label_final[i] == 0:
                pbw += 1
            else:
                abw += 1
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
    error = float(
        mae(np.array(pbw_label), np.array(pbw_pred)) + float(mae(np.array(abw_label), np.array(abw_pred))))
    return error


error = list(map(return_error, train_df['image_id_worm'].values))
print(mean(error))
error_random_picked = []
for i in range(69420):
    error_random_picked.append(np.mean(random.choices(error, k=3000)))
print(min(error_random_picked))
