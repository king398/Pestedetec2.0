__author__ = 'Mithil Salunkhe: https://www.kaggle.com/mithilsalunkhe'

from statistics import mode, mean
from ensemble_boxes import nms
import numpy as np
import pandas as pd
import os
from ensemble_boxes import *
from tqdm import tqdm
from pybboxes import BoundingBox
import yaml

test_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Test.csv')
pred_path = f'/home/mithil/PycharmProjects/Pestedetec2.0/pred_labels/yolov5m6-1536-image-size-25-epoch-mskf-tta-1700'
pred_path_2 = f'/home/mithil/PycharmProjects/Pestedetec2.0/pred_labels/yolov5m6-1536-image-size-25-epoch-mskf'
classifier_df = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/tf_effnet_b2_1024_image_size_inference.csv')
classifer_dict = dict(zip(classifier_df['id'].values, classifier_df['label'].values))
ids = []
labels_final = []

with open(
'/home/mithil/PycharmProjects/Pestedetec2.0/best_values_optuna/ensemble_yolov5m6-1536-image-size-25-epoch-mskf-tta_yolov5m6-1536-image-size-25-epoch-mskf.yaml') as f:
    params = yaml.safe_load(f)


def make_labels(id):
    pbw_list = []
    abw_list = []
    id = id.split('.')[0]

    ids.extend([f"{id}_pbw", f"{id}_abw"])

    classifier_pred = classifer_dict[id]

    for i in range(5):
        pbw = 0
        abw = 0

        bboxes = []
        labels = []
        scores = []

        labels_temp = []
        bbox_temp = []
        score_temp = []
        path = f'{pred_path}/fold_{i}_test/labels/{id}.txt'

        if os.path.exists(path) and classifier_pred > params['classifier_thresh']:
            with open(path) as f:
                preds_per_line = f.readlines()

                for i in preds_per_line:
                    i = i.split(' ')
                    bbox = [float(i[1]), float(i[2]), float(i[3]), float(i[4])]
                    try:
                        bbox = BoundingBox.from_yolo(*bbox, image_size=(1536, 1536))
                        bbox = bbox.to_albumentations().values

                        bbox_temp.append(bbox)
                        score_temp.append(float(i[5]))
                        labels_temp.append(int(i[0]))
                    except:
                        pass
                bbox_temp, score_temp, labels_temp = soft_nms([bbox_temp], [score_temp], [labels_temp],
                                                              iou_thr=params['iou_thr'],
                                                              sigma=params['sigma'], thresh=params['thresh'],
                                                              method=params['method'])
                bboxes.append(bbox_temp)
                scores.append(score_temp)
                labels.append(labels_temp)

        labels_temp = []
        bbox_temp = []
        score_temp = []
        path = f'{pred_path_2}/fold_{i}_test/labels/{id}.txt'

        if os.path.exists(path) and classifier_pred > params['classifier_thresh_2']:
            with open(path) as f:
                preds_per_line = f.readlines()

                for i in preds_per_line:
                    i = i.split(' ')
                    bbox = [float(i[1]), float(i[2]), float(i[3]), float(i[4])]
                    try:
                        bbox = BoundingBox.from_yolo(*bbox, image_size=(1536, 1536))
                        bbox = bbox.to_albumentations().values

                        bbox_temp.append(bbox)
                        score_temp.append(float(i[5]))
                        labels_temp.append(int(i[0]))
                    except:
                        pass
                bbox_temp, score_temp, labels_temp = soft_nms([bbox_temp], [score_temp], [labels_temp],
                                                              iou_thr=params['iou_thr_2'],
                                                              sigma=params['sigma_2'], thresh=params['thresh_2'],
                                                              method=params['method_2'])
                bboxes.append(bbox_temp)
                scores.append(score_temp)
                labels.append(labels_temp)
        if len(bboxes) > 0:
            if len(bboxes) == 1:
                bboxes, scores, labels = weighted_boxes_fusion(
                    boxes_list=bboxes,
                    scores_list=scores,
                    labels_list=labels,
                    weights=None,
                    iou_thr=params['iou_thr_3'],
                )

            else:
                bboxes, scores, labels = weighted_boxes_fusion(boxes_list=bboxes,
                                                               scores_list=scores,
                                                               labels_list=labels,
                                                               iou_thr=params['iou_thr_3'],
                                                               weights=[params['weights'],
                                                                        1 - params['weights']])
        for i in range(len(labels)):
            if labels[i] == 0:
                pbw += 1
            else:
                abw += 1
        pbw_list.append(pbw)
        abw_list.append(abw)

    pbw = int(mode(pbw_list))
    abw = int(mode(abw_list))

    labels_final.extend([pbw, abw])


list(map(make_labels, tqdm(test_df['image_id_worm'].values)))
submission = pd.DataFrame({'image_id_worm': ids, 'label': labels_final}, index=None)
submission.to_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_df/ensemble_yolov5m6-1536-image-size-25-epoch-mskf-tta_yolov5m6-1536-image-size-25-epoch-mskf.csv',
    index=False)
