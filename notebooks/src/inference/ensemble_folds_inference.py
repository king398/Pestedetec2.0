__author__ = 'Mithil Salunkhe: https://www.kaggle.com/mithilsalunkhe'

from statistics import mode
from ensemble_boxes import nms
import numpy as np
import pandas as pd
import os
from ensemble_boxes import *
from tqdm import tqdm
from pybboxes import BoundingBox

test_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Test.csv')
pred_path = f'/home/mithil/PycharmProjects/Pestedetec2.0/pred_labels/yolov5m6-1536-image-size-25-epoch-mskf-tta-1700'
classifier_df = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/tf_effnet_b2_1024_image_size_inference.csv')
classifer_dict = dict(zip(classifier_df['id'].values, classifier_df['label'].values))
ids = []
labels_final = []


def make_labels(id):
    pbw = 0
    abw = 0
    id = id.split('.')[0]

    ids.extend([f"{id}_pbw", f"{id}_abw"])

    classifier_pred = classifer_dict[id]
    bboxes = []
    labels = []
    scores = []

    for i in range(5):

        labels_temp = []
        bbox_temp = []
        score_temp = []
        path = f'{pred_path}/fold_{i}_test/labels/{id}.txt'

        if os.path.exists(path) and classifier_pred > 0.35:
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
                bbox_temp, score_temp, labels_temp = soft_nms([bbox_temp], [score_temp], [labels_temp], iou_thr=0.3,
                                                              sigma=0.9, thresh=0.4, method='nms', )
                bboxes.append(bbox_temp)
                scores.append(score_temp)
                labels.append(labels_temp)

    pbw = 0
    abw = 0
    if len(bboxes) > 0:
        bboxes, scores, labels = weighted_boxes_fusion(bboxes, scores, labels, weights=None, iou_thr=0.3,
                                                       skip_box_thr=0.4)
        for i in range(len(bboxes)):
            if labels[i] == 0:
                pbw += 1
            else:
                abw += 1
    labels_final.extend([pbw, abw])


list(map(make_labels, tqdm(test_df['image_id_worm'].values)))
submission = pd.DataFrame({'image_id_worm': ids, 'label': labels_final}, index=None)
submission.to_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_df/yolov5m6-1536-image-size-25-epoch-mskf-nms-1536-tta.csv',
    index=False)
