__author__ = 'Mithil Salunkhe: https://www.kaggle.com/mithilsalunkhe'

import pandas as pd
import os
from ensemble_boxes import *
from tqdm import tqdm

test_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Test.csv')
pred_path = f'/home/mithil/PycharmProjects/Pestedetec2.0/pred_labels/yolov5s_5_fold_1536_image_size'
ids = []
labels_final = []

weights = [1, 1, 1, 1, 1]
iou_thr = 0.99
skip_box_thr = 0.001


def make_labels(id):
    id = id.split('.')[0]
    ids.extend([f"{id}_pbw", f"{id}_abw"])
    pbw = 0
    abw = 0
    bboxes_list = []
    scores_list = []
    labels_list = []
    for i in range(5):
        bboxes = []
        scores = []
        labels = []

        if os.path.exists(f'{pred_path}/yolov5s6_image_size_1536_upscale_fold_0_test/labels/{id}.txt'):
            with open(f'{pred_path}/yolov5s6_image_size_1536_upscale_fold_0_test/labels/{id}.txt') as f:
                preds_per_line = f.readlines()

                for i in preds_per_line:
                    i = i.split(' ')
                    bboxes.append([float(i[3]), float(i[4]), float(i[1]), float(i[2])])
                    scores.append(float(i[5]))
                    labels.append(int(i[0]))
        bboxes_list.append(bboxes)
        scores_list.append(scores)
        labels_list.append(labels)
    boxes, scores, labels = weighted_boxes_fusion(bboxes_list, scores_list, labels_list, weights=weights,
                                                  iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    for i in range(len(labels)):
        if labels[i] == 0:
            pbw += 1
        else:
            abw += 1
    labels_final.extend([pbw, abw])


list(map(make_labels, tqdm(test_df['image_id_worm'].values)))
submission = pd.DataFrame({'image_id_worm': ids, 'label': labels_final}, index=None)
submission.to_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_df/wbf_try.csv',
    index=False)
