__author__ = 'Mithil Salunkhe: https://www.kaggle.com/mithilsalunkhe'

from statistics import mode

import numpy as np
import pandas as pd
import os
from ensemble_boxes import *
from tqdm import tqdm

test_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Test.csv')
pred_path = f'/home/mithil/PycharmProjects/Pestedetec2.0/pred_labels/yolov5m6-1536-image-size'
classifier_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/inference_classifier.csv')
classifer_dict = dict(zip(classifier_df['id'].values, classifier_df['label'].values))
ids = []
labels_final = []

weights = [1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 0.5]


def make_labels(id):
    id = id.split('.')[0]

    ids.extend([f"{id}_pbw", f"{id}_abw"])
    pbw_list = []
    abw_list = []
    classifier_pred = classifer_dict[id]

    for i in range(5):

        labels = []
        pbw = 0
        abw = 0
        path = f'{pred_path}/yolov5m6-1536-image-size_{i}_test/labels/{id}.txt'

        if os.path.exists(path) and classifier_pred > 0.1:
            with open(path) as f:
                preds_per_line = f.readlines()

                for i in preds_per_line:
                    i = i.split(' ')

                    labels.append(int(i[0]))
        for i in range(len(labels)):
            if labels[i] == 0:
                pbw += 1
            else:
                abw += 1
        pbw_list.append(pbw)
        abw_list.append(abw)


    pbw = int((np.average(pbw_list)))
    abw = int((np.average(abw_list)))
    labels_final.extend([pbw, abw])


list(map(make_labels, tqdm(test_df['image_id_worm'].values)))
submission = pd.DataFrame({'image_id_worm': ids, 'label': labels_final}, index=None)
submission.to_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_df/yolov5m6-1536-image-size-with_classifier-ensemble.csv',
    index=False)
