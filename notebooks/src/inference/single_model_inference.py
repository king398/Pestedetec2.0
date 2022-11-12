__author__ = 'Mithil Salunkhe: https://www.kaggle.com/mithilsalunkhe'

from statistics import mode
from ensemble_boxes import nms
import numpy as np
import pandas as pd
import os
from ensemble_boxes import *
from tqdm import tqdm

test_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Test.csv')
pred_path = f'/home/mithil/PycharmProjects/Pestedetec2.0/pred_labels/yolov5l6-1536-image-size-25-epoch'
classifier_df = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/tf_effnet_b2_1024_image_size_inference.csv')
classifer_dict = dict(zip(classifier_df['id'].values, classifier_df['label'].values))
ids = []
labels_final = []


def make_labels(id):
    id = id.split('.')[0]

    ids.extend([f"{id}_pbw", f"{id}_abw"])
    pbw_list = []
    abw_list = []
    classifier_pred = classifer_dict[id]

    for i in range(5):

        labels = []
        pbw = float(0)
        abw = float(0)
        path = f'{pred_path}/fold_{i}_test/labels/{id}.txt'

        if os.path.exists(path) and classifier_pred > 0.35:
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

    pbw_list = np.array(pbw_list)
    abw_list = np.array(abw_list)

    pbw = int(np.average(pbw_list))
    abw = int(np.average(abw_list))
    labels_final.extend([pbw, abw])


list(map(make_labels, tqdm(test_df['image_id_worm'].values)))
submission = pd.DataFrame({'image_id_worm': ids, 'label': labels_final}, index=None)
submission.to_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_df/yolov5l6-1536-image-size-25-epoch-with-classifier.csv',
    index=False)
