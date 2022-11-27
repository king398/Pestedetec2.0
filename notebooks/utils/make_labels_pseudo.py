import random
from statistics import mode
from ensemble_boxes import nms
import numpy as np
import pandas as pd
import os
from ensemble_boxes import *
from tqdm import tqdm
from pybboxes import BoundingBox
import yaml
import cv2
import matplotlib.pyplot as plt

test_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Test.csv')
pred_path = f'/home/mithil/PycharmProjects/Pestedetec2.0/pred_labels/yolov5l6-1536-image-size-25-epoch-mskf'
classifier_df = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/tf_effnet_b2_1024_image_size_inference.csv')
classifer_dict = dict(zip(classifier_df['id'].values, classifier_df['label'].values))
labels_final = []
with open(
        '/home/mithil/PycharmProjects/Pestedetec2.0/best_values_optuna/yolov5l6-1536-image-size-25-epoch-mskf.yaml') as f:
    params = yaml.safe_load(f)
write_dir = f'/home/mithil/PycharmProjects/Pestedetec2.0/data/pseudo_labels/yolov5l6-1536-image-size-25-epoch-mskf'
os.makedirs(write_dir, exist_ok=True)


def write_label(id, params=params):
    id = id.split('.')[0]

    classifier_pred = classifer_dict[id]

    bboxes = []
    labels = []
    scores = []
    for i in range(5):

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

    if len(bboxes) > 0:
        bboxes, labels, scores = weighted_boxes_fusion(bboxes, labels, scores, iou_thr=params['iou_thr'], )
        if len(bboxes) > 0:
            labels_file = open(f'{write_dir}/{id}.txt', 'w')
        for i in range(len(bboxes)):
            try:
                bbox = BoundingBox.from_albumentations(*bboxes[i], image_size=(1536, 1536))
                bbox = bbox.to_yolo().values
                labels_file.write(f'\n {int(labels[i])} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}')
            except:
                pass

        if len(bboxes) > 0:
            labels_file.close()


list(map(write_label, tqdm(test_df['image_id_worm'].values)))
