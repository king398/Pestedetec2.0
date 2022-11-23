import optuna
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
import yaml

train_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
train_labels_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_modified.csv')
ids = []
labels = []
pred_labels_path = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5m6-1536-image-size-25-epoch-mskf-tta'
pred_labels_path_2 = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/mskf/yolov5m6-2000-image-size-mskf'

id_label_dict = dict(zip(train_labels_df['image_id'].values, train_labels_df['number_of_worms'].values))

classifier_pred = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_classfier_oof/tf_effnet_b2_1024_image_size.csv')
classifier_pred_dict = dict(zip(classifier_pred['id'].values, classifier_pred['pred'].values))


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)


def return_error(id, pred_label_dict):
    id = id.split('.')[0]
    pbw_label = id_label_dict[f'{id}_pbw.jpg']
    abw_label = id_label_dict[f'{id}_abw.jpg']
    pbw_pred = pred_label_dict[f'{id}_pbw.jpg']
    abw_pred = pred_label_dict[f'{id}_abw.jpg']
    error = float(
        mae(np.array(pbw_label), np.array(pbw_pred)) + float(mae(np.array(abw_label), np.array(abw_pred))))
    return error


def make_labels(id, params):
    id = id.split('.')[0]
    pbw_1 = 0
    abw_1 = 0

    classifier_pred = classifier_pred_dict[id] * 1.0
    bboxes_ensemble = []
    labels_ensemble = []
    scores_ensemble = []

    if os.path.exists(
            f'{pred_labels_path}/{id}.txt') and classifier_pred > params['classifier_thresh']:
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

            # bboxes, scores, label = soft_nms([bboxes], [scores], [label], iou_thr=params['iou_thr'],
            # sigma=params['sigma'], thresh=params['thresh'], method=params['method'])
            bboxes_ensemble.append(bboxes)
            scores_ensemble.append(scores)
            labels_ensemble.append(label)
            for i in range(len(label)):
                if label[i] == 0:
                    pbw_1 += 1
                else:
                    abw_1 += 1
    pbw_2 = 0
    abw_2 = 0
    if os.path.exists(
            f'{pred_labels_path_2}/{id}.txt') and classifier_pred > params['classifier_thresh_2']:
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

            # bboxes, scores, label = soft_nms([bboxes], [scores], [label], iou_thr=params['iou_thr_2'],
            # sigma=params['sigma_2'], thresh=params['thresh_2'],
            # method=params['method_2'])
            bboxes_ensemble.append(bboxes)
            scores_ensemble.append(scores)
            labels_ensemble.append(label)
            for i in range(len(label)):
                if label[i] == 0:
                    pbw_2 += 1
                else:
                    abw_2 += 1
    bboxes, scores, label = soft_nms(bboxes_ensemble, scores_ensemble, labels_ensemble,
                                                  iou_thr=params['iou_thr'], sigma=params['sigma'],thresh=params['thresh'], )
    pbw = int(pbw_1 * params['weights'] + pbw_2 * (1 - params['weights']))
    abw = int(abw_1 * params['weights'] + abw_2 * (1 - params['weights']))
    return pbw, abw, f"{id}_pbw.jpg", f"{id}_abw.jpg"


class error_func:
    def __init__(self, pred_label_dict):
        super().__init__()
        self.pred_label_dict = pred_label_dict

    def return_error(self, id):
        id = id.split('.')[0]
        pbw_label = id_label_dict[f'{id}_pbw.jpg']
        abw_label = id_label_dict[f'{id}_abw.jpg']
        pbw_pred = self.pred_label_dict[f'{id}_pbw.jpg']
        abw_pred = self.pred_label_dict[f'{id}_abw.jpg']
        error = float(
            mae(np.array(pbw_label), np.array(pbw_pred)) + float(mae(np.array(abw_label), np.array(abw_pred))))
        return error


def objective(trial):
    params = {
        'iou_thr': trial.suggest_float('iou_thr', 0.1, 0.7),
        'sigma': trial.suggest_float('sigma', 0.3, 1.0),
        'thresh': trial.suggest_float('thresh', 0.2, 0.6),
        'method': trial.suggest_categorical('method', ['nms', 'linear', 'gaussian']),
        'classifier_thresh': trial.suggest_float('classifier_thresh', 0.1, 0.7),
        'iou_thr_2': trial.suggest_float('iou_thr_2', 0.1, 0.7),
        'sigma_2': trial.suggest_float('sigma_2', 0.3, 1.0),
        'thresh_2': trial.suggest_float('thresh_2', 0.2, 0.6),
        'method_2': trial.suggest_categorical('method_2', ['nms', 'linear', 'gaussian']),
        'classifier_thresh_2': trial.suggest_float('classifier_thresh_2', 0.1, 0.7),
        'weights': trial.suggest_float('weights', 0.1, 0.8),

    }

    pred = Parallel(n_jobs=6)(delayed(make_labels)(id, params) for id in train_df['image_id_worm'].values)
    ids = []
    labels = []
    for i in pred:
        ids.append(i[2])
        ids.append(i[3])
        labels.append(i[0])
        labels.append(i[1])
    oof = pd.DataFrame({'image_id_worm': ids, 'label': labels}, index=None)

    pred_label_dict = dict(zip(oof['image_id_worm'].values, oof['label'].values))
    error_fn = error_func(pred_label_dict)

    error = list(map(error_fn.return_error, train_df['image_id_worm'].values))

    return mean(error)


study = optuna.create_study(direction='minimize', study_name='yolov5m6-1536-image-size-25-epoch-mskf-tta')
study.optimize(objective, n_trials=2000)
best_param_save = study.best_params
best_param_save.update({'best_score': study.best_value})
best_param_save.update({'best_trial': study.best_trial.number})
best_param_save.update({'path': pred_labels_path})
with open(
        f'/home/mithil/PycharmProjects/Pestedetec2.0/best_values_optuna/ensemble_{pred_labels_path.split("/")[-1]}_{pred_labels_path_2.split("/")[-1]}_average.yaml',
        'w') as f:
    yaml.dump(best_param_save, f)
