import pandas as pd
import numpy as np

train_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
train_labels_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_modified.csv')
preds = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/oof_df/yolov5m6-1536-image-size-30-epoch.csv')
ids = []
labels = []
pred_labels_path = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5m6-1536-image-size-30-epoch'
id_label_dict = dict(zip(train_labels_df['image_id'].values, train_labels_df['number_of_worms'].values))
pred_label_dict = dict(zip(preds['image_id_worm'].values, preds['label'].values))


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)

mis_pred = 0
correct_pred = 0
errors = []

for i in train_df.index:
    df = train_df.loc[i]
    id = df.image_id_worm.split('.')[0]
    pbw_label = id_label_dict[f'{id}_pbw.jpg']
    abw_label = id_label_dict[f'{id}_abw.jpg']
    if df.number_of_worms == 0:
        if pred_label_dict[f'{id}_pbw.jpg'] != 0 or pred_label_dict[f'{id}_abw.jpg'] != 0:
            mis_pred += 1
            errors.append(float(mae(np.array(pbw_label), np.array(pred_label_dict[f'{id}_pbw.jpg'])) + float(
                mae(np.array(abw_label), np.array(pred_label_dict[f'{id}_abw.jpg'])))))
        else:
            correct_pred += 1

        pred_label_dict[f'{id}_pbw.jpg'] = 0
        pred_label_dict[f'{id}_abw.jpg'] = 0

    else:
        errors.append(float(mae(np.array(pbw_label), np.array(pred_label_dict[f'{id}_pbw.jpg'])) + float(
            mae(np.array(abw_label), np.array(pred_label_dict[f'{id}_abw.jpg'])))))
print(np.mean(errors))