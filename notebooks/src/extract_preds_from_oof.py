import pandas as pd
import os
train_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
ids = []
labels = []
pred_labels_path = '/home/mithil/PycharmProjects/PestDetect/pred_labels/yolo7x_different_aug_1600_image_size'


def make_labels(id):
    id = id.split('.')[0]
    ids.extend([f"{id}_pbw", f"{id}_abw"])
    pbw = 0
    abw = 0
    if os.path.exists(
            f'{pred_labels_path}/{id}.txt'):
        with open(
                f'{pred_labels_path}/{id}.txt') as f:
            preds_per_line = f.readlines()
            for i in preds_per_line:
                if i.split(' ')[0] == '0':
                    pbw += 1
                else:
                    abw += 1
    labels.extend([pbw, abw])


list(map(make_labels, train_df['image_id_worm'].values))
oof = pd.DataFrame({'image_id_worm': ids, 'label': labels}, index=None)
oof.to_csv(
    '/home/mithil/PycharmProjects/PestDetect/submission/yolo7x_different_aug_1600_image_size.csv',
    index=False)
