import pandas as pd
import os

test_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Test.csv')
ids = []
labels = []
pred_labels_path = '/home/mithil/PycharmProjects/Pestedetec2.0/pred_labels/yolov7x-custom-different-augs-part--fold-0-1536-image-size'


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


list(map(make_labels, test_df['image_id_worm'].values))
submission = pd.DataFrame({'image_id_worm': ids, 'label': labels}, index=None)
submission.to_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/pred_df/yolov7x-custom-different-augs-part--fold-0-1536-image-size.csv',
    index=False)
