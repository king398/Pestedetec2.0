import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

test_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/pred_df/yolov5l6-1536-image-size-25-epoch.csv')
test_df_ids = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/Test.csv')
ids = []
pbw = []
abw = []


def append_labels(id):
    ids.append(id)
    id = id.split('.')[0]
    pbw.append(test_df[test_df['image_id_worm'] == f"{id}_pbw"]['label'].values[0])
    abw.append(test_df[test_df['image_id_worm'] == f"{id}_abw"]['label'].values[0])


list(map(append_labels, test_df_ids['image_id_worm'].values))
test_kf_pseudo = pd.DataFrame({'image_id': ids, 'pbw': pbw, 'abw': abw})


def make_fold():
    multilabelstratifiedkfold = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    test_kf_pseudo['fold'] = -1
    num_bins = int(np.floor(1 + np.log2(len(test_kf_pseudo))))
    test_kf_pseudo.loc[:, "bins_1"] = pd.cut(test_kf_pseudo['pbw'], bins=num_bins, labels=False)
    test_kf_pseudo.loc[:, "bins_2"] = pd.cut(test_kf_pseudo['abw'], bins=num_bins, labels=False)
    for fold, (train_idx, val_idx) in enumerate(
            multilabelstratifiedkfold.split(X=test_kf_pseudo, y=test_kf_pseudo[['bins_1', 'bins_2']].values)):
        test_kf_pseudo.loc[val_idx, 'fold'] = fold

    test_kf_pseudo.drop('bins_1', axis=1, inplace=True)
    test_kf_pseudo.drop('bins_2', axis=1, inplace=True)
    labels_len_dict = {}
    for i in range(5):
        labels_len_dict.update({f'pbw_{i}': test_kf_pseudo[test_kf_pseudo['fold'] != i]['pbw'].sum()})
        labels_len_dict.update({f'abw_{i}': test_kf_pseudo[test_kf_pseudo['fold'] != i]['abw'].sum()})
    plt.bar(labels_len_dict.keys(), labels_len_dict.values())
    plt.show()
    return test_kf_pseudo


x = make_fold()
x.to_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/yolov5l6-1536-image-size-25-epoch-pseudo.csv', index=False)
