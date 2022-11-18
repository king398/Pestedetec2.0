import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import matplotlib.pyplot as plt

train_modified = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/train_modified.csv')
train = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
ids = []
pbw = []
abw = []


def append_labels(id):
    ids.append(id)
    id = id.split('.')[0]
    pbw.append(train_modified[train_modified['image_id'] == f"{id}_pbw.jpg"]['number_of_worms'].values[0])
    abw.append(train_modified[train_modified['image_id'] == f"{id}_abw.jpg"]['number_of_worms'].values[0])


list(map(append_labels, train['image_id_worm'].values))
train_kf = pd.DataFrame({'image_id': ids, 'pbw': pbw, 'abw': abw})


def make_fold():
    mskf = MultilabelStratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    train_kf['fold'] = -1
    num_bins = int(np.floor(1 + np.log2(len(train_kf))))
    train_kf.loc[:, "bins_1"] = pd.cut(train_kf['pbw'], bins=num_bins, labels=False)
    train_kf.loc[:, "bins_2"] = pd.cut(train_kf['abw'], bins=num_bins, labels=False)
    for fold, (train_idx, val_idx) in enumerate(mskf.split(X=train_kf, y=train_kf[['bins_1', 'bins_2']].values)):
        train_kf.loc[val_idx, 'fold'] = fold

    train_kf.drop('bins_1', axis=1, inplace=True)
    train_kf.drop('bins_2', axis=1, inplace=True)
    labels_len_dict = {}
    for i in range(5):
        labels_len_dict.update({f'fold_{i}': train_kf[train_kf['fold'] != i]['pbw'].sum()})
        labels_len_dict.update({f'abw_{i}': train_kf[train_kf['fold'] != i]['abw'].sum()})
    plt.bar(labels_len_dict.keys(), labels_len_dict.values())
    plt.show()
    return train_kf


x = make_fold()
x.to_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_mskf.csv', index=False)
