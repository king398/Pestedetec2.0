import pandas as pd

train_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
train_labels_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_modified.csv')
preds = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/oof_df/yolov5m6-1536-image-size-30-epoch.csv')
ids = []
labels = []
pred_labels_path = '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5m6-1536-image-size-30-epoch'
id_label_dict = dict(zip(train_labels_df['image_id'].values, train_labels_df['number_of_worms'].values))
pred_label_dict = dict(zip(preds['image_id_worm'].values, preds['label'].values))
for i in train_df:
    df = train_df.iloc[i]
    print(df)
    break