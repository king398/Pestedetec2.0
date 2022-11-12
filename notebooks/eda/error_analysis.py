import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from shapely.wkt import loads
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import random

fig = plt.figure(figsize=(20, 20))
df_bbox = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/images_bboxes.csv')
train_df = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/Train.csv')
train_labels_df = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/train_modified.csv')
preds = pd.read_csv(
    '/home/mithil/PycharmProjects/Pestedetec2.0/oof_df/yolov5m6-1536-image-size-classifier-tf-tf_effnet_b2_1024_image_size-pred.csv')
ids = []
labels = []
id_label_dict = dict(zip(train_labels_df['image_id'].values, train_labels_df['number_of_worms'].values))
pred_label_dict = dict(zip(preds['image_id_worm'].values, preds['label'].values))


def mae(y_true, y_pred):
    return np.abs(y_true - y_pred)


mis_pred_id = []
errors = []
id_error_dict = {}
for i in train_df['image_id_worm'].values:
    id = i.split('.')[0]
    ids.extend([f"{id}_pbw.jpg", f"{id}_abw.jpg"])
    pbw_label = id_label_dict[f'{id}_pbw.jpg']
    abw_label = id_label_dict[f'{id}_abw.jpg']
    pbw_pred = pred_label_dict[f'{id}_pbw.jpg']
    abw_pred = pred_label_dict[f'{id}_abw.jpg']
    labels.extend([pbw_label, abw_label])
    error = float(mae(np.array(pbw_label), np.array(pbw_pred)) + float(mae(np.array(abw_label), np.array(abw_pred))))
    if error > 30:
        mis_pred_id.append(id)
        id_error_dict.update({id: error})
    errors.append(error)
id = random.choice(mis_pred_id)
label_bbox = df_bbox[df_bbox['image_id'] == f"{id}.jpg"]
img = cv2.imread(
    f"/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5m6-1536-image-size/images/{id}.jpg")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
fig = plt.figure(figsize=(20, 20))
# add title to fig
fig.suptitle(f"{id} Predicted", fontsize=50)
plt.imshow(img)
plt.show()
print(f"Error: {id_error_dict[id]}")
print(f"Actual: {id_label_dict[f'{id}_pbw.jpg']} {id_label_dict[f'{id}_abw.jpg']}")
print(f"Predicted: {pred_label_dict[f'{id}_pbw.jpg']} {pred_label_dict[f'{id}_abw.jpg']}")
img = cv2.imread(
    f'/home/mithil/PycharmProjects/Pestedetec2.0/notebooks/yolov5/dataset/fold_0/images/train/{id}.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if img is None:
    img = cv2.imread(
        f'/home/mithil/PycharmProjects/Pestedetec2.0/notebooks/yolov5/dataset/fold_0/images/val/{id}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
for i in label_bbox.index:
    bbox = label_bbox.loc[i, 'geometry']
    bbox = loads(str(bbox))
    bbox = bbox.bounds
    img = cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
fig = plt.figure(figsize=(20, 20))
# add title to fig
fig.suptitle(f"{id} Label", fontsize=50)
plt.imshow(img)
plt.show()
