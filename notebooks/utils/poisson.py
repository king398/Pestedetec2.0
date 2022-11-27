import pandas as pd
from shapely.wkt import loads
from pybboxes import BoundingBox
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

bboxes = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/images_bboxes.csv')
bboxes = bboxes.dropna(axis=0)
random_row = bboxes.sample(1)
random_row = random_row.reset_index(drop=True)
image_id = random_row['image_id'][0]
image = cv2.imread(f'/home/mithil/PycharmProjects/Pestedetec2.0/data/train_images/{image_id}')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
labels = open(f'/home/mithil/PycharmProjects/Pestedetec2.0/data/labels/{image_id.split(".")[0]}.txt', 'r')
background_image = cv2.imread(
    '/home/mithil/PycharmProjects/Pestedetec2.0/data/train_images/id_0a874b32c61041f70b309f6e.jpg')
labels = labels.readlines()
for i in range(10):
    label = random.choice(labels).split(' ')
    bbox = label[1:5]
    bbox = [float(i) for i in bbox]
    bbox = BoundingBox.from_yolo(*bbox, image_size=(1024, 1024))
    bbox = bbox.to_voc().values
    cropped_image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

    if cropped_image.shape[0] > 50 and cropped_image.shape[1] > 50:
        print(cropped_image.shape)
        mask_cropped = np.ones((cropped_image.shape[0], cropped_image.shape[1]), dtype=np.uint8)
        mask_cropped = mask_cropped * 255
        try:
            background_image = cv2.seamlessClone(cropped_image, background_image, mask_cropped,
                                                 (random.randint(0, 1024), random.randint(0, 1024)), cv2.NORMAL_CLONE)
        except:
            pass
plt.imshow(background_image)
plt.show()
