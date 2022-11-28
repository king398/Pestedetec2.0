import pandas as pd
from shapely.wkt import loads
from pybboxes import BoundingBox
import cv2
import matplotlib.pyplot as plt
import random
import numpy as np

bboxes = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/images_bboxes.csv')
bboxes = bboxes.dropna(axis=0)
for i in range(10):
    random_row = bboxes.sample(1)
    random_row = random_row.reset_index(drop=True)
    image_id = random_row['image_id'][0]
    label_for_each_image_id = bboxes[bboxes['image_id'] == image_id].reset_index(drop=True)
    label_for_each_image_id = label_for_each_image_id['geometry'].values
    image = cv2.imread(f'/home/mithil/PycharmProjects/Pestedetec2.0/data/images/{image_id}')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_shape = (image.shape[0], image.shape[1])
    background_image = cv2.imread(
        '/home/mithil/PycharmProjects/Pestedetec2.0/data/images/id_1d50b693e0374d9f8fa98ca2.jpg')
    x = 0
    label = random.choices(label_for_each_image_id, k=int(len(label_for_each_image_id) * 0.5))
    for i in label:
        bbox = loads(i)
        bbox = bbox.bounds
        cropped_image = image[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

        if cropped_image.shape[0] > 200 and cropped_image.shape[1] > 100:
            print(cropped_image.shape)
            mask_cropped = np.ones((cropped_image.shape[0], cropped_image.shape[1]), dtype=np.uint8)
            mask_cropped = mask_cropped * 255
            try:
                background_image = cv2.seamlessClone(cropped_image, background_image, mask_cropped,
                                                     (random.randint(0, background_image.shape[0]),
                                                      random.randint(0, background_image.shape[1])), cv2.NORMAL_CLONE)
            except:
                pass
            x += 1
    if x > 0:
        plt.imshow(background_image)
        plt.show()
