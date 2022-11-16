import pandas as pd
from tqdm.auto import tqdm
from pybboxes import BoundingBox
from shapely.wkt import loads

bboxes = pd.read_csv('/home/mithil/PycharmProjects/PestDetect/data/images_bboxes.csv')
no_bbox = bboxes[bboxes['geometry'].isnull()]

bboxes = bboxes.dropna(axis=0)
images_len = {}
for i in no_bbox['image_id'].unique():
    i = i.split('.')[0]
    images_len.update({f"{i}_pbw.jpg": 0, f"{i}_abw.jpg": 0})
for i in tqdm(bboxes['image_id'].unique()):
    bboxes_temp_pbw = bboxes[(bboxes['image_id'] == i) & (bboxes['worm_type'] == 'pbw')]
    bboxes_temp_abw = bboxes[(bboxes['image_id'] == i) & (bboxes['worm_type'] == 'abw')]
    i = i.split('.')[0]
    images_len.update({f"{i}_pbw.jpg": len(bboxes_temp_pbw), f"{i}_abw.jpg": len(bboxes_temp_abw)})

train_df = pd.DataFrame(images_len.items(), columns=['image_id', 'number_of_worms'])

train_df.to_csv('/home/mithil/PycharmProjects/PestDetect/data/train_modified.csv', index=False)
