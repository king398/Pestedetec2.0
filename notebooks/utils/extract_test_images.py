import pandas as pd
import os

Test = pd.read_csv('/home/mithil/PycharmProjects/Pestedetec2.0/data/Test.csv')
for i in range(len(Test)):
    image_id = Test['image_id_worm'][i]
    os.system(
        f'cp /home/mithil/PycharmProjects/Pestedetec2.0/data/images/{image_id} /home/mithil/PycharmProjects/Pestedetec2.0/data/test_images')
    os.system(f'rm /home/mithil/PycharmProjects/Pestedetec2.0/data/images/{image_id}')
