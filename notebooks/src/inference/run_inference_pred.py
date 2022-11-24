import glob
import os
import shutil

save_dir = '/home/mithil/PycharmProjects/Pestedetec2.0/pred_labels/yolov5l6-1536-image-size-25-epoch-mskf'
os.makedirs(save_dir, exist_ok=True)
for i in range(5):
    os.system(f"python detect.py --half "
              f"--nosave "
              f"--weights  /home/mithil/PycharmProjects/Pestedetec2.0/models/mskf/yolov5l6-1536-image-size-25-epoch-mskf/yolov5l6-1536-image-size-fold-{i}-25-epoch-mskf/weights/best.pt "
              f" --img-size 1536"
              f" --source test_images "
              f" --name fold_{i}_test"
              f" --save-txt"
              f" --conf 0.01"
              f" --save-conf"
              f" --augment "
              f"")

dirs = os.listdir('runs/detect/')
for i in dirs:
    shutil.move(f'runs/detect/{i}/', f'{save_dir}')

os.system('rm -r runs/detect/')
