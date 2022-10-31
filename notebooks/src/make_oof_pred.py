import os
import glob
import shutil

for i in range(5):
    os.system(
        f"python detect.py --weights /home/mithil/PycharmProjects/Pestedetec2.0/models/yolov5s6_image_size_1024/yolov5s6_image_size_1024_fold_{i}/weights/best.pt --half --nosave --img-size 1024 --source dataset/fold_{i}/images/val --name yolov5s6_image_size_1024_fold_{i}_val --save-txt")
preds_txt = glob.glob(
    'runs/detect/yolov5s6_image_size_1024_fold_*_val/labels/*.txt')
os.makedirs('/home/mithil/PycharmProjects/Pestedetec2.0/oof/yolov5s6_image_size_1024_oof/', exist_ok=True)
for i in preds_txt:
    shutil.copy(i, '/home/mithil/PycharmProjects/Pestedetec2.0/oof/yolov5s6_image_size_1024_oof/')
    os.remove(i)
os.system('rm -r runs/detect/')
