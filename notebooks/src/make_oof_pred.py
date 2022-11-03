import os
import glob
import shutil

for i in range(5):
    os.system(
        f"python detect.py --weights /home/mithil/PycharmProjects/Pestedetec2.0/models/yolov7x-custom-different-augs-image-size-1024/yolov7x-custom-different-augs-part-fold-{i}-image-size-1024/best.pt "

        f"--img-size 1024 "
        f"--source dataset/fold_{i}/images/val "
        f"--name yolov7x-custom-different-augs-image-size-1024-{i}_val --save-txt "
        f"--conf 0.35")
preds_txt = glob.glob(
    'runs/detect/yolov7x-custom-different-augs-image-size-1024-*_val/labels/*.txt')
os.makedirs('/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov7x-custom-different-augs-image-size-1024/',
            exist_ok=True)
for i in preds_txt:
    shutil.copy(i,
                '/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov7x-custom-different-augs-image-size-1024/')
    os.remove(i)
os.system('rm -r runs/detect/')
