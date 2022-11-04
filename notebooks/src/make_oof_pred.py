import os
import glob
import shutil

save_path = "/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5m6-1536-image-size-higher-confidence"
for i in range(5):
    os.system(
        f"python detect.py --weights /home/mithil/PycharmProjects/Pestedetec2.0/models/yolov5m6-1536-image-size/yolov5m6-1536-image-size-fold-{i}/weights/best.pt "

        f"--img-size 1536 --half "
        f"--source dataset/fold_{i}/images/val "
        f"--name yolov7x-custom-different-augs-image-size-1024-{i}_val --save-txt "
        f"--conf 0.35   ")
preds_txt = glob.glob(
    'runs/detect/yolov7x-custom-different-augs-image-size-1024-*_val/labels/*.txt')
os.makedirs(save_path,
            exist_ok=True)
for i in preds_txt:
    shutil.copy(i,
                save_path)
    os.remove(i)
os.system('rm -r runs/detect/')
