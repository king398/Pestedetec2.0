import os
import glob
import shutil

save_path = "/home/mithil/PycharmProjects/Pestedetec2.0/oof_raw_preds/yolov5s6-2200-image-size-25-epoch "
for i in range(5):
    os.system(
        f"python detect.py --weights /home/mithil/PycharmProjects/Pestedetec2.0/models/yolov5s6-2200-image-size-25-epoch/yolov5s6-2200-image-size-fold-{i}-25-epoch/weights/best.pt "

        f"--img-size 2200 --half "
        f"--source dataset/fold_{i}/images/val "
        f"--name yolov7x-custom-different-augs-image-size-1024-{i}_val --save-txt "
        f"--conf 0.1 "
        f"--save-conf  ")
preds_txt = glob.glob(
    'runs/detect/yolov7x-custom-different-augs-image-size-1024-*_val/labels/*.txt')
preds_image = glob.glob(f"runs/detect/yolov7x-custom-different-augs-image-size-1024-*_val/*.jpg") + glob.glob(
    'runs/detect/yolov7x-custom-different-augs-image-size-1024-*_val/*.jpeg')
os.makedirs(save_path,
            exist_ok=True)
os.makedirs(f"{save_path}/images", exist_ok=True)
for i in preds_txt:
    shutil.copy(i,
                save_path)
    os.remove(i)
for i in preds_image:
    shutil.copy(i,
                f"{save_path}/images")
    os.remove(i)
os.system('rm -r runs/detect/')
