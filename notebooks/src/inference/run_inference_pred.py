import os

for i in range(5):
    os.system(f"python detect.py --half "
              f"--nosave "
              f"--weights  /home/mithil/PycharmProjects/Pestedetec2.0/models/mskf/yolov5m6-2000-image-size-mskf/yolov5m6-2000-image-size-fold-{i}-25-epoch-mskf/weights/best.pt "
              f" --img-size 2000"
              f" --source test_images "
              f" --name fold_{i}_test"
              f" --save-txt"
              f" --conf 0.01"
              f" --save-conf"
              f" --augment "
              f"")
