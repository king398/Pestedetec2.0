import os

for i in range(5):
    os.system(f"python detect.py --half "
              f"--nosave "
              f"--weights  /home/mithil/PycharmProjects/Pestedetec2.0/models/yolov5m6-1536-image-size-30-epoch/yolov5m6-1536-image-size-fold-{i}-30-epoch/weights/last.pt "
              f" --img-size 1536"
              f" --source test_images"
              f" --name fold_{i}_test"
              f" --save-txt"
              f" --conf 0.35"
              f" --save-conf "
              f"")
