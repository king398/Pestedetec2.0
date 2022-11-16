import glob
import os

path = f"/home/mithil/PycharmProjects/Pestedetec2.0/notebooks/yolov5/dataset"
for i in range(5):
    removed = 0

    dir = f"{path}/fold_{i}/"
    images = glob.glob(f"{dir}/images/train/*")
    labels = glob.glob(f"{dir}/labels/train/*")
    for i in images:
        id = i.split("/")[-1].split(".")[0]
        if not os.path.exists(f"{dir}/labels/train/{id}.txt") and removed < 1200:
            os.remove(i)
            removed += 1
