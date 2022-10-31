import os

for i in range(5):
    os.system(f"python detect.py --half "
              f"--nosave "
              f"--weights /home/mithil/PycharmProjects/Pestedetec2.0/models/yolov5s6_image_size_1024/yolov5s6_image_size_1024_fold_{i}/weights/best.pt"
              f" --img-size 1536"
              f" --source test_images"
              f" --name yolov5s6_image_size_1536_upscale_fold_{i}_test"
              f" --save-txt"
              f" --conf 0.3"
              f" --save-conf"
              f"")
