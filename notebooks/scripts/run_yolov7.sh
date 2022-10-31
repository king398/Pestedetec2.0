cd yolov7
python train.py  --device 0 --epochs 25 --batch-size 48 --data fold_1.yaml --img 1024 1024 --cfg cfg/training/yolov7x.yaml --weights 'yolov7x_training.pt' \
      --name yolov7x-custom-different-augs-part-fold-1-image-size-1024 --hyp data/hyp.scratch.custom_yolov7x.yaml
python train.py  --device 0 --epochs 25 --batch-size 48 --data fold_2.yaml --img 1024 1024 --cfg cfg/training/yolov7x.yaml --weights 'yolov7x_training.pt' \
      --name yolov7x-custom-different-augs-part-fold-2-image-size-1024 --hyp data/hyp.scratch.custom_yolov7x.yaml
