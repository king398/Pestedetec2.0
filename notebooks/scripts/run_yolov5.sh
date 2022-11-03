cd yolov5
python train.py --device 0 --epochs 25 --batch-size 38 --data fold_0.yaml --img 1536 --weights 'yolov5m6.pt' --name yolov5m6-1536-image-size-fold-0
python train.py --device 0 --epochs 25 --batch-size 38 --data fold_1.yaml --img 1536 --weights 'yolov5m6.pt' --name yolov5m6-1536-image-size-fold-1
python train.py --device 0 --epochs 25 --batch-size 38 --data fold_2.yaml --img 1536 --weights 'yolov5m6.pt' --name yolov5m6-1536-image-size-fold-2
