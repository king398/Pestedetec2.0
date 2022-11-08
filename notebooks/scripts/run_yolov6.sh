cd YOLOv6
python tools/train.py --batch 18 --conf configs/yolov6m_finetune.py --data fold_0.yaml --device 0 --img 1536 --name yolov6m-1536-image-size-fold-0 --epochs 25
python tools/train.py --batch 18 --conf configs/yolov6m_finetune.py --data fold_0.yaml --device 0 --img 1536 --name yolov6m-1536-image-size-fold-1 --epochs 25
 mv YOLOv6/fold_0.yaml YOLOv6/fold_1.yaml YOLOv6/fold_2.yaml YOLOv6/fold_3.yaml YOLOv6/fold_4.yaml yolov5