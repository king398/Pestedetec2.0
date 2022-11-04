cd yolov5
python train.py --device 0 --epochs 25 --batch-size 18 --data fold_0.yaml --img 1536 --weights 'yolov5m6.pt' --name yolov5m6-1536-image-size-fold-0-mid-augs --hyp data/hyps/hyp.scratch-mid.yaml
python train.py --device 0 --epochs 25 --batch-size 18 --data fold_1.yaml --img 1536 --weights 'yolov5m6.pt' --name yolov5m6-1536-image-size-fold-1-mid-augs --hyp data/hyps/hyp.scratch-mid.yaml
