# Benchmark on AlphaPose architecture

## 1. Using different detect network

使用yolo和centerpose作为检测框的对比：

pose arch     | groundtruth | yolo    | centerpose
--------------|-------------|---------|--------------
resnet50      | 0.732       | 0.712   | 0.646

## 2. Using different SPPE network

使用yolo和centerpose作为检测框，使用resnet50/resnet18/resnet18_aug作为SPPE框架：
```
python scripts/validate.py --cfg ./configs/coco/resnet/256x192_res18_lr1e-3_1x.yaml  --checkpoint ./exp/mytrain-res18/final.pth --detector efficientdet
```

pose arch     | groundtruth | yolo    | centerpose   | d0    | d4-0.2 | d4    | d7
--------------|-------------|---------|--------------|-------|--------|-------
resnet152     | 0.758       | 0.733   | 
resnet50-dcn  | 0.753       | 0.728   |              | 0.711 |
resnet50      | 0.743       | 0.720   | 0.652        | 0.705 | 0.674  | 0.717 | 0.710
resent18-aug  | 0.693       | 0.672   | 0.613        |       |        |




## 3. Fine-tuning on resnet50-dcn

