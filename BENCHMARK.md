# Benchmark on AlphaPose architecture

Indeed we could find the best best models from EfficientDet parameters combination,
but the improvement is not that much, even less than 0.1% improvement.

Now my tasks are:
 * find a way to train a even better model than the original author;
   [x] find the best match detector model for AlphaPose;
   [-] use `sgd` to get a better convergence, and see how much improvement could achieve;
   [-] add noise augumentation, and see how much improvement could achieve;
 * validate model on different dataset;
   [-] Huawei;
   [-] CrowdPose;
   [-] AI Challenger;
 * validate model in different data format;
   [-] AI Challenger 14 points, using less points should get higher accuracy;
 * integrate a new model;
   [-] Deep-high-resolution-net;
 * Use different postprocessing method instead of parametric pose-NMS:
   [-] KM algorithm in CrowdPose paper;

From now on, all our test process will be done on ResNet50-dcn model.

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

pose arch     | groundtruth | yolo    | d0-0.5 | d4-0.5 | d7-0.5
--------------|-------------|---------|--------|--------|---------
resnet152     | 0.758       | 0.733   | 0.718  | 0.734  | 0.731
resnet50-dcn  | 0.753       | 0.728   | 0.711  | 0.727  | 0.724


## 3. Fine-tuning on resnet50-dcn

