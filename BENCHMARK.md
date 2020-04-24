# Benchmark on AlphaPose architecture

Indeed we could find the best best models from EfficientDet parameters combination,
but the improvement is not that much, even less than 0.1% improvement.

Now my tasks are:
 * find a way to train a even better model than the original author;
   [x] find the best match detector model for AlphaPose;
   [x] use `sgd` to get a better convergence, and see how much improvement could achieve;
   [-] add noise augumentation, and see how much improvement could achieve;
   [-] understand the training process in AlphaPose and how the flip-test/loss works;
 * validate model on different dataset;
   [-] 理解COCO数据集中的mAP如何计算，如何计算14点的mAP；
   [-] Huawei;
   [-] CrowdPose;
   [-] AI Challenger;
 * validate model in different data format;
   [-] AI Challenger 14 points, using less points should get higher accuracy;
 * integrate a new model;
   [-] Deep-high-resolution-net;
 * Use different postprocessing method instead of parametric pose-NMS:
   [-] KM algorithm in CrowdPose paper;
 * Reconstruct the framework:
   [-] Detector;
   [-] Pose estimator;
   [-] Pose post-processs;
   [-] Integrate different dataset;
   [-] Pose flow;

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

可以看到d4-0.5（366）比Yolo（621）在Mscoco验证集上要少检出许多框。

### 3.1 SGD fine-tuning for another 60 epochs

pose arch     | groundtruth | yolo    | d0-0.5 | d4-0.5 | d7-0.5
--------------|-------------|---------|--------|--------|---------
resnet152     | 0.758       | 0.733   | 0.718  | 0.734  | 0.731
resnet50-dcn  | 0.753       | 0.728   | 0.711  | 0.727  | 0.724
res50-dcn-sgd | 0.753       | 0.728   | -      | 0.726  | -

Using SGD for fine-tuning does not make much difference, the original author's training is pretty fine converged.

 * Q: Why SGD causes GT-as-detect result drops so much, while Yolo-as-detect remains the same?
 * A: noise augumentation只作用在Mscoco类中，没有设置成只在训练阶段，导致结果大幅下降。这也是noise-aug训练过程中精度较低的原因。而使用Mscoco_det类的验证结果则不受影响。这也是noise-aug训练中Yad结果高于Gad的原因；

### 3.2 Add random noise augumentation

pose arch     | groundtruth | yolo    | d0-0.5 | d4-0.5 | d7-0.5
--------------|-------------|---------|--------|--------|---------
resnet152     | 0.758       | 0.733   | 0.718  | 0.734  | 0.731
resnet50-dcn  | 0.753       | 0.728   | 0.711  | 0.727  | 0.724
res50-dcn-ns  | 


可以预见的noise-aug肯定会导致模型的精度下降，训练过程中要低大约4 mAP，只有在别的（现场）数据集上才能知道训练结果是否能提高模型表现。

### 3.3 On


