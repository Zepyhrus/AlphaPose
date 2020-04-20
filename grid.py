"""
A grid search among 35 detectors:
  groundtruth
  yolo
  centerpose
  d0~d7, iou_threshold from 0.2 to 0.5
and 3 posers, cfg from ./configs/coco/resnet/:
  resnet152:      256x192_res152_lr1e-3_1x-duc.yaml,  fast_421_res152_256x192.pth
  resnet50-dcn:   256x192_res50_lr1e-3_2x-dcn.yaml,   fast_dcn_res50_256x192.pth
  resnet18-aug:   256x192_res18_lr1e-3_1x.yaml,       fast_res18_aug_256x192.pth
"""
import argparse
from os.path import join, split
from tqdm import tqdm
import json

import numpy as np

import torch

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.metrics import evaluate_mAP
from alphapose.utils.transforms import (flip, flip_heatmap, get_func_heatmap_to_coord)
from scripts.validate import validate, validate_gt


def get_args():
  paresr = argparse.ArgumentParser(description='Grid search parameters')
  paresr.add_argument('-c', '--cfg', type=str, help='config file')
  paresr.add_argument('-p', '--checkpoint', type=str, help='checkpoint file')
  paresr.add_argument('-b', '--batch-size', dest='batch_size', type=int, help='pose inference batchsize', default=40)
  paresr.add_argument('-d', '--detector', type=str, default='efficientdet')
  paresr.add_argument('-l', '--level', type=int, default=0, help='efficient level')
  paresr.add_argument('-i', '--iou-threshold', dest='iou_threshold', type=float, default=0.4)
  paresr.add_argument('-f', '--flip-test', dest='flip_test', type=bool, default=True)


  args = paresr.parse_args()

  return args



def val(opt):
  cfg = update_config(opt.cfg)
  
  det_root = './exp/json'
  if opt.detector == 'efficientdet':
    det_file = 'det_d{}_{:.1f}.json'.format(opt.level, opt.iou_threshold)
  else :
    det_file = 'det_{}.json'.format(opt.detector)
  cfg.DATASET.TEST.DET_FILE = join(det_root, det_file)

  m = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)

  print(f'Loading model from {opt.checkpoint}...')
  m.load_state_dict(torch.load(opt.checkpoint))
  m = m.cuda()
  heatmap_to_coord = get_func_heatmap_to_coord(cfg)


  with torch.no_grad():
    detbox_AP = validate(m, heatmap_to_coord, opt, cfg)
    gt_AP = validate_gt(m, heatmap_to_coord, opt, cfg)

  det_name = 'error'
  if opt.detector == 'efficientdet':
    det_name = 'd{}-{:.1f}'.format(opt.level, opt.iou_threshold)
  else:
    det_name = opt.detector
  
  return '### gt box: {} mAP | det box: {} mAP ### @ {}*{}'.format(gt_AP, detbox_AP, det_name, opt.cfg)



if __name__ == "__main__":
  # 






  
  # we shall generate detect results first ----------------------------------------
  config_root = 'configs/coco/resnet'
  config_files = [
    '256x192_res18_lr1e-3_1x.yaml',
    '256x192_res50_lr1e-3_2x-dcn.yaml'
  ]

  checkpoints = [
    './models/fast_res18_aug_256x192.pth',
    './models/fast_dcn_res50_256x192.pth'
  ]

  detectors = [
    'yolo',
    'centerpose',
    'efficientdet'
  ]

  with open('grid.txt', 'w') as f:
    for i in range(len(checkpoints)):
      config_file = config_files[i]
      checkpoint = checkpoints[i]

      opt = get_args()
      opt.cfg = join(config_root, config_file)
      opt.checkpoint = checkpoint
      opt.gpus = [0]
      opt.device = torch.device('cuda:0')

      for detector in detectors:
        opt.detector = detector

        if opt.detector == 'efficientdet':
          for iou in range(2, 4):
            opt.iou_threshold = iou / 10

            for level in range(2):
              opt.level = level
              f.write(val(opt)+'\n')
        else:
          f.write(val(opt)+'\n')
    
  














