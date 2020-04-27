"""
This is the benchmark to find the best pose model,
"""
import re
import json

import numpy as np
import pandas as pd

import cv2

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def grid_search():
  with open('grid.cloud.txt', 'r') as f:
    results = f.readlines()

  strip = lambda x: [re.split(r'[-|\ |\*|d]', x)[_] for _ in [3, 10, 15, 16]]

  res_res50 = [strip(_) for _ in results if re.search(r'd[0-7].*256x192_res152_', _) is not None]

  res = pd.DataFrame(res_res50)

  print(res)


if __name__ == "__main__":
  # return grid search result
  # grid_search()

  dataset = json.load(open('data/coco/annotations/person_keypoints_val2017.hie.json'))


  for anno in dataset['annotations']:
    img = cv2.imread('data/coco/val2017/%012d.jpg' % anno['image_id'])
    print(img.shape)

    keypoints = np.array(anno['keypoints']).reshape(-1, 3)

    for kp in keypoints:
      cv2.circle(img, (int(kp[0]), int(kp[1])), 2, (0, 255, 0), 6*int(kp[2]))

    cv2.imshow('_', img)
    if cv2.waitKey(0) == 27: break

  print(len(dataset))











