"""
This is the benchmark to find the best pose model,
Indeed we could find the best best models from EfficientDet parameters combination,
but the improvement is not that much, even less than 0.1% improvement.

Now my tasks are:
 * find a way to train a even better model than the original author;
 * validate model on different dataset;
 * validate model in different data format;
 * integrate a new model;

From now on, all our test process will be done on ResNet50-dcn model.
"""
import re


import numpy as np
import pandas as pd





if __name__ == "__main__":
  with open('grid.cloud.txt', 'r') as f:
    results = f.readlines()

  strip = lambda x: [re.split(r'[-|\ |\*|d]', x)[_] for _ in [3, 10, 15, 16]]

  res_res50 = [strip(_) for _ in results if re.search(r'd[0-7].*256x192_res18_', _) is not None]

  res = pd.DataFrame(res_res50)

  print(res)











