"""
This is the benchmark to find the best pose model,
"""
import re


import numpy as np
import pandas as pd





if __name__ == "__main__":
  with open('grid.cloud.txt', 'r') as f:
    results = f.readlines()

  strip = lambda x: [re.split(r'[-|\ |\*|d]', x)[_] for _ in [3, 10, 15, 16]]

  res_res50 = [strip(_) for _ in results if re.search(r'd[0-7].*256x192_res152_', _) is not None]

  res = pd.DataFrame(res_res50)

  print(res)











