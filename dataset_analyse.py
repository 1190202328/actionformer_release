import os
import numpy as np

# dir_name = './data/thumos/i3d_features'
dir_name = './data/anet_1.3/tsp_features'

count = 0
total_length = len(os.listdir(dir_name))
for i, file in enumerate(os.listdir(dir_name)):
    # load features
    feats = np.load(f'{dir_name}/{file}').astype(np.float32)
    if feats.shape[0] > 2304:
        count += 1
        print(count)
    del feats
    if i % 100 == 0:
        print(f'{i}/{total_length}')
print(count)