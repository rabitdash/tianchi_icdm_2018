import numpy as np
import os
sample_list = os.listdir('../npydata')
batch_size = 500
def load_batch(size=500):
    for sample in sample_list:
        # raw_data = np.load('../npydata/SRAD2018_TRAIN_010_Part1_500batch.npz')
        raw_data = np.load('../npydata/' + sample)['arr_0']
        for i in range(0, len(raw_data), size):
            yield raw_data[i:i+size]
