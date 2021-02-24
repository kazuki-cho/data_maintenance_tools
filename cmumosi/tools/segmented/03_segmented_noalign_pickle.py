import os
import sys
import glob
import pickle

import numpy as np
import scipy.io as sio

COVAREP_DIR = 'segmented_covarep/'
OUTPUT = 'cmumosi_audio_noalign.pkl'

def main():

    data = {}
    mat_files = glob.glob(COVAREP_DIR + '*.mat')
    for mat_file in mat_files:
        segment_name = os.path.splitext(os.path.basename(mat_file))[0]
        mat_content = sio.loadmat(mat_file)
        intervals = []
        data[segment_name] = {}
        for i in range(len(mat_content['features'])):
            start = float(i) / 100
            end = float(i + 1) / 100
            intervals = np.array([start, end])
        data[segment_name]['features'] = mat_content['features']
        data[segment_name]['intervals'] = intervals
    
    with open(OUTPUT, mode='wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()