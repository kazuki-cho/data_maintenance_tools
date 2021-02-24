import os
import sys
import glob
import pandas as pd
import numpy as np

import math
import scipy.io as sio
import time


Label_PATH = '../CMU_MOSI_Raw/Labels/OpinionLevelSentiment.csv'
FULL_COVAREP_DIR = '../CMU_MOSI_Raw/Audio/WAV_16000/COVAREP/'
OUTPUT_DIR = 'segmented_covarep/'

def dev_covarep(row):
    # print(row)
    try:
        start = math.floor(row['start'] * 100)
        end = math.ceil(row['end'] * 100)
        mat_file = FULL_COVAREP_DIR + row['video_id'] + '.mat'
        mat_contents = sio.loadmat(mat_file)
        new_feature = []
        for i in range(start, end + 1):
            new_feature.append(mat_contents['features'][i])
        new_feature = np.array(new_feature)
        mat_contents['features'] = new_feature

        output_file = OUTPUT_DIR + row['video_id'] + '_' + str(row['segment_no']) + '.mat'
        sio.savemat(output_file, mat_contents)
    except Exception as e:
        print(row['video_id'] + '_' + str(row['segment_no']))

def main():

    df = pd.read_csv(Label_PATH, header=None, names=['start', 'end', 'video_id', 'segment_no', 'label'])
    # df = df.iloc[1]
    # print(df.head())
    for index, row in df.iterrows():
        dev_covarep(row)



if __name__ == '__main__':
    main()