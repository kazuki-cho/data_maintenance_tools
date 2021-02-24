import os
import sys
import glob
import pandas as pd
import numpy as np

import pickle
import math
import scipy.io as sio
import time


WORD_ALIGN = 'cmumosi_word_alignments.pkl'
FULL_COVAREP_DIR = '../CMU_MOSI_Raw/Audio/WAV_16000/COVAREP/'
OUTPUT = 'cmumosi_alignmets_full.pkl'

def align(video_id, word_align):
    # print(row)
    try:

        mat_contents = sio.loadmat(FULL_COVAREP_DIR + video_id + '.mat')
        audios = []
        for interval in word_align['intervals']:
            epsilon = 10e-6
            if (abs(interval[0]-interval[1])<epsilon):
                continue
            start = math.floor(interval[0] * 100)
            end = math.ceil(interval[1] * 100)
            audio = np.average(mat_contents['features'][start: end], axis=0)
            audios.append(audio)
        audios = np.array(audios)
        return audios
    except Exception as e:
        print('word_align[features]: ', word_align['features'])
        print('word_align[intervals]: ', word_align['intervals'])

def main():

    word_align = pickle.load(open(WORD_ALIGN, 'rb'))
    # print(word_align['_dI--eQ6qVU']['intervals'])
    for video_id in word_align.keys():
        audios = align(video_id, word_align[video_id])
        word_align[video_id]['audio'] = audios

    with open(OUTPUT, mode='wb') as f:
        pickle.dump(word_align, f)

if __name__ == '__main__':
    main()