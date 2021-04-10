import os
import sys
import glob
import pandas as pd
import numpy as np

import pickle
import math
import scipy.io as sio
import time

sys.path.append('../../../CMU-MultimodalSDK')
import mmsdk
from mmsdk import mmdatasdk as md
from mmsdk.mmdatasdk.computational_sequence.file_ops import *


WORD_ALIGN = '../../../data/cmumosi_word_alignments.pkl'
FULL_COVAREP_DIR = '../../../CMU_MOSI_Raw/Audio/WAV_16000/COVAREP/'
FULL_FACET42 = '../../../CMU-MultimodalSDK/cmumosi/CMU_MOSI_Visual_Facet_42.csd'
OUTPUT = '../../../data/cmumosi_alignmets_full_all.pkl'

def align(video_id, word_align, facet42=None):
    # print(row)
    try:

        mat_contents = sio.loadmat(FULL_COVAREP_DIR + video_id + '.mat')
        audios = []
        videos = []
        for interval in word_align['intervals']:
            epsilon = 10e-6
            if (abs(interval[0]-interval[1])<epsilon):
                continue
            start = math.floor(interval[0] * 100)
            end = math.ceil(interval[1] * 100)
            audio = np.average(mat_contents['features'][start: end], axis=0)
            audios.append(audio)

            if facet42:
                video_start = math.floor(interval[0] / 4)
                video_end = math.ceil(interval[1] / 4)
                video = np.average(facet42['features'][video_start: video_end], axis=0)
                videos.append(video)
        audios = np.array(audios)
        videos = np.array(videos)
        return audios, videos
    except Exception as e:
        print('video_id: ', video_id)

def main():

    word_align = pickle.load(open(WORD_ALIGN, 'rb'))
    # h5handle,data,metadata=read_CSD(os.path.join(FULL_FACET42))
    # print(word_align['_dI--eQ6qVU']['intervals'])
    word_keys = word_align.keys()
    # video_keys = data.keys()

    for video_id in word_align.keys():
        # if video_id == 'c5xsKMxpXnc':
        #     continue
        # audios, videos = align(video_id, word_align[video_id], data.get(video_id))
        audios, videos = align(video_id, word_align[video_id])
        word_align[video_id]['audio'] = audios
        word_align[video_id]['video'] = videos

    with open(OUTPUT, mode='wb') as f:
        pickle.dump(word_align, f)

if __name__ == '__main__':
    main()
