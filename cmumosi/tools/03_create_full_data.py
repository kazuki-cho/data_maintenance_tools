import os
import sys
import glob
import pandas as pd
import numpy as np

import pickle
import math
import scipy.io as sio
import time


TEXT_PATH = '../../../CMU_MOSI_Raw/Transcript/Segmented/'
LABEL_PATH = '../../../CMU_MOSI_Raw/Labels/OpinionLevelSentiment.csv'
AUDIO_PATH = '../../../data/cmumosi_alignmets_full_all.pkl'

OUTPUT = '../../../data/cmumosi_full_all.pkl'

def main():

    audios = pickle.load(open(AUDIO_PATH, 'rb'))
    ret = {}
    df = pd.read_csv(LABEL_PATH, header=None, names=['start', 'end', 'video_id', 'segment_no', 'label'])

    for index, row in df.iterrows():
        video_id = row['video_id']
        # if video_id == 'c5xsKMxpXnc':
        #     continue
        segment_no = row['segment_no']
        segment_name = row['video_id'] + '_' + str(segment_no)
        print(segment_name)

        label = row['label']
        segment_start = row['start']
        segment_end = row['end']

        df_text = pd.read_csv(TEXT_PATH + video_id + '.annotprocessed', sep='@', header=None, names=['text'])
        df_text = df_text['text'].str.split('_DELIM_', expand=True)
        df_text.columns = ['segment_no', 'text']
        # print(df_text.head())
        df_temp = df_text[df_text['segment_no'].astype(int) == int(segment_no)]
        # print(df_temp.head())
        text = df_temp.iat[0, 1]
        # print(text)

        segment_audios = []
        segment_intervals = []
        segment_words = []
        segment_videos = []
        word_cnt = len(audios[video_id]['features'])
        for cnt in range(word_cnt):
            interval = audios[video_id]['intervals'][cnt]
            word_start = interval[0]
            word_end = interval[1]

            if word_end < segment_start:
                continue
            elif segment_end < word_start:
                break
            else:
                segment_words.append(audios[video_id]['features'][cnt])
                segment_audios.append(audios[video_id]['audio'][cnt])
                if len(audios[video_id]['video']) > 0:
                    segment_videos.append(audios[video_id]['video'][cnt])
                else:
                    segment_videos.append([])
                segment_intervals.append(interval)
        
        segment = {}
        segment['video_id'] = video_id
        segment['label'] = label
        segment['text'] = text
        segment['words'] = np.array(segment_words)
        segment['audio'] = np.array(segment_audios)
        segment['video'] = np.array(segment_videos)
        segment['word_intervals'] = np.array(segment_intervals)
        segment['intervals'] = np.array([segment_start, segment_end])

        ret[segment_name] = segment



    with open(OUTPUT, mode='wb') as f:
        pickle.dump(ret, f)

if __name__ == '__main__':
    main()
