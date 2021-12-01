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

DATA_DIR = '../../../CMU_DATA/'
# WORDS_FILE = DATA_DIR + 'CMU_MOSEI_TimestampedWords.csd'
COVAREP_FILE = DATA_DIR + 'CMU_MOSEI_COVAREP.csd'
FACET42_FILE = DATA_DIR + 'CMU_MOSEI_VisualFacet42.csd'
# LABEL_FILE = DATA_DIR + 'CMU_MOSEI_Labels.csd'
TRANSCRIPT_JP_PATH = DATA_DIR + 'transcripts_jp.csv'
OUTPUT = '../data/cmumosei_jp_data.pkl'

def get_segment_data(interval, audio, video):

    epsilon = 10e-6
    if (abs(interval[0]-interval[1])<epsilon):
        return None, None
    start = math.floor(interval[0] * 100)
    end = math.ceil(interval[1] * 100)
    audio_noalign = audio[start: end]

    video_noalign = video[start: end]
    return audio_noalign, video_noalign

def get_text(words):
    
    text = ''
    for w in words:
        text = text + ' ' + w[0].decode().upper()
        text = text.replace(' SP ', ' ')
        text = text.replace(' SP', '')
    return text

def main():

    # h5handle,label_data,metadata=read_CSD(os.path.join(LABEL_FILE))
    # h5handle,words_data,metadata=read_CSD(os.path.join(WORDS_FILE))
    h5handle,audio_data,metadata=read_CSD(os.path.join(COVAREP_FILE))
    h5handle,video_data,metadata=read_CSD(os.path.join(FACET42_FILE))

    df_jp = pd.read_csv(TRANSCRIPT_JP_PATH)

    cmumosei_data = {}
    for index, row in df_jp.iterrows():
        video_id = row['VIDEO_ID']
        segment_no = row['CLIP']
        start = row['start']
        end = row['end']
        interval = [start, end]
        text = row['text']
        sentiment = row['sentiment']
        happiness = row['happiness']
        sadness = row['sadness']
        anger = row['anger']
        fear = row['fear']
        disgust = row['disgust']
        surprise = row['surprise']
        label = [sentiment,happiness,sadness,anger,fear,disgust,surprise]

        # words = words_data.get(video_id)
        audio = audio_data.get(video_id)
        video = video_data.get(video_id)

        # check video_id
        
        if not audio:
            print("the video id of audio doesn't found. miss video_id={video_id}.".format(video_id=video_id))
            continue
        
        if not video:
            print("the video id of video doesn't found. miss video_id={video_id}.".format(video_id=video_id))
            continue
        
        audio = audio['features']
        video = video['features']
        
        # try:
        #     text = get_text(words)
        # except Exception as e:
        #     print("failed words to text. failed video_id={video_id}".format(video_id=video_id))
        #     continue
        try:
            audio_noalign, video_noalign = get_segment_data(interval, audio, video)
        except Exception as e:
            print("align failed. failed video_id={video_id}".format(video_id=video_id))
            continue

        full_data = {}
        full_data['video_id'] = video_id
        full_data['segment_no'] = segment_no
        full_data['interval'] = np.array(interval)
        full_data['label'] = np.array(label)
        full_data['text'] = text
        full_data['audio_noalign'] = np.array(audio_noalign)
        full_data['video_noalign'] = np.array(video_noalign)

        segment_name = str(video_id) + '_' + str(segment_no)
        cmumosei_data[segment_name] = full_data

    with open(OUTPUT, mode='wb') as f:
        pickle.dump(cmumosei_data, f)

if __name__ == '__main__':
    main()
