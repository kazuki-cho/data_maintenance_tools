import os
import sys

import pandas as pd
import numpy as np

import csv
import pickle

sys.path.append('../../../CMU-MultimodalSDK')
import mmsdk
from mmsdk import mmdatasdk as md
from mmsdk.mmdatasdk.dataset.standard_datasets.CMU_MOSEI.cmu_mosei_std_folds import standard_train_fold, standard_valid_fold, standard_test_fold
# [,  ]

# bug_data = ['5W7Z1C_fDaE', '2WGyTLYerpo', 'tIrG4oNLFzE']

# bug_data = ['BioHAh1qJAQ', 'BXuRRbG0Ugk', 'Dg_0XKD0Mf4', '5W7Z1C_fDaE', 'ZUXBRvtny7o', 'yvsjCA6Y5Fc', 'rnaNMUZpvvg', 'tIrG4oNLFzE']

bug_data = []
# # temp [  ]

# standard_train_fold=[ 'tIrG4oNLFzE' ]
# standard_valid_fold=['5W7Z1C_fDaE']
# standard_test_fold=['BXuRRbG0Ugk']

audio_select_cols = [ 1, 3, 6, 25, 60 ]
video_select_cols = []
# video_select_cols = [0, 2, 5, 10, 11, 12, 14, 17, 20, 21, 22, 24, 25, 29, 30, 31, 32, 36, 37, 40]

# label columns
# [sentiment, happiness, sadness, anger, fear, disgust, surprise]

MOSEI_DATA = '../data/cmumosei_jp_data.pkl'
PAIR_SETS = '../data/pair_segments.pkl'
OUTPUT = '../data'
SEQUENCE_LEN = 50
AUDIO_NOALIGNED_LEN = 5000
VIDEO_NOALIGNED_LEN = 1250


def get_data(features):

    video_id = features.get('video_id')
    segment_no = features.get('segment_no')
    audio_noalign = features.get('audio_noalign')
    video_noalign = features.get('video_noalign')
    text = features['text'].strip()
    label = features['label']

    if audio_noalign.shape[0] == 0:
        print('audio noalign data not found. miss video_id={}'.format(video_id))
        return None, None, None
    if video_noalign.shape[0] == 0:
        print('video noalign data not found. miss video_id={}'.format(video_id))
        return None, None, None


    if len(audio_select_cols) != 0:
        # audio = audio[:, audio_select_cols]
        audio_noalign = audio_noalign[:, audio_select_cols]
    if len(video_select_cols) != 0:
        # video = video[:, video_select_cols]
        video_noalign = video_noalign[:, video_select_cols]

    audio_noalign = data_formatter(audio_noalign, AUDIO_NOALIGNED_LEN)
    video_noalign = data_formatter(video_noalign, VIDEO_NOALIGNED_LEN)
    
    text_attr = text.split()
    if len(text_attr) > 50:
        text_attr = text_attr[:50]
    text = ' '.join(text_attr)

    text_data = {
        'video_id': video_id, 'segment_no': int(segment_no), 'text': text, 'sentiment': label[0], 'happiness': label[1], 'sadness': label[2],
        'anger': label[3], 'fear': label[4], 'disgust': label[5], 'surprise': label[6]
    }

    return text_data, audio_noalign, video_noalign


def main():

    input_text_train_data = []
    input_text_valid_data = []
    input_text_test_data = []
    response_text_train_data = []
    response_text_valid_data = []
    response_text_test_data = []
    # audio_train_data = []
    # audio_valid_data = []
    # audio_test_data = []
    # video_train_data = []
    # video_valid_data = []
    # video_test_data = []
    input_audio_noaligned_train_data = []
    input_audio_noaligned_valid_data = []
    input_audio_noaligned_test_data = []
    response_audio_noaligned_train_data = []
    response_audio_noaligned_valid_data = []
    response_audio_noaligned_test_data = []
    input_video_noaligned_train_data = []
    input_video_noaligned_valid_data = []
    input_video_noaligned_test_data = []
    response_video_noaligned_train_data = []
    response_video_noaligned_valid_data = []
    response_video_noaligned_test_data = []
    # lmms_train_data = []
    # lmms_valid_data = []
    # lmms_test_data = []

    # label
    input_label_train_data = []
    input_label_valid_data = []
    input_label_test_data = []
    response_label_train_data = []
    response_label_valid_data = []
    response_label_test_data = []


    mosei_data = pickle.load(open(MOSEI_DATA, 'rb'))
    pair_dict = pickle.load(open(PAIR_SETS, 'rb'))
    # lmms_data = pickle.load(open(LMMS_DATA, 'rb'))

    for segment_name, features in mosei_data.items():
        pair_segment_name = pair_dict[segment_name]
        pair_features = mosei_data[pair_segment_name]
        input_text, input_audio, input_vido = get_data(features)
        response_text, response_audio, response_vido = get_data(pair_features)


        if video_id in standard_train_fold and video_id not in bug_data:
        # if video_id in standard_train_fold:
            # print(audio.shape)
            input_text_train_data.append(input_text)
            # audio_train_data.append(audio)
            # video_train_data.append(video)
            input_audio_noaligned_train_data.append(input_audio)
            input_video_noaligned_train_data.append(input_vido)
            # lmms_train_data.append(lmms)
            response_text_train_data.append(response_text)
            response_audio_noaligned_train_data.append(response_audio)
            response_video_noaligned_train_data.append(response_vido)
        elif video_id in standard_valid_fold and video_id not in bug_data:
        # elif video_id in standard_valid_fold:
            # print(audio.shape)
            input_text_valid_data.append(input_text)
            # audio_valid_data.append(audio)
            # video_valid_data.append(video)
            input_audio_noaligned_valid_data.append(input_audio)
            input_video_noaligned_valid_data.append(input_vido)
            # lmms_valid_data.append(lmms)
            response_text_valid_data.append(response_text)
            response_audio_noaligned_valid_data.append(response_audio)
            response_video_noaligned_valid_data.append(response_vido)
        elif video_id in standard_test_fold and video_id not in bug_data:
        # elif video_id in standard_test_fold:
            # print(audio.shape)
            input_text_test_data.append(input_text)
            # audio_test_data.append(audio)
            # video_test_data.append(video)
            input_audio_noaligned_test_data.append(input_audio)
            input_video_noaligned_test_data.append(input_vido)
            # lmms_test_data.append(lmms)
        else:
            continue

    output_features('/audio/input_audio_noalign.pkl', input_audio_noaligned_train_data, input_audio_noaligned_valid_data, input_audio_noaligned_test_data)
    output_features('/audio/response_audio_noalign.pkl', response_audio_noaligned_train_data, response_audio_noaligned_valid_data, response_audio_noaligned_test_data)
    output_features('/video/input_video_noalign.pkl', input_video_noaligned_train_data, input_video_noaligned_valid_data, input_video_noaligned_test_data)
    output_features('/video/response_video_noalign.pkl', response_video_noaligned_train_data, response_video_noaligned_valid_data, response_video_noaligned_test_data)
    output_text(input_text_train_data, input_text_valid_data, input_text_test_data, 'input_text')
    output_text(response_text_train_data, response_text_valid_data, response_text_test_data, 'response_text')


def data_formatter(data, formatted_length):
    data_len = data.shape[0]
    if data_len > formatted_length:
        formatted_data = data[0: formatted_length]
    elif data_len < formatted_length:
        zero_features = np.zeros((formatted_length - data_len, data.shape[1]))
        formatted_data = np.concatenate([data, zero_features])
    else:
        formatted_data = data
    
    return formatted_data


def output_features(path, train_data, valid_data, test_data):
    new_data = []
    train_data = np.array(train_data)
    # print(train_data.shape)
    valid_data = np.array(valid_data)
    # print(valid_data.shape)
    test_data = np.array(test_data)
    # print(test_data.shape)
    new_data = np.array([train_data, valid_data, test_data])

    with open(OUTPUT + path, mode='wb') as f:
        pickle.dump(new_data, f)

def output_text(word_train_data, word_valid_data, word_test_data, key):
    df_train = pd.DataFrame(word_train_data)
    df_valid = pd.DataFrame(word_valid_data)
    df_test = pd.DataFrame(word_test_data)

    os.makedirs(OUTPUT + '/{key}'.format(key=key), exist_ok=True)
    df_train.to_csv(OUTPUT + '/{key}/train.tsv'.format(key=key), index=False, sep='\t')
    df_valid.to_csv(OUTPUT + '/{key}/dev.tsv'.format(key=key), index=False, sep='\t')
    df_test.to_csv(OUTPUT + '/{key}/test.tsv'.format(key=key), index=False, sep='\t')


if __name__ == '__main__':
    main()
