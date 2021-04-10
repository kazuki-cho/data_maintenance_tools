import os
import sys

import pandas as pd
import numpy as np

import csv
import pickle

standard_train_fold=['2iD-tVS8NPw', '8d-gEyoeBzc', 'Qr1Ca94K55A', 'Ci-AH39fi3Y', '8qrpnFRGt2A', 'Bfr499ggo-0', 'QN9ZIUWUXsY', '9T9Hf74oK10', '7JsX8y1ysxY', '1iG0909rllw', 'Oz06ZWiO20M', 'BioHAh1qJAQ', '9c67fiY0wGQ', 'Iu2PFX3z_1s', 'Nzq88NnDkEk', 'Clx4VXItLTE', '9J25DZhivz8', 'Af8D0E4ZXaw', 'TvyZBvOMOTc', 'W8NXH0Djyww', '8OtFthrtaJM', '0h-zjBukYpk', 'Vj1wYRQjB-o', 'GWuJjcEuzt8', 'BI97DNYfe5I', 'PZ-lDQFboO8', '1DmNV9C1hbY', 'OQvJTdtJ2H4', 'I5y0__X72p0', '9qR7uwkblbs', 'G6GlGvlkxAQ', '6_0THN4chvY', 'Njd1F0vZSm4', 'BvYR0L6f2Ig', '03bSnISJMiM', 'Dg_0XKD0Mf4', '5W7Z1C_fDaE', 'VbQk4H8hgr0', 'G-xst2euQUc', 'MLal-t_vJPM', 'BXuRRbG0Ugk', 'LSi-o-IrDMs', 'Jkswaaud0hk', '2WGyTLYerpo', '6Egk_28TtTM', 'Sqr0AcuoNnk', 'POKffnXeBds', '73jzhE8R1TQ', 'OtBXNcAL_lE', 'HEsqda8_d0Q', 'VCslbP0mgZI', 'IumbAb8q2dM']
standard_valid_fold=['WKA5OygbEKI', 'c5xsKMxpXnc', 'atnd_PF-Lbs', 'bvLlb-M3UXU', 'bOL9jKpeJRs', '_dI--eQ6qVU', 'ZAIRrfG22O0', 'X3j2zQgwYgE', 'aiEXnCPZubE', 'ZUXBRvtny7o']
standard_test_fold=['tmZoasNr4rU', 'zhpQhgha_KU', 'lXPQBPVc5Cw', 'iiK8YX8oH1E', 'tStelxIAHjw', 'nzpVDcQ0ywM', 'etzxEpPuc6I', 'cW1FSBF59ik', 'd6hH302o4v8', 'k5Y_838nuGo', 'pLTX3ipuDJI', 'jUzDDGyPkXU', 'f_pcplsH_V0', 'yvsjCA6Y5Fc', 'nbWiPyCm4g0', 'rnaNMUZpvvg', 'wMbj6ajWbic', 'cM3Yna7AavY', 'yDtzw_Y-7RU', 'vyB00TXsimI', 'dq3Nf_lMPnE', 'phBUpBr1hSo', 'd3_k5Xpfmik', 'v0zCBqDeKcE', 'tIrG4oNLFzE', 'fvVhgmXxadc', 'ob23OKe5a9Q', 'cXypl4FnoZo', 'vvZ4IcEtiZc', 'f9O3YtZ2VfI', 'c7UH_rxdZv4']

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


# WORD_PATH = 'cmumosi_word_alignments.pkl'
# AUDIO_PATH = 'cmumosi_audio_aligned.pkl'
# Label_PATH = '../CMU_MOSI_Raw/Labels/OpinionLevelSentiment.csv'
# INPUT_DATA = '../self_study/cmumosi_full.pkl'
INPUT_DATA = '../../../self_study/cmumosi_full_vocab.pkl'
OUTPUT = '../../../self_study/data'
SEQUENCE_LEN = 50


def main():

    word_train_data = []
    word_valid_data = []
    word_test_data = []
    audio_train_data = []
    audio_valid_data = []
    audio_test_data = []
    video_train_data = []
    video_valid_data = []
    video_test_data = []

    full_data = pickle.load(open(INPUT_DATA, 'rb'))

    for segment_name in full_data.keys():
        segment = full_data[segment_name]
        video_id = segment['video_id']

        audio = segment['audio']
        video = segment['video']
        if len(audio_select_cols) != 0:
            audio = audio[:, audio_select_cols]
        if len(video_select_cols) != 0:
            video = video[:, video_select_cols]
        audio_len = audio.shape[0]
        video_len = video.shape[0]

        if audio_len > SEQUENCE_LEN:
            audio = audio[0: SEQUENCE_LEN]
        if audio_len < SEQUENCE_LEN:
            zero_features = np.zeros((SEQUENCE_LEN - audio_len, audio.shape[1]))
            audio = np.concatenate([audio, zero_features])
        
        if video_len > SEQUENCE_LEN:
            video = video[0: SEQUENCE_LEN]
        if video_len < SEQUENCE_LEN:
            zero_features = np.zeros((SEQUENCE_LEN - video_len, video.shape[1]))
            video = np.concatenate([video, zero_features])
        

        text = {'text': segment['text'].strip(), 'label': segment['label']}

        if video_id in standard_train_fold and video_id not in bug_data:
        # if video_id in standard_train_fold:
            # print(audio.shape)
            word_train_data.append(text)
            audio_train_data.append(audio)
            video_train_data.append(video)
        elif video_id in standard_valid_fold and video_id not in bug_data:
        # elif video_id in standard_valid_fold:
            # print(audio.shape)
            word_valid_data.append(text)
            audio_valid_data.append(audio)
            video_valid_data.append(video)
        elif video_id in standard_test_fold and video_id not in bug_data:
        # elif video_id in standard_test_fold:
            # print(audio.shape)
            word_test_data.append(text)
            audio_test_data.append(audio)
            video_test_data.append(video)
        else:
            continue

    output_audio(audio_train_data, audio_valid_data, audio_test_data)
    output_video(video_train_data, video_valid_data, video_test_data)
    output_text(word_train_data, word_valid_data, word_test_data)


# def main2():

#     word_data = pickle.load(open(WORD_PATH, 'rb'))
#     audio_data = pickle.load(open(AUDIO_PATH, 'rb'))

#     word_train_data = []
#     word_valid_data = []
#     word_test_data = []
#     audio_train_data = []
#     audio_valid_data = []
#     audio_test_data = []

#     df = pd.read_csv(Label_PATH, header=None, names=['start', 'end', 'video_id', 'segment_no', 'label'])

#     for index, row in df.iterrows():
#         video_id = row['video_id']
#         segment_name = row['video_id'] + '_' + str(row['segment_no'])

#         words = word_data.get(segment_name)
#         audios = audio_data.get(segment_name)
#         if not words or not audios:
#             continue

#         sentence = ''
#         for arr in words['features']:
#             # print(arr)
#             word = arr[0].decode()
#             sentence = sentence + ' ' + word
        
#         text = {'text': sentence, 'label': row['label']}
#         audio = audios['features']
#         if audio.shape[0] != 50:
#             print(segment_name, audio.shape[0])

#         # if video_id in standard_train_fold and video_id not in bug_data:
#         if video_id in standard_train_fold:
#             # print(audio.shape)
#             env = 'train'
#             word_train_data.append(text)
#             audio_train_data.append(audio)
#         # elif video_id in standard_valid_fold and video_id not in bug_data:
#         elif video_id in standard_valid_fold:
#             # print(audio.shape)
#             env = 'dev'
#             word_valid_data.append(text)
#             audio_valid_data.append(audio)
#         # elif video_id in standard_test_fold and video_id not in bug_data:
#         elif video_id in standard_test_fold:
#             # print(audio.shape)
#             env = 'test'
#             word_test_data.append(text)
#             audio_test_data.append(audio)
#         else:
#             continue

#     output_audio(audio_train_data, audio_valid_data, audio_test_data)
#     output_text(word_train_data, word_valid_data, word_test_data)


def output_audio(audio_train_data, audio_valid_data, audio_test_data):
    new_audio_data = []
    audio_train_data = np.array(audio_train_data)
    # print(audio_train_data.shape)
    audio_valid_data = np.array(audio_valid_data)
    # print(audio_valid_data.shape)
    audio_test_data = np.array(audio_test_data)
    # print(audio_test_data.shape)
    # new_audio_data.append(audio_train_data)
    # new_audio_data.append(audio_valid_data)
    # new_audio_data.append(audio_test_data)
    new_audio_data = np.array([audio_train_data, audio_valid_data, audio_test_data])

    with open(OUTPUT + '/audio/cmumosi_audio.pkl', mode='wb') as f:
        pickle.dump(new_audio_data, f)

def output_video(video_train_data, video_valid_data, video_test_data):
    new_video_data = []
    video_train_data = np.array(video_train_data)
    print(video_train_data.shape)
    video_valid_data = np.array(video_valid_data)
    print(video_valid_data.shape)
    video_test_data = np.array(video_test_data)
    print(video_test_data.shape)
    new_video_data = np.array([video_train_data, video_valid_data, video_test_data])

    with open(OUTPUT + '/video/cmumosi_video.pkl', mode='wb') as f:
        pickle.dump(new_video_data, f)

def output_text(word_train_data, word_valid_data, word_test_data):
    df_train = pd.DataFrame(word_train_data)
    df_valid = pd.DataFrame(word_valid_data)
    df_test = pd.DataFrame(word_test_data)

    df_train.to_csv(OUTPUT + '/text/train.tsv', index=False, sep='\t')
    df_valid.to_csv(OUTPUT + '/text/dev.tsv', index=False, sep='\t')
    df_test.to_csv(OUTPUT + '/text/test.tsv', index=False, sep='\t')


if __name__ == '__main__':
    main()
