import os
import sys

import pandas as pd
import numpy as np

import csv
import pickle

text_path = '../../../Cross-Modal-BERT/data/text/'
dev_text = '../../../Cross-Modal-BERT/data/text/dev.tsv'
train_text = '../../../Cross-Modal-BERT/data/text/train.tsv'
test_text = '../../../Cross-Modal-BERT/data/text/test.tsv'

full_data_path = '../../../self_study/cmumosi_full_all.pkl'
noaligne_path = '../../../self_study/cmumosi_audio_noalign.pkl'

OUTPUT = '../../../Cross-Modal-BERT/data/audio/'

audio_select_cols = [ 1, 3, 6, 25, 60 ]

def generate_rainbow_map():

    rainbow_map = {}
    full_data = pickle.load(open(full_data_path, 'rb'))
    print('full_data: ', len(full_data.keys()))

    for segment_name in full_data.keys():
        text = full_data[segment_name]['text'].strip().lower()
        rainbow_map[text] = segment_name
    
    return rainbow_map

def get_audio(env, audio_noalign, rainbow_map, max_len):
    audios = []
    df = pd.read_csv(os.path.join(text_path, env + '.tsv'), sep='\t')

    for index, row in df.iterrows():
        text = row[0].strip().lower()
        segment_name = rainbow_map.get(text)
        if not segment_name:
            print('text: ', text)
            continue
        # print(segment_name)
        audio = audio_noalign[segment_name]['features']
        audio = audio[:, audio_select_cols]
        padding = np.zeros((max_len - len(audio), 5))
        audios.append(np.concatenate([audio, padding]))
    
    print(env, len(audios))
    return audios



def main():
    data = []
    rainbow_map = generate_rainbow_map()
    audio_noalign = pickle.load(open(noaligne_path, 'rb'))
    # max_len = max(map(lambda x: len(audio_noalign[x]['features']), audio_noalign.keys()))
    max_len = 5000
    print('max_len: ', max_len)
    for env in ['train', 'dev', 'test']:
        audios = get_audio(env, audio_noalign, rainbow_map, max_len)
        data.append(audios)
    

    with open(OUTPUT + 'audio_noalign.pkl', mode='wb') as f:
        pickle.dump(data, f)


if __name__ == '__main__':
    main()