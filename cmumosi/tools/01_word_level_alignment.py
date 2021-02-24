import os
import sys
import pandas as pd
import glob
import pickle
sys.path.append('../p2fa_py3')
from p2fa import align

# WAV_FILE_PATH = '../CMU_MOSI_Raw/Audio/WAV_16000/Segmented/03bSnISJMiM_1.wav'
WAV_DIR_PATH = '../CMU_MOSI_Raw/Audio/WAV_16000/Full/'
# TRANSCRIPT_FILE_PATH = '../CMU_MOSI_Raw/Transcript/Segmented/03bSnISJMiM.annotprocessed'
# TRANSCRIPT_DIR_PATH = '../CMU_MOSI_Raw/Transcript/Segmented/'
TRANSCRIPT_DIR_PATH = '../CMU_MOSI_Raw/Transcript/Full/'
TEMP_DIR = 'tmp_full/'
OUTPUT_DIR = 'output_full/'
word_alignments_pickle = 'cmumosi_word_alignments.pkl'


def main():

    # transcript_files = glob.glob(TRANSCRIPT_DIR_PATH + '*.annotprocessed')
    transcript_files = glob.glob(TRANSCRIPT_DIR_PATH + '*.textonly')
    tensors = {}

    for transcript_file in transcript_files:
        video_id = os.path.splitext(os.path.basename(transcript_file))[0]
        # 文字列を抽出
        # df = pd.read_csv(transcript_file, sep='_', header=None, names=['no', 'delim', 'sentence'])
        df = pd.read_csv(transcript_file, sep='@', header=None, names=['sentence'])
        df = df['sentence'].str.split(' ', expand=True)

        # wav_files = glob.glob(WAV_DIR_PATH + video_id + '_*.wav')
        wav_files = glob.glob(WAV_DIR_PATH + video_id + '.wav')
        for wav_file in wav_files:
            try:
                segment_name = os.path.splitext(os.path.basename(wav_file))[0]
                # segment_no = int(segment_name.replace(video_id + '_', ''))
                temp_path = TEMP_DIR + segment_name + '.txt'
                output_path = OUTPUT_DIR + segment_name + '.TextGrid'
                # df_temp = df.iloc[segment_no - 1].T
                df_temp = df.T
                df_temp.to_csv(temp_path, header=False, index=False)
                # print(df.head())

                phoneme_alignments, word_alignments, state_alignments = align.align(wav_file, temp_path, output_path, state_align=True)
                
                features, intervals = generat_features(word_alignments)
                tensors[segment_name] = {'features': features, 'intervals': intervals}
            except Exception as e:
                print(e)

    with open(word_alignments_pickle, mode='wb') as f:
        pickle.dump(tensors, f)

            
def generat_features(word_alignments):
    features = []
    intervals = []
    for word_alignment in word_alignments:
        word = word_alignment[0].encode('utf-8')
        start_timestamp = word_alignment[1]
        end_timestamp = word_alignment[2]
        feature = [word]
        features.append(feature)
        interval = [start_timestamp, end_timestamp]
        intervals.append(interval)
    return features, intervals

if __name__ == '__main__':
    main()


