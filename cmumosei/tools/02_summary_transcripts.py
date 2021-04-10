import os
import pandas as pd
import glob

TRANSCRIPT_PATH = '../../../CMU_MOSEI_Raw/Transcript/Segmented/Combined/'
LABELS_PATH = '../../../data/cmumosei_labels_summary.csv'
DATA_PATH = '../../../data/cmumosei/transcripts_summary.csv'

LABELS = ['sentiment', 'happiness', 'sadness', 'anger', 'fear', 'disgust', 'surprise']

COLUMNS = ['VIDEO_ID', 'CLIP', 'start', 'end', 'text']

def main():

    columns = COLUMNS + LABELS
    df = pd.DataFrame(index=[], columns=columns)

    file_names = [ os.path.basename(f) for f in glob.glob(TRANSCRIPT_PATH + '*.txt')]

    # df_file = create_summary('WaaSYRPwQBw.txt')
    for file_name in file_names:
        df_file = create_summary(file_name)
        df = df.append(df_file)
    
    df = annotation(df)
    # df.round({'mean': 1}).sort_values(['VIDEO_ID', 'CLIP']).to_csv('labels_summary.csv', index=False)
    df.to_csv(DATA_PATH, index=False)

    # print(df.groupby('VIDEO_ID').count())
    
def annotation(df_transcripts):
    df_labels = pd.read_csv(LABELS_PATH)

    df = df_transcripts.merge(df_labels, how='inner', on=['VIDEO_ID', 'CLIP'])
    df = df.sort_values(['VIDEO_ID', 'CLIP'])
    df['text'] = df['text'].str.strip().str.upper()
    print(df.head())
    labels = [l + '_mean' for l in LABELS]
    input_columns = COLUMNS + labels
    output_columns = COLUMNS + LABELS
    df = df[input_columns]
    df.columns = output_columns
    # print(df.head())

    # df.to_csv('cmumosei.txt', sep='\t', index=False)
    return df

def create_summary(file_name):
    path = TRANSCRIPT_PATH + file_name
    # print(file_name)
    df = pd.read_csv(path, engine='python', sep='@', header=None, names=[ 'c{0:02d}'.format(i) for i in range(20) ])
    # print(df.head())
    df = df['c00'].str.split('___', expand=True)
    # print(df.head())
    # print(len(df.columns))
    column_length = len(df.columns)
    df.columns = [ 'c{0:02d}'.format(i) for i in range(column_length) ]
    # print(df.head())
    # print(df.columns)
    merge_columns = [ 'c{0:02d}'.format(i) for i in range(5, column_length) ]
    # print(merge_columns)
    if len(df.columns) > 5:
        df_4 = df['c04'].str.cat(df[merge_columns], sep='___', na_rep='No Data')
        df = df[['c00', 'c01', 'c02', 'c03']]
        df['c04'] = df_4
    # print(df.head())

    # df_check = df[['Input.VIDEO_ID', 'Input.CLIP', 'Answer.sentiment']]
    # df_check = df_check[df_check['Input.VIDEO_ID'] == '_0efYOjQYRc']

    # df_check = df_check.groupby(['Input.VIDEO_ID', 'Input.CLIP']).agg(['count', 'mean', max, min])
    # # print(df_check.head())
    # df_check = df_check.reset_index()
    df.columns = COLUMNS
    # print(df.head())
    df['CLIP'] = df['CLIP'].astype(int)
    # print(df.head())

    # df_check.to_csv(file_name + '_summary', index=False)

    return df


if __name__=='__main__':
    main()
