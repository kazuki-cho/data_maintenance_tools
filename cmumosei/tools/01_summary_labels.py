import pandas as pd
import numpy as np

LABELS_PATH = '../../../CMU_MOSEI_Raw/Labels/'
DATA_PATH = '../../../data/cmumosei_labels_summary.csv'
file_names = [
    '5000_batch_raw',
    'Batch_2980374_batch_results',
    'extreme_sentiment_results',
    'mosi_pom_output',
    'mturk_extra_v2',
    'pom_extra_sqa_mono_results'
]

COLUMNS = [
            'VIDEO_ID', 'CLIP',
            'sentiment_count', 'sentiment_mean', 'sentiment_max', 'sentiment_min',
            'happiness_count', 'happiness_mean', 'happiness_max', 'happiness_min',
            'sadness_count', 'sadness_mean', 'sadness_max', 'sadness_min',
            'anger_count', 'anger_mean', 'anger_max', 'anger_min',
            'fear_count', 'fear_mean', 'fear_max', 'fear_min',
            'disgust_count', 'disgust_mean', 'disgust_max', 'disgust_min',
            'surprise_count', 'surprise_mean', 'surprise_max', 'surprise_min',
        ]

def main():

    df = pd.DataFrame(index=[], columns=COLUMNS)

    for file_name in file_names:
        df_file = create_summary(file_name)
        df = df.append(df_file)
    
    round_columns = {'sentiment_mean': 1, 'happiness_mean': 1, 'sadness_mean': 1, 'anger_mean': 1, 'fear_mean': 1, 'disgust_mean': 1, 'surprise_mean': 1}
    
    df.round(round_columns).sort_values(['VIDEO_ID', 'CLIP']).to_csv(DATA_PATH, index=False)

    print(df.groupby('VIDEO_ID').count())
    


def clean_id(video_id):
    if not video_id:
        return video_id
    
    video_id = str(video_id)
    arr = video_id.split('/')
    if len(arr) > 1:
        video_id = arr[1]
    
    return video_id

def create_summary(file_name):
    path = LABELS_PATH + file_name + '.csv'
    df = pd.read_csv(path)
    # print(df.head())

    df_check = df[['Input.VIDEO_ID', 'Input.CLIP', 'Answer.sentiment', 'Answer.happiness', 'Answer.sadness', 'Answer.anger', 'Answer.fear', 'Answer.disgust', 'Answer.surprise']]
    # df_check = df_check[df_check['Input.VIDEO_ID'] == '_0efYOjQYRc']

    df_check = df_check.groupby(['Input.VIDEO_ID', 'Input.CLIP']).agg(['count', 'mean', max, min])
    print(df_check.head())
    df_check = df_check.reset_index()
    df_check.columns = COLUMNS

    # df_check[['VIDEO_ID_0', 'VIDEO_ID_1']] = df_check['VIDEO_ID'].str.split('/', extend=True)
    df_check['VIDEO_ID'] = df_check['VIDEO_ID'].astype(str).apply(clean_id)
    print(df_check.head())
    # print(df_check.columns)

    # df_check.to_csv(file_name + '_summary', index=False)

    return df_check


if __name__=='__main__':
    main()
