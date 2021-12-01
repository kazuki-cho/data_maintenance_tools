import pickle


MOSEI_DATA = '../data/cmumosei_jp_data.pkl'
OUTPUT = '../data/pair_segments.pkl'


def pairing(video_id, segment_no, segment_list):
    pair_no = 0
    segment_no_list = segment_list[str(video_id)]
    for i in segment_no_list:
        if int(segment_no) < int(i):
            if pair_no == 0:
                pair_no = i
            elif int(pair_no) > int(i):
                pair_no = i
    return pair_no == 0 ? None : pair_no



def main():

    mosei_data = pickle.load(open(MOSEI_DATA, 'rb'))

    segment_list = {}
    for segment_name, features in mosei_data.items():
        video_id = features.get('video_id')
        segment_no = features.get('segment_no')
        if not segment_list.get(str(video_id)):
            segment_list[str(video_id)] = []
        
        segment_list[str(video_id)].append(segment_no)
    
    pair_dict = {}
    for segment_name, features in mosei_data.items():
        video_id = features.get('video_id')
        segment_no = features.get('segment_no')
        pair_segment_no = pairing(video_id, segment_no, segment_list)
        if not pair_segment_no:
            continue
        pair_segment_name = str(video_id) + '_' + str(pair_segment_no)
        pair_dict[segment_name] = pair_segment_name
    

    with open(OUTPUT, mode='wb') as f:
        pickle.dump(pair_dict, f)

if __name__ == '__main__':
    main()