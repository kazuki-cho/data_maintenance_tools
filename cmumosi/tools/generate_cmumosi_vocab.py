import os
import sys
import pickle

import numpy as np

import pytorch_pretrained_bert.tokenization  as tokenization

INPUT = '../../../self_study/cmumosi_full.pkl'
OUTPUT = '../../../self_study/cmumosi_full_vocab.pkl'
VOCAB_PATH = '../../../Cross-Modal-BERT/Cross-Modal-BERT-master/pre-trained BERT/vocab.txt'


def main():
    full_data = pickle.load(open(INPUT, 'rb'))
    vocab = tokenization.load_vocab(VOCAB_PATH)
    ret = {}
    remain_word = None
    remain_audio = None
    remain_word_interval = None
    before_video_id = None
    for segment_name in full_data.keys():
        # if segment_name != '_dI--eQ6qVU_8':
        #     continue
        print(segment_name)
        segment = full_data[segment_name]
        new_segment = {}

        video_id = segment['video_id']
        text = segment['text']
        words = segment['words']
        audios = segment['audio']
        videos = segment['video']
        word_intervals = segment['word_intervals']

        if remain_word and remain_word_interval and remain_audio and remain_vodeo and video_id == before_video_id:
            words = np.concatenate([np.array(remain_word), words])
            audios = np.concatenate([np.array(remain_audio), audios])
            videos = np.concatenate([np.array(remain_vodeo), videos])
            deff_start = remain_word_interval[0][0]
            remain_word_interval = np.array(remain_word_interval)
            remain_word_interval = remain_word_interval - np.full_like(remain_word_interval.shape, deff_start)
            deff_end = remain_word_interval[-1][1]
            word_intervals = word_intervals + np.full_like(word_intervals.shape, deff_end)
            word_intervals = np.concatenate([np.array(remain_word_interval) ,word_intervals])

        remain_word = None
        remain_audio = None
        remain_vodeo = None
        remain_word_interval = None
        before_video_id = video_id
        # print(words)
        if len(words) > 0 and words[0][0].decode() == 'sp':
            words = words[1:]
            audios = audios[1:]
            videos = videos[1:]
            word_intervals = word_intervals[1:]
        if len(words) > 0 and words[-1][0].decode() == 'sp':
            words = words[:-1]
            audios = audios[:-1]
            videos = videos[:-1]
            word_intervals = word_intervals[:-1]

        text_words = text.strip().split(' ')

        # print('text_words: ', len(text_words))
        # print('words: ', len(words))
        # print('audios: ', len(audios))
        # print('word_intervals: ', len(word_intervals))
        # print(text_words)
        # print(words)
        tmp_text_words, tmp_words, tmp_audios, tmp_videos, tmp_word_intervals = transcripts_align(text_words, words, audios, videos, word_intervals)
        # print('tmp_text_words: ', tmp_text_words)
        # print('tmp_words: ', tmp_words)


        text_length = len(tmp_text_words)
        words_length = len(tmp_words)
        if words_length > text_length:
            tmp_words = tmp_words[:text_length]
            tmp_audios = tmp_audios[:text_length]
            tmp_videos = tmp_videos[:text_length]
            tmp_word_intervals = tmp_word_intervals[:text_length]

            a = text_length - words_length
            remain_word = [ [x.encode()] for x in tmp_words[a:]]
            remain_audio = tmp_audios[a:]
            remain_vodeo = tmp_videos[a:]
            remain_word_interval = tmp_word_intervals[a:]

        # print('tmp_text_words: ', len(tmp_text_words))
        # print('tmp_words: ', len(tmp_words))
        # print('tmp_audios: ', len(tmp_audios))
        # print('tmp_word_intervals: ', len(tmp_word_intervals))
        output_text_words, output_words, output_audios, output_videos, autput_word_intervals = vocab_align(tmp_text_words, tmp_words, tmp_audios, tmp_videos, tmp_word_intervals, vocab)
        # print('output_text_words: ', output_text_words)
        # print('output_words: ', output_words)
        # print('output_audios: ', output_audios)
        # print('autput_word_intervals: ', autput_word_intervals)
        # print('output_text_words: ', len(output_text_words))
        # print('output_words: ', len(output_words))
        # print('output_audios: ', len(output_audios))
        # print('autput_word_intervals: ', len(autput_word_intervals))

        new_segment['video_id'] = segment['video_id']
        new_segment['label'] = segment['label']
        new_segment['text'] = ' '.join(output_text_words)
        new_segment['words'] = np.array(output_words)
        new_segment['audio'] = np.array(output_audios)
        new_segment['video'] = np.array(output_videos)
        new_segment['word_intervals'] = np.array(autput_word_intervals)
        new_segment['intervals'] = segment['intervals']

        ret[segment_name] = new_segment


    with open(OUTPUT, mode='wb') as f:
        pickle.dump(ret, f)


def transcripts_align(text_words, words, audios, videos, word_intervals):
    new_text_words = []
    new_words = []
    new_audios = []
    new_videos = []
    new_word_intervals = []

    zero_audio = np.zeros(len(audios[0]))
    zero_video = np.zeros(len(videos[0]))

    l = len(text_words)
    m = len(words)
    # print('l: ', l)
    # print('m: ', m)
    p = 0
    
    for i in range(l):
        K = [j for j in range(m) if j >= p]
        # print('K: ', K)
        text_word = text_words[i]
        inter_text = []
        inter_word_arr = []
        inter_audio_arr = []
        inter_video_arr = []
        inter_intervals_arr = []
        is_match = False
        for k in K:
            word = words[k][0].decode()
            audio = audios[k]
            video = videos[k]
            intervals = word_intervals[k]
            # print('text_word.lower(): ', text_word.lower())
            # print('word.lower(): ', word.lower())
            if text_word.lower() == word.lower():
                new_text_words.extend(inter_text)
                new_text_words.append(text_word)
                new_words.extend(inter_word_arr)
                new_words.append(word)
                new_audios.extend(inter_audio_arr)
                new_audios.append(audio)
                new_videos.extend(inter_video_arr)
                new_videos.append(video)
                new_word_intervals.extend(inter_intervals_arr)
                new_word_intervals.append(intervals)
                p = k + 1
                is_match = True

                # print('p:', p)
                # print('new_text_words:', ' '.join(new_text_words))
                # print('new_words:', new_words)
                break
            elif word != 'sp':
                inter_text.append('')
                inter_word_arr.append(word)
                inter_audio_arr.append(audio)
                inter_video_arr.append(video)
                inter_intervals_arr.append(intervals)
        
        if not is_match:
            new_text_words.append(text_word)
            new_words.append('[PAD]')
            new_audios.append(zero_audio)
            new_videos.append(zero_video)
            if len(new_word_intervals) > 0:
                end_timstamp = new_word_intervals[-1][1]
            else:
                end_timstamp = 0.00
            new_word_intervals.append(np.array([end_timstamp, end_timstamp]))

    if p < m:
        new_words.extend([x[0].decode() for x in words[p - m:]])
        new_audios.extend([x for x in audios[p - m:]])
        new_videos.extend([x for x in videos[p - m:]])
        new_word_intervals.extend([x for x in word_intervals[p - m:]])

    return new_text_words, new_words, new_audios, new_videos, new_word_intervals
        




def vocab_align(text_words, words, audios, videos, word_intervals, vocab):

    output_text_words = []
    output_words = []
    output_audios = []
    output_videos = []
    output_word_intervals = []
    
    for i in range(len(words)):
        text_word = text_words[i]
        token = words[i]
        audio = audios[i]
        video = videos[i]
        word_interval = word_intervals[i]

        zero_audio = np.zeros(len(audio))
        zero_video = np.zeros(len(video))
        zero_intervales = np.array([word_interval[1], word_interval[1]])

        chars = list(token)

        is_bad = False
        start = 0
        sub_text = [text_word]
        sub_tokens = []
        sub_audios = [audio]
        sub_videos = [video]
        sub_word_intervals = [word_interval]
        while start < len(chars):
            end = len(chars)
            cur_substr = None
            while start < end:
                substr = "".join(chars[start:end])
                if start > 0:
                    substr = "##" + substr
                if substr in vocab:
                    cur_substr = substr
                    break
                end -= 1
            if cur_substr is None:
                is_bad = True
                break
            sub_text.append('')
            sub_tokens.append(cur_substr)
            sub_audios.append(zero_audio)
            sub_videos.append(zero_video)
            sub_word_intervals.append(zero_intervales)
            start = end

        if is_bad:
            output_text_words.append(text_word)
            output_words.append(token)
            output_audios.append(audio)
            output_videos.append(video)
            output_word_intervals.append(word_interval)
        else:
            sub_text = sub_text[:-1]
            sub_audios = sub_audios[:-1]
            sub_videos = sub_videos[:-1]
            sub_word_intervals = sub_word_intervals[:-1]
            output_text_words.extend(sub_text)
            output_words.extend(sub_tokens)
            output_audios.extend(sub_audios)
            output_videos.extend(sub_videos)
            output_word_intervals.extend(sub_word_intervals)
    return output_text_words, output_words, output_audios, output_videos, output_word_intervals




if __name__ == '__main__':
    main()