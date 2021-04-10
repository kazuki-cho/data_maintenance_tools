import os
import glob
import pandas as pd
DATA_PATH = '../../../CMU_MOSEI_Raw/Transcript/Full/Combined'
OUTPUT_PATH = '../../../CMU_MOSEI_Raw/Transcript/Full/TextOnly'

def main():
    input_paths = glob.glob(DATA_PATH + '/*')
    # full_transcripts = []

    for input_path in input_paths:
        file_name = os.path.splitext(os.path.basename(input_path))[0]
        video_id = file_name.replace('-user', '').replace('.en', '')
        # if video_id != 't80DGfWJ3fI':
        #     continue
        print('video_id: ', video_id)
        print('input_path: ', input_path)
        with open(input_path, 'r') as files:
            lines = files.readlines()
        line_status = 'nomal'
        start_line = 0
        end_line = len(lines)
        if lines[0].strip() == 'WEBVTT':
            start_line = 5
        
        full_text = ''
        for i in range(start_line, end_line):
            line = lines[i].strip()
            # print('line: ', i)
            # print('text: ', line)
            # print('line_status: ', line_status)
            if not line:
                line_status = 'brank'
                continue
            elif line_status == 'brank':
                line_status = 'nomal'
                continue
            else:
                full_text = full_text + ' ' + line
        
        print('full_text: ', full_text)
        

        with open(OUTPUT_PATH + '/' + video_id + '.textonly', 'w') as f:
            f.writelines(full_text.strip())
    #     full_transcripts.append({'video_id': video_id, 'text': full_text})
    # df = pd.DataFrame(full_transcripts)
    # df.sort_values(['video_id']).to_csv(OUTPUT_PATH, index=False)



if __name__ == '__main__':
    main()