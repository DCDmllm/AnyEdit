import json
import re
import os
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', type=str, default='./implicit/gen_sample_history/text_gen.json', help="json file of text samples.")
    parser.add_argument('--output_json', type=str, default='./implicit/gen_sample_history/text_gen_full.json', help="json file of text samples.")
    opt = parser.parse_args()

    FULL_SAMPLES_NUM = 0

    if os.path.exists(opt.input_json):
        with open(opt.input_json) as f:
            input_dict = json.load(f)
    else:
        print('Error: define_json not exist')

    if os.path.exists(opt.output_json):
        with open(opt.output_json) as f:
            json_datas = json.load(f)
        sample_numbers = [int(key.replace("sample", "")) for key in json_datas.keys() if key.startswith("sample")]
        FULL_SAMPLES_NUM = max(sample_numbers)
    else:
        json_datas = {}
    
    for key in input_dict.keys():
        anwser = input_dict[key]
        texts = re.findall(r'\d+\.?\s*(.*)', anwser)
        for txt in texts:
            txt_list = txt.split('; ')
            try:
                txt_dict = {"original_caption": txt_list[0], "instuction": txt_list[1], "target_cation": txt_list[2], 
                            "key_words": txt_list[3]}
            except:
                continue
            FULL_SAMPLES_NUM += 1
            json_datas[f"sample{FULL_SAMPLES_NUM}"] = txt_dict

    with open(opt.output_json, 'w') as f:
        json.dump(json_datas, f, indent=4)
            
