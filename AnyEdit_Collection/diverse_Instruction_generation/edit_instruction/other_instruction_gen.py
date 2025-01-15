
import argparse
import os
from pathlib import Path
import sys 
path = Path(__file__).resolve().parent.parent.parent / "AnyEdit_Collection"
sys.path.insert(0, str(path))
from cv2.gapi import mask
from segment_anything import build_sam, SamPredictor
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import generate_tags, load_model, box_detection
import json
import random
from tqdm import tqdm
import spacy
import warnings
warnings.filterwarnings("ignore")


def generate_tags(raw_text):
    # generate specific categories in the caption by spacy
    nlp = spacy.load("en_core_web_sm")
    tags = {'nouns': [], 'adj': [], 'verb': []}
    words_list = nlp(raw_text)
    for i in range(len(words_list) - 1):
        token = words_list[i]
        next_token = words_list[i + 1]
        if i < len(words_list) - 2:
            next_next_token = words_list[i + 2]
            if token.pos_ == 'ADJ' and next_next_token.pos_ == 'NOUN' and token.text.lower() not in tags['adj']:
                tags['adj'].append(token.text.lower())
                tags['nouns'].append(' '.join([token.text.lower(), next_token.text.lower(), next_next_token.text.lower()]))
        if token.pos_ == 'ADJ' and next_token.pos_ == 'NOUN' and token.text.lower() not in tags['adj']:
            tags['adj'].append(token.text.lower())
            if ' '.join([token.text.lower(), next_token.text.lower()]) not in tags['nouns']:
                tags['nouns'].append(' '.join([token.text.lower(), next_token.text.lower()]))
        elif token.tag_ == 'VBN' and next_token.pos_ == 'NOUN' and token.text.lower() not in tags['adj']:
            tags['adj'].append(token.text.lower())
            if ' '.join([token.text.lower(), next_token.text.lower()]) not in tags['nouns']:
                tags['nouns'].append(' '.join([token.text.lower(), next_token.text.lower()]))
        elif token.pos_ == 'NOUN' and token.tag_ != 'VBG' and token.text.lower() not in tags['nouns'] and next_token.text.lower() not in ['of']:
            if next_token.pos_ == 'NOUN':
                tags['nouns'].append(' '.join([token.text.lower(), next_token.text.lower()])) # fine-grained for attribute
                # tags['nouns'].append(next_token.text.lower())
            else:
                tags['nouns'].append(token.text.lower())
        elif token.pos_ == 'VERB':
            tags['verb'].append(token.text.lower())
    if words_list[-1].pos_ == 'NOUN' and token.text.lower() not in tags['nouns']:
        tags['nouns'].append(words_list[-1].text.lower())
    if words_list[-1].pos_ == 'VERB' and token.text.lower() not in tags['verb']:
        tags['verb'].append(words_list[-1].lemma_.lower())
    return tags

def count_instruction_generation(source_data, target_path):
    number_list = {'0': 'zero', '1':'one', '2':'two', '3':'three', '4':'four', '5':'five', '6':'six', '7':'seven', '8':'eight', '9':'nine'}
    if source_data=="mscoco":  # image_caption.json
        total_data = json.load(open("./AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input/image_caption.json"))
    elif source_data=='llava595':  # new_llava595_1.json
        total_data = json.load(open("./AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input/new_llava595.json"))
    valid_data = []
    for key in tqdm(total_data):
        if source_data == "mscoco":
            captions = total_data[key]['captions']  # use random.choice to get only one caption to speed up
            image_file = total_data[key]['image_file']
        else:
            captions = [key['caption']]
            image_file = key['image_file'] 
        data = dict()
        for caption in captions:
            spacy_results = generate_tags(caption)
            for noun in spacy_results['nouns']:
                cur_num = caption.split(noun)[0].strip(' ').split(' ')[-1]
                for k in number_list.keys():
                    if k == cur_num or number_list[k] == cur_num:
                        if int(k) <= 1:
                            continue
                        data['edited object'] = noun
                        remove_number = random.randint(1,int(k))
                        data['remove_number'] = remove_number
                        left_number = int(k) - remove_number
                        if str(remove_number) in number_list:
                            remove_number = number_list[str(remove_number)]
                        if str(left_number) in number_list:
                            left_number = number_list[str(left_number)]
                        if cur_num in number_list:
                            cur_num = number_list[cur_num]
                        remove_word = random.choice(['Remove', 'Erase', 'Elimate'])
                        left_word = random.choice(['Turn', 'Make', 'Let'])
                        lower_left_word = left_word.lower()
                        count_instructions = [
                            f'{remove_word} {remove_number} of the {noun} in the image',
                            f'{left_word} the total number of the {noun} from {cur_num} to {left_number}',
                            f'{remove_word} {remove_number} of the {noun} to {lower_left_word} the total number from {cur_num} to {left_number}',
                        ]
                        count_instruction = random.choice(count_instructions)
                        data['edit'] = count_instruction
                        break
                if 'edited object' in data and data['edited object'] is not None:
                    break
            if 'edited object' in data and data['edited object'] is not None:
                data['image_file'] = image_file
                data['edit_type'] = 'counting'
                data['input'] = caption
                data['output'] = caption.replace(f' {cur_num} {noun}', f' {left_number} {noun}')
                valid_data.append(data)
                break
    print(len(valid_data))
    with open(f'{target_path}/{source_data}_counting_filtered.jsonl', 'w') as f:
        for instruction in valid_data:
            f.write(json.dumps(instruction) + '\n')

def resize_movement(source_data, target_path, type):
    total_data = [json.loads(f) for f in open(os.path.join(target_path, f"{source_data}_remove_-100.jsonl"))]
    valid_data = []
    for data in total_data:
        data['output'] = None
        data['edit_type'] = type
        if type == 'movement':
            data['edit'] = "Move  " + ' '.join(data['edit'].split(' ')[1:])
        else:
            data['edit'] = "Resize  " + ' '.join(data['edit'].split(' ')[1:])
        valid_data.append(data)
    print(len(valid_data))
    with open(f'{target_path}/{source_data}_{type}_filtered.jsonl', 'w') as f:
        for instruction in valid_data:
            f.write(json.dumps(instruction) + '\n')

def visual_reference_instruction_generation(source_data, target_path):
    total_data = [json.loads(f) for f in open(os.path.join(target_path, f"{source_data}_replace_-100.jsonl"))]
    valid_data = []
    visual_reference_concepts = dict()
    for reference_image in [json.loads(f) for f in open('AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input/visual_reference_concept_mapping.jsonl')]:
        for key in reference_image:
            visual_reference_concepts[key] = reference_image[key]['animals'][0]
    for data in total_data:
        replace_word = random.choice(['replace', 'change', 'alter'])
        edit_object = data['edited object']
        data['edit'] = f'{replace_word} the {edit_object} to the [V*]'
        data['edited object'] = edit_object
        visual_input = random.choice(os.listdir('./AnyEdit/data/visual_reference/visual_input'))
        data['visual_input'] = os.path.join('./AnyEdit/data/visual_reference/visual_input', visual_input)
        data['ref_object'] = visual_reference_concepts[visual_input]["animals"][0]
        total_words = data['input'].split(' ')
        for i, word in enumerate(total_words):
            if word == edit_object:
                total_words[i] = "[V*]"
                break
        data['output'] = ' '.join(total_words)
        data['edit_type'] = 'visual_reference'
        if 'new object' in data:
            del data['new object']
        valid_data.append(data)
        print(len(valid_data))
    with open(f'{target_path}/{source_data}_visual_reference_filtered.jsonl', 'w') as f:
        for instruction in valid_data:
            f.write(json.dumps(instruction) + '\n')
            
import os
import json
import re
from tqdm import tqdm
import spacy

# 替换原始文本中的带连字符的词组
def adjust_text_for_context(text):
    # 用正则表达式查找并替换带连字符的形容词/名词等
    text = re.sub(r'(\w+)-(\w+)', r'\1\2', text)
    return text

def relation_instruction_generation(det_model, source_data, target_path, image_path):
    if source_data=="mscoco":  # image_caption.json
        total_data = json.load(open("/home1/yqf/ssd/AnyEdit/CaptionsGenerator/edit_instruction/input/image_caption.json"))
        # total_data = json.load(open("./AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input/image_caption.json"))
    elif source_data=='llava595':  # new_llava595_1.json
        total_data = json.load(open("./AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/input/new_llava595.json"))
    valid_data = []
    for key in tqdm(total_data):
        if source_data == "mscoco":
            caption = random.choice(total_data[key]['captions'])  # use random.choice to get only one caption to speed up
            image_file = total_data[key]['image_file']
        else:
            caption = key['caption']
            image_file = key['image_file'] 
        data = dict()
        spacy_results = generate_tags(caption)
        init_image_path = os.path.join(image_path, image_file)
        valid_obj = []
        for obj in spacy_results['nouns']:
            mask_bbox = box_detection(det_model, init_image_path, caption, target_object=obj)
            if mask_bbox is None:
                continue
            if (mask_bbox[2] - mask_bbox[0]) / 512 * (mask_bbox[3] - mask_bbox[1]) / 512 < 0.1:
                x = (mask_bbox[2] - mask_bbox[0]) / 2
                y = (mask_bbox[3] - mask_bbox[1]) / 2
                if x < 256 and y < 256: # up-left
                    valid_obj.append((obj, (-1,-1)))
                elif x > 256 and y < 256: # up-right
                    valid_obj.append((obj, (1,-1)))
                elif x < 256 and y > 256: # below-left
                    valid_obj.append((obj, (-1,1)))
                else: # below-right
                    valid_obj.append((obj, (1,1)))
        if len(valid_obj) >= 2:
            if random.random() < 0.5:
                swap_obj = random.sample(valid_obj, 2)
                data['edited object'] = swap_obj[0][0]
                data['target object'] = swap_obj[1][0]
                opearation = random.choice(['swap', 'move'])
                if opearation == 'swap':  
                    data['direction'] = 'swap'
                    random_word = random.choice(['with', 'and'])
                    data['edit'] = f'Swap {swap_obj[0][0]} {random_word} {swap_obj[1][0]}'
                else:
                    random_word = random.choice(['Move', 'Place'])
                    if swap_obj[1][1] == (-1,-1): # 左上角
                        direction = random.choice(['right', 'down'])
                    elif swap_obj[1][1] == (1,-1): # 右上角
                        direction = random.choice(['left', 'down'])
                    elif swap_obj[1][1] == (-1,1): # 左下角 
                        direction = random.choice(['right', 'upper'])
                    else:  # 右下角
                        direction = random.choice(['left', 'upper'])
                    edited_object = swap_obj[0][0]
                    target_object = swap_obj[1][0]
                    if direction == 'left':   
                        instruction = f'{random_word} the {edited_object} to left of the {target_object}'
                    elif direction == 'right':
                        instruction = f'{random_word} the {edited_object} to right of the {target_object}'
                    elif direction == 'down':
                        instruction = f'{random_word} the {edited_object} under the {target_object}'
                    else:
                        instruction = f'{random_word} the {edited_object} on the top of the {target_object}'  
                    
                    data['direction'] = direction
                    data['edit'] = instruction
            else:               
                # non target object
                move_obj = random.choice(valid_obj)
                random_word = random.choice(['Move', 'Place'])
                if move_obj[1] == (-1,-1): # 左上角
                    direction = random.choice(['right', 'down'])
                elif move_obj[1] == (1,-1): # 右上角
                    direction = random.choice(['left', 'down'])
                elif move_obj[1] == (-1,1): # 左下角 
                    direction = random.choice(['right', 'upper'])
                else:  # 右下角
                    direction = random.choice(['left', 'upper'])
                
                
                if direction == 'left':   
                    instruction = random.choice([f'{random_word} the {move_obj[0]} to left', f'{random_word} the {move_obj[0]} from right to left'])
                elif direction == 'right':
                    instruction = random.choice([f'{random_word} the {move_obj[0]} to right', f'{random_word} the {move_obj[0]} from left to right'])
                elif direction == 'down':
                    instruction = random.choice([f'{random_word} the {move_obj[0]} to down', f'{random_word} the {move_obj[0]} from top to down'])
                else:
                    instruction = random.choice([f'{random_word} the {move_obj[0]} to upper', f'{random_word} the {move_obj[0]} from bottom to upper'])                                        
                
                data['edited object'] = move_obj[0]
                data['direction'] = direction
                data['edit'] = instruction
        if 'edit' in data and 'edited object' in data:
            data['input'] = caption
            data['image_file'] = image_file
            data['edit_type'] = 'relation'
            data['output'] = None
            valid_data.append(data)      
    print(len(valid_data))
    with open(f'{target_path}/{source_data}_relation_prefilter.json', 'w') as results_file:
        json.dump(valid_data, results_file, indent=4)

def parse_args():
    parser = argparse.ArgumentParser(description="Other Instruction Generation Pipeline")
    parser.add_argument("--source-data", default='mscoco', choices=['mscoco', 'llava595'])
    parser.add_argument("--target-path", required=True, help="path to save the instruction")
    parser.add_argument("--edit-type", type=str, required=True, help="specific editing types")
    parser.add_argument(
        "--options",
        nargs="+",
        help="in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    source_data = args.source_data
    target_path = args.target_path
    edit_type = args.edit_type
    
    if edit_type == 'counting':
        count_instruction_generation(source_data, target_path)
    elif edit_type == 'resize': 
        # we transfer the instruction of remove to movement/resize to guarantee the effectiveness of edited objects
        resize_movement(source_data, target_path, type="resize")
    elif edit_type == 'movement':
        resize_movement(source_data, target_path, type="movement")    
    elif edit_type == 'visual_reference':
        # we transfer the instruction of replace to visual_reference to guarantee the effectiveness of edited objects
        visual_reference_instruction_generation(source_data, target_path)
    # 先运行 AnyEdit_Collection/diverse_Instruction_generation/edit_instruction/visual_reference_crawler.py 获取visual reference对应的reference images，并保存到对应的visual input文件夹中
    elif edit_type == 'relation':
        config_file = 'GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py' 
        grounded_checkpoint = './checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth'  
        sam_checkpoint = './checkpoints/foundation_models/sam_vit_h_4b8939.pth'
        device = 'cuda'
        image_path = './AnyEdit/original_datasets/coco/train2014/'

        det_model = load_model(config_file, grounded_checkpoint, device=device)
        relation_instruction_generation(det_model, source_data, target_path, image_path)
     