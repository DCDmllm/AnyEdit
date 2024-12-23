import os
import argparse
import re
import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
import json
import torch
import warnings
warnings.filterwarnings('ignore')

from utils import get_clip_score, get_L1_distance, object_detection, blip2_background, blip2_color, cropimage2image, get_directional_clip, ocr_detection

def add_postfilter(edited_image, inst, edited_image_path, edit_mask_path):
    if get_clip_score(edited_image, inst['output']) > 0.2:
        return object_detection(edited_image_path, inst['edit object'], edit_mask_path)
    return False

def replace_postfilter(original_image, edited_image, inst, edited_image_path, edit_mask_path):
    if get_clip_score(edited_image, inst['output']) > 0.2:
        if get_directional_clip(inst, original_image, edited_image) > 0.08:
            return object_detection(edited_image_path, inst['new object'], edit_mask_path)
    return False

def remove_postfilter(original_image, edited_image, inst, edited_image_path, edit_mask_path, second_edited_image_path=None, second_edited_image=None, second_original_image_path=None):
    if second_edited_image != None:
        if get_clip_score(edited_image, inst['output']) >= get_clip_score(second_edited_image, inst['output']):
            edited_image = edited_image 
        else:
            edited_image = second_edited_image
            edited_image_path = second_edited_image_path
            inst['edited_file'] = second_edited_image_path
            inst['image_file'] = second_original_image_path
            edit_mask_path = edited_image_path.replace('edited_img', 'mask').replace('_0.png', '.png')
    if get_clip_score(edited_image, inst['output']) > 0.15:
        return not object_detection(edited_image_path, inst['edit object'], edit_mask_path)
    return False

def action_change_postfilter(original_image, edited_image, inst):
    if get_clip_score(edited_image, inst['output']) > 0.3:
        return get_directional_clip(inst, original_image, edited_image) > 0.05

def appearance_alter_postfilter(original_image, edited_image, inst):
    if get_clip_score(edited_image, inst['output']) > 0.25:
        if get_L1_distance(original_image, edited_image) > 0.3:
            return get_directional_clip(inst, original_image, edited_image) > 0.06
    return False

def tone_transfer_postfilter(original_image, edited_image, inst):
    if get_clip_score(edited_image, inst['output']) > 0.25:
        l1_score = get_L1_distance(original_image, edited_image)
        return l1_score > 0.2 and l1_score < 0.8

def background_change_postfilter(original_image, edited_image, inst, edited_image_path):
    if get_clip_score(edited_image, inst['output']) > 0.15:
        if blip2_background(edited_image, inst, edited_image_path):
            l1_score = get_L1_distance(original_image, edited_image)
            return l1_score > 0.2 and l1_score < 0.9
    return False

def color_alter_postfilter(original_image, edited_image, inst, edited_image_path):
    if get_clip_score(edited_image, inst['output']) > 0.2:
        if get_L1_distance(original_image, edited_image) > 0.3:
            return blip2_color(edited_image, inst, edited_image_path)
    return False

# for textual postfilter

def textual_postfilter(original_image_file, edited_image_file, inst):
    
    words = re.findall(r"'(.*?)'", inst['edit'])
    if len(words) != 2:
        return False
    else:
        tar_ori = words[0]
        tar_edit = words[1]
        res_ori, res_edit = ocr_detection(original_image_file, edited_image_file)
        return tar_ori in res_ori and tar_edit in res_edit

def post_filter(inst, data_root):
    edited_image_path = os.path.join(data_root, inst['edited_file'])
    init_image_path = os.path.join(data_root, inst['image_file'])
    

    if edited_image_path.find('_1') != -1 or edited_image_path.find('_2') != -1:
        return False
    try:
        original_image = Image.open(init_image_path).convert("RGB")
        edited_image = Image.open(edited_image_path).convert("RGB")
    except:
        print('image file type not valid')
        return False
        
    # mask path
    edit_mask_path = edited_image_path.replace('edited_img', 'mask').replace('_0.jpg', '.jpg')

    if instruction_type == 'add':
        return add_postfilter(edited_image, inst, edited_image_path, edit_mask_path)
    if instruction_type == 'replace': 
        return replace_postfilter(original_image, edited_image, inst, edited_image_path, edit_mask_path)
    elif instruction_type == 'remove':
        return remove_postfilter(original_image, edited_image, inst, edited_image_path, edit_mask_path)
    elif instruction_type == 'action_change':
        return action_change_postfilter(original_image, edited_image, inst)
    elif instruction_type == 'appearance_alter':
        return appearance_alter_postfilter(original_image, edited_image, inst)
    elif instruction_type == 'tone_transfer':
        return tone_transfer_postfilter(original_image, edited_image, inst)
    elif instruction_type == 'background_change':
        return background_change_postfilter(original_image, edited_image, inst, edited_image_path)
    elif instruction_type == 'color_alter' or instruction_type == 'color_alter_ip2p':
        return color_alter_postfilter(original_image, edited_image, inst, edited_image_path)
    elif instruction_type == 'textual_change':
        return textual_postfilter(init_image_path, edited_image_path, inst)
    else:
        print('instruction type not found')
        return False
    
if __name__ == '__main__':
    # 加载指令
    # instruction_types = ['add', 'remove', 'replace', 'color_alter', 'tone_transfer', 'appearance_alter', 'background_change', 'action_change']
    parser = argparse.ArgumentParser(description="post_filter")

    parser.add_argument("--instruction-type", help="specify the instruction type.",
                        default='action_change')
    args = parser.parse_args()
    instruction_types = [args.instruction_type]
    for instruction_type in instruction_types:
        # 从筛选出来的指令当中筛选
        try:
            insts = json.load(open(f'anyedit_datasets/{instruction_type}/final_edit_results_-1_-1.json', 'r'))
            success_data = json.load(open(f'anyedit_datasets/{instruction_type}/edit_success_-1_-1.json', 'r'))
        except:
            insts = json.load(open(f'anyedit_datasets/{instruction_type}/final_edit_results_0.json', 'r'))
            success_data = json.load(open(f'anyedit_datasets/{instruction_type}/edit_success_0.json', 'r'))
        print(f'***************************{instruction_type}*******************************')
        success_data_dict = dict()
        for data in success_data:
            key = data['image_file'].split('_')[-1].split('.')[0]
            success_data_dict[key] = data
        selected_inst = []
        fail_inst  = []
        print(instruction_type, len(insts))
        n = 0
        print(len(insts))
        print(len(success_data_dict))

        for inst in tqdm(insts):
            if instruction_type == 'replace':
                key = inst['image_file'].split('_')[-1].split('.')[0]
                inst['new object'] = success_data_dict[key]['new object']
            elif instruction_type == 'background_change':
                key = inst['image_file'].split('_')[-1].split('.')[0]
                inst['new background'] = success_data_dict[key]['new background']
            elif instruction_type == 'color_alter_ip2p':
                key = inst['image_file'].split('_')[-1].split('.')[0]
                inst['edit object'] = success_data_dict[key]['edited object']
                if 'color_alter_ip2p' not in inst['edited_file']:
                    inst['edited_file'] = inst['edited_file'].replace('color_alter', 'color_alter_ip2p')
            if inst['input'] is None:
                key = inst['image_file'].split('_')[-1].split('.')[0]
                inst['input'] = success_data_dict[key]['input']
            if inst['output'] is None:
                key = inst['image_file'].split('_')[-1].split('.')[0]
                inst['output'] = success_data_dict[key]['output']
            if post_filter(inst, data_root=''):
                n += 1
                edited_image_path = inst['edited_file']     
                original_image_path = inst['image_file']

                if instruction_type == 'remove' or instruction_type == 'background_change':

                    edit_mask_path = inst['edited_file'].replace('edited_img', 'mask').replace('_0.png', '.png')
                    
                    original_image = Image.open(inst['image_file']).convert("RGB")
                    edited_image = Image.open(inst['edited_file']).convert("RGB")

                    masked_edited_image = cropimage2image(edited_image, original_image, edit_mask_path, 1.0)
                    os.makedirs(f"anyedit_datasets/{instruction_type}/update_edited_img", exist_ok=True)
                    update_edited_image_path = edited_image_path.replace('edited_img', 'update_edited_img').replace('/final_anyedit_datasets', '/anyedit_datasets')
                    if masked_edited_image is not None:
                        masked_edited_image.save(update_edited_image_path)
                        inst['edited_file'] = update_edited_image_path
                selected_inst.append(inst)
            else:
                fail_inst.append(inst)
        print(n)
        json.dump(selected_inst, open(f'final_output_postfilter/{instruction_type}_postfilter.json', 'w'))
        json.dump(fail_inst, open(f'final_output_postfilter/{instruction_type}_failed_postfilter.json', 'w'))
        print(f'Select {len(selected_inst)} from {len(selected_inst)+len(fail_inst)}')