import os
import sys
from diffusers import DiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionPipeline, AutoPipelineForText2Image
import torch
from PIL import Image
import random
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import warnings
import json
from tqdm import tqdm
import argparse
import spacy
from omegaconf import OmegaConf
from pathlib import Path
lama_path = Path(__file__).resolve().parent.parent.parent / "AnyEdit_Collection/other_modules/lama"
sys.path.insert(0, str(lama_path))
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.utils import move_to_device
import yaml
from tool import is_human_variant, return_parameters, maskgeneration
from termcolor import cprint
warnings.filterwarnings("ignore")

def lama_builder(
    config_p: str,
    ckpt_p: str,
    device="cuda"
):
    predict_config = OmegaConf.load(config_p)
    predict_config.model.path = ckpt_p
    # device = torch.device(predict_config.device)
    device = torch.device(device)

    train_config_path = os.path.join(
        predict_config.model.path, 'config.yaml')

    with open(train_config_path, 'r') as f:
        train_config = OmegaConf.create(yaml.safe_load(f))

    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(
        predict_config.model.path, 'models',
        predict_config.model.checkpoint
    )
    model = load_checkpoint(
        train_config, checkpoint_path, strict=False, map_location='cpu')
    model.to(device)
    model.freeze()
    return model

def inpaint_img_with_lama(
        img: np.ndarray,
        mask: np.ndarray,
        model,
        mod=8,
        device="cuda"
):
    assert len(mask.shape) == 2
    if np.max(mask) == 1:
        mask = mask * 255
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()

    batch = {}
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)
    batch['mask'] = mask[None, None]
    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1

    batch = model(batch)
    cur_res = batch["inpainted"][0].permute(1, 2, 0)
    cur_res = cur_res.detach().cpu().numpy()

    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]

    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res


def remove_noise(mask, min_area=100):
    """
    去除 mask 中的小面积噪声点。

    参数:
    mask: 输入的二值化物体 mask，1 表示物体区域，0 表示背景。
    min_area: 最小的连通组件面积，小于这个面积的组件将被移除。

    返回:
    denoised_mask: 去除噪声后的 mask。
    """
    # 找到所有连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    
    # 创建一个新的 mask 只保留大于最小面积的连通组件
    denoised_mask = np.zeros_like(mask)
    
    for i in range(1, num_labels):  # 跳过第一个标签（背景）
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            denoised_mask[labels == i] = 1

    return denoised_mask

def check_occlusion_by_segmentation(mask, threshold=0.3, min_area=100):
    """
    检查一个物体的 mask 是否被遮挡，通过计算不相连部分的面积比例。
    
    参数:
    mask: 输入的二值化物体 mask，1 表示物体区域，0 表示背景。
    threshold: 用于判断遮挡的面积比例阈值，默认为 0.3。
    min_area: 最小的连通组件面积，小于这个面积的组件将被移除。

    返回:
    bool: 如果有不相连部分占总面积的比例超过阈值，返回 True，否则返回 False。
    """
    # 先去除噪声和干扰点
    denoised_mask = remove_noise(mask, min_area=min_area)
    
    # 找到所有连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(denoised_mask.astype(np.uint8), connectivity=8)
    
    # 检查是否有足够的连通组件
    if num_labels <= 1:
        # 只有背景或没有足够的连通组件
        print("Object is not occluded: insufficient components.")
        return False
    
    # 计算每个连通组件的面积
    total_area = np.sum(denoised_mask)
    largest_component_area = np.max(stats[1:, cv2.CC_STAT_AREA])  # 跳过第一个标签，因为它是背景
    
    # 检查最大连通组件面积是否小于总面积的 70% （即超过 30% 的其他部分）
    if largest_component_area / total_area < 1 - threshold:
        print(f"Object is occluded: largest component area is {largest_component_area}, total area is {total_area}.")
        return True
    else:
        print("Object is not occluded.")
        return False

def masks_overlap(mask1, mask2):
    if mask1 is None or mask2 is None:
        return False
    
    # 读取灰度图像
    if isinstance(mask1, str) and isinstance(mask2, str):
        mask1 = cv2.imread(mask1, cv2.IMREAD_GRAYSCALE)
        mask2 = cv2.imread(mask2, cv2.IMREAD_GRAYSCALE)
    
    # 计算两个掩码的交集
    intersected = cv2.bitwise_and(mask1, mask2)
    
    # 检查交集是否有任何非零像素
    overlap_exists = np.any(intersected > 0)
    
    return overlap_exists

def bboxes_overlap(bbox_pil1, bbox_pil2: list):
    for foreground in bbox_pil2:
        if masks_overlap(bbox_pil1, foreground):
            return True
    return False

def load_sam_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    image_pil = image_pil.resize((512, 512), resample=Image.Resampling.LANCZOS)
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image_pil, image
def load_sam_image_from_Image(image_pil):
    # load image
    transform = T.Compose(
        [
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    image, _ = transform(image_pil, None)  # 3, h, w
    return image
def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model
def generate_tags(raw_text):
    # generate specific categories in the caption by spacy
    nlp = spacy.load("en_core_web_sm")
    tags = {'nouns':[], 'adj':[]}
    words_list = nlp(raw_text)
    for i in range(len(words_list)-1):
        token = words_list[i]
        next_token = words_list[i+1]
        if token.pos_ == 'ADJ' and next_token.pos_ == 'NOUN' and token.text.lower() not in tags['adj']:
            tags['adj'].append(token.text.lower())
        elif token.pos_ == 'NOUN' and token.tag_ != 'VBG' and token.text.lower() not in tags['nouns'] and next_token.text.lower() not in ['of']:
            if next_token.pos_ == 'NOUN':
                tags['nouns'].append(' '.join([token.text.lower(), next_token.text.lower()])) # fine-grained for attribute
                # tags['nouns'].append(next_token.text.lower())
            else:
                tags['nouns'].append(token.text.lower())
    if words_list[-1].pos_ == 'NOUN' and token.text.lower() not in tags['nouns']:
        tags['nouns'].append(words_list[-1].text.lower())
    return tags

def sdxl_pipe(base, prompt):
    # Define how many steps and what % of steps to be run on each experts (80/20) here
    # n_steps = 50
    # high_noise_frac = 0.8
    # prompt = prompt.strip('.')
    # prompt = f'{prompt}, high resolution, realistic'
    # # run both experts
    # image = base(
    #     prompt=prompt,
    #     num_inference_steps=n_steps,
    #     denoising_end=high_noise_frac,
    #     output_type="latent",
    # ).images
    # image = refiner(
    #     prompt=prompt,
    #     num_inference_steps=n_steps,
    #     denoising_start=high_noise_frac,
    #     image=image,
    # ).images[0]
    image = base(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    return image.resize((512,512), resample=Image.Resampling.LANCZOS)

def sd_pipe_generation(sd_model, prompt):
    image = sd_model(prompt).images[0]  
        
    return image.resize((512,512), resample=Image.Resampling.LANCZOS)

def cropimage2image(original_image, mask, background_image, scale):
    original_image_array = np.array(original_image)
    mask_array = np.array(mask)
    background_image_array = np.array(background_image)

    coords = np.argwhere(mask_array > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)

    extracted_content = original_image_array[min_y:max_y+1, min_x:max_x+1]
    extracted_content_mask=mask_array[min_y:max_y+1, min_x:max_x+1]

    original_w = max_x - min_x
    original_h = max_y - min_y
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)
    resized_content = Image.fromarray(extracted_content).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))
    resized_content_mask = Image.fromarray(extracted_content_mask).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))

    result_array = background_image_array
    resized_content_array=np.array(resized_content)
    resized_content_mask_array=np.array(resized_content_mask)
    result_array[min_y:min_y+scaled_h, min_x:min_x+scaled_w][resized_content_mask_array > 0] = resized_content_array[resized_content_mask_array > 0]

    result_image = Image.fromarray(result_array)
    return result_image

def resize_cropimage2image(original_image, mask, background_image, x, y, scale):
    original_image_array = np.array(original_image)
    mask_array = np.array(mask)
    background_image_array = np.array(background_image)

    coords = np.argwhere(mask_array > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)

    extracted_content = original_image_array[min_y:max_y+1, min_x:max_x+1]
    extracted_content_mask=mask_array[min_y:max_y+1, min_x:max_x+1]

    original_w = max_x - min_x
    original_h = max_y - min_y
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)
    resized_content = Image.fromarray(extracted_content).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))
    resized_content_mask = Image.fromarray(extracted_content_mask).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))
    result_array = background_image_array
    resized_content_array=np.array(resized_content)
    resized_content_mask_array=np.array(resized_content_mask)
    result_array[y:y+scaled_h, x:x+scaled_w][resized_content_mask_array > 0] = resized_content_array[resized_content_mask_array > 0]

    result_image = Image.fromarray(result_array)
    return result_image

def load_tool_model(args):
    config_file = args.groundingdino_config_file #'GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py' 
    grounded_checkpoint = args.groundingdino_ckpt #'checkpoints/groundingdino_swinb_cogcoor.pth'  
    sam_checkpoint = args.sam_ckpt #'checkpoints/sam_vit_h_4b8939.pth'

    device = 'cuda'

    det_model = load_model(config_file, grounded_checkpoint, device=device)
    sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))
    
    
    lama_pipe = lama_builder("./lama/configs/prediction/default.yaml", "./checkpoints/foundation_models/big-lama", device=device)
    
    return lama_pipe, det_model, sam_model


def adjust_scale_if_necessary(image_size, x, y, mask_size, scale):
    scaled_w = int(mask_size[0] * scale)
    scaled_h = int(mask_size[1] * scale)
    
    # 检查是否越界
    if x + scaled_w > image_size[0] or y + scaled_h > image_size[1]:
        # 如果越界了，将scale改为缩小的比例
        scale = 0.7
        scaled_w = int(mask_size[0] * scale)
        scaled_h = int(mask_size[1] * scale)
    
    return scale, scaled_w, scaled_h
def relation_change(det_model, sam_model, inpaint_pipe, output, edited_object, direction, instruction, 
                    init_image_path, edited_image_path_pre, instruction_type, target_object=None):
    invalid_words = ['people', 'man', 'woman', 'boy', 'girl', 'person', 'adult', 'child', 'hand', 'foot']
    if edited_object in invalid_words:
        return []
    # generate init image: x
    original_size = Image.open(init_image_path).convert("RGB").size
    total_objects = generate_tags(output)['nouns']
    if edited_object not in total_objects:
        total_objects.append(edited_object)
        total_objects.append(edited_object.split(' ')[-1])
    if target_object is not None and target_object not in total_objects:
        total_objects.append(target_object)

    prompt = ', '.join(total_objects)
    mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.jpg'

    mask_pil, image_pil, mask_bbox, union_region = maskgeneration(det_model, sam_model, init_image_path, prompt, target_object=edited_object, box_threshold=0.2, text_threshold=0.2)

    if mask_pil is None:
        print(f'Skip: mask for {edited_object} is none, image refers to {init_image_path}')
        return None
                
    mask_array = np.array(mask_pil)
    mask_array = remove_noise(mask_array)

    
    coords = np.argwhere(mask_array > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)
    mask_size = [max_x-min_x, max_y-min_y]
    
    if target_object is not None:
        target_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt, box_threshold=0.2, text_threshold=0.2, target_object=target_object)
        if target_mask_pil is not None:
            target_mask_array = np.array(target_mask_pil)
            target_mask_pil.save(edited_image_path_pre.replace("edited_img", "mask") + f'_target.jpg')
            target_coords = np.argwhere(target_mask_array > 0)
            target_min_y, target_min_x = np.min(target_coords, axis=0)
            target_max_y, target_max_x = np.max(target_coords, axis=0)
            target_mask_size = [target_max_x-target_min_x, target_max_y-target_min_y]
        else:
            target_object = None
    total_edited_images = []
    if target_object is not None and direction == 'swap':
        mask_pil.save(mask_path)
        cv2_mask_image = cv2.imread(mask_path)
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((15, 15), np.uint8))
        maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
        edited_image = inpaint_img_with_lama(np.array(image_pil), np.array(maskimage_dilate), inpaint_pipe)
        edited_image = Image.fromarray(edited_image)
        os.remove(mask_path)
        edit_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, edited_image, prompt, target_object=edited_object)

        target_mask_pil.save(mask_path)
        cv2_mask_image = cv2.imread(mask_path)
        maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((15, 15), np.uint8))
        maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
        edited_image = inpaint_img_with_lama(np.array(edited_image), np.array(maskimage_dilate), inpaint_pipe)
        edited_image = Image.fromarray(edited_image)
        os.remove(mask_path)
        target_edit_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, edited_image, prompt, target_object=target_object)
        
        if edit_mask_pil is None and target_edit_mask_pil is None:
            valid = True
        else:
            if edit_mask_pil is not None:
                intersection_mask = np.asarray(mask_pil) & np.asarray(edit_mask_pil)
                num_inter_ones = np.sum(intersection_mask == 1)
                num_original_ones = np.sum(np.asarray(mask_pil)== 1)
                percentage = (num_inter_ones / num_original_ones)
            if target_edit_mask_pil is not None:
                target_intersection_mask = np.asarray(target_mask_pil) & np.asarray(target_edit_mask_pil)
                target_num_inter_ones = np.sum(target_intersection_mask == 1)
                target_num_original_ones = np.sum(np.asarray(target_mask_pil)== 1)
                target_percentage = (target_num_inter_ones / target_num_original_ones)    
            
            if (edit_mask_pil is None or percentage < 0.2) and (target_edit_mask_pil is None or target_percentage < 0.2):
                valid = True
        if valid:
            scale = 1
            # first step
            x = max(1, int(0.5*(int(target_max_x)+int(target_min_x))-0.5*(max_x-min_x)))
            y = max(1, int(0.5*(int(target_max_y)+int(target_min_y))-0.5*(max_y-min_y)))
            
            image_size = image_pil.size
            scale, scaled_w, scaled_h = adjust_scale_if_necessary(image_size, x, y, mask_size, scale)
            result_image_pil = resize_cropimage2image(image_pil, mask_pil, edited_image, x, y, scale)
            if result_image_pil is None:
                print('Skip: resize_cropimage2image fails')
                return None
            
            target_x = max(1, int(0.5*(int(max_x)+int(min_x))-0.5*(int(target_max_x)-int(target_min_x))))
            target_y = max(1, int(0.5*(int(max_y)+int(min_y))-0.5*(int(target_max_y)-int(target_min_y))))
            
            image_size = image_pil.size
            scale, scaled_w, scaled_h = adjust_scale_if_necessary(image_size, target_x, target_y, target_mask_size, scale)
            target_result_image_pil = resize_cropimage2image(image_pil, target_mask_pil, result_image_pil, target_x, target_y, scale)
            if result_image_pil is None:
                print('Skip: resize_cropimage2image fails')
                return None
            edited_image = target_result_image_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
            image_pil = image_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
            mask_pil = mask_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
            cur_edited_image_path = edited_image_path_pre+f'.jpg'
        

            cur_data = {"edit": instruction, "edit object": edited_object,
                    "output": output, "input": None, "edit_type": "relation", "image_file": cur_edited_image_path.replace("edited_img", "input_img"), "edited_file": cur_edited_image_path}
            image_pil.save(cur_edited_image_path.replace("edited_img", "input_img"))
            mask_pil.save(mask_path)
            edited_image.save(cur_edited_image_path)
            total_edited_images.append(cur_data)
            return total_edited_images
        else:
            return None
    elif direction == 'swap':
        return None
    
    mask_pil.save(mask_path)
    cv2_mask_image = cv2.imread(mask_path)
    maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((15, 15), np.uint8))
    maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
    edited_image = inpaint_img_with_lama(np.array(image_pil), np.array(maskimage_dilate), inpaint_pipe)
    edited_image = Image.fromarray(edited_image)
    # edited_image.save(edited_image_path_pre+f'_test.jpg')
    os.remove(mask_path)
    edit_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, edited_image, prompt, target_object=edited_object)
    valid = False
    if edit_mask_pil is None:
        valid = True
    else:
        intersection_mask = np.asarray(mask_pil) & np.asarray(edit_mask_pil)
        num_inter_ones = np.sum(intersection_mask == 1)
        num_original_ones = np.sum(np.asarray(mask_pil)== 1)
        percentage = (num_inter_ones / num_original_ones)
        if percentage < 0.2:
            valid = True
    if valid:
        scale = 1
        if target_object is not None:
            if direction == 'left':
                delta_x = random.randint(int(1-min_x), max(int(1-min_x)+1, int(target_min_x-1-max_x)))
                delta_y = 0
            elif direction == 'right':
                delta_x = random.randint(min(int(target_max_x+1-min_x), image_pil.size[0]-1-int(max_x)-1), image_pil.size[0]-1-int(max_x))
                delta_y = 0
            elif direction == 'upper':
                delta_x = random.randint(-10,10)
                delta_y = random.randint(int(1-min_y), max(int(1-min_y)+1, int(target_min_y-1-max_y)))
            elif direction == 'down':
                delta_x = random.randint(-10,10)
                delta_y = random.randint(min(int(target_max_y+1-min_y), image_pil.size[1]-1-int(max_y)-1), image_pil.size[1]-1-int(max_y))   
            elif direction == 'inside':
                delta_x = int(0.5*(int(target_max_x)+int(target_min_x))-0.5*(max_x+min_x))
                delta_y = int(0.5*(int(target_max_x)+int(target_min_x))-0.5*(max_x+min_x))
            
        else:
            if direction == 'left':
                delta_x = random.randint(min(-51,int(1-min_x)), -50)
                delta_y = 0
            elif direction == 'right':
                delta_x = random.randint(50, max(51, image_pil.size[0]-1-int(max_x)))
                delta_y = 0
            elif direction == 'upper':
                delta_x = random.randint(-10,10)
                delta_y = random.randint(min(-51,int(1-min_y)), -50)
            elif direction == 'down':
                delta_x = random.randint(-10,10)
                delta_y = random.randint(50, max(51, image_pil.size[1]-1-int(max_y)))                     

        x = max(1, min_x + delta_x)
            # y = min(1, max_y - int(mask_size[1]*scale) + delta_y)
        y = max(1, min_y + delta_y)
        
        image_size = image_pil.size
        scale, scaled_w, scaled_h = adjust_scale_if_necessary(image_size, x, y, mask_size, scale)
        # mask_pil = Image.fromarray(cv2.cvtColor(cv2.GaussianBlur(cv2_mask_image, (15, 15), 0),cv2.COLOR_BGR2GRAY))
        print(image_pil.size, mask_pil.size, edited_image.size)
        result_image_pil = resize_cropimage2image(image_pil, mask_pil, edited_image, x, y, scale)
        if result_image_pil is None:
            print('Skip: resize_cropimage2image fails')
            return None
        edited_image = result_image_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
        image_pil = image_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
        mask_pil = mask_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
        i = 0
        cur_edited_image_path = edited_image_path_pre+f'.jpg'
    

        cur_data = {"edit": instruction, "edit object": edited_object,
                "output": output, "input": None, "edit_type": "relation", "image_file": cur_edited_image_path.replace("edited_img", "input_img"), "edited_file": cur_edited_image_path}
        image_pil.save(cur_edited_image_path.replace("edited_img", "input_img"))
        mask_pil.save(mask_path)
        edited_image.save(cur_edited_image_path)
        total_edited_images.append(cur_data)
    return total_edited_images



def parse_args():
    parser = argparse.ArgumentParser(description="Editing Pipeline")
    parser.add_argument("--groundingdino-config-file", required=True, help="path to groundingdino config file.")
    parser.add_argument("--groundingdino-ckpt", required=True, help="path to groundingdino model file")
    parser.add_argument("--sam-ckpt", required=True, help="path to sam model file")
    parser.add_argument("--instruction-path", required=True, help="path to instruction path")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--start-idx", type=str, default='-1', help="specify the experiment id.")
    parser.add_argument("--end-idx", type=str, default='-1', help="specify the experiment id.")
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

    try:
        with open(args.json_path) as f:
            edit_instruction_data = json.load(f)
    except:
        with open(args.json_path, 'r') as f:
            edit_instruction_data = [json.loads(line) for line in f]
    instruction_path = f'{args.instruction_path}/{args.instruction_type}'
    final_edit_results_path = f'{instruction_path}/final_edit_results_{args.start_idx}_{args.end_idx}.json'
    edit_success_path = f'{instruction_path}/edit_success_{args.start_idx}_{args.end_idx}.json'
    edit_failure_path = f'{instruction_path}/edit_failure_{args.start_idx}_{args.end_idx}.json'

    final_edited_results = []
    success_data = []
    failure_data = []

    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/input_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/mask', exist_ok=True)
    
    
    lama_pipe, det_model, sam_model = load_tool_model(args)
    valid_number = 0

    iter = 0
    if args.start_idx != '-1' and args.end_idx != '-1':
        print(args.start_idx, args.end_idx)
        st_idx = int(args.start_idx)
        ed_idx = int(args.end_idx)
        print(st_idx, ed_idx)
        edit_instruction_data = edit_instruction_data[st_idx:ed_idx]
    print(len(edit_instruction_data))
    for data in tqdm(edit_instruction_data):
        instruction_type = data['edit_type']
        if instruction_type == 'relation' and ('edited object' not in data or data['edited object'] is None):
            print(f'Skip: no edited object in data')
            continue
        input, edited_object, target_object, direction, output, init_image_path, instruction, edited_image_path_pre = \
            return_parameters(data, args, init_image_root=args.image_path)
        if target_object in ['left', 'right', 'upper', 'down']:
            target_object = None
        if 'ball' in edited_object:
            edited_object = 'ball'
        instruction_data = relation_change(det_model, sam_model, lama_pipe, output, edited_object, direction, instruction, 
                    init_image_path, edited_image_path_pre, instruction_type, target_object)

        if instruction_data is not None and len(instruction_data) > 0:
            valid_number += 1
            final_edited_results.extend(instruction_data)
            success_data.append(data)
        else:
            failure_data.append(data)
        

    print(f"valid editing insturction data: {valid_number}")
    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_{args.start_idx}_{args.end_idx}.json', 'w') as results_file:
        json.dump(final_edited_results, results_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_success_{args.start_idx}_{args.end_idx}.json', 'w') as success_file:
        json.dump(success_data, success_file, indent=4)
    with open(f'{args.instruction_path}/{args.instruction_type}/edit_failure_{args.start_idx}_{args.end_idx}.json', 'w') as failure_file:
        json.dump(failure_data, failure_file, indent=4)