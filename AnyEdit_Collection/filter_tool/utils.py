import os

import cv2
from PIL import Image
import numpy as np
from tqdm import tqdm
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import re
import json
import torch
import warnings
from transformers import AutoModel, AutoTokenizer
warnings.filterwarnings('ignore')

device1 = 'cuda:0'
device2 = 'cuda:1'

# Clip score
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', local_files_only=True).to(device1)
clip_processor = CLIPProcessor.from_pretrained('laion/CLIP-ViT-H-14-laion2B-s32B-b79K', local_files_only=True)
 
def get_clip_score(image, text):
    images = [image]
    inputs = clip_processor(text=text, images=images, return_tensors='pt')
    if inputs['input_ids'].shape[1] > 77:
        text = text.split('. ')[0] + '.'
        inputs = clip_processor(text=text, images=images, return_tensors='pt')
        if inputs['input_ids'].shape[1] > 77:
            text = text.split(', ')[0] + '.'
            inputs = clip_processor(text=text, images=images, return_tensors='pt')
            if inputs['input_ids'].shape[1] > 77:
                inputs['input_ids'] = inputs['input_ids'][:,:77]
                inputs['attention_mask'] = inputs['attention_mask'][:,:77]
    inputs = {name: tensor.to(device1) for name, tensor in inputs.items()} 
    outputs = clip_model(**inputs) 
    logits_per_image = outputs.logits_per_image
    torch.cuda.synchronize(device1) 
    return logits_per_image.item() / 100

# OCR_model
ocr_tokenizer = AutoTokenizer.from_pretrained('./GOT_OCR2', trust_remote_code=True)
ocr_model = AutoModel.from_pretrained('./GOT_OCR2', trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda', use_safetensors=True, pad_token_id=ocr_tokenizer.eos_token_id).eval().cuda() 

def ocr_detection(original_image_file, edited_image_file):
    res_ori = ocr_model.chat(ocr_tokenizer, original_image_file, ocr_type='ocr')
    res_edit = ocr_model.chat(ocr_tokenizer, edited_image_file, ocr_type='ocr')
    return res_ori, res_edit

# Blip2   
blip2_processor = Blip2Processor.from_pretrained('Salesforce/blip2-flan-t5-xl', local_files_only=True)
blip2_model = Blip2ForConditionalGeneration.from_pretrained('Salesforce/blip2-flan-t5-xl', local_files_only=True).to(device2)

def blip2_color(image, inst, image_path=''):
    inputs = blip2_processor(images=image, return_tensors='pt').to('cuda:1')
    generated_ids = blip2_model.generate(**inputs)
    generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    # objects = words[index+1]
    objects = inst['edit object']
    words = re.findall('\w+', inst['edit'])
    color = words[-1]
    prompt = f"Question: Is the color of {objects} close to {color}? Answer:"
    inputs = blip2_processor(images=image, text=prompt, return_tensors='pt').to(device2)
    generated_ids = blip2_model.generate(**inputs)
    generated_text = blip2_processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0].strip()

    affirmative_words = ['yes', 'true', 'right', 'correct']
    generated_words = generated_text.split(' ')
    torch.cuda.synchronize(device2)  # 同步 device2
    if any(word in generated_words for word in affirmative_words):
        return True
    else:
        return False

# Blip2   
def blip2_background(image, inst, image_path=''):
    inputs = blip2_processor(images=image, return_tensors='pt').to(device2)
    generated_ids = blip2_model.generate(**inputs)
    generated_text = blip2_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    
    background = inst['new background']
    prompt = f"Question: Is the background of this image similar to {background}? Answer:"
    inputs = blip2_processor(images=image, text=prompt, return_tensors='pt').to(device2)
    generated_ids = blip2_model.generate(**inputs)
    generated_text = blip2_processor.batch_decode(generated_ids.cpu(), skip_special_tokens=True)[0].strip()

    affirmative_words = ['yes', 'true', 'right', 'correct']
    generated_words = generated_text.split(' ')
    torch.cuda.synchronize(device2)  # 同步 device2
    if any(word in generated_words for word in affirmative_words):
        return True
    else:
        return False

def normalized(a, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

# L1 score
def get_L1_distance(origin_img, edited_img):
    origin_img_rgb = origin_img.convert('RGB')
    edited_img_rgb = edited_img.convert('RGB')
    origin_img_rgb = np.array(origin_img_rgb)
    edited_img_rgb = np.array(edited_img_rgb)
    l1_distance = np.sum(np.abs(origin_img_rgb - edited_img_rgb))
    num_pixels = origin_img_rgb.shape[0] * origin_img_rgb.shape[1] * origin_img_rgb.shape[2]
    normalized_l1_distance = l1_distance / num_pixels / 255
    return normalized_l1_distance

# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity

def get_cosine_similarity(origin_img, edited_img):
    origin_img = np.array(origin_img)
    edited_img = np.array(edited_img)
    if len(origin_img.shape) == 3:
        origin_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)
    if len(edited_img.shape) == 3:
        edited_img = cv2.cvtColor(edited_img, cv2.COLOR_BGR2GRAY)
        
    if origin_img.shape != edited_img.shape:
        edited_img = cv2.resize(edited_img, (origin_img.shape[1], origin_img.shape[0]))
        
    origin_img_flat = origin_img.flatten().astype(np.float32)
    edited_img_flat = edited_img.flatten().astype(np.float32)
    origin_tensor = torch.from_numpy(origin_img_flat).view(1, -1).to(device2)
    edited_tensor = torch.from_numpy(edited_img_flat).view(1, -1).to(device2)
    
    cosine_sim = torch.nn.functional.cosine_similarity(origin_tensor, edited_tensor, dim=1)
    cosine_sim = cosine_sim.cpu().numpy()    
    torch.cuda.synchronize(device2)  # 同步 device2
    return cosine_sim

# object detection
from segment_anything import build_sam, SamPredictor 

import sys
sys.path.insert(0, 'AnyEdit')
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import load_model
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

config_file = './GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py' 
grounded_checkpoint = './checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth' 
sam_checkpoint = './checkpoints/foundation_models/sam_vit_h_4b8939.pth'

det_model = load_model(config_file, grounded_checkpoint, device=device1)
sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device1))

def load_sam_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB")  # load image
    # image_pil = image_pil.resize((512, 512))
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
def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device1)
    image = image.to(device1)
    with torch.no_grad():
        torch.cuda.synchronize(device2)  # 同步 device2
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
    logits.shape[0]

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
    logits_filt.shape[0]

    # get phrase
    tokenlizer = model.tokenizer
    tokenized = tokenlizer(caption)
    # build pred
    pred_phrases = []
    for logit, box in zip(logits_filt, boxes_filt):
        pred_phrase = get_phrases_from_posmap(logit > text_threshold, tokenized, tokenlizer)
        if with_logits:
            pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
        else:
            pred_phrases.append(pred_phrase)

    return boxes_filt, pred_phrases

def object_detection(image_path, det_prompt, edit_mask_path, box_threshold=0.25, text_threshold=0.25, target_object=None):
    # load image
    if isinstance(image_path, str):
        image_pil, image = load_sam_image(image_path)
    else:
        image_pil = image_path.convert("RGB")  # load image
        image_pil = image_pil.resize((512, 512))
        image = load_sam_image_from_Image(image_pil)

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(det_model, image, det_prompt, box_threshold, text_threshold)
    image = cv2.cvtColor(np.array(image_pil),cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_model.set_image(image)

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    valid_boxes_filt = []

    if target_object is not None:
        for box, phrase in zip(boxes_filt, pred_phrases):
            if phrase.split('(')[0] == target_object or phrase.split('(')[0] in target_object:
                valid_boxes_filt.append(box)
        if len(valid_boxes_filt) > 0:
            boxes_filt = torch.stack(valid_boxes_filt)
        else:
            torch.cuda.synchronize(device1) 
            return False

    boxes_filt = boxes_filt.cpu()
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device1)

    if len(boxes_filt) == 0:
        torch.cuda.synchronize(device1) 
        return False
    masks, _, _ = sam_model.predict_torch(
        point_coords = None,
        point_labels = None,
        boxes = transformed_boxes.to(device1),
        multimask_output = False,
    )
    mask = masks[0][0].cpu().numpy() # just preserve the max mask, else keep the top-k masks (k=remove_number)
    
    # detecting whether the mask is valid
    coords = np.argwhere(mask > 0)
    if isinstance(coords, np.ndarray) and coords.size == 0:
        torch.cuda.synchronize(device1) 
        return False
    mask = mask.astype(np.float32)  # 将 mask 转换为 float32 类型
    mask = cv2.resize(mask, (512, 512))    
    edit_mask = Image.open(edit_mask_path)
    # 如果尺寸不同则都修改为512*512
    if edit_mask.size != (512, 512):
        edit_mask = edit_mask.resize((512, 512))
        
    intersection_mask = np.asarray(mask) * np.asarray(edit_mask)
    num_inter_ones = np.sum(intersection_mask == 1)
    num_original_ones = np.sum(np.asarray(mask)== 1)
    percentage = num_inter_ones / num_original_ones
    torch.cuda.synchronize(device1) 
    return percentage > 0.2

# Directional CLIP
import clip
clipViT_B_model, clipViT_B_preprocess = clip.load("ViT-B/32")
clipViT_B_model = clipViT_B_model.to(device1)
import torch.nn.functional as F

def get_directional_clip(inst, original_image, edited_image):
    image_a = clipViT_B_preprocess(original_image).unsqueeze(0).to(device1)
    image_b = clipViT_B_preprocess(edited_image).unsqueeze(0).to(device1)
    image_a_features = clipViT_B_model.encode_image(image_a)
    image_b_features = clipViT_B_model.encode_image(image_b)

    text_a = clip.tokenize(inst['input']).to(device1)
    text_a_features = clipViT_B_model.encode_text(text_a)
    text_b = clip.tokenize(inst['output']).to(device1)
    text_b_features = clipViT_B_model.encode_text(text_b)
    
    image_diff = image_b_features - image_a_features
    text_diff = text_b_features - text_a_features
    image_diff_normalized = F.normalize(image_diff, p=2, dim=-1)
    text_diff_normalized = F.normalize(text_diff, p=2, dim=-1)

    similarity = torch.nn.functional.cosine_similarity(image_diff_normalized, text_diff_normalized, dim=1).item()
    return similarity

# 贴图
def cropimage2image(edited_image, background_image, mask_path, scale=1.0):
    if not os.path.exists(mask_path):
        print("Not exist mask path")
        return None
    cv2_mask_image = cv2.imread(mask_path)
    # more mask region for remove artifact
    maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((30, 30), np.uint8)) 
    maskimage_dilate = cv2.GaussianBlur(maskimage_dilate, (5, 5), 0) 
    mask = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY)).resize(edited_image.size)
    
    edited_image_array = np.array(edited_image)
    mask_array = np.array(mask)
    background_image_array = np.array(background_image)

    coords = np.argwhere(mask_array > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)

    extracted_content = edited_image_array[min_y:max_y+1, min_x:max_x+1]
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