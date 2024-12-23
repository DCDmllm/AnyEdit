import torch
import torchvision
import spacy
import cv2
import os
from PIL import Image
import numpy as np
import nltk
from nltk.corpus import wordnet as wn
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
# nltk.download('wordnet')

def prepare_output_dir(output_root_dir):
    os.makedirs(output_root_dir, exist_ok=True)
    
    input_image_dir = os.path.join(output_root_dir, 'input_img')
    output_edited_image_dir = os.path.join(output_root_dir, 'edited_img')
    output_init_dir = os.path.join(output_root_dir, 'init')
    output_mask_dir = os.path.join(output_root_dir, 'mask')
    os.makedirs(input_image_dir, exist_ok=True)
    os.makedirs(output_edited_image_dir, exist_ok=True)
    os.makedirs(output_init_dir, exist_ok=True)
    os.makedirs(output_mask_dir, exist_ok=True)


def return_parameters(instruction_data, args, init_image_root=None):
    edit_type = instruction_data['edit_type']
    if init_image_root is None:
        init_image_root = f'{args.instruction_path}/{args.instruction_type}/input_img'
    input = instruction_data['input']
    output = instruction_data['output']
    init_image_path = f"{init_image_root}/{instruction_data['image_file']}"
    instruction = instruction_data['edit']

    # the value of ['edited_file'] is similar like ['image_file'], but they are in different root path
    if instruction_data['image_file'] is not None:
        edited_image_path = f'{args.instruction_path}/{args.instruction_type}' + '/edited_img/' + \
                                init_image_path.split('.')[-2].split('/')[-1]
    else:
        edited_image_path = f'{args.instruction_path}/{args.instruction_type}' + '/edited_img/' + \
                            instruction_data['edited_file'].split('.')[-2]

    if edit_type in ['color_alter', 'remove', 'appearance_alter', 'material_alter', 'add', 'resize', 'movement']:
        edited_object = instruction_data['edited object']
        return input, edited_object, output, init_image_path, instruction, edited_image_path
    elif edit_type == 'replace':
        edited_object = instruction_data['edited object']
        new_object = instruction_data['new object']
        return input, edited_object, new_object, output, init_image_path, instruction, edited_image_path
    elif edit_type == 'background_change' or edit_type == 'tone_transfer':
        return input, output, init_image_path, instruction, edited_image_path
    elif edit_type == 'counting':
        edited_object = instruction_data['edited object']
        remove_number = instruction_data['remove_number']
        return input, edited_object, output, init_image_path, instruction, edited_image_path, remove_number
    elif edit_type == 'visual_reference':
        edited_object = instruction_data['edited object']
        visual_image_path = instruction_data['visual_input']  # new object
        ref_object = instruction_data['ref_object'] # the category of the new object (from the jellycat)
        return input, edited_object, ref_object, visual_image_path, output, init_image_path, instruction, edited_image_path
    else:
        raise NotImplementedError

def generate_tags(raw_text):
    # generate specific categories in the caption by spacy
    nlp = spacy.load("en_core_web_sm")
    tags = {'nouns': [], 'adj': [], 'verb': []}
    words_list = nlp(raw_text)
    for i in range(len(words_list) - 1):
        token = words_list[i]
        next_token = words_list[i + 1]
        if token.pos_ == 'ADJ' and next_token.pos_ == 'NOUN' and token.text.lower() not in tags['adj']:
            tags['adj'].append(token.text.lower())
            tags['nouns'].append(' '.join([token.text.lower(), next_token.text.lower()]))
        elif token.pos_ == 'NOUN' and token.tag_ != 'VBG' and token.text.lower() not in tags['nouns'] and next_token.text.lower() not in ['of']:
            if next_token.pos_ == 'NOUN':
                tags['nouns'].append(' '.join([token.text.lower(), next_token.text.lower()])) # fine-grained for attribute
            else:
                tags['nouns'].append(token.text.lower())
        elif token.pos_ == 'VERB':
            tags['verb'].append(token.text.lower())
    if words_list[-1].pos_ == 'NOUN' and words_list[-1].text.lower() not in tags['nouns']:
        tags['nouns'].append(words_list[-1].text.lower())
    if words_list[-1].pos_ == 'VERB' and words_list[-1].text.lower() not in tags['verb']:
        tags['verb'].append(words_list[-1].lemma_.lower())
    return tags

def load_sam_image(image_path):
    # load image
    image_pil = Image.open(image_path).convert("RGB").resize((512, 512), resample=Image.Resampling.LANCZOS)  # load image
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

def get_grounding_output(model, image, caption, box_threshold, text_threshold, with_logits=True, device="cuda"):
    caption = caption.lower()
    caption = caption.strip()
    if not caption.endswith("."):
        caption = caption + "."
    model = model.to(device)
    image = image.to(device)
    with torch.no_grad():
        outputs = model(image[None], captions=[caption])
    logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
    boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)

    # filter output
    logits_filt = logits.clone()
    boxes_filt = boxes.clone()
    filt_mask = logits_filt.max(dim=1)[0] > box_threshold
    logits_filt = logits_filt[filt_mask]  # num_filt, 256
    boxes_filt = boxes_filt[filt_mask]  # num_filt, 4

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

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cpu")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def is_human_variant(word):
    human_suffixes = ["er", "or", "ist", "ian", "ee"]
    # Check if the word ends with any of the common human-related suffixes
    if any(word.endswith(suffix) for suffix in human_suffixes):
        return True
    return False


def maskgeneration(det_model, sam_model, image_path, det_prompt, mask_mode='max', box_threshold=0.25,
                   text_threshold=0.25, target_object=None, device="cuda"):
    # load image
    if isinstance(image_path, str):
        image_pil, image = load_sam_image(image_path)
    else:
        image_pil = image_path.convert("RGB")  # load image
        image_pil = image_pil.resize((512, 512), resample=Image.Resampling.LANCZOS)
        image = load_sam_image_from_Image(image_pil)

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        det_model, image, det_prompt, box_threshold, text_threshold, device=device
    )
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sam_model.set_image(image)
    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    valid_boxes_filt = []
    valid_boxes_score = []
    if target_object is not None:
        if isinstance(target_object, str):
            for box, phrase in zip(boxes_filt, pred_phrases):
                if phrase.split('(')[0] == target_object:
                    valid_boxes_filt.append(box)
                    valid_boxes_score.append(float(phrase.split('(')[1].strip(')')))
            if len(valid_boxes_filt)==0:
                for box, phrase in zip(boxes_filt, pred_phrases):
                    if phrase.split('(')[0] in target_object.split(' ') or target_object.split(' ')[-1] in phrase.split('(')[0].split(' '):
                        valid_boxes_filt.append(box)
                        valid_boxes_score.append(float(phrase.split('(')[1].strip(')')))
            if len(valid_boxes_filt) > 0:
                boxes_filt = torch.stack(valid_boxes_filt)
                boxes_score = torch.Tensor(valid_boxes_score)
            else:
                return None, image_pil, None, None
        elif isinstance(target_object, list):
            for obj in target_object:
                for box, phrase in zip(boxes_filt, pred_phrases):
                    if phrase.split('(')[0] == obj:
                        valid_boxes_filt.append(box)
                        valid_boxes_score.append(float(phrase.split('(')[1].strip(')')))
            for obj in target_object:
                if len(valid_boxes_filt)==0:
                    for box, phrase in zip(boxes_filt, pred_phrases):
                        if phrase.split('(')[0] in obj.split(' ') or obj.split(' ')[-1] in phrase.split('(')[0].split(' '):
                            valid_boxes_filt.append(box)
                            valid_boxes_score.append(float(phrase.split('(')[1].strip(')')))
            if len(valid_boxes_filt) > 0:
                boxes_filt = torch.stack(valid_boxes_filt)
                boxes_score = torch.Tensor(valid_boxes_score)
            else:
                return None, image_pil, None, None
        box_nms_idx = torchvision.ops.nms(boxes_filt, boxes_score, iou_threshold=0.5).numpy().tolist() 
        boxes_filt = boxes_filt[box_nms_idx]
    boxes_filt = boxes_filt.cpu()
    transformed_boxes = sam_model.transform.apply_boxes_torch(boxes_filt, image.shape[:2]).to(device)
    if len(boxes_filt) == 0:
        return None, image_pil, None, None
    union_region = ((boxes_filt[0][2] - boxes_filt[0][0]).item() / 512) * (
                (boxes_filt[0][3] - boxes_filt[0][1]).item() / 512)
    masks, _, _ = sam_model.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes.to(device),
        multimask_output=False,
    )
    # masks: [1, 1, 512, 512]

    if mask_mode == 'merge':
        masks = torch.sum(masks, dim=0).unsqueeze(0)
        masks = torch.where(masks > 0, True, False)
    elif mask_mode == 'count':  # preserve all masks in the image
        # masks = torch.sum(masks[:-1,:,:,:], dim=0).unsqueeze(0)
        masks = torch.where(masks > 0, True, False)
        return masks, Image.fromarray(image), None, union_region
    mask = masks[0][0].cpu().numpy()  # max or merge, just preserve the max mask, else keep the top-k masks (k=remove_number)

    # detecting whether the mask is valid
    coords = np.argwhere(mask > 0)
    if isinstance(coords, np.ndarray) and coords.size == 0:
        return None, image_pil, None, None
    # box_mask generation
    bbox = boxes_filt[0].numpy()
    if mask_mode == 'merge':
        box_mask = np.zeros((H, W), dtype=np.uint8)
        for bbox in boxes_filt:
            bbox = bbox.cpu().numpy()  # to numpy
            # merge the bboxes
            box_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255
    else:
        box_mask = np.zeros((H, W))
        box_mask = box_mask.astype(np.uint8)
        box_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255

    mask_pil = Image.fromarray(mask)
    image_pil = Image.fromarray(image)
    mask_bbox = Image.fromarray(box_mask)
    return mask_pil, image_pil, mask_bbox, union_region

def union_generation(det_model, image_path, det_prompt, box_threshold=0.25, text_threshold=0.25, target_object=None, device="cuda"):

    # load image
    if isinstance(image_path, str):
        image_pil, image = load_sam_image(image_path)
    else:
        image_pil = image_path.convert("RGB")  # load image
        image_pil = image_pil.resize((512, 512))
        image = load_sam_image_from_Image(image_pil)

    # run grounding dino model
    boxes_filt, pred_phrases = get_grounding_output(
        det_model, image, det_prompt, box_threshold, text_threshold, device=device
    )

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
            return 0.0

    boxes_filt = boxes_filt.cpu()
    if len(boxes_filt) == 0:
        return 0.0
    union_region = ((boxes_filt[0][2]-boxes_filt[0][0]).item()/512) * ((boxes_filt[0][3]-boxes_filt[0][1]).item()/512)
    return union_region

def img4outpainting(det_model, original_image, caption, box_threshold=0.25, text_threshold=0.25, scale=1.0):
    image_pil = original_image.convert("RGB")  # load image
    image = load_sam_image_from_Image(image_pil)
    device = 'cuda'
    # run grounding dino model
    total_objects = generate_tags(caption)['nouns']
    det_prompt = ', '.join(total_objects)
    boxes_filt, pred_phrases = get_grounding_output(
        det_model, image, det_prompt, box_threshold, text_threshold, device=device
    )

    size = image_pil.size
    H, W = size[1], size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    box_mask = None
    for i in range(boxes_filt.size(0)):
        union_region = ((boxes_filt[i][2]-boxes_filt[i][0]).item()/W) * ((boxes_filt[i][3]-boxes_filt[i][1]).item()/H)
        if union_region > 0.1 and union_region < 0.5:
            bbox = boxes_filt[i].numpy()
            box_mask = np.zeros((H, W))
            box_mask = box_mask.astype(np.uint8)
            box_mask[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] = 255
            break
    if box_mask is None:
        return None

    original_image_array = np.array(original_image)

    coords = np.argwhere(box_mask > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)

    extracted_content = original_image_array[min_y:max_y+1, min_x:max_x+1]

    original_w = max_x - min_x
    original_h = max_y - min_y
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)
    resized_content = Image.fromarray(extracted_content).resize((scaled_w, scaled_h),
                                                                resample=Image.Resampling.LANCZOS) #(max_x - min_x + 1, max_y - min_y + 1))

    return resized_content, box_mask

def mask_score(mask):
    '''Scoring the mask according to connectivity.'''
    mask = mask.astype(np.uint8)
    if mask.sum() < 10:
        return 0
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt_area = [cv2.contourArea(cnt) for cnt in contours]
    conc_score = np.max(cnt_area) / sum(cnt_area)
    return conc_score


def sobel(img, mask, thresh = 50):
    '''Calculating the high-frequency map.'''
    H,W = img.shape[0], img.shape[1]
    img = cv2.resize(img,(256,256))
    mask = (cv2.resize(mask,(256,256)) > 0.5).astype(np.uint8)
    kernel = np.ones((5,5),np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 2)
    
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)
    sobel_X = cv2.convertScaleAbs(sobelx)
    sobel_Y = cv2.convertScaleAbs(sobely)
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1) * mask    
    
    scharr[scharr < thresh] = 0.0
    scharr = np.stack([scharr,scharr,scharr],-1)
    scharr = (scharr.astype(np.float32)/255 * img.astype(np.float32) ).astype(np.uint8)
    scharr = cv2.resize(scharr,(W,H))
    return scharr


def resize_and_pad(image, box):
    '''Fitting an image to the box region while keeping the aspect ratio.'''
    y1,y2,x1,x2 = box
    H,W = y2-y1, x2-x1
    h,w =  image.shape[0], image.shape[1]
    r_box = W / H 
    r_image = w / h
    if r_box >= r_image:
        h_target = H
        w_target = int(w * H / h) 
        image = cv2.resize(image, (w_target, h_target))

        w1 = (W - w_target) // 2
        w2 = W - w_target - w1
        pad_param = ((0,0),(w1,w2),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    else:
        w_target = W 
        h_target = int(h * W / w)
        image = cv2.resize(image, (w_target, h_target))

        h1 = (H-h_target) // 2 
        h2 = H - h_target - h1
        pad_param =((h1,h2),(0,0),(0,0))
        image = np.pad(image, pad_param, 'constant', constant_values=255)
    return image



def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask


def resize_box(yyxx, H,W,h,w):
    y1,y2,x1,x2 = yyxx
    y1,y2 = int(y1/H * h), int(y2/H * h)
    x1,x2 = int(x1/W * w), int(x2/W * w)
    y1,y2 = min(y1,h), min(y2,h)
    x1,x2 = min(x1,w), min(x2,w)
    return (y1,y2,x1,x2)


def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)


def expand_bbox(mask,yyxx,ratio=[1.2,2.0], min_crop=0):
    y1,y2,x1,x2 = yyxx
    ratio = np.random.randint( ratio[0] * 10,  ratio[1] * 10 ) / 10
    H,W = mask.shape[0], mask.shape[1]
    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)


def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if H > W:
        pad_param = ((0,0),(padd_1,padd_2),(0,0))
    else:
        pad_param = ((padd_1,padd_2),(0,0),(0,0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)
    return image



def box_in_box(small_box, big_box):
    y1,y2,x1,x2 = small_box
    y1_b, _, x1_b, _ = big_box
    y1,y2,x1,x2 = y1 - y1_b ,y2 - y1_b, x1 - x1_b ,x2 - x1_b
    return (y1,y2,x1,x2 )



def shuffle_image(image, N):
    height, width = image.shape[:2]
    
    block_height = height // N
    block_width = width // N
    blocks = []
    
    for i in range(N):
        for j in range(N):
            block = image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width]
            blocks.append(block)
    
    np.random.shuffle(blocks)
    shuffled_image = np.zeros((height, width, 3), dtype=np.uint8)

    for i in range(N):
        for j in range(N):
            shuffled_image[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = blocks[i*N+j]
    return shuffled_image


def get_mosaic_mask(image, fg_mask, N=16, ratio = 0.5):
    ids = [i for i in range(N * N)]
    masked_number = int(N * N * ratio)
    masked_id = np.random.choice(ids, masked_number, replace=False)
    

    
    height, width = image.shape[:2]
    mask = np.ones((height, width))
    
    block_height = height // N
    block_width = width // N
    
    b_id = 0
    for i in range(N):
        for j in range(N):
            if b_id in masked_id:
                mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] = mask[i*block_height:(i+1)*block_height, j*block_width:(j+1)*block_width] * 0
            b_id += 1
    mask = mask * fg_mask
    mask3 = np.stack([mask,mask,mask],-1).copy().astype(np.uint8)
    noise = q_x(image)
    noise_mask = image * mask3 + noise * (1-mask3)
    return noise_mask

def extract_canney_noise(image, mask, dilate=True):
    h,w = image.shape[0],image.shape[1]
    mask = cv2.resize(mask.astype(np.uint8),(w,h)) > 0.5
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask =  cv2.erode(mask.astype(np.uint8), kernel, 10)

    canny = cv2.Canny(image, 50,100) * mask
    kernel = np.ones((8, 8), dtype=np.uint8)
    mask = (cv2.dilate(canny, kernel, 5) > 128).astype(np.uint8)
    mask = np.stack([mask,mask,mask],-1)

    pure_noise = q_x(image, t=1) * 0 + 255
    canny_noise = mask * image + (1-mask) * pure_noise
    return canny_noise


def get_random_structure(size):
    choice = np.random.randint(1, 5)

    if choice == 1:
        return cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    elif choice == 2:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    elif choice == 3:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size//2))
    elif choice == 4:
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size//2, size))

def random_dilate(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.dilate(seg,kernel,iterations = 1)
    return seg

def random_erode(seg, min=3, max=10):
    size = np.random.randint(min, max)
    kernel = get_random_structure(size)
    seg = cv2.erode(seg,kernel,iterations = 1)
    return seg

def compute_iou(seg, gt):
    intersection = seg*gt
    union = seg+gt
    return (np.count_nonzero(intersection) + 1e-6) / (np.count_nonzero(union) + 1e-6)


def select_max_region(mask):
    nums, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    background = 0
    for row in range(stats.shape[0]):
        if stats[row, :][0] == 0 and stats[row, :][1] == 0:
            background = row
    stats_no_bg = np.delete(stats, background, axis=0)
    max_idx = stats_no_bg[:, 4].argmax()
    max_region = np.where(labels==max_idx+1, 1, 0)

    return max_region.astype(np.uint8)



def perturb_mask(gt, min_iou = 0.3,  max_iou = 0.99):
    iou_target = np.random.uniform(min_iou, max_iou)
    h, w = gt.shape
    gt = gt.astype(np.uint8)
    seg = gt.copy()
    
    # Rare case
    if h <= 2 or w <= 2:
        print('GT too small, returning original')
        return seg

    # Do a bunch of random operations
    for _ in range(250):
        for _ in range(4):
            lx, ly = np.random.randint(w), np.random.randint(h)
            lw, lh = np.random.randint(lx+1,w+1), np.random.randint(ly+1,h+1)

            # Randomly set one pixel to 1/0. With the following dilate/erode, we can create holes/external regions
            if np.random.rand() < 0.1:
                cx = int((lx + lw) / 2)
                cy = int((ly + lh) / 2)
                seg[cy, cx] = np.random.randint(2) * 255

            # Dilate/erode
            if np.random.rand() < 0.5:
                seg[ly:lh, lx:lw] = random_dilate(seg[ly:lh, lx:lw])
            else:
                seg[ly:lh, lx:lw] = random_erode(seg[ly:lh, lx:lw])
            
            seg = np.logical_or(seg, gt).astype(np.uint8)
            #seg = select_max_region(seg) 

        if compute_iou(seg, gt) < iou_target:
            break
    seg = select_max_region(seg.astype(np.uint8)) 
    return seg.astype(np.uint8)


def q_x(x_0,t=65):
    '''Adding noise for and given image.'''
    x_0 = torch.from_numpy(x_0).float() / 127.5 - 1
    num_steps = 100
    
    betas = torch.linspace(-6,6,num_steps)
    betas = torch.sigmoid(betas)*(0.5e-2 - 1e-5)+1e-5

    alphas = 1-betas
    alphas_prod = torch.cumprod(alphas,0)
    
    alphas_prod_p = torch.cat([torch.tensor([1]).float(),alphas_prod[:-1]],0)
    alphas_bar_sqrt = torch.sqrt(alphas_prod)
    one_minus_alphas_bar_log = torch.log(1 - alphas_prod)
    one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)
    
    noise = torch.randn_like(x_0)
    alphas_t = alphas_bar_sqrt[t]
    alphas_1_m_t = one_minus_alphas_bar_sqrt[t]
    return (alphas_t * x_0 + alphas_1_m_t * noise).numpy()  * 127.5 + 127.5 


def extract_target_boundary(img, target_mask):
    Ksize = 3
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=Ksize)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=Ksize)

    # sobel-x
    sobel_X = cv2.convertScaleAbs(sobelx)
    # sobel-y
    sobel_Y = cv2.convertScaleAbs(sobely)
    # sobel-xy
    scharr = cv2.addWeighted(sobel_X, 0.5, sobel_Y, 0.5, 0)
    scharr = np.max(scharr,-1).astype(np.float32)/255
    scharr = scharr *  target_mask.astype(np.float32)
    return scharr

def is_transitive(verb):
    synsets = wn.synsets(verb, pos=wn.VERB)
    for synset in synsets:
        for lemma in synset.lemmas():
            frame_strings = lemma.frame_strings()
            if any("something" in frame for frame in frame_strings):
                return True
    return False

def read_json_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    else:
        print(f"File {file_path} does not exist.")
        return []