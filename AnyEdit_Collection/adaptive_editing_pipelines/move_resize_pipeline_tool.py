import os
import sys
from diffusers import AutoPipelineForText2Image, FluxPipeline
import torch
from PIL import Image
import random
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict
from segment_anything import build_sam, SamPredictor
import cv2
import numpy as np
import warnings
import json
from tqdm import tqdm
import argparse
from omegaconf import OmegaConf
from pathlib import Path
lama_path = Path(__file__).resolve().parent.parent.parent / "AnyEdit_Collection/other_modules/lama"
sys.path.insert(0, str(lama_path))
from saicinpainting.training.trainers import load_checkpoint
from saicinpainting.evaluation.data import pad_tensor_to_modulo
from saicinpainting.evaluation.utils import move_to_device
import yaml
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import return_parameters, maskgeneration, generate_tags
warnings.filterwarnings("ignore")                                     


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

def check_occlusion_by_segmentation(mask, threshold=0.15, min_area=50):
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

def image_generation_pipe(base, prompt):
    # sdxl-turbo, but not good for human
    # image = base(prompt=prompt, num_inference_steps=4, guidance_scale=0.0).images[0]
    # flux
    seed = random.randint(0, 2<<32)
    image = base(
        prompt,
        output_type="pil",
        num_inference_steps=4, #use a larger number if you are using [dev]
        generator=torch.Generator("cuda").manual_seed(seed)
    ).images[0]
    return image.resize((512,512), resample=Image.Resampling.LANCZOS)

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
    extracted_content_mask = mask_array[min_y:max_y+1, min_x:max_x+1]

    original_w = max_x - min_x
    original_h = max_y - min_y
    scaled_w = int(original_w * scale)
    scaled_h = int(original_h * scale)
    
    # 检查是否超出背景图像的边界
    if (x + scaled_w > background_image_array.shape[1]) or (y + scaled_h > background_image_array.shape[0]):
        print(f"Error: Scaled content exceeds background image boundaries. "
              f"Scaled size: ({scaled_w}, {scaled_h}), "
              f"Position: ({x}, {y}), "
              f"Background size: {background_image_array.shape[:2]}")
        return None

    # 调整后的内容和掩码
    resized_content = Image.fromarray(extracted_content).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS)
    resized_content_mask = Image.fromarray(extracted_content_mask).resize((scaled_w, scaled_h), resample=Image.Resampling.LANCZOS)

    resized_content_array = np.array(resized_content)
    resized_content_mask_array = np.array(resized_content_mask)

    # 重新定义 result_array 来指代背景图像数组
    result_array = background_image_array.copy()

    # 确保 result_array 切片与 resized_content_mask_array 形状一致
    result_sub_array = result_array[y:y+scaled_h, x:x+scaled_w]

    # 如果形状不匹配，打印错误并返回 None
    if result_sub_array.shape[:2] != resized_content_mask_array.shape:
        print(f"Error: Mismatched shapes between result sub-array {result_sub_array.shape[:2]} and mask array {resized_content_mask_array.shape}.")
        return None

    # 进行内容的替换
    result_sub_array[resized_content_mask_array > 0] = resized_content_array[resized_content_mask_array > 0]

    # 更新结果数组
    result_array[y:y+scaled_h, x:x+scaled_w] = result_sub_array

    result_image = Image.fromarray(result_array)
    return result_image


def load_tool_model(args):
    config_file = args.groundingdino_config_file #'GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py' 
    grounded_checkpoint = args.groundingdino_ckpt #'checkpoints/groundingdino_swinb_cogcoor.pth'  
    sam_checkpoint = args.sam_ckpt #'checkpoints/sam_vit_h_4b8939.pth'
    sdxl_ckpt = args.sdxl_ckpt # "stabilityai/sdxl-turbo"
    lama_config = args.lama_config_file # "./lama/configs/prediction/default.yaml"
    lama_checkpoint = args.lama_ckpt # "./checkpoints/foundation_models/big-lama"
    device = 'cuda'

    det_model = load_model(config_file, grounded_checkpoint, device=device)
    sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to(device))

    if 'FLUX' in sdxl_ckpt or 'Flux' in sdxl_ckpt or 'flux' in sdxl_ckpt: 
        sdxl_base = FluxPipeline.from_pretrained(sdxl_ckpt, torch_dtype=torch.bfloat16).to(device)
    else:
        sdxl_base = AutoPipelineForText2Image.from_pretrained(sdxl_ckpt, torch_dtype=torch.float16, variant="fp16").to(device)

    lama_pipe = lama_builder(lama_config, lama_checkpoint, device=device)
    return sdxl_base, lama_pipe, det_model, sam_model

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
def resize_movement(det_model, sam_model, lama_pipe, input, edited_object,
                    init_image_path, edited_image_path_pre, instruction_type, foreground=None, sdxl_base=None):
    # generate init image: x
    if not os.path.exists(init_image_path):
        init_image = image_generation_pipe(sdxl_base, input)
        init_image.save(init_image_path)

    original_size = Image.open(init_image_path).convert("RGB").size
    total_objects = generate_tags(input)['nouns']

    if edited_object not in total_objects:
        total_objects.append(edited_object)

    prompt = ', '.join(total_objects)
    mask_path = edited_image_path_pre.replace("edited_img", "mask") + f'.jpg'
    mask_pil, image_pil, mask_bbox, union_region = maskgeneration(det_model, sam_model, init_image_path, prompt, target_object=edited_object)

    if mask_pil is None:
        print(f'Skip: mask for {edited_object} is none, image refers to {init_image_path}')
        return None
    # check if the edited object intersects with the foreground object
    if foreground is not None:
        for obj in foreground:
            if obj == edited_object:
                print(f'Skip: foreground object is the same as the edited object')
                continue
            mask_pil_fore, _, _, _ = maskgeneration(det_model, sam_model, init_image_path, prompt, target_object=obj)
            if mask_pil_fore is not None:
                if masks_overlap(np.array(mask_pil, dtype='int'), np.array(mask_pil_fore,dtype='int')):
                    print(f'Skip: foreground object({obj}) intersects with edited object({edited_object})')
                    return None
                
    mask_array = np.array(mask_pil)
    
    #check if the edited object is occluded
    if check_occlusion_by_segmentation(mask_array, threshold = 0.1):
        print('Skip: occlusion')
        return
    
    coords = np.argwhere(mask_array > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)
    mask_size = [max_x-min_x, max_y-min_y]
    total_edited_images = []
    
    
    mask_pil.save(mask_path)
    cv2_mask_image = cv2.imread(mask_path)
    maskimage_dilate = cv2.dilate(cv2_mask_image, np.ones((15, 15), np.uint8))
    maskimage_dilate = Image.fromarray(cv2.cvtColor(maskimage_dilate,cv2.COLOR_BGR2GRAY))
    edited_image = inpaint_img_with_lama(np.array(image_pil), np.array(maskimage_dilate), lama_pipe)
    edited_image = Image.fromarray(edited_image)
    
    edit_mask_pil, _, _, _ = maskgeneration(det_model, sam_model, edited_image, prompt, target_object=edited_object)
    if edit_mask_pil is None:
        # now to add movement/resize object
        if instruction_type == 'movement':
            scale = 1
            if int(1-min_x) >= -50 and image_pil.size[0]-1-int(mask_size[0]*scale)-int(min_x) > 50:
                delta_x = random.randint(50, image_pil.size[0]-1-int(mask_size[0]*scale)-int(min_x))
            elif image_pil.size[0]-1-int(mask_size[0]*scale)-int(min_x) > 50:
                dis_0 = random.randint(int(1-min_x), -50)
                dis_1 = random.randint(50, image_pil.size[0]-1-int(mask_size[0]*scale)-int(min_x))
                delta_x = random.choice([dis_0, dis_1])
            elif int(1-min_x) < -50:
                delta_x = random.randint(int(1-min_x), -50)
            else:
                if int(1-min_x) != image_pil.size[0]-1-int(mask_size[0]*scale)-int(min_x):
                    delta_x = random.randint(int(1-min_x), image_pil.size[0]-1-int(mask_size[0]*scale)-int(min_x))
                else:
                    delta_x = int(1-min_x)
            x = max(1, min_x + delta_x)
            y = max_y - int(mask_size[1]*scale)
        else:
            if int((image_pil.size[0]-min_x)/mask_size[0]) < 1.5 or int((image_pil.size[1]-min_y)/mask_size[1]) < 1.5:
                scale = 0.7
            else:
                scale = 1.3
            x = min_x
            y = max_y - int(mask_size[1]*scale)

        result_image_pil = resize_cropimage2image(image_pil, mask_pil, edited_image, x, y, scale)
        edited_image = result_image_pil.resize(original_size, resample=Image.Resampling.LANCZOS)
        cur_edited_image_path = edited_image_path_pre+f'.jpg'
        if scale == 1:
            instruction_word = random.choice(['Move', 'Shift'])
            cur_data = {"edit": f"{instruction_word} the {edited_object} in the image", "edit object": edited_object,
                        "output": None, "input": input, "edit_type": "movement", "image_file": init_image_path,
                        "edited_file": cur_edited_image_path}
        elif scale == 1.3:
            instruction_word = random.choice(['Zoom in', 'Enlarge', 'Amplify'])
            cur_data = {"edit": f"{instruction_word} the {edited_object} in the image", "edit object": edited_object,
                        "output": None, "input": input, "edit_type": "resize", "image_file": init_image_path,
                        "edited_file": cur_edited_image_path}
        else:
            instruction_word = random.choice(['Zoom out', 'Minify'])
            cur_data = {"edit": f"{instruction_word} the {edited_object} in the image", "edit object": edited_object,
                        "output": None, "input": input, "edit_type": "resize", "image_file": init_image_path,
                        "edited_file": cur_edited_image_path}
        edited_image.save(cur_edited_image_path)
        total_edited_images.append(cur_data)
    return total_edited_images


def parse_args():
    parser = argparse.ArgumentParser(description="Editing Pipeline")
    parser.add_argument("--groundingdino-config-file", required=True, help="path to groundingdino config file.")
    parser.add_argument("--groundingdino-ckpt", required=True, help="path to groundingdino model file")
    parser.add_argument("--sam-ckpt", required=True, help="path to sam model file")
    parser.add_argument("--sdxl-ckpt", required=True, help="path to sdxl model file")
    parser.add_argument("--lama-config-file", required=True, help="path to inpaint model file")
    parser.add_argument("--lama-ckpt", required=True, help="path to inpaint model file")
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
    return parser.parse_args()



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
    final_edited_results = []
    
    edit_success_path = f'{instruction_path}/edit_success_{args.start_idx}_{args.end_idx}.json'
    success_data = []

    edit_failure_path = f'{instruction_path}/edit_failure_{args.start_idx}_{args.end_idx}.json'
    failure_data = []

    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/mask', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/input_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)

    sdxl_base, lama_pipe, det_model, sam_model = load_tool_model(args)

    valid_number  = 0
        
    for data in tqdm(edit_instruction_data[:2]):
        # data['edit_type'] = random.choice(['movement', 'resize'])
        data['edit_type'] = args.instruction_type
        if 'edited object' not in data and 'foreground' in data:
            data['edited object'] = random.choice(data['foreground'])
        instruction_type = data['edit_type']
        if 'input' not in data or 'output' not in data:
            print(f'Skip: input or output in data')
            continue

        if instruction_type in ["resize", "movement"] and 'edited object' not in data:
            print(f'Skip: no edited object in data')
            continue

        if instruction_type in ["resize", "movement"]:
            input, edited_object, output, init_image_path, instruction, edited_image_path_pre = \
            return_parameters(data, args, init_image_root=args.image_path)
        else:
            raise NotImplementedError

        instruction_data = resize_movement(det_model=det_model, sam_model=sam_model, lama_pipe=lama_pipe,
                                    input=input, edited_object=edited_object, init_image_path=init_image_path, edited_image_path_pre=edited_image_path_pre,foreground = data['foreground'] if 'foreground' in data.keys() else None, instruction_type=instruction_type, sdxl_base=sdxl_base)


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
        






