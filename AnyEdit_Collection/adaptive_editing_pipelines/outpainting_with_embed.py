import warnings
from tqdm import tqdm
from PIL import Image
import random
import glob
import os
import json
import numpy as np


from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import load_model, img4outpainting
import AnyEdit_Collection.adaptive_editing_pipelines.tools.tool as tool


groundingdino_config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
groundingdino_ckpt = "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth"
device = 'cuda'

mscoco_path = './AnyEdit/original_datasets/coco/train2014'
image_files_paths = sorted(glob.glob(os.path.join(mscoco_path, '*.jpg')))
image_filenames = [os.path.basename(image_file) for image_file in image_files_paths]
txt_files = sorted(glob.glob(os.path.join(mscoco_path, '*.txt')))
total_edited_images = []
debug = False
if not debug:
    det_model = load_model(groundingdino_config_file, groundingdino_ckpt, device=device)

# Create the output directory if it doesn't exist
output_dir = './AnyEdit/data/outpaint'
tool.prepare_output_dir(output_dir)

def save_data(data, output_jsonl_path):
    if isinstance(data, int):
        with open(output_jsonl_path, 'w') as f:
            f.write(str(data) + '\n')
    else:
        with open(output_jsonl_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d) + '\n')
    


def embed_crop_image(crop_image, box_mask):
    # 确保crop_image是Image类型，并转换为numpy array
    crop_image = np.array(crop_image)
    
    # 获取box_mask中白色区域的坐标
    coords = np.argwhere(box_mask > 0)
    min_y, min_x = np.min(coords, axis=0)
    max_y, max_x = np.max(coords, axis=0)
    
    # 计算白色区域的宽高
    mask_height = max_y - min_y + 1
    mask_width = max_x - min_x + 1
    
    if len(crop_image.shape) == 2:
        crop_image = np.stack([crop_image]*3, axis=-1)
    
    # 调整crop_image的大小以匹配白色区域
    resized_crop_image = np.array(Image.fromarray(crop_image).resize((mask_width, mask_height)))
    
    # 创建与box_mask相同大小的黑色背景，并确保它有3个通道（RGB）
    result_image = np.zeros((box_mask.shape[0], box_mask.shape[1], 3), dtype=np.uint8)
    
    # 将裁剪后的图片填充到白色区域
    result_image[min_y:max_y+1, min_x:max_x+1, :] = resized_crop_image
    
    # 返回处理后的图片
    return Image.fromarray(result_image)



state_path = os.path.join(output_dir, 'state.json')
start_ind = 0
if os.path.exists(state_path):
    with open(state_path, 'r') as f:
        state = json.load(f)
        start_ind = state
        
jsonl_file_path = os.path.join(output_dir, 'final_edit_results_all.jsonl')
if os.path.exists(jsonl_file_path):
    with open(jsonl_file_path, 'r') as f:
        for line in f:
            total_edited_images.append(json.loads(line))

for i, image_file in tqdm(enumerate(image_files_paths), total=len(image_files_paths)):
        if i < start_ind:
            continue
        
        original_image_file = image_file
        
        # Read the caption from the corresponding txt file
        with open(txt_files[i], 'r') as f:
            original_caption = f.readline().strip()
        
        # Generate out painting images
        original_image = Image.open(original_image_file)
        
        if original_caption == '':
            print(f'{image_file} has no caption')
            continue
        res= img4outpainting(det_model=det_model, original_image=original_image, caption=original_caption)
        
        if res is not None:
            result   = res[0]
            box_mask = res[1]
            
            box_mask_pil = Image.fromarray(box_mask)
            box_mask_pil = box_mask_pil.convert('L')
            box_mask_pil.save(output_dir +  '/mask/' + os.path.basename(image_file))
            
            original_image_path = os.path.join(output_dir,"edited_img",os.path.basename(image_file))
            original_image.save(original_image_path)
            
            crop_image_path = os.path.join(output_dir, "input_img",f'crop_{image_filenames[i]}')
            result.save(crop_image_path)
            
            embed = embed_crop_image(result, box_mask)
            embed_image_path = os.path.join(output_dir, "embed_img",f'embed_{image_filenames[i]}')
            embed.save(embed_image_path)
            
            operate = random.choice(["Outpaint", "Imagine", "Complete"])
            cur_data = {
                "edit": f"{operate} the image as you can",
                "edit object": None,
                "input": None,
                "output": None,
                "edit_type": "outpainting",
                "visual_input": None,
                "image_file": os.path.basename(crop_image_path),
                "edited_file": os.path.basename(original_image_path),
            }
            total_edited_images.append(cur_data)
        else:
            print(f'{image_file} is not suitable for outpainting')
        
        if i % 400 == 0:
                
                save_data(total_edited_images, jsonl_file_path)
                save_data(i, state_path)
                print(f'Saved {i} images to {jsonl_file_path}')





