import warnings
from tqdm import tqdm
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import load_model, img4outpainting
from PIL import Image
import random
import glob
import os
import json
import AnyEdit_Collection.adaptive_editing_pipelines.tools.tool as tool
if __name__ == '__main__':
    # presets and model loading
    groundingdino_config_file = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    groundingdino_ckpt = "./checkpoints/foundation_models/groundingDINO/groundingdino_swinb_cogcoor.pth"
    device = 'cuda'
    
    mscoco_path = "./AnyEdit/original_datasets/coco/train2014"
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
    
    
    # Generate images
    for i, image_file in tqdm(enumerate(image_files_paths), total=len(image_files_paths)):
        original_image_file = image_file
        
        # Read the caption from the corresponding txt file
        with open(txt_files[i], 'r') as f:
            original_caption = f.readline().strip()
        
        # Generate out painting images
        original_image = Image.open(original_image_file)
        
        result = img4outpainting(det_model=det_model, original_image=original_image, caption=original_caption)
        if result is not None:
            original_image_path = os.path.join(output_dir,"edited_img",os.path.basename(image_file))
            original_image.save(original_image_path)
            crop_image_path = os.path.join(output_dir, "input_img",f'crop_{image_filenames[i]}')
            result.save(crop_image_path)
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
    
    # Save insturctions
    jsonl_file_path = os.path.join(output_dir, 'final_edit_results_all.jsonl')
    with open(jsonl_file_path, 'w') as jsonl_file:
        for data in total_edited_images:
            jsonl_file.write(json.dumps(data) + '\n')

    print("JSONL dump to {}".format(jsonl_file_path))