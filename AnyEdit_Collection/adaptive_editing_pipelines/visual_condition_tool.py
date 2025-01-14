import os
import sys
import numpy as np
import argparse
import json
from tqdm import tqdm
import GroundingDINO.groundingdino.datasets.transforms as T
from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap
from segment_anything import build_sam, SamPredictor
import warnings
import torch
import cv2
from torchvision.transforms import Compose
from pathlib import Path
lama_path = Path(__file__).resolve().parent.parent.parent / "AnyEdit_Collection/other_modules"
sys.path.insert(0, str(lama_path))
from AnyEdit_Collection.other_modules.depth_anything_v2.dpt import DepthAnythingV2
import matplotlib
import glob
from AnyEdit_Collection.other_modules.DPT.util import io
from AnyEdit_Collection.other_modules.DPT.dpt.transforms import Resize, NormalizeImage, PrepareForNet
from termcolor import cprint
from AnyEdit_Collection.other_modules.uniformer.mmseg.datasets.pipelines import Compose
from AnyEdit_Collection.other_modules.uniformer.mmseg.apis import init_segmentor, inference_segmentor, show_result_pyplot
from AnyEdit_Collection.other_modules.uniformer.mmseg.core.evaluation import get_palette
from AnyEdit_Collection.other_modules.HED import HEDdetector
from AnyEdit_Collection.adaptive_editing_pipelines.tools.tool import maskgeneration, get_bbox_from_mask 
warnings.filterwarnings("ignore")


def img2sketch(image_path, output_path):
    # Load the image
    img = cv2.imread(image_path)
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    gaussian_blur = cv2.GaussianBlur(gray, (3, 3), 0)
    # Apply edge detection
    edges = cv2.Canny(gaussian_blur, 50, 150)
    # Invert the colors of the edges
    edges = cv2.bitwise_not(edges)
    # Save the result
    cv2.imwrite(output_path, edges)

def run_depth(input_image, output_image, model):
    """Run MonoDepthNN to compute depth maps.

    Args:
        input_path (str): path to input image
        output_path (str): path to output image
        model_path (str): path to saved model
    """

    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    net_w = net_h = 384
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    model.eval()

    if device == torch.device("cuda"):
        model = model.to(memory_format=torch.channels_last)
        model = model.half()

    model.to(device)
    img = io.read_image(input_image)

    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

        if device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)
            sample = sample.half()

        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )
    cprint(output_image, 'red')
    io.write_depth(output_image, prediction, bits=2, absolute_depth=False)


def img2depth(image_path, output_path, depth_anything = None):
    '''
    change to depth anything V2
    '''
    

    if os.path.isfile(image_path):
        if image_path.endswith('txt'):
            with open(image_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [image_path]
    else:
        filenames = glob.glob(os.path.join(image_path, '**/*'), recursive=True)

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    for k, filename in enumerate(filenames):
        raw_image = cv2.imread(filename)
        depth = depth_anything.infer_image(raw_image)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
        cv2.imwrite(output_path, depth)


def img2seg(image_path, output_path, model):
    # Load the image
    img = cv2.imread(image_path)
    result = inference_segmentor(model, img)
    res_img = show_result_pyplot(model, img, result, get_palette('ade'), opacity=1)
    # Save the result
    cv2.imwrite(output_path, res_img)

def draw_bbox(image, output_path, bbox):
    if isinstance(image, str):
        image = cv2.imread(image_path)
    
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imwrite(output_path, image)
    
def img2bbox(input_path, output_path, target_obj, det_model, sam_model):
    image = cv2.imread(input_path)
    mask_pil,image_pil,bbox_pil,union = maskgeneration(det_model, sam_model, input_path, target_obj)
    if mask_pil is None:
        print(f'Can not find {target_obj} in {input_path}')
        return False, None
    y1, y2, x1, x2 = get_bbox_from_mask(np.array(mask_pil))
    image_pil = np.array(image_pil)
    draw_bbox(image_pil, output_path, [x1, y1, x2, y2])
    
    return True, mask_pil

def load_model(model_config_path, model_checkpoint_path, device):
    args = SLConfig.fromfile(model_config_path)
    args.device = device
    model = build_model(args)
    checkpoint = torch.load(model_checkpoint_path, map_location="cuda")
    load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
    _ = model.eval()
    return model

def load_tool_model(args):
    # visual segment
    seg_config = args.seg_config
    seg_ckpt = args.seg_ckpt
    if args.instruction_type == 'visual_segment':
        seg_model = init_segmentor(config=seg_config, device='cuda', checkpoint=seg_ckpt)
    else:
        seg_model = None
    # visual scribble
    scribble_ckpt = args.scribble_ckpt
    if args.instruction_type == 'visual_scribble':
        hed_model = HEDdetector(path=scribble_ckpt)
    else:
        hed_model = None
    # visual depth
    if args.instruction_type == 'visual_depth':
        depth_ckpt = args.depth_ckpt
        depth_model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        depth_model.load_state_dict(
            torch.load(depth_ckpt, map_location='cpu'))
        depth_model = depth_model.to('cuda').eval()
    else:
        depth_model = None
    # visual bbox
    if args.instruction_type == 'visual_bbox':
        config_file = args.groundingdino_config_file 
        grounded_checkpoint = args.groundingdino_ckpt 
        sam_checkpoint = args.sam_ckpt 

        det_model = load_model(config_file, grounded_checkpoint, device='cuda')
        sam_model = SamPredictor(build_sam(checkpoint=sam_checkpoint).to('cuda'))
    else:
        det_model = None
        sam_model = None
    return seg_model, hed_model, depth_model, det_model, sam_model

def parse_args():
    parser = argparse.ArgumentParser(description="Visual Condition Pipeline")
    parser.add_argument("--seg-config", default=None, help="path to segmentation config file")
    parser.add_argument("--seg-ckpt", default=None, help="path to segmentation checkpoint file")
    parser.add_argument("--scribble-ckpt", default=None, help="path to scribble checkpoint file")
    parser.add_argument("--depth-ckpt", default=None, help="path to depth checkpoint file")
    parser.add_argument("--groundingdino-config-file", default=None, help="path to groundingdino config file.")
    parser.add_argument("--groundingdino-ckpt", default=None, help="path to groundingdino model file")
    parser.add_argument("--sam-ckpt", default=None, help="path to sam model file")
    parser.add_argument("--instruction-path", required=True, help="path to instruction path")
    parser.add_argument("--instruction-type", required=True, help="specify the instruction type.")
    parser.add_argument("--json-path", default=None)
    parser.add_argument("--image-path", default=None)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    edit_instruction_data = []
    try:
        with open(args.json_path) as f:
            for line in f:
                edit_instruction_data.append(json.loads(line.strip()))
    except:
        edit_instruction_data = json.load(open(args.json_path))
    seg_model, hed_model, depth_model, det_model, sam_model = load_tool_model(args)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/input_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/edited_img', exist_ok=True)
    os.makedirs(f'{args.instruction_path}/{args.instruction_type}/visual_input', exist_ok=True) 
    valid_data = []
    for data in tqdm(edit_instruction_data):
        edited_image_file_pre = data['image_file'].split('/')[-1].split('.')[0]
        image_path = data['edited_file']
        if not os.path.exists(image_path):
            continue
        if 'remove' in data['edited_file'].split('/'):  
            visual_path = os.path.join(f'{args.instruction_path}/{args.instruction_type}/visual_input', f'remove_{edited_image_file_pre}.jpg')
        elif 'replace' in data['edited_file'].split('/'):
            visual_path = os.path.join(f'{args.instruction_path}/{args.instruction_type}/visual_input', f'replace_{edited_image_file_pre}.jpg')       
        if args.instruction_type == 'visual_segment':
            img2seg(image_path=image_path, output_path=visual_path, model=seg_model)
            cur_data = {
                'edit': data['edit'],
                'edit object': data['edit object'],
                'input': data['input'],
                'output': data['output'],
                'edit_type': args.instruction_type,
                'image_file': data['image_file'],
                'edited_file': data['edited_file'],
                'visual_input': visual_path
            }
            valid_data.append(cur_data)
        elif args.instruction_type == 'visual_scribble':
            hed_model(image_path, visual_path)
            cur_data = {
                'edit': data['edit'],
                'edit object': data['edit object'],
                'input': data['input'],
                'output': data['output'],
                'edit_type': args.instruction_type,
                'image_file': data['image_file'],
                'edited_file': data['edited_file'],
                'visual_input': visual_path
            }
            valid_data.append(cur_data)
        elif args.instruction_type == 'visual_sketch':
            img2sketch(image_path=image_path, output_path=visual_path)
            cur_data = {
                'edit': data['edit'],
                'edit object': data['edit object'],
                'input': data['input'],
                'output': data['output'],
                'edit_type': args.instruction_type,
                'image_file': data['image_file'],
                'edited_file': data['edited_file'],
                'visual_input': visual_path
            }
            valid_data.append(cur_data)
        elif args.instruction_type == 'visual_depth':
            img2depth(image_path=image_path,output_path=visual_path, depth_anything=depth_model)
            cur_data = {
                'edit': data['edit'],
                'edit object': data['edit object'],
                'input': data['input'],
                'output': data['output'],
                'edit_type': args.instruction_type,
                'image_file': data['image_file'],
                'edited_file': data['edited_file'],
                'visual_input': visual_path
            }
            valid_data.append(cur_data)
        elif args.instruction_type == 'visual_bbox':
            if 'flux' in data['image_file']:
                image_path = os.path.join('/home1/yqf/ssd', data['image_file'].replace('.jpg', '.png'))
            else:
                image_path = os.path.join(args.image_path, data['image_file'])
            valid, _ = img2bbox(input_path=image_path, output_path=visual_path, target_obj = data['edit object'], det_model = det_model, sam_model = sam_model)
            if valid:
                cur_data = {
                    'edit': data['edit'],
                    'edit object': data['edit object'],
                    'input': data['input'],
                    'output': data['output'],
                    'edit_type': args.instruction_type,
                    'image_file': data['image_file'],
                    'edited_file': data['edited_file'],
                    'visual_input': visual_path
                }
                valid_data.append(cur_data)
    print(f"valid editing insturction data: {len(valid_data)}")   
    with open(f'{args.instruction_path}/{args.instruction_type}/final_edit_results_-1_-1.json', 'w') as f:
        json.dump(valid_data, f, indent=4)
            
    
    